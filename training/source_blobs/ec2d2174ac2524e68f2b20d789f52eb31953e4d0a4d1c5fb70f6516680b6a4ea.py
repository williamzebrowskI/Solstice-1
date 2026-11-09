"""Public Parquet readers, including bounded schema/question/SQL rendering."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

PARSER_TREE_SHA256 = 'daa3692255791d46338eb807686a3b3724f73cbad336f25170b2a08de201145e'
FIELDS = {'text':'string','repo_name':'string','path':'string','language':'string',
          'blob_id':'string','src_encoding':'string','length_bytes':'int64',
          'int_score':'int64','license_type':'string','detected_licenses':['string'],
          'download_success':'bool','_code_kind':'string','_rejection':'string'}


@lru_cache(maxsize=1)
def parser():
    vendor=Path(__file__).parent/'_vendor'
    files={str(p.relative_to(vendor)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (vendor/'sqlglot').rglob('*.py')}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if digest != PARSER_TREE_SHA256:
        raise ValueError('Vendored SQL parser integrity mismatch')
    sys.path.insert(0,str(vendor))
    import sqlglot
    if sqlglot.__version__ != '30.29.0' or not Path(sqlglot.__file__).resolve().is_relative_to(vendor.resolve()):
        raise ValueError('Unexpected SQL parser installation')
    return sqlglot


def raw_record(row):
    return dict(text=row['text'],repo_name=row['repo_name'],path=row['path'],language=row['language'],
        blob_id=row['blob_id'],src_encoding=row.get('src_encoding') or 'utf-8',length_bytes=row['length_bytes'],
        int_score=row['int_score'],license_type=row['license_type'],detected_licenses=row['detected_licenses'],
        download_success=row.get('download_success',True),_code_kind='raw',_rejection='')


def synsql_record(row):
    result=dict(text='',repo_name='synsql/'+str(row.get('db_id','')),path='synthetic.sql',language='SQL',
        blob_id='',src_encoding='utf-8',length_bytes=0,int_score=-1,license_type='synthetic-apache-2.0',
        detected_licenses=['Apache-2.0'],download_success=True,_code_kind='synsql',_rejection='sql_schema_or_query')
    if not row.get('db_id') or any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('schema','sql','question')):
        return result
    if len(row['schema'])>100000 or len(row['sql'])>20000 or len(row['question'])>10000:
        result['_rejection']='oversized_synthetic_input'
        return result
    sqlglot=parser()
    exp=sqlglot.exp
    try:
        queries=sqlglot.parse(row['sql'],read='sqlite')
        if len(queries)!=1 or not isinstance(queries[0],exp.Query):
            return result
        query=queries[0]
        required={t.name.casefold() for t in query.find_all(exp.Table)}
        ctes={t.alias.casefold() for t in query.find_all(exp.CTE)}
        required-=ctes
        schema=sqlglot.parse(row['schema'],read='sqlite')
        selected={}
        for statement in schema:
            if (not isinstance(statement,exp.Create) or statement.args.get('kind')!='TABLE'
                    or not isinstance(statement.this,exp.Schema) or statement.args.get('expression') is not None):
                return result
            name=statement.this.this.name.casefold()
            if name in required:
                selected[name]=statement.sql(dialect='sqlite',comments=False)
        if not required or set(selected)!=required:
            return result
        sql=query.sql(dialect='sqlite',comments=False)
        # Compile against empty in-memory tables. No query is executed and no
        # downloaded Python is executed. Block SQLite filesystem/extension APIs.
        db=sqlite3.connect(':memory:')
        try:
            db.execute('PRAGMA temp_store=MEMORY')
            denied={sqlite3.SQLITE_ATTACH,sqlite3.SQLITE_DETACH,sqlite3.SQLITE_PRAGMA,
                    sqlite3.SQLITE_CREATE_VTABLE,sqlite3.SQLITE_DROP_VTABLE}
            def authorize(action,a,b,*unused):
                if action in denied or (action==sqlite3.SQLITE_FUNCTION and str(b).lower() in ('load_extension','readfile','writefile')):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorize)
            db.set_progress_handler(lambda:1,100000)
            for ddl in selected.values():
                db.execute(ddl)
            db.execute('EXPLAIN '+sql).fetchall()
        finally:
            db.close()
        context=row.get('external_knowledge') or ''
        text='Schema:\n'+';\n'.join(selected.values())+';\n\nQuestion:\n'+row['question']
        if context:
            text+='\n\nContext:\n'+context
        text+='\n\nSQL:\n'+sql+'\n'
        result.update(text=text,length_bytes=len(text.encode()),_rejection='')
    except (sqlglot.errors.SqlglotError,sqlite3.Error,ValueError,RecursionError):
        pass
    return result


def open_component(source):
    from datasets import load_dataset
    kwargs={k:source[k] for k in ('columns','data_files','data_dir','filters') if k in source}
    return load_dataset(source['repo'],source.get('config'),revision=source['revision'],
                        split=source['split'],streaming=True,**kwargs)


class ComponentStream:
    """Resume raw Parquet positions, then render; never rely on map/concat cursors."""
    FORMAT='sml-public-code-components-v1'

    def __init__(self,source):
        primary={k:v for k,v in source.items() if k!='additional_sources'}
        self.components=[primary,*source.get('additional_sources',[])]
        self.index=0
        self.position=None
        self.active=None

    def state_dict(self):
        position=self.active.state_dict() if self.active is not None else self.position
        return dict(format=self.FORMAT,component=self.index,position=copy.deepcopy(position))

    def load_state_dict(self,state):
        if (not isinstance(state,dict) or set(state)!={'format','component','position'}
                or state['format']!=self.FORMAT or type(state['component']) is not int
                or not 0<=state['component']<=len(self.components)
                or (state['position'] is not None and not isinstance(state['position'],dict))):
            raise ValueError('Invalid component cursor')
        self.index=state['component'];self.position=copy.deepcopy(state['position']);self.active=None

    def __iter__(self):
        while self.index<len(self.components):
            source=self.components[self.index]
            self.active=open_component(source)
            if self.position is not None:
                self.active.load_state_dict(copy.deepcopy(self.position))
            iterator=iter(self.active)
            try:
                for row in iterator:
                    yield synsql_record(row) if source.get('renderer')=='synsql-schema-query-v1' else raw_record(row)
            finally:
                self.position=copy.deepcopy(self.active.state_dict())
                self.active=None
                if hasattr(iterator,'close'):iterator.close()
            self.index+=1;self.position=None


def open_group(source):
    return ComponentStream(source)
