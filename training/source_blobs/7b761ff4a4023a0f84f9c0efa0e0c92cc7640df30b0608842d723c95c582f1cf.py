from pathlib import Path
import json
from sml_v1.common import atomic_json, read_json, file_sha256
from .text_quality import key

def exclusions(project):
    keys=set();receipts=[]
    import pyarrow as pa
    benchmark=project/'pretraining/data/benchmark_inputs_20261102'
    manifest=read_json(benchmark/'manifest.json')
    for name,digest in manifest['files'].items():
        path=benchmark/name
        if file_sha256(path)!=digest:raise ValueError('Benchmark exclusion source changed')
        if path.suffix=='.arrow':
            with pa.memory_map(str(path),'r') as mm:
                table=pa.ipc.open_stream(mm).read_all()
                for field in ('question','prompt','goal','ctx'):
                    if field in table.column_names:keys.update(key(v) for v in table[field].to_pylist() if isinstance(v,str))
        if name=='ifeval_inputs.json':
            value=read_json(path)
            rows=value if isinstance(value,list) else value.get('examples',value.get('rows',[]))
            for row in rows:
                if isinstance(row,dict) and isinstance(row.get('prompt'),str):keys.add(key(row['prompt']))
        receipts.append(dict(path=str(path),sha256=digest))
    for path in (project/'pretraining/data').glob('sft_*/**/*.jsonl'):
        if path.name not in ('dev.jsonl','test.jsonl'):continue
        for line in path.read_text().splitlines():
            row=json.loads(line)
            for m in row.get('messages',[]):
                if m['role']=='user' and len(m['content'].split())>=5:keys.add(key(m['content']))
            if row.get('question'):keys.add(key(row['question']))
        receipts.append(dict(path=str(path),sha256=file_sha256(path)))
    # The already inspected pretraining questions stay diagnostics, never targets.
    for prompt in read_json(project/'pretraining/prepared/python_sql_20261104/sample_prompts.json')['prompts']:
        keys.add(key(prompt));keys.add(key(prompt.rsplit('Question:',1)[-1].split('SQL:')[0]))
    for directory in ('openai___openai_humaneval','google-research-datasets___mbpp'):
        for path in (project/'pretraining/data/.sft_hf_cache'/directory).rglob('*.arrow'):
            if path.name.startswith('cache-'):continue
            with pa.memory_map(str(path),'r') as mm:
                table=pa.ipc.open_stream(mm).read_all()
                for field in ('prompt','text'):
                    if field in table.column_names:keys.update(key(v) for v in table[field].to_pylist() if isinstance(v,str))
            receipts.append(dict(path=str(path),sha256=file_sha256(path)))
    return keys,receipts

