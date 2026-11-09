"""Whole-answer paired instruction checks; raw replies retained for review."""
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
from .behavior_v8 import python_checks,typed_equal
from .quality import repeated


def check(row,result):
    text=result['text'].strip();gold=row['messages'][-1]['content']
    natural=result['stop_reason'] in ('turn_end','document_end')
    clean=bool(text) and not repeated(text)
    if row.get('functional_assertions'):
        detail=python_checks(text,row['functional_assertions'],**row['python_contract']);semantic=all(detail.values())
    elif row.get('trace_contract'):
        try:semantic=typed_equal(ast.literal_eval(text),ast.literal_eval(gold))
        except (ValueError,SyntaxError):semantic=False
        detail=dict(python_value=semantic)
    elif gold.startswith('{'):
        try:semantic=json.loads(text)==json.loads(gold)
        except ValueError:semantic=False
        detail=dict(json_contract=semantic)
    elif row.get('rules'):
        semantic=text==gold;detail=dict(literal_contract=semantic)
    else:
        # Only an explicitly authored missing-input request appears in the
        # scored dialogue set; exact wording is not required.
        semantic=bool(re.search(r'\b(?:send|share|paste|provide)\b',text,re.I) and re.search(r'\b(?:note|draft|text|recipient|tone)\b',text,re.I)) and len(text.split())<=45
        detail=dict(requests_missing_input=semantic)
    return dict(**detail,natural_stop=natural,nonrepetitive_nonempty=clean,passed=bool(semantic and natural and clean))


def run(model,tokenizer,prepared,split='dev'):
    from .evaluate import generate
    rows=json.loads((Path(prepared)/'contrast_panels.json').read_text())[split]
    samples=[];pairs=defaultdict(list);bybucket=defaultdict(list)
    for row in rows:
        history=[];transcript=[]
        for message in row['messages']:
            if message['role']=='system':history.append(message)
            elif message['role']=='user':
                history.append(message)
                result=generate(model,tokenizer,history,limit=160)
                transcript.append(dict(request=[dict(m) for m in history],**result))
                history.append(dict(role='assistant',content=result['text'] or '[empty response]'))
        checked=check(row,result)
        checked['passed']=checked['passed'] and all(t['stop_reason'] in ('turn_end','document_end') and bool(t['text'].strip()) and not repeated(t['text']) for t in transcript)
        pairs[row['pair_id']].append(bool(checked['passed']))
        samples.append(dict(pair_id=row['pair_id'],bucket=row['bucket'],group=row['group'],gold=row['messages'][-1]['content'],transcript=transcript,checks=checked))
    for pair_id,values in pairs.items():
        if len(values)!=2:raise ValueError('Incomplete diagnostic pair')
        bucket=next(r['bucket'] for r in rows if r['pair_id']==pair_id)
        bybucket[bucket].append(all(values))
    rates={bucket:sum(values)/len(values) for bucket,values in bybucket.items()}
    return dict(split=split,rates=rates,macro_pair_pass=sum(rates.values())/len(rates),pair_count=len(pairs),samples=samples,
        scope='Both instructions in each held-out group pair must pass. Actual generated prior replies used for dialogue. Literal/JSON/Python contracts are bounded; missing-input language remains heuristic. Not a universal chat or factual score.')
