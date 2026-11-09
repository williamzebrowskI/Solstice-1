"""Separate task-completion, free-generation, knowledge and prose guards."""
from collections import defaultdict
import json
import math
from pathlib import Path
import mlx.core as mx
import numpy as np
from sml_v1.common import atomic_json,read_json
from .data import Corpus
from .evaluate import generate,conversation_losses
from .balanced_tasks import check
from .quality_gate import looping,unchanged_edit_claim
from .retention import verify as verify_retention


def choice_batch(tokenizer,question,choices,max_length):
    """Mask the question; account for any token merge at the answer boundary."""
    prompt='Question: '+question.strip()+'\nAnswer:'
    prefix=tokenizer.encode(prompt);sequences=[];starts=[]
    for answer in choices:
        ids=tokenizer.encode(prompt+' '+answer.strip())
        common=0
        for a,b in zip(prefix,ids):
            if a!=b:break
            common+=1
        if common<1 or len(ids)<=common or len(ids)>max_length:
            raise ValueError('Invalid/overlong knowledge-guard answer; no truncation allowed')
        sequences.append(ids);starts.append(common)
    length=max(map(len,sequences))-1
    x=np.zeros((len(choices),length),np.int32);y=np.full_like(x,-100)
    for i,(ids,start) in enumerate(zip(sequences,starts)):
        x[i,:len(ids)-1]=ids[:-1];y[i,start-1:len(ids)-1]=ids[start:]
    return x,y


def knowledge(model,tokenizer,data_dir):
    rows=read_json(Path(data_dir)/'knowledge_guard.json')['records'];scores=defaultdict(list);details=[]
    for row in rows:
        x,y=choice_batch(tokenizer,row['question'],row['choices'],model.cfg.max_seq_len)
        token_loss=model(mx.array(x),targets=mx.array(y))['token_loss']
        losses=(token_loss*mx.array(y!=-100)).sum(axis=1)
        values=losses.tolist()
        if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite knowledge guard')
        predicted=int(np.argmin(values));correct=predicted==row['gold'];scores[row['task']].append(correct)
        details.append(dict(task=row['task'],index=row['index'],prediction=predicted,gold=row['gold'],choice_nll=values))
    if not rows:raise ValueError('Empty knowledge guard')
    return dict(correct=sum(sum(s) for s in scores.values()),total=len(rows),
        tasks={k:dict(correct=sum(v),total=len(v)) for k,v in scores.items()},
        scoring='Raw continuation log likelihood; fixed development guard, no chat wrapper.',details=details)


def evaluate(model,tokenizer,data_dir,plan,step,output,retention,*,split='dev',checker=check):
    data_dir=Path(data_dir);output=Path(output);corpus=Corpus(data_dir,split)
    rows=[json.loads(s) for s in (data_dir/(split+'.jsonl')).read_text().splitlines()]
    by_component=defaultdict(list)
    for i,row in enumerate(rows):by_component[row['component']].append(i)
    nll={};counts={};samples=[];rates=defaultdict(list)
    model.eval()
    try:
        for component,indices in sorted(by_component.items()):
            indices=indices[:max(1,plan['eval_examples']//len(by_component))]
            total=count=tokens=0
            for start in range(0,len(indices),2):
                x,y=corpus.batch(indices[start:start+2]);values,n=conversation_losses(model,mx.array(x),mx.array(y))
                total+=float(values.sum().item());count+=len(x);tokens+=int(n.sum().item())
            nll[component]=total/count;counts[component]=tokens
        for row in read_json(data_dir/('behavior_'+split+'.json')):
            value=generate(model,tokenizer,row['messages'][:-1],limit=row['limit'])
            loop=looping(value['text']);meaningless=unchanged_edit_claim(value['text'])
            natural=value['stop_reason'] in ('turn_end','document_end')
            rules=row.get('verification');passed=None
            if rules:
                passed=bool(checker(value['text'],rules['checks']) and natural and not loop and not meaningless and value['text'].strip())
                rates[row['family']].append(passed)
            samples.append(dict(id=row['id'],component=row['component'],family=row['family'],
                messages=row['messages'][:-1],reference=row['messages'][-1]['content'],
                **value,repeated=loop,meaningless_edit=meaningless,passed=passed))
        receipt=verify_retention(retention);losses={}
        with np.load(Path(retention)/'panel.npz',allow_pickle=False) as panel:
            for source in receipt['weights']:
                values=[float(model(mx.array(x),targets=mx.array(y))['loss'].item()) for x,y in zip(panel[source+'_x'],panel[source+'_y'])]
                losses[source]=sum(values)/len(values)
        assistant_nll=sum(nll.values())/len(nll)
        if not math.isfinite(assistant_nll) or not all(math.isfinite(v) for v in losses.values()) or not rates or not samples:
            raise RuntimeError('Invalid balanced evaluation')
        knowledge_result=knowledge(model,tokenizer,data_dir)
        behavior=dict(total=len(samples),cap_hits=sum(s['stop_reason']=='token_limit' for s in samples),
            repetition_hits=sum(s['repeated'] for s in samples),
            invalid_stop_hits=sum(s['stop_reason']=='unexpected_structure' for s in samples),
            meaningless_edit_hits=sum(s['meaningless_edit'] for s in samples))
        generated_rates={k:sum(v)/len(v) for k,v in rates.items()}
        result=dict(step=step,split=split,assistant_nll=assistant_nll,components=nll,supervised_tokens=counts,
            checked_passed=sum(s['passed'] is True for s in samples),checked_total=sum(s['passed'] is not None for s in samples),
            generated_rates=generated_rates,generated_macro_pass=sum(generated_rates.values())/len(generated_rates),
            natural_stop=sum(s['stop_reason'] in ('turn_end','document_end') for s in samples)/len(samples),
            repetition_rate=behavior['repetition_hits']/len(samples),behavior=behavior,knowledge=knowledge_result,samples=samples,
            retention=dict(val_loss=sum(losses[s]*w for s,w in receipt['weights'].items())/sum(receipt['weights'].values()),
                sources=losses,validation_fingerprint=receipt['validation_fingerprint']),
            scope='Separate authored task combinations, natural held-out behavior, fixed development knowledge and original prose retention. Rule checks and repetition heuristics cannot certify all semantic correctness or unseen abilities.')
        atomic_json(output/('evaluations' if split=='dev' else 'final_test')/f'eval_{step:07d}.json',result)
        print(f'[eval-sft {step}] assistant_nll={assistant_nll:.6f} generated_macro_pass={result["generated_macro_pass"]:.3f} natural_stop={result["natural_stop"]:.3f} repetition_rate={result["repetition_rate"]:.3f}',flush=True)
        print(f'[eval-behavior {step}] {behavior} knowledge={knowledge_result["correct"]}/{knowledge_result["total"]}',flush=True)
        print(f'[eval-retention {step}] val_loss={result["retention"]["val_loss"]:.6f} sources={losses}',flush=True)
        for sample in samples[:8]:print(f'[sample-sft {step}] component={sample["component"]} passed={sample["passed"]} stop={sample["stop_reason"]}\n{sample["text"]}',flush=True)
        return {k:({a:b for a,b in v.items() if a!='details'} if k=='knowledge' else v) for k,v in result.items() if k!='samples'}
    finally:model.train()
