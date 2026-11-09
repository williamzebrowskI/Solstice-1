"""Frozen grouped dev checks, with a separate test used only after training."""
import json
import math
from pathlib import Path
import mlx.core as mx
import numpy as np
from sml_v1.common import atomic_json,read_json
from .data import Corpus
from .evaluate import generate,conversation_losses
from .quality import repeated
from .retention import verify as verify_retention


def answer_key(value):return ' '.join(value.strip().strip('"').rstrip('.!?').casefold().split())


def checked_answer(text,verification,stop_reason):
    return bool(stop_reason in ('turn_end','document_end') and text.strip() and not repeated(text)
        and answer_key(text) in {answer_key(a) for a in verification['answers']})


def selection(result,baseline,max_increase):
    eligible=result['retention']['val_loss']<=baseline+max_increase
    return [int(eligible),result['generated_macro_pass'],result['natural_stop'],-result['assistant_nll']]


def evaluate(model,tokenizer,data_dir,plan,step,output,retention,*,split='dev'):
    data_dir=Path(data_dir);output=Path(output);corpus=Corpus(data_dir,split)
    rows=[json.loads(s) for s in (data_dir/(split+'.jsonl')).read_text().splitlines()]
    components=sorted({r['component'] for r in rows});per_component=max(1,plan['eval_examples']//len(components))
    nll,counts,samples,rates={},{},[],{}
    model.eval()
    try:
        for component in components:
            indices=[i for i,r in enumerate(rows) if r['component']==component][:per_component]
            total,count,tokens=0.,0,0
            for start in range(0,len(indices),2):
                x,y=corpus.batch(indices[start:start+2]);values,n=conversation_losses(model,mx.array(x),mx.array(y))
                total+=float(values.sum().item());count+=len(x);tokens+=int(n.sum().item())
            nll[component]=total/count;counts[component]=tokens
            candidates=[i for i in indices if rows[i].get('verification')][:plan['generated_eval_per_component']]
            if not candidates:candidates=indices[:2]
            checked=[]
            for i in candidates:
                row=rows[i];value=generate(model,tokenizer,row['messages'][:-1],limit=96)
                passed=checked_answer(value['text'],row['verification'],value['stop_reason']) if row.get('verification') else None
                if passed is not None:checked.append(passed)
                samples.append(dict(component=component,messages=row['messages'][:-1],reference=row['messages'][-1]['content'],
                    **value,repeated=repeated(value['text']),passed=passed))
            if checked:rates[component]=sum(checked)/len(checked)
        assistant_nll=sum(nll.values())/len(nll)
        if not math.isfinite(assistant_nll):raise RuntimeError('Nonfinite grounded SFT validation')
        receipt=verify_retention(retention);losses={}
        with np.load(Path(retention)/'panel.npz',allow_pickle=False) as panel:
            for source in receipt['weights']:
                values=[float(model(mx.array(x),targets=mx.array(y))['loss'].item()) for x,y in zip(panel[source+'_x'],panel[source+'_y'])]
                losses[source]=sum(values)/len(values)
        result=dict(step=step,split=split,assistant_nll=assistant_nll,components=nll,supervised_tokens=counts,
            generated_rates=rates,generated_macro_pass=sum(rates.values())/len(rates),
            natural_stop=sum(s['stop_reason'] in ('turn_end','document_end') for s in samples)/len(samples),
            repetition_rate=sum(s['repeated'] for s in samples)/len(samples),samples=samples,
            retention=dict(val_loss=sum(losses[s]*w for s,w in receipt['weights'].items())/sum(receipt['weights'].values()),sources=losses,validation_fingerprint=receipt['validation_fingerprint']),
            scope='Exact short-answer/span and generated-template checks; human/dialogue outputs require review. This is not a general factual-accuracy benchmark.')
        atomic_json(output/('evaluations' if split=='dev' else 'final_test')/f'eval_{step:07d}.json',result)
        print(f'[eval-sft {step}] assistant_nll={assistant_nll:.6f} generated_macro_pass={result["generated_macro_pass"]:.3f} natural_stop={result["natural_stop"]:.3f} repetition_rate={result["repetition_rate"]:.3f}',flush=True)
        print(f'[eval-retention {step}] val_loss={result["retention"]["val_loss"]:.6f} sources={losses}',flush=True)
        for sample in samples[:8]:print(f'[sample-sft {step}] component={sample["component"]} passed={sample["passed"]} stop={sample["stop_reason"]}\n{sample["text"]}',flush=True)
        return {k:v for k,v in result.items() if k!='samples'}
    finally:model.train()
