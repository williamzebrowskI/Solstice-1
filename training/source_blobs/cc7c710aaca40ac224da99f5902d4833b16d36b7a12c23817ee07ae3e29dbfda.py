"""FP32 text evaluation with checked answers separate from qualitative chat."""
import json
import math
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten,tree_unflatten
from sml_v1.common import atomic_json
from .data import Corpus,serialize
from .evaluate import generate,conversation_losses
from .quality_gate import looping
from .text_checks import answer_pass
from .retention import verify as verify_retention


@contextmanager
def reference_inference(model):
    """Use the same reference operators as the final FP32 benchmark suite."""
    saved=[]
    for obj,key,value in [(model.cfg,'attention_impl','vanilla'),(model.cfg,'ffn_impl','reference'),(model.cfg,'ce_impl','reference')]+[
            item for block in model.blocks for item in [(block.attn,'attention_impl','vanilla'),(block.ffn,'ffn_impl','reference')]]:
        saved.append((obj,key,getattr(obj,key)));setattr(obj,key,value)
    try:yield
    finally:
        for obj,key,value in reversed(saved):setattr(obj,key,value)


def checked_panel(rows,indices,limit):
    """Round-robin task families; balance answerable and missing-answer passages."""
    from collections import defaultdict
    groups=defaultdict(list)
    for i in indices:
        row=rows[i];v=row['verification']
        group=str(v.get('answerable')) if v['kind']=='span' else row.get('provenance',{}).get('family',v.get('operation','other'))
        groups[group].append(i)
    selected=[]
    while any(groups.values()) and len(selected)<limit:
        for group in sorted(groups):
            if groups[group]:selected.append(groups[group].pop(0))
            if len(selected)==limit:break
    return selected


def with_masters(model,optimizer,fn):
    saved=model.parameters();masters={n[:-7]:v for n,v in tree_flatten(optimizer.state) if n.endswith('.master')}
    if set(masters)!=set(dict(tree_flatten(saved))):raise ValueError('Missing exact evaluation masters')
    try:
        model.update(tree_unflatten(list(masters.items())));mx.eval(model.parameters())
        with reference_inference(model):return fn()
    finally:model.update(saved);mx.eval(model.parameters());model.train();mx.clear_cache()


def evaluate(model,tokenizer,data_dir,plan,step,output,retention,*,split='dev',heartbeat=None,precision_label='exact-fp32-masters'):
    data_dir=Path(data_dir);output=Path(output);corpus=Corpus(data_dir,split)
    if any(v.dtype!=mx.float32 for _,v in tree_flatten(model.parameters())):raise ValueError('Evaluation must use exact FP32 masters')
    rows=[json.loads(l) for l in (data_dir/(split+'.jsonl')).read_text().splitlines()]
    nll={};samples=[];checked={c:[] for c in plan['checked_components']};model.eval()
    def progress(component,done,total):
        if heartbeat is not None:heartbeat()
        atomic_json(output/'evaluation_progress.json',dict(step=step,split=split,component=component,completed=done,total=total))
        print(f'[eval-progress {step}] {split} {component} {done}/{total}',flush=True)
    try:
        for component in plan['components']:
            indices=[i for i,r in enumerate(rows) if r['component']==component]
            selected=indices[:plan['eval_examples_per_component']] if split=='dev' else indices
            progress(component+'-loss',0,1);total=0.
            for start in range(0,len(selected),2):
                x,y=corpus.batch(selected[start:start+2]);values,_=conversation_losses(model,mx.array(x),mx.array(y));total+=float(values.sum().item())
            nll[component]=total/len(selected)
            verified=[i for i in indices if rows[i].get('verification') and rows[i]['verification'].get('selection_eligible',True)]
            qualitative=[i for i in indices if not rows[i].get('verification') or not rows[i]['verification'].get('selection_eligible',True)][:plan.get('qualitative_eval_per_component',8)]
            generated=(checked_panel(rows,verified,plan['generated_eval_per_component']) if split=='dev' else verified)+qualitative
            progress(component,0,len(generated))
            for number,i in enumerate(generated,1):
                row=rows[i];prompt=row['messages'][:-1];limit=plan['answer_max_tokens'][component]+1
                ids,_=serialize(tokenizer,prompt,generation=True)
                if len(ids)+limit>model.cfg.max_seq_len:raise ValueError('Generation would truncate')
                value=generate(model,tokenizer,prompt,limit=limit)
                natural=value['stop_reason'] in ('turn_end','document_end');repeat=looping(value['text']);empty=not value['text'].strip();matched=None;passed=None
                if row.get('verification') and row['verification'].get('selection_eligible',True):
                    matched=answer_pass(value['text'],row['verification']);passed=matched and natural and not repeat and not empty;checked[component].append(passed)
                samples.append(dict(component=component,subset=row['subset'],messages=prompt,reference=row['messages'][-1]['content'],
                                    **value,answer_match=matched,passed=passed,repeated=repeat,empty=empty))
                if number%8==0 or number==len(generated):progress(component,number,len(generated))
        progress('prose-retention',0,1);receipt=verify_retention(retention);losses={}
        with np.load(Path(retention)/'panel.npz',allow_pickle=False) as panel:
            for source in receipt['weights']:
                values=[float(model(mx.array(x),targets=mx.array(y))['loss'].item()) for x,y in zip(panel[source+'_x'],panel[source+'_y'])]
                losses[source]=sum(values)/len(values)
        if not all(math.isfinite(v) for v in [*nll.values(),*losses.values()]):raise ValueError('Nonfinite evaluation')
        if not all(checked.values()):raise ValueError('Missing checked domain')
        rates={k:sum(v)/len(v) for k,v in checked.items()}
        result=dict(step=step,split=split,precision=precision_label,assistant_nll=sum(nll.values())/len(nll),components=nll,
                    generated_rates=rates,generated_pass_counts={k:sum(v) for k,v in checked.items()},generated_counts={k:len(v) for k,v in checked.items()},
                    generated_macro_pass=sum(rates.values())/len(rates),natural_stop=sum(s['stop_reason'] in ('turn_end','document_end') for s in samples)/len(samples),
                    repetition_rate=sum(s['repeated'] for s in samples)/len(samples),empty_rate=sum(s['empty'] for s in samples)/len(samples),
                    stop_reason_counts=dict(Counter(s['stop_reason'] for s in samples)),
                    turn_end_rate=sum(s['stop_reason']=='turn_end' for s in samples)/len(samples),
                    replacement_character_rate=sum('\ufffd' in s['text'] for s in samples)/len(samples),
                    qualitative_count=sum(s['passed'] is None for s in samples),samples=samples,
                    retention=dict(val_loss=sum(losses[k]*w for k,w in receipt['weights'].items())/sum(receipt['weights'].values()),sources=losses,validation_fingerprint=receipt['validation_fingerprint']),
                    scope='Checked grounded answers, deterministic tasks and authored conversation context. Public general/dialogue/social outputs require semantic review. No code, SQL, JSON, or public benchmark scores select checkpoints.')
        atomic_json(output/('evaluations' if split=='dev' else 'final_test')/f'eval_{step:07d}.json',result);progress('complete',1,1)
        print(f'[eval-sft {step}] assistant_nll={result["assistant_nll"]:.6f} checked={rates} natural_stop={result["natural_stop"]:.3f} repetition={result["repetition_rate"]:.3f}',flush=True)
        print(f'[eval-retention {step}] prose={result["retention"]["val_loss"]:.6f}',flush=True)
        for sample in samples[:6]:print(f'[sample-sft {step}] {sample["component"]} checked={sample["passed"]} stop={sample["stop_reason"]}\n{sample["text"]}',flush=True)
        return {k:v for k,v in result.items() if k!='samples'}
    finally:model.train()
