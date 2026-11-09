"""Bounded fresh-conversation SFT update from immutable parent 2000 adapters."""
import argparse,json,math,os,signal,threading,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map,tree_flatten
from sml_v1.common import read_json,atomic_json,fingerprint,file_sha256
from sml_v1.tokenization import Tokenizer
from sml_v1.pretrain import clip_gradients
from .data import learning_rate,training_order
from .lora_search_common import setup,data_receipt,load_baseline,gate,severe,correct,SHARED
from .lora_adapter import dense_model,assert_frozen,base_digest
from .lora_checkpoint import save,restore,resolve
from .lora_replay_objective import batch,log_probs,backward,orders
from .text_eval import evaluate,reference_inference

def gradients(model,reference,tokenizer,fresh,replay,plan,new_backward,replay_backward):
    total=None;values=[0.,0.,0.,0.];tokens=assistant_tokens=0
    for rows,final_only,weight,one,use_kl in [(fresh,True,plan['fresh_weight'],new_backward,False),(replay,False,1-plan['fresh_weight'],replay_backward,plan['kl_weight']>0)]:
        for start in range(0,len(rows),plan['microbatch_rows']):
            part=rows[start:start+plan['microbatch_rows']]
            x,y,inputs,targets=batch(tokenizer,part,final_only,plan['max_sequence_length'])
            teacher=mx.stop_gradient(log_probs(reference(x)['logits'])) if use_kl else mx.array(0.,mx.float32)
            mx.eval(teacher)
            (loss,(nll,kl)),grad=one(x,y,teacher);mx.eval(loss,nll,kl,grad)
            if not all(math.isfinite(x.item()) for x in (loss,nll,kl)):raise ValueError('Nonfinite objective')
            share=weight*len(part)/len(rows)
            grad=tree_map(lambda x:x.astype(mx.float32)*share,grad)
            total=grad if total is None else tree_map(lambda a,b:a+b,total,grad);mx.eval(total)
            values[0]+=loss.item()*share
            values[1 if final_only else 2]+=nll.item()*len(part)/len(rows)
            if use_kl:values[3]+=kl.item()*len(part)/len(rows)
            tokens+=inputs;assistant_tokens+=targets
    total,norm=clip_gradients(total,plan['grad_clip']);mx.eval(total,norm)
    if not math.isfinite(norm.item()) or not all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(total)):raise ValueError('Nonfinite gradient')
    return values,total,float(norm.item()),tokens,assistant_tokens

def run(staging,output):
    plan=read_json(staging/'plan.json');contract=read_json(staging/'contract.json')
    if contract['fresh_data']!=fingerprint(data_receipt()):raise ValueError('Fresh-data contract changed')
    tok=Tokenizer(staging/'tokenizer')
    if tok.fingerprint!=contract['tokenizer']:raise ValueError('Tokenizer changed')
    mx.random.seed(plan['seed']);mx.set_cache_limit(12*2**30)
    model,opt,base,cfg,meta,digest,count,ancestry=setup(staging,plan,contract)
    reference,refopt,refbase,_,_,refdigest,_,_=setup(staging,plan,contract)
    del refopt,refbase
    reference.freeze();reference.eval();mx.eval(reference.parameters())
    reference_before=base_digest(dict(tree_flatten(reference.parameters())))
    fresh=[json.loads(l) for l in (SHARED/'fresh_data/train.jsonl').open()]
    replay=[json.loads(l) for l in (SHARED/'fresh_data/replay.jsonl').open()]
    fresh_order,replay_order=orders(tok,fresh,replay,plan)
    order_hash=fingerprint([fresh_order,replay_order]);baseline=load_baseline(staging,'dev')
    total_steps=len(fresh_order)
    state=dict(step=0,tokens=0,assistant_tokens=0,contract=contract,order_hash=order_hash,source_step=meta['step'],
               source_tokens=meta['tokens'],validation_fingerprint=meta['validation_fingerprint'],adaptation=plan['adaptation'],
               trainable_parameters=count,quality_stop=False,failed_eval_count=0,total_steps=total_steps,selection_baseline={k:v for k,v in baseline.items() if k!='samples'},
               has_eligible_candidate=False,selection_key=None,method=plan['method'],dpo_weight=0,kl_weight=plan['kl_weight'],**ancestry)
    if (output/'latest.json').exists():state=restore(resolve(output/'latest.json'),model,opt,contract,order_hash)
    atomic_json(output/'jobs/local/worker_rank0.json',dict(pid=os.getpid()))
    atomic_json(output/'worker.json',dict(pid=os.getpid(),steps=total_steps,parent=ancestry['parent_checkpoint'],method=plan['method']))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:(output/'STOP').touch())
    heartbeat=[time.monotonic()];finished=threading.Event()
    def watchdog():
        while not finished.wait(10):
            if time.monotonic()-heartbeat[0]>1800:os._exit(70)
    threading.Thread(target=watchdog,daemon=True).start()
    new_backward=backward(model,0.);replay_backward=backward(model,plan['kl_weight'])
    last_saved=state['step'] if state['step'] else -1;started=time.monotonic()
    try:
        if not state['step']:
            # This checks the actual BF16-base/FP32-adapter training path,
            # not the numerically different dense FP32 inference export.
            x,y,_,_=batch(tok,replay[:plan['microbatch_rows']],False,plan['max_sequence_length'])
            p,r=log_probs(model(x)['logits']),log_probs(reference(x)['logits']);mx.eval(p,r)
            if not bool(mx.all(p==r).item()):raise ValueError('Policy/reference do not start identically')
            atomic_json(output/'initialization.json',dict(exact_parent_fp32=True,training_reference_exact=True,optimizer='fresh',parent_step=2000))
        print(f'[search-start] {plan["method"]} parent=2000 fresh={len(fresh)} replay={len(replay)} steps={total_steps}',flush=True)
        while state['step']<total_steps and not (output/'STOP').exists():
            heartbeat[0]=time.monotonic();step=state['step']
            selected=[fresh[i] for i in fresh_order[step]]
            repeated=[replay[i] for i in replay_order[step]]
            values,grad,norm,tokens,assistant_tokens=gradients(model,reference,tok,selected,repeated,plan,new_backward,replay_backward)
            lr=learning_rate(step+1,total_steps,plan['peak_lr'],plan['final_lr'],plan['warmup_fraction'])
            opt.learning_rate=lr;opt.update(model,grad);mx.eval(model.parameters(),opt.state)
            state['step']+=1;state['tokens']+=tokens;state['assistant_tokens']+=assistant_tokens;step=state['step']
            if step==1 or step%plan['log_every']==0:
                elapsed=time.monotonic()-started
                print(f'[sft {step}/{total_steps}] assistant_loss={values[1]:.6f} objective_loss={values[0]:.6f} replay_loss={values[2]:.6f} lr={lr:.9e} grad_norm={norm:.4f} input_tokens={state["tokens"]} assistant_tokens={state["assistant_tokens"]} elapsed={elapsed:.1f}s tok/s={state["tokens"]/max(elapsed,1e-9):,.0f} replay_kl={values[3]:.6f}',flush=True)
            if step%plan['eval_every']==0 or step==total_steps:
                assert_frozen(model,digest)
                dense=dense_model(model,opt,base,cfg)
                with reference_inference(dense):
                    result=evaluate(dense,tok,staging/'corpus',plan,step,output,staging/'retention',heartbeat=lambda:heartbeat.__setitem__(0,time.monotonic()),precision_label='fp32-parent2000-fresh-sft')
                del dense;mx.clear_cache();check=gate(result,baseline,split='dev')
                state['failed_eval_count']=state['failed_eval_count']+1 if severe(result,baseline) else 0
                state['quality_stop']=state['failed_eval_count']>=2
                # Keep only dev-selected best plus latest; final/test is never
                # used to choose among this trial's intermediate checkpoints.
                key=[correct(result),result['natural_stop'],-result['repetition_rate'],-result['assistant_nll']]
                best=check['passed'] and (state['selection_key'] is None or key>state['selection_key'])
                if best:state.update(selection_key=key,has_eligible_candidate=True)
                atomic_json(output/'evaluations'/f'quality_{step:07d}.json',dict(check,consecutive_severe=state['failed_eval_count']))
                path=save(output,model,opt,base,cfg,state,best=best);last_saved=step
                print(f'[quality] step={step} checked={correct(result)}/320 gain={check["gain"]} eligible={check["passed"]} checkpoint={path.name}',flush=True)
                # Remove superseded exports within this disposable trial only.
                keep={resolve(output/'latest.json')}
                if (output/'best.json').exists():keep.add(resolve(output/'best.json'))
                if not (output/'KEEP.json').exists():
                    import shutil
                    for p in output.glob('step_*'):
                        if p.is_dir() and p not in keep and not p.is_symlink():shutil.rmtree(p)
                if state['quality_stop']:break
        assert_frozen(model,digest)
        if base_digest(dict(tree_flatten(reference.parameters())))!=reference_before:raise ValueError('Frozen parent reference changed')
        if last_saved!=state['step']:save(output,model,opt,base,cfg,state)
        status='quality-stopped' if state['quality_stop'] else 'complete' if state['step']==total_steps else 'stopped'
        atomic_json(output/'result.json',dict(state,status=status))
        atomic_json(output/'jobs/local/result.json',dict(state,status=status))
        print(f'[finished-search-trial] status={status} step={state["step"]}',flush=True)
    finally:finished.set()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
