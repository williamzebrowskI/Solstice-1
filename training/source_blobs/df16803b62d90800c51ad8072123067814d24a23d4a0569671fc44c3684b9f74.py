"""One bounded repair run, with dev-only selection and exact adapter ancestry."""
import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v1.common import read_json, atomic_json, fingerprint
from sml_v1.tokenization import Tokenizer
from .data import learning_rate
from .lora_repair_common import SHARED, setup, receipt, selection, severe
from .lora_search_train import gradients
from .lora_replay_objective import batch, log_probs, backward, orders
from .lora_adapter import dense_model, assert_frozen, base_digest
from .lora_checkpoint import save, restore, resolve
from .lora_repair_eval import evaluate as repair_evaluate
from .text_eval import evaluate, reference_inference


def run(staging,output):
    plan=read_json(staging/'plan.json');contract=read_json(staging/'contract.json')
    if contract['repair_data']!=fingerprint(receipt()):raise ValueError('Repair data changed')
    tok=Tokenizer(staging/'tokenizer')
    if tok.fingerprint!=contract['tokenizer']:raise ValueError('Tokenizer changed')
    mx.random.seed(plan['seed']);mx.set_cache_limit(12*2**30)
    model,opt,base,cfg,meta,digest,count,ancestry=setup(staging,plan,contract)
    ancestry['sft_stage']=read_json(staging/'initial_parent/state.json').get('sft_stage',1)+1
    reference,refopt,refbase,_,_,_,_,_=setup(staging,plan,contract)
    del refopt,refbase
    reference.freeze();reference.eval();mx.eval(reference.parameters())
    reference_digest=base_digest(dict(tree_flatten(reference.parameters())))
    fresh=[json.loads(l) for l in (SHARED/'data/train.jsonl').read_text().splitlines()]
    replay=[json.loads(l) for l in (SHARED/'data/replay.jsonl').read_text().splitlines()]
    fresh_order,replay_order=orders(tok,fresh,replay,plan);order_hash=fingerprint([fresh_order,replay_order])
    total=len(fresh_order)
    state=dict(step=0,tokens=0,assistant_tokens=0,contract=contract,order_hash=order_hash,
               source_step=meta['step'],source_tokens=meta['tokens'],validation_fingerprint=meta['validation_fingerprint'],
               adaptation=plan['adaptation'],trainable_parameters=count,quality_stop=False,failed_eval_count=0,total_steps=total,
               has_eligible_candidate=False,selection_key=None,method=plan['method'],kl_weight=plan['kl_weight'],dpo_weight=0,**ancestry)
    if (output/'latest.json').exists():state=restore(resolve(output/'latest.json'),model,opt,contract,order_hash)
    atomic_json(output/'jobs/local/worker_rank0.json',dict(pid=os.getpid()))
    atomic_json(output/'worker.json',dict(pid=os.getpid(),steps=total,parent=ancestry['parent_checkpoint'],method=plan['method']))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:(output/'STOP').touch())
    heartbeat=[time.monotonic()];finished=threading.Event()
    def beat():heartbeat[0]=time.monotonic()
    def watchdog():
        while not finished.wait(10):
            if time.monotonic()-heartbeat[0]>1800:os._exit(70)
    threading.Thread(target=watchdog,daemon=True).start()
    new_backward=backward(model,0.);replay_backward=backward(model,plan['kl_weight'])
    last_saved=state['step'] if state['step'] else -1;started=time.monotonic()
    try:
        if not state['step']:
            x,y,_,_=batch(tok,replay[:plan['microbatch_rows']],False,plan['max_sequence_length'])
            a,b=log_probs(model(x)['logits']),log_probs(reference(x)['logits']);mx.eval(a,b)
            if not bool(mx.all(a==b).item()):raise ValueError('Policy/reference initialization differs')
            atomic_json(output/'initialization.json',dict(exact_parent_fp32=True,training_reference_exact=True,
                        optimizer='fresh',parent_step=state['parent_step'],parent_manifest=state['parent_manifest'],
                        order_hash=order_hash,selected_fresh=len(fresh_order)*plan['rows_per_group'],selected_replay=len(replay_order)*plan['rows_per_group']))
        print(f'[repair-start] {plan["method"]} parent={plan["repair_parent"]} steps={total}',flush=True)
        while state['step']<total and not (output/'STOP').exists():
            beat();step=state['step']
            selected=[fresh[i] for i in fresh_order[step]];repeated=[replay[i] for i in replay_order[step]]
            values,grad,norm,tokens,assistant_tokens=gradients(model,reference,tok,selected,repeated,plan,new_backward,replay_backward)
            lr=learning_rate(step+1,total,plan['peak_lr'],plan['final_lr'],plan['warmup_fraction'])
            opt.learning_rate=lr;opt.update(model,grad);mx.eval(model.parameters(),opt.state)
            state['step']+=1;state['tokens']+=tokens;state['assistant_tokens']+=assistant_tokens;step=state['step']
            if step==1 or step%plan['log_every']==0:
                elapsed=time.monotonic()-started
                print(f'[sft {step}/{total}] assistant_loss={values[1]:.6f} objective_loss={values[0]:.6f} replay_loss={values[2]:.6f} lr={lr:.9e} grad_norm={norm:.4f} input_tokens={state["tokens"]} assistant_tokens={state["assistant_tokens"]} elapsed={elapsed:.1f}s tok/s={state["tokens"]/max(elapsed,1e-9):,.0f} replay_kl={values[3]:.6f}',flush=True)
            if step%plan['eval_every']==0 or step==total:
                assert_frozen(model,digest);dense=dense_model(model,opt,base,cfg)
                new=repair_evaluate(dense,tok,output/'evaluations'/f'repair_{step:07d}.json',heartbeat=beat)
                with reference_inference(dense):
                    old=evaluate(dense,tok,staging/'corpus',plan,step,output,staging/'retention',heartbeat=beat,precision_label='fp32-chat-repair')
                del dense;mx.clear_cache();check=selection(new,old)
                state['failed_eval_count']=state['failed_eval_count']+1 if severe(new,old) else 0
                state['quality_stop']=state['failed_eval_count']>=2
                best=state['selection_key'] is None or check['key']>state['selection_key']
                if best:state.update(selection_key=check['key'],has_eligible_candidate=check['eligible'])
                atomic_json(output/'evaluations'/f'quality_{step:07d}.json',dict(check,consecutive_severe=state['failed_eval_count']))
                path=save(output,model,opt,base,cfg,state,best=best);last_saved=step
                print(f'[repair-quality] step={step} new={new["correct"]}/{new["total"]} legacy={sum(old["generated_pass_counts"].values())}/320 eligible={check["eligible"]} best={best} checkpoint={path.name}',flush=True)
                keep={resolve(output/'latest.json'),resolve(output/'best.json')}
                if not (output/'KEEP.json').exists():
                    import shutil
                    for p in output.glob('step_*'):
                        if p.is_dir() and p not in keep and not p.is_symlink():shutil.rmtree(p)
                if state['quality_stop']:break
        assert_frozen(model,digest)
        if reference_digest!=base_digest(dict(tree_flatten(reference.parameters()))):raise ValueError('Frozen reference changed')
        if last_saved!=state['step']:save(output,model,opt,base,cfg,state)
        status='quality-stopped' if state['quality_stop'] else 'complete' if state['step']==total else 'stopped'
        atomic_json(output/'result.json',dict(state,status=status));atomic_json(output/'jobs/local/result.json',dict(state,status=status))
        print(f'[finished-repair] status={status} step={state["step"]}',flush=True)
    finally:finished.set()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
