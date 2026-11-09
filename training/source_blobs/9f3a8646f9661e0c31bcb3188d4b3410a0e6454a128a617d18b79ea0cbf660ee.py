"""Full-parameter SFT with deterministic batches, checkpoint resume and quality gates."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v1.checkpoint_bundle import resolve_bundle, save_bundle
from sml_v1.jaccl_control import ControlChannel
from sml_v1.model import TransformerConfig, TransformerLM, count_parameters
from sml_v1.precision import MasterAdamW
from sml_v1.pretrain import clip_gradients
from sml_v1.tokenization import Tokenizer
from .data import Corpus, learning_rate, training_order
from .grounded_corpus import verify
from .mixed import MixedBatcher


def state_digest(model, optimizer):
    """Canonical tensor order remains stable after safetensors reload."""
    result=hashlib.sha256()
    for prefix,state in (('model',model.parameters()),('optimizer',optimizer.state)):
        for name,value in sorted(tree_flatten(state)):
            result.update((prefix+name+str(value.shape)+str(value.dtype)).encode())
            result.update(np.asarray(value.astype(mx.float32) if value.dtype==mx.bfloat16 else value).tobytes())
    return result.hexdigest()


def weighted_backward(model, accum, *, compile_one=True):
    """Sum token losses/global token count, including unequal rank/micro masks.

    Rank contributions are SUMMED across the ring. Averaging rank-local means
    would overweight short answers and partially padded final batches.
    """
    def objective(x, y, denominator):
        return model(x, targets=y)['loss'] * (y != -100).sum() / denominator
    one = nn.value_and_grad(model, objective)
    if compile_one:
        one = mx.compile(one, inputs=[model.state], outputs=[model.state])
    def backward(xs, ys, denominator):
        total, value = None, mx.array(0., mx.float32)
        for i in range(accum):
            loss, grads = one(xs[i], ys[i], mx.array(denominator, mx.float32))
            grads = tree_map(lambda x: x.astype(mx.float32), grads)
            total = grads if total is None else tree_map(lambda a, b: a+b, total, grads)
            value = value + loss.astype(mx.float32)
            mx.eval(total, value)
        return value, total
    return backward


def mixed_backward(model,accum,plan,*,compile_one=True):
    def objective(x,y,weights,replay):
        token_loss=model(x,targets=y)['token_loss']
        weighted=token_loss*weights
        sft=(weighted*(~replay[:,None])).sum()
        document=(weighted*replay[:,None]).sum()
        return sft+document,sft/plan['sft_objective_weight'],document/plan['replay_objective_weight']
    one=nn.value_and_grad(model,objective)
    if compile_one:one=mx.compile(one,inputs=[model.state],outputs=[model.state])
    def backward(xs,ys,weights,replay):
        total,values=None,[mx.array(0.,mx.float32) for _ in range(3)]
        for i in range(accum):
            metrics,grads=one(xs[i],ys[i],weights[i],replay[i])
            grads=tree_map(lambda x:x.astype(mx.float32),grads)
            total=grads if total is None else tree_map(lambda a,b:a+b,total,grads)
            values=[a+b.astype(mx.float32) for a,b in zip(values,metrics)];mx.eval(total,values)
        return mx.stack(values),total
    return backward


def order_digest(order):
    h = hashlib.sha256()
    for indices in order:
        h.update(np.asarray(indices, dtype='<i8').tobytes())
        h.update(b'|')
    return h.hexdigest()


from .grounded_eval import evaluate, selection


def run(resources):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'data', 'tokenizer', 'source', 'save-dir', 'job-dir', 'contract', 'retention'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--source-masters', type=Path)
    p.add_argument('--reference',type=Path)
    args = p.parse_args()
    plan = read_json(args.config)
    evaluate_run=evaluate
    fresh=plan.get('format')=='opensml-deep44-curriculum-sft-v4'
    balanced=fresh or plan.get('format')=='opensml-deep44-balanced-sft-v3'
    if balanced:
        from .balanced_eval import evaluate as evaluate_run
        from .quality_gate import compact_baseline,eligibility,selection as balanced_selection
        from .quality_gate import fresh_progress
    args.job_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.job_dir/f'worker_rank{os.getenv("MLX_RANK", "0")}.json', dict(pid=os.getpid(),output=str(args.job_dir)))
    batches, accum = plan['batches'], plan['grad_accum']
    if len(batches)>1 and os.getenv('MLX_JACCL_RING')!='1':
        raise ValueError('Use the verified five-Mac launcher')
    group = mx.distributed.init(backend='jaccl', strict=True) if len(batches)>1 else mx.distributed.init(strict=False)
    rank, world = group.rank(), group.size()
    if world != len(batches): raise ValueError('SFT rank count differs from recipe')
    mx.set_cache_limit(int(plan.get('cache_limit_gib',24 if rank<2 else 8)*2**30))
    control = ControlChannel(mx, group, ring_mode='native')
    heartbeat, finished = [time.monotonic()], threading.Event()
    resources.callback(finished.set)
    def watchdog():
        while not finished.wait(10):
            if time.monotonic()-heartbeat[0]>1800:
                print('[fatal] SFT made no progress for 30 minutes', flush=True); os._exit(70)
    threading.Thread(target=watchdog, daemon=True).start()
    def root_call(label, fn):
        value, error = None, None
        if rank==0:
            try: value=fn()
            except Exception as exc: error=f'{type(exc).__name__}: {exc}'[:1000]
        error=control.broadcast(error, label+'-status')
        if error: raise RuntimeError(error)
        return value
    contract = read_json(args.contract)
    identity=fingerprint(contract)
    if any(c!=identity for c in control.exchange(identity, 'contract')):
        raise ValueError('SFT worker contracts differ')
    root=Path(__file__).resolve().parents[1]
    modules={str(p.relative_to(root)):file_sha256(p) for package in ('sml_v1','sft')
             for p in sorted((root/package).rglob('*.py'))}
    if modules!=contract['modules']:
        raise ValueError('SFT worker code differs from the staged contract')
    tokenizer = Tokenizer(args.tokenizer)
    manifest = root_call('corpus-verify', lambda: verify(args.data, plan))
    if tokenizer.fingerprint!=contract['tokenizer']:
        raise ValueError('Frozen SFT tokenizer changed')
    source = read_json(args.source/'model.safetensors.json')
    from .retention import verify as verify_retention
    from .parent import context,initialize_masters
    ancestry=context(source,args.retention)
    verify_retention(args.retention,ancestry['validation_fingerprint'])
    if file_sha256(args.retention/'manifest.json')!=contract['retention']:
        raise ValueError('SFT retention contract changed')
    if file_sha256(args.source/'model.safetensors.json')!=contract['source_metadata']:
        raise ValueError('SFT source model metadata changed')
    if file_sha256(args.source/'model.safetensors')!=contract['source_weights']:
        raise ValueError('SFT parent weights changed')
    if fresh:
        if args.reference is None:raise ValueError('Fresh SFT requires the Part 1 reference')
        for name,key in (('model.safetensors','weights'),('model.safetensors.json','metadata')):
            if file_sha256(args.reference/name)!=contract['reference'][key]:raise ValueError('Part 1 reference changed')
        reference_metadata=read_json(args.reference/'model.safetensors.json')
        context(reference_metadata,args.retention)
        if reference_metadata['recipe']['model']!=source['recipe']['model']:raise ValueError('Reference architecture differs')
    mx.random.seed(plan['seed'])
    model = TransformerLM(TransformerConfig(**source['recipe']['model']))
    model.load_weights(str(args.source/'model.safetensors'))
    optimizer = MasterAdamW(learning_rate=plan['peak_lr'], betas=tuple(plan['betas']),
        eps=plan['eps'], weight_decay=plan['weight_decay'])
    optimizer.init(model.trainable_parameters())
    if contract.get('source_masters'):
        if args.source_masters is None or file_sha256(args.source_masters)!=contract['source_masters']:
            raise ValueError('SFT parent FP32 masters changed')
        if not args.resume:initialize_masters(model,optimizer,args.source_masters)
    step, tokens, assistant_tokens, best = 0, 0, 0, None
    best_key,baseline_retention,quality_stop=None,None,False
    selection_baseline,failed_eval_count=None,0
    has_candidate=False
    train = Corpus(args.data, 'train')
    order = training_order(train.lengths, plan['seed'], plan['sft_rows_per_update'], plan['epochs'])
    mixed=MixedBatcher(train,args.data,plan)
    if len(order)*plan['replay_rows_per_update']>len(mixed.replay):raise ValueError('Replay pool cannot cover this run without recycling')
    order_hash = order_digest(order)
    if args.resume:
        weights = resolve_bundle(args.resume)
        metadata = read_json(weights+'.json')
        if metadata.get('sft_contract')!=contract or metadata.get('order_hash')!=order_hash:
            raise ValueError('SFT resume code/data/recipe/order differs')
        if metadata.get('quality_stop'):raise ValueError('Quality-stopped SFT requires a new reviewed recipe, not automatic resume')
        model.load_weights(weights)
        optimizer.state = tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
        step, tokens, assistant_tokens, best = (metadata[k] for k in ('step','tokens','assistant_tokens','best_val_loss'))
        best_key,baseline_retention=metadata['selection_key'],metadata['baseline_retention']
        if balanced:
            selection_baseline=metadata['selection_baseline'];failed_eval_count=metadata['failed_eval_count']
            has_candidate=metadata.get('has_eligible_candidate',True)
        with gzip.open(weights+'.rank0.data_state.json.gz','rt') as f: cursor=json.load(f)
        if cursor!={'step':step,'stream_state':dict(next_batch=step,order_hash=order_hash)}:
            raise ValueError('SFT saved batch position differs')
    mx.eval(model.parameters(), optimizer.state)
    if any(v.dtype!=mx.bfloat16 for _,v in tree_flatten(model.parameters())):
        raise ValueError('Expected unchanged BF16 source weights')
    if any(mx.issubdtype(v.dtype,mx.floating) and v.dtype!=mx.float32 for _,v in tree_flatten(optimizer.state)):
        raise ValueError('Expected FP32 SFT optimizer state')
    initial_digest=state_digest(model,optimizer)
    if args.resume and initial_digest!=metadata['replica_hash']:
        raise ValueError('SFT checkpoint model/optimizer did not restore exactly')
    if any(d!=initial_digest for d in control.exchange(initial_digest,'initial-state')):
        raise ValueError('SFT initial replicas differ')
    backward = mixed_backward(model, accum,plan)
    last_saved = -1
    def save(is_best=False):
        nonlocal last_saved
        digests=control.exchange(state_digest(model,optimizer),'checkpoint-state')
        if len(set(digests))!=1: raise ValueError('SFT replicas diverged')
        metadata = dict(training_kind='sft', step=step, tokens=tokens, assistant_tokens=assistant_tokens,
            best_val_loss=best, sft_contract=contract, order_hash=order_hash, replica_hash=digests[0],
            selection_key=best_key,baseline_retention=baseline_retention,quality_stop=quality_stop,
            selection_baseline=selection_baseline,failed_eval_count=failed_eval_count,
            has_eligible_candidate=has_candidate,selection_reference='Part 1' if fresh else 'source checkpoint',
            selection_policy=('Baseline-relative behavior, knowledge, task completion and prose gates, then generated dev macro pass, repetition, caps and assistant NLL' if balanced else 'Retention bound, then generated dev macro pass, then stopping, then assistant NLL'),
            source_step=source['step'], source_tokens=source['tokens'],
            pretraining_step=ancestry['pretraining_step'],pretraining_tokens=ancestry['pretraining_tokens'],
            sft_stage=ancestry['sft_stage'],validation_fingerprint=ancestry['validation_fingerprint'],
            lifetime_sft_assistant_tokens=ancestry['prior_assistant_tokens']+assistant_tokens,
            source_masters_preserved=bool(contract.get('source_masters')),
            recipe=dict(model=source['recipe']['model'], **plan), optimizer_reset=True)
        path=root_call('checkpoint-write', lambda: save_bundle(args.save_dir,model,optimizer,metadata,
            dict(next_batch=step,order_hash=order_hash),best=is_best,keep=plan['keep_checkpoints']))
        if rank==0: print(f'[ckpt] {"best + latest" if is_best else "latest"} {path}',flush=True)
        last_saved=step
    if not args.resume:
        def baseline():
            if not fresh:return evaluate_run(model,tokenizer,args.data,plan,step,args.save_dir,args.retention)
            reference=TransformerLM(TransformerConfig(**reference_metadata['recipe']['model']))
            reference.load_weights(str(args.reference/'model.safetensors'));mx.eval(reference.parameters())
            result=evaluate_run(reference,tokenizer,args.data,plan,0,args.save_dir/'reference',args.retention)
            del reference;mx.clear_cache()
            return result
        result=control.broadcast(root_call('baseline',baseline),'baseline-result')
        baseline_retention=result['retention']['val_loss'];best=result['assistant_nll']
        if balanced:
            selection_baseline=compact_baseline(result)
            best_key=balanced_selection(result,selection_baseline,dict(plan,min_task_gain_cases=0) if fresh else plan)
        else:best_key=selection(result,baseline_retention,plan['max_retention_loss_increase'])
        has_candidate=not fresh
        save(not fresh)
    started, start_tokens = time.monotonic(), tokens
    if rank==0:
        print(f'[sft-start] parameters={count_parameters(model):,} pretraining_tokens={ancestry["pretraining_tokens"]:,} source_step={source["step"]} sft_stage={ancestry["sft_stage"]} steps={len(order)} epochs={plan["epochs"]} optimizer=fresh-FP32-AdamW',flush=True)
    total_rows=sum(batches)*accum
    while step<len(order):
        heartbeat[0]=time.monotonic()
        if control.broadcast((args.save_dir/'STOP').exists() if rank==0 else None,'stop'): break
        def next_batch():
            return mixed.batch(order[step],step)
        value=root_call('batch-read',next_batch)
        info=control.broadcast(value[1] if rank==0 else None,'batch-info')
        if info['assistant_tokens']<=0: raise ValueError('Empty supervised SFT batch')
        digest=control.broadcast(hashlib.sha256(value[0].tobytes()).hexdigest() if rank==0 else None,'batch-hash')
        with mx.stream(mx.cpu):
            wire=mx.array(value[0]) if rank==0 else mx.zeros((3,total_rows,info['length']),mx.int32)
            wire=mx.distributed.all_sum(wire,group=group,stream=mx.cpu);mx.eval(wire);array=np.asarray(wire)
        if not all(control.exchange(hashlib.sha256(array.tobytes()).hexdigest()==digest,'batch-verified')):
            raise ValueError('SFT batch transfer corrupted')
        array=array.reshape(3,accum,sum(batches),info['length'])
        begin,end=sum(batches[:rank]),sum(batches[:rank+1])
        flags=np.tile(np.arange(begin,end)%4==3,(accum,1))
        loss,grads=backward(mx.array(array[0,:,begin:end].copy()),mx.array(array[1,:,begin:end].copy()),
            mx.array(array[2,:,begin:end].copy().view(np.float32)),mx.array(flags))
        mx.eval(loss,grads)
        losses=control.exchange(loss.tolist() if bool(mx.all(mx.isfinite(loss)).item()) else None,'loss')
        if any(x is None for x in losses): raise RuntimeError('Nonfinite SFT loss')
        if world>1:
            grads=tree_map(lambda x:x*world,grads)
            grads=nn.average_gradients(grads,group=group,all_reduce_size=32*1024**2,communication_stream=mx.cpu)
        grads,norm=clip_gradients(grads,plan['grad_clip']);mx.eval(grads,norm)
        if not all(control.exchange(bool(mx.isfinite(norm).item()),'finite-gradient')):
            raise RuntimeError('Nonfinite SFT gradient')
        lr=learning_rate(step+1,len(order),plan['peak_lr'],plan['final_lr'],plan['warmup_fraction'])
        optimizer.learning_rate=lr;optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state)
        step+=1;tokens+=info['input_tokens'];assistant_tokens+=info['assistant_tokens']
        if rank==0 and (step==1 or step%plan['log_every']==0):
            elapsed=time.monotonic()-started
            print(f'[sft {step}/{len(order)}] assistant_loss={sum(v[1] for v in losses):.6f} objective_loss={sum(v[0] for v in losses):.6f} replay_loss={sum(v[2] for v in losses):.6f} lr={lr:.9e} grad_norm={norm.item():.4f} input_tokens={tokens} assistant_tokens={assistant_tokens} elapsed={elapsed:.1f}s tok/s={(tokens-start_tokens)/max(elapsed,1e-6):,.0f}',flush=True)
        if step%plan['eval_every']==0 or step==len(order):
            result=control.broadcast(root_call('evaluate',lambda:evaluate_run(model,tokenizer,args.data,plan,step,args.save_dir,args.retention)),'eval-result')
            if balanced:
                if fresh:
                    checks=fresh_progress(result,selection_baseline,plan,has_candidate,failed_eval_count)
                    failed_eval_count=checks['consecutive_failures'];quality_stop=checks['quality_stop']
                else:
                    checks=eligibility(result,selection_baseline,plan)
                    failed_eval_count=0 if checks['eligible'] else failed_eval_count+1
                    quality_stop=failed_eval_count>=plan['regression_patience']
                key=balanced_selection(result,selection_baseline,plan)
                if rank==0:
                    atomic_json(args.save_dir/'evaluations'/f'quality_{step:07d}.json',dict(checks,consecutive_failures=failed_eval_count,baseline=selection_baseline))
                    print(f'[quality-gates {step}] eligible={checks["eligible"]} failures={checks["failures"]} consecutive={failed_eval_count}/{plan["regression_patience"]}',flush=True)
            else:
                key=selection(result,baseline_retention,plan['max_retention_loss_increase']);quality_stop=not bool(key[0])
            improved=bool(key[0]) and key>best_key
            if improved:best=result['assistant_nll'];best_key=key;has_candidate=True
            save(improved)
            if quality_stop:
                if rank==0:print('[quality-stop] Candidate failed the quality gates; best eligible checkpoint preserved',flush=True)
                break
    if last_saved!=step:save()
    result=dict(status='quality-stopped' if quality_stop else 'complete' if step==len(order) else 'stopped',step=step,tokens=tokens,assistant_tokens=assistant_tokens,total_steps=len(order),selection_key=best_key,baseline_retention=baseline_retention,selection_baseline=selection_baseline,failed_eval_count=failed_eval_count,has_eligible_candidate=has_candidate)
    root_call('result',lambda:atomic_json(args.job_dir/'result.json',result))
    control.exchange(True,'finished')
    print(f'[finished-sft] rank={rank} status={result["status"]} step={step}',flush=True)


if __name__=='__main__':
    with ExitStack() as resources: run(resources)
