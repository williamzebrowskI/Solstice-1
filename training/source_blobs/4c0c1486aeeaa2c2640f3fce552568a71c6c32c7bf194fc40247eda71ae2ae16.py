"""Single-Mac LoRA SFT; frozen base, exact resume, FP32 evaluation and exports."""
import argparse
from pathlib import Path
import threading
import time
import os
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from sml_v1.common import read_json, atomic_json, file_sha256, fingerprint
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.precision import MasterAdamW
from sml_v1.pretrain import clip_gradients
from sml_v1.tokenization import Tokenizer
from .data import Corpus, training_order, learning_rate
from .mixed import MixedBatcher
from .text_corpus import verify
from .text_train import mixed_backward, order_digest
from .text_eval import evaluate, reference_inference
from .text_checks import eligibility, selection
from .retention import verify as verify_retention
from .lora_adapter import attach, assert_frozen, dense_model, base_digest
from . import lora_checkpoint as checkpoint


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'tokenizer', 'source', 'source-masters', 'contract', 'save-dir', 'job-dir', 'retention', 'reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    plan = read_json(args.config)
    if plan['execution'] != 'single-mac' or plan['batches'] != [8] or plan['grad_accum'] != 8:
        raise ValueError('LoRA worker requires the validated single-Mac batch configuration')
    if os.getenv('MLX_RANK') not in (None, '0'):
        raise ValueError('Distributed LoRA launch is not supported')
    contract = read_json(args.contract)
    if contract['plan'] != fingerprint(plan) or contract['data'] != fingerprint(verify(args.data, plan)):
        raise ValueError('LoRA plan/data contract mismatch')
    runtime = Path(__file__).resolve().parents[1]
    modules = {str(p.relative_to(runtime)): file_sha256(p) for package in ('sml_v1', 'sft')
               for p in sorted((runtime / package).rglob('*.py'))}
    if modules != contract['modules']:
        raise ValueError('Frozen LoRA worker modules changed')
    for path, key in ((args.source / 'model.safetensors', 'source_weights'),
                      (args.source / 'model.safetensors.json', 'source_metadata'),
                      (args.source_masters, 'source_masters'),
                      (args.retention / 'manifest.json', 'retention')):
        if file_sha256(path) != contract[key]:
            raise ValueError('LoRA input checksum mismatch: ' + key)
    for name, digest in contract['comparison_files'].items():
        if file_sha256(args.reference / name) != digest:
            raise ValueError('Reference checkpoint changed')
    source = read_json(args.source / 'model.safetensors.json')
    if source.get('training_kind') == 'sft':
        raise ValueError('Expected a pretrained base')
    verify_retention(args.retention, source['validation_fingerprint'])
    tokenizer = Tokenizer(args.tokenizer)
    if tokenizer.fingerprint != contract['tokenizer']:
        raise ValueError('Tokenizer changed')
    args.job_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.job_dir / 'worker_rank0.json', dict(pid=os.getpid(), output=str(args.job_dir)))
    heartbeat, finished = [time.monotonic()], threading.Event()
    def watchdog():
        while not finished.wait(10):
            if time.monotonic() - heartbeat[0] > 1800:
                print('[fatal] No LoRA progress for 30 minutes', flush=True)
                os._exit(70)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        mx.set_cache_limit(int(plan.get('cache_limit_gib', 16) * 2**30))
        mx.random.seed(plan['seed'])
        config = source['recipe']['model']
        model = TransformerLM(TransformerConfig(**config))
        original = mx.load(str(args.source / 'model.safetensors'))
        base = mx.load(str(args.source_masters))
        if set(base) != set(original) or any(v.dtype != mx.float32 or v.shape != original[n].shape for n, v in base.items()):
            raise ValueError('FP32 base master names, shapes or precision differ')
        if not all(bool(mx.all(v.astype(original[n].dtype) == original[n]).item()) for n, v in base.items()):
            raise ValueError('FP32 masters do not match the pretrained base')
        model.load_weights(list(original.items()), strict=True)
        frozen_digest = base_digest(original)
        trainable_count = attach(model, plan['adaptation'])
        optimizer = MasterAdamW(learning_rate=plan['peak_lr'], betas=tuple(plan['betas']),
                               eps=plan['eps'], weight_decay=plan['weight_decay'])
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        assert_frozen(model, frozen_digest)
        corpus = Corpus(args.data, 'train')
        order = training_order(corpus.lengths, plan['seed'], plan['sft_rows_per_update'], plan['epochs'])
        order_hash = order_digest(order)
        batcher = MixedBatcher(corpus, args.data, plan)
        if len(order) * plan['replay_rows_per_update'] > len(batcher.replay):
            raise ValueError('Insufficient unique prose replay rows')
        state = dict(step=0, tokens=0, assistant_tokens=0, contract=contract, order_hash=order_hash,
                     source_step=source['step'], source_tokens=source['tokens'],
                     validation_fingerprint=source['validation_fingerprint'], adaptation=plan['adaptation'],
                     trainable_parameters=trainable_count, quality_stop=False, failed_eval_count=0,
                     has_eligible_candidate=False, selection_key=None, selection_baseline=None,
                     baseline_retention=None, total_steps=len(order))
        def measure(step):
            assert_frozen(model, frozen_digest)
            dense = dense_model(model, optimizer, base, config)
            try:
                with reference_inference(dense):
                    return evaluate(dense, tokenizer, args.data, plan, step, args.save_dir, args.retention,
                                    precision_label='fp32-base-plus-lora-and-token-deltas',
                                    heartbeat=lambda: heartbeat.__setitem__(0, time.monotonic()))
            finally:
                del dense
                mx.clear_cache()
        last_saved = -1
        def save(best=False):
            nonlocal last_saved
            assert_frozen(model, frozen_digest)
            path = checkpoint.save(args.save_dir, model, optimizer, base, config, state, best=best)
            print(f'[ckpt] {"best + latest" if best else "latest"} {path}', flush=True)
            last_saved = state['step']
        if args.resume:
            state = checkpoint.restore(args.resume, model, optimizer, contract, order_hash)
            assert_frozen(model, frozen_digest)
            last_saved = state['step']
        else:
            if plan.get('baseline_cache'):
                from .baseline_cache import read as cached_baseline
                # New adapters and token deltas must be exactly zero-effect.
                from .lora_adapter import merged_weights
                merged = merged_weights(model, optimizer, base)
                if not all(bool(mx.all(v == base[n]).item()) for n,v in merged.items()):
                    raise ValueError('Cannot reuse baseline for a nonzero adaptation')
                del merged
                parent = cached_baseline(runtime, 'parent_dev', args.save_dir/'evaluations/eval_0000000.json')
                comparison = cached_baseline(runtime, 'reference_dev', args.save_dir/'comparison_167/evaluations/eval_0000167.json')
            else:
                parent = measure(0)
                reference_meta = read_json(args.reference / 'model.safetensors.json')
                reference = TransformerLM(TransformerConfig(**config))
                reference.load_weights(str(args.reference / 'model.safetensors'), strict=True)
                mx.eval(reference.parameters())
                with reference_inference(reference):
                    comparison = evaluate(reference, tokenizer, args.data, plan, reference_meta['step'],
                                          args.save_dir / 'comparison_167', args.retention,
                                          heartbeat=lambda: heartbeat.__setitem__(0, time.monotonic()))
                del reference
                mx.clear_cache()
            baseline = dict(comparison, retention=parent['retention'])
            state.update(selection_baseline=baseline, baseline_retention=parent['retention']['val_loss'],
                         selection_key=selection(parent, baseline, plan))
            atomic_json(args.save_dir / 'selection_baseline.json', dict(behavior_reference='checkpoint 167',
                        retention_reference='42B cooled pretrained parent', comparison=comparison,
                        pretrained=parent, combined=baseline))
            save(True)
        backward = mixed_backward(model, plan['grad_accum'], plan)
        started = time.monotonic()
        start_tokens = state['tokens']
        print(f'[sft-start] method=lora+chat-tokens base_parameters={sum(v.size for v in base.values()):,} '
              f'trainable_parameters={trainable_count:,} pretraining_tokens={source["tokens"]:,} '
              f'steps={len(order)} epochs={plan["epochs"]} base_weights=frozen', flush=True)
        while state['step'] < len(order):
            heartbeat[0] = time.monotonic()
            if (args.save_dir / 'STOP').exists():
                break
            wire, info = batcher.batch(order[state['step']], state['step'])
            if info['assistant_tokens'] <= 0:
                raise ValueError('Empty supervised batch')
            array = wire.reshape(3, plan['grad_accum'], plan['batches'][0], info['length'])
            flags = np.tile(np.arange(plan['batches'][0]) % 4 == 3, (plan['grad_accum'], 1))
            values, grads = backward(mx.array(array[0].copy()), mx.array(array[1].copy()),
                                     mx.array(array[2].copy().view(np.float32)), mx.array(flags))
            mx.eval(values, grads)
            grads, norm = clip_gradients(grads, plan['grad_clip'])
            mx.eval(grads, norm)
            if not bool(mx.all(mx.isfinite(values)).item()) or not bool(mx.isfinite(norm).item()):
                raise RuntimeError('Nonfinite adapter loss or gradient')
            lr = learning_rate(state['step'] + 1, len(order), plan['peak_lr'], plan['final_lr'], plan['warmup_fraction'])
            optimizer.learning_rate = lr
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            state['step'] += 1
            state['tokens'] += info['input_tokens']
            state['assistant_tokens'] += info['assistant_tokens']
            step = state['step']
            if step == 1 or step % plan['log_every'] == 0:
                elapsed = time.monotonic() - started
                loss, assistant, replay = values.tolist()
                print(f'[sft {step}/{len(order)}] assistant_loss={assistant:.6f} objective_loss={loss:.6f} '
                      f'replay_loss={replay:.6f} lr={lr:.9e} grad_norm={norm.item():.4f} '
                      f'input_tokens={state["tokens"]} assistant_tokens={state["assistant_tokens"]} '
                      f'elapsed={elapsed:.1f}s tok/s={(state["tokens"]-start_tokens)/max(elapsed,1e-6):,.0f}', flush=True)
            if step % plan['eval_every'] == 0 or step == len(order):
                result = measure(step)
                checks = eligibility(result, state['selection_baseline'], plan)
                state['failed_eval_count'] = state['failed_eval_count'] + 1 if checks['foundation_failures'] else 0
                state['quality_stop'] = state['failed_eval_count'] >= plan['regression_patience']
                key = selection(result, state['selection_baseline'], plan)
                improved = bool(key[0]) and key > state['selection_key']
                if improved:
                    state['selection_key'] = key
                    state['has_eligible_candidate'] = True
                atomic_json(args.save_dir / 'evaluations' / f'quality_{step:07d}.json',
                            dict(checks, consecutive_failures=state['failed_eval_count'], frozen_base_verified=True))
                print(f'[quality-gates {step}] eligible={checks["eligible"]} failures={checks["failures"]} '
                      f'consecutive={state["failed_eval_count"]}/{plan["regression_patience"]} frozen_base=verified', flush=True)
                save(improved)
                if state['quality_stop']:
                    print('[quality-stop] Adapter exceeded retention bound twice; no automatic restart', flush=True)
                    break
        if last_saved != state['step']:
            save()
        status = 'quality-stopped' if state['quality_stop'] else 'complete' if state['step'] == len(order) else 'stopped'
        atomic_json(args.job_dir / 'result.json', dict(state, status=status))
        print(f'[finished-sft] rank=0 status={status} step={state["step"]}', flush=True)
    finally:
        finished.set()


if __name__ == '__main__':
    run()
