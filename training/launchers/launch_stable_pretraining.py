#!/usr/bin/env python3
"""Plan an open-ended pretraining continuation; training requires --run."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v1.common import ROOT, file_sha256, fingerprint, load_corpus, lock, read_json
from sml_v1.checkpoint_metadata import verify_bundle_metadata
from sml_v1.continuation import check_resume
from sml_v1.lr_scheduler import restore_scheduler
from sml_v1.recipe import tokens_per_update, validate
from sml_v1.stable_schedule import (KEY, DECAY_KEY, GRADUAL_KEY, cooldown_recipe, stable_recipe, gradual_recipe,
    effective_learning_rate, phase, training_target)


def checkpoint_path(path):
    """Pin the pointer once so planning and staging use the same immutable bundle."""
    path = path.absolute()
    if path.is_symlink():
        raise ValueError('Checkpoint path must not be a symlink')
    if path.suffix == '.json':
        pointer = read_json(path)
        name = pointer.get('bundle', '')
        if (pointer.get('format') != 'sml-pretrain-bundle-v1'
                or not name.startswith('step_') or Path(name).name != name):
            raise ValueError('Invalid checkpoint pointer')
        path = path.parent / name
    return path


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='Full pretraining checkpoint for a new, separate output')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Resume this output latest.json, preserving its schedule')
    p.add_argument('--stable-lr', type=float, help='Optional explicit LR change; default holds saved effective LR')
    p.add_argument('--warmup-tokens', type=int, default=0, help='Required nonzero ramp when changing LR')
    p.add_argument('--all-available-data', action='store_true',
                   help='Remove artificial row caps and continue remaining sources when one ends')
    p.add_argument('--cooldown-tokens', type=int, help='Fork a final 1-sqrt decay from a stable checkpoint')
    p.add_argument('--final-lr', type=float, default=0.)
    p.add_argument('--gradual-decay-until', type=int, help='Absolute total token count at which linear decay reaches its floor')
    p.add_argument('--decay-floor-lr', type=float, help='Nonzero LR held after the gradual decay milestone; training remains open-ended')
    p.add_argument('--corpus', type=Path, default=ROOT.parent/'configs/corpus_20b.json')
    p.add_argument('--tokenizer', type=Path, default=ROOT.parent/'reference/opensml-150m/tokenizer/bytebpe32k_v1')
    p.add_argument('--sample-prompts-file', type=Path, default=ROOT.parent/'configs/sample_prompts.json')
    p.add_argument('--network', type=Path)
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--stop-after-steps', type=int, default=0)
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    if (bool(a.source) == a.resume or a.warmup_tokens < 0 or a.stop_after_steps < 0
            or (a.resume and (a.stable_lr is not None or a.warmup_tokens or a.cooldown_tokens is not None or a.all_available_data or a.gradual_decay_until is not None))
            or (a.cooldown_tokens is not None and (a.stable_lr is not None or a.warmup_tokens or a.all_available_data))
            or ((a.gradual_decay_until is None) != (a.decay_floor_lr is None))
            or (a.gradual_decay_until is not None and (a.cooldown_tokens is not None or a.stable_lr is not None or a.warmup_tokens or a.all_available_data))
            or (a.final_lr != 0 and a.cooldown_tokens is None)
            or (a.run and a.network is None)):
        p.error('Choose --source OR --resume; schedule changes require a separate output; --run requires --network')
    return a


def main(argv=None):
    a = arguments(argv)
    output = a.output.resolve()
    if not output.is_relative_to(ROOT/'runs') or output == ROOT/'runs':
        raise ValueError('Outputs must stay inside pretraining/runs')
    if (output/'latest.json').exists() and not a.resume:
        raise ValueError('Existing output must resume its own latest.json with --resume')
    source = checkpoint_path(output/'latest.json' if a.resume else a.source)
    if not a.resume and (output == source.parent.resolve() or output.is_relative_to(source.parent.resolve())):
        raise ValueError('Use a separate output; preserve the source run')
    metadata = verify_bundle_metadata(source)
    if a.resume:
        recipe = validate(copy.deepcopy(metadata['recipe']))
        if KEY not in recipe:
            raise ValueError('Resume requires an existing stable-schedule output')
    elif a.gradual_decay_until is not None:
        recipe = gradual_recipe(metadata, decay_end_tokens=a.gradual_decay_until, floor_lr=a.decay_floor_lr)
    elif a.cooldown_tokens is not None:
        recipe = cooldown_recipe(metadata, duration_tokens=a.cooldown_tokens, final_lr=a.final_lr)
    else:
        recipe = stable_recipe(metadata, stable_lr=a.stable_lr, warmup_tokens=a.warmup_tokens,
                               all_available_data=a.all_available_data)
    if len(recipe['batches']) != 5 or recipe.get('data_mode') != 'hf_stream':
        raise ValueError('This launcher requires five ranks and the existing HF stream')
    target = training_target(recipe)
    if target is not None and metadata['tokens'] >= target:
        raise ValueError('Cooldown target already reached; return to a pre-cooldown stable checkpoint')
    from sml_v1.tokenization import Tokenizer
    from sml_v1.stream import stream_manifest
    from sml_v1.sample_prompts import read_prompts
    tokenizer = Tokenizer(a.tokenizer)
    if tokenizer.vocab_size != recipe['model']['vocab_size']:
        raise ValueError('Frozen tokenizer vocabulary differs from model')
    corpus = load_corpus(a.corpus)
    manifest = stream_manifest(corpus, tokenizer, recipe['stream'])
    read_prompts(a.sample_prompts_file)
    modules = {p.name: file_sha256(p) for p in (ROOT/'sml_v1').glob('*.py')}
    contract = dict(recipe=fingerprint(recipe), tokenizer=tokenizer.fingerprint,
                    data=fingerprint(manifest), code=fingerprint(modules))
    transition = check_resume(metadata, contract, recipe, manifest)
    scheduler = restore_scheduler(recipe, metadata)
    update = tokens_per_update(recipe)
    plan = dict(mode='run' if a.run else 'plan only', source=str(source), output=str(output),
        transition=transition, start_step=metadata['step'], start_tokens=metadata['tokens'],
        schedule=recipe.get(DECAY_KEY, recipe.get(GRADUAL_KEY, recipe[KEY])), phase=phase(metadata['tokens'], recipe),
        target_tokens=target, remaining_updates=None if target is None else (target-metadata['tokens'])//update,
        old_target_tokens_ignored=recipe['target_tokens'],
        next_lr=effective_learning_rate(metadata['tokens']+update, recipe, scheduler),
        plateau_controller='disabled; historical state retained unchanged',
        optimizer='restore full FP32 AdamW state', data_cursor='restore including pending input',
        eval_every_tokens=recipe['eval_every_tokens'], validation_fingerprint=metadata['validation_fingerprint'],
        corpus_shares=manifest['weights'], tokens_per_update=update, stop_after_steps=a.stop_after_steps)
    plan['source_policy'] = recipe['stream'].get('source_policy', 'legacy scan caps')
    plan['effective_row_limit_per_source'] = None if recipe['stream'].get('source_policy') == 'all-available-v1' else corpus['max_rows_per_source']
    print(json.dumps(plan, indent=2), flush=True)
    if not a.run:
        print('No training or SSH. No automatic token endpoint unless a final cooldown is selected.')
        return plan
    from five_mac_cluster import Cluster, PYTHON
    from sml_v1.launch import stage
    with lock(a.network/'.training-launch.lock'), lock(output/'.launcher.lock'):
        if (output/'STOP').exists() and not a.clear_stop:
            raise ValueError('STOP exists; use --clear-stop explicitly')
        proof_path = a.network/'five_rank_correctness.json'
        proof = read_json(proof_path) if proof_path.exists() else {}
        receipts = proof.get('receipts', [])
        if (proof.get('modules') != modules or len(receipts) != 5
                or {r.get('rank') for r in receipts} != set(range(5))
                or any(r.get('status') != 'passed' for r in receipts)):
            # Includes an idle-worker preflight; it never stops an existing job.
            subprocess.run([sys.executable, str(Path(__file__).with_name('check_five_mac.py')),
                            '--network', str(a.network), '--run'], check=True)
            proof = read_json(proof_path)
            if proof.get('modules') != modules:
                raise ValueError('Worker code changed after planning; rerun the command')
        cluster = Cluster(a.network)
        job = output/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir(parents=True)
        cluster.preflight(job)
        source_dir = stage(job, recipe, a.tokenizer, source, corpus)
        staged_modules = {p.name: file_sha256(p) for p in (source_dir/'sml_v1').glob('*.py')}
        if staged_modules != modules:
            raise ValueError('Worker code changed after planning; refusing launch')
        cluster.stage(source_dir)
        if a.clear_stop:
            (output/'STOP').unlink(missing_ok=True)
        worker = ['/usr/bin/caffeinate', '-i', PYTHON, '-u', '-m', 'sml_v1.pretrain',
            '--config', str(source_dir/'recipe.json'), '--tokenizer', str(source_dir/'tokenizer'),
            '--corpus', str(source_dir/'corpus.json'), '--resume', str(source_dir/'checkpoint'),
            '--save-dir', str(output), '--job-dir', str(job), '--ring-control', 'native',
            '--stop-after-steps', str(a.stop_after_steps),
            '--sample-prompts-file', str(a.sample_prompts_file.resolve())]
        cluster.launch(job, source_dir, worker, timeout=None, stop_dir=output)
        result = read_json(job/'result.json')
        if result.get('status') not in ('stopped', 'complete', 'data-exhausted'):
            raise RuntimeError('Missing clean worker completion')
        print(f'[verified-stop] {result}; log: {job/"train.log"}', flush=True)
    return plan


if __name__ == '__main__':
    main()
