#!/usr/bin/env python3
"""Resume the preserved 40B base into a separate, bounded 2B-token cooldown."""
import argparse
import copy
from datetime import datetime
import gzip
import json
from pathlib import Path
import shutil
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'pretraining'
PREPARED = ROOT/'prepared/deep44_cooldown_2b_20261107'
sys.path.insert(0, str(PREPARED/'runtime'))
from sml_v1.common import atomic_json, file_sha256, fingerprint, load_corpus, lock, read_json
from sml_v1.checkpoint_metadata import verify_bundle_metadata
from sml_v1.code_phase import CodePhaseStream, training_manifest
from sml_v1.code_validation import load_panel
from sml_v1.continuation import check_resume
from sml_v1.lr_scheduler import restore_scheduler
from sml_v1.recipe import tokens_per_update, validate
from sml_v1.stable_schedule import DECAY_KEY, cooldown_recipe, effective_learning_rate, training_target
from sml_v1.tokenization import Tokenizer

SOURCE = ROOT/'runs/deep44_prose_python_sql_80_15_5_20261104/step_0305186_cea4b9aa534e'
OUTPUT = ROOT/'runs/deep44_cooldown_2b_20261107'
NETWORK = ROOT/'diagnostics/m5_migration_20260922/network_20261102_234513'
TOKENIZER = PROJECT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
DURATION = 2_000_000_000
FINAL_LR = 1e-5


def modules():
    return {p.name: file_sha256(p) for p in (PREPARED/'runtime/sml_v1').glob('*.py')}


def checkpoint_path(pointer):
    info = read_json(pointer)
    name = info.get('bundle', '')
    if (pointer.is_symlink() or info.get('format') != 'sml-pretrain-bundle-v1'
            or Path(name).name != name or not name.startswith('step_')):
        raise ValueError('Invalid cooldown checkpoint pointer')
    path = pointer.parent/name
    if path.is_symlink():
        raise ValueError('Checkpoint cannot be a symlink')
    return path


def verify_readiness():
    receipt = read_json(PREPARED/'readiness.json')
    if receipt.get('status') != 'ready':
        raise ValueError('Cooldown preparation is incomplete')
    for name, digest in receipt['files'].items():
        if file_sha256(PROJECT/name) != digest:
            raise ValueError('Prepared input changed: '+name)
    return receipt


def plan(*, full_verify=True):
    resume = (OUTPUT/'latest.json').exists()
    source = checkpoint_path(OUTPUT/'latest.json') if resume else SOURCE
    metadata = verify_bundle_metadata(source) if full_verify else read_json(source/'model.safetensors.json')
    if not resume and (metadata['step'] != 305186 or metadata['tokens'] != 40001339392):
        raise ValueError('Expected the explicit preserved 40B base')
    if OUTPUT == SOURCE.parent or OUTPUT.is_relative_to(SOURCE.parent):
        raise ValueError('Cooldown must use a separate output')
    tokenizer = Tokenizer(TOKENIZER)
    corpus = load_corpus(PREPARED/'corpus.json')
    manifest = training_manifest(corpus, tokenizer, metadata['recipe']['stream'])
    recipe = validate(copy.deepcopy(metadata['recipe'])) if resume else cooldown_recipe(
        metadata, duration_tokens=DURATION, final_lr=FINAL_LR)
    t = recipe[DECAY_KEY]
    parent = read_json(SOURCE/'model.safetensors.json')
    if recipe != cooldown_recipe(parent, duration_tokens=DURATION, final_lr=FINAL_LR):
        raise ValueError('Cooldown settings differ from the exact planned branch')
    if fingerprint(manifest) != parent['v2_contract']['data']:
        raise ValueError('The training data, filters or mixture changed')
    contract = dict(recipe=fingerprint(recipe), tokenizer=tokenizer.fingerprint,
                    data=fingerprint(manifest), code=fingerprint(modules()))
    transition = check_resume(metadata, contract, recipe, manifest)
    load_panel(PREPARED/'code_validation.npz', manifest['code_validation'], manifest, tokenizer, recipe)
    from sml_v1.sample_prompts import read_prompts
    read_prompts(PREPARED/'sample_prompts.json')
    if full_verify:
        with gzip.open(source/'model.safetensors.rank0.data_state.json.gz', 'rt') as f:
            saved = json.load(f)
        cursor = saved['stream_state']
        stream = CodePhaseStream(manifest, tokenizer, 'train', recipe['model']['max_seq_len'], cursor,
                                 exclusions=manifest['code_validation']['excluded_content_keys'])
        stream.close()  # Validate pending input/accounting without fetching any rows.
        if cursor.get('exhausted_sources'):
            raise ValueError('A required source is exhausted')
    scheduler = restore_scheduler(recipe, metadata)
    update = tokens_per_update(recipe)
    target = training_target(recipe)
    result = dict(mode='resume cooldown' if resume else 'new cooldown from preserved 40B base',
        training_started=False, source=str(source), output=str(OUTPUT), transition=transition,
        start_step=metadata['step'], start_tokens=metadata['tokens'],
        requested_added_tokens=DURATION, actual_added_tokens=t['duration_tokens'],
        target_tokens=target, final_step=t['parent_step']+t['duration_tokens']//update,
        remaining_updates=max(0,(target-metadata['tokens'])//update), tokens_per_update=update,
        starting_lr=t['start_lr'], next_lr=effective_learning_rate(metadata['tokens']+update, recipe, scheduler),
        final_lr=t['final_lr'], decay_shape=t['shape'],
        schedule_samples=[dict(added_tokens=round(t['duration_tokens']*fraction),
            learning_rate=effective_learning_rate(t['start_tokens']+t['duration_tokens']*fraction, recipe, scheduler))
            for fraction in (0,.25,.5,.75,1)],
        token_shares=manifest['weights'], sequence_length=recipe['model']['max_seq_len'], ranks=5,
        data='Same pinned public live streams; preserve source positions and pending input; no corpus download or recycling.',
        optimizer='Restore original FP32 masters, AdamW moments and update count; BF16 training computation.',
        evaluation='Unchanged fixed prose validation and separate Python/SQL losses every 100M total-token boundary, plus final.',
        checkpoint_policy='Separate output; inherited best-loss criterion; original checkpoints remain intact.',
        source_validation_fingerprint=metadata['validation_fingerprint'],
        source_manifest_sha256=file_sha256(source/'manifest.json'),
        launch='bash ${WORKSPACE}/pretraining/scripts/run_deep44_cooldown_2b.sh')
    return result, source, metadata, recipe, corpus


def correctness_probe(cluster, job):
    from five_mac_cluster import PYTHON, ORDER
    probe = job/'correctness'
    probe.mkdir()
    cluster.preflight(probe)
    staged = probe/'input'
    staged.mkdir()
    shutil.copytree(PREPARED/'runtime/sml_v1', staged/'sml_v1', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(ROOT/'scripts/probe_five_mac.py', staged/'probe_five_mac.py')
    cluster.stage(staged)
    cluster.launch(probe, staged, [PYTHON,'-u',str(staged/'probe_five_mac.py'),'--job-dir',str(probe)], timeout=360)
    receipts=[]
    for rank,physical in enumerate(ORDER):
        r=json.loads(cluster.remote(physical,['/bin/cat',str(probe/f'probe_rank{rank}.json')]))
        if r.get('rank')!=rank or r.get('status')!='passed':
            raise ValueError('Five-Mac correctness probe failed')
        receipts.append(r)
    if len({r['checkpoint_digest'] for r in receipts})!=1:
        raise ValueError('Probe replicas differ')
    atomic_json(job/'correctness.json',dict(modules=modules(),receipts=receipts))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--check-cluster',action='store_true')
    p.add_argument('--clear-stop',action='store_true')
    args=p.parse_args(argv)
    if args.run and args.check_cluster:
        p.error('Choose --run or --check-cluster')
    verify_readiness()
    result,source,metadata,recipe,corpus=plan()
    print(json.dumps(result,indent=2),flush=True)
    if not args.run and not args.check_cluster:
        print('[prepared] No training started. Use the launch command to begin or resume.',flush=True)
        return result
    from five_mac_cluster import Cluster,PYTHON
    from sml_v1.launch import stage
    with lock(NETWORK/'.training-launch.lock'),lock(ROOT/'.local-sft-launch.lock'),lock(OUTPUT/'.launcher.lock'):
        cluster=Cluster(NETWORK)
        if args.check_cluster:
            job=PREPARED/'cluster_checks'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
            job.mkdir(parents=True)
            cluster.preflight(job)
            atomic_json(PREPARED/'cluster_check.json',dict(status='passed',job=str(job),training_started=False))
            return result
        if result['remaining_updates']==0:
            print('[complete] Cooldown target already reached; no restart.',flush=True)
            return result
        if (OUTPUT/'STOP').exists() and not args.clear_stop:
            raise ValueError('STOP marker exists; use --clear-stop to resume explicitly')
        result,source,metadata,recipe,corpus=plan()
        job=OUTPUT/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir(parents=True)
        cluster.preflight(job)
        atomic_json(job/'plan.json',result)
        correctness_probe(cluster,job)
        verify_readiness()
        staged=stage(job,recipe,TOKENIZER,source,corpus)
        for name in ('code_validation.npz','sample_prompts.json'):
            shutil.copyfile(PREPARED/name,staged/name)
        manifest=read_json(staged/'input_manifest.json')
        for name in ('code_validation.npz','sample_prompts.json'):
            manifest['files'][name]=file_sha256(staged/name)
        atomic_json(staged/'input_manifest.json',manifest)
        if {p.name:file_sha256(p) for p in (staged/'sml_v1').glob('*.py')} != modules():
            raise ValueError('Staged runtime differs from the reviewed copy')
        cluster.stage(staged)
        if args.clear_stop:
            (OUTPUT/'STOP').unlink(missing_ok=True)
        worker=['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','sml_v1.pretrain',
            '--config',str(staged/'recipe.json'),'--tokenizer',str(staged/'tokenizer'),
            '--corpus',str(staged/'corpus.json'),'--resume',str(staged/'checkpoint'),
            '--save-dir',str(OUTPUT),'--job-dir',str(job),'--ring-control','native',
            '--sample-prompts-file',str(staged/'sample_prompts.json'),
            '--code-validation-panel',str(staged/'code_validation.npz')]
        cluster.launch(job,staged,worker,timeout=None,stop_dir=OUTPUT)
        result=read_json(job/'result.json')
        if result.get('status') not in ('stopped','complete','data-exhausted'):
            raise RuntimeError('No clean worker completion')
        print(f'[verified-stop] {result}; log: {job/"train.log"}',flush=True)
        return result


if __name__=='__main__':
    main()
