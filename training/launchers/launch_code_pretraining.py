#!/usr/bin/env python3
"""Prepare or explicitly launch the isolated 80/15/5 continuation."""
import argparse
import copy
from datetime import datetime
import gzip
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'prepared/python_sql_20261104'
sys.path.insert(0,str(PREPARED/'runtime'))
from sml_v1.common import atomic_json,file_sha256,fingerprint,load_corpus,lock,read_json
from sml_v1.code_phase import KEY,phase_recipe,training_manifest,migrate_cursor,CodePhaseStream
from sml_v1.code_validation import load_panel
from sml_v1.continuation import check_resume
from sml_v1.recipe import tokens_per_update,validate
from sml_v1.stable_schedule import effective_learning_rate
from sml_v1.lr_scheduler import restore_scheduler
from sml_v1.tokenization import Tokenizer

PARENT=ROOT/'runs/deep44_prose_gradual_100b_20261028'
OUTPUT=ROOT/'runs/deep44_prose_python_sql_80_15_5_20261104'
NETWORK=ROOT/'diagnostics/m5_migration_20260922/network_20261102_234513'
TOKENIZER=ROOT.parent/'reference/opensml-150m/tokenizer/bytebpe32k_v1'


def checkpoint_path(pointer):
    if pointer.is_symlink():raise ValueError('Checkpoint pointer cannot be a symlink')
    info=read_json(pointer);name=info.get('bundle','')
    if info.get('format')!='sml-pretrain-bundle-v1' or not name.startswith('step_') or Path(name).name!=name:
        raise ValueError('Invalid full checkpoint pointer')
    source=pointer.parent/name
    if source.is_symlink():raise ValueError('Checkpoint cannot be a symlink')
    return source


def modules():
    return {p.name:file_sha256(p) for p in (PREPARED/'runtime/sml_v1').glob('*.py')}


def verify_readiness():
    readiness=read_json(PREPARED/'readiness.json')
    if readiness.get('status')!='ready-for-idle-cluster-check':
        raise ValueError('Prepared continuation has not passed its CPU/data checks')
    for name,digest in readiness['files'].items():
        path=ROOT.parent/name
        if not path.is_file() or file_sha256(path)!=digest:
            raise ValueError('Prepared input changed after validation: '+name)
    from sml_v1.code_sources import parser
    parser()  # Verify the separately vendored parser without touching MLX/GPU.
    return readiness


def plan(*,full_verify=False):
    readiness=verify_readiness()
    resume=(OUTPUT/'latest.json').exists()
    source=checkpoint_path((OUTPUT if resume else PARENT)/'latest.json')
    if full_verify:
        from sml_v1.checkpoint_metadata import verify_bundle_metadata
        metadata=verify_bundle_metadata(source)
    else:
        metadata=read_json(source/'model.safetensors.json')
    tokenizer=Tokenizer(TOKENIZER)
    corpus=load_corpus(PREPARED/'corpus.json')
    from sml_v1.sample_prompts import read_prompts
    read_prompts(PREPARED/'sample_prompts.json')
    manifest=training_manifest(corpus,tokenizer,metadata['recipe']['stream'])
    recipe=validate(copy.deepcopy(metadata['recipe'])) if resume else phase_recipe(metadata,manifest)
    if KEY not in recipe or recipe[KEY]['manifest']!=fingerprint(manifest):
        raise ValueError('Existing phase must resume its own unchanged data contract')
    if tokenizer.vocab_size!=recipe['model']['vocab_size'] or len(recipe['batches'])!=5:
        raise ValueError('Model/tokenizer or five-rank configuration differs')
    contract=dict(recipe=fingerprint(recipe),tokenizer=tokenizer.fingerprint,
                  data=fingerprint(manifest),code=fingerprint(modules()))
    transition=check_resume(metadata,contract,recipe,manifest)
    load_panel(PREPARED/'code_validation.npz',manifest['code_validation'],manifest,tokenizer,recipe)
    if full_verify:
        with gzip.open(source/'model.safetensors.rank0.data_state.json.gz','rt') as f:
            saved=json.load(f)['stream_state']
        cursor=saved if resume else migrate_cursor(saved,metadata,manifest,recipe)
        check=CodePhaseStream(manifest,tokenizer,'train',recipe['model']['max_seq_len'],cursor,
                              exclusions=manifest['code_validation']['excluded_content_keys'])
        check.close()  # No rows fetched; checks saved buffers/accounting on CPU.
        if cursor.get('exhausted_sources'):
            raise ValueError('A required source is already exhausted; cannot preserve the fixed mixture')
    scheduler=restore_scheduler(recipe,metadata)
    start=recipe[KEY]['start_tokens']
    remaining=max(0,100_000_000_000-start)
    capacity=readiness['capacity']
    if (remaining*.15>capacity['python_conservative_tokens']
            or remaining*.05>capacity['sql_conservative_tokens']):
        raise ValueError('Sampled filtered supply is insufficient for the proposed 100B-total milestone')
    result=dict(mode='resume this code phase' if resume else 'new code phase from latest full pretraining checkpoint',
        source=str(source),output=str(OUTPUT),start_step=metadata['step'],start_tokens=metadata['tokens'],
        transition=transition,next_lr=effective_learning_rate(metadata['tokens']+tokens_per_update(recipe),recipe,scheduler),
        schedule='preserve existing gradual decay to 3e-5 at 100B total tokens, then hold',
        token_shares=manifest['weights'],effective_row_limit=None,
        data='live public Parquet streaming; bounded buffers; no corpus download or source recycling',
        sql_order='Stack-Edu SQL first, then schema/question/query SynSQL; outer SQL allocation stays 5%',
        optimizer='restore full FP32 AdamW state',cursor='preserve old source positions and pending input',
        validation='same prose validation/best criterion plus separate frozen Python and SQL losses',
        supply='sample estimates, not a full token inventory or global uniqueness guarantee',capacity=capacity)
    return result,source,metadata,recipe,corpus


def correctness_probe(cluster,job):
    """Only called after the user explicitly runs this launcher on an idle rack."""
    from five_mac_cluster import PYTHON,ORDER
    probe_job=job/'correctness';probe_job.mkdir()
    cluster.preflight(probe_job)
    source=probe_job/'input';source.mkdir()
    shutil.copytree(PREPARED/'runtime/sml_v1',source/'sml_v1',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(ROOT/'scripts/probe_five_mac.py',source/'probe_five_mac.py')
    cluster.stage(source)
    cluster.launch(probe_job,source,[PYTHON,'-u',str(source/'probe_five_mac.py'),'--job-dir',str(probe_job)],timeout=360)
    receipts=[]
    for rank,physical in enumerate(ORDER):
        receipt=json.loads(cluster.remote(physical,['/bin/cat',str(probe_job/f'probe_rank{rank}.json')]))
        if receipt.get('rank')!=rank or receipt.get('status')!='passed':
            raise RuntimeError('Missing five-rank correctness receipt')
        receipts.append(receipt)
    if len({r['checkpoint_digest'] for r in receipts})!=1:
        raise RuntimeError('Five-rank replicas differ')
    atomic_json(job/'correctness.json',dict(modules=modules(),receipts=receipts))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--clear-stop',action='store_true')
    args=parser.parse_args(argv)
    if not args.run:
        result,*_=plan()
        print(json.dumps(result,indent=2))
        print('Plan only: no SSH, GPU work, training start or active-run changes.')
        return result
    # The current launcher holds the same network lock. Do not interrupt it.
    with lock(NETWORK/'.training-launch.lock'),lock(PARENT/'.launcher.lock'),lock(OUTPUT/'.launcher.lock'):
        from five_mac_cluster import Cluster,PYTHON
        from sml_v1.launch import stage
        result,source,metadata,recipe,corpus=plan(full_verify=True)
        if (OUTPUT/'STOP').exists() and not args.clear_stop:
            raise ValueError('New output STOP marker exists; use --clear-stop to resume explicitly')
        job=OUTPUT/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir(parents=True)
        cluster=Cluster(NETWORK)
        cluster.preflight(job)  # Rejects active workers; never stops an existing run.
        print(json.dumps(result,indent=2),flush=True)
        atomic_json(job/'plan.json',result)
        correctness_probe(cluster,job)
        verify_readiness()
        staged=stage(job,recipe,TOKENIZER,source,corpus)
        shutil.copyfile(PREPARED/'code_validation.npz',staged/'code_validation.npz')
        shutil.copyfile(PREPARED/'sample_prompts.json',staged/'sample_prompts.json')
        inputs=read_json(staged/'input_manifest.json')
        for name in ('code_validation.npz','sample_prompts.json'):
            inputs['files'][name]=file_sha256(staged/name)
        atomic_json(staged/'input_manifest.json',inputs)
        if {p.name:file_sha256(p) for p in (staged/'sml_v1').glob('*.py')}!=modules():
            raise ValueError('Staged code differs from the prepared version')
        cluster.stage(staged)
        if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        worker=['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','sml_v1.pretrain',
            '--config',str(staged/'recipe.json'),'--tokenizer',str(staged/'tokenizer'),
            '--corpus',str(staged/'corpus.json'),'--resume',str(staged/'checkpoint'),
            '--save-dir',str(OUTPUT),'--job-dir',str(job),'--ring-control','native',
            '--sample-prompts-file',str(staged/'sample_prompts.json'),
            '--code-validation-panel',str(staged/'code_validation.npz')]
        cluster.launch(job,staged,worker,timeout=None,stop_dir=OUTPUT)
        result=read_json(job/'result.json')
        if result.get('status') not in ('stopped','complete','data-exhausted'):
            raise RuntimeError('Missing clean worker completion')
        print(f'[verified-stop] {result}; log: {job/"train.log"}',flush=True)
        return result


if __name__=='__main__':main()
