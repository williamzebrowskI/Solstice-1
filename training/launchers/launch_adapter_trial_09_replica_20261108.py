#!/usr/bin/env python3
"""Plan or explicitly launch the prepared text-only chatbot SFT on this Mac."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'prepared/adapter_trial_09_replica_20261108'
RUNTIME=PREPARED/'runtime'
NETWORK=ROOT/'diagnostics/m5_migration_20260922/network_20261102_234513'
sys.path.insert(0,str(RUNTIME))
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sml_v1.checkpoint_metadata import verify_bundle_metadata
from sml_v1.tokenization import Tokenizer
from sft.text_corpus import verify
from sft.retention import verify as verify_retention


def module_hashes(root):
    return {str(p.relative_to(root)):file_sha256(p) for package in ('sml_v1','sft')
            for p in sorted((root/package).rglob('*.py'))}


def checkpoint(pointer):
    pointer=Path(pointer)
    if pointer.is_symlink():raise ValueError('Checkpoint pointer cannot be a symlink')
    info=read_json(pointer);name=info.get('bundle','')
    if info.get('format')!='opensml-lora-sft-bundle-v1' or Path(name).name!=name or not name.startswith('step_'):
        raise ValueError('Invalid checkpoint pointer')
    path=pointer.parent/name
    if path.is_symlink():raise ValueError('Checkpoint cannot be a symlink')
    return path


def idle_local():
    result=subprocess.run(['/bin/ps','-axo','pid=,command='],capture_output=True,text=True,check=True)
    busy=[line.strip() for line in result.stdout.splitlines() if ' -c ' not in line and any(term in line for term in
        ('-m sml_v1.pretrain','-m sft.train','-m sft.text_train','-m sft.text_assess',
         '-m sft.probe','-m sft.lora_train','-m sft.lora_assess','-m preference.train','-m preference.preflight','mlx.launch',
         'scripts/benchmark_best.py','scripts/benchmark_ifeval.py','scripts/benchmark_chat_text.py'))]
    if busy:raise ValueError('Pretraining or another training/evaluation worker is active. Stop it cleanly first; this launcher will not interrupt it: '+str(busy))


def verify_readiness():
    receipt=read_json(PREPARED/'readiness.json')
    if receipt['status']!='ready-for-user-launch':raise ValueError('CPU/data checks have not passed')
    for name,digest in receipt['files'].items():
        path=ROOT.parent/name
        if not path.is_file() or file_sha256(path)!=digest:raise ValueError('Prepared file changed after checks: '+name)
    return receipt


def inspect():
    verify_readiness();plan=read_json(PREPARED/'plan.json');manifest=verify(plan['prepared'],plan)
    if (plan['format']!='opensml-verified-chat-v2' or plan['execution']!='single-mac' or plan['batches']!=[8]
        or plan['grad_accum']!=8 or plan['epochs'] not in (1,2) or plan['sft_rows_per_update']!=48 or plan['replay_rows_per_update']!=16
        or abs(plan['sft_objective_weight']+plan['replay_objective_weight']-1)>1e-8):raise ValueError('Unsupported training plan')
    output=Path(plan['output'])
    if output.parent!=ROOT/'runs' or output==Path(plan['source_run']):raise ValueError('Use a separate SFT output')
    pinned=output/'input'
    source=pinned/'source' if pinned.exists() else Path(plan['source_checkpoint'])
    metadata=read_json(source/'model.safetensors.json');tokenizer=Tokenizer(plan['tokenizer'])
    model=metadata['recipe']['model']
    if metadata.get('training_kind')=='sft' or 'sft_contract' in metadata:raise ValueError('Expected a pretrained parent, not an earlier SFT attempt')
    if (metadata['v2_contract']['tokenizer']!=tokenizer.fingerprint or manifest['tokenizer']!=tokenizer.fingerprint
        or model['vocab_size']!=tokenizer.vocab_size or model['max_seq_len']!=plan['max_sequence_length']
        or model['n_layers']!=44 or model['d_model']!=576):raise ValueError('Parent model/tokenizer/context differs')
    verify_retention(plan['retention'],metadata['validation_fingerprint'])
    comparison=read_json(Path(plan['comparison_checkpoint'])/'model.safetensors.json')
    if (comparison['step']!=167 or comparison['compute_precision']!='float32'
        or comparison['recipe']['model']!=model or comparison['sft_contract']['tokenizer']!=tokenizer.fingerprint
        or comparison.get('inference_export',{}).get('precision')!='exact FP32 optimizer masters'):
        raise ValueError('Expected the preserved exact FP32 checkpoint 167 with matching architecture and tokenizer')
    description=dict(execution='single-mac',training_started=False,source=str(source),source_step=metadata['step'],source_tokens=metadata['tokens'],
        source_selection='Pinned parent retained for this SFT run' if pinned.exists() else 'Final 42.001B pretrained checkpoint; frozen base, fresh LoRA and tied chat-token deltas',
        minimum_source_tokens=plan['minimum_source_tokens'],source_ready=metadata['tokens']>=plan['minimum_source_tokens'],
        counts=manifest['counts'],sequence_length=plan['max_sequence_length'],truncated=0,total_steps=manifest['total_steps'],
        peak_lr=plan['peak_lr'],final_lr=plan['final_lr'],optimizer='FP32 AdamW for LoRA and two tied chat-token deltas; pretrained base frozen',
        adaptation=plan['adaptation'],epochs=plan['epochs'],comparison='Preserved checkpoint 167; behavior measured on identical private development cases',
        evaluation='320 checked responses plus 40 qualitative responses every 200 updates; reserved test assessment; historical cases reused',
        output=str(output),launch='bash pretraining/scripts/run_deep44_token_lora_chat_sft.sh')
    if pinned.exists():
        if read_json(pinned/'plan.json')!=plan:raise ValueError('Pinned SFT plan changed')
        c=read_json(pinned/'contract.json')
        if c['modules']!=module_hashes(RUNTIME) or c['data']!=fingerprint(manifest):raise ValueError('Pinned SFT code/data changed')
    return plan,manifest,source,metadata,description


def stage_input(plan,manifest,source,output):
    pinned=output/'input'
    if pinned.exists():return pinned
    stage=output/('.input-'+uuid.uuid4().hex);stage.mkdir(parents=True)
    try:
        verify_bundle_metadata(source)
        for package in ('sml_v1','sft'):shutil.copytree(RUNTIME/package,stage/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for original,name in ((source,'source'),(plan['prepared'],'corpus'),(plan['retention'],'retention'),(plan['tokenizer'],'tokenizer')):
            shutil.copytree(original,stage/name)
        if plan.get('comparison_checkpoint'):
            comparison=Path(plan['comparison_checkpoint']);(stage/'comparison').mkdir()
            for name in ('model.safetensors','model.safetensors.json'):
                shutil.copyfile(comparison/name,stage/'comparison'/name)
        verify_bundle_metadata(stage/'source');verify(stage/'corpus',plan)
        # Export in a short-lived process so all temporary model memory is
        # released before the SFT worker starts.
        subprocess.run([sys.executable,'-u','-m','sft.text_preserve','--source',str(stage/'source'),
                        '--output',str(stage/'source/masters')],cwd=stage,
                       env=dict(os.environ,PYTHONPATH=str(stage)),check=True)
        modules=module_hashes(stage)
        if modules!=module_hashes(RUNTIME):raise ValueError('Worker code changed during staging')
        # Keep the established checkpoint envelope so existing SFT readers work.
        contract=dict(format='opensml-grounded-sft-contract-v1',plan=fingerprint(plan),data=fingerprint(manifest),
            tokenizer=manifest['tokenizer'],source_weights=file_sha256(stage/'source/model.safetensors'),
            source_metadata=file_sha256(stage/'source/model.safetensors.json'),source_masters=file_sha256(stage/'source/masters/model.safetensors'),
            retention=file_sha256(stage/'retention/manifest.json'),modules=modules)
        if plan.get('comparison_checkpoint'):
            contract['comparison_files']={name:file_sha256(stage/'comparison'/name) for name in ('model.safetensors','model.safetensors.json')}
        atomic_json(stage/'plan.json',plan);atomic_json(stage/'contract.json',contract)
        atomic_json(stage/'parent.json',dict(original_checkpoint=str(source),original_manifest=read_json(source/'manifest.json')))
        verify_readiness();stage.rename(pinned)
        return pinned
    finally:
        if stage.exists():shutil.rmtree(stage)


def run_local(job,staging,command,output):
    env=worker_environment(staging)
    atomic_json(job/'launch.json',dict(command=command,execution='single-mac',ssh=False))
    process=subprocess.Popen(command,cwd=staging,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    def stop(sig,frame):
        (output/'STOP').touch();print('[stop] Save/stop requested; waiting for the local worker',flush=True)
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with (job/'train.log').open('w') as log:
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        status=process.wait()
        if status:raise RuntimeError(f'SFT exited with code {status}; see {job/"train.log"}')
    finally:
        if process.poll() is None:
            (output/'STOP').touch()
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        process.stdout.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return env


def worker_environment(staging):
    env=dict(os.environ,PYTHONPATH=str(staging),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for key in list(env):
        if key.startswith(('MLX_','JACCL_','SML5_','SML_JACCL_')):env.pop(key)
    return env


def assess(staging,output):
    subprocess.run(['/usr/bin/caffeinate','-i',sys.executable,'-u','-m','sft.lora_assess','--input',str(staging),'--output',str(output)],
                   cwd=staging,env=worker_environment(staging),check=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--clear-stop',action='store_true');a=p.parse_args(argv)
    plan,manifest,source,metadata,description=inspect();print(json.dumps(description,indent=2),flush=True)
    if not a.run:print('Prepared; no training started.');return description
    output=Path(plan['output'])
    # Share the five-Mac launch lock so no continuation starts during local SFT.
    with lock(NETWORK/'.training-launch.lock'),lock(ROOT/'.local-sft-launch.lock'),lock(output/'.launcher.lock'):
        idle_local();plan,manifest,source,metadata,_=inspect()
        if metadata['tokens']<plan['minimum_source_tokens']:raise ValueError('Wait for a saved checkpoint at or beyond 42B pretraining tokens')
        if (output/'STOP').exists() and not a.clear_stop:raise ValueError('Saved STOP marker exists; use --clear-stop to resume intentionally')
        staging=stage_input(plan,manifest,source,output);contract=read_json(staging/'contract.json');resume=None
        if (output/'latest.json').exists():
            from sft.lora_checkpoint import verify as verify_lora_checkpoint
            resume=checkpoint(output/'latest.json');saved=verify_lora_checkpoint(resume)
            if saved['contract']!=contract:raise ValueError('SFT resume contract changed')
            if saved['quality_stop']:raise ValueError('This candidate failed retention checks; do not automatically continue it')
            if saved['step']>=manifest['total_steps']:
                print('This SFT run has already completed.')
                if not (output/'assessment/summary.json').exists():assess(staging,output)
                return saved
        job=output/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]);job.mkdir(parents=True)
        if a.clear_stop:(output/'STOP').unlink(missing_ok=True)
        command=['/usr/bin/caffeinate','-i',sys.executable,'-u','-m','sft.lora_train',
            '--config',str(staging/'plan.json'),'--data',str(staging/'corpus'),'--tokenizer',str(staging/'tokenizer'),
            '--source',str(staging/'source'),'--source-masters',str(staging/'source/masters/model.safetensors'),
            '--contract',str(staging/'contract.json'),'--save-dir',str(output),'--job-dir',str(job),'--retention',str(staging/'retention')]
        if plan.get('comparison_checkpoint'):command+=['--reference',str(staging/'comparison')]
        if resume:command+=['--resume',str(resume)]
        env=run_local(job,staging,command,output);result=read_json(job/'result.json')
        if result['status'] not in ('stopped','complete','quality-stopped'):raise ValueError('Missing clean completion')
        print(f'[verified-stop-sft] {result}; log: {job/"train.log"}',flush=True)
        if result['status'] in ('complete','quality-stopped'):
            assess(staging,output)
        return result


if __name__=='__main__':main()
