#!/usr/bin/env python3
"""Five authorized repair trials; freeze one dev-selected finalist and retain it."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT/'pretraining/prepared/chat_repair_search_20261109'
REPORT=ROOT/'reports/sft_search/chat_repair_search_20261109'
sys.path.insert(0,str(SHARED/'runtime'))
from sml_v1.common import atomic_json, read_json, file_sha256, fingerprint, lock
from sft.lora_repair_common import PARENTS, ASSETS, SPECS, REPLAY_MIX, DOMAINS, receipt, modules, compact, legacy
from sft.lora_checkpoint import verify, resolve
from sft.baseline_cache import identity

STOP=False
CURRENT=None


def run_path(number):
    if type(number)!=int or not 1<=number<=5:raise ValueError('Only the authorized five trials')
    return ROOT/f'pretraining/runs/deep44_chat_repair_{number:02d}_20261109'


def status(phase,**values):
    atomic_json(REPORT/'status.json',dict(updated=time.time(),phase=phase,maximum_attempts=5,
                run=str(CURRENT) if CURRENT else None,**values))


def stopping():
    return STOP or (REPORT/'STOP').exists() or (CURRENT is not None and (CURRENT/'STOP').exists())


def protected():
    for path,sha in read_json(REPORT/'protected.json').items():
        if file_sha256(path)!=sha:raise ValueError('Protected file changed: '+path)


def ready():
    d=read_json(REPORT/'readiness.json')
    if d['status']!='tests-passed':raise ValueError('Workflow tests have not passed')
    for path,sha in d['files'].items():
        if file_sha256(ROOT/path)!=sha:raise ValueError('Frozen workflow changed: '+path)
    receipt()


def active(path):
    lines=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines()
    return [l for l in lines if str(path) in l and any(s in l for s in ('-m sft.', 'chat_adapter_search_bench.py')) and int(l.split()[0])!=os.getpid()]


def copy(source,dest):
    if source.is_dir():shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)


def combo():
    results=[read_json(REPORT/f'trial_{n:02d}.json') for n in range(1,5)]
    parent=max(results[:2],key=lambda r:r['quality']['key'])['parent']
    old=read_json(REPORT/'baselines'/parent/'dev.json')
    scores={d:1+max(0,max(r['new_dev']['counts'][d] for r in results)-old['counts'][d]) for d in DOMAINS}
    weights={d:.1+.5*s/sum(scores.values()) for d,s in scores.items()}
    for _ in range(10):
        excess=sum(max(0,w-.4) for w in weights.values())
        if excess<1e-12:break
        weights={d:min(w,.4) for d,w in weights.items()};free=[d for d in weights if weights[d]<.4]
        for d in free:weights[d]+=excess/len(free)
    weights={d:w/sum(weights.values()) for d,w in weights.items()}
    result=dict(name=SPECS[4]['name'],parent=parent,mixture=weights,
                decision='Original parent chosen by dev-ranked matched repairs 1/2; domain weights use only dev gains from trials 1-4. Never chain candidate weights.')
    atomic_json(REPORT/'trial_05_design.json',result);return result


def prepare(number,spec=None):
    output=run_path(number);staging=output/'input'
    if staging.exists():
        if active(output):raise ValueError('Existing trial worker is active')
        if not (staging/'contract.json').exists():raise ValueError('Incomplete staging; inspect before recovery')
        return output,staging
    spec=spec or SPECS[number-1];parent=PARENTS[spec['parent']]
    output.mkdir(exist_ok=False);staging.mkdir()
    for package in ('sft','sml_v1'):copy(SHARED/'runtime'/package,staging/package)
    for name in ('corpus','retention','tokenizer','comparison'):copy(ASSETS/name,staging/name)
    for name in ('model.safetensors','model.safetensors.json','masters/model.safetensors'):copy(ASSETS/'source'/name,staging/'source'/name)
    copy(parent,staging/'initial_parent')
    plan=read_json(SHARED/'template_plan.json');lr=1.5e-5 if number==5 else 2e-5
    plan.update(initial_checkpoint=str(parent),prepared=str(staging/'corpus'),output=str(output),
                repair_parent=spec['parent'],method=spec['name'],seed=2026110915,
                eval_every=64,log_every=10,microbatch_rows=4,rows_per_group=16,max_steps=256,
                fresh_mixture=spec['mixture'],replay_mixture=REPLAY_MIX,fresh_weight=.4,kl_weight=.25,
                peak_lr=lr,final_lr=lr*.1,warmup_fraction=.1,
                selection_reference='New 300-case development test, with historical 320-case and prose retention guards. Final 1000-case test is unopened until a single finalist is frozen.',
                strategy='Independent small adapter updates from parent2000 or preserved attempt5; verified text tasks and 60% replay objective with frozen-source probability preservation.')
    contract=read_json(ASSETS/'contract.json')
    contract.update(plan=fingerprint(plan),modules=modules(staging),initial_parent_manifest=file_sha256(parent/'manifest.json'),repair_data=fingerprint(receipt()))
    atomic_json(staging/'plan.json',plan);atomic_json(staging/'contract.json',contract);verify(staging/'initial_parent')
    if identity(staging)!=read_json(REPORT/'baselines/legacy_manifest.json')['identity']:
        raise ValueError('Historical evaluation identity changed')
    return output,staging


def worker(command,log,staging,phase):
    env=dict(os.environ,PYTHONPATH=str(staging),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for key in list(env):
        if key.startswith(('MLX_','JACCL_','SML5_','SML_JACCL_')):env.pop(key)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',*command],cwd=staging,
                env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(REPORT/'worker_launch.json',dict(pid=child.pid,command=command,phase=phase,log=str(log)))
        signalled=False
        while child.poll() is None:
            status(phase,worker_pid=child.pid)
            if stopping():
                if phase.startswith('training-'):(CURRENT/'STOP').touch()
                elif not signalled:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    signalled=True
            time.sleep(3)
    if child.returncode and not stopping():raise RuntimeError(f'{phase} failed ({child.returncode}); artifacts retained in {log}')


def dashboard(path):
    for row in subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines():
        if 'scripts/training_dashboard.py --' in row and '--port 8768' in row:
            if not any(n in row for n in ('deep44_chat_repair_', 'deep44_parent2000_generalize_05_replica_')):return
            os.kill(int(row.split()[0]),signal.SIGTERM)
    with (REPORT/'dashboard.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'pretraining/scripts/training_dashboard.py'),
            '--run-dir',str(ROOT/'pretraining/runs/deep44_cooldown_2b_20261107'),'--port','8768',
            '--sft-run-dir',str(path),'--parent-step','320445'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(REPORT/'dashboard.json',dict(pid=p.pid,run=str(path)))


def describe(number):
    output=run_path(number);bundle=resolve(output/'best.json');state=verify(bundle);step=state['step']
    value=dict(attempt=number,parent=read_json(output/'input/plan.json')['repair_parent'],checkpoint=str(bundle),
               checkpoint_manifest=file_sha256(bundle/'manifest.json'),step=step,
               quality=read_json(output/'evaluations'/f'quality_{step:07d}.json'),
               new_dev=compact(read_json(output/'evaluations'/f'repair_{step:07d}.json')),
               legacy_dev=compact(read_json(output/'evaluations'/f'eval_{step:07d}.json')),
               status=read_json(output/'result.json')['status'],plan=read_json(output/'input/plan.json'))
    atomic_json(REPORT/f'trial_{number:02d}.json',value);return value


def final_report(winner,public):
    from sft.lora_repair_eval import paired
    candidate=read_json(REPORT/'final/test.json');new=[];reasons=[]
    for name in PARENTS:
        base=read_json(REPORT/'baselines'/name/'test.json');comparison=paired(candidate,base)
        new.append(dict(baseline=name,**comparison))
        if comparison['gain']<.05-1e-12 or comparison['confidence_interval_97_5'][0]<=0:
            reasons.append(name+': new-test gain below 5pp or interval includes zero')
        for d in DOMAINS:
            if candidate['counts'][d]<base['counts'][d]-6:reasons.append(name+': new-test '+d+' regression over 3pp')
        if candidate['natural_stop']<base['natural_stop']-.01 or candidate['repetition_rate']>base['repetition_rate']+.01:
            reasons.append(name+': response control regression')
        if candidate['strict_correct']<base['strict_correct']-10:
            reasons.append(name+': strict instruction compliance fell over 1pp')
    old=read_json(REPORT/'final/legacy/final_test'/f'eval_{winner["step"]:07d}.json')
    for name in PARENTS:
        base=legacy(name,'test')
        if sum(old['generated_pass_counts'].values())<sum(base['generated_pass_counts'].values())-7:
            reasons.append(name+': historical total fell over 7/704')
        for d in base['generated_pass_counts']:
            if old['generated_pass_counts'][d]<base['generated_pass_counts'][d]-4:reasons.append(name+': historical '+d+' fell over 4 cases')
        if old['retention']['val_loss']>base['retention']['val_loss']+.003:reasons.append(name+': prose retention')
    baselines=[read_json(REPORT/'baselines'/(n+'_public.json')) for n in PARENTS]
    names=list(public['choices'])
    if sum(public['choices'][n]['acc_norm'] for n in names)/len(names)<max(sum(b['choices'][n]['acc_norm'] for n in names)/len(names) for b in baselines)-.0025:
        reasons.append('mean choice accuracy fell over 0.25pp')
    for n in names:
        if public['choices'][n]['acc_norm']<max(b['choices'][n]['acc_norm'] for b in baselines)-.01:reasons.append(n+' fell over 1pp')
    for kind in ('strict','loose'):
        for metric in ('prompt_accuracy','instruction_accuracy'):
            tolerance=0 if kind=='strict' else .005
            target=max(b['ifeval'][kind][metric] for b in baselines)-tolerance
            if public['ifeval'][kind][metric]+1e-12<target:reasons.append('IFEval '+kind+' '+metric)
    for name in PARENTS:
        b=read_json(REPORT/'baselines'/(name+'_public.json'));a=public['chat'];c=b['chat']
        if a['objective']['answer_match_with_neutral_wrapper']<c['objective']['answer_match_with_neutral_wrapper']-2:reasons.append(name+': chat accuracy fell over 2/180')
        if a['response_control_all']['natural_stop']<c['response_control_all']['natural_stop']-1:reasons.append(name+': chat stopping')
        if a['response_control_all']['repeated']>c['response_control_all']['repeated']+1 or a['response_control_all']['empty']:reasons.append(name+': chat repetition or empty')
    if not winner['quality']['eligible']:reasons.append('No development candidate passed preservation guards')
    result=dict(checkpoint=winner['checkpoint'],attempt=winner['attempt'],preserved=True,
                significant_on_reserved_panel_and_passed_guards=not reasons,reasons=reasons,
                comparisons=new,new_test=compact(candidate),legacy_test=compact(old),public=public,
                scope='One dev-selected finalist; reserved authored task panel is not a complete measure of conversational quality. Public and historical suites were already used in earlier experiments.')
    atomic_json(REPORT/'final/comparison.json',result);return result


def cleanup(winner):
    protected();receipts=[]
    for number in range(1,6):
        path=run_path(number)
        if number==winner['attempt'] or (path/'KEEP.json').exists():continue
        if active(path):raise ValueError('Refuse cleanup while a trial is active')
        if path.is_symlink():raise ValueError('Refuse cleanup of a symlink')
        if path.exists():
            size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file());shutil.rmtree(path)
            receipts.append(dict(path=str(path),freed_bytes=size))
    protected();atomic_json(REPORT/'cleanup.json',receipts)


def main():
    global CURRENT
    REPORT.mkdir(parents=True,exist_ok=True)
    network=ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261102_234513/.training-launch.lock'
    with lock(REPORT/'.controller.lock'),lock(network),lock(ROOT/'pretraining/.local-sft-launch.lock'):
        ready();protected()
        if (REPORT/'complete.json').exists():print('All five attempts already finished.');return
        if stopping():status('stopped');return
        # Only development baselines are visible before candidate selection.
        for name,bundle in PARENTS.items():
            output=REPORT/'baselines'/name/'dev.json'
            if not output.exists():worker(['-m','sft.lora_repair_eval','--bundle',str(bundle),'--output',str(output)],REPORT/(name+'_dev.log'),SHARED/'runtime','baseline-'+name)
            if stopping():status('stopped');return
        if not (REPORT/'selection_frozen.json').exists():
            for number in range(1,6):
                if stopping():status('stopped');return
                if (REPORT/f'trial_{number:02d}.json').exists():continue
                spec=SPECS[number-1] if number<5 else combo()
                ready();CURRENT,staging=prepare(number,spec);dashboard(CURRENT)
                if not (CURRENT/'result.json').exists() or read_json(CURRENT/'result.json')['status']=='stopped':
                    worker(['-m','sft.lora_repair_train','--input',str(staging),'--output',str(CURRENT)],CURRENT/'jobs/local/train.log',staging,f'training-{number}')
                if stopping():status('stopped');return
                describe(number)
            candidates=[read_json(REPORT/f'trial_{n:02d}.json') for n in range(1,6)]
            winner=max(candidates,key=lambda r:r['quality']['key'])
            atomic_json(REPORT/'selection_frozen.json',dict(winner=winner,selection='Highest predeclared development key; public and final-test results cannot change this choice',
                        test_data_sha256=file_sha256(SHARED/'data/test.jsonl'),allowed_test_manifests=[winner['checkpoint_manifest'],*[file_sha256(p/'manifest.json') for p in PARENTS.values()]]))
        winner=read_json(REPORT/'selection_frozen.json')['winner'];CURRENT=Path(winner['checkpoint']).parent;staging=CURRENT/'input'
        atomic_json(CURRENT/'KEEP.json',dict(reason='User requested keeping the best of five; selected using development results only',checkpoint=winner['checkpoint'],preserve_regardless_of_final_results=True))
        dashboard(CURRENT)
        for name,bundle,output in [(n,p,REPORT/'baselines'/n/'test.json') for n,p in PARENTS.items()]+[('final',Path(winner['checkpoint']),REPORT/'final/test.json')]:
            if not output.exists():worker(['-m','sft.lora_repair_eval','--bundle',str(bundle),'--output',str(output),'--split','test'],REPORT/(name+'_test.log'),staging,'reserved-test-'+name)
            if stopping():status('stopped');return
        legacy_output=REPORT/'final/legacy/final_test'/f'eval_{winner["step"]:07d}.json'
        if not legacy_output.exists():
            worker(['-m','sft.lora_repair_legacy','--bundle',winner['checkpoint'],'--input',str(staging),'--output',str(REPORT/'final/legacy')],REPORT/'final_legacy.log',staging,'historical-final-test')
        if stopping():status('stopped');return
        bench=REPORT/'final/public'
        if not (bench/'summary.json').exists():worker([str(ROOT/'pretraining/scripts/chat_adapter_search_bench.py'),winner['checkpoint'],str(staging),str(bench)],REPORT/'final_public.log',staging,'full-public-benchmarks')
        if stopping():status('stopped');return
        result=final_report(winner,read_json(bench/'summary.json'))
        cleanup(winner)
        atomic_json(REPORT/'complete.json',dict(status='complete-best-preserved',attempt=winner['attempt'],checkpoint=winner['checkpoint'],
                    significant_on_reserved_panel_and_passed_guards=result['significant_on_reserved_panel_and_passed_guards']))
        status('complete-best-preserved',attempt=winner['attempt'],checkpoint=winner['checkpoint'],significant_on_reserved_panel_and_passed_guards=result['significant_on_reserved_panel_and_passed_guards'])


def stop(*_):
    global STOP
    STOP=True


if __name__=='__main__':
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:main()
    except Exception as error:
        status('error-artifacts-retained',error=repr(error));raise
