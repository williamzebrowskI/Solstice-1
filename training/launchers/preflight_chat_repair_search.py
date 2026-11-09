"""Freeze inputs and validate the real model before the bounded search starts."""
import json
from pathlib import Path
import re
import shutil
import sys
import run_chat_repair_search as run
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint,lock
from sml_v1.tokenization import Tokenizer
from sft.lora_repair_common import PARENTS,ASSETS,SHARED,REPORT,SPECS,setup
from sft.lora_replay_objective import orders,encode
from sft.lora_repair_audit import audit
from sft.baseline_cache import identity
import chat_adapter_search_bench as benchmark


def main():
    if (REPORT/'readiness.json').exists():raise ValueError('Preflight is already frozen; verify rather than overwrite')
    tests=(REPORT/'tests.log').read_text()
    if 'Ran 9 tests' not in tests or '\nOK\n' not in tests:raise ValueError('Required tests have not passed')
    audit();benchmark.verify_scoring()
    protected=read_json(run.ROOT/'reports/sft_search/parent2000_generalize_20261109/protected.json')
    replica=PARENTS['attempt5'].parent
    for bundle in replica.glob('step_*'):
        if bundle.is_dir():
            for p in bundle.rglob('*'):
                if p.is_file():protected[str(p)]=file_sha256(p)
    protected[str(replica/'KEEP.json')]=file_sha256(replica/'KEEP.json')
    atomic_json(REPORT/'protected.json',protected);run.protected()
    sources={
      'parent2000_legacy_dev.json':PARENTS['parent2000'].parent/'evaluations/eval_0002000.json',
      'parent2000_legacy_test.json':PARENTS['parent2000'].parent/'assessment/latest_unselected/final_test/eval_0002000.json',
      'attempt5_legacy_dev.json':replica/'evaluations/eval_0000256.json',
      'attempt5_legacy_test.json':replica/'assessment/final_test/eval_0000256.json'}
    b=REPORT/'baselines';b.mkdir(exist_ok=True)
    for name,source in sources.items():shutil.copy2(source,b/name)
    atomic_json(b/'legacy_manifest.json',dict(identity=identity(ASSETS),files={n:file_sha256(b/n) for n in sources},
                parents={n:file_sha256(p/'manifest.json') for n,p in PARENTS.items()}))
    public=run.ROOT/'reports/sft_search/parent2000_generalize_05_replica_20261109'
    for name,folder in [('parent2000','parent_benchmarks'),('attempt5','replica_benchmarks')]:
        value=read_json(public/folder/'summary.json')
        if not value['complete'] or value['model_sha256']!=file_sha256(PARENTS[name]/'inference/model.safetensors'):
            raise ValueError('Public baseline is incomplete or uses different weights')
        shutil.copy2(public/folder/'summary.json',b/(name+'_public.json'))
    tok=Tokenizer(ASSETS/'tokenizer')
    fresh=[json.loads(l) for l in (SHARED/'data/train.jsonl').read_text().splitlines()]
    replay=[json.loads(l) for l in (SHARED/'data/replay.jsonl').read_text().splitlines()]
    maximum=0
    for r in replay:
        texts=[m['content'] for m in r['messages']]
        if any(re.search(r'\b(?:python|sql|json)\b|```',t,re.I) for t in texts):raise ValueError('Excluded task in inherited replay')
        length=len(encode(tok,r,False)[0]);maximum=max(maximum,length)
        if length>2048:raise ValueError('Replay truncation would be required')
    network=run.ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261102_234513/.training-launch.lock'
    with lock(network),lock(run.ROOT/'pretraining/.local-sft-launch.lock'):
        output,staging=run.prepare(1)
        plan=read_json(staging/'plan.json');contract=read_json(staging/'contract.json')
        checks=[]
        for n,spec in enumerate(SPECS,1):
            mixture=spec['mixture'] if n<5 else {d:.2 for d in run.DOMAINS}
            # Plans are saved with sorted JSON keys; choose() visits mixture
            # entries in that stored order when drawing deterministic samples.
            candidate=json.loads(json.dumps(dict(plan,fresh_mixture=mixture),sort_keys=True))
            new,old=orders(tok,fresh,replay,candidate)
            assert len(new)==len(old)==256
            assert len(set(i for group in new for i in group))==4096
            assert len(set(i for group in old for i in group))==4096
            checks.append(dict(attempt=n,order_hash=fingerprint([new,old]),fresh_selected=4096,replay_selected=4096,steps=256,
                               combined_trial_uses_provisional_balanced_mixture=n==5))
        assert checks[0]['order_hash']==checks[1]['order_hash']
        model,opt,base,cfg,meta,digest,count,ancestry=setup(staging,plan,contract)
        if count!=3785856 or ancestry['parent_step']!=256:raise ValueError('Unexpected real-model initialization')
        atomic_json(REPORT/'production_preflight.json',dict(exact_attempt5_initialization=True,trainable_parameters=count,
                    fresh_optimizer=True,maximum_replay_tokens=maximum,orders=checks,training_started=False))
    design=dict(maximum_attempts=5,trials=SPECS,updates_per_attempt=256,selected_new_conversations=4096,selected_replay_conversations=4096,
                peak_lr=2e-5,combined_peak_lr=1.5e-5,new_objective_weight=.4,replay_objective_weight=.6,replay_kl_weight=.25,
                final_lr_fraction=.1,warmup_fraction=.1,seed=2026110915,
                reserved_test_cases=1000,development_cases=300,legacy_dev_cases=320,legacy_test_cases=704,
                fifth_trial='Choose original starting checkpoint using matched repair trials 1/2 dev scores; allocate 10%-40% per domain using positive dev gains from first four. No chaining of experimental checkpoints.',
                selection='Highest dev key with preservation guards; one finalist is frozen before reserved test or new public benchmark evaluation.',
                practical_success='At least 5pp new heldout content correctness gain over EACH baseline with 97.5% paired scenario-bootstrap intervals above zero; preserve strict instruction compliance, domains and general benchmarks within predeclared tolerances.',
                public_tolerances=dict(choice_mean_pp=.25,choice_per_task_pp=1,ifeval_strict_pp=0,ifeval_loose_pp=.5,chat_answer_cases=2,chat_stop_cases=1,chat_repetition_cases=1),
                preservation='Both starting models and previous protected checkpoints remain immutable. Keep the highest-dev-ranked new candidate even if final results do not establish an improvement. Remove other new trial artifacts only after all final assessments complete.')
    atomic_json(REPORT/'design.json',design)
    if (SHARED/'draft_data').exists():shutil.rmtree(SHARED/'draft_data')
    paths=[p for p in (SHARED/'runtime').rglob('*.py')]
    paths += [SHARED/'template_plan.json',SHARED/'data/manifest.json',*list((SHARED/'data').glob('*.jsonl'))]
    paths += [Path(__file__),run.ROOT/'pretraining/scripts/run_chat_repair_search.py',run.ROOT/'pretraining/scripts/chat_adapter_search_bench.py',run.ROOT/'pretraining/tests/test_chat_repair_search.py',REPORT/'tests.log',REPORT/'protected.json',REPORT/'design.json',*list(b.glob('*.json'))]
    files={str(p.relative_to(run.ROOT)):file_sha256(p) for p in paths}
    atomic_json(REPORT/'readiness.json',dict(status='tests-passed',tests=9,files=files))
    run.ready();print(json.dumps(dict(status='ready',trials=5,exact_initialization=True,tests=9,training_started=False)))


if __name__=='__main__':main()
