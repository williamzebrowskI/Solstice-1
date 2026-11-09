#!/usr/bin/env python3
"""Reproduce and permanently retain one trial, without restarting the search."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_parent2000_generalize as original
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json

ROOT = original.ROOT
RUN = ROOT/'pretraining/runs/deep44_parent2000_generalize_05_replica_20261109'
REPORT = ROOT/'reports/sft_search/parent2000_generalize_05_replica_20261109'
EXPECTED = original.REPORT/'trial_05_result.json'
STOP = False


def status(phase, **values):
    atomic_json(REPORT/'status.json', dict(updated=time.time(), phase=phase,
                run=str(RUN), preserved=True, maximum_training_runs=1, **values))


def stopping():
    return STOP or (REPORT/'STOP').exists() or (RUN/'STOP').exists()


def prepare():
    original.verify_workflow()
    original.protected()
    receipt = read_json(EXPECTED)
    if receipt['recipe'] != original.SPECS[4]:
        raise ValueError('Original attempt 5 recipe differs from the saved result')
    old_path = original.run_path
    try:
        def replica_path(number):
            if number != 5:
                raise ValueError('Only attempt 5 is authorized here')
            return RUN
        original.run_path = replica_path
        output, staging = original.prepare(5)
    finally:
        original.run_path = old_path
    atomic_json(output/'KEEP.json', dict(reason='User explicitly requested reproduction and preservation of additional-search attempt 5',
                preserve_all_checkpoints=True, preserve_regardless_of_scores=True,
                parent=str(original.PARENT), original_result=str(EXPECTED)))
    plan = read_json(staging/'plan.json')
    contract = read_json(staging/'contract.json')
    for key, value in dict(original.SPECS[4], seed=202611097, max_steps=256,
                           eval_every=64, rows_per_group=16, microbatch_rows=4).items():
        if plan[key] != value:
            raise ValueError('Replica setting mismatch: '+key)
    if contract['plan'] != fingerprint(plan) or contract['modules'] != original.modules(staging):
        raise ValueError('Replica frozen input contract changed')
    if contract['modules'] != original.modules(original.SHARED/'runtime'):
        raise ValueError('Replica runtime differs from the tested original')
    if contract['fresh_data'] != fingerprint(original.data_receipt()):
        raise ValueError('Training data differs from the original attempt')
    from sft.lora_search_common import load_baseline
    for split in ('dev', 'test'):
        load_baseline(staging, split)
    atomic_json(REPORT/'reproduction_inputs.json', dict(parent=str(original.PARENT),
                parent_manifest=file_sha256(original.PARENT/'manifest.json'),
                original_result=str(EXPECTED), original_result_sha256=file_sha256(EXPECTED),
                plan=plan, contract=contract, expected_development=receipt['development'],
                expected_assessment=receipt['assessment'], retained_before_training=True))
    return staging


def worker(command, log, staging, phase):
    env = dict(os.environ, PYTHONPATH=str(staging), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for key in list(env):
        if key.startswith(('MLX_', 'JACCL_', 'SML5_', 'SML_JACCL_')):
            env.pop(key)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        child = subprocess.Popen(['/usr/bin/caffeinate', '-i', sys.executable, '-u', *command],
                cwd=staging, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(REPORT/(phase+'_launch.json'), dict(pid=child.pid, command=command, log=str(log)))
        signalled = False
        while child.poll() is None:
            status(phase, worker_pid=child.pid)
            if stopping():
                if phase == 'training':
                    (RUN/'STOP').touch()
                elif not signalled:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    signalled = True
            time.sleep(3)
    if child.returncode and not stopping():
        raise RuntimeError(f'{phase} exited {child.returncode}; all files retained: {log}')


def compare(expected, actual, path='', differences=None):
    """Compare saved evaluation fields, allowing only tiny floating point noise."""
    differences = [] if differences is None else differences
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            differences.append(dict(field=path, expected=expected, actual=actual))
        else:
            for key, value in expected.items():
                compare(value, actual.get(key), path+'/'+key, differences)
    elif isinstance(expected, float) and isinstance(actual, (float, int)):
        if not math.isclose(expected, actual, rel_tol=0, abs_tol=1e-6):
            differences.append(dict(field=path, expected=expected, actual=actual))
    elif expected != actual:
        differences.append(dict(field=path, expected=expected, actual=actual))
    return differences


def run(prepare_only=False, benchmarks=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    network = ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261102_234513/.training-launch.lock'
    with lock(REPORT/'.controller.lock'), lock(network), lock(ROOT/'pretraining/.local-sft-launch.lock'):
        staging = prepare()
        if prepare_only:
            status('prepared')
            print(json.dumps(dict(run=str(RUN), preserved=True, steps=256, training_started=False)))
            return
        if stopping():
            status('stopped')
            return
        # Only the dashboard destination is changed; historical search files stay intact.
        old_report = original.REPORT
        try:
            original.REPORT = REPORT
            original.dashboard(RUN)
        finally:
            original.REPORT = old_report
        if not (RUN/'result.json').exists() or read_json(RUN/'result.json')['status'] == 'stopped':
            worker(['-m', 'sft.lora_search_train', '--input', str(staging), '--output', str(RUN)],
                   RUN/'jobs/local/train.log', staging, 'training')
        if stopping():
            status('stopped')
            return
        result = read_json(RUN/'result.json')
        if result['status'] != 'complete' or result['step'] != 256:
            status('retained-incomplete', training_status=result['status'], step=result['step'])
            return
        if not (RUN/'assessment/summary.json').exists():
            worker(['-m', 'sft.lora_search_assess', '--input', str(staging), '--output', str(RUN)],
                   RUN/'assessment.log', staging, 'private-assessment')
        if stopping():
            status('stopped')
            return
        assessment = read_json(RUN/'assessment/summary.json')
        expected = read_json(EXPECTED)
        development = {p.stem: read_json(p) for p in sorted((RUN/'evaluations').glob('eval_*.json'))}
        differences = compare(expected['development'], development, 'development')
        for field in ('candidate', 'paired', 'fresh'):
            compare(expected['assessment'][field], assessment[field], field, differences)
        comparison = dict(matches_original_metrics=not differences, float_tolerance=1e-6,
                          differences=differences, checkpoint=assessment['checkpoint'], preserved=True)
        atomic_json(REPORT/'reproduction_comparison.json', comparison)
        if benchmarks:
            from sft.lora_checkpoint import resolve
            # Benchmark the fixed final update, regardless of the private selection gate.
            final = resolve(RUN/'latest.json')
            for name, bundle in [('replica_benchmarks', final), ('parent_benchmarks', original.PARENT)]:
                destination = REPORT/name
                if not (destination/'summary.json').exists():
                    worker([str(ROOT/'pretraining/scripts/chat_adapter_search_bench.py'),
                            str(bundle), str(staging), str(destination)],
                           REPORT/(name+'.log'), staging, name)
                if stopping():
                    status('stopped')
                    return
            public = original.public_gate(read_json(REPORT/'parent_benchmarks/summary.json'),
                                          read_json(REPORT/'replica_benchmarks/summary.json'))
            atomic_json(REPORT/'public_comparison.json', public)
        original.protected()
        atomic_json(REPORT/'complete.json', dict(status='complete-preserved', **comparison,
                    public_benchmarks_complete=benchmarks))
        status('complete-preserved', checkpoint=assessment['checkpoint'],
               matches_original_metrics=not differences, public_benchmarks_complete=benchmarks)


def request_stop(*_):
    global STOP
    STOP = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--benchmarks', action='store_true')
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        run(args.prepare_only, args.benchmarks)
    except Exception as error:
        status('error-retained', error=repr(error))
        raise
