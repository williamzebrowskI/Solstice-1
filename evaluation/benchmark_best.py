#!/usr/bin/env python3
"""Prepare/run a frozen zero-shot multiple-choice benchmark (CPU by default).

Task prompts and character-normalized accuracy follow EleutherAI's task
definitions. This is a native MLX runner, not an lm-eval-harness execution.
No training imports, optimizer loads or distributed initialization. An explicit
device='gpu' configuration runs locally on the GPU after training has stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pretraining'))
DEFAULT_CONFIG = ROOT / 'configs/benchmark_best_low_impact.json'
from sft.metadata import sft_info


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_tokenizer(metadata):
    info=sft_info(metadata)
    if info is not None:
        return info['contract']['tokenizer']
    return metadata['v2_contract']['tokenizer']


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_gib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**3 if sys.platform == 'darwin' else peak / 1024**2


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_process(cache):
    # Process-local settings; do not change the training runtime or its cache.
    for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_CACHE'] = str(cache / 'hub')
    os.environ['HF_DATASETS_CACHE'] = str(cache / 'datasets')
    if hasattr(os, 'setpriority'):
        os.setpriority(os.PRIO_PROCESS, 0, 19)


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'checkpoint'
    if not frozen.exists():
        staging = output / ('.checkpoint-' + uuid.uuid4().hex)
        staging.mkdir()
        try:
            source_run = local_path(config['run_dir'])
            name = read_json(source_run / 'best.json')['bundle']
            if not re.fullmatch(r'step_\d+_[a-f0-9]+', name):
                raise ValueError('Invalid best checkpoint name')
            source = source_run / name
            if source.is_symlink():
                raise ValueError('Checkpoint cannot be a symlink')
            manifest = read_json(source / 'manifest.json')
            records = {}
            for filename in ('model.safetensors', 'model.safetensors.json'):
                origin = source / filename
                record = manifest['files'][filename]
                if origin.is_symlink():
                    raise ValueError('Checkpoint files cannot be symlinks')
                target = staging / filename
                shutil.copyfile(origin, target)
                if target.stat().st_size != record['bytes'] or sha256(target) != record['sha256']:
                    raise ValueError('Checkpoint changed or failed its checksum; retry preparation')
                records[filename] = record
            shutil.copytree(local_path(config['tokenizer_dir']), staging / 'tokenizer')
            from sml_v1.tokenization import Tokenizer
            tokenizer = Tokenizer(staging / 'tokenizer')
            metadata = read_json(staging / 'model.safetensors.json')
            info=sft_info(metadata)
            if tokenizer.fingerprint != checkpoint_tokenizer(metadata):
                raise ValueError('Checkpoint/tokenizer fingerprints do not match')
            if tokenizer.vocab_size != metadata['recipe']['model']['vocab_size']:
                raise ValueError('Checkpoint/tokenizer vocabularies do not match')
            atomic_json(staging / 'manifest.json', {
                'format': 'sml-benchmark-snapshot-v1', 'source_bundle': name,
                'step': metadata['step'], 'tokens': metadata['tokens'],
                'best_val_loss': metadata.get('best_val_loss'), 'files': records,
                **({'phase': 'SFT', 'parent_step': info['parent_step'],
                    'loss_scope': 'SFT uses assistant NLL; pre-training val_loss is evaluated separately.'} if info is not None else {}),
            })
            staging.rename(frozen)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    saved_config = output / 'config.json'
    if saved_config.exists() and read_json(saved_config) != config:
        raise ValueError('Prepared configuration differs; use a fresh output directory')
    atomic_json(saved_config, config)
    # Load/download public data before loading any model; no extra packages needed.
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    from datasets import DownloadConfig, load_dataset, load_from_disk
    for task, spec in config['tasks'].items():
        target = output / 'datasets' / task
        if target.exists():
            dataset = load_from_disk(str(target))
        else:
            print(f'[prepare] {task} · {spec["split"]} · {spec["revision"]}', flush=True)
            dataset = load_dataset(spec['path'], spec['name'], split=spec['split'],
                                   revision=spec['revision'],
                                   download_config=DownloadConfig(num_proc=1, max_retries=3))
            if len(dataset) != spec['expected_examples']:
                raise ValueError(f'{task}: unexpected example count {len(dataset)}')
            staging = target.with_name('.' + task + '-' + uuid.uuid4().hex)
            dataset.save_to_disk(str(staging))
            staging.rename(target)
        if len(dataset) != spec['expected_examples']:
            raise ValueError(f'{task}: prepared dataset is incomplete')
    print('[prepared] ' + str(frozen), flush=True)
    print(json.dumps(read_json(frozen / 'manifest.json'), indent=2), flush=True)


def example(task, doc):
    if task.startswith('arc_'):
        return ('Question: ' + doc['question'] + '\nAnswer:',
                doc['choices']['text'], doc['choices']['label'].index(doc['answerKey']))
    if task == 'piqa':
        return ('Question: ' + doc['goal'] + '\nAnswer:',
                [doc['sol1'], doc['sol2']], int(doc['label']))
    if task == 'hellaswag':
        def clean(text):
            text = text.strip().replace(' [title]', '. ')
            return re.sub(r'\[.*?\]', '', text).replace('  ', ' ')
        context = doc['activity_label'] + ': ' + doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize()
        return clean(context), [clean(text) for text in doc['endings']], int(doc['label'])
    raise ValueError('Unknown task: ' + task)


def pair_tokens(tokenizer, context, choice, max_length):
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + ' ' + choice
        context = context[:-trailing]
    else:
        continuation = ' ' + choice
    prefix = tokenizer.encode(context)
    joint = tokenizer.encode(context + continuation)
    if not prefix or joint[:len(prefix)] != prefix:
        raise ValueError('Tokenizer boundary mismatch; do not silently mis-score an answer')
    answer = joint[len(prefix):]
    if not answer or len(answer) > max_length:
        raise ValueError('Answer cannot be scored within the checkpoint context window')
    # At most max_length input tokens + the final token being predicted.
    removed = max(0, len(joint) - (max_length + 1))
    return joint[removed:], len(prefix) - removed, removed


def predictions(scores, choices):
    if any(not math.isfinite(score) for score in scores) or any(not text for text in choices):
        raise ValueError('Non-finite scores or empty choices')
    raw = max(range(len(scores)), key=lambda i: scores[i])
    normalized = max(range(len(scores)), key=lambda i: scores[i] / len(choices[i]))
    return raw, normalized


def training_records(run):
    paths = list((run / 'jobs').glob('*/train.log'))
    if not paths:
        return []
    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    with path.open('rb') as file:
        file.seek(max(0, path.stat().st_size - 128 * 1024))
        text = file.read().decode('utf-8', errors='replace')
    return [(int(step), float(rate.replace(',', '')))
            for step, rate in re.findall(r'\[step\s+(\d+)\].*?\stok/s=([\d,\.]+)', text)
            if float(rate.replace(',', '')) > 0]


class TrainingGuard:
    def __init__(self, config):
        self.run = local_path(config['run_dir'])
        self.ratio = config['throughput_floor_ratio']
        records = training_records(self.run)
        self.baseline = statistics.median(rate for _, rate in records[-20:]) if records else None
        self.initial_step = self.last_step = records[-1][0] if records else None
        self.last_advance = time.monotonic()
        self.last_check = 0.0
        self.low_checks = 0
        self.checked_step = None

    def check(self):
        now = time.monotonic()
        if now - self.last_check < 10:
            return
        self.last_check = now
        records = training_records(self.run)
        if not records:
            return
        step = records[-1][0]
        if self.last_step is None or step > self.last_step:
            self.last_step = step
            self.last_advance = now
        elif now - self.last_advance > 180:
            raise RuntimeError('Benchmark stopped: training has not advanced for 3 minutes; inspect its logs')
        fresh = [rate for s, rate in records if self.initial_step is not None and s > self.initial_step]
        if self.baseline is not None and len(fresh) >= 6 and step != self.checked_step:
            self.checked_step = step
            recent = statistics.median(fresh[-6:])
            self.low_checks = self.low_checks + 1 if recent < self.baseline * self.ratio else 0
            if self.low_checks >= 2:
                raise RuntimeError(f'Benchmark stopped: recent training throughput {recent:,.0f} '
                                   f'is below {self.ratio:.0%} of baseline {self.baseline:,.0f}')


class Scorer:
    def __init__(self, output, config, guard):
        import mlx.core as mx
        device=config.get('device','cpu')
        if device not in ('cpu','gpu'):raise ValueError('Benchmark device must be cpu or gpu')
        mx.set_default_device(mx.gpu if device=='gpu' else mx.cpu)
        mx.set_cache_limit(32 * 1024**2)
        mx.set_memory_limit(int(config['rss_stop_gib'] * 1024**3))  # MLX guideline, not an OS cap.
        from sml_v1.model import TransformerConfig, TransformerLM
        from sml_v1.inference import _cast_model_floats
        from sml_v1.tokenization import Tokenizer
        frozen = output / 'checkpoint'
        manifest = read_json(frozen / 'manifest.json')
        for name, record in manifest['files'].items():
            path = frozen / name
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise ValueError('Frozen checkpoint integrity check failed: ' + name)
        metadata = read_json(frozen / 'model.safetensors.json')
        model_config = dict(metadata['recipe']['model'])
        # Same weights and equations, CPU-supported reference operations.
        model_config.update(attention_impl='vanilla', ce_impl='reference', ffn_impl='reference')
        self.model = TransformerLM(TransformerConfig(**model_config))
        precision = mx.float32 if metadata.get('compute_precision') == 'float32' else mx.bfloat16
        _cast_model_floats(self.model, precision)
        self.model.load_weights(str(frozen / 'model.safetensors'), strict=True)
        from mlx.utils import tree_flatten
        if any(value.dtype != precision for _, value in tree_flatten(self.model.parameters())):
            raise ValueError('Saved weight dtype differs from declared inference precision')
        self.model.eval()
        mx.eval(self.model.parameters())
        self.tokenizer = Tokenizer(frozen / 'tokenizer')
        if self.tokenizer.fingerprint != checkpoint_tokenizer(metadata):
            raise ValueError('Frozen tokenizer integrity mismatch')
        self.mx, self.config, self.guard = mx, config, guard
        self.max_length = model_config['max_seq_len']
        self.check_memory()
        print(f'[model] step={metadata["step"]} tokens={metadata["tokens"]:,} '
              f'device={mx.default_device()} batch={config.get("batch_size", 1)} '
              f'chunk={config["chunk_tokens"]}', flush=True)

    def check_memory(self):
        if peak_rss_gib() > self.config['rss_stop_gib']:
            raise RuntimeError('Benchmark stopped: process peak RSS exceeded its configured budget')

    def score(self, context, choice):
        return self.score_choices(context, [choice])[0]

    def score_choices(self, context, choices):
        """Score right-padded answer sequences together, excluding all padding.

        Causal attention prevents trailing padding from affecting real tokens.
        Each row retains its own continuation boundary and truncation count.
        """
        if not choices:
            return []
        mx = self.mx
        pairs = [pair_tokens(self.tokenizer, context, choice, self.max_length) for choice in choices]
        totals, caches = [0.0] * len(pairs), None
        input_length = max(len(tokens) - 1 for tokens, _, _ in pairs)
        chunk = self.config['chunk_tokens']
        try:
            for start in range(0, input_length, chunk):
                self.guard.check()
                stop = min(start + chunk, input_length)
                rows = []
                for tokens, _, _ in pairs:
                    row = tokens[:-1][start:stop]
                    rows.append(row + [0] * (stop - start - len(row)))
                ids = mx.array(rows, dtype=mx.int32)
                logits, next_caches = self.model.logits(ids, caches=caches)
                scores = []
                for index, (tokens, prefix_length, _) in enumerate(pairs):
                    first = max(start, prefix_length - 1)
                    end = min(stop, len(tokens) - 1)
                    if first >= end:
                        continue
                    selected = logits[index, first - start:end - start, :].astype(mx.float32)
                    targets = mx.array(tokens[first + 1:end + 1], dtype=mx.int32)
                    target_logits = mx.take_along_axis(selected, targets[:, None], axis=-1)[:, 0]
                    score = (target_logits - mx.logsumexp(selected, axis=-1)).sum()
                    scores.append((index, score))
                    del selected, targets, target_logits, score
                mx.eval(next_caches, [score for _, score in scores])
                for index, score in scores:
                    totals[index] += float(score.item())
                del scores
                caches = next_caches
                del logits, ids, next_caches
                self.check_memory()
            if any(not math.isfinite(total) for total in totals):
                raise ValueError('Non-finite likelihood')
            return [(total, pair[2]) for total, pair in zip(totals, pairs)]
        finally:
            caches = None
            mx.clear_cache()
            time.sleep(self.config['pause_seconds'])


def summarize(output, config, limit, status):
    results = {}
    for task, spec in config['tasks'].items():
        path = output / f'{task}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        selected = [row for row in rows if limit == 0 or row['index'] < limit]
        count = len(selected)
        intended = min(limit or spec['expected_examples'], spec['expected_examples'])
        results[task] = {
            'evaluated': count, 'expected': intended, 'complete': count == intended,
            'acc': sum(row['correct'] for row in selected) / count if count else None,
            'acc_norm': sum(row['correct_norm'] for row in selected) / count if count else None,
            'context_truncated_choices': sum(sum(n > 0 for n in row['truncated']) for row in selected),
        }
    report = {
        'format': 'sml-choice-results-v1', 'status': status,
        'checkpoint': read_json(output / 'checkpoint/manifest.json'),
        'settings': {'device': config.get('device','cpu'), 'batch_size': config.get('batch_size', 1),
                     'chunk_tokens': config['chunk_tokens'],
                     'limit_per_task': limit, 'num_fewshot': 0, 'chat_template': False,
                     'scoring': 'sum continuation log-likelihood; acc_norm divides by choice characters',
                     'implementation': 'native MLX; lm-eval-style task prompts; not harness-validated'},
        'datasets': config['tasks'], 'results': results,
    }
    metadata_path = output / 'checkpoint/model.safetensors.json'
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        report['settings'].update(
            compute_precision=metadata.get('compute_precision', 'bfloat16'),
            weight_provenance=metadata.get('inference_export'),
            mlx_enable_tf32=os.environ.get('MLX_ENABLE_TF32', 'default'))
    atomic_json(output / 'results.json', report)
    return report


def run(config, output, limit):
    if not (output / 'config.json').exists():
        raise ValueError('Run --prepare first; it freezes the best checkpoint and downloads data')
    if read_json(output / 'config.json') != config:
        raise ValueError('Settings differ from the frozen run configuration')
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    from datasets import load_from_disk
    guard = TrainingGuard(config)
    print(f'[guard] training baseline={guard.baseline}; stop below '
          f'{config["throughput_floor_ratio"]:.0%}; RSS stop={config["rss_stop_gib"]} GiB', flush=True)
    started = time.monotonic()
    status = 'running'
    try:
        scorer = Scorer(output, config, guard)
        for task in config['tasks']:
            dataset = load_from_disk(str(output / 'datasets' / task))
            count = min(limit or len(dataset), len(dataset))
            path = output / f'{task}.jsonl'
            done = {json.loads(line)['index'] for line in path.read_text().splitlines()} if path.exists() else set()
            with path.open('a') as log:
                for index in range(count):
                    if index in done:
                        continue
                    context, choices, gold = example(task, dataset[index])
                    scores, truncated = [], []
                    batch_size = config.get('batch_size', 1)
                    for start in range(0, len(choices), batch_size):
                        for score, removed in scorer.score_choices(context, choices[start:start + batch_size]):
                            scores.append(score)
                            truncated.append(removed)
                    raw, normalized = predictions(scores, choices)
                    log.write(json.dumps({'index': index, 'scores': scores, 'gold': gold,
                                          'prediction': raw, 'prediction_norm': normalized,
                                          'correct': raw == gold, 'correct_norm': normalized == gold,
                                          'truncated': truncated}, allow_nan=False) + '\n')
                    log.flush()
                    if (index + 1) % 10 == 0 or index + 1 == count:
                        summarize(output, config, limit, 'running')
                        print(f'[{task}] {index + 1}/{count} elapsed={time.monotonic() - started:.1f}s', flush=True)
            summarize(output, config, limit, 'running')
        status = 'complete'
    except (KeyboardInterrupt, Exception) as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'stopped'
        print(f'[{status}] {error or "Ctrl+C"}; completed examples are saved. Re-run to resume.', flush=True)
        if not isinstance(error, KeyboardInterrupt):
            raise
    finally:
        report = summarize(output, config, limit, status)
        report['runtime'] = {'elapsed_seconds': time.monotonic() - started,
                             'peak_rss_gib': peak_rss_gib(),
                             'training_baseline_tokens_per_second': guard.baseline,
                             'last_observed_training_step': guard.last_step}
        atomic_json(output / 'results.json', report)
        print(json.dumps(report['results'], indent=2), flush=True)
        print(f'[memory] peak process RSS={peak_rss_gib():.3f} GiB', flush=True)
        print('[results] ' + str(output / 'results.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='First N examples per task; 0 runs full splits')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    config = read_json(args.config)
    if config.get('device','cpu') not in ('cpu','gpu'):parser.error('device must be cpu or gpu')
    output = local_path(config['output_dir'])
    if not 1 <= config['chunk_tokens'] <= 128:
        parser.error('Chunk size must be between 1 and 128 tokens')
    if type(config.get('batch_size', 1)) is not int or not 1 <= config.get('batch_size', 1) <= 8:
        parser.error('Batch size must be an integer between 1 and 8 answer sequences')
    configure_process(output.parent / 'cache')
    if args.prepare:
        prepare(config, output)
    else:
        # Prevent two evaluators appending to the same result logs.
        import fcntl
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.run.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(config, output, args.limit)


if __name__ == '__main__':
    main()
