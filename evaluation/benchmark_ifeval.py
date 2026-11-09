#!/usr/bin/env python3
"""Generate SFT answers and score with the existing pinned Google IFEval code."""
import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'reports/benchmarks/ifeval-runtime-20261008'
sys.path[:0] = [str(ROOT/'pretraining'), str(RUNTIME/'site-packages'), str(RUNTIME)]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def persist_rows(path, rows):
    temp = path.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    temp.replace(path)


def resolve_weights(path):
    """Accept a verified inference snapshot as well as a training bundle."""
    directory = Path(path).absolute()
    if directory.name == 'model.safetensors':
        directory = directory.parent
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest.get('format') == 'sml-benchmark-snapshot-v1':
            required = {'model.safetensors', 'model.safetensors.json'}
            if directory.is_symlink() or not required <= manifest['files'].keys():
                raise ValueError('Incomplete inference snapshot')
            for name, record in manifest['files'].items():
                if Path(name).name != name:
                    raise ValueError('Unsafe inference snapshot filename')
                file = directory/name
                if file.is_symlink() or file.stat().st_size != record['bytes'] or digest(file) != record['sha256']:
                    raise ValueError('Inference snapshot integrity check failed: '+name)
            return str(directory/'model.safetensors')
    from sml_v1.checkpoint_bundle import resolve_bundle
    return resolve_bundle(path)


def score(inputs, records):
    from instruction_following_eval import evaluation_lib as ev
    outputs = {'strict': [], 'loose': []}
    for inp, record in zip(inputs, records):
        example = ev.InputExample(**inp)
        response = {inp['prompt']: record['response']}
        for kind in outputs:
            random.seed(2026)
            check = getattr(ev, 'test_instruction_following_'+kind)
            outputs[kind].append(dataclasses.asdict(check(example, response)))
    metrics = {}
    for kind, rows in outputs.items():
        count = sum(len(row['follow_instruction_list']) for row in rows)
        metrics[kind] = {
            'prompt_accuracy': sum(row['follow_all_instructions'] for row in rows)/len(rows) if rows else None,
            'instruction_accuracy': sum(sum(row['follow_instruction_list']) for row in rows)/count if count else None,
            'prompts': len(rows), 'instructions': count,
        }
    return metrics, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=0, help='Smoke/subset only; 0 runs all 541 prompts')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.max_tokens <= 2048:
        parser.error('limit must be nonnegative; max-tokens must be 1–2048')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['NLTK_DATA'] = str(RUNTIME/'nltk_data')
    with (args.output/'.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from sml_v1.tokenization import Tokenizer
        weights = resolve_weights(args.checkpoint)
        metadata = read(weights+'.json')
        precision = metadata.get('compute_precision', 'bfloat16')
        if precision not in ('float32', 'bfloat16'):
            raise ValueError('Unsupported inference precision: '+str(precision))
        model_config = dict(metadata['recipe']['model'])
        if precision == 'float32':
            model_config.update(attention_impl='vanilla', ce_impl='reference', ffn_impl='reference')
        from sft.metadata import sft_info
        info=sft_info(metadata)
        if info is None:
            raise ValueError('This runner requires an SFT chat checkpoint')
        tokenizer = Tokenizer(args.tokenizer)
        if tokenizer.fingerprint != info['contract']['tokenizer']:
            raise ValueError('Tokenizer mismatch')
        source = read(RUNTIME/'source.json')
        for name,expected in source.get('evaluator_blob_sha1',{}).items():
            raw=(RUNTIME/'instruction_following_eval'/name).read_bytes()
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=expected:
                raise ValueError('Pinned IFEval evaluator changed: '+name)
        identity = {
            'format': 'opensml-ifeval-v1', 'checkpoint': str(Path(weights).parent),
            'step': metadata['step'], 'model_sha256': digest(weights),
            'metadata_sha256': digest(weights+'.json'), 'tokenizer': tokenizer.fingerprint,
            'dataset': source['dataset'], 'dataset_revision': source['dataset_revision'],
            'google_research_commit': source['google_research_commit'],
            'evaluator_files': {p.name: digest(p) for p in sorted((RUNTIME/'instruction_following_eval').glob('*.py'))},
            'runner_sha256': digest(__file__), 'temperature': 0,
            'repetition_penalty': 1, 'max_tokens': args.max_tokens,
            'system_prompt': None, 'serialization': 'native SFT chat template',
            'prompt_truncation': False, 'context_limit': metadata['recipe']['model']['max_seq_len'],
            'compute_precision': precision, 'model_implementation': model_config,
            'weight_provenance': metadata.get('inference_export'),
            'mlx_enable_tf32': os.environ.get('MLX_ENABLE_TF32', 'default'),
        }
        identity_path = args.output/'protocol.json'
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError('Protocol differs; choose a fresh output directory')
        write(identity_path, identity)
        inputs_path = args.output/'inputs.json'
        receipt_path = args.output/'inputs.sha256.json'
        if inputs_path.exists():
            if read(receipt_path)['sha256'] != digest(inputs_path):
                raise ValueError('Frozen IFEval inputs changed')
            inputs = read(inputs_path)
        else:
            from datasets import load_dataset
            dataset = load_dataset(source['dataset'], 'default', split='train', revision=source['dataset_revision'],
                                   cache_dir=str(ROOT/'pretraining/data/.sft_hf_cache'))
            inputs = [dict(key=int(row['key']), prompt=row['prompt'],
                           instruction_id_list=list(row['instruction_id_list']),
                           kwargs=[{k: v for k, v in item.items() if v is not None} for item in row['kwargs']])
                      for row in dataset]
            if len(inputs) != 541 or len({row['key'] for row in inputs}) != 541:
                raise ValueError('Unexpected IFEval prompt count or duplicate keys')
            write(inputs_path, inputs)
            write(receipt_path, {'sha256': digest(inputs_path), 'expected': 541})
        if len(inputs) != 541:
            raise ValueError('Frozen prompt count differs')
        import nltk
        from langdetect import DetectorFactory
        DetectorFactory.seed = 2026
        nltk.data.path.insert(0, str(RUNTIME/'nltk_data'))
        try:
            nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        except LookupError:
            for resource in ('punkt', 'punkt_tab'):
                if not nltk.download(resource, download_dir=str(RUNTIME/'nltk_data'), quiet=True):
                    raise RuntimeError('Could not prepare NLTK '+resource)
            nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        # Exercise the scorer on every prompt before loading the model.
        score(inputs, [{'response': ''} for _ in inputs])
        if args.prepare_only:
            print(f'[prepared] IFEval 541 prompts; SFT step {metadata["step"]}; {args.output}', flush=True)
            return
        responses_path = args.output/'responses.jsonl'
        records = [json.loads(line) for line in responses_path.read_text().splitlines()] if responses_path.exists() else []
        for inp, record in zip(inputs, records):
            if record['key'] != inp['key'] or record['prompt'] != inp['prompt']:
                raise ValueError('Saved responses differ from frozen prompt order')
        if len(records) > len(inputs):
            raise ValueError('Too many saved responses')
        selected = inputs[:args.limit] if args.limit else inputs
        if len(records) < len(selected):
            import mlx.core as mx
            from sml_v1.model import TransformerConfig, TransformerLM
            from sft.data import serialize
            from sft.evaluate import generate
            model = TransformerLM(TransformerConfig(**model_config))
            model.load_weights(weights)
            dtype = mx.float32 if precision == 'float32' else mx.bfloat16
            from mlx.utils import tree_flatten
            if any(value.dtype != dtype for _, value in tree_flatten(model.parameters())):
                raise ValueError('Saved weight dtype differs from declared inference precision')
            model.eval()
            mx.eval(model.parameters())
            print(f'[model] compute_precision={precision} attention={model_config["attention_impl"]}', flush=True)
            started = time.monotonic()
            for inp in selected[len(records):]:
                messages = [{'role': 'user', 'content': inp['prompt']}]
                ids, _ = serialize(tokenizer, messages, generation=True)
                available = model.cfg.max_seq_len-len(ids)
                if available < 1:
                    raise ValueError(f'Prompt {inp["key"]} exceeds context; no silent truncation')
                result = generate(model, tokenizer, messages, limit=args.max_tokens, temperature=0)
                records.append(dict(key=inp['key'], prompt=inp['prompt'], response=result['text'],
                                    tokens=result['tokens'], stop_reason=result['stop_reason'],
                                    prompt_tokens=len(ids), effective_max_tokens=min(args.max_tokens, available)))
                persist_rows(responses_path, records)
                if len(records)%10 == 0 or len(records)==len(selected):
                    print(f'[ifeval] {len(records)}/{len(selected)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        metrics, outputs = score(inputs[:len(records)], records)
        for kind, rows in outputs.items():
            persist_rows(args.output/f'{kind}.jsonl', rows)
        result = dict(protocol=identity, evaluated=len(records), expected=541,
                      complete=len(records)==541, metrics=metrics,
                      token_limited=sum(row['stop_reason']=='token_limit' for row in records),
                      context_limited=sum(row['effective_max_tokens']<args.max_tokens for row in records),
                      scope='Official pinned IFEval strict/loose checks; native SFT generation, no harness parity claim.')
        write(args.output/'results.json', result)
        print(json.dumps(result, indent=2))
        print('[results] '+str(args.output/'results.json'), flush=True)


if __name__ == '__main__':
    main()
