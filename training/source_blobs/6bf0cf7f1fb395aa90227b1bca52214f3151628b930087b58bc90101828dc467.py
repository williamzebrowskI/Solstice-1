"""Matched reserved-test assessment of base, retained 167 and LoRA exports."""
import argparse
from pathlib import Path
import mlx.core as mx
from sml_v1.common import read_json, atomic_json, file_sha256
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.tokenization import Tokenizer
from .text_corpus import verify
from .text_eval import evaluate, reference_inference
from .lora_checkpoint import resolve, verify as verify_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.input / 'plan.json')
    contract = read_json(args.input / 'contract.json')
    verify(args.input / 'corpus', plan)
    tokenizer = Tokenizer(args.input / 'tokenizer')
    source = args.input / 'source'
    metadata = read_json(source / 'model.safetensors.json')
    masters = source / 'masters/model.safetensors'
    if file_sha256(masters) != contract['source_masters']:
        raise ValueError('FP32 base changed')
    for name, digest in contract['comparison_files'].items():
        if file_sha256(args.input / 'comparison' / name) != digest:
            raise ValueError('Retained reference changed')
    candidates = [('parent', masters, 0), ('reference_167', args.input / 'comparison/model.safetensors', 167)]
    selected = resolve(args.output / 'best.json')
    latest = resolve(args.output / 'latest.json')
    for label, bundle in [('selected', selected), ('latest_unselected', latest)]:
        state = verify_checkpoint(bundle)
        if state['contract'] != contract:
            raise ValueError('Adapter assessment contract mismatch')
        if state['step'] and (label == 'selected' or latest != selected):
            candidates.append((label, bundle / 'inference/model.safetensors', state['step']))
    results = {}
    for label, weights, step in candidates:
        if label in ('parent', 'reference_167') and plan.get('baseline_cache'):
            from .baseline_cache import read as cached_baseline
            key = 'parent_test' if label == 'parent' else 'reference_test'
            results[label] = cached_baseline(args.input, key, args.output/'assessment'/label/'final_test'/f'eval_{step:07d}.json')
            continue
        model = TransformerLM(TransformerConfig(**metadata['recipe']['model']))
        model.load_weights(str(weights), strict=True)
        mx.eval(model.parameters())
        with reference_inference(model):
            results[label] = evaluate(model, tokenizer, args.input / 'corpus', plan, step,
                                      args.output / 'assessment' / label, args.input / 'retention', split='test',
                                      precision_label='fp32-base-plus-lora-and-token-deltas' if label in ('selected','latest_unselected') else 'exact-fp32-masters')
        del model
        mx.clear_cache()
    atomic_json(args.output / 'assessment/summary.json', dict(results=results, selected=str(selected),
                status='candidate-needs-benchmark-review' if 'selected' in results else 'no-eligible-improvement',
                note='Reserved grouped test set; reused historical cases. Does not replace full public benchmarks.'))
    print('[assessment] ' + str(args.output / 'assessment/summary.json'), flush=True)


if __name__ == '__main__':
    main()
