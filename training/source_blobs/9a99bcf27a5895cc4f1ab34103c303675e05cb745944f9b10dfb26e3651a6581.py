"""Reuse only identical frozen-base/reference measurements across adapter trials."""
from pathlib import Path
import shutil
from sml_v1.common import read_json, atomic_json, file_sha256


def identity(staging):
    staging = Path(staging)
    plan = read_json(staging / 'plan.json')
    contract = read_json(staging / 'contract.json')
    evaluation_keys = ('components', 'checked_components', 'eval_examples_per_component',
                       'generated_eval_per_component', 'qualitative_eval_per_component', 'answer_max_tokens')
    files = {}
    for split in ('dev', 'test'):
        for suffix in ('jsonl', 'tokens.npy', 'mask.npy', 'offsets.npy'):
            name = f'corpus/{split}.{suffix}'
            files[name] = file_sha256(staging / name)
    for name in ('retention/manifest.json', 'retention/panel.npz'):
        files[name] = file_sha256(staging / name)
    # Include all evaluator dependencies; adapter training itself is immaterial
    # to these measurements of the unchanged dense base and checkpoint 167.
    modules = {n: h for n, h in contract['modules'].items()
               if not n.startswith(('sft/lora_', 'sft/baseline_cache.'))}
    return dict(source_masters=contract['source_masters'], reference=contract['comparison_files'],
                tokenizer=contract['tokenizer'], files=files, modules=modules,
                evaluation={k:plan.get(k) for k in evaluation_keys})


def read(staging, key, destination):
    staging = Path(staging)
    plan = read_json(staging / 'plan.json')
    cache = Path(plan['baseline_cache'])
    receipt = read_json(cache / 'manifest.json')
    if receipt['identity'] != identity(staging):
        raise ValueError('Cached baseline input or evaluation protocol differs')
    source = cache / (key + '.json')
    if file_sha256(source) != receipt['files'][source.name]:
        raise ValueError('Cached baseline checksum differs')
    value = read_json(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, value)
    print('[baseline-cache] verified identical ' + key, flush=True)
    return {k:v for k,v in value.items() if k != 'samples'}
