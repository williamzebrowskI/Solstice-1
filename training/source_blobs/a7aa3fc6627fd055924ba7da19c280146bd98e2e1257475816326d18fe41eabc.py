"""Pinned parent-2000 experiment inputs and comparison rules."""
from pathlib import Path
import json
import os
import hashlib
import mlx.core as mx
from sml_v1.common import read_json, file_sha256, fingerprint
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.precision import MasterAdamW
from .lora_adapter import attach, base_digest, assert_frozen
from .lora_stage2 import initialize

# Runtime is copied into run/input/sft as well as shared/runtime/sft. Use a
# project-relative anchor, never a checkpoint supplied by model-generated text.
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pretraining/scripts').is_dir())
SHARED = ROOT/'pretraining/prepared/parent2000_generalize_20261109'
REPORT = ROOT/'reports/sft_search/parent2000_generalize_20261109'
PARENT_RUN = ROOT/'pretraining/runs/deep44_adapter_trial_09_replica_20261108'
PARENT = PARENT_RUN/'step_0002000_f3d1ffd185f9'
ASSETS = PARENT_RUN/'input'

def modules(path):
    return {str(p.relative_to(path)):file_sha256(p) for package in ('sft','sml_v1')
            for p in sorted((path/package).rglob('*.py'))}

def compact(value):
    return {k:v for k,v in value.items() if k!='samples'}

def correct(value):
    return sum(value['generated_pass_counts'].values())

def gate(result, baseline, *, split):
    """Predeclared practical thresholds, not a claim of statistical significance."""
    reasons=[]
    gain=correct(result)-correct(baseline)
    minimum=4 if split=='dev' else 8
    if gain<minimum:reasons.append(f'accuracy gain {gain} below {minimum}')
    for name,count in baseline['generated_pass_counts'].items():
        if result['generated_pass_counts'][name]<count:reasons.append(name+' accuracy regression')
    for name in ('natural_stop',):
        if result[name]+1e-12<baseline[name]:reasons.append(name+' regression')
    for name in ('repetition_rate','empty_rate','replacement_character_rate'):
        if result.get(name,0)>baseline.get(name,0)+1e-12:reasons.append(name+' regression')
    cap=min(baseline['retention']['val_loss']+.003,2.8353193145245315+.03)
    if result['retention']['val_loss']>cap:reasons.append('prose retention')
    return dict(passed=not reasons,reasons=reasons,gain=gain,minimum_gain=minimum,retention_cap=cap)

def severe(result,baseline):
    return (correct(result)<correct(baseline)-8 or
            result['natural_stop']<baseline['natural_stop']-.02 or
            result['repetition_rate']>baseline['repetition_rate']+.02 or
            result['retention']['val_loss']>min(baseline['retention']['val_loss']+.003,2.8653193145245315))

def setup(staging,plan,contract):
    staging=Path(staging)
    if contract['plan']!=fingerprint(plan) or contract['modules']!=modules(staging):
        raise ValueError('Frozen plan/runtime changed')
    for name,key in [('source/model.safetensors','source_weights'),('source/model.safetensors.json','source_metadata'),
                     ('source/masters/model.safetensors','source_masters'),('retention/manifest.json','retention')]:
        if file_sha256(staging/name)!=contract[key]:raise ValueError('Input changed: '+name)
    metadata=read_json(staging/'source/model.safetensors.json')
    config=metadata['recipe']['model'];model=TransformerLM(TransformerConfig(**config))
    original=mx.load(str(staging/'source/model.safetensors'))
    base=mx.load(str(staging/'source/masters/model.safetensors'))
    if set(base)!=set(original) or any(v.dtype!=mx.float32 or v.shape!=original[n].shape for n,v in base.items()):
        raise ValueError('Invalid base masters')
    if not all(bool(mx.all(v.astype(original[n].dtype)==original[n]).item()) for n,v in base.items()):
        raise ValueError('Base master round trip differs')
    model.load_weights(list(original.items()),strict=True);digest=base_digest(original)
    count=attach(model,plan['adaptation'])
    optimizer=MasterAdamW(learning_rate=plan['peak_lr'],betas=tuple(plan['betas']),eps=plan['eps'],weight_decay=0.)
    ancestry=initialize(model,optimizer,base,staging/'initial_parent',plan['adaptation'],contract)
    assert_frozen(model,digest)
    return model,optimizer,base,config,metadata,digest,count,ancestry

def data_receipt():
    data=SHARED/'fresh_data';receipt=read_json(data/'manifest.json')
    if receipt['format']!='parent2000-fresh-conversations-v1':raise ValueError('Unexpected fresh-data format')
    if receipt['parent_manifest']!=file_sha256(PARENT/'manifest.json'):raise ValueError('Fresh-data parent changed')
    for name,h in receipt['files'].items():
        if file_sha256(data/name)!=h:raise ValueError('Fresh data changed: '+name)
    return receipt

def load_baseline(staging,split):
    from .baseline_cache import identity
    receipt=read_json(SHARED/'baseline/manifest.json')
    if identity(staging)!=receipt['identity']:raise ValueError('Evaluation differs from cached parent')
    if file_sha256(PARENT/'manifest.json')!=receipt['parent_manifest']:raise ValueError('Cached parent changed')
    path=SHARED/'baseline'/(split+'.json')
    if file_sha256(path)!=receipt['files'][path.name]:raise ValueError('Baseline changed')
    return read_json(path)
