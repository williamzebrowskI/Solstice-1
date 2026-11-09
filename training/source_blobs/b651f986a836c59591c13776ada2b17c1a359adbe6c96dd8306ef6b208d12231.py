#!/usr/bin/env python3
"""Preserve exact master parameters while leaving the source bundle unchanged."""
import argparse
from pathlib import Path
import sys
import uuid
import shutil
from sml_v1.common import read_json,atomic_json,file_sha256


def preserve(source,dest):
    source=Path(source);dest=Path(dest);manifest=read_json(source/'manifest.json')
    names=('model.safetensors','model.safetensors.json','model.safetensors.optimizer.safetensors')
    for name in names:
        p=source/name;r=manifest['files'][name]
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or file_sha256(p)!=r['sha256']:raise ValueError('Source checksum mismatch: '+name)
    if dest.exists():
        receipt=read_json(dest/'manifest.json')
        if receipt['source_files']!={n:manifest['files'][n] for n in names} or file_sha256(dest/'model.safetensors')!=receipt['sha256']:
            raise ValueError('Preserved masters have a different source')
        return receipt
    import mlx.core as mx
    original=mx.load(str(source/names[0]));state=mx.load(str(source/names[2]))
    masters={n[:-7]:v for n,v in state.items() if n.endswith('.master')}
    if set(masters)!=set(original):raise ValueError('Master names differ')
    for n,v in masters.items():
        if v.dtype!=mx.float32 or v.shape!=original[n].shape or not bool(mx.all(v.astype(original[n].dtype)==original[n]).item()):
            raise ValueError('Master dtype/shape/rounding differs: '+n)
    stage=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True)
    try:
        target=stage/'model.safetensors';mx.save_safetensors(str(target),masters)
        reload=mx.load(str(target))
        if set(reload)!=set(masters) or not all(bool(mx.all(reload[n]==v).item()) for n,v in masters.items()):raise ValueError('Master export not exact')
        receipt=dict(format='opensml-preserved-fp32-masters-v1',source_bundle=str(source),source_files={n:manifest['files'][n] for n in names},
            sha256=file_sha256(target),rounds_to_original_bf16=True,export_round_trip_exact=True,parameters=sum(v.size for v in masters.values()))
        atomic_json(stage/'manifest.json',receipt)
        for name in names:
            if file_sha256(source/name)!=manifest['files'][name]['sha256']:raise ValueError('Source changed during export')
        stage.rename(dest);return receipt
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(preserve(a.source,a.output))
