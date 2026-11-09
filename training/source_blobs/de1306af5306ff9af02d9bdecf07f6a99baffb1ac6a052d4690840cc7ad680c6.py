"""Transactional adapter resume bundles with separate dense inference exports."""
import os
from pathlib import Path
import shutil
import time
import uuid
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from sml_v1.common import atomic_json, read_json, file_sha256
from .lora_adapter import merged_weights

FORMAT = 'opensml-lora-sft-bundle-v1'


def resolve(pointer):
    pointer = Path(pointer)
    if pointer.is_symlink():
        raise ValueError('Adapter pointer cannot be a symlink')
    value = read_json(pointer)
    name = value.get('bundle', '')
    if value.get('format') != FORMAT or Path(name).name != name or not name.startswith('step_'):
        raise ValueError('Invalid adapter checkpoint pointer')
    return pointer.parent / name


def verify(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Adapter checkpoint cannot be a symlink')
    manifest = read_json(path / 'manifest.json')
    required = {'adapters.safetensors', 'optimizer.safetensors', 'state.json',
                'inference/model.safetensors', 'inference/model.safetensors.json', 'inference/manifest.json'}
    if manifest.get('format') != FORMAT or not required <= set(manifest['files']):
        raise ValueError('Incomplete adapter checkpoint')
    for name, record in manifest['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or len(p.parts) > 2:
            raise ValueError('Unsafe adapter checkpoint filename')
        target = path / p
        if target.is_symlink() or target.parent.is_symlink() or target.stat().st_size != record['bytes'] or file_sha256(target) != record['sha256']:
            raise ValueError('Adapter checkpoint checksum differs: ' + name)
    state = read_json(path / 'state.json')
    if state['step'] != manifest['step']:
        raise ValueError('Adapter checkpoint step mismatch')
    return state


def restore(path, model, optimizer, contract, order_hash):
    path = Path(path)
    state = verify(path)
    if state['contract'] != contract or state['order_hash'] != order_hash:
        raise ValueError('Adapter resume contract or batch order changed')
    if state['quality_stop']:
        raise ValueError('Do not resume a quality-stopped adapter')
    values = mx.load(str(path / 'adapters.safetensors'))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(values) != set(expected) or any(values[n].dtype != mx.float32 or values[n].shape != v.shape for n, v in expected.items()):
        raise ValueError('Adapter tensor names, shapes or precision changed')
    model.update(tree_unflatten(list(values.items())))
    optimizer.state = tree_unflatten(list(mx.load(str(path / 'optimizer.safetensors')).items()))
    mx.eval(model.parameters(), optimizer.state)
    masters = {n[:-7]: v for n, v in tree_flatten(optimizer.state) if n.endswith('.master')}
    if set(masters) != set(values) or not all(bool(mx.all(values[n] == v).item()) for n, v in masters.items()):
        raise ValueError('Adapter optimizer masters differ from adapter weights')
    return state


def save(output, model, optimizer, base, config, state, *, best=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 3 * 2**30:
        raise OSError('Less than 3 GiB free; previous checkpoints preserved')
    name = f'step_{state["step"]:07d}_{uuid.uuid4().hex[:12]}'
    stage = output / ('.' + name + '.tmp')
    final = output / name
    stage.mkdir()
    try:
        mx.save_safetensors(str(stage / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
        mx.save_safetensors(str(stage / 'optimizer.safetensors'), dict(tree_flatten(optimizer.state)))
        atomic_json(stage / 'state.json', state)
        export = stage / 'inference'
        export.mkdir()
        mx.save_safetensors(str(export / 'model.safetensors'), merged_weights(model, optimizer, base))
        metadata = dict(training_kind='sft', sft_method='lora+trainable-tokens', step=state['step'],
                        tokens=state['tokens'], assistant_tokens=state['assistant_tokens'],
                        pretraining_step=state['source_step'], pretraining_tokens=state['source_tokens'],
                        source_step=state['source_step'], source_tokens=state['source_tokens'],
                        validation_fingerprint=state['validation_fingerprint'], sft_stage=1,
                        sft_contract=state['contract'], compute_precision='float32',
                        recipe=dict(model=config, adaptation=state['adaptation']),
                        inference_export=dict(precision='FP32 pretrained masters plus FP32 LoRA and tied chat-token deltas',
                                              trainable_parameters=state['trainable_parameters'],
                                              resume='Use the enclosing adapter bundle; this is an inference export'))
        atomic_json(export / 'model.safetensors.json', metadata)
        def records(directory):
            return {str(p.relative_to(directory)): dict(bytes=p.stat().st_size, sha256=file_sha256(p))
                    for p in sorted(directory.rglob('*')) if p.is_file()}
        atomic_json(export / 'manifest.json', dict(format='sml-pretrain-bundle-v1', step=state['step'], files=records(export)))
        atomic_json(stage / 'manifest.json', dict(format=FORMAT, step=state['step'], created_ns=time.time_ns(), files=records(stage)))
        for p in stage.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:
                    os.fsync(f.fileno())
        verify(stage)
        stage.rename(final)
        pointer = dict(format=FORMAT, bundle=name)
        if best:
            atomic_json(output / 'best.json', pointer)
        atomic_json(output / 'latest.json', pointer)
        return final
    finally:
        if stage.exists():
            shutil.rmtree(stage)
