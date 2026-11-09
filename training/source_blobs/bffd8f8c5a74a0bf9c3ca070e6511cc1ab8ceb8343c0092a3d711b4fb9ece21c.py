"""Publish an independent inference-only best snapshot at regular save points."""
import json
from pathlib import Path
import re
import shutil
import uuid

from .common import atomic_json, file_sha256, read_json


FILE_NAMES = ('model.safetensors', 'model.safetensors.json')


def publish_best(run, *, event_step, event_tokens):
    run = Path(run)
    pointer = read_json(run/'best.json')
    name = pointer.get('bundle', '')
    if not isinstance(name, str) or not re.fullmatch(r'step_\d+_[a-f0-9]+', name):
        raise ValueError('Invalid best checkpoint pointer')
    source = run/name
    if source.is_symlink():
        raise ValueError('Best checkpoint must not be a symlink')
    manifest = read_json(source/'manifest.json')
    target = run/'playground'; target.mkdir(parents=True, exist_ok=True)
    staging = target/('.publish_'+uuid.uuid4().hex)
    destination = target/name
    if not destination.exists():
        staging.mkdir()
        try:
            for filename in FILE_NAMES:
                origin = source/filename
                if origin.is_symlink():
                    raise ValueError('Checkpoint files must not be symlinks')
                record = manifest['files'][filename]
                copied = staging/filename
                shutil.copyfile(origin, copied)
                if copied.stat().st_size != record['bytes'] or file_sha256(copied) != record['sha256']:
                    raise ValueError('Playground copy failed checkpoint integrity checks')
            metadata = read_json(staging/'model.safetensors.json')
            if metadata['step'] != manifest['step'] or 'v2_contract' not in metadata:
                raise ValueError('Invalid pretrained checkpoint metadata')
            # Inference-only: no optimizer, data cursor, or training bundle claim.
            atomic_json(staging/'manifest.json', dict(format='sml-playground-snapshot-v1', step=manifest['step'],
                        files={n:manifest['files'][n] for n in FILE_NAMES}))
            staging.rename(destination)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    metadata = read_json(destination/'model.safetensors.json')
    atomic_json(target/'best.json', dict(format='sml-playground-pointer-v1', bundle=name,
                checkpoint_step=metadata['step'], checkpoint_tokens=metadata['tokens'],
                best_val_loss=metadata['best_val_loss'], event_step=event_step, event_tokens=event_tokens))
    # Preserve one prior snapshot for requests already opening it. The server
    # copies weights privately while loading and retries if this crosses cleanup.
    old = sorted((p for p in target.glob('step_*') if p.is_dir() and not p.is_symlink() and p != destination),
                 key=lambda p:p.stat().st_mtime_ns, reverse=True)
    for path in old[1:]: shutil.rmtree(path)
    return dict(bundle=str(destination), step=metadata['step'], tokens=metadata['tokens'])


def publish_safely(run, *, event_step, event_tokens):
    """A playground failure must not fail a committed training checkpoint."""
    try:
        result = publish_best(run, event_step=event_step, event_tokens=event_tokens)
        print(f'[playground] best pretrained step={result["step"]} tokens={result["tokens"]:,} published', flush=True)
        return result
    except Exception as exc:
        try:
            atomic_json(Path(run)/'playground/last_error.json', dict(event_step=event_step, event_tokens=event_tokens,
                                                                   error=str(exc)))
        except OSError:
            pass
        print(f'[playground] update skipped: {exc}; training checkpoint remains saved', flush=True)
        return None
