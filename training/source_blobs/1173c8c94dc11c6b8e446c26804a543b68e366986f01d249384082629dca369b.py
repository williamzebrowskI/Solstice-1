"""Live qualitative samples; prompt text is reread, never part of validation."""
import hashlib
import json
from pathlib import Path

from .common import atomic_json


def read_prompts(path):
    raw = Path(path).read_bytes()
    if len(raw) > 256 * 1024:
        raise ValueError('Prompt file exceeds 256 KiB')
    value = json.loads(raw)
    prompts = value.get('prompts') if isinstance(value, dict) else value
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 32:
        raise ValueError('Use a JSON prompts array containing 1–32 prompts')
    if any(not isinstance(p, str) or not p.strip() or len(p) > 8192 for p in prompts):
        raise ValueError('Prompts must be nonempty strings of at most 8192 characters')
    return prompts, hashlib.sha256(raw).hexdigest()


def sample_event(step, tokens, generate, output, *, prompts_file=None, legacy_prompts=None):
    """One coherent file snapshot per event; malformed edits skip that event."""
    receipt = dict(step=step, tokens=tokens, kind='qualitative-samples', results=[])
    if prompts_file is not None:
        receipt['prompts_file'] = str(prompts_file)
        try:
            prompts, digest = read_prompts(prompts_file)
        except (OSError, ValueError) as exc:
            receipt.update(status='skipped', error=str(exc))
            atomic_json(Path(output) / 'samples' / f'step_{step:07d}.json', receipt)
            print(f'[sample {step}] skipped: {exc}; training continues', flush=True)
            return receipt
        receipt['prompts_sha256'] = digest
    else:
        prompts = legacy_prompts or []
    for prompt in prompts:
        try:
            text = generate(prompt)
        except ValueError as exc:
            receipt['results'].append(dict(prompt=prompt, status='skipped', error=str(exc)))
            print(f'[sample {step}] skipped prompt: {exc}', flush=True)
        else:
            receipt['results'].append(dict(prompt=prompt, status='generated', completion=text))
            print(f'[sample {step}] temperature=0 prompt={prompt!r}\n{text}', flush=True)
    receipt['status'] = 'complete'
    atomic_json(Path(output) / 'samples' / f'step_{step:07d}.json', receipt)
    return receipt
