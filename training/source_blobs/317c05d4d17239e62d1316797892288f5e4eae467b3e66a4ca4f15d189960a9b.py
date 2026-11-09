"""Frozen new-source NLL, evaluated inline and kept outside reference selection."""
import hashlib
import math
import time

import numpy as np

from .common import atomic_json, file_sha256, read_json


def panel_fingerprint(x, y):
    digest = hashlib.sha256()
    for name, values in (('x', x), ('y', y)):
        digest.update(name.encode())
        digest.update(values.astype('<i4').tobytes())
    return digest.hexdigest()


def validate_policy(policy, recipe):
    keys = {'format', 'source', 'panel_fingerprint', 'panel_sha256', 'tokenizer', 'sequences', 'seq_len'}
    if (not isinstance(policy, dict) or set(policy) != keys
            or policy['format'] != 'sml-stage-b-diagnostic-v1'
            or not isinstance(policy['source'], str) or not policy['source']
            or type(policy['sequences']) is not int or policy['sequences'] != 32
            or policy['seq_len'] != recipe['model']['max_seq_len']
            or any(not isinstance(policy[k], str) or len(policy[k]) != 64
                   or any(c not in '0123456789abcdef' for c in policy[k])
                   for k in ('panel_fingerprint', 'panel_sha256', 'tokenizer'))):
        raise ValueError('Invalid frozen Stage B diagnostic policy')


def policy_from_receipt(receipt_path, source, tokenizer, recipe):
    receipt = read_json(receipt_path)
    spec = receipt['spec']
    # Token allocation does not change the held-out documents or panel bytes.
    if (dict(spec['source'], weight=source['weight']) != source
            or spec['partition'] != 'validation' or spec['tokenizer'] != tokenizer.fingerprint
            or spec['seq_len'] != recipe['model']['max_seq_len']):
        raise ValueError('Diagnostic source, tokenizer or reserved partition differs')
    policy = dict(format='sml-stage-b-diagnostic-v1', source=source['label'],
        panel_fingerprint=receipt['fingerprint'], panel_sha256=receipt['sha256'],
        tokenizer=spec['tokenizer'], sequences=receipt['sequences'], seq_len=spec['seq_len'])
    validate_policy(policy, recipe)
    return policy


def load_panel(path, policy, tokenizer, recipe):
    validate_policy(policy, recipe)
    if tokenizer.fingerprint != policy['tokenizer'] or file_sha256(path) != policy['panel_sha256']:
        raise ValueError('Diagnostic panel or tokenizer integrity failure')
    with np.load(path, allow_pickle=False) as arrays:
        x, y = arrays['x'].copy(), arrays['y'].copy()
    shape = (policy['sequences'], policy['seq_len'])
    if (x.shape != shape or y.shape != shape or x.dtype.kind not in 'iu' or y.dtype.kind not in 'iu'
            or np.min(x) < 0 or np.min(y) < 0
            or np.max(x) >= recipe['model']['vocab_size'] or np.max(y) >= recipe['model']['vocab_size']
            or panel_fingerprint(x, y) != policy['panel_fingerprint']):
        raise ValueError('Diagnostic shape, token IDs or fingerprint differs')
    return x, y


def evaluate(model, panel, policy, *, step, tokens, output):
    import mlx.core as mx
    started = time.monotonic()
    values = []
    model.eval()
    try:
        for x, y in zip(*panel):
            value = float(model(mx.array(x[None]), targets=mx.array(y[None]))['loss'].item())
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite new-source diagnostic loss')
            values.append(value)
    finally:
        model.train()
    loss = sum(values)/len(values)
    result = dict(format='sml-hq-validation-result-v1', step=step, tokens=tokens,
        hq_loss=loss, hq_ppl=math.exp(min(80, loss)), sequence_losses=values,
        panel=policy['panel_fingerprint'], token_positions=policy['sequences']*policy['seq_len'],
        device='inline training device', precision='BF16 model; training evaluation kernels; same method for baseline and continuation',
        elapsed_seconds=time.monotonic()-started,
        scope='HQ-only diagnostic; excluded from reference val_loss and best checkpoint selection.')
    path = output/'hq_validation.json'
    history = read_json(path) if path.exists() else dict(format='sml-hq-validation-history-v1', results=[])
    if any(r['panel'] != policy['panel_fingerprint'] for r in history['results']):
        raise ValueError('Existing HQ history uses a different panel')
    if any(r.get('device') != result['device'] or r.get('precision') != result['precision'] for r in history['results']):
        raise ValueError('Existing HQ history uses a different evaluation method')
    by_step = {r['step']: r for r in history['results']}
    by_step[step] = result
    history.update(status='Inline evaluation at each reference evaluation; no separate service needed.',
                   results=[by_step[s] for s in sorted(by_step)])
    atomic_json(path, history)
    print(f'[eval-hq {step}] hq_loss={loss:.6f} hq_ppl={result["hq_ppl"]:.2f} '
          f'tokens={tokens} sequences={len(values)}; separate from reference val_loss', flush=True)
    return result
