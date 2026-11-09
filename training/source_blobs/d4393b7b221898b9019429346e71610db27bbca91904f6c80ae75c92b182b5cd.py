"""Small frozen code panels, reported separately from the unchanged prose metric."""
import math
import copy
import re
import numpy as np

from .common import atomic_json, file_sha256, fingerprint, read_json

FORMAT = 'sml-python-sql-heldout-v1'


def source_identity(manifest):
    return fingerprint([s for s in manifest['corpus']['sources'] if s['label'] in ('python','sql')])


def validate_policy(policy, manifest, tokenizer):
    if (policy.get('format') != FORMAT or policy.get('sources') != source_identity(manifest)
            or policy.get('tokenizer') != tokenizer.fingerprint
            or type(policy.get('seq_len')) is not int or policy['seq_len'] < 1
            or type(policy.get('sequences')) is not int or not 1 <= policy['sequences'] <= 32
            or not isinstance(policy.get('excluded_content_keys'),list)
            or not all(isinstance(k,str) and re.fullmatch('[0-9a-f]{64}',k) for k in policy['excluded_content_keys'])
            or not re.fullmatch('[0-9a-f]{64}',policy.get('sha256',''))):
        raise ValueError('Code validation policy differs from source/tokenizer contract')


def prepare_panel(manifest, tokenizer, seq_len, destination, sequences=16):
    from .code_phase import CodePhaseStream
    arrays = {}
    excluded = set()
    for label in ('python','sql'):
        source = next(s for s in manifest['corpus']['sources'] if s['label']==label)
        primary = {k:v for k,v in source.items() if k!='additional_sources'}
        components = [primary,*source.get('additional_sources',[])]
        pairs = []
        for index,component in enumerate(components):
            temporary = copy.deepcopy(manifest)
            temporary['corpus']['sources'] = [
                dict(primary,**component) if s['label']==label else s for s in temporary['corpus']['sources']]
            count = sequences//len(components) + (index < sequences%len(components))
            stream = CodePhaseStream(temporary,tokenizer,'validation',seq_len)
            try:
                pairs.append(stream.batch(count,label))
                excluded.update(stream.seen)
                print(f'[panel] {label} component={index} sequences={count} scanned={stream.sources[label].rows}',flush=True)
            finally:
                stream.close()
        for axis,name in enumerate(('x','y')):
            arrays[label+'_'+name] = np.concatenate([p[axis] for p in pairs])
    np.savez(destination, **arrays)
    policy = dict(format=FORMAT, sources=source_identity(manifest), tokenizer=tokenizer.fingerprint,
        seq_len=seq_len, sequences=sequences, sha256=file_sha256(destination), excluded_content_keys=sorted(excluded))
    validate_policy(policy,manifest,tokenizer)
    return policy


def load_panel(path, policy, manifest, tokenizer, recipe):
    validate_policy(policy,manifest,tokenizer)
    if policy['seq_len'] != recipe['model']['max_seq_len'] or file_sha256(path) != policy['sha256']:
        raise ValueError('Code validation bytes/context length changed')
    result = {}
    with np.load(path,allow_pickle=False) as values:
        for source in ('python','sql'):
            pair = tuple(values[source+'_'+axis].copy() for axis in ('x','y'))
            for array in pair:
                if (array.shape != (policy['sequences'],policy['seq_len']) or array.dtype.kind not in 'iu'
                        or np.min(array)<0 or np.max(array)>=tokenizer.vocab_size):
                    raise ValueError('Invalid code validation token array')
            result[source] = pair
    return result


def evaluate(model, panel, *, step, tokens, output, panel_sha256):
    import mlx.core as mx
    losses = {}
    model.eval()
    try:
        for source,(x,y) in panel.items():
            values = [float(model(mx.array(a[None]),targets=mx.array(b[None]))['loss'].item()) for a,b in zip(x,y)]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError('Nonfinite code validation loss')
            losses[source] = sum(values)/len(values)
    finally:
        model.train()
    path = output/'code_validation.json'
    history = read_json(path) if path.exists() else dict(format=FORMAT,panel_sha256=panel_sha256,results=[])
    if history['panel_sha256'] != panel_sha256:
        raise ValueError('Code validation panel changed within run')
    result = dict(step=step,tokens=tokens,losses=losses)
    by_step = {r['step']:r for r in history['results']}
    by_step[step] = result
    history['results'] = [by_step[s] for s in sorted(by_step)]
    atomic_json(path,history)
    print(f'[eval-code {step}] python_loss={losses["python"]:.6f} sql_loss={losses["sql"]:.6f}; separate from prose best selection',flush=True)
    return result
