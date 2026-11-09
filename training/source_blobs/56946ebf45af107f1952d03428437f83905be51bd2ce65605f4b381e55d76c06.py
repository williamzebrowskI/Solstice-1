"""SFT-parent provenance and fresh optimizer moments with preserved FP32 masters."""
import gzip
from pathlib import Path
from sml_v1.common import file_sha256, read_json
from .metadata import sft_info
from .retention import verify as verify_retention


def context(metadata, retention):
    receipt=verify_retention(retention)
    info=sft_info(metadata)
    if info is None:
        if receipt['validation_fingerprint']!=metadata['validation_fingerprint']:
            raise ValueError('Pretraining validation fingerprint differs')
        return dict(pretraining_step=metadata['step'],pretraining_tokens=metadata['tokens'],sft_stage=1,
                    prior_assistant_tokens=0,validation_fingerprint=receipt['validation_fingerprint'])
    if info['contract']['retention']!=file_sha256(Path(retention)/'manifest.json'):
        raise ValueError('SFT parent uses a different original retention panel')
    return dict(pretraining_step=info['parent_step'],pretraining_tokens=info['parent_tokens'],
                sft_stage=metadata.get('sft_stage',1)+1,
                prior_assistant_tokens=metadata.get('lifetime_sft_assistant_tokens',metadata['assistant_tokens']),
                validation_fingerprint=receipt['validation_fingerprint'])


def verify_sft_parent(directory, retention):
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('Parent cannot be a symlink')
    manifest=read_json(directory/'manifest.json')
    required={'model.safetensors','model.safetensors.json','model.safetensors.optimizer.safetensors','model.safetensors.rank0.data_state.json.gz'}
    if manifest.get('format')!='sml-pretrain-bundle-v1' or not required<=manifest['files'].keys():
        raise ValueError('Incomplete SFT parent')
    for name,record in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Unsafe parent filename')
        file=directory/name
        if file.is_symlink() or file.stat().st_size!=record['bytes'] or file_sha256(file)!=record['sha256']:
            raise ValueError('SFT parent checksum failed: '+name)
    metadata=read_json(directory/'model.safetensors.json')
    if sft_info(metadata) is None or metadata['step']!=manifest['step']:
        raise ValueError('Invalid SFT parent metadata')
    with gzip.open(directory/'model.safetensors.rank0.data_state.json.gz','rt') as stream:cursor=__import__('json').load(stream)
    if cursor!={'step':metadata['step'],'stream_state':dict(next_batch=metadata['step'],order_hash=metadata['order_hash'])}:
        raise ValueError('SFT parent cursor differs')
    context(metadata,retention)
    return metadata


def initialize_masters(model, optimizer, path):
    """Keep the trained FP32 parameters, resetting Adam's step and moments."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    masters=mx.load(str(path));parameters=dict(tree_flatten(model.parameters()))
    if set(masters)!=set(parameters):raise ValueError('Source master tensor names differ')
    checks=[]
    for name,value in masters.items():
        parameter=parameters[name]
        if value.dtype!=mx.float32 or value.shape!=parameter.shape:
            raise ValueError('Source master dtype/shape differs: '+name)
        checks.append(mx.all(value.astype(parameter.dtype)==parameter))
    mx.eval(checks)
    if not all(bool(x.item()) for x in checks):raise ValueError('Source masters do not round to source model')
    state=dict(tree_flatten(optimizer.state))
    for name,value in masters.items():state[name+'.master']=value
    optimizer.state=tree_unflatten(list(state.items()))
