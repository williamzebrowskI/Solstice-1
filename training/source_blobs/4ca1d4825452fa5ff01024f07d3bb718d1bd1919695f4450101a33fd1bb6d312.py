"""Start a new stage from learned adapters without carrying optimizer moments."""
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from sml_v1.common import file_sha256
from .lora_checkpoint import verify
from .lora_adapter import merged_weights


def initialize(model, optimizer, base, parent, adaptation, contract):
    state = verify(parent)
    if state['adaptation'] != adaptation:
        raise ValueError('Parent adapter architecture changed')
    for name in ('source_weights', 'source_masters', 'tokenizer', 'retention'):
        if state['contract'][name] != contract[name]:
            raise ValueError('Parent frozen input changed: ' + name)
    if file_sha256(parent / 'manifest.json') != contract['initial_parent_manifest']:
        raise ValueError('Initial parent checkpoint changed')
    values = mx.load(str(parent / 'adapters.safetensors'))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(values) != set(expected) or any(values[k].shape != v.shape or values[k].dtype != mx.float32
                                          for k, v in expected.items()):
        raise ValueError('Initial adapter tensors changed')
    if not all(bool(mx.all(mx.isfinite(v)).item()) for v in values.values()):
        raise ValueError('Nonfinite initial adapter')
    model.update(tree_unflatten(list(values.items())))
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    merged = merged_weights(model, optimizer, base)
    reference = mx.load(str(parent / 'inference/model.safetensors'))
    if set(merged) != set(reference) or not all(bool(mx.all(v == reference[k]).item()) for k, v in merged.items()):
        raise ValueError('Second-stage initialization does not reproduce parent FP32 weights')
    return dict(sft_stage=2, parent_step=state['step'], parent_checkpoint=str(parent),
                parent_manifest=contract['initial_parent_manifest'], optimizer_initialization='fresh moments')
