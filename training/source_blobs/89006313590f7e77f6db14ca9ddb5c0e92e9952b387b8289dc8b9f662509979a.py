"""Attention LoRA and tied chat-token deltas, with an immutable pretrained base."""
import hashlib
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from sml_v1.model import TransformerLM, TransformerConfig


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        out_dim, in_dim = linear.weight.shape
        if not 0 < rank <= min(out_dim, in_dim):
            raise ValueError('Invalid adapter rank')
        self.linear = linear
        self.linear.freeze()
        self.scale = float(alpha) / rank
        bound = 1 / math.sqrt(in_dim)
        self.lora_a = mx.random.uniform(-bound, bound, (in_dim, rank)).astype(mx.float32)
        self.lora_b = mx.zeros((rank, out_dim), mx.float32)

    def __call__(self, x):
        y = self.linear(x)
        delta = (x.astype(mx.float32) @ self.lora_a) @ self.lora_b
        return y + (self.scale * delta).astype(y.dtype)


class TrainableChatEmbedding(nn.Module):
    """Share two small row updates between input lookup and the tied LM head."""
    def __init__(self, embedding, token_ids):
        super().__init__()
        if list(token_ids) != [2, 3] or embedding.weight.shape[0] <= 3:
            raise ValueError('Only the existing chat-boundary tokens 2 and 3 are supported')
        self.linear = embedding
        self.linear.freeze()
        self._token_ids = tuple(token_ids)
        self.token_delta = mx.zeros((len(token_ids), embedding.weight.shape[1]), mx.float32)

    def __call__(self, ids):
        value = self.linear(ids)
        # Keep token deltas in FP32 for optimization; cast only the residual.
        for row, token_id in enumerate(self._token_ids):
            value = value + ((ids == token_id)[..., None] * self.token_delta[row]).astype(value.dtype)
        return value

    def as_linear(self, x):
        value = self.linear.as_linear(x)
        delta = x.astype(mx.float32) @ self.token_delta.T
        return value.at[..., mx.array(self._token_ids)].add(delta.astype(value.dtype))


def adapter_names(model):
    return [f'blocks.{i}.attn.{name}' for i, block in enumerate(model.blocks)
            for name in ('qkv_proj', 'proj')
            if isinstance(getattr(block.attn, name), LoRALinear)]


def attach(model, config):
    if (config['method'] != 'lora' or config['layers'] != 'all'
            or config['targets'] != ['attn.qkv_proj', 'attn.proj']
            or config['dropout'] != 0 or config['alpha'] <= 0):
        raise ValueError('Unsupported adapter configuration')
    model.freeze()
    if not isinstance(model.embed, nn.Embedding):
        raise ValueError('Expected the original tied embedding')
    model.embed = TrainableChatEmbedding(model.embed, config['trainable_token_ids'])
    for block in model.blocks:
        for name in ('qkv_proj', 'proj'):
            layer = getattr(block.attn, name)
            if not isinstance(layer, nn.Linear):
                raise ValueError('Expected unmodified attention projection')
            setattr(block.attn, name, LoRALinear(layer, config['rank'], config['alpha']))
    trainable = dict(tree_flatten(model.trainable_parameters()))
    expected = {n + suffix for n in adapter_names(model) for suffix in ('.lora_a', '.lora_b')}
    expected.add('embed.token_delta')
    if set(trainable) != expected:
        raise ValueError('Only adapter A/B matrices and the two tied token deltas may be trainable')
    count = sum(v.size for v in trainable.values())
    if config.get('expected_trainable_parameters', count) != count:
        raise ValueError('Unexpected trainable parameter count')
    return count


def base_digest(weights):
    """Independent value fingerprint; array aliases cannot hide in-place changes."""
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        digest.update((name + str(value.shape) + str(value.dtype)).encode())
        digest.update(np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value).tobytes())
    return digest.hexdigest()


def assert_frozen(model, expected_digest):
    actual = {n.replace('.linear.', '.'): v for n, v in tree_flatten(model.parameters())
              if not n.endswith(('.lora_a', '.lora_b', '.token_delta'))}
    if base_digest(actual) != expected_digest:
        raise ValueError('A frozen pretrained weight changed')


def merged_weights(model, optimizer, fp32_base):
    if any(v.dtype != mx.float32 for v in fp32_base.values()):
        raise ValueError('Merge requires exact FP32 pretrained masters')
    masters = {n[:-7]: v for n, v in tree_flatten(optimizer.state) if n.endswith('.master')}
    expected = set(dict(tree_flatten(model.trainable_parameters())))
    if set(masters) != expected or any(v.dtype != mx.float32 for v in masters.values()):
        raise ValueError('Missing FP32 adapter masters')
    weights = dict(fp32_base)
    weights['embed.weight'] = fp32_base['embed.weight'].at[mx.array(model.embed._token_ids)].add(masters['embed.token_delta'])
    for i, block in enumerate(model.blocks):
        for name in ('qkv_proj', 'proj'):
            layer = getattr(block.attn, name)
            prefix = f'blocks.{i}.attn.{name}'
            weights[prefix + '.weight'] = fp32_base[prefix + '.weight'] + (
                (masters[prefix + '.lora_a'] @ masters[prefix + '.lora_b']).T * layer.scale)
    mx.eval(weights)
    return weights


def dense_model(model, optimizer, fp32_base, config):
    result = TransformerLM(TransformerConfig(**config))
    result.load_weights(list(merged_weights(model, optimizer, fp32_base).items()), strict=True)
    mx.eval(result.parameters())
    return result
