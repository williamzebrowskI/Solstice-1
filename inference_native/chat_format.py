"""One complete conversation per sequence; deterministic, resumable batches."""
import math
from pathlib import Path
import numpy as np


def serialize(tokenizer, messages, *, generation=False):
    ids, mask = [], []
    previous = None
    for i, message in enumerate(messages):
        role, text = message['role'], message['content']
        if role not in ('system', 'user', 'assistant') or not isinstance(text, str) or not text.strip():
            raise ValueError('Invalid chat message')
        if role == 'system':
            if i: raise ValueError('System message must be first')
        elif role == 'user':
            if previous not in (None, 'system', 'assistant'): raise ValueError('User turn order')
        elif previous != 'user': raise ValueError('Assistant turn order')
        header = [2] + tokenizer.encode(role + '\n')
        content = tokenizer.assert_roundtrip(text)
        ids += header + content + [3]
        mask += [False] * len(header) + [role == 'assistant'] * (len(content) + 1)
        previous = role
    if generation:
        if previous != 'user': raise ValueError('Generation requires a final user turn')
        header = [2] + tokenizer.encode('assistant\n')
        ids += header; mask += [False] * len(header)
    else:
        if previous != 'assistant': raise ValueError('Training requires an assistant answer')
        ids.append(1); mask.append(True)
    return np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=np.bool_)


def shifted(ids, mask):
    return ids[:-1], np.where(mask[1:], ids[1:], -100).astype(np.int32)


def epoch_order(lengths, seed, batch_size=64):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    # Randomized length buckets reduce padding without changing membership.
    chunks = []
    window_size = ((1024 + batch_size - 1) // batch_size) * batch_size
    for start in range(0, len(order), window_size):
        window = order[start:start+window_size]
        window = window[np.argsort(np.asarray(lengths)[window], kind='stable')]
        chunks.extend(window[i:i+batch_size] for i in range(0, len(window), batch_size))
    # Every complete window is divisible by the requested batch size.
    full = [c for c in chunks if len(c) == batch_size]
    partial = [c for c in chunks if len(c) != batch_size]
    rng.shuffle(full)
    return full + partial


def training_order(lengths, seed, batch_size=64, epochs=1):
    if type(epochs)!=int or not 1<=epochs<=2: raise ValueError('Expected one or two SFT passes')
    return [batch for epoch in range(epochs) for batch in epoch_order(lengths,seed+epoch,batch_size)]


class Corpus:
    def __init__(self, directory, split):
        path = Path(directory)
        self.ids = np.load(path/f'{split}.tokens.npy', mmap_mode='r', allow_pickle=False)
        self.mask = np.load(path/f'{split}.mask.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(path/f'{split}.offsets.npy', mmap_mode='r', allow_pickle=False)
        self.lengths = np.diff(self.offsets).astype(np.int64)-1

    def __len__(self): return len(self.offsets)-1

    def batch(self, indices, size=None, max_length=2048):
        size = size or len(indices)
        if len(indices) > size or not len(indices): raise ValueError('Empty/oversized batch')
        length = int(max(self.lengths[indices]))
        if length > max_length: raise ValueError('Conversation exceeds context')
        length = min(max_length, max(16, 2**math.ceil(math.log2(length))))
        x = np.zeros((size, length), np.int32)
        y = np.full((size, length), -100, np.int32)
        for row, index in enumerate(indices):
            a, b = map(int, self.offsets[index:index+2])
            xi, yi = shifted(self.ids[a:b], self.mask[a:b])
            x[row, :len(xi)] = xi; y[row, :len(yi)] = yi
        return x, y


def learning_rate(step, total, peak, final, warmup_fraction):
    warmup = max(1, math.ceil(total*warmup_fraction))
    if step <= warmup: return peak*step/warmup
    progress = min(1., (step-warmup)/max(1, total-warmup))
    return final+(peak-final)*(1+math.cos(math.pi*progress))/2
