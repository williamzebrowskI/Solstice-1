#!/usr/bin/env python3
"""Offline native MLX chat inference for the selected OpenSML checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ChatModel:
    def __init__(self, directory=ROOT, weights=None):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from inference_native.model import TransformerConfig, TransformerLM, count_parameters
        from inference_native.tokenization import Tokenizer
        self.mx = mx
        self.directory = Path(directory)
        self.config = json.loads((self.directory/'config.json').read_text())
        weights = Path(weights) if weights else self.directory/'model.safetensors'
        if not weights.is_file():
            raise ValueError('Missing model.safetensors; download the complete Hugging Face package.')
        if file_hash(weights) != self.config['weight_sha256']:
            raise ValueError('Weight checksum differs from the selected release checkpoint.')
        self.tokenizer = Tokenizer(self.directory/'tokenizer')
        if self.tokenizer.fingerprint != self.config['tokenizer_fingerprint']:
            raise ValueError('Tokenizer differs from the frozen release tokenizer.')
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(256 * 1024**2)
        self.model = TransformerLM(TransformerConfig(**self.config['model']))
        self.model.load_weights(str(weights), strict=True)
        mx.eval(self.model.parameters())
        self.model.eval()
        if count_parameters(self.model) != self.config['parameters']:
            raise ValueError('Parameter count differs from the release configuration.')
        if any(v.dtype != mx.float32 for _, v in tree_flatten(self.model.parameters())):
            raise ValueError('This release requires its original FP32 weights.')

    def prepare(self, messages, max_new_tokens):
        from inference_native.chat_format import serialize
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2047:
            raise ValueError('max_new_tokens must be an integer from 1 through 2047.')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Supply a nonempty list of chat messages ending with a user turn.')
        if any(not isinstance(m, dict) or set(m) != {'role', 'content'} for m in messages):
            raise ValueError('Each message must contain exactly role and content.')
        ids, _ = serialize(self.tokenizer, messages, generation=True)
        if len(ids) + max_new_tokens > self.config['context_limit']:
            raise ValueError('Conversation plus output budget exceeds 2048 tokens; shorten it or start a new chat. No truncation is performed.')
        return ids

    def generate(self, messages, max_new_tokens=128, temperature=0., seed=2026):
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be finite and between 0 and 2.')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be an unsigned 32-bit integer.')
        ids = self.prepare(messages, max_new_tokens)
        mx = self.mx
        mx.random.seed(seed)
        inputs = mx.array(ids[None])
        caches = None
        generated = []
        stop = 'token_limit'
        started = time.perf_counter()
        try:
            for _ in range(max_new_tokens):
                logits, caches = self.model.logits(inputs, caches)
                scores = logits[0, -1].astype(mx.float32)
                token = int((mx.argmax(scores) if temperature == 0 else mx.random.categorical(scores/temperature)).item())
                if token in (0, 1, 2, 3):
                    stop = {0: 'unexpected_structure', 1: 'document_end', 2: 'unexpected_structure', 3: 'turn_end'}[token]
                    break
                generated.append(token)
                inputs = mx.array([[token]], dtype=mx.int32)
            return dict(model=self.config['model_name'], text=self.tokenizer.decode(generated),
                        token_ids=generated, prompt_tokens=len(ids), output_tokens=len(generated),
                        stop_reason=stop, seconds=time.perf_counter()-started,
                        protocol=dict(precision='float32', attention='vanilla', temperature=temperature,
                                      seed=seed, max_new_tokens=max_new_tokens, repetition_penalty=1,
                                      serialization='explicit-ids-assistant-mask-v1', truncation=False))
        finally:
            caches = None
            mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--messages-file', type=Path, help='JSON array of role/content messages, ending in a user turn.')
    source.add_argument('--interactive', action='store_true', help='Keep previous turns; /reset clears history and /quit exits.')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--json', action='store_true', help='Print the generated answer and inference metadata as JSON.')
    args = parser.parse_args()
    try:
        model = ChatModel(weights=args.weights)
        if args.interactive:
            messages = []
            while True:
                try: text = input('You: ').strip()
                except (EOFError, KeyboardInterrupt): break
                if text == '/quit': break
                if text == '/reset': messages = []; print('Conversation cleared.'); continue
                if not text: continue
                request = messages + [dict(role='user', content=text)]
                try: result = model.generate(request, args.max_new_tokens, args.temperature, args.seed)
                except ValueError as exc: print(str(exc)); continue
                print(json.dumps(result, ensure_ascii=False) if args.json else 'Assistant: '+result['text'])
                if result['text'].strip():
                    messages = request + [dict(role='assistant', content=result['text'])]
        else:
            messages = json.loads(args.messages_file.read_text()) if args.messages_file else [dict(role='user', content=args.prompt)]
            result = model.generate(messages, args.max_new_tokens, args.temperature, args.seed)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else result['text'])
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
