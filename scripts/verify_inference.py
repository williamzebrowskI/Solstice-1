#!/usr/bin/env python3
"""Check native inference against a fixed subset of saved benchmark outputs."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import ChatModel
from inference_native.chat_format import serialize
from inference_native.tokenization import FIXTURES, SPECIALS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = ChatModel(weights=args.weights)
    guards = []
    for text in FIXTURES:
        assert model.tokenizer.decode(model.tokenizer.encode(text)) == text
        assert not set(model.tokenizer.encode(text)) & {0, 1, 2, 3}
    guards.append('Tokenizer round trips and literal structural-token isolation')
    messages = [{'role': 'user', 'content': 'hello'}]
    expected = [2]+model.tokenizer.encode('user\n')+model.tokenizer.encode('hello')+[3, 2]+model.tokenizer.encode('assistant\n')
    assert model.prepare(messages, 128).tolist() == expected
    guards.append('Explicit chat token boundaries')
    for invalid in [[], [{'role':'assistant','content':'hi'}], [{'role':'user','content':''}],
                    [{'role':'user','content':'a'},{'role':'user','content':'b'}],
                    [{'role':'system','content':'s'}], [{'role':'tool','content':'x'}]]:
        try: model.prepare(invalid, 128)
        except ValueError: pass
        else: raise AssertionError(f'Accepted invalid conversation: {invalid}')
    guards.append('Invalid roles, empty messages and turn orders rejected')
    for invalid in [0, 2048, True, -1]:
        try: model.prepare(messages, invalid)
        except ValueError: pass
        else: raise AssertionError('Accepted invalid output budget')
    try: model.prepare([{'role':'user','content':'word '*2200}], 128)
    except ValueError: pass
    else: raise AssertionError('Context overflow was not rejected')
    guards.append('Output budget bounds and context overflow rejected')
    for value in [float('nan'), float('inf'), -1, 2.1]:
        try: model.generate(messages, temperature=value)
        except ValueError: pass
        else: raise AssertionError('Accepted invalid temperature')
    guards.append('Non-finite and out-of-range temperatures rejected')
    groups = defaultdict(list)
    for line in (ROOT/'results/public/chat/responses.jsonl').read_text().splitlines():
        row = json.loads(line)
        groups[row['case']['category']].append(row)
    # Chosen by fixed positions, including failures; not by answer correctness.
    selected = [rows[i] for rows in groups.values() for i in (0, 1, 5, 15) if i < len(rows)]
    comparisons = []
    for row in selected:
        case = row['case']; prior = row['response']
        result = model.generate(case['messages'], max_new_tokens=case['max_tokens'])
        matches = result['text'] == prior['text'] and result['stop_reason'] == prior['stop_reason'] and result['output_tokens'] == prior['tokens']
        comparisons.append({'id':case['id'], 'matches_saved_fp32':matches, 'previous':prior, 'observed':result})
        print(f"{case['id']}: {'match' if matches else 'DIFFERENT'}", flush=True)
    # Verify a sampled invocation is reproducible, without asserting its quality.
    sample = model.generate(messages, max_new_tokens=32, temperature=.6, seed=2026)
    repeated = model.generate(messages, max_new_tokens=32, temperature=.6, seed=2026)
    assert sample['token_ids'] == repeated['token_ids']
    guards.append('Fixed-seed sampling is repeatable in this runtime')
    report = {'passed': all(r['matches_saved_fp32'] for r in comparisons),
              'full_benchmark_rerun':False, 'weight_sha256':model.config['weight_sha256'],
              'environment': {'python':platform.python_version(), 'macos':platform.mac_ver()[0],
                'machine':platform.machine(), **{k:importlib.metadata.version(k) for k in ['mlx','numpy','tokenizers']}},
              'checks':guards, 'saved_response_comparisons':len(comparisons),
              'matching_responses':sum(r['matches_saved_fp32'] for r in comparisons),
              'comparisons':comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(not report['passed'])


if __name__ == '__main__':
    main()
