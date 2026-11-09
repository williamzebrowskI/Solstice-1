#!/usr/bin/env python3
"""Restore an archived training source tree into a new directory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--snapshot')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    index = json.loads((ROOT/'training/SNAPSHOTS.json').read_text())
    if args.list:
        print(json.dumps(index, indent=2)); return
    if args.snapshot not in index or args.output is None:
        parser.error('Choose --snapshot from --list and supply a new --output directory.')
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory. No files were changed.')
    manifest = json.loads((ROOT/index[args.snapshot]['snapshot']).read_text())
    checked = []
    for name, record in manifest['files'].items():
        destination = (output/name).resolve()
        source = (ROOT/record['file']).resolve()
        if not destination.is_relative_to(output) or not source.is_relative_to(ROOT):
            raise ValueError('Invalid snapshot path')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['packaged_sha256']:
            raise ValueError(f'Snapshot integrity mismatch: {name}')
        checked.append((destination, data))
    output.mkdir(parents=True, exist_ok=False)
    for destination, data in checked:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(json.dumps({'snapshot': args.snapshot, 'files': len(checked), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
