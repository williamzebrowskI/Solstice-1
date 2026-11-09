#!/usr/bin/env python3
"""Verify release files without loading the model or accessing the network."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true', help='Allow the GitHub checkout to omit model.safetensors.')
    args = parser.parse_args()
    files = json.loads((ROOT/'FILE_MANIFEST.json').read_text())['files']
    failures = []
    checked = 0
    for name, record in files.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(f'Missing or invalid path: {name}')
        elif path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            failures.append(f'Integrity mismatch: {name}')
        else:
            checked += 1
    weights = ROOT/'model.safetensors'
    if weights.is_file():
        expected = json.loads((ROOT/'config.json').read_text())['weight_sha256']
        if 'model.safetensors' not in files:
            if digest(weights) != expected:
                failures.append('Weight checksum does not match config.json')
            else:
                checked += 1
    elif not args.source_only:
        failures.append('Missing model.safetensors; download it from OpenSML/Solstice-1.')
    print(json.dumps({'passed': not failures, 'files_verified': checked, 'weights_present': weights.is_file(), 'failures': failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
