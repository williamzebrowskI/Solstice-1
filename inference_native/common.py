import hashlib
import json
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text())

def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
