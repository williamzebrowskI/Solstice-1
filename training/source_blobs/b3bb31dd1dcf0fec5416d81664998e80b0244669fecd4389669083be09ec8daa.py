"""CPU-only source selection. Parsing checks syntax; downloaded code is never run."""
import ast
import hashlib
import math
import re
import warnings

from .corpus import content_key, partition

SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}')
SQL_START = re.compile(r'\b(?:SELECT|WITH|CREATE|ALTER|INSERT|UPDATE|DELETE|MERGE|DROP|BEGIN|DECLARE|GRANT|SET)\b', re.I)
BENCHMARK = re.compile(r'(?:human[_-]?eval|mbpp|ds[_-]?1000|evalplus|spider|bird[_-]?(?:bench|sql))', re.I)


def repository_key(row):
    name = row.get('repo_name', row.get('repo_id'))
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Code record is missing repository provenance')
    return hashlib.sha256(name.casefold().encode()).hexdigest()


def metadata_reason(row, source):
    if row.get('download_success') is False:
        return 'content_unavailable'
    if row.get('_rejection'):
        return row['_rejection']
    language = source['code_filter']['language']
    if row.get('language', language) != language:
        return 'wrong_language'
    if not row.get('repo_name', row.get('repo_id')):
        return 'missing_provenance'
    path = row.get('path', row.get('file_path', ''))
    if BENCHMARK.search(str(path) + ' ' + str(row.get('repo_name', row.get('repo_id', '')))):
        return 'benchmark_path'
    if row.get('is_generated') or row.get('is_vendor'):
        return 'generated_or_vendor'
    if source.get('min_score') and row.get('_code_kind') != 'synsql':
        policy = source['min_score']
        score = row.get(policy['field'])
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('Missing or nonfinite code quality score')
        if score < policy['value']:
            return 'educational_score'
    if source['code_filter'].get('require_explicit_license'):
        if row.get('license_type') != 'permissive' or not row.get('detected_licenses'):
            return 'missing_permissive_license'
    return None


def select(row, source, config, split, exclusions=(), tokenizer=None):
    reason = metadata_reason(row, source)
    if reason:
        return None, reason
    text = row.get(source['text_field'])
    if not isinstance(text, str):
        raise ValueError('Code source does not contain its declared text field')
    if len(text) < config['min_chars'] or len(text.encode('utf-8')) > config['max_bytes']:
        return None, 'length'
    if '\x00' in text or '\ufffd' in text:
        return None, 'encoding'
    if row.get('_code_kind') != 'synsql':
        try:
            raw = text.encode(row.get('src_encoding') or 'utf-8', errors='strict')
            verified = hashlib.sha1(raw).hexdigest() == row.get('blob_id')
        except (UnicodeError, LookupError):
            verified = False
        if not verified:
            return None, 'content_hash'
    key = content_key(text)
    group = partition(repository_key(row))
    # Repositories reserved for validation/final never contribute training data.
    # Content-based train exclusion also preserves the existing pending-buffer checks.
    if group != split or (split == 'train' and partition(key) != 'train'):
        return None, 'partition'
    if key in exclusions:
        return None, 'heldout_content'
    if SECRET.search(text):
        return None, 'credential_pattern'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 20 and len(set(lines))/len(lines) < .35:
        return None, 'repetition'
    if max(map(len, lines), default=0) > 4000:
        return None, 'oversized_line'
    language = source['code_filter']['language']
    if language == 'Python':
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return None, 'python_syntax'
    elif language == 'SQL':
        if not SQL_START.search(text):
            return None, 'no_sql_statement'
        if len(re.findall(r'\bINSERT\s+INTO\b', text, re.I)) > 50 or len(re.findall(r'\),\s*\(', text)) > 100:
            return None, 'bulk_insert_dump'
        if row.get('_code_kind') == 'synsql':
            if tokenizer is None:
                raise ValueError('Synthetic SQL length filtering requires the frozen tokenizer')
            if len(tokenizer.encode(text)) > 2047:
                return None, 'synthetic_context_length'
        # This heuristic is deliberately not called SQL syntax/execution validation.
    else:
        raise ValueError('Unsupported code language')
    return text, None


