"""Opt-in, dependency-free English prose heuristics for live input.

Version english-prose-v1 mirrors the reviewed offline heuristics without its
language model, benchmark exclusion, or global exact/near-duplicate passes.
Kept self-contained so staged workers do not depend on preparation artifacts.
"""
import re

CODE_LINE = re.compile(
    r'^\s*(?:def\s+\w+\s*\(|async\s+def\s|class\s+\w+.*:|'
    r'from\s+[\w.]+\s+import\s|import\s+[\w.]+\s*$|@\w+(?:\.|\()|'
    r'(?:elif\s+.{1,100}:|for\s+\w+\s+in\s+.{1,100}:|'
    r'with\s+\w+\([^\n]*\)\s+as\s+\w+\s*:|try\s*:|except\s+\w*\s*:)|'
    r'(?:return|raise|yield)\s+|(?:const|let|var)\s+\w+\s*=|'
    r'#include\s*[<"]|(?:public|private)\s+(?:static\s+)?\w+\s+\w+\(|'
    r'(?:print|range|open|np\.\w+|pd\.\w+|torch\.\w+|plt\.\w+)\([^\n]*\)\s*$|>>>\s+)')
PYTHON_FENCE = re.compile(r'```\s*(?:python\d*|py|ipython)\b', re.I)
JUNK = re.compile(r'\b(?:enable javascript|accept all cookies|cookie preferences|'
                  r'subscribe to our newsletter|sign in to continue|all rights reserved)\b', re.I)
HTML = re.compile(r'</?(?:div|span|script|style|iframe|nav)\b[^>]*>', re.I)
WORDS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SPAM = re.compile(r'\b(?:Solvusoft|WinThruster|DriverDoc|RegistryBooster|'
                  r'registry cleaner|free registry scan|download our repair tool)\b',re.I)


def heuristic_reason(text, min_chars=200, max_bytes=200000):
    if not isinstance(text, str):
        return 'missing_text'
    if len(text) < min_chars:
        return 'too_short'
    if len(text.encode('utf-8')) > max_bytes:
        return 'too_long'
    if '\x00' in text or '\ufffd' in text:
        return 'broken_encoding'
    if sum(ord(c) < 32 and c not in '\t\n\r' for c in text) > 0:
        return 'control_characters'
    if PYTHON_FENCE.search(text):
        return 'python_block'
    if SPAM.search(text):
        return 'software_download_spam'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    code = [line for line in lines if CODE_LINE.search(line)]
    if len(code) >= 3 or sum(map(len, code)) > max(80, len(text) * .10):
        return 'code_heavy'
    if len(lines) >= 10 and len(set(lines)) / len(lines) < .65:
        return 'repeated_lines'
    if len(HTML.findall(text)) >= 3:
        return 'html_residue'
    if len(JUNK.findall(text)) >= 4:
        return 'navigation_boilerplate'
    words = WORDS.findall(text)
    if len(words) < 40:
        return 'too_little_prose'
    if sum(map(len, words)) / len(text) < .50:
        return 'low_prose_density'
    if len(words) >= 100 and len(set(w.lower() for w in words)) / len(words) < .12:
        return 'repetitive_vocabulary'
    return None

