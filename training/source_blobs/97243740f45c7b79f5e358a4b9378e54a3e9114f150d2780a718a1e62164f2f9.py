"""Deterministic target checks; no benchmark questions or answers are imported."""
import re


def words(text):
    return re.findall(r"\b[\w'-]+\b", text)


def compare(value, op, target):
    return {'exactly': value == target, 'at least': value >= target,
            'less than': value < target, 'at most': value <= target}[op]


def check_rules(text, rules):
    for rule in rules:
        kind = rule['kind']
        if kind == 'case':
            letters = ''.join(c for c in text if c.isalpha())
            if not letters or letters != (letters.upper() if rule['value'] == 'upper' else letters.lower()): return False
        elif kind == 'end':
            if not text.rstrip().endswith(rule['value']): return False
        elif kind == 'title':
            if not re.match(r'^\s*<<[^<>\n]+>>', text): return False
        elif kind == 'quote':
            if not (text.startswith('"') and text.endswith('"')): return False
        elif kind == 'postscript':
            if not re.search(r'(?im)^p\.s\.', text): return False
        elif kind == 'forbid':
            if any(re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)', text, re.I) for s in rule['values']): return False
        elif kind == 'include':
            if not all(re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)', text, re.I) for s in rule['values']): return False
        else:
            count = {'words': lambda: len(words(text)),
                     'sentences': lambda: len([s for s in re.split(r'[.!?]+(?:\s+|$)', text.strip()) if s.strip()]),
                     'bullets': lambda: len(re.findall(r'(?m)^\s*[-*] +\S', text)),
                     'sections': lambda: len(re.findall(r'(?m)^SECTION \d+', text)),
                     'paragraphs': lambda: len([s for s in re.split(r'\n\s*\n', text.strip()) if s.strip()]),
                     'divided_paragraphs': lambda: len([s for s in text.split('***') if s.strip()]),
                     'placeholders': lambda: len(re.findall(r'\[[^\[\]\n]+\]', text)),
                     'highlights': lambda: len(re.findall(r'\*[^*\n]+\*', text))}[kind]()
            if not compare(count, rule['op'], rule['count']): return False
    return True


def public_rules(prompt):
    """Recognize source constraints conservatively; remaining semantics need review.

    These rows are training examples, not objective checkpoint-selection cases.
    The checks certify only the recognized properties, not factual correctness.
    """
    rules=[]
    for m in re.finditer(r'(exactly|at least|less than|at most)\s+(\d+)\s+(words|sentences|paragraphs|bullet points|sections|placeholders|highlights?)', prompt, re.I):
        kind={'bullet points':'bullets','highlight':'highlights'}.get(m[3].lower(),m[3].lower())
        rules.append(dict(kind=kind,op=m[1].lower(),count=int(m[2])))
    for m in re.finditer(r'must have\s+(\d+)\s+paragraphs',prompt,re.I):
        rules.append(dict(kind='divided_paragraphs' if '***' in prompt else 'paragraphs',op='exactly',count=int(m[1])))
    if re.search(r'all lowercase|no capital letters',prompt,re.I): rules.append(dict(kind='case',value='lower'))
    if re.search(r'all uppercase|all capital letters',prompt,re.I): rules.append(dict(kind='case',value='upper'))
    if re.search(r'title.*double angular brackets',prompt,re.I): rules.append(dict(kind='title'))
    if re.search(r'postscript starting with p\.s\.',prompt,re.I): rules.append(dict(kind='postscript'))
    if re.search(r'quotation marks',prompt,re.I): rules.append(dict(kind='quote'))
    end=re.search(r'\[ender\] is\s*["“]([^"”]+)["”]',prompt,re.I)
    if end: rules.append(dict(kind='end',value=end[1]))
    elif '[ender]' in prompt: raise ValueError('unresolved_ender')
    forbidden=re.search(r'\[forbidden_words\] are:\s*([^\n.]+)',prompt,re.I)
    if forbidden: rules.append(dict(kind='forbid',values=[s.strip() for s in forbidden[1].split(',')]))
    elif '[forbidden_words]' in prompt:raise ValueError('unresolved_forbidden_words')
    required=re.search(r'\[keywords\] (?:should be|are):\s*([^\n.]+)',prompt,re.I)
    if required:rules.append(dict(kind='include',values=[s.strip() for s in required[1].split(',')]))
    elif '[keywords]' in prompt:raise ValueError('unresolved_keywords')
    if re.search(r'\[(?:topic|language|number|word|sentence|paragraph)\]',prompt,re.I):raise ValueError('unresolved_slot')
    # Cases requiring operations not implemented here are not admitted.
    if re.search(r'JSON|repeat.*prompt|two different responses|letter.*frequency|keyword.*times|nth|[1-9](?:st|nd|rd|th) paragraph|language.*(?:French|Spanish|German)|in (?:French|Spanish|German|Chinese|Arabic)',prompt,re.I):
        raise ValueError('unsupported_constraint')
    if not rules: raise ValueError('no_checkable_constraint')
    return rules
