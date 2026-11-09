"""Conservative answer matching, separate from response control measures."""
from collections import Counter
import re


def normalize(text, case_sensitive=False):
    text = text.replace('’', "'").strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text).rstrip('.!?')
    text = re.sub(r'\s*,\s*', ', ', text)
    return text if case_sensitive else text.casefold()


def looping(text):
    words = re.findall(r'\w+', text.casefold())
    if len(words) >= 24:
        grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(set(grams)) / len(grams) < .68:
            return True
    for chunks in (text.splitlines(), re.split(r'[.!?]+', text)):
        normalized = [normalize(s) for s in chunks if len(s.split()) >= 4]
        if normalized and max(Counter(normalized).values()) >= 3:
            return True
    return False


def score(case, response):
    text = response['text']
    normal = response['stop_reason'] in ('turn_end', 'document_end')
    repeat = looping(text)
    empty = not bool(text.strip())
    exact = None
    wrapped = None
    if not case['qualitative']:
        aliases = {normalize(a, case['case_sensitive']) for a in case['answers']}
        exact = normalize(text, case['case_sensitive']) in aliases
        # Only strip these neutral wrappers; substring matches are never correctness.
        unwrapped = re.sub(r'^(?:answer\s*:\s*|the answer is\s+|the result is\s+|it is\s+|it\x27s\s+)', '', text, flags=re.I)
        wrapped = normalize(unwrapped, case['case_sensitive']) in aliases
    return dict(answer_match=exact, answer_match_with_neutral_wrapper=wrapped,
                natural_stop=normal, repeated=repeat, empty=empty,
                token_limit=response['stop_reason'] == 'token_limit',
                unexpected_structure=response['stop_reason'] == 'unexpected_structure',
                clean_response=normal and not repeat and not empty,
                strict_task_success=None if exact is None else exact and normal and not repeat and not empty)


def summarize(rows):
    objective = [r for r in rows if not r['case']['qualitative']]
    def metrics(selected):
        n = len(selected)
        return dict(count=n, **{k:sum(bool(r['scores'][k]) for r in selected) for k in
                    ('answer_match','answer_match_with_neutral_wrapper','natural_stop','repeated','empty','token_limit','unexpected_structure','clean_response','strict_task_success')})
    categories = sorted({r['case']['category'] for r in objective})
    return dict(objective=metrics(objective), categories={c:metrics([r for r in objective if r['case']['category']==c]) for c in categories},
                response_control_all=metrics(rows),
                answerability={str(a):metrics([r for r in objective if r['case']['category']=='answerability' and r['case']['answerable']==a]) for a in (True,False)},
                caveat='Answer match is conservative automatic scoring, not a complete semantic correctness judgment. Non-matching answers and open chat require review. Repetition detection is heuristic.')
