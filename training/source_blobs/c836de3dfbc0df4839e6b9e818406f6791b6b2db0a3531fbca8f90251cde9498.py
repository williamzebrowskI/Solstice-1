"""Baseline-relative behavioral gates; no public benchmark score selects weights."""
from collections import Counter
import re


def canonical(text):return ' '.join(re.findall(r'\w+',text.casefold()))


def looping(text):
    tokens=canonical(text).split()
    if len(tokens)>=24:
        grams=[tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        if len(set(grams))/len(grams)<.68:return True
    lines=[canonical(s) for s in text.splitlines() if len(canonical(s).split())>=4]
    if lines and max(Counter(lines).values())>=3:return True
    sentences=[canonical(s) for s in re.split(r'[.!?]+',text) if len(canonical(s).split())>=4]
    if sentences and max(Counter(sentences).values())>=3:return True
    return False


def unchanged_edit_claim(text):
    pattern=r'(?:changed|replaced)\s+["\u201c\u2018\x27]([^"\u201d\u2019\x27]+)["\u201d\u2019\x27]\s+(?:to|with)\s+["\u201c\u2018\x27]([^"\u201d\u2019\x27]+)["\u201d\u2019\x27]'
    return any(canonical(a)==canonical(b) and len(canonical(a).split())>=2 for a,b in re.findall(pattern,text,re.I))


def eligibility(result,baseline,plan):
    failures=[]
    if result['retention']['val_loss']>baseline['retention']['val_loss']+plan['max_retention_loss_increase']:
        failures.append('prose_retention')
    if result['knowledge']['correct']<baseline['knowledge']['correct']-plan['max_knowledge_drop_cases']:
        failures.append('knowledge_accuracy')
    if plan.get('max_knowledge_drop_per_task') is not None:
        for task,previous in baseline['knowledge']['tasks'].items():
            if result['knowledge']['tasks'][task]['correct']<previous['correct']-plan['max_knowledge_drop_per_task']:
                failures.append('knowledge_'+task)
    for key in ('cap_hits','repetition_hits','invalid_stop_hits','meaningless_edit_hits'):
        if result['behavior'][key]>baseline['behavior'][key]:failures.append(key)
    if result['generated_macro_pass']+1e-12<baseline['generated_macro_pass']:
        failures.append('task_completion')
    if plan.get('min_task_gain_cases',0) and result['checked_passed']<baseline['checked_passed']+plan['min_task_gain_cases']:
        failures.append('no_task_gain')
    return dict(eligible=not failures,failures=failures)


def selection(result,baseline,plan):
    allowed=eligibility(result,baseline,plan)['eligible']
    return [int(allowed),result['generated_macro_pass'],-result['behavior']['repetition_hits'],
            -result['behavior']['cap_hits'],-result['assistant_nll']]


def compact_baseline(result):
    return {key:result[key] for key in ('retention','knowledge','behavior','generated_macro_pass','checked_passed','checked_total') if key in result}


def fresh_progress(result,baseline,plan,has_candidate,failed_count):
    """Fresh pretraining weights must learn chat before behavioral regression exists.

    Foundation failures count immediately. Before any candidate meets Part 1,
    inferior chat is logged and cannot be selected, but is not an early-stop
    reason. Once a candidate qualifies, all regression gates are active.
    """
    checks=eligibility(result,baseline,plan)
    foundation=any(f=='prose_retention' or f.startswith('knowledge_') for f in checks['failures'])
    counted=not checks['eligible'] and (has_candidate or foundation)
    consecutive=failed_count+1 if counted else 0
    return dict(**checks,consecutive_failures=consecutive,quality_stop=consecutive>=plan['regression_patience'])
