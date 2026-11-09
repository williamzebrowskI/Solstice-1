"""Explicit answer checks and retention-aware selection for text-only SFT."""
import re
import string


def answer_pass(text,verification):
    if verification['kind']=='rules':
        from .verified_rules import check_rules
        return check_rules(text,verification['rules'])
    if verification['kind']=='verbatim_format':
        # Newlines, case, quotation marks and punctuation are part of the task.
        return text.strip() in {a.strip() for a in verification['answers']}
    if verification['kind']=='span':
        def normalize(value):
            value=''.join(c for c in value.casefold() if c not in string.punctuation)
            return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())
    else:
        def normalize(value):
            value=re.sub(r'\s+',' ',value.strip().strip('"').strip("'").rstrip('.!?'))
            return value if verification.get('case_sensitive',False) else value.casefold()
    return normalize(text) in {normalize(a) for a in verification['answers']}


def eligibility(result,baseline,plan):
    foundation=[];behavior=[]
    if result['retention']['val_loss']>baseline['retention']['val_loss']+plan['max_retention_loss_increase']:foundation.append('prose retention')
    for component,rate in result['generated_rates'].items():
        if rate+1e-12<baseline['generated_rates'][component]:behavior.append(component+' checked answers')
    if result['natural_stop']+1e-12<max(baseline['natural_stop'],plan.get('min_natural_stop',0)):
        behavior.append('natural stopping')
    if result['repetition_rate']>min(baseline['repetition_rate'],plan.get('max_repetition_rate',1))+1e-12:
        behavior.append('repetition')
    if result.get('empty_rate',0)>0:behavior.append('empty answers')
    for component,minimum in plan.get('min_domain_pass_cases',{}).items():
        if result['generated_pass_counts'][component]<minimum:behavior.append(component+' minimum checks')
    gain=sum(result['generated_pass_counts'].values())-sum(baseline['generated_pass_counts'].values())
    return dict(eligible=not foundation and not behavior and gain>=plan.get('min_task_gain_cases',1),foundation_failures=foundation,
                failures=foundation+behavior,additional_checked_successes=gain)


def selection(result,baseline,plan):
    check=eligibility(result,baseline,plan)
    return [int(check['eligible']),result['generated_macro_pass'],result['natural_stop'],-result['repetition_rate'],-result['assistant_nll']]
