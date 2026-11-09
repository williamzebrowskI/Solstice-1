"""Frozen protocol for five independent text-chat repair experiments."""
from pathlib import Path
from sml_v1.common import read_json, file_sha256, fingerprint
from .lora_search_common import setup, modules, ASSETS, PARENT

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pretraining/scripts').is_dir())
SHARED = ROOT/'pretraining/prepared/chat_repair_search_20261109'
REPORT = ROOT/'reports/sft_search/chat_repair_search_20261109'
ATTEMPT5 = ROOT/'pretraining/runs/deep44_parent2000_generalize_05_replica_20261109/step_0000256_6074b98fe66b'
PARENTS = {'parent2000': PARENT, 'attempt5': ATTEMPT5}
DOMAINS = ('grounded', 'abstention', 'dialogue', 'constraints', 'reasoning')
SPECS = [
 dict(name='Extraction and appropriate abstention', parent='attempt5', mixture=dict(grounded=.4, abstention=.4, dialogue=.2)),
 dict(name='Matched repair from parent2000', parent='parent2000', mixture=dict(grounded=.4, abstention=.4, dialogue=.2)),
 dict(name='Practical combined instructions', parent='attempt5', mixture=dict(constraints=.65, dialogue=.15, grounded=.1, abstention=.1)),
 dict(name='Short checked reasoning', parent='attempt5', mixture=dict(reasoning=.6, grounded=.15, dialogue=.15, abstention=.1)),
 dict(name='Development-selected combination', parent=None, mixture=None),
]
REPLAY_MIX = dict(general=.1875, knowledge=.1875, grounded=.125, dialogue=.125,
                  skills=.125, constraints=.125, social=.0625, rewrite=.0625)


def receipt():
    value = read_json(SHARED/'data/manifest.json')
    if value['format'] != 'chat-repair-data-v1':
        raise ValueError('Unexpected repair data')
    for name, sha in value['files'].items():
        if file_sha256(SHARED/'data'/name) != sha:
            raise ValueError('Repair data changed: '+name)
    for name, path in PARENTS.items():
        if file_sha256(path/'manifest.json') != value['parents'][name]:
            raise ValueError('Protected starting checkpoint changed')
    return value


def compact(value):
    return {k:v for k,v in value.items() if k != 'samples'}


def correct(value):
    return sum(value['generated_pass_counts'].values())


def baseline(name, split='dev'):
    p = REPORT/'baselines'/name/(split+'.json')
    value = read_json(p)
    expected = dict(parent=file_sha256(PARENTS[name]/'manifest.json'),
                    data=file_sha256(SHARED/'data'/(split+'.jsonl')),
                    evaluator=file_sha256(SHARED/'runtime/sft/lora_repair_eval.py'))
    if value['identity'] != expected:
        raise ValueError('New baseline identity differs')
    return value


def legacy(name, split='dev'):
    manifest = read_json(REPORT/'baselines/legacy_manifest.json')
    p = REPORT/'baselines'/(name+'_legacy_'+split+'.json')
    if file_sha256(p) != manifest['files'][p.name]:
        raise ValueError('Historical baseline changed')
    return read_json(p)


def selection(new, old):
    """Predeclared dev-only ranking; tolerances allow small count fluctuations."""
    previous = [baseline(n) for n in PARENTS]
    historical = [legacy(n) for n in PARENTS]
    reasons = []
    if correct(old) < max(map(correct, historical))-6:
        reasons.append('historical total fell over 6/320')
    for domain in historical[0]['generated_pass_counts']:
        floor = max(v['generated_pass_counts'][domain] for v in historical)-3
        if old['generated_pass_counts'][domain] < floor:
            reasons.append('historical '+domain+' fell over 3 cases')
    if old['retention']['val_loss'] > min(v['retention']['val_loss'] for v in historical)+.003:
        reasons.append('prose retention')
    if old['natural_stop'] < max(v['natural_stop'] for v in historical)-.01:
        reasons.append('historical stopping')
    if old['repetition_rate'] > min(v['repetition_rate'] for v in historical)+.01:
        reasons.append('historical repetition')
    if new['natural_stop'] < max(v['natural_stop'] for v in previous)-.01:
        reasons.append('new stopping')
    if new['repetition_rate'] > min(v['repetition_rate'] for v in previous)+.01 or new['empty']:
        reasons.append('new repetition or empty answers')
    if new['strict_correct'] < max(v['strict_correct'] for v in previous)-3:
        reasons.append('new strict instruction compliance fell over 3/300')
    # Answerable questions must remain answerable after learning to abstain.
    if new['counts']['grounded'] < max(v['counts']['grounded'] for v in previous)-2:
        reasons.append('unnecessary abstention or extraction regression')
    key = [int(not reasons), new['correct'], correct(old), new['natural_stop'],
           -new['repetition_rate'], -old['assistant_nll']]
    return dict(eligible=not reasons, reasons=reasons, key=key)


def severe(new, old):
    historical = [legacy(n) for n in PARENTS]
    return (correct(old) < min(map(correct, historical))-12 or
            old['retention']['val_loss'] > min(v['retention']['val_loss'] for v in historical)+.008 or
            new['natural_stop'] < .9 or new['repetition_rate'] > .1)
