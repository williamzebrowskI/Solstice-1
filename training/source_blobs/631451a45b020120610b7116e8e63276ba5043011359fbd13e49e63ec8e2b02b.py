"""Explicit open-ended constant LR and optional, separately forked final decay.

Historical recipe fields are retained verbatim as provenance. target_tokens is
not an active stopping condition after this transition; training_target is.
"""
import copy
import math

from .common import fingerprint

KEY = 'stable_schedule_transition'
DECAY_KEY = 'stable_cooldown_transition'
FORMAT = 'sml-v2-open-ended-stable-v1'
DECAY_FORMAT = 'sml-v2-stable-cooldown-v1'


def _parent(recipe, key):
    from .recipe import validate
    parent = copy.deepcopy(recipe)
    parent.pop(key)
    validate(parent)
    return parent


def _provenance(t, parent):
    contract = t.get('parent_contract')
    if (not isinstance(contract, dict) or set(contract) != {'code', 'recipe', 'data', 'tokenizer'}
            or fingerprint(parent) != contract['recipe']
            or not t.get('parent_replica_hash') or not t.get('validation_fingerprint')):
        raise ValueError('Stable schedule changed unrelated settings or lost parent provenance')
    for key in ('parent_step', 'start_tokens'):
        if type(t.get(key)) is not int or t[key] < 1:
            raise ValueError('Invalid stable schedule counter: ' + key)


def _rate(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value < 1


def validate_transition(recipe):
    if DECAY_KEY in recipe:
        t = recipe[DECAY_KEY]
        keys = {'format', 'parent_step', 'start_tokens', 'parent_contract', 'parent_replica_hash',
                'validation_fingerprint', 'duration_tokens', 'start_lr', 'final_lr', 'shape'}
        if not isinstance(t, dict) or set(t) != keys or t['format'] != DECAY_FORMAT:
            raise ValueError('Invalid stable cooldown transition')
        parent = _parent(recipe, DECAY_KEY)
        _provenance(t, parent)
        if KEY not in parent or parent[KEY]['warmup_tokens'] + parent[KEY]['start_tokens'] > t['start_tokens']:
            raise ValueError('Cooldown requires a completed stable LR warmup')
        if (type(t['duration_tokens']) is not int or t['duration_tokens'] < 1
                or t['shape'] != '1-sqrt' or not _rate(t['start_lr'])
                or type(t['final_lr']) not in (int, float) or not math.isfinite(t['final_lr'])
                or not 0 <= t['final_lr'] < t['start_lr']
                or t['start_lr'] != schedule_rate(t['start_tokens'], parent)):
            raise ValueError('Invalid cooldown duration, shape or learning rates')
        return
    t = recipe[KEY]
    keys = {'format', 'parent_step', 'start_tokens', 'parent_contract', 'parent_replica_hash',
            'validation_fingerprint', 'parent_scheduler', 'start_lr', 'stable_lr', 'warmup_tokens'}
    if not isinstance(t, dict) or set(t) != keys or t['format'] != FORMAT:
        raise ValueError('Invalid open-ended stable schedule transition')
    parent = _parent(recipe, KEY)
    _provenance(t, parent)
    if any(k in parent for k in ('lr_trial_transition', 'research_stage_b_transition', 'stage_b_transition')):
        raise ValueError('Open-ended schedule requires a regular pretraining checkpoint')
    if (not _rate(t['start_lr']) or not _rate(t['stable_lr'])
            or type(t['warmup_tokens']) is not int or t['warmup_tokens'] < 0
            or (t['stable_lr'] != t['start_lr'] and t['warmup_tokens'] < 1)):
        raise ValueError('Changing the stable LR requires an explicit nonzero ramp')
    from .lr_scheduler import restore_scheduler
    from .recipe import learning_rate
    metadata = dict(recipe=parent, step=t['parent_step'], tokens=t['start_tokens'],
                    lr_scheduler=t['parent_scheduler'],
                    best_val_loss=t['parent_scheduler']['best'] if t['parent_scheduler'] else None)
    scheduler = restore_scheduler(parent, metadata)
    base = learning_rate(t['start_tokens'], parent)
    if t['start_lr'] != (scheduler.rate(base) if scheduler else base):
        raise ValueError('Stable starting LR differs from the saved effective rate')


def _verified_parent(metadata):
    from .recipe import validate
    parent = validate(copy.deepcopy(metadata['recipe']))
    if metadata.get('sft') or fingerprint(parent) != metadata['v2_contract']['recipe']:
        raise ValueError('Stable schedule requires a verified pretraining checkpoint')
    if not metadata.get('validation_fingerprint') or not metadata.get('replica_hash'):
        raise ValueError('Checkpoint is missing validation or replica provenance')
    return parent


def stable_recipe(metadata, *, stable_lr=None, warmup_tokens=0):
    from .lr_scheduler import restore_scheduler
    from .recipe import learning_rate, validate
    parent = _verified_parent(metadata)
    if KEY in parent:
        raise ValueError('Already on the stable schedule; resume it without a new transition')
    scheduler = restore_scheduler(parent, metadata)
    base = learning_rate(metadata['tokens'], parent)
    start_lr = scheduler.rate(base) if scheduler else base
    parent[KEY] = dict(format=FORMAT, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']), parent_replica_hash=metadata['replica_hash'],
        validation_fingerprint=metadata['validation_fingerprint'],
        parent_scheduler=scheduler.state_dict() if scheduler else None,
        start_lr=start_lr, stable_lr=start_lr if stable_lr is None else stable_lr, warmup_tokens=warmup_tokens)
    return validate(parent)


def cooldown_recipe(metadata, *, duration_tokens, final_lr=0.):
    from .recipe import validate, tokens_per_update
    parent = _verified_parent(metadata)
    if KEY not in parent or DECAY_KEY in parent:
        raise ValueError('Fork cooldown from a stable checkpoint, never a cooled checkpoint')
    if type(duration_tokens) is not int or duration_tokens < 1:
        raise ValueError('Cooldown duration must be positive integer tokens')
    update = tokens_per_update(parent)
    duration = ((duration_tokens + update - 1) // update) * update
    parent[DECAY_KEY] = dict(format=DECAY_FORMAT, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']), parent_replica_hash=metadata['replica_hash'],
        validation_fingerprint=metadata['validation_fingerprint'], duration_tokens=duration,
        start_lr=schedule_rate(metadata['tokens'], parent), final_lr=final_lr, shape='1-sqrt')
    return validate(parent)


def schedule_rate(tokens, recipe):
    if DECAY_KEY in recipe:
        t = recipe[DECAY_KEY]
        p = min(1., max(0., (tokens - t['start_tokens']) / t['duration_tokens']))
        return t['final_lr'] + (t['start_lr'] - t['final_lr']) * (1. - math.sqrt(p))
    t = recipe[KEY]
    span = t['warmup_tokens']
    p = min(1., max(0., (tokens - t['start_tokens']) / span)) if span else 1.
    return t['start_lr'] + (t['stable_lr'] - t['start_lr']) * p


def training_target(recipe):
    if DECAY_KEY in recipe:
        t = recipe[DECAY_KEY]
        return t['start_tokens'] + t['duration_tokens']
    return None if KEY in recipe else recipe['target_tokens']


def effective_learning_rate(tokens, recipe, scheduler):
    from .recipe import learning_rate
    base = learning_rate(tokens, recipe)
    if KEY in recipe or 'research_stage_b_transition' in recipe:
        return base
    return scheduler.rate(base) if scheduler else base


def phase(tokens, recipe):
    if DECAY_KEY in recipe:
        return 'final-decay'
    t = recipe[KEY]
    return 'warmup' if tokens < t['start_tokens'] + t['warmup_tokens'] else 'stable'


def verify_transition(metadata, recipe):
    if DECAY_KEY in recipe and DECAY_KEY not in metadata['recipe']:
        t = recipe[DECAY_KEY]
        expected = cooldown_recipe(metadata, duration_tokens=t['duration_tokens'], final_lr=t['final_lr'])
        action = 'explicit-stable-cooldown-transition'
    else:
        t = recipe[KEY]
        expected = stable_recipe(metadata, stable_lr=t['stable_lr'], warmup_tokens=t['warmup_tokens'])
        action = 'explicit-open-ended-stable-transition'
    if recipe != expected:
        raise ValueError('Resume rejected: stable schedule changed unrelated settings or parent state')
    return action
