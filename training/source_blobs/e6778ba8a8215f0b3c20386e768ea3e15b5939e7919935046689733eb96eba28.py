"""Explicit microbatch change with preserved checkpoint and token schedule."""
import copy
import math
from .common import fingerprint
FORMAT='sml-research-batch-transition-v1'
REBALANCE_FORMAT='sml-research-batch-rebalance-v1'


def validate_cache_policy(recipe):
    values = recipe.get('cache_gib_by_mac')
    if (not isinstance(values, list) or len(values) != len(recipe['batches']) or
            any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 32 for v in values)):
        raise ValueError('Require finite cache limits in physical Mac order, 0 < GiB <= 32')


def cache_limit_for_rank(recipe, rank, fallback=4):
    """The five-rank ring visits physical Macs 1, 2, 3, 5, 4."""
    if 'cache_gib_by_mac' not in recipe:
        return fallback
    validate_cache_policy(recipe)
    physical = [0, 1, 2, 4, 3][rank] if len(recipe['batches']) == 5 else rank
    return recipe['cache_gib_by_mac'][physical]


def validate_rebalance(recipe):
    t = recipe['batch_rebalance_transition']
    keys = {'format', 'parent_step', 'start_tokens', 'parent_contract',
            'parent_replica_hash', 'parent_batches'}
    if not isinstance(t, dict) or set(t) != keys or t['format'] != REBALANCE_FORMAT:
        raise ValueError('Invalid batch rebalance transition')
    old, new = t['parent_batches'], recipe['batches']
    if (not isinstance(old, list) or len(old) != 5 or len(new) != 5 or old == new or
            any(type(b) is not int or b < 1 for b in old + new) or sum(old) != sum(new)):
        raise ValueError('Rebalance must preserve the global batch across five ranks')
    if any(type(t[k]) is not int or t[k] <= 0 for k in ('parent_step', 'start_tokens')):
        raise ValueError('Invalid rebalance counters')
    if set(t['parent_contract']) != {'code', 'recipe', 'data', 'tokenizer'}:
        raise ValueError('Missing rebalance parent contract')
    if not isinstance(t['parent_replica_hash'], str) or len(t['parent_replica_hash']) != 64:
        raise ValueError('Missing rebalance parent replica hash')
    validate_cache_policy(recipe)


def rebalance_batches(metadata, batches, cache_gib_by_mac):
    """Explicitly reviewed redistribution; retain earlier transition provenance."""
    from .recipe import validate
    previous = validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Parent recipe fingerprint differs')
    if 'batch_rebalance_transition' in previous:
        if batches != previous['batches'] or cache_gib_by_mac != previous['cache_gib_by_mac']:
            raise ValueError('Further rebalance requires separate review')
        return previous
    if not 0 < metadata['tokens'] < previous['target_tokens']:
        raise ValueError('Require an unfinished trained checkpoint')
    r = copy.deepcopy(previous)
    r['batches'] = copy.deepcopy(batches)
    r['cache_gib_by_mac'] = copy.deepcopy(cache_gib_by_mac)
    r['batch_rebalance_transition'] = dict(format=REBALANCE_FORMAT,
        parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']),
        parent_replica_hash=metadata['replica_hash'], parent_batches=previous['batches'].copy())
    return validate(r)


def validate_transition(recipe):
    t=recipe['batch_transition']
    keys={'format','parent_step','start_tokens','parent_contract','parent_replica_hash','parent_batches'}
    if not isinstance(t,dict) or set(t)!=keys or t['format']!=FORMAT:
        raise ValueError('Invalid batch transition')
    old=t['parent_batches'];new=recipe['batches']
    if (not isinstance(old,list) or len(old)!=5 or len(new)!=5
            or any(type(b)is not int or b<1 for b in old) or old==new):
        raise ValueError('Batch transition requires changed positive five-rank batches')
    if any(type(t[k])is not int or t[k]<=0 for k in ('parent_step','start_tokens')):
        raise ValueError('Invalid batch transition counters')
    if set(t['parent_contract'])!={'code','recipe','data','tokenizer'}:
        raise ValueError('Missing batch parent contract')
    if not isinstance(t['parent_replica_hash'],str) or len(t['parent_replica_hash'])!=64:
        raise ValueError('Missing parent replica hash')


def change_batches(metadata,batches):
    from .recipe import validate
    previous=validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous)!=metadata['v2_contract']['recipe']:
        raise ValueError('Parent recipe fingerprint differs')
    if (not isinstance(batches,list) or len(batches)!=5 or len(previous['batches'])!=5
            or any(type(b)is not int or b<1 for b in batches)):
        raise ValueError('Require five positive microbatches')
    if 'batch_transition' in previous:
        if batches!=previous['batches']:raise ValueError('Further batch transition requires separate review')
        return previous
    if batches==previous['batches'] or not 0<metadata['tokens']<previous['target_tokens']:
        raise ValueError('Require changed batches and an unfinished trained checkpoint')
    r=copy.deepcopy(previous);r['batches']=batches.copy()
    r['batch_transition']=dict(format=FORMAT,parent_step=metadata['step'],start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']),parent_replica_hash=metadata['replica_hash'],
        parent_batches=previous['batches'].copy())
    return validate(r)
