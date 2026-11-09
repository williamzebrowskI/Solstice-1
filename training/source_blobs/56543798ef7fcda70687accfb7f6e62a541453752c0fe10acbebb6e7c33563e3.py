"""Explicit budget extension; never reset weights, optimizer, or stream cursors."""

import copy
from pathlib import Path

from .common import file_sha256, fingerprint
from .recipe import learning_rate, validate

PRE_BATCH_CODE = '58f75d5d92576c268370e8f3dbb59f7cbc5ef58bcd1cf247a7cf6008c47e7e3c'
PILOT_CODE = '3af3f1a2405b7c447ee79cd1e9504ea9558d0f6636d7f9379f15cd216a7e4d58'
PRE_GATHER_CODE = 'fdaa5eceee48990f8e20adef266d9357348288e83f4c287f35256d5e31690d80'
PRE_PLATEAU_CODE = 'b1c40eb3a2639560d9ccec963d5295914bdbff5c0e0b8275ad6fd57b5482862f'
PRE_FIVE_RANK_CODE = '7ba6f57878f6ea5866c0193c1e90e24aa08218d8ee93effcd350f7fcd805ab9f'
PRE_LR_FLOOR_CODE = '407fbbde739e255344b5f622933189ac3c4ad6488f780cd0c8bba5d0baba9d5b'
PRE_DATA_REVIEW_CODE = '510c74f556a9a9505581d6ec800abb9f96ef8747c9fb3aae0a9194b2d4a28dcf'
PRE_LR_TRIAL_CODE = 'c20eebd79580c0d2c5b93b035750830422955d7c7022480ea67b83b616e40a54'
PRE_STAGE_B_CODE = '1659589be418196d424adf912d28c4b0f14ed210a16f367cf0b590f226d026ba'
PRE_ACCUMULATION_CODE = 'e65ef90906717d81b4f9f10820be1a6b167f535d0e09838bd8bdc3361adfe3a9'
PRE_REBALANCE_CODE = '2c7e9a69271f9817017c2371ae5e874c9229da489b6c0128aadda943c109201d'
PRE_RESEARCH_STAGE_B_CODE = 'db42e945868bbf75a15d49a3fd95692493d1e1262b465834779ce40ee9f0e59a'
PRE_RESEARCH_PILOT_CODE = 'c9d474a611ff1c2d78aaafaa7a0547e94f2b16126850cf9b2a0a3ffca79037ba'
PRE_OPEN_ENDED_CODE = '6e444060dba8c63ba33e128a43da03df6d927d289318ec51a2c0026fb494ab9f'
PRE_ALL_AVAILABLE_CODE = '680fdd3a84c6cae314e9d9038ca00e1f2712284e255d5d8d66599984f57ebb6c'
PRE_GRADUAL_CODE = '1bc7d03c972319212ea775d4667fc0d3f5c0d2d738f3bfff4967c0093913df00'
# Pins the reviewed upgrade, including opt-in streaming prose filters and curated token pools,
# excluding this module to avoid a self-hash cycle.
UPGRADE_MODULES = 'd27ea49d38667900ba2fe99a728d771967b489f9c9b5c93162a31122e35d3740'
CODE_PHASE_PARENT = 'b7ff0ae2c1f095bd36300267f5dc2fd60d58be71d828a1b85a796ddf176e942e'
CODE_PHASE_MODULES = '83898e87cfa33607868c4512dd014e13a4e8bd2665828b7c49635827c652b68a'


def reviewed_upgrade(previous, current):
    if previous == CODE_PHASE_PARENT:
        hashes = {p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
        return (fingerprint(hashes) == current and
                fingerprint({k:v for k,v in hashes.items() if k != 'continuation.py'}) == CODE_PHASE_MODULES)
    if previous not in (PRE_BATCH_CODE, PILOT_CODE, PRE_GATHER_CODE, PRE_PLATEAU_CODE, PRE_FIVE_RANK_CODE, PRE_LR_FLOOR_CODE, PRE_DATA_REVIEW_CODE, PRE_LR_TRIAL_CODE, PRE_STAGE_B_CODE, PRE_ACCUMULATION_CODE, PRE_REBALANCE_CODE, PRE_RESEARCH_STAGE_B_CODE, PRE_RESEARCH_PILOT_CODE, PRE_OPEN_ENDED_CODE, PRE_ALL_AVAILABLE_CODE, PRE_GRADUAL_CODE):
        return False
    hashes = {p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    return (fingerprint(hashes) == current and
            fingerprint({k: v for k, v in hashes.items() if k != 'continuation.py'}) == UPGRADE_MODULES)


def extend_recipe(metadata, target_tokens, rewarm_tokens=50_000_000):
    previous = validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Source checkpoint recipe fingerprint mismatch')
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError('Continuation target must be a positive total token budget')
    if 'continuation' in previous:
        if target_tokens != previous['target_tokens']:
            raise ValueError('Existing continuation has a different target; refusing to change its schedule')
        return previous
    if target_tokens <= max(previous['target_tokens'], metadata['tokens']):
        raise ValueError('Continuation must extend the source token budget')
    recipe = copy.deepcopy(previous)
    recipe['name'] = previous['name'] + f'-continued-to-{target_tokens}'
    recipe['target_tokens'] = target_tokens
    recipe['continuation'] = dict(
        format='sml-v2-continuation-v1', start_tokens=metadata['tokens'],
        start_lr=learning_rate(metadata['tokens'], previous), rewarm_tokens=rewarm_tokens,
        parent_step=metadata['step'], parent_contract=copy.deepcopy(metadata['v2_contract']),
        parent_replica_hash=metadata['replica_hash'])
    return validate(recipe)


def check_resume(metadata, contract, recipe, manifest=None):
    """Allow only the reviewed code upgrade and an exactly derived extension."""
    previous = metadata.get('v2_contract', {})
    if fingerprint(metadata['recipe']) != previous.get('recipe'):
        raise ValueError('Resume rejected: source recipe fingerprint mismatch')
    if fingerprint(recipe) != contract.get('recipe'):
        raise ValueError('Resume rejected: requested recipe fingerprint mismatch')
    if previous == contract:
        return 'unchanged'
    if previous.get('tokenizer') != contract.get('tokenizer'):
        raise ValueError('Resume rejected: data or tokenizer changed')
    if previous.get('code') != contract.get('code') and not reviewed_upgrade(previous.get('code'), contract.get('code')):
        raise ValueError('Resume rejected: unreviewed code change')
    if 'code_data_transition' in recipe and 'code_data_transition' not in metadata['recipe']:
        from .code_phase import verify_transition
        verify_transition(metadata, manifest, contract, recipe)
        return 'explicit-code-data-transition'
    if 'research_stage_b_transition' in recipe and 'research_stage_b_transition' not in metadata['recipe']:
        from .research_stage_b import verify_manifest
        verify_manifest(metadata, manifest, contract, recipe)
        return 'explicit-research-stage-b-transition'
    if 'stage_b_transition' in recipe and 'stage_b_transition' not in metadata['recipe']:
        from .stage_b import verify_manifest
        verify_manifest(metadata, manifest, contract, recipe)
        return 'explicit-stage-b-transition'
    if 'data_review_transition' in recipe and 'data_review_transition' not in metadata['recipe']:
        from .data_review import reviewed_recipe, verify_manifest
        expected = reviewed_recipe(metadata, recipe['stream']['shuffle_buffer'], recipe['eval_batches_per_source'])
        if recipe != expected:
            raise ValueError('Resume rejected: data review changed unrelated settings')
        verify_manifest(metadata, manifest, contract)
        if manifest['settings'] != recipe['stream']:
            raise ValueError('Resume rejected: data review manifest differs from the recipe')
        return 'explicit-data-review-transition'
    if (('stable_schedule_transition' in recipe and 'stable_schedule_transition' not in metadata['recipe'])
            or ('stable_cooldown_transition' in recipe and 'stable_cooldown_transition' not in metadata['recipe'])
            or ('gradual_decay_transition' in recipe and 'gradual_decay_transition' not in metadata['recipe'])):
        from .stable_schedule import verify_transition
        if ('parent_stream' not in recipe['stable_schedule_transition']
                and previous.get('data') != contract.get('data')):
            raise ValueError('Resume rejected: data or tokenizer changed')
        if (any(k in recipe for k in ('stable_cooldown_transition','gradual_decay_transition'))
                and previous.get('data') != contract.get('data')):
            raise ValueError('Resume rejected: data or tokenizer changed')
        return verify_transition(metadata, recipe, manifest, contract)
    if previous.get('data') != contract.get('data'):
        raise ValueError('Resume rejected: data or tokenizer changed')
    if 'batch_rebalance_transition' in recipe and 'batch_rebalance_transition' not in metadata['recipe']:
        from .batch_transition import rebalance_batches
        if recipe != rebalance_batches(metadata, recipe['batches'], recipe['cache_gib_by_mac']):
            raise ValueError('Resume rejected: rebalance changed unrelated settings')
        return 'explicit-batch-rebalance-transition'
    if 'batch_transition' in recipe and 'batch_transition' not in metadata['recipe']:
        from .batch_transition import change_batches
        if recipe != change_batches(metadata, recipe['batches']):
            raise ValueError('Resume rejected: batch transition changed unrelated settings')
        return 'explicit-batch-transition'
    if 'lr_trial_transition' in recipe and 'lr_trial_transition' not in metadata['recipe']:
        from .lr_trial import trial_recipe
        t = recipe['lr_trial_transition']
        expected = trial_recipe(metadata, t['arm'], target_lr=t['target_lr'],
                                warmup_steps=t['warmup_steps'], trial_steps=t['trial_steps'])
        if recipe != expected:
            raise ValueError('Resume rejected: LR trial changed unrelated settings')
        return 'explicit-lr-trial-transition'
    if 'topology_transition' in recipe and 'topology_transition' not in metadata['recipe']:
        from .topology_transition import migrate_recipe
        expected = migrate_recipe(metadata, recipe['batches'], recipe['topology_transition']['physical_order'])
        if recipe != expected:
            raise ValueError('Resume rejected: topology migration changed unrelated settings')
        return 'explicit-five-rank-transition'
    if previous['recipe'] == contract['recipe']:
        if previous.get('code') == PRE_REBALANCE_CODE:
            return 'reviewed-batch-cache-code-upgrade'
        if previous.get('code') == PRE_ACCUMULATION_CODE:
            return 'reviewed-sequential-accumulation-code-upgrade'
        if previous.get('code') == PRE_STAGE_B_CODE:
            return 'reviewed-stage-b-code-upgrade'
        if previous.get('code') == PRE_LR_TRIAL_CODE:
            return 'reviewed-lr-trial-code-upgrade'
        if previous.get('code') == PRE_DATA_REVIEW_CODE:
            return 'reviewed-data-review-code-upgrade'
        if previous.get('code') == PRE_LR_FLOOR_CODE:
            return 'reviewed-lr-floor-code-upgrade'
        if previous.get('code') == PRE_FIVE_RANK_CODE:
            return 'reviewed-five-rank-code-upgrade'
        if previous.get('code') == PRE_PLATEAU_CODE:
            return 'reviewed-scheduler-code-upgrade'
        return ('reviewed-control-code-upgrade' if previous.get('code') == PRE_GATHER_CODE
                else 'reviewed-pilot-code-upgrade')
    if 'lr_floor_transition' in recipe and 'lr_floor_transition' not in metadata['recipe']:
        from .lr_scheduler import lower_floor_recipe
        expected = lower_floor_recipe(metadata, recipe['plateau_scheduler']['min_lr'])
        if recipe != expected:
            raise ValueError('Resume rejected: LR floor transition changed unrelated settings')
        return 'explicit-lr-floor-transition'
    if 'plateau_scheduler' in recipe and 'plateau_scheduler' not in metadata['recipe']:
        from .lr_scheduler import adaptive_recipe
        expected = adaptive_recipe(metadata, recipe['plateau_scheduler'])
        if recipe != expected:
            raise ValueError('Resume rejected: scheduler transition changed unrelated settings')
        return 'explicit-validation-scheduler-transition'
    extension = recipe.get('continuation')
    if not extension or 'continuation' in metadata['recipe']:
        raise ValueError('Resume rejected: settings changed without an explicit budget extension')
    expected = extend_recipe(metadata, recipe['target_tokens'], extension['rewarm_tokens'])
    if recipe != expected:
        raise ValueError('Resume rejected: extension changed settings beyond the budget and LR ramp')
    return 'explicit-token-budget-extension'
