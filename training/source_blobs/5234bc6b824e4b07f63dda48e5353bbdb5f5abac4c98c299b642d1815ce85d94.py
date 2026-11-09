"""Bounded data-mixture pilots for the 176M run, with fixed reference validation.

This is separate from the historical prose-only Stage B transition. It retains
code/math, model geometry, optimizer state, and the parent's scheduler history.
Only the pilot's data allocation and explicit learning-rate policy are new.
"""
import copy
import math
import re
from collections import Counter, OrderedDict

from .common import fingerprint
from .corpus import accepted, content_key, partition
from .stage_b_data import document_keys
from .stream import StreamingTokens, stream_manifest

KEY = 'research_stage_b_transition'
FORMAT = 'sml-research-stage-b-v1'
FULL_LR_FORMATS = ('rewarm-cosine-to-target-v1', 'parent-cosine-to-target-v1')
NOISE = re.compile(r'page (?:was )?not found|page you requested does not exist|'
    r'what is the duplicate article|why is this article offensive|'
    r'did you find what you were looking for|enter your email address|'
    r'users browsing this forum|accept all cookies|enable javascript', re.I)


def select_novel(row, source, config, tokenizer):
    """Explicit extra quality screen for the new source only; retain code/math."""
    text = accepted(row, source, config)
    if text is None: return None, 'source_filter'
    score = float(row.get('quality_score', float('nan')))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid HQ quality score')
    if row.get('language') != 'en': return None, 'non_english'
    policy = source['pilot_filter']
    url = str(row.get('url', '')).lower()
    if any(part in url for part in policy['exclude_url_parts']): return None, 'excluded_url_type'
    if NOISE.search(text): return None, 'boilerplate'
    ids = tokenizer.encode(text)
    if not policy['min_tokens'] <= len(ids) <= policy['max_tokens']:
        return None, 'token_length'
    return text, None


def training_manifest(corpus, tokenizer, settings):
    reference = stream_manifest(corpus['reference_corpus'], tokenizer, settings)
    training = copy.deepcopy(corpus)
    training.pop('reference_corpus')
    result = stream_manifest(training, tokenizer, settings)
    result.update(format=FORMAT, reference=reference)
    old = {s['label']: s for s in reference['corpus']['sources']}
    new = {s['label']: s for s in training['sources']}
    if not old.keys() <= new.keys():
        raise ValueError('Stage B must retain all reference sources')
    for label, source in old.items():
        if new[label] != dict(source, weight=new[label]['weight']):
            raise ValueError('Existing source changed beyond its allocation: ' + label)
    for key, value in reference['corpus'].items():
        if key not in ('sources', 'name') and training.get(key) != value:
            raise ValueError('Stage B changed corpus filtering/packing: ' + key)
    return result


def validate_transition(recipe):
    from .recipe import learning_rate, tokens_per_update, validate
    from .lr_scheduler import restore_scheduler
    t = recipe[KEY]
    keys = {'format', 'arm', 'parent_step', 'start_tokens', 'parent_contract',
            'parent_replica_hash', 'parent_scheduler', 'validation_fingerprint',
            'baseline_loss', 'manifest', 'token_budget', 'trial_steps', 'learning_rate'}
    if not isinstance(t, dict) or set(t) - {'lr_policy', 'diagnostic'} != keys or t['format'] != FORMAT:
        raise ValueError('Invalid research Stage B transition')
    parent = copy.deepcopy(recipe); parent.pop(KEY)
    validate(parent)
    if any(k in parent for k in ('stage_b_transition', 'lr_trial_transition')):
        raise ValueError('Cannot stack unrelated historical experiments')
    if (set(t['parent_contract']) != {'code', 'data', 'recipe', 'tokenizer'}
            or fingerprint(parent) != t['parent_contract']['recipe']
            or not t['parent_replica_hash'] or not t['manifest']
            or not t['validation_fingerprint'] or t['arm'] not in ('control', 'hq')):
        raise ValueError('Stage B lost its exact parent provenance')
    for key in ('parent_step', 'start_tokens', 'token_budget', 'trial_steps'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError('Invalid Stage B counter: ' + key)
    full_run = t.get('lr_policy',{}).get('format') in FULL_LR_FORMATS
    end_tokens = t['start_tokens'] + t['trial_steps'] * tokens_per_update(parent)
    budget_ok = (t['token_budget'] == parent['target_tokens']-t['start_tokens']
                 if full_run else end_tokens < parent['target_tokens'])
    if (not math.isfinite(t['baseline_loss']) or t['baseline_loss'] <= 0
            or t['trial_steps'] != math.ceil(t['token_budget'] / tokens_per_update(parent))
            or not budget_ok):
        raise ValueError('Invalid Stage B baseline or bounded budget')
    meta = dict(recipe=parent, step=t['parent_step'], tokens=t['start_tokens'],
                best_val_loss=t['baseline_loss'], lr_scheduler=t['parent_scheduler'])
    scheduler = restore_scheduler(parent, meta)
    if scheduler is None:
        raise ValueError('Stage B requires saved parent scheduler state')
    state = scheduler.state_dict()
    if (state['last_step'], state['last_tokens'], state['last_loss']) != (
            t['parent_step'], t['start_tokens'], t['baseline_loss']):
        raise ValueError('Stage B must start from an evaluated best checkpoint')
    expected = scheduler.rate(learning_rate(t['start_tokens'], parent))
    if not math.isfinite(t['learning_rate']) or t['learning_rate'] != expected:
        raise ValueError('Stage B must retain the parent effective learning rate')
    validate_lr_policy(t.get('lr_policy'), t['learning_rate'], parent['peak_lr'], t['trial_steps'],
                       t['start_tokens'], tokens_per_update(parent), parent['target_tokens'])
    if 'diagnostic' in t:
        from .diagnostic_eval import validate_policy
        validate_policy(t['diagnostic'], parent)


def validate_lr_policy(policy, start_lr, maximum_lr, trial_steps, start_tokens, update_tokens, target_tokens):
    if policy is None: return
    if isinstance(policy, dict) and policy.get('format') == 'parent-cosine-to-target-v1':
        if (set(policy) != {'format', 'final_lr'}
                or not isinstance(policy['final_lr'], (int, float))
                or not math.isfinite(policy['final_lr'])
                or not 0 < policy['final_lr'] < start_lr):
            raise ValueError('Invalid Stage B parent-rate cosine policy')
        return
    production = isinstance(policy,dict) and policy.get('format') == 'rewarm-cosine-to-target-v1'
    keys = {'format','target_lr','warmup_steps'}
    if production: keys |= {'decay_start_tokens','final_lr'}
    if (not isinstance(policy, dict)
            or set(policy) != keys
            or policy['format'] not in ('linear-rewarm-hold-v1','rewarm-cosine-to-target-v1')
            or type(policy['warmup_steps']) is not int
            or not 1 <= policy['warmup_steps'] < trial_steps
            or not isinstance(policy['target_lr'], (float, int))
            or not math.isfinite(policy['target_lr'])
            or not start_lr < policy['target_lr'] <= maximum_lr):
        raise ValueError('Invalid Stage B rewarming policy')
    if production and (type(policy['decay_start_tokens']) is not int
            or not start_tokens+policy['warmup_steps']*update_tokens < policy['decay_start_tokens'] < target_tokens
            or not isinstance(policy['final_lr'],(float,int)) or not math.isfinite(policy['final_lr'])
            or not 0 < policy['final_lr'] <= start_lr):
        raise ValueError('Invalid full Stage B final decay')


def stage_learning_rate(tokens, recipe):
    """Pure token-based schedule, identical after a checkpoint resume."""
    from .recipe import tokens_per_update
    t = recipe[KEY]
    policy = t.get('lr_policy')
    if policy is None: return t['learning_rate']
    if policy['format'] == 'parent-cosine-to-target-v1':
        progress = min(1., max(0., (tokens-t['start_tokens']) /
                              (recipe['target_tokens']-t['start_tokens'])))
        return policy['final_lr'] + .5*(t['learning_rate']-policy['final_lr'])*(1+math.cos(math.pi*progress))
    warm_end = t['start_tokens'] + policy['warmup_steps'] * tokens_per_update(recipe)
    if tokens < warm_end:
        fraction = max(0., (tokens-t['start_tokens']) / (warm_end-t['start_tokens']))
        return t['learning_rate'] + (policy['target_lr']-t['learning_rate']) * fraction
    if policy['format'] == 'linear-rewarm-hold-v1' or tokens <= policy['decay_start_tokens']:
        return policy['target_lr']
    progress = min(1., (tokens-policy['decay_start_tokens']) /
                      (recipe['target_tokens']-policy['decay_start_tokens']))
    return policy['final_lr'] + .5*(policy['target_lr']-policy['final_lr'])*(1+math.cos(math.pi*progress))


def stage_recipe(metadata, manifest, arm, token_budget, lr_policy=None, diagnostic=None):
    from .recipe import learning_rate, tokens_per_update, validate
    from .lr_scheduler import restore_scheduler
    previous = validate(copy.deepcopy(metadata['recipe']))
    if metadata.get('sft') or fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Stage B requires a verified pretraining checkpoint')
    if KEY in previous:
        t = previous[KEY]
        if (t['manifest'], t['arm'], t['token_budget']) != (fingerprint(manifest), arm, token_budget):
            raise ValueError('Saved Stage B mixture/arm/budget cannot be changed on resume')
        if t.get('lr_policy') != lr_policy:
            raise ValueError('Saved Stage B learning-rate policy cannot be changed on resume')
        if t.get('diagnostic') != diagnostic:
            raise ValueError('Saved Stage B diagnostic cannot be changed on resume')
        return previous
    if fingerprint(manifest['reference']) != metadata['v2_contract']['data']:
        raise ValueError('Stage B reference data differs from parent')
    scheduler = restore_scheduler(previous, metadata)
    if scheduler is None:
        raise ValueError('Parent scheduler is missing')
    previous[KEY] = dict(format=FORMAT, arm=arm, parent_step=metadata['step'],
        start_tokens=metadata['tokens'], parent_contract=copy.deepcopy(metadata['v2_contract']),
        parent_replica_hash=metadata['replica_hash'], parent_scheduler=scheduler.state_dict(),
        validation_fingerprint=metadata['validation_fingerprint'], baseline_loss=metadata['best_val_loss'],
        manifest=fingerprint(manifest), token_budget=token_budget,
        trial_steps=math.ceil(token_budget / tokens_per_update(previous)),
        learning_rate=scheduler.rate(learning_rate(metadata['tokens'], metadata['recipe'])))
    if lr_policy is not None:
        previous[KEY]['lr_policy'] = copy.deepcopy(lr_policy)
    if diagnostic is not None:
        previous[KEY]['diagnostic'] = copy.deepcopy(diagnostic)
    return validate(previous)


def verify_manifest(metadata, manifest, contract, recipe):
    t = recipe[KEY]
    if (manifest is None or manifest.get('format') != FORMAT
            or fingerprint(manifest) != contract['data'] or fingerprint(manifest) != t['manifest']
            or manifest['tokenizer'] != contract['tokenizer']
            or manifest['settings'] != recipe['stream']
            or fingerprint(manifest['reference']) != metadata['v2_contract']['data']
            or recipe != stage_recipe(metadata, manifest, t['arm'], t['token_budget'], t.get('lr_policy'), t.get('diagnostic'))):
        raise ValueError('Stage B changed unrelated settings or reference validation')
    if t.get('diagnostic') and t['diagnostic']['source'] not in set(manifest['weights']) - set(manifest['reference']['weights']):
        raise ValueError('Diagnostic must evaluate the new training source')


def end_step(recipe):
    t = recipe.get(KEY)
    return t['parent_step'] + t['trial_steps'] if t else None


def migrate_cursor(state, metadata, manifest):
    """Preserve all pending original-source input; new sources start at zero.

Mixture selection uses pilot counters, so lifetime source balances cannot create
an unintended catch-up period. Lifetime token accounting remains untouched.
"""
    if (state['fingerprint'] != metadata['v2_contract']['data'] or state['split'] != 'train'
            or sum(state['offsets'].values()) != metadata['tokens']
            or fingerprint(manifest['reference']) != metadata['v2_contract']['data']):
        raise ValueError('Stage B source cursor mismatch')
    result = copy.deepcopy(state)
    result['fingerprint'] = fingerprint(manifest)
    for label in manifest['weights']:
        if label not in result['sources']:
            result['sources'][label] = dict(dataset=None, rows=0, buffer=[])
            result['offsets'][label] = 0
            result['shuffle']['sources'][label] = dict(documents=[], draws=0, exhausted=False)
    result['research_stage_b'] = dict(format=FORMAT,
        parent_offsets=copy.deepcopy(result['offsets']), offsets={s: 0 for s in manifest['weights']},
        exclusions=None, seen_keys=['content:' + h for h in result['recent_document_hashes']],
        selection_counts={s: {} for s in manifest['weights']})
    return result


class ResearchStageBStream(StreamingTokens):
    def __init__(self, manifest, tokenizer, split, seq_len, state, *, exclusions):
        if split != 'train' or state is None:
            raise ValueError('Stage B requires explicit migrated training state')
        super().__init__(manifest, tokenizer, split, seq_len, state)
        self.stage = copy.deepcopy(state['research_stage_b'])
        if self.stage['format'] != FORMAT:
            raise ValueError('Unknown research Stage B state')
        self.exclusions = set(exclusions)
        digest = fingerprint(sorted(self.exclusions))
        if self.stage['exclusions'] not in (None, digest):
            raise ValueError('Stage B held-out exclusions changed')
        self.stage['exclusions'] = digest
        self.stage_offsets = self.stage['offsets']
        if (set(self.stage_offsets) != set(self.weights)
                or any(type(v) is not int or v < 0 or v % seq_len for v in self.stage_offsets.values())
                or any(self.offsets[s] != self.stage['parent_offsets'][s] + self.stage_offsets[s]
                       for s in self.weights)):
            raise ValueError('Stage B token accounting mismatch')
        new_sources=[s for s in self.config['sources'] if s['label'] not in manifest['reference']['weights']]
        capacities={s.get('provenance_window_keys') for s in new_sources}
        if len(capacities)>1: raise ValueError('New sources disagree on provenance window')
        self.provenance_capacity=next(iter(capacities),None)
        if self.provenance_capacity is not None and (type(self.provenance_capacity) is not int
                or not 100_000 <= self.provenance_capacity <= 3_000_000):
            raise ValueError('Invalid bounded provenance window')
        self.provenance_seen = OrderedDict.fromkeys(self.stage.pop('seen_keys'))
        if self.provenance_capacity is not None and len(self.provenance_seen)>self.provenance_capacity:
            raise ValueError('Saved provenance exceeds configured window')
        self.counts = {s: Counter(v) for s, v in self.stage.pop('selection_counts').items()}
        if set(self.counts) != set(self.sources):
            raise ValueError('Stage B source counters mismatch')

    def state_dict(self):
        state = super().state_dict()
        state['research_stage_b'] = dict(copy.deepcopy(self.stage),
            seen_keys=list(self.provenance_seen), selection_counts={s: dict(c) for s, c in self.counts.items()})
        return state

    def _accepted_document(self, source):
        cursor, counts = self.sources[source], self.counts[source]
        while True:
            if cursor.rows >= cursor.source.get('max_rows_per_source',self.config['max_rows_per_source']):
                raise RuntimeError(source + ': scan limit reached; no source substitution')
            row = cursor.next(); counts['scanned'] += 1
            novel = source not in self.manifest['reference']['weights']
            if novel:
                text, reason = select_novel(row, cursor.source, self.config, self.tokenizer)
            else:
                text, reason = accepted(row,cursor.source,self.config), 'source_filter'
            if text is None:
                counts[reason] += 1; continue
            keys = document_keys(row, text)
            key = content_key(text)
            if partition(key) != 'train':
                counts['reserved_partition'] += 1; continue
            # Keep the original-source control's bounded dedup behavior. New
            # provenance exclusions apply to the new source only; otherwise the
            # control would change training selection as well as its LR policy.
            if key in self.seen or (novel and keys & self.provenance_seen.keys()):
                counts['duplicate'] += 1; continue
            if novel and keys & self.exclusions:
                counts['reference_overlap'] += 1; continue
            if self.provenance_capacity is None and len(self.provenance_seen) + len(keys) > 3_000_000:
                raise RuntimeError('Stage B provenance limit reached; save and review')
            self.provenance_seen.update(dict.fromkeys(sorted(keys)))
            if self.provenance_capacity is not None:
                while len(self.provenance_seen)>self.provenance_capacity:
                    self.provenance_seen.popitem(last=False)
            self.seen[key] = None
            if len(self.seen) > self.manifest['settings']['dedup_window']:
                self.seen.popitem(last=False)
            counts['accepted'] += 1
            return text

    def row(self, source=None):
        counters = self.offsets if self.weights == self.manifest['reference']['weights'] else self.stage_offsets
        source = source or min(self.weights, key=lambda s: counters[s] / self.weights[s])
        pair = super().row(source)
        self.stage_offsets[source] += self.seq_len
        return pair
