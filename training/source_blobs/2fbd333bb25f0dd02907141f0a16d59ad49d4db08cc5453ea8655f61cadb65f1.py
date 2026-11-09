"""Explicit prose/Python/SQL transition, preserving lifetime state and schedule."""
from collections import Counter
import copy
import re

from .common import fingerprint
from .corpus import content_key
from .stream import StreamingTokens, StreamExhausted, stream_manifest, ALL_AVAILABLE

KEY = 'code_data_transition'
FORMAT = 'sml-prose-python-sql-phase-v1'
WEIGHTS = {'web':56, 'dclm':16, 'wiki':8, 'python':15, 'sql':5}


def training_manifest(corpus, tokenizer, settings):
    reference_corpus = corpus['reference_corpus']
    reference = stream_manifest(reference_corpus, tokenizer, settings)
    training = copy.deepcopy(corpus)
    training.pop('reference_corpus')
    panel = training.pop('code_validation', None)
    old = {s['label']:s for s in reference_corpus['sources']}
    new = {s['label']:s for s in training['sources']}
    if set(old) != {'web','dclm','wiki'} or set(new) != set(WEIGHTS):
        raise ValueError('Require exactly the original prose sources plus Python and SQL')
    if settings.get('source_policy') != ALL_AVAILABLE:
        raise ValueError('Code continuation requires all-available source policy')
    if {s['label']:s['weight'] for s in training['sources']} != WEIGHTS:
        raise ValueError('Code continuation must use the approved 80/15/5 token ratio')
    for label, source in old.items():
        if new[label] != dict(source, weight=WEIGHTS[label]):
            raise ValueError('Existing prose sources may only change token allocation')
    for k,v in reference_corpus.items():
        if k not in ('name','sources') and training.get(k) != v:
            raise ValueError('Existing prose filter settings must remain unchanged')
    for label, language in [('python','Python'),('sql','SQL')]:
        source = new[label]
        if source.get('code_filter',{}).get('language') != language:
            raise ValueError('New code source requires language-specific filtering')
        for component in [source,*source.get('additional_sources',[])]:
            if component.get('content_fetch') or component.get('local_jsonl'):
                raise ValueError('Code continuation requires public full-text streaming sources')
            if not re.fullmatch('[0-9a-f]{40}',component.get('revision','')):
                raise ValueError('Every code source revision must be pinned')
            if component.get('renderer') not in (None,'synsql-schema-query-v1'):
                raise ValueError('Unknown code renderer')
            if component.get('renderer') and label != 'sql':
                raise ValueError('SQL renderer can only be used in SQL source')
    result = stream_manifest(training, tokenizer, settings)
    result.update(format=FORMAT, reference=reference)
    if panel is not None:
        from .code_validation import validate_policy
        validate_policy(panel, result, tokenizer)
        result['code_validation'] = copy.deepcopy(panel)
    return result


def validate_transition(recipe):
    from .recipe import validate
    parent = copy.deepcopy(recipe)
    t = parent.pop(KEY)
    keys = {'format','parent_step','start_tokens','parent_contract','parent_replica_hash',
            'validation_fingerprint','manifest'}
    if not isinstance(t,dict) or set(t) != keys or t['format'] != FORMAT:
        raise ValueError('Invalid code-data transition')
    validate(parent)
    if ('gradual_decay_transition' not in parent or fingerprint(parent) != t['parent_contract']['recipe']
            or not t['parent_replica_hash'] or not t['validation_fingerprint']
            or type(t['parent_step']) is not int or t['parent_step'] < 1
            or type(t['start_tokens']) is not int or t['start_tokens'] < 1
            or not isinstance(t['manifest'],str) or len(t['manifest']) != 64):
        raise ValueError('Code transition changed unrelated settings or lost parent provenance')
    return parent


def phase_recipe(metadata, manifest):
    from .recipe import validate
    previous = validate(copy.deepcopy(metadata['recipe']))
    if metadata.get('sft') or KEY in previous or fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('New data phase requires the original pretrained parent')
    if fingerprint(manifest['reference']) != metadata['v2_contract']['data']:
        raise ValueError('Reference corpus is not the parent training corpus')
    if not metadata.get('replica_hash') or not metadata.get('validation_fingerprint'):
        raise ValueError('Parent is missing checkpoint provenance')
    previous[KEY] = dict(format=FORMAT, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']), parent_replica_hash=metadata['replica_hash'],
        validation_fingerprint=metadata['validation_fingerprint'], manifest=fingerprint(manifest))
    return validate(previous)


def verify_transition(metadata, manifest, contract, recipe):
    if (manifest is None or fingerprint(manifest) != contract['data']
            or manifest.get('format') != FORMAT or manifest['settings'] != recipe['stream']
            or recipe != phase_recipe(metadata, manifest)):
        raise ValueError('Code transition manifest, parent or recipe differs')


def migrate_cursor(state, metadata, manifest, recipe):
    verify_transition(metadata, manifest, dict(data=fingerprint(manifest)), recipe)
    old_labels = set(manifest['reference']['weights'])
    if (state['fingerprint'] != metadata['v2_contract']['data'] or state['split'] != 'train'
            or state['seq_len'] != recipe['model']['max_seq_len']
            or set(state['offsets']) != old_labels or set(state['sources']) != old_labels
            or sum(state['offsets'].values()) != metadata['tokens'] or KEY in state):
        raise ValueError('Parent stream cursor does not match the saved checkpoint')
    result = copy.deepcopy(state)
    result['fingerprint'] = fingerprint(manifest)
    for name in set(manifest['weights']) - old_labels:
        result['offsets'][name] = 0
        result['sources'][name] = dict(dataset=None, rows=0, buffer=[])
        if result.get('shuffle') is not None:
            result['shuffle']['sources'][name] = dict(documents=[],draws=0,exhausted=False)
    result[KEY] = dict(format=FORMAT, parent_offsets=copy.deepcopy(result['offsets']),
                       offsets={name:0 for name in manifest['weights']}, selection_counts={})
    return result


class CodePhaseStream(StreamingTokens):
    def __init__(self, manifest, tokenizer, split, seq_len, state=None, *, exclusions=()):
        super().__init__(manifest, tokenizer, split, seq_len, state)
        self.code_exclusions = set(exclusions)
        self.code_sources = set(manifest['weights']) - set(manifest['reference']['weights'])
        if state is not None and KEY not in state:
            raise ValueError('Code phase cursor migration was not performed')
        self.phase = copy.deepcopy(state[KEY]) if state is not None else dict(
            format=FORMAT, parent_offsets={s:0 for s in self.weights},
            offsets={s:0 for s in self.weights}, selection_counts={})
        if (self.phase.get('format') != FORMAT or set(self.phase['offsets']) != set(self.weights)
                or set(self.phase['parent_offsets']) != set(self.weights)
                or any(type(v) is not int or v < 0 or v % seq_len
                       for offsets in (self.phase['offsets'],self.phase['parent_offsets']) for v in offsets.values())
                or any(self.offsets[s] != self.phase['parent_offsets'][s] + self.phase['offsets'][s] for s in self.weights)):
            raise ValueError('Invalid phase/lifetime token accounting')
        self.counts = {s:Counter(self.phase['selection_counts'].get(s,{})) for s in self.code_sources}

    def state_dict(self):
        result = super().state_dict()
        result[KEY] = copy.deepcopy(self.phase)
        result[KEY]['selection_counts'] = {s:dict(c) for s,c in self.counts.items()}
        return result

    def _accepted_document(self, source):
        if source not in self.code_sources:
            return super()._accepted_document(source)
        from .code_quality import select, metadata_reason
        cursor = self.sources[source]
        counts = self.counts[source]
        while True:
            raw = cursor.next()
            counts['scanned'] += 1
            reason = metadata_reason(raw,cursor.source)
            if reason:
                counts[reason] += 1
                continue
            text, reason = select(raw,cursor.source,self.config,self.split,self.code_exclusions,self.tokenizer)
            if reason:
                counts[reason] += 1
                continue
            key = content_key(text)
            if key in self.seen:
                counts['recent_duplicate'] += 1
                continue
            self.seen[key] = None
            if len(self.seen) > self.manifest['settings']['dedup_window']:
                self.seen.popitem(last=False)
            counts['accepted'] += 1
            return text

    def row(self, source=None):
        # Explicit source prevents the base class silently renormalizing after EOF.
        # Exhaustion propagates to the trainer's coordinated save/stop path.
        name = source or min(self.weights, key=lambda s:self.phase['offsets'][s]/self.weights[s])
        try:
            result = super().row(name)
        except StreamExhausted as error:
            raise StreamExhausted(f'{name}: source ended; preserving 80/15/5 by saving and stopping; no recycling') from error
        self.phase['offsets'][name] += self.seq_len
        return result
