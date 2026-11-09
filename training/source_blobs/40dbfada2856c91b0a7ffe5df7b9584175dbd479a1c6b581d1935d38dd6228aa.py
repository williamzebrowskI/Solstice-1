"""FP32 evaluation of new task families with untouched final-test enforcement."""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v1.common import atomic_json, read_json, file_sha256
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.tokenization import Tokenizer
from .lora_checkpoint import verify
from .lora_repair_common import SHARED, REPORT, ASSETS, compact
from .text_eval import reference_inference
from .evaluate import generate
from .text_checks import answer_pass
from .quality_gate import looping


def load(bundle):
    state=verify(bundle);meta=read_json(bundle/'inference/model.safetensors.json')
    model=TransformerLM(TransformerConfig(**meta['recipe']['model']))
    model.load_weights(str(bundle/'inference/model.safetensors'),strict=True);mx.eval(model.parameters())
    return model,state


def abstention_signal(text):
    return len(text.split())<=30 and bool(re.search(r"not (?:provided|stated|specified|available|given)|(?:does not|doesn't) (?:say|specify|provide)|(?:cannot|can't) (?:tell|determine|be determined)|not enough information|(?:don't|do not) know",text,re.I))


def content_match(text,row):
    """Accept finite, unambiguous answer paraphrases; never substring matches."""
    if row['component']=='constraints':return answer_pass(text,row['verification'])
    def norm(value):return ' '.join(value.replace('’',"'").strip().strip('"').strip("'").rstrip('.!?').casefold().split())
    value=norm(row['messages'][-1]['content']);actual=norm(text)
    aliases={value}
    if row['component']=='abstention':
        aliases.update(norm(a) for a in ('Not provided.','Not stated.','Not specified.','Unknown.','I do not know.',"I don't know.",
                         'Not enough information.','The requested fact is absent.','The requested detail is missing.',
                         'This cannot be determined from the message.','This cannot be determined from the supplied information.'))
        field_subjects={'contact':('the contact',), 'time':('the time','the pickup time'),
                        'quantity':('the quantity','the number of items'), 'label':('the label','the label color'),
                        'location':('the location','the collection point')}[row['spec']['field']]
        for subject in ('the information','the supplied information','the provided information','the requested information','the requested fact','the requested detail','this detail','the answer','the supplied details','the provided details',*field_subjects):
            for ending in ('not provided','not stated','not specified','not available','not given','missing','absent','unknown'):
                for verb in ('is','was','are'):aliases.add(norm(subject+' '+verb+' '+ending))
        for source in ('these notes','the notes','this information','the supplied information','the provided information','these details','the details','the message','the passage'):
            for ending in ('not provided','not stated','not specified','not available','not given'):
                aliases.add(norm(ending+' in '+source))
            for prefix in ('I cannot tell from ','I cannot determine this from ','I do not know from '):aliases.add(norm(prefix+source))
            for verb in ('does not say','does not specify','does not provide this','do not say','do not specify'):
                aliases.add(norm(source+' '+verb))
    else:
        aliases.update(norm(prefix+value) for prefix in ('Answer: ','The answer is ','The result is ',"It's ",'It is '))
        field=row['spec'].get('field')
        if field:
            subjects={'contact':('the contact','the current contact'), 'time':('pickup time','the pickup time','the current time'),
                      'quantity':('the quantity','the number of items','the current quantity'), 'label':('the label','the label color','the current label'),
                      'location':('the collection point','the location','the current location')}[field]
            for subject in subjects:
                aliases.add(norm(subject+' is '+value))
                if field=='location':aliases.add(norm(subject+' is the '+value))
            if field=='location':aliases.add(norm('the '+value))
            if field=='quantity':aliases.update([norm('There are '+value+' items'),norm(value+' items')])
    return actual in aliases


def evaluate(model,tokenizer,output,split='dev',parent_manifest=None,heartbeat=lambda:None):
    if split=='test':
        if not (REPORT/'selection_frozen.json').exists():raise ValueError('Final test cannot be opened before selecting the finalist')
        frozen=read_json(REPORT/'selection_frozen.json')
        if parent_manifest not in frozen['allowed_test_manifests'] or file_sha256(SHARED/'data/test.jsonl')!=frozen['test_data_sha256']:
            raise ValueError('Final test model or data differs from the frozen selection')
    if any(v.dtype!=mx.float32 for _,v in tree_flatten(model.parameters())):raise ValueError('FP32 evaluation required')
    path=SHARED/'data'/(split+'.jsonl');rows=[json.loads(l) for l in path.read_text().splitlines()]
    samples=[];counts=defaultdict(int);strict_counts=defaultdict(int);totals=defaultdict(int);model.eval()
    try:
        with reference_inference(model):
            for i,row in enumerate(rows,1):
                heartbeat();v=generate(model,tokenizer,row['messages'][:-1],limit=96)
                normal=v['stop_reason'] in ('turn_end','document_end');repeat=looping(v['text']);empty=not v['text'].strip()
                matched=content_match(v['text'],row);strict_matched=answer_pass(v['text'],row['verification'])
                passed=matched and normal and not repeat and not empty;strict_passed=strict_matched and normal and not repeat and not empty
                counts[row['component']]+=int(passed);totals[row['component']]+=1
                strict_counts[row['component']]+=int(strict_passed)
                samples.append(dict(group=row['group'],cluster=row['cluster'],component=row['component'],family=row['family'],
                                    messages=row['messages'][:-1],reference=row['messages'][-1]['content'],**v,
                                    matched=matched,passed=passed,strict_matched=strict_matched,strict_passed=strict_passed,
                                    repeated=repeat,empty=bool(empty),abstention_signal=abstention_signal(v['text'])))
                if i%50==0:print(f'[repair-eval {split}] {i}/{len(rows)}',flush=True)
        value=dict(correct=sum(counts.values()),total=len(rows),counts=dict(counts),totals=dict(totals),
                   strict_correct=sum(strict_counts.values()),strict_counts=dict(strict_counts),
                   scoring='Conservative whole-answer equivalence for factual tasks; exact format for transformation tasks. Strict format compliance is reported separately. Whole-passage copying and contradictory extra claims are not accepted.',
                   natural_stop=sum(s['stop_reason'] in ('turn_end','document_end') for s in samples)/len(rows),
                   repetition_rate=sum(s['repeated'] for s in samples)/len(rows),empty=sum(s['empty'] for s in samples),
                   false_abstention_signals=sum(s['abstention_signal'] and s['component']=='grounded' for s in samples),
                   missing_fact_abstention_signals=sum(s['abstention_signal'] and s['component']=='abstention' for s in samples),
                   samples=samples,identity=dict(parent=parent_manifest,data=file_sha256(path),evaluator=file_sha256(Path(__file__))))
        atomic_json(output,value);return compact(value)
    finally:model.train()


def paired(candidate,baseline):
    a,b=candidate['samples'],baseline['samples']
    if len(a)!=len(b) or any(x['group']!=y['group'] or x['messages']!=y['messages'] for x,y in zip(a,b)):
        raise ValueError('Paired evaluation inputs differ')
    clusters=defaultdict(list)
    for x,y in zip(a,b):clusters[x['cluster']].append(int(x['passed'])-int(y['passed']))
    sums=np.array([sum(v) for v in clusters.values()]);sizes=np.array([len(v) for v in clusters.values()])
    rng=np.random.default_rng(2026110914);boots=[]
    for _ in range(10000):
        indices=rng.integers(0,len(sums),len(sums));boots.append(float(sums[indices].sum()/sizes[indices].sum()))
    return dict(gain=float(sums.sum()/sizes.sum()),gained=sum(x['passed'] and not y['passed'] for x,y in zip(a,b)),
                lost=sum(y['passed'] and not x['passed'] for x,y in zip(a,b)),
                confidence_interval_97_5=list(map(float,np.quantile(boots,[.0125,.9875]))),
                method='Scenario-cluster paired percentile bootstrap, 10000 draws; 97.5% interval for each of two baseline comparisons. Covers sampling uncertainty within this authored panel, not all chat tasks.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],default='dev');a=p.parse_args()
    m,_=load(a.bundle);evaluate(m,Tokenizer(ASSETS/'tokenizer'),a.output,a.split,file_sha256(a.bundle/'manifest.json'))
