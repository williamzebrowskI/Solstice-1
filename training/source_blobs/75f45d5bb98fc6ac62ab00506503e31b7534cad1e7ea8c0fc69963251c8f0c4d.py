"""Computed English conversations with disjoint wording, entities and final tests."""
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from sml_v1.common import atomic_json, file_sha256, fingerprint
from sml_v1.tokenization import Tokenizer
from .data import serialize
from .lora_repair_common import SHARED, REPORT, ROOT, ASSETS, PARENTS, DOMAINS

BANKS = {
 'train': ('Adele Basil Corin Daria Elin Faye Galen Hilda Idris Jada Keir Livia Miro Nessa Orla Pavel Rina Soren Talia Vito Willa Zane'.split(),
           'apron badge cable drum folder glass handle jacket lamp medal mug pen plate ring rope soap stamp tin tool tube'.split()),
 'dev': ('Aisha Bao Cora Dilan Enzo Frida Imani Jules Leif Mira'.split(),
         'brush candle coin dish frame gate hanger jar leaf mask'.split()),
 'test': ('Amina Boaz Celeste Davin Edda Farah Hester Ilya Kellan Liora Niko Odette Priya Remy Saira Tomas'.split(),
          'bell cap crate cup doll fan flute harp lid net pin rack sack slate sock sponge'.split())}
FIELDS = ('contact', 'time', 'quantity', 'label', 'location')
COLORS = ('red','blue','green','orange','white','black','purple','yellow')
PLACES = ('north entrance','south desk','west counter','east gate','main office','back door')
PHRASES = {
 'train': ('Use only the details I provide.', 'Please answer from this information.', 'Read these notes and answer my question.'),
 'dev': ('Base your answer entirely on these notes.', 'Use the supplied details to respond.'),
 'test': ('Treat the following message as your only source.', 'Answer solely from the information in this conversation.')}
MISSING = {
 'train': ('Not provided.', 'The information does not say.', 'I cannot tell from these details.'),
 'dev': ('Not stated in these notes.', 'This detail is not available.'),
 'test': ('The supplied information does not specify this.', 'This cannot be determined from the message.')}
QUESTION = {
 'train': ('Who is the contact?', 'At what time is the pickup?', 'How many items are there?', 'What color is the label?', 'Where is the collection point?'),
 'dev': ('Which person should be contacted?', 'What pickup time was given?', 'What is the number of items?', 'Which label color was specified?', 'What location should I use for collection?'),
 'test': ('Whom should I contact?', 'When am I supposed to collect it?', 'How many items should I expect?', 'What is the color of its label?', 'Where should I go to collect it?')}


def pair(prompt, answer):
    return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]


def normalize(text):
    return ' '.join(text.casefold().split())


def facts(rng, split):
    names, words = BANKS[split]
    return dict(contact=rng.choice(names),time=f'{rng.randrange(7,20)}:{rng.choice((0,10,15,20,30,40,45,50)):02d}',
                quantity=str(rng.randrange(2,80)),label=rng.choice(COLORS),location=rng.choice(PLACES)),rng.choice(words)


def render(values, item, rng):
    clauses = dict(contact=f'The contact is {values["contact"]}.' if 'contact' in values else '',
                   time=f'Pickup time is {values["time"]}.' if 'time' in values else '',
                   quantity=f'There are {values["quantity"]} items.' if 'quantity' in values else '',
                   label=f'The label is {values["label"]}.' if 'label' in values else '',
                   location=f'The collection point is the {values["location"]}.' if 'location' in values else '')
    pieces=[v for v in clauses.values() if v];rng.shuffle(pieces)
    return f'I have a {item} order to collect. '+' '.join(pieces)


def row(split, domain, index, messages, spec, family, cluster, strict=False):
    answer = messages[-1]['content']
    return dict(component=domain,group=f'repair:{split}:{domain}:{index}',cluster=cluster,family=family,
                messages=messages,spec=spec,verification=dict(kind='verbatim_format' if strict else 'exact',answers=[answer]),
                provenance=dict(origin='computed and independently audited scenario',license='CC0-1.0'),split=split)


def grounding_pair(rng, split, index):
    values,item=facts(rng,split);field=FIELDS[index%len(FIELDS)]
    question=QUESTION[split][FIELDS.index(field)];missing=rng.choice(MISSING[split])
    instruction=rng.choice(PHRASES[split])+f' If the requested fact is absent, reply exactly "{missing}" Otherwise give only the requested value.'
    cluster=f'{split}:facts:{index}';rows=[]
    for absent,domain in [(False,'grounded'),(True,'abstention')]:
        supplied={k:v for k,v in values.items() if not (absent and k==field)}
        passage=render(supplied,item,rng);answer=missing if absent else values[field]
        messages=pair(instruction+'\n\n'+passage+'\n\n'+question,answer)
        rows.append(row(split,domain,index,messages,dict(kind='facts',values=supplied,field=field,missing=missing),field,cluster))
    return rows


def dialogue(rng,split,index):
    values,item=facts(rng,split);field=FIELDS[index%5]
    updated,_=facts(rng,split)
    while updated[field]==values[field]:updated,_=facts(rng,split)
    latest=updated[field];initial=values[field]
    messages=pair(render(values,item,rng),'Understood.')
    update=f'Correction: the {field} is now {latest}. Please use this instead of the earlier {field}.'
    if split=='dev':update=f'I need to revise one detail: replace the old {field} with {latest}.'
    if split=='test':update=f'The previous {field} is out of date. The current {field} is {latest}.'
    messages+=pair(update,'I have noted the change.')
    if index%2:messages+=pair('Please remember that I will bring a reusable bag.','Understood.')
    messages+=pair(QUESTION[split][FIELDS.index(field)]+' Give only the current value.',latest)
    return row(split,'dialogue',index,messages,dict(kind='latest',field=field,initial=initial,current=latest),field,f'{split}:dialogue:{index}')


def constraints(rng,split,index):
    names,words=BANKS[split];values=rng.sample(words,rng.randrange(3,7));family=index%8
    style='Please edit this list: ' if split=='train' else 'Reformat these entries: ' if split=='dev' else 'Here are the entries I want you to transform: '
    instructions=[
      'Put the entries in alphabetical order and use one "- " bullet per entry.',
      'Use uppercase letters and put each entry on a separate line.',
      'Use lowercase letters and join the entries with a comma and a space.',
      'Reverse the entry order and use a numbered list starting with 1., one entry per line.',
      'Keep only the first two entries and join them with a comma and a space.',
      'Put the entries in alphabetical order, join them with a comma and a space, and wrap the entire result in double quotes.',
      'Keep the entry order, use uppercase letters, and join entries with a comma and a space.',
      'Keep the entry order and use one entry per line. Then add a final line containing only Done.'
    ]
    if split!='train':
        instructions=[s.replace('Put','Arrange').replace('Keep','Retain').replace('Use','Apply').replace('join','separate').replace('Reverse the entry order','Start from the last entry and work backwards') for s in instructions]
    shown=[v.upper() if family==2 else v for v in values]
    prompt=style+', '.join(shown)+'.\n'+instructions[family]+' Include no introduction or explanation.'
    spec=dict(kind='layout',family=family,values=shown)
    return row(split,'constraints',index,pair(prompt,solve(spec)),spec,str(family),f'{split}:constraints:{index}',True)


def reasoning(rng,split,index):
    names,words=BANKS[split];a,b=rng.randrange(2,40),rng.randrange(2,30);name,other=rng.sample(names,2);item=rng.choice(words);family=index%6
    if family==0:story=f'{name} has {a} tickets and receives {b} more.';question='How many tickets does '+name+' have now?';answer=str(a+b)
    elif family==1:
        a+=b;story=f'A box contains {a} items. {b} are removed.';question='How many items remain?';answer=str(a-b)
    elif family==2:
        a,b=rng.randrange(2,10),rng.randrange(2,10);story=f'For {name}\'s {item} order, there are {a} packs with {b} labels in each pack.';question='How many labels are there altogether?';answer=str(a*b)
    elif family==3:
        a,b=rng.sample(range(2,60),2);story=f'{name} has {a} points and {other} has {b} points.';question='Who has more points?';answer=name if a>b else other
    elif family==4:
        a,b=rng.randrange(2,12),rng.randrange(2,12);story=f'{name} divides {a*b} items equally among {a} boxes.';question='How many items go in each box?';answer=str(b)
    else:
        a,b=rng.sample(range(2,60),2);story=f'The {item} item costs {a} credits. A second item costs {b} credits.';question='What is the difference between their prices?';answer=str(abs(a-b))
    if split=='dev':
        story=story.replace('receives','is given').replace('are removed','are taken out').replace('with','containing').replace('has','holds')
        question=question.replace('How many','What number of').replace('Who has more points?','Which person has the higher score?')
    if split=='test':
        story=story.replace('receives','gets').replace('are removed','leave the box').replace('in each pack','per pack').replace('divides','shares').replace('has','owns')
        question=question.replace('How many','Count how many').replace('Who has more points?','Name the person with more points.').replace('difference between their prices','gap between the two prices')
    prefix={'train':'Work this out from the details.','dev':'Solve this short situation.','test':'What follows from these facts?'}[split]
    messages=pair(prefix+'\n'+story+'\n'+question+' Return only the number or name, without an explanation.',answer)
    spec=dict(kind='reasoning',family=family,a=a,b=b,name=name,other=other)
    return row(split,'reasoning',index,messages,spec,str(family),f'{split}:reasoning:{index}')


def solve(spec):
    k=spec['kind']
    if k=='facts':return spec['values'].get(spec['field'],spec['missing'])
    if k=='latest':return spec['current']
    if k=='reasoning':
        a,b=spec['a'],spec['b'];f=spec['family']
        return [lambda:str(a+b),lambda:str(a-b),lambda:str(a*b),lambda:spec['name'] if a>b else spec['other'],lambda:str(b),lambda:str(abs(a-b))][f]()
    if k=='layout':
        v=spec['values'];f=spec['family']
        return [lambda:'\n'.join('- '+x for x in sorted(v)),lambda:'\n'.join(v).upper(),lambda:', '.join(v).lower(),
                lambda:'\n'.join(f'{i+1}. {x}' for i,x in enumerate(reversed(v))),lambda:', '.join(v[:2]),
                lambda:'"'+', '.join(sorted(v))+'"',lambda:', '.join(v).upper(),lambda:'\n'.join(v+['Done.'])][f]()
    raise ValueError(k)


def prepare():
    destination=SHARED/'data'
    if destination.exists():raise ValueError('Do not overwrite frozen repair data')
    tok=Tokenizer(ASSETS/'tokenizer');excluded=set()
    for split in ('train','dev','test'):
        for line in (ASSETS/f'corpus/{split}.jsonl').read_text().splitlines():
            r=json.loads(line);excluded.add(fingerprint(r['messages'][:-1]))
    public=ROOT/'reports/benchmarks/deep44-reliable-chat-167-fp32-20261106'
    public_prompts={normalize(r['prompt']) for r in json.loads((public/'ifeval/inputs.json').read_text())}
    public_prompts.update(normalize(json.loads(l)['messages'][-1]['content']) for l in (public/'chat/cases.jsonl').read_text().splitlines())
    seen=set(excluded);rows_by_split={};max_length=0
    for split,count,seed in [('train',4096,2026110911),('dev',60,2026110912),('test',200,2026110913)]:
        rng=random.Random(seed);rows=[];counts=Counter();index=0
        while min([counts[d] for d in DOMAINS])<count:
            candidates=grounding_pair(rng,split,index)
            candidates += [dialogue(rng,split,index),constraints(rng,split,index),reasoning(rng,split,index)]
            index+=1
            # Preserve complete fact-present/fact-absent pairs when deduplicating.
            if any(fingerprint(r['messages'][:-1]) in seen for r in candidates):continue
            for r in candidates:
                if counts[r['component']]>=count:continue
                if normalize(r['messages'][-2]['content']) in public_prompts:raise ValueError('Public prompt contamination')
                if r['messages'][-1]['content']!=solve(r['spec']):raise ValueError('Incorrect generated label')
                ids,_=serialize(tok,r['messages'])
                if len(ids)>513:raise ValueError('No truncation: conversation exceeds 512 training tokens')
                if any(i in (0,1,2,3) for m in r['messages'] for i in tok.encode(m['content'])):raise ValueError('Reserved token in text')
                max_length=max(max_length,len(ids)-1);seen.add(fingerprint(r['messages'][:-1]));rows.append(r);counts[r['component']]+=1
        rng.shuffle(rows);rows_by_split[split]=rows
    destination.mkdir()
    for split,rows in rows_by_split.items():
        with (destination/(split+'.jsonl')).open('w') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    source=ROOT/'pretraining/prepared/parent2000_generalize_20261109/fresh_data/replay.jsonl'
    shutil.copy2(source,destination/'replay.jsonl')
    receipt=dict(format='chat-repair-data-v1',counts={k:dict(Counter(r['component'] for r in v)) for k,v in rows_by_split.items()},
                 maximum_sequence_tokens=max_length,truncated=0,parents={k:file_sha256(v/'manifest.json') for k,v in PARENTS.items()},
                 tokenizer=tok.fingerprint,public_prompt_overlap=0,replay_source=str(source),replay_sha256=file_sha256(source),
                 files={p.name:file_sha256(p) for p in destination.glob('*.jsonl')},
                 scope='Verified fictional records and computable text tasks. Historical and public evaluations separately check broader abilities. Test reserved for one development-selected finalist.')
    atomic_json(destination/'manifest.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))


if __name__=='__main__':prepare()
