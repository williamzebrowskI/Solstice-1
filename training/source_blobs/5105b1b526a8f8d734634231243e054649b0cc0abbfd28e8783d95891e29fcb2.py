"""Fresh computed conversations, separated by facts, wording and entity banks."""
import json,random,re
from collections import Counter,defaultdict
from pathlib import Path
from sml_v1.common import atomic_json,file_sha256,fingerprint,read_json
from sml_v1.tokenization import Tokenizer
from .data import serialize
from .lora_search_common import SHARED,REPORT,ASSETS,PARENT

BANKS={
 'train':('Amelia Bruno Clara Diego Emilia Felix Greta Hugo Ines Jonas Kira Lucio Marta Noel Oren Petra Quentin Rosa Silas Thea Ugo Vera Walter Yvonne'.split(),
          'basket blanket button cabinet carton compass cushion envelope feather helmet ladder lantern mirror notebook parcel pencil pillow ribbon sandal scarf spoon ticket towel trumpet vase wagon wallet whistle window zipper'.split()),
 'fresh_test':('Alina Boris Celia Duncan Estelle Finn Gemma Hector Ivana Julian'.split(),
               'album bowl comb desk funnel glove hook jug key locket mat napkin paddle quilt ruler tray umbrella vest wheel yo-yo'.split())}

def operation(op,values):
    if op=='add':return str(sum(values))
    if op=='subtract':return str(values[0]-values[1])
    if op=='multiply':return str(values[0]*values[1])
    if op=='sort':return ', '.join(map(str,sorted(values)))
    if op=='descending':return ', '.join(map(str,sorted(values,reverse=True)))
    if op=='even':return ', '.join(str(x) for x in values if x%2==0) or 'none'
    if op=='odd':return ', '.join(str(x) for x in values if x%2!=0) or 'none'
    if op=='reverse':return ', '.join(reversed(values))
    if op=='alphabetical':return ', '.join(sorted(values))
    if op=='lower':return values[0].lower()
    if op=='upper':return values[0].upper()
    if op=='replace':return values[0].replace(values[1],values[2])
    if op=='last':return ', '.join(values[-2:])
    if op=='first':return ', '.join(values[:2])
    if op=='count':return str(len(values))
    if op=='unique':return ', '.join(dict.fromkeys(values))
    raise ValueError(op)

PHRASINGS={
 'add':(['What is the combined amount of {x}?','Please total these amounts: {x}.','I need the sum: {x}.'],['Work out the total for {x}.']),
 'subtract':(['There are {a} tickets, and {b} are used. How many remain?','Calculate {a} minus {b}.','What is left after taking {b} away from {a}?'],['A box held {a} items. After removing {b}, how many are left?']),
 'multiply':(['There are {a} packets with {b} buttons each. How many buttons altogether?','Calculate {a} times {b}.','What is the product of {a} and {b}?'],['Each of {a} containers has {b} items. Find the total.']),
 'sort':(['Give this list in ascending numerical order: {x}.','Arrange {x} so that each number is larger than the one before it.','Smallest first, please: {x}.'],['Sequence these numbers numerically, low to high: {x}.']),
 'descending':(['Largest first, please: {x}.','Give {x} in descending numerical order.','Arrange {x} from highest to lowest.'],['Sequence these numbers numerically, high to low: {x}.']),
 'even':(['Which of {x} are even? Keep their order.','From {x}, include only multiples of 2, in their current order.','Delete the odd entries in {x}; do not reorder the rest.'],['Extract the even-valued entries without rearranging them: {x}.']),
 'odd':(['Which of {x} are odd? Keep their order.','From {x}, include only numbers that are not multiples of 2, in their current order.','Delete the even entries in {x}; do not reorder the rest.'],['Extract the odd-valued entries without rearranging them: {x}.']),
 'reverse':(['Please flip the item order: {x}.','Start with the last entry and work back to the first: {x}.','Give the backwards version of this list: {x}.'],['Invert the sequence of items: {x}.']),
 'alphabetical':(['Please alphabetize {x}.','Put {x} in dictionary order.','Give these words in A-to-Z order: {x}.'],['Order this collection lexicographically: {x}.']),
 'lower':(['Make this text entirely lowercase: {x}','Rewrite using small letters only: {x}','Lowercase every letter here: {x}'],['Remove the capitalization from this text: {x}']),
 'upper':(['Make this text entirely uppercase: {x}','Rewrite using capital letters only: {x}','Uppercase every letter here: {x}'],['Capitalize every letter of this text: {x}']),
 'replace':(['In "{text}", replace "{old}" with "{new}".','Change every "{old}" in "{text}" to "{new}".','Use "{new}" instead of "{old}" in this text: {text}'],['Substitute "{new}" for each occurrence of "{old}": {text}']),
 'last':(['I only need the final two items, in order: {x}.','Give the last pair of entries from {x}.','Drop everything except the two items at the end: {x}.'],['Report the trailing two entries of this sequence: {x}.']),
 'first':(['I only need the initial two items, in order: {x}.','Give the first pair of entries from {x}.','Drop everything except the two items at the beginning: {x}.'],['Report the leading two entries of this sequence: {x}.']),
 'count':(['How many entries are shown in {x}?','Give the item count for {x}.','Count these entries: {x}.'],['State how many items this sequence contains: {x}.']),
 'unique':(['Remove duplicate entries, keeping the first occurrence of each: {x}.','List {x} without repeats; keep the original order.','Deduplicate {x} without sorting.'],['Keep one occurrence per item, in first-seen order: {x}.'])}

def pair(prompt,answer):return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]

def make_row(rng,split,component,index):
    names,words=BANKS[split];name,other,third=rng.sample(names,3);w,w2,w3,w4=rng.sample(words,4)
    a,b=rng.sample(range(3,96),2);family=index%16;test=split!='train';spec={}
    if component=='skills':
        op=list(PHRASINGS)[family]
        values=rng.sample(range(1,120),rng.randint(3,6))
        if op in ('add','subtract'):values=[a,b]
        elif op=='multiply':values=[rng.randint(2,12),rng.randint(2,12)]
        elif op in ('reverse','alphabetical','last','first','count'):values=rng.sample(words,rng.randint(3,6))
        elif op=='unique':values=[w,w2,w,w3,w2,w4]
        elif op in ('lower','upper'):values=[' '.join(rng.sample(words,3)).swapcase() if op=='lower' else ' '.join(rng.sample(words,3))]
        elif op=='replace':values=[f'{name} packed a {w} and a {w2}.',w,w3]
        wording=rng.choice(PHRASINGS[op][1 if test else 0])
        prompt=wording.format(x=', '.join(map(str,values)) if op not in ('lower','upper') else values[0],a=values[0],b=values[1] if len(values)>1 else '',text=values[0],old=values[1] if len(values)>1 else '',new=values[2] if len(values)>2 else '')
        list_op=op in ('sort','descending','even','odd','reverse','alphabetical','last','first','unique')
        if list_op:prompt+=' Separate multiple items with a comma and a space.'
        if op in ('even','odd'):prompt+=' If nothing qualifies, write none.'
        prompt+=rng.choice([' Return just the result.',' No explanation is needed.',' Answer without an introduction.']) if not test else ' Supply only the requested output.'
        spec=dict(kind='operation',op=op,values=values);answer=operation(op,values);messages=pair(prompt,answer)
    elif component=='constraints':
        sentences=[f'{name} packed {a} {w} items.',f'{other} carried {b} {w2} items.',f'{third} kept the {w3} item.'][:rng.randint(2,3)]
        family=index%10;spec=dict(kind='format',family=family,sentences=sentences,title=f'{w.title()} inventory',ending=f'Filed by {name}.')
        instructions=[f'Make one bullet per sentence using "- ".', 'Join the sentences into a single paragraph with one space between sentences.',
            'Use one sentence per paragraph, with a blank line between paragraphs.', 'Join the sentences with spaces and use lowercase letters.',
            'Join the sentences with spaces and use uppercase letters.', 'Quote the first sentence using double quotes.',
            f'Place the heading {spec["title"]} on the first line, a blank line next, and the first sentence after it.',
            f'Give the first sentence, then a space, then exactly: {spec["ending"]}',
            'Make a numbered list using 1., 2., and so on, one sentence per line.',
            'Give the first and second sentences on separate lines.']
        instruction=instructions[family]
        if test:instruction=instruction.replace('Make','Create').replace('Give','Present').replace('Join','Combine').replace('Use','Apply').replace('Place','Put')
        prompt=rng.choice(['Please reformat the following text. ','I need this text presented differently. ','Follow this layout request. '])+instruction+' Keep the wording unless a case change is requested. Include no commentary.\n\n'+ '\n'.join(sentences)
        answer=solve(spec);messages=pair(prompt,answer)
    elif component=='dialogue':
        family=index%8
        if family==0:
            messages=pair(f'The delivery reference is {a:02d}-{b:02d}.','I have noted that reference.')+pair(f'Update the reference to {b:02d}-{a:02d}. The earlier one is no longer correct.','The delivery reference has been updated.')
            question='Which delivery reference should be used?';answer=f'{b:02d}-{a:02d}';spec=dict(kind='latest',initial=f'{a:02d}-{b:02d}',current=answer)
        elif family==1:
            messages=pair(f'{name} chose the {w} item; {other} chose the {w2} item.','I have noted their choices.')
            question=f'What item did {other} choose?';answer=w2;spec=dict(kind='lookup',mapping={name:w,other:w2},key=other)
        elif family==2:
            messages=pair(f'I want the {w} design.','I have noted your design preference.')+pair(f'The order also needs {a} labels.','I have noted the label count.')
            question='Which design did I request?';answer=w;spec=dict(kind='lookup',mapping={'design':w,'labels':str(a)},key='design')
        elif family==3:
            messages=pair(f'The package is on shelf {a}.','The package is on that shelf.')+pair(f'It has now been moved to shelf {b}.','I have noted the new location.')
            question='Which shelf has the package now?';answer=str(b);spec=dict(kind='latest',initial=str(a),current=answer)
        elif family==4:
            messages=pair(f'We need {a} seats.','I have noted the seat count.')+pair(f'Change that to {b} seats. The venue stays the same.','The seat count has been changed.')
            question='How many seats do we need now?';answer=str(b);spec=dict(kind='latest',initial=str(a),current=answer)
        elif family==5:
            messages=pair(f'Each {w} item costs {a} credits.','I have noted the unit price.')+pair(f'The updated price is {b} credits each.','I have noted the updated price.')
            question='What is the cost of two items at the updated price?';answer=str(b*2);spec=dict(kind='operation',op='multiply',values=[b,2])
        elif family==6:
            messages=pair(f'{name} will collect the {w} parcel. No collection time has been specified.','The collection time is unspecified.')
            question='At what time will the parcel be collected?';answer='The collection time has not been provided.';spec=dict(kind='missing',field='collection time',answer=answer)
        else:
            messages=pair(f'{name} owns {a} {w} items and {other} owns {b} {w2} items.','I have noted the two collections.')+pair(f'{name} gives one item away. The other collection does not change.','Only the first collection changes.')
            question=f'How many items does {other} still own?';answer=str(b);spec=dict(kind='lookup',mapping={name:str(a-1),other:str(b)},key=other)
        if test:question='From the latest information, '+question[0].lower()+question[1:]
        else:question=rng.choice(['','Please tell me: ','Using those details, '])+question
        messages+=pair(question+' Answer briefly with only the requested information.',answer)
    else:
        # These are explicit fictional records; answerability is computable.
        records=[dict(person=name,item=w,quantity=str(a),shelf=str(rng.randint(1,30))),dict(person=other,item=w2,quantity=str(b),shelf=str(rng.randint(1,30)))]
        chosen=rng.choice(records);field=rng.choice(['item','quantity','shelf','date'])
        # 80% answerable, matching the intent to avoid excessive abstention.
        if index%5:field=rng.choice(['item','quantity','shelf'])
        else:field='date'
        passage=' '.join(f'{r["person"]} stored {r["quantity"]} {r["item"]} items on shelf {r["shelf"]}.' for r in records)
        passage+=f' The storage room also contains a {w3} and a {w4}.'
        questions=dict(item=f'What kind of items did {chosen["person"]} store?',quantity=f'How many items did {chosen["person"]} store?',shelf=f'Which shelf holds the items stored by {chosen["person"]}?',date=f'On which date did {chosen["person"]} store the items?')
        question=questions[field]
        if test:question=question.replace('What kind of','Which kind of').replace('How many','What number of').replace('Which shelf','What shelf number')
        spec=dict(kind='record',records=records,person=chosen['person'],field=field)
        answer=solve(spec)
        prompt=rng.choice(['Answer using the record below.','Use only this supplied record.','Read the record and give a short answer.'])+' If the requested detail is missing, say "The passage does not say."\n\n'+passage+'\n\n'+question+' Give only the answer.'
        messages=pair(prompt,answer)
    assert solve(spec)==answer
    return dict(component=component,group=f'fresh-v2:{split}:{component}:{index}',messages=messages,spec=spec,verification=dict(kind='verbatim_format',answers=[answer]),split=split,provenance=dict(origin='computed fresh examples',license='CC0-1.0'))

def solve(spec):
    kind=spec['kind']
    if kind=='operation':return operation(spec['op'],spec['values'])
    if kind=='latest':return spec['current']
    if kind=='lookup':return spec['mapping'][spec['key']]
    if kind=='missing':return spec['answer']
    if kind=='record':return next(r for r in spec['records'] if r['person']==spec['person']).get(spec['field'],'The passage does not say.')
    if kind=='format':
        s=spec['sentences'];f=spec['family']
        return [lambda:'\n'.join('- '+x for x in s),lambda:' '.join(s),lambda:'\n\n'.join(s),lambda:' '.join(s).lower(),lambda:' '.join(s).upper(),lambda:'"'+s[0]+'"',lambda:spec['title']+'\n\n'+s[0],lambda:s[0]+' '+spec['ending'],lambda:'\n'.join(f'{i+1}. {x}' for i,x in enumerate(s)),lambda:'\n'.join(s[:2])][f]()
    raise ValueError(kind)

def prepare():
    output=SHARED/'fresh_data'
    if output.exists():raise ValueError('Fresh data already exists')
    tok=Tokenizer(ASSETS/'tokenizer');old={split:[json.loads(l) for l in (ASSETS/f'corpus/{split}.jsonl').open()] for split in ('train','dev','test')}
    excluded={fingerprint(r['messages'][:-1]) for rows in old.values() for r in rows}
    allseen=set(excluded);counts={};maximum=0;rejected=Counter();generated={}
    for split,count in [('train',2000),('fresh_test',64)]:
        rows=[];rng=random.Random(202611094 if split=='train' else 202611095)
        for component in ('skills','constraints','dialogue','grounded'):
            accepted=0;i=0
            while accepted<count:
                row=make_row(rng,split,component,i);i+=1;key=fingerprint(row['messages'][:-1])
                if key in allseen:rejected['duplicate']+=1;continue
                ids,_=serialize(tok,row['messages'])
                if len(ids)>513:raise ValueError('New conversation exceeds 512-token training budget')
                if any(x in (0,1,2,3) for m in row['messages'] for x in tok.encode(m['content'])):raise ValueError('Reserved token in content')
                maximum=max(maximum,len(ids)-1);allseen.add(key);rows.append(row);accepted+=1
        rng.shuffle(rows);generated[split]=rows;counts[split]=dict(Counter(r['component'] for r in rows))
    # Original training replay is explicitly labeled; it is not new data.
    replay=[];rng=random.Random(202611096);quotas=dict(general=1500,knowledge=1500,grounded=1000,dialogue=1000,skills=1000,constraints=1000,social=500,rewrite=500)
    pools=defaultdict(list)
    for row in old['train']:
        ids,_=serialize(tok,row['messages'])
        if len(ids)>513:continue
        if row['component'] in ('knowledge','grounded'):
            answer=row['messages'][-1]['content'];v=row.get('verification') or {}
            support=row.get('provenance',{}).get('support','') if row['component']=='knowledge' else row['messages'][0]['content'].split('Question:',1)[0]
            if v.get('answerable',True) and (len(answer)<4 or not re.search(r'(?<!\w)'+re.escape(answer)+r'(?!\w)',support,re.I)):continue
        pools[row['component']].append(row)
    for component,count in quotas.items():
        rng.shuffle(pools[component])
        if len(pools[component])<count:raise ValueError('Insufficient replay: '+component)
        for row in pools[component][:count]:replay.append(dict(row,provenance=dict(row.get('provenance',{}),replay_of='parent2000 original training')))
    rng.shuffle(replay);generated['replay']=replay;counts['replay']=dict(Counter(r['component'] for r in replay))
    output.mkdir()
    for split,rows in generated.items():(output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest=dict(format='parent2000-fresh-conversations-v1',parent_manifest=file_sha256(PARENT/'manifest.json'),tokenizer=tok.fingerprint,counts=counts,
        maximum_new_sequence=maximum,truncated=0,rejected=dict(rejected),source_training_hash=file_sha256(ASSETS/'corpus/train.jsonl'),
        evaluation_exclusions={split:file_sha256(ASSETS/f'corpus/{split}.jsonl') for split in ('dev','test')},
        limitations=['Fresh examples are computed and templated, not a broad human-chat corpus.','Replay includes previously curated public responses; not all public facts have been independently verified.','Fresh evaluation separates wording and entities but shares task families. Historical 320/704 panels are reused.'],
        files={p.name:file_sha256(p) for p in output.iterdir()})
    atomic_json(output/'manifest.json',manifest);atomic_json(REPORT/'data_audit.json',manifest)
    print(json.dumps(dict(counts=counts,maximum_new_sequence=maximum,truncated=0),indent=2))

if __name__=='__main__':prepare()
