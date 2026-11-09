"""Small training supplement and separately authored evaluation instructions."""
import json
import re
from .grounded_corpus import pair
from .quality_gate import canonical


def json_equal(a,b):
    if isinstance(a,bool) or isinstance(b,bool):return type(a)==type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return a==b
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(json_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(json_equal(x,y) for x,y in zip(a,b))
    return a==b


def check(text,rules):
    passed=[]
    for rule in rules:
        kind=rule['kind'];value=text.strip()
        if kind=='exact':ok=value in rule['answers']
        elif kind=='answer':ok=canonical(value) in {canonical(x) for x in rule['answers']}
        elif kind=='contains':ok=all((' '+canonical(v)+' ') in (' '+canonical(value)+' ') for v in rule['values'])
        elif kind=='forbidden':ok=not any((' '+canonical(v)+' ') in (' '+canonical(value)+' ') for v in rule['values'])
        elif kind=='json':
            try:ok=json_equal(json.loads(value),rule['value'])
            except (ValueError,TypeError):ok=False
        elif kind=='case':ok=bool(re.search('[a-zA-Z]',value)) and value==getattr(value,rule['mode'])()
        elif kind=='bullets':ok=len(re.findall(r'^\s*[-*+]\s+\S',value,re.M))==rule['count']
        elif kind=='numbered':ok=len(re.findall(r'^\s*\d+[.)]\s+\S',value,re.M))==rule['count']
        elif kind=='length':
            if rule['unit']=='bullets':n=len(re.findall(r'^\s*[-*+]\s+\S',value,re.M))
            elif rule['unit']=='words':n=len(value.split())
            elif rule['unit']=='sentences':n=len(re.findall(r'[.!?](?:["\u201d\x27)]*)?(?:\s|$)',value))
            else:raise ValueError('Unsupported length unit')
            ok=rule.get('min',0)<=n<=rule.get('max',10**9)
        elif kind=='sentences':
            n=len(re.findall(r'[.!?](?:["\u201d\x27)]*)?(?:\s|$)',value));ok=rule.get('min',rule.get('count',1))<=n<=rule.get('max',rule.get('count',1))
        elif kind=='suffix':ok=value.endswith(rule['value'])
        else:raise ValueError('Unknown balanced check '+kind)
        passed.append(bool(ok))
    return bool(passed) and all(passed)


def exact(answer):return [dict(kind='exact',answers=[answer])]
def contains(*values):return dict(kind='contains',values=list(values))


def training_rows():
    """Single operations only; evaluation uses different tasks/compositions."""
    vocabulary=['amber','bridge','cloud','desk','elm','field','garden','harbor','island','jar','kite','leaf','maple','notebook','oak','pond']
    for i in range(1200):
        items=[vocabulary[(i+j*3)%len(vocabulary)]+(' '+str(i+21) if j==0 else '') for j in range(3)]
        family=i%6
        if family==0:
            request='Convert this label to uppercase and return only the label: '+' '.join(items)
            answer=' '.join(items).upper();rules=[dict(kind='exact',answers=[answer])]
        elif family==1:
            request='Put these three supplied items into a bullet list: '+', '.join(items)
            answer='\n'.join('- '+x for x in items);rules=[contains(*items),dict(kind='bullets',count=3)]
        elif family==2:
            request='Return these supplied words in alphabetical order, separated by commas: '+', '.join(items)
            answer=', '.join(sorted(items));rules=[dict(kind='exact',answers=[answer])]
        elif family==3:
            value=dict(first=items[0],second=items[1],third=items[2])
            request='Represent these three values as JSON using the keys first, second, third: '+', '.join(items)
            answer=json.dumps(value);rules=[dict(kind='json',value=value)]
        elif family==4:
            request='Copy the entire identifier exactly: '+' / '.join(items)
            answer=' / '.join(items);rules=[dict(kind='exact',answers=[answer])]
        else:
            request='Number each supplied item in its original order: '+', '.join(items)
            answer='\n'.join(f'{n+1}. {x}' for n,x in enumerate(items));rules=[contains(*items),dict(kind='numbered',count=3)]
        yield dict(component='verified',group='redo-single-operation:'+str(i),messages=pair(request,answer),
            verification=dict(checks=rules),family='train/single-'+str(family),
            provenance=dict(repo='local/balanced-instruction-supplement',revision='20261102-redo-v1',license='CC0-1.0'))


def panel(split):
    """Explicit reference tasks; no event-template generator is shared with train."""
    if split=='dev':
        specs=[
          ('agreement','Fix only the verb agreement. Return the corrected sentence: The boxes is beside the door.','The boxes are beside the door.',exact('The boxes are beside the door.')),
          ('spelling','Correct the spelling and return only the sentence: Plese put teh folder on the shelf.','Please put the folder on the shelf.',exact('Please put the folder on the shelf.')),
          ('two-facts','The museum opens at 10:30 and closes at 17:00. Give both times in that order, separated by a slash.','10:30 / 17:00',exact('10:30 / 17:00')),
          ('whole-label','Copy the complete label without dropping any part: Cedar / Nora / 482','Cedar / Nora / 482',[dict(kind='exact',answers=['Cedar / Nora / 482'])]),
          ('missing','A notice says the reading room opens on Friday. No other information is given. What is the entry fee? If absent, answer Not provided.','Not provided.',exact('Not provided.')),
          ('lower-extract','From this note give only the destination, in lowercase: Destination: Green Harbor. Departure: noon.','green harbor',[dict(kind='exact',answers=['green harbor']),dict(kind='case',mode='lower')]),
          ('json-pair','A parcel weighs 4 kilograms and contains 6 books. Return JSON using the keys weight_kg and books, with numeric values.','{"weight_kg": 4, "books": 6}',[dict(kind='json',value=dict(weight_kg=4,books=6))]),
          ('list-evidence','A school notice asks families to bring a notebook and a water bottle. Give exactly two bullet points listing those items.','- notebook\n- water bottle',[contains('notebook','water bottle'),dict(kind='bullets',count=2)]),
          ('ordered-numbered','Number these names alphabetically: Zoe, Amir, Bella. Give one name per line.','1. Amir\n2. Bella\n3. Zoe',[dict(kind='exact',answers=['1. Amir\n2. Bella\n3. Zoe'])]),
          ('summary-time','Summarize in one sentence, keeping the day and time exactly: The book club meeting moved from Wednesday to Friday. It now starts at 14:15.','The book club now meets on Friday at 14:15.',[contains('Friday','14:15'),dict(kind='forbidden',values=['Wednesday']),dict(kind='sentences',count=1)]),
          ('index-choice','The shelf labels are North, South, East, West. Return only the third label in uppercase.','EAST',[dict(kind='exact',answers=['EAST'])]),
          ('omit-private','Booking: customer Elena; room Birch Suite. Return only the room name, without the customer name.','Birch Suite',[dict(kind='answer',answers=['Birch Suite']),dict(kind='forbidden',values=['Elena'])]),
          ('negation','The green bag contains pencils. The blue bag does not contain pencils. Which bag contains pencils? Answer with just its color.','green',exact('green')),
          ('classification','For this task a triangle has three sides and a square has four. Which has four sides? Give only its name.','square',exact('square')),
          ('contrast','Kian leaves on Monday. Rosa leaves on Thursday. Give Rosa\'s departure day and do not mention Kian.','Thursday',[dict(kind='answer',answers=['Thursday']),dict(kind='forbidden',values=['Kian'])]),
          ('only-rewrite','Correct the grammar only. Return the revised sentence without commentary: We was waiting when the bus arrive.','We were waiting when the bus arrived.',exact('We were waiting when the bus arrived.')),
          ('combined-copy','Read the label Sunflower Garden. Return it in uppercase followed by a period.','SUNFLOWER GARDEN.',[dict(kind='exact',answers=['SUNFLOWER GARDEN.'])]),
          ('suffix-content','The library closes at 18:00. Write one sentence giving that time, then finish with the exact sentence Plan ahead.','The library closes at 18:00. Plan ahead.',[contains('18:00'),dict(kind='suffix',value='Plan ahead.'),dict(kind='sentences',count=2)]),
          ('factual-edit','Rewrite clearly without adding facts: The cafe is on Pine Street and opening time is 08:45 and it is closed Sundays.','The cafe on Pine Street opens at 08:45 and is closed on Sundays.',[contains('Pine Street','08:45','Sundays'),dict(kind='forbidden',values=['Saturday','09:00'])]),
          ('short-explanation','Use this passage: A wet towel loses water through evaporation. Air movement helps carry water vapor away. Explain in one or two sentences why moving air helps dry the towel.','Moving air carries water vapor away, helping water evaporate from the towel.',[contains('water','away'),dict(kind='sentences',min=1,max=2)])]
    elif split=='test':
        specs=[
          ('agreement-two','Correct both agreement errors and return only the sentence: These chairs is heavy but that table are light.','These chairs are heavy but that table is light.',exact('These chairs are heavy but that table is light.')),
          ('spelling-two','Correct the misspellings without changing the meaning: The adress is writen on the envelope.','The address is written on the envelope.',exact('The address is written on the envelope.')),
          ('ordered-facts','A train leaves at 07:20 and reaches Lakeside at 09:05. Give the arrival time first, then the departure time, separated by a slash.','09:05 / 07:20',exact('09:05 / 07:20')),
          ('label-punctuation','Return this complete label exactly: Maple | Imani | 736','Maple | Imani | 736',[dict(kind='exact',answers=['Maple | Imani | 736'])]),
          ('absent-location','A note gives the concert date as July 12 and the start time as 19:30. It gives no venue. What is the venue? Reply Information missing if not given.','Information missing.',exact('Information missing.')),
          ('upper-extract','Service record: vehicle color silver; appointment Tuesday. Output only the vehicle color in uppercase.','SILVER',[dict(kind='exact',answers=['SILVER'])]),
          ('json-arrays','A basket contains pears and plums. Return a JSON array of those two fruit names in the stated order.','["pears", "plums"]',[dict(kind='json',value=['pears','plums'])]),
          ('bullets-upper','Supplies: tape and scissors. List both as exactly two uppercase bullet points.','- TAPE\n- SCISSORS',[contains('tape','scissors'),dict(kind='bullets',count=2),dict(kind='case',mode='upper')]),
          ('reverse-numbered','Number these three words in reverse order: morning, afternoon, evening.','1. evening\n2. afternoon\n3. morning',[dict(kind='exact',answers=['1. evening\n2. afternoon\n3. morning'])]),
          ('summary-revision','The workshop was planned for May 2 at 11:00. It has moved to May 9 at 13:30. State only the new date and time in one sentence.','The workshop is now on May 9 at 13:30.',[contains('May 9','13:30'),dict(kind='forbidden',values=['May 2','11:00']),dict(kind='sentences',count=1)]),
          ('filtered-choice','From red, oak, blue, elm select only the tree names. Return the names separated by a comma in their original order.','oak, elm',exact('oak, elm')),
          ('exclude-field','Contact card: name Hugo; town Riverbend; phone 555-0182. Give the town alone and omit the phone number.','Riverbend',[dict(kind='answer',answers=['Riverbend']),dict(kind='forbidden',values=['555'])]),
          ('negative-correction','The first drawer is empty. The second drawer holds the keys. Which drawer should be opened to get the keys? Give just first or second.','second',exact('second')),
          ('classification-supplied','In this exercise a dax is a striped object and a wug is a plain object. What is the name for a plain object?','wug',exact('wug')),
          ('factual-distinction','Mina bought two notebooks. Oren bought five folders. Who bought folders? Return only the name.','Oren',exact('Oren')),
          ('rewrite-tense','Fix the verb forms and return only the corrected sentence: She have finished and they was ready.','She has finished and they were ready.',exact('She has finished and they were ready.')),
          ('copy-lower-period','Take the phrase HARBOR WALK, convert it to lowercase, and end it with a period.','harbor walk.',[dict(kind='exact',answers=['harbor walk.'])]),
          ('ending-after-fact','A shop opens at 09:15. State this in one sentence. Add a separate final sentence that is exactly Doors open soon.','The shop opens at 09:15. Doors open soon.',[contains('09:15'),dict(kind='suffix',value='Doors open soon.'),dict(kind='sentences',count=2)]),
          ('meaning-preservation','Rewrite this as one clear sentence, retaining the price and location: Entry is 8 credits and the gate is beside Willow House and payment is at the gate.','Entry costs 8 credits, payable at the gate beside Willow House.',[contains('8','credits','Willow House','gate'),dict(kind='sentences',count=1)]),
          ('cause-from-passage','Passage: A lid reduces water loss because it blocks escaping water vapor. In one or two sentences explain why a lid helps keep water in the pot.','The lid blocks escaping water vapor, reducing water loss.',[contains('water','vapor'),dict(kind='sentences',min=1,max=2)])]
    else:raise ValueError('Panel split must be dev or test')
    result=[]
    for i,(family,prompt,answer,rules) in enumerate(specs):
        result.append(dict(id=split+'-'+family,family=split+'/'+family,group='redo-panel:'+split+':'+str(i),
            component='verified',messages=pair(prompt,answer),verification=dict(checks=rules),limit=256,
            provenance=dict(repo='local/independent-behavior-panels',revision='20261102-redo-v1',license='CC0-1.0')))
    # Different follow-up compositions, outside the authored single-operation training bank.
    if split=='dev':
        messages=pair('Draft a notice: the class meets at Cedar Hall on Monday at 09:00.','The class meets at Cedar Hall on Monday at 09:00.')
        messages+=pair('Move it to Thursday and keep the location and time. Give only the revised notice.','The class meets at Cedar Hall on Thursday at 09:00.')
        rules=[contains('Cedar Hall','Thursday','09:00'),dict(kind='forbidden',values=['Monday'])]
    else:
        messages=pair('Make a note: the club meets in Elm Room at 16:00 on Friday.','The club meets in Elm Room at 16:00 on Friday.')
        messages+=pair('Use Saturday instead and change the room to Oak Room. Keep the time. Return only the revised note.','The club meets in Oak Room at 16:00 on Saturday.')
        rules=[contains('Oak Room','Saturday','16:00'),dict(kind='forbidden',values=['Elm Room','Friday'])]
    result.append(dict(id=split+'-followup',family=split+'/followup-composition',group='redo-panel:'+split+':followup',
        component='verified',messages=messages,verification=dict(checks=rules),limit=512,
        provenance=dict(repo='local/independent-behavior-panels',revision='20261102-redo-v1',license='CC0-1.0')))
    for row in result:
        if not check(row['messages'][-1]['content'],row['verification']['checks']):raise ValueError('Invalid panel reference: '+row['id'])
    return result
