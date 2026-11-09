"""SFT-039: natural short instructions with small, explicit contrast pairs.

Source responses are preserved, not teacher-rewritten. Ratings and lexical
screening are not proof of factual correctness. No benchmark/probe targets enter
training. Held-out panels check actual generated histories in both presentations.
"""
from collections import Counter, defaultdict
import json
import random
import re
from pathlib import Path
from sml_v1.common import atomic_json, file_sha256, read_json
from sml_v1.tokenization import Tokenizer
from .curate import local
from .chat_search import SOURCES, POLICY, DRIFT
from .reliable_chat import cached, encode
from .balanced import oasst_paths
from .conversation import split_group
from .quality import digest, normalized, repeated
from .evidence_instructions import row, value, exact, check_contract, validate_row, benchmark_exclusions

FOLDER='reports/sft/SFT-039-natural-short-contrast-20261008'
SEED=263900
BUCKETS=('conversation','source_qa','intent','text_contrast','python')


def short_check(contract, text):
    if contract['kind']=='request':
        aliases={'traceback':('traceback','error','code'), 'post':('post','text','article'),
                 'script':('script','code'), 'collection':('collection','items','list')}
        return (bool(text.strip()) and len(text.split())<=35
                and bool(re.search(r'\b(?:paste|share|send|provide|show)\b',text,re.I))
                and all(any(re.search(r'\b'+re.escape(a)+r'\b',text,re.I) for a in aliases.get(obj,(obj,))) for obj in contract['objects'])
                and not re.search(r'\b(?:fixed it|the error is|revised version)\b',text,re.I))
    if contract['kind']!='meaning':return check_contract(contract,text)
    if not text.strip() or repeated(text) or len(text.split())>contract.get('max_words',45):return False
    return (all(re.search(p,text,re.I) for p in contract.get('require',[]))
            and not any(re.search(p,text,re.I) for p in contract.get('forbid',[])))


def meaning(*patterns, forbid=(), max_words=45):
    return dict(kind='meaning',require=list(patterns),forbid=list(forbid),max_words=max_words)


def make(bucket,group,turns,family):
    # Reuse serialization schema, never SFT-038 data or targets.
    r=row(bucket,group,turns,dict(repo='local/natural-short-contrast',revision='v1',license='CC0-1.0',split='authored',method='Reviewed concrete request/target pairs; bounded Python oracle'),family)
    r['component']='python' if bucket=='python' else 'chat'
    return r


def python_pairs():
    """Twenty genuinely different operations, contrasted under shared names.

    No thousand-row parameter multiplier; three identifier presentations per
    pair remain correlated and are reported as such. Split whole operation pairs.
    Tests include unrequested-operation detection, zero and negative boundaries.
    """
    specs=[
      ('case','text','remove leading and trailing whitespace, preserving letter case','text.strip()',
       'remove leading and trailing whitespace, then lowercase the text','text.strip().lower()', [['  MiXeD  '],[' a '],[''],['  '],['ABC']],['MiXeD','a','','','ABC'],['mixed','a','','','abc']),
      ('case_upper','text','uppercase the entire text without trimming it','text.upper()',
       'trim surrounding whitespace, then uppercase the text','text.strip().upper()', [[' xY '],['z'],[''],[' a b ']],[' XY ','Z','',' A B '],['XY','Z','','A B']),
      ('prefix','text','return whether the text begins with the exact prefix "key:"','text.startswith("key:")',
       'return whether the text ends with the exact suffix ":key"','text.endswith(":key")',[['key:a'],['a:key'],['key'],[''],['key:key']],[True,False,False,False,True],[False,True,False,False,True]),
      ('presence','items','return the number of items','len(items)',
       'return True if the list is nonempty, otherwise False','bool(items)',[[[]],[[0]],[[1,2]],[[False]]],[0,1,2,1],[False,True,True,True]),
      ('zero_filter','items','keep only strictly positive numbers, preserving order','[x for x in items if x > 0]',
       'keep nonnegative numbers, including zero, preserving order','[x for x in items if x >= 0]',[[[-2,0,4,1]],[[]],[[0]],[[2,-1,2]]],[[4,1],[],[],[2,2]],[[0,4,1],[],[0],[2,2]]),
      ('sign_filter','items','keep only negative numbers in their original order','[x for x in items if x < 0]',
       'keep only nonpositive numbers in their original order','[x for x in items if x <= 0]',[[[-3,0,2,-1]],[[]],[[0]],[[2]]],[[-3,-1],[],[],[]],[[-3,0,-1],[],[0],[]]),
      ('parity','items','keep even integers in their original order','[x for x in items if x % 2 == 0]',
       'keep odd integers in their original order','[x for x in items if x % 2 != 0]',[[[-3,-2,0,1,4]],[[]],[[5,5,2]]],[[-2,0,4],[],[2]],[[-3,1],[],[5,5]]),
      ('reversal','text','reverse the text, preserving spaces and letter case','text[::-1]',
       'lowercase the text, then reverse it, preserving spaces','text.lower()[::-1]',[['Ab C'],[''],[' X ']],['C bA','',' X '],['c ba','',' x ']),
      ('sorting','items','sort numbers from smallest to largest, keeping duplicates','sorted(items)',
       'sort numbers from largest to smallest, keeping duplicates','sorted(items, reverse=True)',[[[3,1,3,2]],[[]],[[-2,0,1]]],[[1,2,3,3],[],[-2,0,1]],[[3,3,2,1],[],[1,0,-2]]),
      ('spaces','text','replace each underscore with a space','text.replace("_", " ")',
       'remove underscores without adding spaces','text.replace("_", "")',[['a_b'],['__'],[''],[' a_ ']],['a b','  ','',' a  '],['ab','','',' a ']),
      ('equality','left, right','return True if the two values are equal, otherwise False','left == right',
       'return True if the two values differ, otherwise False','left != right',[[2,2],[2,3],['a','a'],['','x']],[True,False,True,False],[False,True,False,True]),
      ('square','number','return the square of the number','number * number',
       'return twice the number','number * 2',[[0],[3],[-2],[1]],[0,9,4,1],[0,6,-4,2]),
      ('sum_count','items','return the sum of the numbers; an empty list gives zero','sum(items)',
       'return the count of the numbers; an empty list gives zero','len(items)',[[[]],[[3,4]],[[-2,5,0]]],[0,7,3],[0,2,3]),
      ('first_last','items','return the first element, or None if the list is empty','items[0] if items else None',
       'return the last element, or None if the list is empty','items[-1] if items else None',[[[]],[[0,4]],[[None,2]],[[5]]],[None,0,None,5],[None,4,2,5]),
      ('join_split','text','split the text into whitespace-separated words','text.split()',
       'split the text only at commas, retaining empty fields','text.split(",")',[['a, b'],[''],['a,,b'],[' a  b ']], [['a,','b'],[],['a,,b'],['a','b']],[['a',' b'],[''],['a','','b'],[' a  b ']]),
      ('lengths','items','return the length of every string, preserving order','[len(x) for x in items]',
       'return only strings with more than two characters, preserving order','[x for x in items if len(x) > 2]',[[[]],[['','ab','abcd']],[['cat','a']]], [[],[0,2,4],[3,1]],[[],['abcd'],['cat']]),
      ('boundary','number','return True when number is at least ten, otherwise False','number >= 10',
       'return True when number is greater than ten, otherwise False','number > 10',[[9],[10],[11],[-1]],[False,True,True,False],[False,False,True,False]),
      ('absolute','number','return the absolute value','abs(number)',
       'return the negated value','-number',[[-3],[0],[4]],[3,0,4],[3,0,-4]),
      ('key_default','mapping, key','return the value under key, or None if key is absent','mapping.get(key)',
       'return True if key exists in the mapping, otherwise False','key in mapping',[[{'a':0},'a'],[{'a':None},'a'],[{},'a'],[{'a':2},'b']],[0,None,None,None],[True,True,False,False]),
      ('single_replace','text','replace only the first occurrence of "cat" with "dog"','text.replace("cat", "dog", 1)',
       'replace every occurrence of "cat" with "dog"','text.replace("cat", "dog")',[['cat cat'],['a cat'],[''],['catcat']],['dog cat','a dog','','dogcat'],['dog dog','a dog','','dogdog']),
    ]
    for family,args,u1,b1,u2,b2,inputs,e1,e2 in specs:
        group='039:program-pair:'+family
        for fn in ('process','compute','apply_change'):
            for i,(u,body,expected) in enumerate(((u1,b1,e1),(u2,b2,e2))):
                tests=[dict(args=x,expected=y) for x,y in zip(inputs,expected)]
                contract=dict(kind='python',function=fn,tests=tests)
                answer=f'```python\ndef {fn}({args}):\n    return {body}\n```'
                prompt=f'Write Python function {fn}({args}) to {u}. Return only a Python code block.'
                yield make('python',group,[(prompt,answer,contract)],family+':'+str(i))


def intent_rows():
    # Each pair contrasts different intents; not 489 greeting-task-goodbye copies.
    pairs=[
      ('Could we have a quick conversation?', 'Of course. What would you like to talk about?', meaning(r'\b(?:talk|discuss|mind)\b',forbid=(r'\b(?:goodbye|bye)\b',))),
      ('What sorts of things can you do for someone?', 'I can explain ideas, answer questions, help with writing, and work on simple Python code.', meaning(r'\b(?:explain|questions|writing|python)\b')),
      ('Thanks, your explanation cleared that up.', "You're welcome!", meaning(r'\b(?:welcome|glad)\b')),
      ('I am signing off for the night.', 'Good night! Take care.', meaning(r'\b(?:good night|take care|bye)\b')),
      ('Can you make an email clearer for me?', 'Please paste the email you want me to revise.', dict(kind='request',objects=['email'])),
      ('I have a Python exception. Can you work out the cause?', 'Please share the code and the error message.', dict(kind='request',objects=['code','error'])),
      ('Can you arrange some items alphabetically?', 'Please send the list of items you want sorted.', dict(kind='request',objects=['list'])),
      ('Can you give me a summary of an article?', 'Please share the article you want summarized.', dict(kind='request',objects=['article'])),
      ('I want to practice asking you questions.', 'Go ahead. What would you like to ask?', meaning(r'\b(?:ask|question)\b',forbid=(r'\b(?:goodbye|grammar tutor)\b',))),
      ('Are you a person with a family?', 'No. I am an AI assistant and do not have a family.', meaning(r'\b(?:no|not)\b',r'\b(?:AI|assistant)\b')),
      ('Are you able to help me write code?', 'Yes, I can help with simple Python programs and explain how they work.', meaning(r'\bpython\b')),
      ('Can we keep talking for a bit?', 'Sure. What is on your mind?', meaning(r'\b(?:mind|talk|discuss)\b',forbid=(r'\b(?:goodbye|bye)\b',))),
      ('What is going on in this chat?', 'I am here to answer your questions. What would you like to discuss?', meaning(r'\b(?:questions|discuss)\b',forbid=(r'\b(?:goodbye|bye)\b',))),
      ('Can you rewrite a sentence if I give you one?', 'Yes. Share the sentence and tell me what you want to change.', dict(kind='request',objects=['sentence'])),
      ('Would you explain a topic I find confusing?', 'Yes. Tell me the topic and what part is confusing.', meaning(r'\b(?:tell|share)\b',r'\btopic\b')),
      ('Can you tell me a good question to ask about a book?', 'You could ask what motivates its main character.', meaning(r'\b(?:character|book)\b')),
    ]
    for i,(u,a,c) in enumerate(pairs):
        yield make('intent','039:intent:'+str(i),[(u,a,c)],str(i))
    # Complete ordinary exchanges, with relevant context and no compulsory farewell.
    for topic,task,answer,contract in [
      ('a reminder','Please turn "water the plants tomorrow" into a polite reminder.','Please remember to water the plants tomorrow.',meaning(r'water',r'plants',r'tomorrow')),
      ('a shopping list','My list contains rice and milk. Which two things am I buying?','Rice and milk.',value('Rice and milk')),
      ('a short note','Make this more polite: Send the file.','Please send the file.',exact('Please send the file.')),
      ('a heading','Give this heading in uppercase: quiet garden.','QUIET GARDEN',exact('QUIET GARDEN')),
      ('a question','In my note, Maya lends a pen to Jules. Who receives it?','Jules.',value('Jules')),
      ('a tiny program','What does len([]) return in Python? Answer with just the number.','0',value('0')),
    ]:
        turns=[(f'I would like help with {topic}.',f'Sure. Please share {topic}.',meaning(r'\b(?:share|send|tell)\b')),(task,answer,contract)]
        yield make('intent','039:exchange:'+digest(topic),turns,topic)


def text_pairs():
    # Concrete contrasting instructions on the same text. Groups stay together.
    items=[('night sky','dawn'),('green leaf','root'),('small garden','pond'),('quiet road','path'),('amber lamp','window'),('silver cup','shelf'),('fresh bread','plate'),('plain paper','folder'),('wooden door','latch'),('soft wool','cotton'),('blue scarf','coat'),('stone wall','gate')]
    for a,b in items:
        group='039:text-pair:'+digest(a)
        for prompt,answer,c in [
          (f'Write only the uppercase form of {a}.',a.upper(),exact(a.upper())),
          (f'Write only the lowercase form of {a.upper()}.',a,exact(a)),
          (f'Write {a} and {b} on two separate lines, with no bullets.',a+'\n'+b,exact(a+'\n'+b)),
          (f'Write {a} and {b} as two bullet points.',f'- {a}\n- {b}',dict(kind='bullets',items=[a,b])),
        ]:yield make('text_contrast',group,[(prompt,answer,c)],prompt.split('only')[0])


def transfer_cases(split):
    """Frozen independently authored probes, never added to training.

    Development and test use different concrete requests/records/program specs;
    recurring intent/operation concepts make these diagnostic, not benchmark
    accuracy. No system and training-system are scored separately.
    """
    dev=split=='dev';tag='dev' if dev else 'test';rows=[]
    def add(bucket,name,turns):rows.append(make(bucket,'039:'+tag+':'+bucket+':'+name,turns,name))
    if dev:
        social=[('I am available to chat now. Are you?', 'Sure. What would you like to discuss?'),('Can we just talk casually?', 'Of course. What is on your mind?'),('Describe your useful skills in a short sentence.','I can explain topics, help with writing, answer questions, and assist with Python.'),('I appreciate the explanation you gave.','You are welcome!')]
        clar=[('Please improve an invitation for me.', ['invitation']),('Can you look at a traceback for me?', ['traceback']),('I want you to summarize a blog post.', ['post']),('Would you sort my collection?', ['collection'])]
        source=[('A note says Sera owns a red notebook. Ivo owns a green folder. Who owns the notebook?', 'Sera'),('The bus leaves at 17:20. The train leaves at 18:05. What time does the bus leave?', '17:20'),('The record lists a price of 8 euros but does not mention delivery time. When will delivery happen?', 'Unknown'),('There are two bowls. Only the metal bowl contains peaches. Which bowl contains peaches?', 'metal')]
        word='mint';names=['pine','ash'];who='Tess';project='Lantern';old='Friday';new='Sunday'
    else:
        social=[('I have a moment for a relaxed conversation.','Sure. What would you like to discuss?'),('Are we still chatting?', 'Yes. What is on your mind?'),('What kinds of tasks could I ask you to do?', 'I can answer questions, help with writing, explain topics, and assist with Python.'),('That made the idea clearer, much appreciated.','You are welcome!')]
        clar=[('Can you help polish a newsletter?', ['newsletter']),('Could you inspect my failing script?', ['script']),('Please shorten my announcement.', ['announcement']),('Can you explain a chart I have?', ['chart'])]
        source=[('The diary says Aron carries a yellow backpack. Elin carries a blue case. Who carries the backpack?', 'Aron'),('The ferry departs at 09:35. The flight departs at 10:10. When does the ferry leave?', '09:35'),('The listing gives a weight of 2 kg but no purchase date. When was it bought?', 'Unknown'),('Only the glass jar holds cherries; the tin jar is empty. Which jar holds cherries?', 'glass')]
        word='fern';names=['birch','elm'];who='Milo';project='Beacon';old='Tuesday';new='Thursday'
    for i,(q,a) in enumerate(social):
        c=meaning(r'\b(?:welcome|glad)\b') if i==3 else meaning(r'\b(?:explain|writing|python|questions)\b') if i==2 else meaning(r'\b(?:mind|discuss|talk)\b',forbid=(r'\b(?:bye|goodbye|take care)\b',))
        add('social',str(i),[(q,a,c)])
    for i,(q,objects) in enumerate(clar):add('clarification',str(i),[(q,'Please share the '+objects[0]+'.',dict(kind='request',objects=objects))])
    for i,(q,a) in enumerate(source):
        instruction=' Reply Unknown if it is not stated; otherwise give only the requested value.'
        add('source_qa',str(i),[(q+instruction,a,value(a))])
    add('followup','topic',[(f'My project is named {project}. What is the name? Give only the name.',project,value(project)),('Say the project name again, and nothing else.',project,value(project))])
    add('followup','date',[(f'{who} is visiting on {old}. Give only the day.',old,value(old)),(f'Change that visit to {new}. Which day is it now? Give only the day.',new,value(new))])
    add('followup','contrast',[(f'Write only {word.upper()}.',word.upper(),exact(word.upper())),('Now give the same word in lowercase, with nothing else.',word,exact(word))])
    add('followup','clarify_then_do',[(f'Would you lowercase a phrase for me?','Please share the phrase.',dict(kind='request',objects=['phrase'])),(f'Here it is: {project.upper()}. Give only the lowercase version.',project.lower(),exact(project.lower()))])
    add('format','bare',[(f'Copy only the word {word}, with no introduction.',word,exact(word))])
    add('format','lines',[(f'Put {names[0]} and {names[1]} on separate lines without bullets.', '\n'.join(names),exact('\n'.join(names)))])
    add('format','bullets',[(f'Use exactly two bullets to list {names[0]} and {names[1]}.', '- '+names[0]+'\n- '+names[1],dict(kind='bullets',items=names))])
    add('format','json',[(f'Return just a JSON list containing "{names[0]}" and "{names[1]}" in that order.',json.dumps(names),dict(kind='json',expected=names))])
    # Shared generic names and contrasting specs; tests are not shown in prompt.
    fn='handle' if dev else 'convert'
    for i,(spec,body,tests,args) in enumerate([
      ('return text in uppercase without removing any characters','text.upper()',[dict(args=[' aB '],expected=' AB '),dict(args=[''],expected=''),dict(args=['x'],expected='X')],'text') if dev else ('remove only trailing whitespace, preserving leading whitespace','text.rstrip()',[dict(args=[' aB '],expected=' aB'),dict(args=[''],expected=''),dict(args=[' q'],expected=' q')],'text'),
      ('return text in lowercase without removing any characters','text.lower()',[dict(args=[' aB '],expected=' ab '),dict(args=[''],expected=''),dict(args=['X'],expected='x')],'text') if dev else ('remove only leading whitespace, preserving trailing whitespace','text.lstrip()',[dict(args=[' aB '],expected='aB '),dict(args=[''],expected=''),dict(args=['q '],expected='q ')],'text'),
      ('return numbers greater than limit in their original order','[x for x in values if x > limit]',[dict(args=[[2,3,4],3],expected=[4]),dict(args=[[],3],expected=[]),dict(args=[[0,-1],-1],expected=[0])],'values, limit') if dev else ('keep numbers at least limit in their original order','[x for x in values if x >= limit]',[dict(args=[[4,3,2,3],3],expected=[4,3,3]),dict(args=[[],3],expected=[]),dict(args=[[-1,0,-2],-1],expected=[-1,0])],'values, limit'),
      ('return numbers less than limit in their original order','[x for x in values if x < limit]',[dict(args=[[2,3,4],3],expected=[2]),dict(args=[[],3],expected=[]),dict(args=[[0,-1],0],expected=[-1])],'values, limit') if dev else ('keep numbers at most limit in their original order','[x for x in values if x <= limit]',[dict(args=[[4,3,2,3],3],expected=[3,2,3]),dict(args=[[],3],expected=[]),dict(args=[[-1,0,-2],-1],expected=[-1,-2])],'values, limit'),
    ]):
        c=dict(kind='python',function=fn,tests=tests);code=f'```python\ndef {fn}({args}):\n    return {body}\n```'
        add('python',str(i),[(f'Implement {fn}({args}) in Python to {spec}. Give only a Python code block.',code,c)])
    assert all(len([m for m in r['messages'] if m['role']=='assistant'])==len(r['turn_contracts']) for r in rows)
    return rows


def panel(model,tok,rows,system=None):
    from .evaluate import generate
    scores=defaultdict(list);samples=[]
    for r in rows:
        history=[dict(role='system',content=system)] if system else [];turns=[]
        for m,c in zip([m for m in r['messages'] if m['role']=='user'],r['turn_contracts']):
            history.append(m);generated=generate(model,tok,history,limit=192,temperature=0)
            ok=short_check(c,generated['text']) and generated['stop_reason'] in ('turn_end','document_end')
            turns.append(dict(request=[dict(x) for x in history],result=generated,contract=c,passed=bool(ok)))
            history.append(dict(role='assistant',content=generated['text']))
        ok=all(t['passed'] for t in turns);scores[r['bucket']].append(ok)
        samples.append(dict(group=r['group'],bucket=r['bucket'],turns=turns,passed=bool(ok)))
    rates={k:sum(v)/len(v) for k,v in scores.items()}
    return dict(rates=rates,macro_pass=sum(rates.values())/len(rates),samples=samples)


def development(model,tok,prepared,plan):
    rows=read_json(Path(prepared)/'transfer-dev.json')['cases'];results={}
    for variant,system in [('no-system',None),('training-system',plan['conversation_system'])]:results[variant]=panel(model,tok,rows,system)
    return dict(metrics=results,macro_pass=sum(p['macro_pass'] for p in results.values())/2,
                scope='Small whole-answer, whole-history diagnostic panel. Intent checks are heuristics; Python checks are functional. Raw answers and an independent test review are required.')


def eligibility(result,plan):
    ref=result.get('selection_reference')
    retained=result['retention']['val_loss']<=plan['parent_val_loss']+plan['max_retention_increase']
    if ref is None:return dict(retained=retained,broad_nonregression=True,core_nonregression=True,core_floor=True,transfer_nonregression=True,transfer_gain=True)
    broad=result['chat_behavior'];core=('greeting','casual','capability','clarify','context')
    checks=dict(retained=retained,broad_nonregression=broad['macro_pass']>=ref['chat_behavior']['macro_pass']-1e-9,
      core_nonregression=all(broad['rates'][k]>=ref['chat_behavior']['rates'][k]-1e-9 for k in core))
    checks['core_floor']=all(broad['rates'][k]>=plan.get('core_chat_floor',.5) for k in ('greeting','casual','capability','clarify'))
    checks['transfer_nonregression']=all(result['natural_dev']['metrics'][v]['rates'][b]>=rate-1e-9
        for v,p in ref['natural_dev']['metrics'].items() for b,rate in p['rates'].items())
    checks['transfer_gain']=result['natural_dev']['macro_pass']>=ref['natural_dev']['macro_pass']+plan['min_transfer_gain']
    return checks


def ranking(result,plan):
    checks=eligibility(result,plan);eligible=all(checks.values())
    # An ineligible candidate cannot displace an eligible parent, regardless of
    # low target NLL or high narrow-schema score. Broad chat precedes transfer.
    return [int(checks['retained']),int(eligible),result['chat_behavior']['macro_pass'],result['natural_dev']['macro_pass'],-result['assistant_nll']]


def build(config):
    plan=read_json(config);folder=local(FOLDER);folder.mkdir(parents=True,exist_ok=True)
    target=folder/'catalog.jsonl'
    if target.exists():raise ValueError('Frozen catalog already exists; use a new setup revision')
    blocked,grams,benchmark_receipt=benchmark_exclusions()
    # Historical heldouts and inspection queries are protected. Reuse historical
    # TRAIN source rows transparently, never relabel former validation as train.
    protected=[];reserved_groups=set();reserved_sources=set()
    def remember(r):
        reserved_groups.add(r.get('group'))
        prov=r.get('provenance',{})
        for key in ('row','id','message_id','tree'):
            if prov.get(key) is not None:reserved_sources.add((prov.get('repo'),key,str(prov[key])))
        blocked.update(digest(m['content']) for m in r['messages'] if m['role']=='user')
    for p in local('reports/sft').rglob('catalog.jsonl'):
        if p.is_relative_to(folder):continue # Unlaunched drafts are not historical evidence.
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r.get('split') not in ('train',None):
                remember(r)
    for p in local('pretraining/data').glob('sft_*/dev.jsonl'):
        for name in ('dev.jsonl','test.jsonl'):
            q=p.parent/name
            if not q.exists():continue
            for line in q.read_text().splitlines():
                r=json.loads(line);remember(r)
    for pattern in ('**/probes.json','**/protocol.json'):
        for p in local('reports/sft').glob(pattern):
            try:obj=read_json(p)
            except ValueError:continue
            cases=obj.get('cases',obj.get('probes',[])) if isinstance(obj,dict) else obj if isinstance(obj,list) else []
            for c in cases:
                if isinstance(c,dict):
                    blocked.update(digest(q) for q in c.get('prompts',[]))
                    blocked.update(digest(m['content']) for m in c.get('messages',[]) if m['role']=='user')
            protected.append(dict(path=str(p.relative_to(local('.'))),sha256=file_sha256(p)))
    for s in ('dev','test'):
        for r in transfer_cases(s):blocked.update(digest(m['content']) for m in r['messages'] if m['role']=='user')
    from .behavior import cases
    for s in ('dev','test'):
        blocked.update(digest(m['content']) for r in cases(s,'instruction-v8') for m in r['messages'] if m['role']=='user')
    tok=Tokenizer(local(plan['tokenizer']));rows=[];seen=set();rejected=Counter();sources=[]
    unwanted=re.compile(r'\b(?:Open ?Assistant|ChatGPT|Claude|copyright|my name|my family|I live|grammar tutor|as an? (?:AI|language model)|treatment|disease|medic|tutor|religion|politic|president|favourite|favorite|personal opinion)\b',re.I)
    review_exclusions=list(plan.get('review_exclusions',[]))
    for p in local('configs').glob('sft*.json'):
        if p.resolve()==config.resolve():continue
        review_exclusions.extend(read_json(p).get('review_exclusions',[]))
    def accept(r,forced=None):
        try:
            if r['group'] in reserved_groups:raise ValueError('historical_heldout_group')
            prov=r['provenance']
            if any((prov.get('repo'),k,str(prov[k])) in reserved_sources for k in ('row','id','message_id','tree') if k in prov):raise ValueError('historical_heldout_source')
            if any(all(prov.get(k)==v for k,v in ex.items() if k not in ('reason','note')) for ex in review_exclusions):raise ValueError('manual_target_review')
            messages=r['messages'];users=[m['content'] for m in messages if m['role']=='user']
            if len(messages)>8 or any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)):raise ValueError('history_shape')
            if any(digest(q) in blocked and normalized(q) not in ('hi','hello','hey','hi there','hello there','good morning','good evening') for q in users):raise ValueError('reserved_request')
            for q in users:
                words=normalized(q).split()
                if any(' '.join(words[i:i+8]) in grams for i in range(len(words)-7)):raise ValueError('benchmark_overlap')
            if r['bucket']=='conversation':
                full=' '.join(m['content'] for m in messages)
                if POLICY.search(full) or DRIFT.search(full) or unwanted.search(full):raise ValueError('persona_specialist_time_or_identity')
                # "How can I help you today?" is an ordinary greeting, not a
                # claim about current events. Keep it while excluding timed
                # factual/recommendation requests and unsupported live facts.
                if re.search(r'\b(?:right now|currently|tonight|trending|latest|current)\b',full,re.I):raise ValueError('unsupported_current_claim')
                if re.search(r'\b(?:diagnosis|doctor|germs|vaccine|food poisoning|medication|diet|healthy|chicken|knives|behaviorist)\b',full,re.I):raise ValueError('outside_reviewed_task_scope')
                if any(len(m['content'].split())>90 or repeated(m['content']) for m in messages if m['role']=='assistant'):raise ValueError('long_or_repetitive')
                if any(len(m['content'].split())>160 for m in messages if m['role']=='user'):raise ValueError('long_input')
                if any('```' in m['content'] for m in messages):raise ValueError('unverified_source_code')
            for m,c in zip([m for m in messages if m['role']=='assistant'],r.get('turn_contracts',[])):
                if not short_check(c,m['content']):raise ValueError('failed_target_contract')
            encode(messages,tok,plan,r['component'],r['group'])
            key=digest(json.dumps(messages,sort_keys=True))
            if key in seen:raise ValueError('duplicate_history')
            seen.add(key);r['split']=forced or split_group(r['group'],SEED,12);rows.append(r)
        except ValueError as e:rejected[r['bucket']+':'+str(e)]+=1
    # Whole English reviewed human exchanges, one path per source tree.
    data,receipt=cached(SOURCES[1]);sources.append(dict(source=SOURCES[1],**receipt))
    def high_quality(r):
        return len(r['text'].split())<=90 and (r['role']!='assistant' or dict(zip(r['labels']['name'],r['labels']['value'])).get('quality',0)>=.8)
    for messages,group,identity in oasst_paths(data,node_filter=high_quality):
        accept(dict(bucket='conversation',component='chat',group=group,messages=messages,
          provenance=dict(SOURCES[1],**identity,transformation='Whole original reviewed English rank-zero nonsynthetic path; quality >= .8, short targets; no clipping')))
    data,receipt=cached(SOURCES[0]);sources.append(dict(source=SOURCES[0],**receipt))
    for i,r in enumerate(data):
        if r['source']!='everyday-conversations':continue
        messages=[dict(m) for m in r['messages']]
        # Never cut a long answer or take a greeting out of its task context.
        while messages and messages[-1]['role']=='user':messages.pop()
        topical=next((m['content'] for m in messages if m['role']=='user' and normalized(m['content']) not in ('hi','hello','hey','hi there','hello there','good morning','good evening')),messages[0]['content'])
        accept(dict(bucket='conversation',component='chat',group='039:natural-topic:'+digest(topical),messages=messages,
          provenance=dict(SOURCES[0],row=i,source_subset=r['source'],transformation='Complete original short everyday history; no answer clipping; source is synthetic')))
    data,receipt=cached(SOURCES[2]);sources.append(dict(source=SOURCES[2],**receipt))
    for i,r in enumerate(data):
        if r['category']!='closed_qa' or not r['context'] or not 20<=len(r['context'].split())<=160:continue
        a=r['response'].strip();q=r['instruction'].strip();ctx=r['context']
        # Complete original answers only, exact evidence spans and a simple
        # explicit direct question; reject explanation/list targets and metadata.
        if not 1<=len(a.split())<=12 or normalized(a) not in normalized(ctx) or not q.endswith('?'):continue
        if POLICY.search(q+' '+ctx) or unwanted.search(q+' '+ctx):continue
        messages=[dict(role='user',content=q+'\n\nUse this passage:\n'+ctx),dict(role='assistant',content=a)]
        group='039:dolly-context:'+digest(normalized(ctx))
        accept(dict(bucket='source_qa',component='qa',group=group,messages=messages,answers=[a],answerable=True,
           provenance=dict(SOURCES[2],row=i,category=r['category'],context_sha256=digest(ctx),transformation='Whole source passage, original question and complete original answer; normalized evidence alignment, not universal fact verification')))
    for r in (*list(intent_rows()),*list(text_pairs()),*list(python_pairs())):accept(r,forced='train')
    selected=[]
    for split in ('train','dev','test'):
        for bucket in BUCKETS:
            available=[r for r in rows if r['split']==split and r['bucket']==bucket]
            available.sort(key=lambda r:digest('039'+r['group']+json.dumps(r['messages'],sort_keys=True)))
            cap=plan['bucket_caps'][bucket] if split=='train' else 30
            selected.extend(available[:cap])
    # Explicit authored training contrasts do not furnish the transfer heldouts.
    # Natural dev/test components may be sparse, so use source rows only for NLL.
    for s in ('dev','test'):
        if sum(r['split']==s for r in selected)<16:raise ValueError('Insufficient natural '+s+' coverage')
    groups={s:{r['group'] for r in selected if r['split']==s} for s in ('train','dev','test')}
    if any(groups[a]&groups[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):raise ValueError('Group leakage')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    info=dict(format='opensml-natural-short-contrast-v1',sha256=file_sha256(target),source_receipts=sources,protected_files=protected,
      benchmark_exclusions=benchmark_receipt,rejected=dict(rejected),counts={s:dict(Counter(r['bucket'] for r in selected if r['split']==s)) for s in groups},
      limitations=['Source ratings, heuristics and normalized evidence alignment are not independent verification of all facts.','Some historical TRAIN source rows may be replayed; historical heldouts and inspection requests are excluded.','Authored intent/contrast/Python coverage is small and correlated; no thousands of renamed programs are counted as new skills.'])
    atomic_json(folder/'catalog.json',info)
    for s in ('dev','test'):
        cases=transfer_cases(s)
        assert all(short_check(c,m['content']) for r in cases for m,c in zip([m for m in r['messages'] if m['role']=='assistant'],r['turn_contracts']))
        atomic_json(folder/('transfer-'+s+'.json'),dict(cases=cases,scope='Small independent generated-answer panel; actual model history. No benchmark accuracy claim.'))
    plan['catalog_sha256']=info['sha256'];atomic_json(config,plan)
    print(json.dumps(info['counts'],indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);build(p.parse_args().config)
