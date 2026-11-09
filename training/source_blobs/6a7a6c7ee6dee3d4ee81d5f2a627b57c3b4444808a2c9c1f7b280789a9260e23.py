"""SFT-016: proportional chat/direct QA/CoQA/Python, frozen provenance.

Quotas count conversations, not tokens. No repeated rows fill a scarce quota.
This builder changes only the new corpus; historical run inputs stay immutable.
"""
from collections import Counter, defaultdict
import ast
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from sml_v1.common import atomic_json, file_sha256, read_json
from sml_v1.tokenization import Tokenizer
from .curate import local, verify, write_split, blocked_questions
from .data import serialize
from .conversation import prior_exclusions, split_group
from .quality import digest, normalized, repeated, social_prompt, direct_general
from .short_chat import check_focus, focused_curriculum
from .balanced import pair
from .bounded_python import verify_target
from .pure_python import assertions


def cached(source):
    folder=local('pretraining/data/.sft_hf_cache')/source['repo'].replace('/','___')/source['config']/'0.0.0'/source['revision']
    paths=sorted(folder.glob('*-'+source['split']+'.arrow')) or sorted(folder.glob('*-'+source['split']+'-*.arrow'))
    if not paths:raise ValueError('Missing pinned source cache: '+str(folder))
    datasets=[Dataset.from_file(str(p)) for p in paths]
    return (datasets[0] if len(datasets)==1 else concatenate_datasets(datasets)),dict(files={str(p):file_sha256(p) for p in paths})


def encode(messages,tokenizer,plan,component,group):
    """Preserve supplied systems and whole histories; never truncate targets."""
    messages=[dict(m) for m in messages]
    fraction=plan.get('system_fraction',.5)
    if not 0<=fraction<=1:raise ValueError('Invalid system fraction')
    use_system=int(digest(group)[:8],16)/2**32<fraction if 'system_fraction' in plan else bool(int(digest(group)[:8],16)%2)
    if messages[0]['role']!='system' and use_system:
        messages.insert(0,dict(role='system',content=plan['conversation_system']))
    for m in messages:
        text=m['content']
        if not text.strip() or '\x00' in text or '�' in text or any(x in text for x in ('<|pad|>','<|doc_end|>','<|turn_start|>','<|turn_end|>')):raise ValueError('invalid_text')
        if m['role']=='assistant':
            if len(tokenizer.encode(text))>plan['python_max_tokens' if component=='python' else 'assistant_max_tokens']:raise ValueError('reply_length')
            if repeated(text):raise ValueError('repeated_target')
    ids,mask=serialize(tokenizer,messages)
    if len(ids)>plan['max_seq_len']:raise ValueError('context_length')
    return messages,ids,mask


def python_row(raw,index,source):
    tests=assertions(raw['instruction'])
    if len({json.dumps(t,sort_keys=True) for t in tests})<2:raise ValueError('nonindependent_tests')
    tests=[dict(t,args_literal=repr(t['args']),expected_literal=repr(t['expected'])) for t in tests]
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',raw['response'],re.S)
    if len(blocks)!=1:raise ValueError('one_code_block_required')
    response='```python\n'+blocks[0].strip()+'\n```'
    prompt=re.split(r'Your code should pass the following assertions:',raw['instruction'],maxsplit=1)[0].strip()
    # Remove assertion-only blocks; never supply dev/test assertions to the model.
    prompt=re.sub(r'```(?:python|py)?\s*\n(?:(?!```).)*?assert(?:(?!```).)*?```','',prompt,flags=re.S).strip()
    prompt=re.sub(r'(?:Your code should pass|The function should satisfy) the following (?:test cases?|assertions):\s*$','',prompt,flags=re.I).strip()
    prompt=re.sub(r'(?:Here are some examples|Examples|Example):\s*$','',prompt,flags=re.I).strip()
    if re.search(r'```[^`]*\breturn\b',prompt):raise ValueError('solution_in_prompt')
    if len(prompt.split())>180:raise ValueError('complex_python_prompt')
    if re.search(r'\b(?:SQL|secure|security|cryptograph|decorator|recursive|recursion|class|regex|regular expression|do nothing|musical|QR code|alignment patterns|nearest|rounds?|modulo)\b',prompt,re.I):raise ValueError('specialist_or_ambiguous_contract')
    tree=ast.parse(blocks[0]);nodes=list(ast.walk(tree))
    if re.search(r'f[- ]strings?',prompt,re.I) and not any(isinstance(n,ast.JoinedStr) for n in nodes):raise ValueError('fstring_spec_conflict')
    if re.search(r'string slicing',prompt,re.I) and not any(isinstance(n,ast.Slice) for n in nodes):raise ValueError('slicing_spec_conflict')
    if re.search(r'list comprehension',prompt,re.I) and not any(isinstance(n,ast.ListComp) for n in nodes):raise ValueError('comprehension_spec_conflict')
    if re.search(r'without.*(?:built-in|builtin)',prompt,re.I) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) for n in nodes):raise ValueError('builtin_spec_conflict')
    if re.search(r'without.*(?:loop|conditional)',prompt,re.I) and any(isinstance(n,(ast.For,ast.If,ast.IfExp)) for n in nodes):raise ValueError('control_spec_conflict')
    if verify_target('',response,tests=tests,strict=True) is None:raise ValueError('functional_contract_failed_or_unsupported')
    fn=tree.body[0]
    if len(fn.args.args)==1 and re.search(r'(?:input list|list).*empty.*return an empty list',prompt,re.I|re.S):
        edge=dict(function=fn.name,args=[[]],expected=[])
        if verify_target('',response,tests=[*tests,edge],strict=True) is None:raise ValueError('empty_list_spec_conflict')
        tests=[*tests,edge]
    # Equivalent functions with renamed identifiers share a split.
    class Canonical(ast.NodeTransformer):
        def __init__(self):self.names={}
        def name(self,s):return self.names.setdefault(s,'v'+str(len(self.names)))
        def visit_FunctionDef(self,n):
            n.name='function';n.returns=None
            n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
            return self.generic_visit(n)
        def visit_arg(self,n):n.arg=self.name(n.arg);n.annotation=None;return n
        def visit_Name(self,n):
            if n.id not in ('len','sum','min','max','abs','sorted','tuple','list','set','str','int','bool','round','all','any','reversed','enumerate','zip','range'):n.id=self.name(n.id)
            return n
    canonical=Canonical().visit(ast.parse(blocks[0]))
    return dict(component='python',group='python-ast:'+digest(ast.dump(canonical,include_attributes=False)),
        messages=pair(prompt+'\nReturn only a Python code block.',response),functional_assertions=tests,
        provenance=dict(source,row=index,transformation='Original function block; removed source explanation and exposed assertion block; source tests retained offline'))


def coqa_row(raw,index,source):
    if len(raw['questions'])!=len(raw['answers']['input_text']):raise ValueError('field_mismatch')
    messages=[];answers=raw['answers']['input_text']
    for j,(q,a,start,end) in enumerate(zip(raw['questions'],answers,raw['answers']['answer_start'],raw['answers']['answer_end'])):
        if not a.strip() or len(a.split())>40 or re.search(r'^[,.]',a.strip()):raise ValueError('questionable_coqa_target')
        if normalized(a)!='unknown' and not 0<=start<=end<=len(raw['story']):raise ValueError('invalid_evidence')
        prompt=('Answer using only the passage. If the answer is not supported, reply Unknown.\n\nPassage:\n'+raw['story']+'\n\nQuestion: '+q) if j==0 else q
        messages.extend(pair(prompt,'Unknown' if normalized(a)=='unknown' else a))
    return dict(component='coqa',group='coqa:'+digest(raw['story']),messages=messages,
                answers=[answers[-1]],answerable=normalized(answers[-1])!='unknown',
                raw_questions=raw['questions'],story=raw['story'],
                provenance=dict(source,row=index,domain=raw['source'],source_url=raw.get('filename',raw.get('id','')),transformation='Full passage once and every original Q/A; unknown capitalization standardized; gold evidence not exposed'))


def correction_curriculum():
    """Finite authored updates; variants of one record share a split group."""
    rows=[]
    provenance=dict(repo='local/reliable-chat-v1',revision='reliable-chat-v1',split='authored',license='CC0-1.0',method='Deterministic record corrections and authored conversational replies; correlated variants share groups')
    objects=('shipment','notebook','appointment','storage box','booking','work folder')
    for i in range(120):
        obj=objects[i%len(objects)];old=f'BC-{100+i}';new=f'DF-{600+i}'
        for wording in (0,1):
            first=f'Please note that the code for my {obj} is {old}.'
            correction=(f'That code has changed. Use {new} instead of {old}.' if wording else f'I gave the wrong code. The correct code is {new}.')
            question=(f'Which code should I use for the {obj} now?' if wording else f'Tell me the updated code for my {obj}.')
            messages=pair(first,f'The code for your {obj} is {old}.')+pair(correction,f'Updated: the code for your {obj} is {new}.')+pair(question,f'The current code is {new}.')
            rows.append(dict(component='chat',foundation_kind='correction',group=f'local-reliable:record-{i}',messages=messages,provenance=provenance))
    tasks=[('an unfamiliar word','I can explain a word and give a short example. Which word would you like to discuss?'),
           ('a small Python program','I can help write, explain, or debug simple Python code. What should the program do?'),
           ('a confusing paragraph','Share the paragraph, and I can help explain its meaning.'),
           ('a short message','Tell me what you want to communicate, and I can help draft a short message.'),
           ('a list of ideas','Describe the topic, and we can brainstorm a few ideas.'),
           ('planning a task','Describe the task, and I can help break it into smaller steps.')]
    starts=[('Can I ask you something?','Of course. What would you like to ask?'),
            ('Are you free to talk?','Yes, I am here and ready to chat.'),
            ('I feel like talking for a bit.','Sure. What would you like to talk about?'),
            ('What are you doing at the moment?','I am here to respond to your messages. What is on your mind?')]
    for i,(task,answer) in enumerate(tasks):
        for opening,reply in starts:
            rows.append(dict(component='chat',foundation_kind='social-capability',group=f'local-reliable:capability-{i}',
                messages=pair(opening,reply)+pair(f'Could you help me with {task}?',answer),provenance=provenance))
    return rows


def prepare(config):
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
        result=verify(dest)
        if result['plan_sha256']!=file_sha256(config):raise ValueError('Prepared plan changed')
        return result
    tokenizer=Tokenizer(local(plan['tokenizer']));rng=random.Random(plan['seed'])
    blocked,bench=blocked_questions(local('pretraining/data/.sft_hf_cache'))
    oldgroups,oldprompts,oldreceipts=prior_exclusions(include_all=True);blocked.update(oldprompts)
    from .behavior import cases
    from .behavior_v5 import rollouts
    diagnostic=[]
    for suite in ('chat-relevance-v2','conversation-v3','conversation-v4','conversation-v5'):
        diagnostic.extend(c for s in ('dev','test') for c in cases(s,suite))
    for c in diagnostic:
        blocked.update(digest(m['content']) for m in c['messages'] if m['role']=='user' and not social_prompt(m['content']))
    blocked.update(digest(q) for s in ('dev','test') for c in rollouts(s) for q in c['prompts'])
    # Exact and eight-word ARC screens over all six splits, exclusions only.
    arcgrams=set()
    for cfg in ('ARC-Easy','ARC-Challenge'):
        folder=local('pretraining/data/.sft_hf_cache/allenai___ai2_arc')/cfg/'0.0.0'/'210d026faf9955653af8916fad021475a3f00453'
        for p in folder.glob('*.arrow'):
            for q in Dataset.from_file(str(p))['question']:
                w=normalized(q).split();arcgrams.update(' '.join(w[i:i+8]) for i in range(len(w)-7))
    pools=defaultdict(lambda:defaultdict(list));rejected=Counter();source_receipts=[];seen=set();first_groups={}
    def accept(row,split=None):
        c=row['component']
        try:
            group=row['group'];messages=row['messages']
            prov=row['provenance']
            if any(prov.get('repo')==x['repo'] and (prov.get('row')==x.get('row') if 'row' in x else prov.get('message_id')==x.get('message_id')) for x in plan['review_exclusions']):raise ValueError('manual_review_rejection')
            if group in oldgroups:raise ValueError('previous_holdout_group')
            qs=row.get('raw_questions',[m['content'] for m in messages if m['role']=='user'])
            if any(digest(q) in blocked for q in qs if not social_prompt(q)):raise ValueError('benchmark_or_diagnostic_prompt')
            text=' '.join(m['content'] for m in messages)
            w=normalized(text).split()
            if any(' '.join(w[i:i+8]) in arcgrams for i in range(len(w)-7)):raise ValueError('arc_eight_word_overlap')
            key=digest(json.dumps(messages,sort_keys=True))
            if key in seen:raise ValueError('duplicate_conversation')
            # Entire conversations sharing the initial substantive question stay grouped.
            if c in ('chat','general'):
                check_focus(messages)
                prompts=' '.join(m['content'] for m in messages if m['role'] in ('user','system'))
                if re.search(r'\b(?:allerg\w*|epipen|emergency|disease|cancer|medical|treatments?|medications?|health benefits|statistical|ARPU|Radeon|GeForce|ruby|rails|typescript|javascript|recursion|hardware|processor)\b',text,re.I):raise ValueError('specialist_topic')
                if re.search(r'\b(?:first (?:two )?sentences?|different words|suggested revision|opening sentences|i made.*changes|my home town)\b',text,re.I):raise ValueError('editing_or_persona_drift')
                if re.search(r'\b(?:this weekend|best places|best.*restaurants?|hair dryer|lotter\w*|Powerball|betting|FanDuel|DraftKings|Germans love|Amazon Echo|Google Home|Philips Hue|Millennium tower)\b',text,re.I) or re.search(r'\b(?:latest|today.s (?:date|news)|currently (?:president|available)|events today)\b',prompts,re.I):raise ValueError('stale_or_questionable_advice')
                if re.search(r'\b(?:pretend|act as|diagnos\w*|prescrib\w*|stocks|investment|latest|current president|my hometown|near me)\b',prompts,re.I):raise ValueError('persona_specialist_or_stale')
                answers=' '.join(m['content'] for m in messages if m['role']=='assistant')
                if re.search(r'\b(?:my name is|i live in|i was born|my (?:wife|husband|children)|you can help me|i am (?:alex|sarah|a teacher))\b',answers,re.I):raise ValueError('role_or_identity_conflict')
                if len([m for m in messages if m['role']=='user'])>8:raise ValueError('chat_turn_limit')
                if any(len(m['content'].split())>plan['short_reply_words'] for m in messages if m['role']=='assistant'):raise ValueError('not_short_reply')
                first=next((m['content'] for m in messages if m['role']=='user' and not social_prompt(m['content'])),messages[0]['content'])
                group=first_groups.setdefault(digest(first),group)
            split=split or split_group(group,plan['seed'],plan['heldout_group_percent'])
            messages,ids,mask=encode(messages,tokenizer,plan,c,group)
            row=dict(row,group=group,messages=messages,split=split,_ids=ids,_mask=mask)
            pools[c][split].append(row);seen.add(key)
        except (ValueError,SyntaxError) as e:rejected[c+':'+str(e)]+=1
    # Reuse eligible source identities, reread original HF dialogues to preserve
    # supplied systems and complete history, not SFT-015 truncated presentations.
    prior=local('pretraining/data/sft_015_short-chat-python_20261008/train.jsonl')
    source_receipts.append(dict(previous_training_candidates=str(prior),sha256=file_sha256(prior)))
    raw_by={}
    for source in plan['sources']:
        if source['component'] in ('chat','general'):
            raw,receipt=cached(source);raw_by[source['repo']]=(raw,source);source_receipts.append(receipt)
    for line in prior.read_text().splitlines():
        row=json.loads(line)
        if row['component'] not in ('chat','general','constraints'):continue
        if row['provenance']['repo'].startswith('local/'):continue
        if row['bucket'] not in ('everyday','magpie_chat','magpie_direct','dolly','oasst'):continue
        prov=row['provenance'];component='general' if row['bucket'] in ('dolly','magpie_direct') else 'chat'
        if prov['repo']=='OpenAssistant/oasst2':
            accept(dict(row,component=component));continue
        raw,source=raw_by[prov['repo']];r=raw[prov['row']]
        if prov['repo']=='databricks/databricks-dolly-15k':messages=pair(r['instruction'],r['response'])
        else:
            messages=[dict(m) for m in r['messages']]
            while messages and messages[-1]['role']=='user':messages.pop()
            if component=='general':
                offset=1 if messages[0]['role']=='system' else 0
                messages=messages[:offset+2]
        if component=='general' and not direct_general(messages):continue
        accept(dict(row,component=component,messages=messages))
    for row in [*focused_curriculum(),*correction_curriculum()]:
        accept(dict(row,component='chat'))
    source=next(s for s in plan['sources'] if s['component']=='python');raw,receipt=cached(source);source_receipts.append(receipt)
    exclusions={x['row'] for x in plan['review_exclusions'] if x['repo']==source['repo']}
    for i,r in enumerate(raw):
        if i in exclusions:rejected['python:manual_review']+=1;continue
        try:accept(python_row(r,i,source))
        except (ValueError,SyntaxError,TypeError) as e:rejected['python:'+str(e)]+=1
    source=next(s for s in plan['sources'] if s['component']=='coqa')
    audit=local(plan['coqa_audit']);receipt=read_json(audit/'source-receipt.json');source_receipts.append(receipt)
    if receipt['repo']!=source['repo'] or receipt['revision']!=source['revision']:raise ValueError('CoQA revision mismatch')
    for name,info in receipt['files'].items():
        if file_sha256(audit/'source'/name)!=info['sha256']:raise ValueError('CoQA source changed')
    coqas={s:pq.read_table(audit/'source/data'/f'{s}-00000-of-00001.parquet').to_pylist() for s in ('train','validation')}
    valid_hashes={digest(r['story']) for r in coqas['validation']}
    for s,data in coqas.items():
        for i,r in enumerate(data):
            if r['source'] not in ('wikipedia','gutenberg'):continue
            if s=='train' and (digest(r['story']) in valid_hashes or i in plan['coqa_excluded_train_rows']):rejected['coqa:validation_story_or_manual']+=1;continue
            try:accept(coqa_row(r,i,dict(source,split=s)),split='test' if s=='validation' else None)
            except ValueError as e:rejected['coqa:'+str(e)]+=1
    proportions=plan['mixture'];units=min(len(pools[c]['train'])//n for c,n in proportions.items())
    units=min(units,plan['max_conversations']//sum(proportions.values()))
    if units<20:raise ValueError('Insufficient clean data for proposed shares: '+str({c:{s:len(rs) for s,rs in p.items()} for c,p in pools.items()}))
    selected=defaultdict(list);counts={}
    for c,share in proportions.items():
        for s in ('train','dev','test'):
            candidates=pools[c][s];rng.shuffle(candidates)
            n=units*share if s=='train' else min(plan['heldout_per_component'],len(candidates))
            if n<8:raise ValueError('Insufficient '+c+' '+s)
            selected[s].extend(candidates[:n])
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        reviews=[]
        for s,rows in selected.items():
            rng.shuffle(rows)
            counts[s]=dict(conversations=len(rows),semantic_groups=len({r['group'] for r in rows}),components=dict(Counter(r['component'] for r in rows)),
                input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),
                assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in proportions},
                multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows))
            for c in proportions:
                reviews.extend({k:v for k,v in r.items() if not k.startswith('_')} for r in [r for r in rows if r['component']==c][:12])
            write_split(staging,s,rows)
        atomic_json(staging/'review_examples.json',reviews)
        atomic_json(staging/'curation.json',dict(counts=counts,pool_counts={c:{s:len(rs) for s,rs in p.items()} for c,p in pools.items()},rejected=dict(rejected),benchmark_exclusions=bench,
            previous_holdout_exclusions=oldreceipts,source_receipts=source_receipts,mixture_units=units,proportions=proportions,
            limitations=['HF factual/dialogue targets structurally filtered and spot-reviewed, not exhaustively verified.','Code tests check supplied inputs, not every natural-language requirement.','Exact/eight-word screens are not complete semantic decontamination.','CoQA annotation/source errors may remain; passage grounding is not world truth.','Prior source training candidates reused with new grouping and complete original histories; previous holdouts excluded.']))
        atomic_json(staging/'diagnostic_protocol.json',dict(suite='conversation-v5',selection='dev only',dev=cases('dev','conversation-v5'),test=cases('test','conversation-v5'),dev_rollouts=rollouts('dev'),test_rollouts=rollouts('test')))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        result=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=counts,files=files,seed=plan['seed'],serialization='explicit-ids-assistant-mask-v1',epochs=1,final_quality_version='reliable-chat-v1',total_steps=math.ceil(counts['train']['conversations']/64))
        atomic_json(staging/'manifest.json',result);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(counts,indent=2),flush=True);return result
