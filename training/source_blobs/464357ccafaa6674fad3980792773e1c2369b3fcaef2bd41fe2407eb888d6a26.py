"""Frozen natural-conversation catalog and controlled ten-run chat experiments.

Whole exchanges only. No ARC targets, response clipping, or new teacher output.
Source labels/review ratings are filters, not exhaustive factual verification.
"""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid
import numpy as np
from sml_v1.common import atomic_json,read_json,file_sha256
from sml_v1.tokenization import Tokenizer
from .curate import local,verify,write_split,blocked_questions
from .reliable_chat import cached,encode
from .balanced import pair,oasst_paths
from .conversation import prior_exclusions,split_group
from .quality import digest,normalized,repeated,social_prompt
from .data import serialize
from .behavior import cases
from .behavior_v8 import python_checks

EVIDENCE='reports/sft/chat-search-20261008'
SOURCES=[
 dict(component='chat',repo='HuggingFaceTB/smol-smoltalk',config='default',revision='f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc',split='train',license='Apache-2.0'),
 dict(component='chat',repo='OpenAssistant/oasst2',config='default',revision='179dd21fc55192153d94adb0e0ce8f69e222bf75',split='train',license='Apache-2.0'),
 dict(component='chat',repo='databricks/databricks-dolly-15k',config='default',revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',split='train',license='CC-BY-SA-3.0; preserve attribution/share-alike'),
]
SYSTEM='You are a helpful assistant. Answer the user directly and briefly. Ask a question when you need more information.'
POLICY=re.compile(r'\b(?:role[- ]?play|pretend|act as|impersonate|as if you|you are (?:a|an|the|dr\b|professor)|i want you to be|assume you|in character|diagnos\w*|prescrib\w*|medications?|medical advice|legal advice|investments?|stocks|current president|latest|today.s date|suicid\w*|murder|bomb|weapons?|casino|gambling|Powerball|FanDuel|porn\w*|sexual\w*|conspirac\w*)\b',re.I)
DRIFT=re.compile(r'\b(?:my name is|i live in|i was born|my (?:wife|husband|children)|you can help me|as (?:a|an) (?:human|doctor|lawyer)|mathematical journey|connections and beauty of mathematics|first sentence about|your english (?:grammar|learning))\b',re.I)

def social_rows():
    """Small intent anchors, labeled as correlated authored examples, not HF data."""
    intents=[
      ('Hello, could you help me get started?', 'Hello! Tell me what you would like to do.'),
      ('Hi! I have a question for you.', 'Hi! What would you like to ask?'),
      ('Good morning. May I ask something?', 'Good morning! Go ahead.'),
      ('Good evening. Are you available?', 'Good evening! Yes, I am here to help.'),
      ('Hey, can we talk?', 'Sure. What is on your mind?'),
      ('Are you there?', 'Yes, I am here. How can I help?'),
      ('Can I talk to you?', 'Of course. What would you like to discuss?'),
      ('I just feel like chatting.', 'Sure. Is there something you would like to talk about?'),
      ('What do you do?', 'I can answer questions, explain ideas, help with writing, and assist with simple Python code.'),
      ('Which tasks can you help with?', 'I can explain topics, organize ideas, work with text, and help write or debug simple Python.'),
      ('Do you have a real job?', 'I do not have a job outside this chat. I am here to help with your questions.'),
      ('Do you have children?', 'No. I am an AI assistant and do not have a family.'),
      ('Thanks for your time and effort.', "You're welcome!"),
      ('I understand the answer now. Thank you.', 'Glad it helped.'),
      ('That cleared it up for me.', 'I am glad it is clearer now.'),
      ('I have to go. See you later.', 'See you later! Take care.'),
      ('This is all I needed. Bye.', 'Goodbye! Glad I could help.'),
      ('Can you shorten my message?', 'Please paste the message you want shortened.'),
      ('Could you fix an error in my code?', 'Please share the code, the error message, and what you expected it to do.'),
      ('I do not understand a paragraph.', 'Please share the paragraph and tell me which part is confusing.'),
    ]
    topics=('schoolwork','a letter','a short story','a recipe','a science question','a list of tasks','a Python exercise','a book','a presentation','a project idea')
    for i,(u,a) in enumerate(intents):
        yield dict(bucket='social',component='chat',group='chat-search:social-'+str(i),messages=pair(u,a),
          provenance=dict(repo='local/chat-search-intents',revision='v1',license='CC0-1.0',split='authored',review='Reviewed intent, role, factual identity and missing-input action'))
        for j,topic in enumerate(topics):
            # Each whole history is distinct; prompts/targets within an intent
            # remain correlated. Expose ordinary text tasks after the opener.
            history=pair(u,a)+pair(f'I would like some help with {topic}.',f'What would you like help with? Please share the details of {topic}.')
            yield dict(bucket='social',component='chat',group='chat-search:social-'+str(i),messages=history,
              provenance=dict(repo='local/chat-search-intents',revision='v1',license='CC0-1.0',split='authored',review='Reviewed opener and genuinely missing details; correlated contextual variant',variant=j))

def catalog():
    folder=local(EVIDENCE);folder.mkdir(parents=True,exist_ok=True)
    target=folder/'catalog.jsonl';receipt=folder/'catalog.json'
    if target.exists():
        info=read_json(receipt);assert file_sha256(target)==info['sha256'];return info
    blocked,bench=blocked_questions(local('pretraining/data/.sft_hf_cache'))
    oldgroups,oldprompts,oldreceipts=prior_exclusions(include_all=True);blocked.update(oldprompts)
    for suite in ('instruction-v6','instruction-v7','instruction-v8','conversation-v5'):
      for split in ('dev','test'):
       for case in cases(split,suite):
        blocked.update(digest(m['content']) for m in case['messages'] if m['role']=='user' and not social_prompt(m['content']))
    fresh=read_json(local('reports/sft/SFT-020-balanced-instructions-20261008/fresh-diagnostics/protocol.json'))
    blocked.update(digest(p) for case in fresh['cases'] for p in case['prompts'])
    arcgrams=set()
    from datasets import Dataset
    for cfg in ('ARC-Easy','ARC-Challenge'):
      for path in local('pretraining/data/.sft_hf_cache/allenai___ai2_arc').glob(cfg+'/0.0.0/*/*.arrow'):
       for q in Dataset.from_file(str(path))['question']:
        words=normalized(q).split();arcgrams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    exclusions=defaultdict(set)
    for path in local('configs').glob('sft*.json'):
      for x in read_json(path).get('review_exclusions',[]):
       exclusions[x['repo']].add(x.get('row',x.get('message_id')))
    rows=[];rejected=Counter();seen=set();sources=[];first_groups={}
    limits=dict(everyday=2000,direct=16000,conversation=8000,hermes=2500,dolly=8000,oasst=4000,python=1000,social=500)
    bucketcounts=Counter()
    def accept(row):
      try:
       messages=row['messages'];bucket=row['bucket'];group=row['group']
       if group in oldgroups:raise ValueError('old_reserved_group')
       users=[m['content'] for m in messages if m['role']=='user']
       if any(digest(p) in blocked for p in users):raise ValueError('reserved_prompt')
       for p in users:
        words=normalized(p).split()
        if any(' '.join(words[i:i+8]) in arcgrams for i in range(len(words)-7)):raise ValueError('arc_overlap')
       if bucket!='python':
        full=' '.join(m['content'] for m in messages)
        if POLICY.search(full) or DRIFT.search(' '.join(m['content'] for m in messages if m['role']=='assistant')):raise ValueError('persona_specialist_or_role_drift')
        if any(len(m['content'].split())>96 or repeated(m['content']) for m in messages if m['role']=='assistant'):raise ValueError('long_or_repetitive_target')
        if any(len(m['content'].split())>300 for m in messages if m['role']=='user'):raise ValueError('long_input')
        if any(m['role']=='system' for m in messages):raise ValueError('task_system')
        for i,m in enumerate(messages):
         if m['role']=='assistant' and re.search(r'\b(?:mathemat|equation|algebra|calculus|grammar lesson)\b',m['content'],re.I) and not re.search(r'\b(?:math|mathemat|equation|algebra|calculus|grammar)\b',' '.join(x['content'] for x in messages[:i] if x['role']=='user'),re.I):raise ValueError('unsupported_tutoring_context')
       first=next((p for p in users if not social_prompt(p)),users[0])
       group=first_groups.setdefault(digest(first),group)
       key=digest(json.dumps(messages,sort_keys=True))
       if key in seen:raise ValueError('duplicate')
       if bucketcounts[bucket]>=limits[bucket]:return
       split=split_group(group,262200,8)
       rows.append(dict(row,group=group,split=split));seen.add(key);bucketcounts[bucket]+=1
      except ValueError as e:rejected[row['bucket']+':'+str(e)]+=1
    raw,rec=cached(SOURCES[0]);sources.append(dict(source=SOURCES[0],**rec))
    # Fixed source-row shuffle, shared by all attempts. Screening is done before
    # quota selection; never silently use SFT-018's known bad row identities.
    order=list(range(len(raw)));random.Random(262200).shuffle(order)
    for index in order:
      row=raw[index];source=row['source']
      if source not in ('everyday-conversations','smol-magpie-ultra-short','openhermes-50k') or index in exclusions[SOURCES[0]['repo']]:continue
      messages=[dict(m) for m in row['messages']]
      while messages and messages[-1]['role']=='user':messages.pop()
      if not messages or len(messages)>8 or any(not m['content'].strip() for m in messages):continue
      bucket='everyday' if source=='everyday-conversations' else 'hermes' if source=='openhermes-50k' else 'direct' if len(messages)==2 else 'conversation'
      accept(dict(bucket=bucket,component='chat',group='natural-first:'+digest(messages[0]['content']),messages=messages,
       provenance=dict(SOURCES[0],row=index,source_subset=source,transformation='All complete source exchanges; no shortened responses; exclude explicit task systems/personas and known audit defects')))
    raw,rec=cached(SOURCES[1]);sources.append(dict(source=SOURCES[1],**rec))
    for messages,group,identity in oasst_paths(raw):
      if identity['message_id'] in exclusions[SOURCES[1]['repo']]:continue
      accept(dict(bucket='oasst',component='chat',group=group,messages=messages,provenance=dict(SOURCES[1],**identity,transformation='Complete English nonsynthetic rank-zero reviewed path, quality >=.7; no clipping')))
    raw,rec=cached(SOURCES[2]);sources.append(dict(source=SOURCES[2],**rec))
    for i,row in enumerate(raw):
      if i in exclusions[SOURCES[2]['repo']]:continue
      context=row['context'].strip();prompt=row['instruction'].strip()
      if context:prompt+='\n\nUse this context:\n'+context
      if row['category'] not in ('open_qa','closed_qa','general_qa','information_extraction','classification','brainstorming','summarization'):continue
      accept(dict(bucket='dolly',component='chat',group='dolly:'+digest(row['instruction']),messages=pair(prompt,row['response']),
       provenance=dict(SOURCES[2],row=i,category=row['category'],transformation='Original instruction, complete context and complete response; no answer rewriting')))
    # Previously reviewed Python training contracts, not historical heldouts.
    for path in ('pretraining/data/sft_019_reviewed-instructions_20261008/train.jsonl','pretraining/data/sft_021_contrast-instructions_20261008/train.jsonl'):
      for line in local(path).read_text().splitlines():
       row=json.loads(line)
       if row['component']!='python' or not row.get('functional_assertions') or not row['messages'][-1]['content'].startswith('```python'):continue
       if any((row['python_contract'].get('function') or '').startswith(f) for f in ('count_positive','clamp','lookup','get_default','lengths','offset_values','first_positive')):continue
       messages=[m for m in row['messages'] if m['role']!='system']
       if not all(python_checks(messages[-1]['content'],row['functional_assertions'],**row['python_contract']).values()):continue
       accept(dict(row,bucket='python',messages=messages,provenance=dict(row['provenance'],reuse_training_file=path)))
    for row in social_rows():accept(row)
    with target.open('w') as f:
      for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    info=dict(sha256=file_sha256(target),counts={s:dict(Counter(r['bucket'] for r in rows if r['split']==s)) for s in ('train','dev','test')},
      rejected=dict(rejected),source_receipts=sources,historical_exclusions=oldreceipts,benchmark_exclusions=bench,
      scope='Pinned natural instructions and complete histories plus a small reviewed intent/Python anchor set. Automated screening and spot review do not establish all source facts correct.')
    atomic_json(receipt,info);print(json.dumps(info['counts'],indent=2),flush=True);return info

def supervision(ids,mask,tokenizer,messages,plan):
    result=mask.copy()
    if plan.get('last_turn_only'):
        prefix,_=serialize(tokenizer,messages[:-1],generation=True)
        result[:len(prefix)]=False
    if plan.get('no_document_end_target'):result[-1]=False
    if not result.any():raise ValueError('No supervised target')
    return result

def prepare(config):
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
      result=verify(dest);assert result['plan_sha256']==file_sha256(config);return result
    path=local(plan.get('catalog_path',EVIDENCE+'/catalog.jsonl'));receipt=read_json(local(plan.get('catalog_receipt',EVIDENCE+'/catalog.json')))
    assert file_sha256(path)==plan['catalog_sha256']==receipt['sha256']
    tokenizer=Tokenizer(local(plan['tokenizer']));pools=defaultdict(lambda:defaultdict(list));rejected=Counter()
    for line in path.read_text().split('\n'):
      if not line:continue
      row=json.loads(line)
      if row['bucket'] not in plan['bucket_caps']:continue
      try:
       messages=row['messages']
       if any(any(c in m['content'] for c in ('\u0085','\u2028','\u2029')) for m in messages):raise ValueError('nonstandard_line_separator')
       if plan.get('single_turn_only') and len(messages)!=2:continue
       if any(len(m['content'].split())>plan['short_reply_words'] for m in messages if m['role']=='assistant' and row['component']!='python'):continue
       messages,ids,mask=encode(messages,tokenizer,plan,row['component'],row['group'])
       mask=supervision(ids,mask,tokenizer,messages,plan)
       pools[row['split']][row['bucket']].append(dict(row,messages=messages,_ids=ids,_mask=mask))
      except ValueError as e:rejected[row['bucket']+':'+str(e)]+=1
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    counts={};reviews=[]
    try:
      for split in ('train','dev','test'):
       rows=[]
       for bucket,cap in plan['bucket_caps'].items():
        candidates=pools[split][bucket];candidates.sort(key=lambda r:digest('262200'+json.dumps(r['messages'],sort_keys=True)))
        quota=cap if split=='train' else plan['heldout_per_bucket'] if plan.get('evidence_instructions') or plan.get('matched_comparison') else 30
        rows.extend(candidates[:quota])
       if len(rows)<32:raise ValueError('Insufficient '+split+' rows')
       random.Random(plan['seed']).shuffle(rows)
       counts[split]=dict(conversations=len(rows),semantic_groups=len({r['group'] for r in rows}),
        components=dict(Counter(r['component'] for r in rows)),buckets=dict(Counter(r['bucket'] for r in rows)),
        input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),
        multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows))
       reviews.extend({k:v for k,v in r.items() if not k.startswith('_')} for r in rows[:30]);write_split(staging,split,rows)
      atomic_json(staging/'review_examples.json',reviews)
      if plan.get('natural_transfer'):
       for name in ('transfer-dev.json','transfer-test.json'):
        shutil.copyfile(local(plan['catalog_receipt']).parent/name,staging/name)
      if plan.get('continuation_trials'):
       for name in ('trial-dev.json','trial-test.json'):
        shutil.copyfile(local(plan['catalog_receipt']).parent/name,staging/name)
      if plan.get('manual_review_panel'):
       panel=local(plan['manual_review_panel'])
       assert file_sha256(panel)==plan['manual_review_panel_sha256']
       shutil.copyfile(panel,staging/'manual-review.json')
      atomic_json(staging/'curation.json',dict(catalog_receipt=receipt,rejected=dict(rejected),counts=counts,mask_policy=dict(last_turn_only=plan.get('last_turn_only',False),no_document_end_target=plan.get('no_document_end_target',False)),limitations=['Source quality labels and structural screening are not a universal fact oracle.','Social/Python anchors are correlated finite templates; actual numbers are reported.']))
      files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
      manifest=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),sources=plan['sources'],counts=counts,files=files,
       seed=plan['seed'],tokenizer=tokenizer.fingerprint,epochs=plan['epochs'],serialization='explicit-ids-assistant-mask-v1',
       final_quality_version='chat-search-v1',total_steps=math.ceil(counts['train']['conversations']/64)*plan['epochs'])
      atomic_json(staging/'manifest.json',manifest);staging.rename(dest)
    finally:
      if staging.exists():shutil.rmtree(staging)
    print(json.dumps(counts,indent=2),flush=True);return manifest
