"""Balanced human/HF instruction corpus plus explicit local foundations.

No model-generated targets, benchmark answers, or old SFT weights are used.
Local copy/extraction tasks are mechanically checked; human facts still need review.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid
from .curate import local, verify, write_split, source_rows, filter_messages, re_identity
from .quality import Deduplicator, digest, normalized, direct_general
from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v1.tokenization import Tokenizer


def pair(prompt, answer):
    return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]


def foundations():
    """Finite, auditable curriculum. Split by semantic value, not paraphrase."""
    rows=[]
    def add(kind, group, messages, rule=None):
        extra={'foundation_kind':kind}
        if rule: extra['rules']=[rule]
        rows.append(dict(component='constraints' if rule else 'chat', group='local:'+group,
                         messages=messages,provenance={'repo':'local/conversation-foundations','revision':'conversation-foundations-v1',
                         'split':'authored','license':'CC0-1.0','method':'reviewed social targets or deterministic copy/context target'},**extra))
    greetings=['Hi there','Greetings','Hello there','Hey','Good afternoon','Hi','Hello','Good morning','Hey there','Good evening']
    for i,u in enumerate(greetings):
        add('greeting',f'greeting-{i}',pair(u,['Hello!','Hi! How can I help?','Hello! What would you like to discuss?'][i%3]))
    thanks=['Thanks for explaining','I appreciate your help','Thank you so much','Thanks, I understand now','That was useful, thank you','Many thanks','Thanks a lot','Much appreciated','Thank you for the explanation','Thanks for your time',"You've been helpful, thanks",'I appreciate that','Thank you kindly','That answers my question, thanks','Great, thank you','Thanks for the answer','I understand now, thanks','Thank you for assisting me','Thanks, that makes sense','Thanks for clarifying']
    suffixes=['','!',' — that cleared things up.','; I appreciate it.']
    for i,u in enumerate(thanks):
        for j,s in enumerate(suffixes):
            add('thanks',f'thanks-{i}',pair(u+s,["You're welcome!","Glad I could help.","You're welcome. Happy to help!","Anytime!"][(i+j)%4]))
    farewells=['See you tomorrow','Bye for today','I have to leave now','Talk to you later','Good night','See you soon','I am heading out','Until next time','Have a nice day','That is all for today','I need to go','Take care','Catch you later','We can talk another time','I will sign off now','I am done for now','Have a good evening','Bye now','See you next time','Time for me to leave']
    for i,u in enumerate(farewells):
        for s in ['', '!', ', thanks for chatting.']:
            add('farewell',f'farewell-{i}',pair(u+s,['Goodbye! Take care.','See you later!','Have a good day!','Bye!'][i%4]))
    objects=['document','message','paragraph','email','draft','plan','diagram','schedule','design','list','table','description','outline','summary','report','invitation','announcement','note','explanation','presentation']
    for i,obj in enumerate(objects):
        for verb in ['review','check','improve','look over']:
            add('clarify',f'clarify-{obj}',pair(f'Could you {verb} my {obj}?',f'Could you share the {obj} and tell me what you would like help with?'))
        add('clarify',f'clarify-{obj}',pair(f'I need some help with a {obj}.',f'What would you like help with? Please share the {obj}.'))
    # No ambiguous referent is treated as a physical or medical repair request.
    for i,u in enumerate(['Help me with this','This is not working','I am stuck','Something went wrong','Can you help me figure this out','I need your advice','I do not understand this','Can you explain that','Could you take a look','What should I do next','Help please','I need assistance']):
        add('clarify',f'ambiguous-{i}',pair(u,['Could you share more details about what you need help with?','What are you working on, and what happened?'][i%2]))
    words=['amber','birch','cobalt','daisy','ember','fern','granite','hazel','indigo','jade','kiwi','lavender','maple','nectar','olive','pearl','quartz','ruby','sage','teal','umber','violet','willow','zinc','apricot','basil','cherry','dill','elm','fig','grape','honey','iris','jasmine','kelp','lemon','mint','nutmeg','oak','plum','rose','silk','thyme','walnut','yarrow','azure','bronze','coral','denim','ivory']
    for i,w in enumerate(words):
        for prompt,ans in [(f'Output exactly this word and nothing else: {w}',w),(f'Write only the uppercase form of {w}.',w.upper()),(f'Return only the lowercase form of {w.upper()}.',w)]:
            add('literal',f'word-{w}',pair(prompt,ans),dict(kind='exact',args=[ans]))
        record=f'The selected label is {w}. The spare label is {words[(i+7)%len(words)]}.'
        add('extract',f'word-{w}',pair(f'{record}\nWhich label is selected? Output only its name.',w),dict(kind='exact',args=[w]))
        history=pair(f'For this conversation, my project is called {w.title()}.',f'Understood. Your project is called {w.title()}.')
        add('context',f'word-{w}',[*history,*pair('What did I call my project?',f'Your project is called {w.title()}.')])
        # Asking about an absent field must not produce invented context.
        add('missing-context',f'word-{w}',pair(f'The project is named {w.title()}. What is its deadline?', 'The deadline was not provided.'))
    return rows


def acceptable_node(row):
    if row['lang']!='en' or row['deleted'] or row['synthetic'] or not row['review_result']: return False
    if row['role']=='assistant':
        if row['rank']!=0 or not row.get('labels'): return False
        labels=dict(zip(row['labels']['name'],row['labels']['value']))
        if labels.get('quality',0)<.7: return False
        if any(labels.get(k,0)>.1 for k in ('spam','fails_task','lang_mismatch','pii','hate_speech','sexual_content')):return False
    return True


def oasst_paths(rows, node_filter=lambda row: True):
    """Use complete alternating paths with ranked, reviewed human answers."""
    by={r['message_id']:r for r in rows}; best={}
    for row in rows:
        if row['role']!='assistant' or not acceptable_node(row) or not node_filter(row):continue
        chain=[]; node=row; seen=set()
        while node:
            if node['message_id'] in seen or not acceptable_node(node) or not node_filter(node):chain=[];break
            seen.add(node['message_id']);chain.append(node)
            if node['parent_id'] is None:break
            node=by.get(node['parent_id'])
            if node is None:chain=[];break
        if not chain or chain[-1]['parent_id'] is not None:continue
        chain.reverse()
        if len(chain)>6 or any(r['role']!=('prompter' if i%2==0 else 'assistant') for i,r in enumerate(chain)):continue
        messages=[dict(role='user' if r['role']=='prompter' else 'assistant',content=r['text']) for r in chain]
        key=row['message_tree_id']
        old=best.get(key)
        if old is None or (len(chain),row['message_id'])>(len(old[0]),old[1]):best[key]=(messages,row['message_id'])
    return [(m,'oasst:'+g,{'message_id':mid,'tree':g}) for g,(m,mid) in sorted(best.items())]


def prepare(config):
    plan=read_json(config); dest=local(plan['prepared'])
    if dest.exists():
        m=verify(dest)
        if m['plan_sha256']!=file_sha256(config):raise ValueError('Use a fresh prepared destination')
        return m
    tokenizer=Tokenizer(local(plan['tokenizer'])); origin=local('pretraining/data/sft_176m_chat_ready_20261007'); old=verify(origin)
    # Build exact benchmark exclusions from cached, pinned original sources.
    from .curate import blocked_questions
    blocked,bench=blocked_questions(dest.parent/'.sft_hf_cache')
    from .behavior import cases
    blocked.update(digest(m['content']) for split in ('dev','test') for c in cases(split,plan['behavior_suite']) for m in c['messages'] if m['role']=='user' and normalized(m['content']) not in {'hello','hi','hey','hi there','hello there','hey there','good morning','good evening'})
    dedup=Deduplicator(blocked); selected=defaultdict(list); counts=Counter(); rejects=Counter()
    policy=re.compile(r'\b(?:roleplay|pretend|act as|impersonate|diagnos\w*|prescri\w*|medication|suicide|kill yourself|investment|stocks|legal advice|tax advice|latest|current|currently|right now|today.s date)\b',re.I)
    def accept(row,bucket,forced_split=None):
        try:
            msgs=row['messages']
            if policy.search(' '.join(m['content'] for m in msgs)):raise ValueError('specialist_persona_or_time_sensitive')
            if any(len(m['content'].split())>120 for m in msgs if m['role']=='user') and bucket not in ('qa','dolly_grounded'):raise ValueError('complex_prompt')
            if any(re.search(r'\b(?:i live|my (?:wife|husband|children)|i was born|my name is)\b',m['content'],re.I) for m in msgs if m['role']=='assistant'):raise ValueError('invented_personal_history')
            if bucket in ('dolly_general','oasst') and any(re.search(r'```|\b(?:rewrite|rephrase|proofread|essay|poem|story|limerick|translate|role|character|persona|constitution\w*|legal|tax|crypto\w*|stocks|shares|elon musk)\b',m['content'],re.I) for m in msgs if m['role']=='user'):raise ValueError('wrong_general_task')
            if bucket=='dolly_general':
                prompt=msgs[0]['content'];answer=msgs[-1]['content']
                if not direct_general(msgs):raise ValueError('not_direct_general')
                if row['provenance']['category']=='open_qa' and not re.match(r'^(?:what (?:is|are|do|does)|why|how)\b',prompt,re.I):raise ValueError('long_tail_trivia')
                if re.search(r'\b(?:aliens|harry potter|weasley\w*|game of thrones|house \w+|song of ice and fire|anime|pokemon)\b',prompt,re.I):raise ValueError('niche_fiction_or_speculation')
                if re.match(r'^(?:why|how)\b',prompt,re.I) and len(answer.split())<6:raise ValueError('underspecified_explanation')
            if bucket=='oasst' and any(re.search(r'\bas (?:an? )?(?:artificial intelligence|ai assistant)\b',m['content'],re.I) for m in msgs if m['role']=='assistant'):raise ValueError('identity_boilerplate')
            if row['component']=='chat' and any(re.search(r'^(?:hello|hi)[!.]?(?:\s+how can i help.*)?$',m['content'].strip(),re.I) for m in msgs[3::2]):raise ValueError('greeting_used_as_followup')
            group=row['group'];split=forced_split
            if split is None:
                n=int(fingerprint({'seed':plan['seed'],'group':group})[:8],16)%100
                split='test' if n<4 else 'dev' if n<8 else 'train'
            if split=='train' and counts[bucket]>=plan['balanced_caps'].get(bucket,100000):raise ValueError('cap')
            if not any(m['role']=='system' for m in msgs) and int(digest(group)[:8],16)%2:
                msgs=[dict(role='system',content=plan['conversation_system']),*msgs]
            ids,mask=filter_messages(msgs,tokenizer,row['component'],plan)
            if bucket in ('dolly_general','oasst') and any(len(tokenizer.encode(m['content']))>96 for m in msgs if m['role']=='assistant'):raise ValueError('long_human_answer')
            # The same context-recall question legitimately has different
            # answers in different histories. Screen each raw prompt against
            # benchmarks, then deduplicate the complete context, not the final
            # question alone. Semantic-value groups stay in one split.
            dedup_msgs=msgs
            if row.get('foundation_kind')=='context':
                if any(digest(m['content']) in blocked for m in msgs if m['role']=='user'):raise ValueError('eval_overlap')
                dedup_msgs=pair('\n'.join(m['content'] for m in msgs if m['role']!='system'),msgs[-1]['content'])
            if not dedup.accept(dedup_msgs,row['component']):raise ValueError('duplicate_or_eval_overlap')
            row=dict(row,messages=msgs,split=split,bucket=bucket,_ids=ids,_mask=mask)
            selected[split].append(row)
            if split=='train':counts[bucket]+=1
        except ValueError as e:rejects[bucket+':'+str(e)]+=1
    for row in foundations():accept(row,'foundations')
    for source in plan['sources'][:2]:
        raw=source_rows(source,dest.parent/'.sft_hf_cache')
        provenance={k:source[k] for k in ('repo','config','revision','split','license')}
        if source['repo'].startswith('databricks/'):
            indices=list(range(len(raw)));random.Random(plan['seed']).shuffle(indices)
            for i in indices:
                r=raw[i];cat=r['category'];context=r['context'].strip();prompt=r['instruction'].strip();answer=r['response'].strip()
                if cat in ('open_qa','general_qa') and not context:
                    accept(dict(component='general',group='dolly:'+digest(prompt),messages=pair(prompt,answer),provenance=dict(provenance,row=i,category=cat)),'dolly_general')
                elif cat in ('closed_qa','information_extraction') and context and answer in context and len(answer.split())<=15:
                    accept(dict(component='qa',group='dolly-context:'+digest(context),messages=pair('Use only the context. Give just the answer.\n\nContext:\n'+context+'\n\nQuestion: '+prompt,answer),answers=[answer],answerable=True,provenance=dict(provenance,row=i,category=cat)),'dolly_grounded')
        else:
            paths=oasst_paths(raw,lambda row:len(tokenizer.encode(row['text']))<=96 if row['role']=='assistant' else len(row['text'].split())<=120);random.Random(plan['seed']).shuffle(paths)
            for messages,group,extra in paths:accept(dict(component='chat',group=group,messages=messages,provenance=dict(provenance,**extra)),'oasst')
    # Reuse original pinned HF rows, never weights or optimizer from retired SFT.
    for split in ('train','dev','test'):
        rows=[json.loads(s) for s in (origin/f'{split}.jsonl').read_text().splitlines()];random.Random(plan['seed']).shuffle(rows)
        for r in rows:
            c=r['component']
            if c not in ('chat','qa','constraints','python'):continue
            bucket='everyday' if c=='chat' else c
            if c=='chat':
                msgs=r['messages'];system=[m for m in msgs if m['role']=='system'];body=[m for m in msgs if m['role']!='system']
                if normalized(body[0]['content']) in {'hello','hi','hey','hi there','hello there','hey there','good morning','good evening'}:
                    body=body[2:]
                if not body:continue
                r=dict(r,messages=system+body,opening_removed=True)
            accept(r,bucket,split)
    stats={}
    for split,rows in selected.items():
        random.Random(plan['seed']).shuffle(rows)
        stats[split]=dict(conversations=len(rows),components=dict(Counter(r['component'] for r in rows)),buckets=dict(Counter(r['bucket'] for r in rows)),
            input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),
            assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in {r['component'] for r in rows}},
            multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows))
    if stats['train']['conversations']<5000 or any(stats[s]['components'].get(c,0)<8 for s in ('dev','test') for c in stats['train']['components']):raise ValueError('Insufficient balanced corpus: '+json.dumps(stats))
    reviews={b:[{k:v for k,v in r.items() if not k.startswith('_')} for r in selected['train'] if r['bucket']==b][:30] for b in counts}
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        for s,rows in selected.items():write_split(staging,s,rows)
        atomic_json(staging/'review_examples.json',reviews)
        atomic_json(staging/'curation.json',dict(counts=stats,rejected=dict(rejects),benchmark_exclusions=bench,
            original_hf_manifest=dict(path=str(origin),sha256=file_sha256(origin/'manifest.json')),local_curriculum=plan['local_curriculum'],
            limitations=['Human-written factual answers are not exhaustively verified.','Local social targets are authored; copy/extraction are deterministic, not a new HF dataset.','Generic intents recur across splits; grouping prevents the same tree/context/value from crossing, not all semantic overlap.','Exact/shingle benchmark exclusion is not semantic decontamination.']))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        m=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=stats,files=files,seed=plan['seed'],serialization='explicit-ids-assistant-mask-v1',total_steps=math.ceil(stats['train']['conversations']/64)*plan['epochs'],epochs=plan['epochs'],final_quality_version='balanced-chat-v1')
        atomic_json(staging/'manifest.json',m);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(stats,indent=2),flush=True);return m
