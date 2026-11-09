"""Versioned conversation corpus: complete HF targets and checked local skills.

Paired system/no-system rows are two presentations of one semantic conversation,
not two unique discoveries. HF answers pass structural checks, not a truth oracle.
"""
from collections import Counter, defaultdict
import json
import math
import random
import re
import shutil
import uuid
from .balanced import pair, oasst_paths
from .curate import local, verify, write_split, source_rows, filter_messages, convert, blocked_questions
from .quality import Deduplicator, digest, normalized, direct_general, check_rules, social_prompt
from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v1.tokenization import Tokenizer


def split_group(group, seed, heldout_percent=4):
    n=int(fingerprint({'seed':seed,'group':group})[:8],16)%1000
    return 'test' if n<10*heldout_percent else 'dev' if n<20*heldout_percent else 'train'


def presentations(messages, system):
    body=[dict(m) for m in messages if m['role']!='system']
    return [body,[dict(role='system',content=system),*body]]


def curriculum():
    """Finite, auditable arithmetic/context/exact/social curriculum, no teacher."""
    rows=[]
    def add(kind,group,messages,exact=None):
        component='constraints' if exact is not None else 'chat'
        r=dict(component=component,group='local-v2:'+group,messages=messages,foundation_kind=kind,
            provenance=dict(repo='local/conversation-skills-v2',revision='conversation-skills-v2',split='authored',license='CC0-1.0',method='authored social replies or deterministic target'))
        if exact is not None:r['rules']=[dict(kind='exact',args=[exact])]
        rows.append(r)
    # Arithmetic is checked by construction; equivalent commutative pairs share
    # a split group. Different prompt wordings do not create held-out examples.
    for a in range(1,31):
        for b in range(a,31):
            group=f'numbers-{a}-{b}'
            add('arithmetic',group,pair(f'What is {a} plus {b}? Give only the number.',str(a+b)),str(a+b))
            if a<=12 and b<=12:
                add('arithmetic',group,pair(f'Calculate {a} times {b}. Reply with the number only.',str(a*b)),str(a*b))
            if b>a:add('arithmetic',group,pair(f'What is {b} minus {a}? Give only the number.',str(b-a)),str(b-a))
    labels=['amber','birch','cobalt','daisy','ember','fern','granite','hazel','indigo','jade','kiwi','lavender','maple','nectar','olive','pearl','quartz','ruby','sage','teal','umber','violet','willow','zinc','apricot','basil','cherry','dill','elm','fig','grape','honey','iris','jasmine','kelp','lemon','mint','nutmeg','oak','plum','rose','silk','thyme','walnut','yarrow','azure','bronze','coral','denim','ivory']
    for i,w in enumerate(labels):
        other=labels[(i+17)%len(labels)];group='label-'+w
        for u,a in [(f'Return just the uppercase version of {w}.',w.upper()),(f'Write {w.upper()} in lowercase. No additional text.',w),(f'Copy this label exactly: {w}',w)]:add('exact',group,pair(u,a),a)
        for entity in ('parcel','folder','project','box','ticket'):
            context=f'The {entity} has label {w}. The spare label is {other}.'
            add('grounded',group,pair(context+f'\nGive only the label of the {entity}.',w),w)
            history=pair(f'The {entity} was named {other}.',f'Understood: {other}.')+pair(f'Correction: it is now named {w}.',f'Got it. The current name is {w}.')
            add('correction',group,history+pair(f'What is the current name of the {entity}? Reply with the name only.',w),w)
            add('missing',group,pair(context+f'\nWhat date does the {entity} arrive?', 'The arrival date was not provided.'))
            # Greeting is tied to a real subsequent task, never removed.
            greeting=['Hello','Hi there','Good morning','Good evening','Hey'][i%5]
            add('social-context',group,pair(greeting,'Hello! How can I help?')+pair(f'Please remember that my {entity} is called {w}.',f'Your {entity} is called {w}.')+pair('Thanks for your help.',"You're welcome!")+pair('I have to go now.','Goodbye! Take care.'))
    objects=['document','paragraph','message','schedule','diagram','email','table','plan','draft','invitation','report','list','design','description','outline','presentation','announcement','note','summary','explanation']
    for i,obj in enumerate(objects):
        group='clarify-'+obj
        for verb in ('review','check','improve','explain'):
            add('clarify',group,pair(f'Can you {verb} my {obj}?',f'Could you share the {obj} and describe what you would like help with?'))
            add('clarify-social',group,pair('Hello!','Hi! How can I help?')+pair(f'I need help with my {obj}.',f'Could you share the {obj} and describe what you need help with?'))
    # Single-turn social training is modest; multiple distinct phrasings are
    # authored, not thousands of duplicated "hello" strings.
    greetings=['Hi','Hello','Hey','Hi there','Hello there','Hey there','Good morning','Good afternoon','Good evening','Greetings']
    for i,u in enumerate(greetings):
        add('greeting','greet-'+str(i),pair(u,['Hello!','Hi! How can I help?','Hello! What would you like to talk about?'][i%3]))
    for i,u in enumerate(['Thank you','Thanks for the explanation','That was helpful, thanks','I appreciate your help','Thank you for your time','Much appreciated','Thanks, that makes sense','That answers my question','Many thanks','Thanks for clarifying']):
        add('thanks','thanks-'+str(i),pair(u,["You're welcome!",'Glad I could help.'][i%2]))
    for i,u in enumerate(['Goodbye','See you soon','Bye for now','Talk to you later','Take care','I need to leave','That is all for now','Have a nice evening','Until next time','I am signing off']):
        add('farewell','farewell-'+str(i),pair(u,['Goodbye! Take care.','See you later!'][i%2]))
    return rows


def prior_exclusions(include_all=False):
    """Old inspected holdouts are regression evidence, never new training rows."""
    groups=set(); prompts=set(); receipts=[]
    names=['sft_176m_stage_a_20261007','sft_176m_chat_ready_20261007','sft_176m_balanced_chat_v1_20261007']
    if include_all:names=sorted(set(names)|{p.name for p in local('pretraining/data').glob('sft_*') if p.is_dir()})
    for name in names:
        directory=local('pretraining/data/'+name)
        for split in ('dev','test'):
            path=directory/f'{split}.jsonl'
            if not path.exists():continue
            receipts.append(dict(path=str(path),sha256=file_sha256(path)))
            for line in path.read_text().splitlines():
                r=json.loads(line);groups.add(r['group'])
                for m in r['messages']:
                    if m['role']=='user' and not social_prompt(m['content']):prompts.add(digest(m['content'].rsplit('Question:',1)[-1]))
    probe=local('reports/sft-independent-20261007-probes/declared_cases.json')
    if probe.exists():
        obj=read_json(probe)
        def walk(x):
            if isinstance(x,dict):
                if x.get('role')=='user' and not social_prompt(x.get('content','')):prompts.add(digest(x['content']))
                if isinstance(x.get('prompt'),str):prompts.add(digest(x['prompt']))
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(obj);receipts.append(dict(path=str(probe),sha256=file_sha256(probe)))
    return groups,prompts,receipts


def prepare(config):
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
        result=verify(dest)
        if result['plan_sha256']!=file_sha256(config):raise ValueError('Prepared configuration changed; use a new version')
        return result
    tokenizer=Tokenizer(local(plan['tokenizer']));cache=dest.parent/'.sft_hf_cache'
    blocked,bench=blocked_questions(cache);oldgroups,oldprompts,oldreceipts=prior_exclusions(include_all=plan.get('short_chat_focus',False));blocked.update(oldprompts)
    from .behavior import cases
    for suite in ('chat-relevance-v2','conversation-v3',*(['conversation-v4'] if plan.get('short_chat_focus') else [])):
        blocked.update(digest(m['content']) for split in ('dev','test') for c in cases(split,suite) for m in c['messages'] if m['role']=='user' and not social_prompt(m['content']))
    dedup=Deduplicator(blocked);selected=defaultdict(list);counts=Counter();rejects=Counter();scanned=Counter();groups_seen=set()
    policy=re.compile(r'\b(?:roleplay|pretend|act as|impersonate|diagnos\w*|prescri\w*|medication|investment|stocks|legal advice|tax advice|latest|currently|right now|today.s date)\b',re.I)
    wrongtask=re.compile(r'```|\b(?:rewrite|rephrase|proofread|essay|poem|story|limerick|translate|roleplay|persona|python|javascript|sql|algorithm|calculus)\b',re.I)
    def accept(row,bucket):
        scanned[bucket]+=1
        try:
            from .pure_python import verify_target
            if any(row['provenance'].get('repo')==x['repo'] and row['provenance'].get('row')==x['row'] for x in plan.get('review_exclusions',[])):raise ValueError('manual_review_rejection')
            group=row['group'];msgs=[m for m in row['messages'] if m['role']!='system']
            if group in oldgroups:raise ValueError('previous_holdout_group')
            split=split_group(group,plan['seed'],plan.get('heldout_group_percent',4));cap=plan['conversation_caps'].get(bucket,0) if split=='train' else plan['heldout_per_bucket']
            if counts[(split,bucket)]>=cap:raise ValueError('cap')
            variants_count=1 if plan.get('short_chat_focus') else 2
            if split=='train' and len(selected['train'])+variants_count>plan['max_conversations']:raise ValueError('total_cap')
            if bucket=='qa':
                subcap=int(cap*(.8 if row['answerable'] else .2))
                if counts[(split,'qa-'+str(row['answerable']))]>=subcap:raise ValueError('qa_balance_cap')
            if plan.get('short_chat_focus') and bucket in ('magpie_direct','magpie_chat','dolly','oasst','everyday','constraints'):
                from .short_chat import check_focus
                check_focus(msgs)
            if policy.search(' '.join(m['content'] for m in msgs)):raise ValueError('time_sensitive_or_persona')
            if any(len(m['content'].split())>140 for m in msgs if m['role']=='user') and row['component']!='qa':raise ValueError('complex_prompt')
            if bucket in ('everyday','magpie_chat','oasst') and re.search(r'\b(?:my hometown|this city|our city|our town|my local|in my area|near me)\b',' '.join(m['content'] for m in msgs),re.I):raise ValueError('underspecified_local_context')
            if bucket in ('magpie_direct','magpie_chat','dolly','oasst'):
                if re.search(r'\b(?:game of thrones|targaryen|jon snow|daenerys|harry potter|weasley|pokemon|anime)\b',' '.join(m['content'] for m in msgs),re.I):raise ValueError('niche_fiction')
                if re.search(r'\b(?:your database|your training data|my training data|my knowledge cutoff|i was trained|i have a vast amount of training data)\b',' '.join(m['content'] for m in msgs),re.I):raise ValueError('unsupported_model_identity')
                if re.search(r'\b(?:luau|roblox|code|script|programming|function|proof|theorem|derivative|integral|differential equation)\b',' '.join(m['content'] for m in msgs if m['role']=='user'),re.I):raise ValueError('specialist_general_task')
                if wrongtask.search(' '.join(m['content'] for m in msgs if m['role']=='user')):raise ValueError('wrong_task')
                if bucket in ('magpie_direct','dolly') and not direct_general(msgs):raise ValueError('not_direct_qa')
            if any(re.search(r'\b(?:my (?:wife|husband|children)|i live in|i was born|my name is|i am a (?:teacher|doctor))\b',m['content'],re.I) for m in msgs if m['role']=='assistant'):raise ValueError('invented_personal_identity')
            if row['component']=='python':
                if plan.get('short_chat_focus') and any(len(tokenizer.encode(m['content']))>plan['python_max_tokens'] for m in msgs if m['role']=='assistant'):raise ValueError('python_answer_length')
                tests=verify_target(msgs[0]['content'],msgs[-1]['content'])
                if tests is None:raise ValueError('unverified_python_contract')
                row=dict(row,functional_assertions=tests)
            if row['component']=='constraints' and not all(check_rules(msgs[-1]['content'],row['rules']) for _ in [0]):raise ValueError('failed_constraint')
            # Context-dependent prompts recur legitimately. Deduplicate full
            # dialogue, but screen raw questions against heldout/benchmark hashes.
            dependent=row.get('foundation_kind') in ('correction','social-context','clarify-social') or (plan.get('short_chat_focus') and sum(m['role']=='user' for m in msgs)>1)
            if any(digest(m['content'].rsplit('Question:',1)[-1]) in blocked for m in msgs if m['role']=='user' and not social_prompt(m['content'])):raise ValueError('excluded_prompt')
            dm=pair('\n'.join(m['content'] for m in msgs),msgs[-1]['content']) if dependent else msgs
            if plan.get('short_chat_focus'):
                from .short_chat import presentation
                variants=presentation(msgs,plan['conversation_system'],group)
            else:variants=presentations(msgs,plan['conversation_system'])
            encoded=[filter_messages(v,tokenizer,row['component'],plan) for v in variants]
            if not dedup.accept(dm,row['component']):raise ValueError('duplicate_or_overlap')
            # Both variants share their semantic group and split. Cap heldout
            # rows without fabricating extra independent examples.
            for variant,(ids,mask) in enumerate(encoded):
                selected[split].append(dict(row,messages=variants[variant],split=split,bucket=bucket,system_variant=variants[variant][0]['role']=='system',_ids=ids,_mask=mask))
            counts[(split,bucket)]+=1;groups_seen.add(group)
            if bucket=='qa':counts[(split,'qa-'+str(row['answerable']))]+=1
        except ValueError as e:rejects[bucket+':'+str(e)]+=1
    if plan.get('short_chat_focus'):
        from .short_chat import focused_curriculum
        skills=focused_curriculum()
    else:skills=curriculum()
    random.Random(plan['seed']).shuffle(skills)
    for r in skills:accept(r,'skills')
    for source in plan['sources']:
        raw=source_rows(source,cache);provenance={k:source[k] for k in ('repo','config','revision','split','license')}
        rng=random.Random(plan['seed']);indices=list(range(len(raw)));rng.shuffle(indices)
        repo=source['repo']
        if repo=='OpenAssistant/oasst2':
            paths=oasst_paths(raw,lambda r:len(tokenizer.encode(r['text']))<=plan['assistant_max_tokens'] if r['role']=='assistant' else len(r['text'].split())<=140)
            rng.shuffle(paths)
            for msgs,group,extra in paths:accept(dict(component='chat',group=group,messages=msgs,provenance=dict(provenance,**extra)),'oasst')
            print(f'[conversation] oasst complete eligible paths={len(paths)} selected={counts[("train","oasst")]}',flush=True);continue
        for n,i in enumerate(indices):
            r=raw[i]
            try:
                extra={}
                if repo=='databricks/databricks-dolly-15k':
                    if r['category'] not in ('open_qa','general_qa') or r['context'].strip():continue
                    msgs=pair(r['instruction'].strip(),r['response'].strip());group='dolly:'+digest(msgs[0]['content']);bucket='dolly';component='general';extra['category']=r['category']
                elif repo=='HuggingFaceTB/smol-smoltalk':
                    src=r['source']
                    if src not in ('smol-magpie-ultra-short','everyday-conversations'):continue
                    # Retain complete histories, no answer chopping/truncation.
                    msgs=[dict(role=m['role'],content=m['content']) for m in r['messages']]
                    while msgs and msgs[-1]['role']!='assistant':msgs.pop()
                    if not msgs:continue
                    users=[m['content'] for m in msgs if m['role']=='user']
                    group='dialogue:'+digest('\n'.join(p for p in users if not social_prompt(p)))
                    # Most Magpie records are multi-turn. Reserve half of their
                    # source rows for the initial standalone direct QA pair;
                    # the remaining half retains the full dialogue. The answer
                    # itself is unchanged and complete, never token-truncated.
                    if plan.get('short_chat_focus'):
                        from .short_chat import complete_prefix
                        msgs=complete_prefix(msgs,tokenizer,plan)
                        # Prefixes and longer conversations with the same first
                        # question must share a split, regardless of formatting.
                        users=[m['content'] for m in msgs if m['role']=='user']
                        substantive=[p for p in users if not social_prompt(p)]
                        group='dialogue:'+digest(substantive[0] if substantive else '\n'.join(users))
                        users=[m['content'] for m in msgs if m['role']=='user']
                    direct=src=='smol-magpie-ultra-short' and (i%(4 if plan.get('short_chat_focus') else 2)==0 or len(users)==1)
                    bucket='everyday' if src=='everyday-conversations' else 'magpie_direct' if direct else 'magpie_chat'
                    if direct:
                        msgs=msgs[:2];group='dialogue:'+digest(msgs[0]['content'])
                        extra['selection']='complete first direct question/answer pair'
                    component='general' if bucket=='magpie_direct' else 'chat';extra['subset']=src
                    # Do not let broad Magpie histories be editing lessons.
                    if bucket=='magpie_chat' and not plan.get('short_chat_focus') and not direct_general(msgs):raise ValueError('not_direct_first_turn')
                else:
                    msgs,group,extra=convert(r,dict(source,chat_redo=True));component=source['component'];bucket=component
                accept(dict(component=component,group=group,messages=msgs,provenance=dict(provenance,row=i,**{k:v for k,v in extra.items() if k in ('category','subset','selection')}),**{k:v for k,v in extra.items() if k not in ('category','subset','selection')}),bucket)
            except ValueError as e:rejects[repo+':'+str(e)]+=1
            if (n+1)%50000==0:print(f'[conversation] {repo} scanned={n+1}/{len(raw)} train semantic counts={dict((b,c) for (s,b),c in counts.items() if s=="train")}',flush=True)
        print(f'[conversation] finished {repo}; train semantic counts={dict((b,c) for (s,b),c in counts.items() if s=="train")}',flush=True)
    stats={}
    for split,rows in selected.items():
        random.Random(plan['seed']).shuffle(rows)
        stats[split]=dict(conversations=len(rows),semantic_conversations=len(rows) if plan.get('short_chat_focus') else len(rows)//2,semantic_groups=len({r['group'] for r in rows}),components=dict(Counter(r['component'] for r in rows)),buckets=dict(Counter(r['bucket'] for r in rows)),
            input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in {r['component'] for r in rows}},multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows))
    if stats['train']['semantic_conversations']<10000 or any(stats[s]['components'].get(c,0)<8 for s in ('dev','test') for c in ('chat','general','qa','constraints','python')):raise ValueError('Insufficient conversation corpus: '+json.dumps(stats))
    if plan.get('short_chat_focus') and (stats['train']['multi_turn']/stats['train']['conversations']<.45 or stats['train']['components'].get('python',0)<100):raise ValueError('Insufficient multi-turn/Python coverage: '+json.dumps(stats['train']))
    groups={s:{r['group'] for r in rows} for s,rows in selected.items()}
    if any(groups[a]&groups[b] for a,b in (('train','dev'),('train','test'),('dev','test'))):raise ValueError('Semantic group leakage')
    reviews={b:[{k:v for k,v in r.items() if not k.startswith('_')} for r in selected['train'] if r['bucket']==b and not r['system_variant']][:30] for b in plan['conversation_caps']}
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        for s,rows in selected.items():write_split(staging,s,rows)
        atomic_json(staging/'review_examples.json',reviews)
        limitations=['HF factual/synthetic answers are structurally filtered, not exhaustively fact checked.','Local curriculum is authored/deterministic, not an HF dataset.','Semantic decontamination cannot be established by exact/shingle screening.','General prompt topics can recur; complete tree/context groups do not cross splits.']
        if not plan.get('short_chat_focus'):limitations.append('System/no-system variants are correlated presentations, not unique conversations.')
        atomic_json(staging/'curation.json',dict(counts=stats,rejected=dict(rejects),scanned=dict(scanned),benchmark_exclusions=bench,previous_holdout_exclusions=oldreceipts,local_curriculum=plan['local_curriculum'],
            limits=plan['conversation_caps'],limitations=limitations))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        result=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=stats,files=files,seed=plan['seed'],serialization='explicit-ids-assistant-mask-v1',epochs=plan['epochs'],final_quality_version='short-chat-focus-v1' if plan.get('short_chat_focus') else 'conversation-v2',total_steps=math.ceil(stats['train']['conversations']/64)*plan['epochs'])
        atomic_json(staging/'manifest.json',result);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(stats,indent=2),flush=True);return result
