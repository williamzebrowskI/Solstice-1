"""SFT-038: short evidence and explicit whole-exchange contracts.

Source offsets verify supplied evidence, not universal factual truth. Authored
records are synthetic and schema-correlated. Never execute generated Python.
Historical SFT-037 scoring remains unchanged.
"""
from collections import Counter, defaultdict
import ast
import hashlib
import json
from pathlib import Path
import random
import re

from sml_v1.common import atomic_json, file_sha256, read_json
from sml_v1.tokenization import Tokenizer
from .curate import local, blocked_questions
from .conversation import split_group
from .quality import digest, normalized, repeated
from .reliable_chat import cached, encode
from .behavior_v8 import python_checks
from .verified_extension import SOURCE, RESERVED_TOPICS, source_span

FOLDER = 'reports/sft/SFT-038-short-evidence-dialogue-20261008'
QUOTAS = dict(natural_qa=2000, evidence=2000, dialogue=2000, format=1500, python=2000, social=500)


def json_equal(actual, expected):
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict):
        return actual.keys()==expected.keys() and all(json_equal(actual[k],v) for k,v in expected.items())
    if isinstance(expected, list):
        return len(actual)==len(expected) and all(json_equal(a,b) for a,b in zip(actual,expected))
    return actual==expected


def value_key(text):
    # Preserve signs, decimals and internal punctuation; optional sentence end
    # is permitted only for semantic-value contracts, not exact formatting.
    return ' '.join(text.strip().rstrip('.!?').casefold().split())


def check_contract(contract, text):
    text=text.strip(); kind=contract['kind']
    if not text or repeated(text): return False
    if kind=='value': return value_key(text) in {value_key(a) for a in contract['answers']}
    if kind=='exact': return text==contract['expected']
    if kind=='json':
        try: return json_equal(json.loads(text),contract['expected'])
        except (ValueError,TypeError): return False
    if kind=='bullets':
        lines=text.splitlines()
        return len(lines)==len(contract['items']) and all(
            re.fullmatch(r'[-*] '+re.escape(item),line.strip()) for item,line in zip(contract['items'],lines))
    if kind=='sentences':
        sentences=[x.strip() for x in re.split(r'(?<=[.!?])\s+',text) if x.strip()]
        return (len(sentences)==contract['count'] and len(text.split())<=contract['max_words']
                and not text.startswith(('-', '*'))
                and all(value_key(fact) in value_key(text) for fact in contract['facts']))
    if kind=='request':
        aliases={'text':('text','message','passage','paragraph')}
        return (len(text.split())<=35 and bool(re.search(r'\b(?:paste|send|share|provide)\b',text,re.I))
                and all(any(re.search(r'\b'+re.escape(alias)+r'\b',text,re.I) for alias in aliases.get(item,(item,))) for item in contract['objects'])
                and not re.search(r'\b(?:fixed it|the error is|revised version)\b',text,re.I))
    if kind=='social':
        return (len(text.split())<=45 and bool(re.search(contract['pattern'],text,re.I))
                and not re.search(r'\b(?:my name|my day|your day is|my family|tornadoes|earthquake|grammar tutor)\b',text,re.I))
    if kind=='python':
        valid=all(python_checks(text,contract['tests'],function=contract['function'],
                                code_only=contract.get('code_only',True)).values())
        if contract.get('explanation'):
            # A working code block alone does not satisfy an explanation request.
            prose=re.sub(r'```.*?```','',text,flags=re.S).strip()
            valid=valid and len(prose.split())>=5 and prose.endswith(('.', '!', '?'))
        return bool(valid)
    raise ValueError('Unknown contract '+kind)


def value(answer): return dict(kind='value',answers=[str(answer)])
def exact(answer): return dict(kind='exact',expected=str(answer))
def jcontract(answer): return dict(kind='json',expected=answer)


def row(bucket, group, turns, provenance, family):
    """Every assistant turn, not just the final answer, has a contract."""
    messages=[];contracts=[]
    for prompt,answer,contract in turns:
        messages.extend([dict(role='user',content=prompt),dict(role='assistant',content=answer)])
        contracts.append(contract)
    component='python' if bucket=='python' else 'qa' if bucket in ('natural_qa','evidence') else 'constraints' if bucket=='format' else 'chat'
    r=dict(bucket=bucket,component=component,group=group,messages=messages,turn_contracts=contracts,
           provenance=provenance,family=str(family))
    final=contracts[-1]
    if bucket in ('natural_qa','evidence'):r.update(answers=final.get('answers',[answer]),answerable=True)
    if bucket=='python':r.update(functional_assertions=final['tests'],python_contract=dict(function=final['function'],code_only=True,docstring=False))
    if bucket=='format':r['rules']=[dict(kind='exact',args=[answer])]
    return r


def validate_row(r):
    assistants=[m['content'] for m in r['messages'] if m['role']=='assistant']
    return len(assistants)==len(r['turn_contracts']) and all(check_contract(c,a) for c,a in zip(r['turn_contracts'],assistants))


def authored_rows():
    people=['Rina','Omar','Vera','Kato','Elsa','Bryn','Dara','Finn','Lena','Cora','Noel','Yuri','Ada','Evan','Ruth','Seth']
    things=['ticket','badge','coupon','bookmark','card','label','voucher','receipt','token','tag','note','slip','sheet','form','seal','pass']
    places=['studio','office','lobby','attic','basement','terrace','cellar','garage','porch','loft','balcony','workshop']
    colors=['violet','orange','silver','gold','brown','black','white','pink']
    for i in range(1100):
        rng=random.Random(263800+i);a,b=rng.sample(people,2);x,y,z,w=rng.sample(things,4);p,q=rng.sample(places,2)
        n=5+i%21; delta=1+i%4; extra=2+i%5; tag=f'LOG-{i+7000}'; color=colors[i%len(colors)]
        group='038:record:'+str(i); split=split_group(group,263800,12)
        # Wording held out by split, in addition to record/parameter grouping.
        ask={'train':'Give only','dev':'Respond with just','test':'Answer using only'}[split]
        prov=dict(repo='local/evidence-instructions',revision='v1',license='CC0-1.0',split='authored',
                  record=dict(person=a,other=b,item=x,other_item=y,place=p,excluded_place=q,quantity=n,delta=delta,added=extra,color=color,identifier=tag),
                  method='Explicit fictional record and independent state oracle; correlated schema variants')
        prompt=f'Fictional note: The {x} is not in the {q}. The {x} is in the {p}. The {y} is in the {q}.\nWhere is the {x}? {ask} the location.'
        yield row('evidence',group,[(prompt,p,value(p))],prov,'negation')
        prompt=f'Fictional roster: {a} supervises the {p}. {b} works in the {q}. {b} delivered a {x} to the {p}.\nWho supervises the {p}? {ask} the name.'
        yield row('evidence',group,[(prompt,a,value(a))],prov,'relation')
        prompt=f'Record: {a} has {n} {x} items. The record has no arrival date.\nWhat is the arrival date? Say Unknown if absent; otherwise give the recorded date.'
        yield row('evidence',group,[(prompt,'Unknown',value('Unknown'))],prov,'missing')
        prompt=f'Record: {a} has {n} {x} items and arrives on day {i%28+1}.\nWhat is the arrival date? Say Unknown if absent; otherwise give the recorded date.'
        yield row('evidence',group,[(prompt,f'day {i%28+1}',value(f'day {i%28+1}'))],prov,'present')
        limit=i+20
        prompt=f'Policy: A parcel is accepted when its count is at least {limit}. This parcel has {limit-1 if i%2 else limit} items.\nIs it accepted? {ask} yes or no.'
        answer='no' if i%2 else 'yes'
        yield row('evidence',group,[(prompt,answer,value(answer))],prov,'conditional')
        prompt=f'I have {n} {x} items. {ask} the current count.'
        turns=[(prompt,str(n),value(n)),(f'I use {delta} of those items. {ask} the remaining count.',str(n-delta),value(n-delta)),
               (f'I obtain {extra} more. {ask} the total now.',str(n-delta+extra),value(n-delta+extra))]
        yield row('dialogue',group,turns,prov,'arithmetic')
        old=f'{i%24:02d}:00';new=f'{(i+5)%24:02d}:00';other_time=f'{(i+3)%24:02d}:00'
        turns=[(f'{a} booked the {p} at {old}. {b} booked the {q} at {other_time}. {ask} JSON mapping each person to their time.',json.dumps({a:old,b:other_time}),jcontract({a:old,b:other_time})),
               (f'Change only {a} to {new}. {ask} JSON mapping each person to their time.',json.dumps({a:new,b:other_time}),jcontract({a:new,b:other_time})),
               (f'What was {a}\'s original time? {ask} the time.',old,value(old))]
        yield row('dialogue',group,turns,prov,'two-entity')
        turns=[(f'My ordered items are {x}, {y}, {z}. {ask} the list as JSON.',json.dumps([x,y,z]),jcontract([x,y,z])),
               (f'Remove {y} and append {w}. {ask} the updated list as JSON.',json.dumps([x,z,w]),jcontract([x,z,w])),
               (f'Now {ask.lower()} the updated items separated by comma-space.',f'{x}, {z}, {w}',exact(f'{x}, {z}, {w}'))]
        yield row('dialogue',group,turns,prov,'representation-change')
        turns=[(f'{a}\'s meeting is on day {i%28+1} at {old}. {ask} the day and time.',f'day {i%28+1} at {old}',value(f'day {i%28+1} at {old}')),
               (f'Change only the day to day {(i+5)%28+1}. {ask} the day and time.',f'day {(i+5)%28+1} at {old}',value(f'day {(i+5)%28+1} at {old}'))]
        yield row('dialogue',group,turns,prov,'preserve-field')
        prompt=f'Record: color={color}; count={i%4}; reference={tag}; item={x}.\n{ask} JSON with exactly keys "color", "count" and "reference"; count must be an integer.'
        out=dict(color=color,count=i%4,reference=tag)
        yield row('format',group,[(prompt,json.dumps(out),jcontract(out))],prov,'json-fields')
        prompt=f'{ask} the exact identifier {tag}. Keep its case and punctuation unchanged.'
        yield row('format',group,[(prompt,tag,exact(tag))],prov,'literal')
        prompt=f'List {x}, {y}, {z} in that order, using exactly three bullet points. Include nothing else.' if split=='train' else f'Use one bullet per item and no introduction: {x}, {y}, {z}. Preserve the order.'
        yield row('format',group,[(prompt,f'- {x}\n- {y}\n- {z}',dict(kind='bullets',items=[x,y,z]))],prov,'bullets')
        article='an' if color[0] in 'aeiou' else 'a'
        note=f'{a} has {article} {color} {x}. {b} has a {y}.'
        prompt=f'Note: {note}\nRestate both facts in exactly two short sentences, without an introduction.' if split=='train' else f'Give a two-sentence summary of this note, retaining both facts: {note}'
        yield row('format',group,[(prompt,note,dict(kind='sentences',count=2,max_words=24,facts=[f'{a} has {article} {color} {x}',f'{b} has a {y}']))],prov,'sentences')
        opener=['Can we discuss something?','I need a little assistance.','Are you available to talk?','Could we work through a question?'][i%4]
        greeting='Yes, I am here to help. What would you like to discuss?'
        closing='Goodbye. Take care.'
        turns=[(opener,greeting,dict(kind='social',pattern=r'\b(?:help|discuss|talk)\b')),
               (f'Read this note: {a} has {article} {color} {x}. What color is it? {ask} the color.',color,value(color)),
               ('That helps. I must leave now.',closing,dict(kind='social',pattern=r'\b(?:goodbye|take care|bye)\b'))]
        yield row('social',group,turns,prov,'complete-exchange')
        prompt=f'I want you to revise a {x} message, but I have not included the text.'
        yield row('social',group,[(prompt,'Please paste the text you want revised.',dict(kind='request',objects=['text']))],prov,'clarify')


def python_rows():
    # Explicit TRAIN replay plus additional behavior-changing parameter specs.
    # Families recur across groups: this is not an unseen-program-family test.
    from .verified_extension import python_rows as originals
    # Reuse verified task definitions with new specification groups, then add
    # complementary boundary contracts. Historical results remain untouched.
    for r in originals():
        family=r['provenance']['family'];n=r['provenance']['parameter']
        if split_group(r['group'],263700,12)!='train':continue
        fn=f'checked_{family}_{n}';target=r['messages'][-1]['content'];prompt=r['messages'][0]['content']
        tests=list(r['functional_assertions'])
        if family==17:tests += [dict(args=[f'tag{n}'],expected=False),dict(args=[f'tag{n}x'],expected=False),dict(args=[f'TAG{n}:'],expected=False)]
        if family==1:tests += [dict(args=[[n]],expected=[])]
        if family==13:tests += [dict(args=[[n]],expected=1)]
        c=dict(kind='python',function=fn,tests=tests)
        yield row('python',f'038:python-spec:{family}:{n}',[(prompt,target,c)],dict(r['provenance'],replay='SFT-037 training family definitions only; old heldout families excluded',method='Verified TRAIN task replay with all original assertions; never represented as new independent source data'),'reviewed-'+str(family))
    # Broader instruction coverage: two arguments, types, boundary guards,
    # mutation-free transformations. Each parameter changes actual behavior.
    for i in range(500):
        n=i+2; label=f'unit{n}'; token=f'/{n}/';group='038:code-spec:'+str(i)
        prov=dict(repo='local/evidence-instructions',revision='v1',license='CC0-1.0',split='authored',parameter=n,method='Authored precise contract with literal ordinary/edge assertions, bounded interpreter')
        tasks=[
          ('guard', 'items, k', f'[] if k <= 0 else items[:min(k, {n})]', f'take the first k items but never more than {n}; return an empty list for zero or negative k', [([[],3],[]),([[n,n+1],-1],[]),([[n,n+1],0],[]),([[n,n+1],1],[n]),([list(range(n+2)),n+10],list(range(n)))]),
          ('lookup', 'record, key', f'record.get(key, {-n})', f'return a stored dictionary value, preserving zero and None; return {-n} if the key is absent', [([{label:0},label],0),([{label:None},label],None),([{},label],-n),([{label:n},label],n)]),
          ('suffix', 'text', f'text.endswith("{token}")', f'return whether text ends with the exact case-sensitive suffix "{token}"', [(['x'+token],True),([token+'x'],False),([token+' '],False),([str(n)],False),([''],False)]),
          ('spaces', 'text', f'text.strip().upper()[{n}:]', f'remove leading and trailing whitespace, uppercase the text, then remove the first {n} characters; preserve remaining internal spaces', [(['  '+('a'*n)+' bc  '],' BC'),(['\t\n'],''),([''],''),([('x'*n)+' ab cd'],' AB CD')]),
          ('threshold', 'items', f'[x for x in items if x < {n}]', f'keep only numeric values strictly less than {n}, preserving order', [([[]],[]),([[n,n-1,n+1]],[n-1]),([[-1,0]],[-1,0]),([[n,n]],[])])]
        for name,args,expr,desc,test_pairs in tasks:
            fn=f'practice_{name}_{n}';tests=[dict(args=a,expected=b) for a,b in test_pairs]
            target=f'```python\ndef {fn}({args}):\n    return {expr}\n```'
            yield row('python',group,[(f'Write {fn}({args}) in Python to {desc}. Return one Python code block only.',target,dict(kind='python',function=fn,tests=tests))],prov,name)


def benchmark_exclusions():
    blocked,receipt=blocked_questions(local('pretraining/data/.sft_hf_cache'));grams=set()
    from datasets import Dataset
    for repo,field in [('allenai___ai2_arc','question'),('google___ifeval','prompt')]:
        for path in local('pretraining/data/.sft_hf_cache/'+repo).rglob('*.arrow'):
            for r in Dataset.from_file(str(path)):
                words=normalized(r[field]).split();grams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    frozen=local('reports/benchmarks/ifeval-SFT-036-step96-20261008/inputs.json')
    for r in read_json(frozen):
        words=normalized(r['prompt']).split();grams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    return blocked,grams,receipt


def build(config):
    plan=read_json(config);folder=local(FOLDER);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'catalog.jsonl'
    if path.exists():raise ValueError('Existing immutable SFT-038 catalog; do not overwrite')
    tok=Tokenizer(local(plan['tokenizer']));blocked,grams,bench=benchmark_exclusions()
    prior=local('reports/sft/SFT-037-verified-extension-20261008/catalog.jsonl')
    oldrows=[json.loads(line) for line in prior.read_text().splitlines()]
    forbidden_titles={r['provenance'].get('title') for r in oldrows if r['split']!='train'}
    forbidden_ids={r['provenance'].get('id') for r in oldrows}
    protected_paths=[local('reports/sft/sft036-extension-diagnostics-20261008/protocol.json'),local('reports/sft/SFT-037-verified-extension-20261008/fresh-playground-diagnostics/probes.json')]
    for p in protected_paths:
        obj=read_json(p);cases=obj['cases'] if isinstance(obj,dict) else obj
        blocked.update(digest(q) for c in cases for q in c['prompts'])
    blocked.update(digest(m['content']) for r in oldrows if r['split']!='train' for m in r['messages'] if m['role']=='user')
    pools=defaultdict(list);rejected=Counter();seen=set()
    def accept(r):
        try:
            if not validate_row(r):raise ValueError('target_contract')
            users=[m['content'] for m in r['messages'] if m['role']=='user']
            for q in users:
                if digest(q) in blocked:raise ValueError('reserved_probe')
                words=normalized(q).split()
                if any(' '.join(words[i:i+8]) in grams for i in range(len(words)-7)):raise ValueError('benchmark_ngram')
            encode(r['messages'],tok,plan,r['component'],r['group'])
            key=digest(json.dumps(r['messages'],sort_keys=True))
            if key in seen:raise ValueError('duplicate_history')
            seen.add(key);r['split']=split_group(r['group'],263800,12);pools[(r['split'],r['bucket'])].append(r)
        except ValueError as e:rejected[str(e)]+=1
    data,source_receipt=cached(SOURCE)
    for i,raw in enumerate(data):
        if raw['id'] in forbidden_ids or raw['title'] in forbidden_titles:continue
        ctx=raw['context'];q=raw['question']
        if digest(q) in blocked:rejected['reserved_raw_question']+=1;continue
        if not 20<=len(ctx.split())<=90 or not 4<=len(q.split())<=30 or RESERVED_TOPICS.search(ctx+' '+q) or '[citation needed]' in ctx:continue
        try:a=source_span(raw)
        except ValueError:continue
        if len(a.split())>6:continue
        # Exclude the exact fresh factual families, not just their raw wording.
        if re.search(r'\b(?:stomata|nitrogen|carbon dioxide|alpha centauri|closest star|largest ocean|seasons|conducts? electricity)\b',ctx+' '+q,re.I):continue
        prov=dict(SOURCE,row=i,id=raw['id'],title=raw['title'],context=ctx,answer_offsets=raw['answers']['answer_start'],context_sha256=hashlib.sha256(ctx.encode()).hexdigest(),method='Whole short paragraph; human question and answer offsets retained; not fabricated short answers')
        group='038:article:'+digest(raw['title']);prompt=f'Use this passage: {ctx}\nQuestion: {q}\nGive a short answer using the passage.'
        contract=dict(kind='value',answers=raw['answers']['text'])
        accept(row('natural_qa',group,[(prompt,a,contract)],prov,'source-span'))
    for r in authored_rows():accept(r)
    for r in python_rows():accept(r)
    rows=[]
    for split in ('train','dev','test'):
        for bucket,quota in QUOTAS.items():
            pool=pools[(split,bucket)];pool.sort(key=lambda r:digest('038'+json.dumps(r['messages'],sort_keys=True)))
            cap=quota if split=='train' else 60
            # Round-robin families, not first N easy examples. Source article
            # cap prevents one prolific source title dominating the QA quota.
            families=defaultdict(list)
            for r in pool:families[r['family']].append(r)
            selected=[];group_count=Counter()
            while len(selected)<cap and any(families.values()):
                for family in sorted(families):
                    if not families[family]:continue
                    r=families[family].pop()
                    if bucket=='natural_qa' and group_count[r['group']]>=20:continue
                    selected.append(r);group_count[r['group']]+=1
                    if len(selected)==cap:break
            if len(selected)!=cap:raise ValueError(f'Insufficient {split}/{bucket}: {len(selected)}/{cap}; pool {len(pool)}')
            rows.extend(selected)
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','dev','test')}
    if any(groups[a]&groups[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):raise ValueError('Group leakage')
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    receipt=dict(format='opensml-evidence-instructions-catalog-v1',sha256=file_sha256(path),
                 counts={s:dict(conversations=sum(r['split']==s for r in rows),groups=len(groups[s]),buckets=dict(Counter(r['bucket'] for r in rows if r['split']==s)),families={b:dict(Counter(r['family'] for r in rows if r['split']==s and r['bucket']==b)) for b in QUOTAS}) for s in groups},
                 source_receipts=[dict(source=SOURCE,**source_receipt)],rejected=dict(rejected),benchmark_exclusions=bench,
                 prior_catalog_sha256=file_sha256(prior),protected_files={str(p.relative_to(local('.'))):file_sha256(p) for p in protected_paths},
                 checks='Every assistant target has a checked contract; grouped splits; full serialization; no duplicate padding; ordinary/edge code tests; whole source offsets; exact/8-gram benchmark exclusion.',
                 limitations=['Synthetic templates, social intents and program families recur across groups; these are parameter/record holdouts, not unseen task families.','Source annotations and evidence alignment do not certify all world facts.','New data principally targets instruction transfer, not a guaranteed increase in open-domain knowledge.','Exact/8-gram exclusion does not prove semantic benchmark decontamination.'])
    atomic_json(folder/'catalog.json',receipt)
    # Frozen test selection is stratified by family, includes all Python
    # families (at most 25), and is never consulted by in-run ranking.
    panel=[]
    for bucket in QUOTAS:
        available=[r for r in rows if r['split']=='test' and r['bucket']==bucket];families=defaultdict(list)
        for r in available:families[r['family']].append(r)
        n=25 if bucket=='python' else 20
        while n and any(families.values()):
            for family in sorted(families):
                if families[family] and n:panel.append(dict(families[family].pop(),name=f'{bucket}-{len(panel)}'));n-=1
    atomic_json(folder/'acceptance-panel.json',dict(format='opensml-evidence-instructions-test-v1',cases=panel,
        policy=dict(min_macro_gain=.10,min_evidence_gain=.10,min_dialogue_gain=.10,max_category_regression=.05,min_python=.50,min_format=.50,retention_ceiling=2.6814481621980665,require_manual_review=True),
        scope='Fresh record/source/spec holdouts; all consequential turns checked. Synthetic wording and task schemas are correlated. Not a general chatbot or benchmark accuracy estimate.'))
    plan['catalog_sha256']=receipt['sha256'];atomic_json(config,plan)
    print(json.dumps(receipt['counts'],indent=2),flush=True)


def generated_panel(model,tokenizer,rows,*,system=None,limit=192):
    from .evaluate import generate
    samples=[];scores=defaultdict(list)
    for r in rows:
        history=[dict(role='system',content=system)] if system else [];turns=[];ci=0
        for m in r['messages']:
            if m['role']!='user':continue
            history.append(dict(m));result=generate(model,tokenizer,history,limit=limit,temperature=0)
            contract=r['turn_contracts'][ci];ci+=1
            ok=check_contract(contract,result['text']) and result['stop_reason'] in ('turn_end','document_end')
            turns.append(dict(request=[dict(x) for x in history],result=result,contract=contract,passed=bool(ok)))
            history.append(dict(role='assistant',content=result['text']))
            if not result['text'].strip():break
        passed=len(turns)==len(r['turn_contracts']) and all(t['passed'] for t in turns)
        scores[r['bucket']].append(passed)
        samples.append(dict(name=r.get('name',r['group']),bucket=r['bucket'],family=r['family'],turns=turns,passed=bool(passed)))
    rates={b:sum(v)/len(v) for b,v in scores.items()}
    return dict(rates=rates,macro_pass=sum(rates.values())/len(rates),samples=samples,
                scope='All target-bearing turns must pass; macro across six buckets; source/record/spec holdouts, correlated schemas; manual semantic review still required.')


def development(model,tokenizer,prepared,plan):
    rows=[json.loads(line) for line in (Path(prepared)/'dev.jsonl').read_text().splitlines()];selected=[]
    for bucket in QUOTAS:
        families=defaultdict(list)
        for r in rows:
            if r['bucket']==bucket:families[r['family']].append(r)
        n=plan.get('evidence_dev_per_bucket',8)
        if bucket=='python':n=max(n,len(families))
        while n and any(families.values()):
            for family in sorted(families):
                if families[family] and n:selected.append(families[family].pop());n-=1
    panels=[generated_panel(model,tokenizer,selected,system=s,limit=192) for s in (None,plan['conversation_system'])]
    rates={b:sum(p['rates'][b] for p in panels)/2 for b in QUOTAS}
    return dict(rates=rates,macro_pass=sum(rates.values())/len(rates),samples=[x for p in panels for x in p['samples']],scope=panels[0]['scope'])


def compare(baseline,candidate,policy,retention):
    checks={'original_retention_ceiling':retention<=policy['retention_ceiling']}
    for variant,b in baseline['metrics'].items():
        c=candidate['metrics'][variant]
        checks[variant+':macro_gain']=c['macro_pass']>=b['macro_pass']+policy['min_macro_gain']
        for bucket in ('evidence','dialogue'):
            checks[variant+':'+bucket+'_gain']=c['rates'][bucket]>=b['rates'][bucket]+policy['min_'+bucket+'_gain']
        for bucket,v in b['rates'].items():checks[variant+':'+bucket+'_retained']=c['rates'][bucket]>=v-policy['max_category_regression']
        checks[variant+':python_floor']=c['rates']['python']>=policy['min_python']
        checks[variant+':format_floor']=c['rates']['format']>=policy['min_format']
    passed=all(checks.values())
    return dict(checks=checks,numerical_pass=passed,automatic_promotion=False,status='provisional; inspect every relevant turn and semantic failures' if passed else 'not accepted; preserve report and inspect failures')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);build(p.parse_args().config)
