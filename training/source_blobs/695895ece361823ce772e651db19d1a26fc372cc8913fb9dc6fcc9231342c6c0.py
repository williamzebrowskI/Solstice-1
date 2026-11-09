"""SFT-037: source-span QA and typed contracts, not a fact-oracle claim.

Natural QA is licensed, annotated SQuAD TRAIN data. Synthetic state/format/code
records are explicitly labeled and grouped by source article or task family.
No downloaded Python is executed; bounded interpretation checks literal tests.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from sml_v1.common import atomic_json, file_sha256, read_json
from sml_v1.tokenization import Tokenizer
from .curate import local, blocked_questions
from .quality import digest, normalized, repeated
from .reliable_chat import cached, encode
from .conversation import split_group
from .behavior_v8 import python_checks, typed_equal

SOURCE = dict(component='qa',repo='rajpurkar/squad_v2',config='squad_v2',
              revision='3ffb306f725f7d2ce8394bc1873b24868140c412',split='train',license='CC-BY-SA-4.0')
FOLDER='reports/sft/SFT-037-verified-extension-20261008'
QUOTAS=dict(qa=6000,unanswerable=500,dialogue=1500,contrast=1200,python=600,social=200)
# These fact families and diagnostic objects are reserved, not patch targets.
RESERVED_TOPICS=re.compile(r'\b(?:jupiter|largest planet|freez\w*|insect\w*|photosynth\w*|oxygen|vacuum|rainbow|ocean tides|yeast|north poles|seed|pyramids|largest continent|cassowary|lantern|bamboo|quartz|ribbon|baxter|zora|malik|nessa|davin|elora|take_first|smallest_or_none|count_equal|sign_label)\b',re.I)


def pair(u,a): return [dict(role='user',content=u),dict(role='assistant',content=a)]


def check_target(row,text):
    """Strict specified-output checks; acceptable paraphrases need human review."""
    v=row['verification']; kind=v['kind']; text=text.strip()
    if kind=='span':
        # Preserve meaningful signs/decimals; a generic word normalizer would
        # incorrectly certify -5 as 5. Allow only case, spacing and final prose
        # punctuation differences on an explicitly requested short answer.
        def span_key(value):
            return ' '.join(value.strip().strip('"').rstrip('.!?').casefold().split())
        ok=span_key(text) in {span_key(x) for x in v['answers']}
    elif kind=='exact': ok=text==v['expected']
    elif kind=='json':
        try: ok=typed_equal(json.loads(text),v['expected'])
        except (ValueError,TypeError): ok=False
    elif kind=='python': ok=all(python_checks(text,row['functional_assertions'],**row['python_contract']).values())
    elif kind=='request':
        ok=bool(re.search(r'\b(?:paste|send|share|provide)\b',text,re.I) and re.search(v['object_pattern'],text,re.I)) and len(text.split())<=30
    elif kind=='social':
        ok=bool(re.search(v['pattern'],text,re.I)) and len(text.split())<=40 and not re.search(r'\b(?:my name|my family|english tutor|you can help me|tornadoes|earthquake)\b',text,re.I)
    else: raise ValueError('Unknown verification kind')
    return bool(ok and text and not repeated(text))


def source_span(raw):
    answers=raw['answers']; texts=answers['text']; starts=answers['answer_start']
    if not texts or len(texts)!=len(starts): raise ValueError('missing_span')
    if any(not t.strip() or raw['context'][p:p+len(t)]!=t for t,p in zip(texts,starts)):
        raise ValueError('span_offset_mismatch')
    if any(len(t.split())>10 for t in texts): raise ValueError('long_answer')
    return texts[0].strip()


def python_rows():
    """Twenty independently specified task families; parameters change behavior.

    Expected values use separate reference formulas/loops rather than executing
    the target expression. Each family is kept in a single split.
    """
    for family in range(20):
      for k in range(48):
        n=k+2; b=k%7+1; fn=f'checked_{family}_{n}'
        if family==0:
            args='value';expr=f'value * {n} + {b}';desc=f'multiply a number by {n}, then add {b}'
            xs=[[x] for x in (0,-2,3,1.5)]; ys=[sum([x[0]]*n)+b for x in xs]
        elif family==1:
            args='values';expr=f'[x for x in values if x > {n}]';desc=f'keep only list numbers strictly greater than {n}, preserving order'
            xs=[[[]],[[n,n+1,n-1,n+2]],[[n+1,n+1]],[[-1,0]]];ys=[[],[n+1,n+2],[n+1,n+1],[]]
        elif family==2:
            args='value';expr=f'max({-n}, min({n}, value))';desc=f'clamp a number to the inclusive interval from {-n} to {n}'
            xs=[[0],[-n-1],[n+1],[n]];ys=[0,-n,n,n]
        elif family==3:
            args='text';expr=f'text.replace("{n}", "[{n}]")';desc=f'replace every occurrence of the literal string "{n}" with "[{n}]"'
            xs=[[''],[str(n)],['q'+str(n)+'q'],[str(n)+' '+str(n)]];ys=['',f'[{n}]',f'q[{n}]q',f'[{n}] [{n}]']
        elif family==4:
            args='items';expr=f'" /{n}/ ".join(items)';desc=f'join a list of strings using the exact separator " /{n}/ "'
            xs=[[[]],[['a']],[['a','b']],[['','z']]];ys=['','a',f'a /{n}/ b',f' /{n}/ z']
        elif family==5:
            args='number';expr=f'number % {n} == 0';desc=f'return a Boolean indicating whether an integer is divisible by {n}'
            xs=[[0],[n],[n+1],[-n]];ys=[True,True,False,True]
        elif family==6:
            args='values';expr=f'[x + {n} for x in values]';desc=f'add {n} to each number in a list, preserving duplicates and order'
            xs=[[[]],[[0,-1]],[[n,n]],[[2]]];ys=[[],[n,n-1],[2*n,2*n],[n+2]]
        elif family==7:
            args='values';expr=f'sum(x for x in values if x < {-n})';desc=f'sum only list numbers strictly less than {-n}; return zero if none qualify'
            xs=[[[]],[[-n,-n-1,0]],[[-n-2,-n-2]],[[n]]];ys=[0,-n-1,-2*n-4,0]
        elif family==8:
            args='value';expr=f'value / {n}';desc=f'divide a number by {n} without integer rounding'
            xs=[[0],[n],[-n],[n/2]];ys=[0,1,-1,.5]
        elif family==9:
            args='key';expr=f'{{"alpha": {n}, "beta": {n+1}}}.get(key, -1)';desc=f'look up a string key in the mapping alpha={n}, beta={n+1}, returning -1 for an absent key'
            xs=[['alpha'],['beta'],[''],['gamma']];ys=[n,n+1,-1,-1]
        elif family==10:
            args='minutes';expr=f'(minutes + {n}) % 1440';desc=f'add {n} to an integer minute value and wrap the result into 0 through 1439'
            xs=[[0],[1440-n],[-n-1],[1440]];ys=[n,0,1439,n]
        elif family==11:
            args='values';expr=f'[x for x in values if x != {n}]';desc=f'remove every occurrence of the number {n} from a list while preserving order'
            xs=[[[]],[[n,-1,n,n+1]],[[-1,n+1]],[[n,n]]];ys=[[],[-1,n+1],[-1,n+1],[]]
        elif family==12:
            args='text';expr=f'text[{b}:{b+n}]';desc=f'return the substring starting at zero-based position {b}, taking up to {n} characters; use normal Python slicing'
            s='abcdefghijklmnopqrstuvwxyz'*3;xs=[[''],['a'],[s],[s[:b+1]]];ys=['','', ''.join(s[i] for i in range(b,min(b+n,len(s)))),s[b]]
        elif family==13:
            args='values';expr=f'len([x for x in values if x >= {n}])';desc=f'count list numbers greater than or equal to {n}'
            xs=[[[]],[[n,n-1,n+1]],[[n,n]],[[0]]];ys=[0,2,2,0]
        elif family==14:
            args='text';expr=f'text + "!" * {b}';desc=f'append exactly {b} exclamation marks to a string, including an empty string'
            xs=[[''],['hello'],['!'],[' a ']];ys=[x[0]+''.join('!' for _ in range(b)) for x in xs]
            # Only seven distinct specs: do not rename the function to pad quota.
            if k>=7: continue
        elif family==15:
            args='values';expr=f'[x * {n} for x in values if x != 0]';desc=f'drop zero entries and multiply each remaining list number by {n}'
            xs=[[[]],[[0,1,-2]],[[0,0]],[[2,2]]];ys=[[],[n,-2*n],[],[2*n,2*n]]
        elif family==16:
            args='number';expr=f'number // {n}';desc=f'floor-divide an integer by {n}, including negative inputs'
            xs=[[0],[n+1],[-1],[-n-1]];ys=[0,1,-1,-2]
        elif family==17:
            args='text';expr=f'text.startswith("tag{n}:")';desc=f'return whether a string starts with the exact prefix "tag{n}:"'
            xs=[[''],[f'tag{n}:'],[f'tag{n}:x'],[f'xtag{n}:']];ys=[False,True,True,False]
        elif family==18:
            args='values';expr=f'[x for x in values if {-n} <= x <= {n}]';desc=f'keep list numbers inside the inclusive interval {-n} through {n}, in their original order'
            xs=[[[]],[[-n-1,-n,0,n,n+1]],[[n,n]],[[n+1]]];ys=[[],[-n,0,n],[n,n],[]]
        else:
            args='values';expr=f'sum(values) + {n}';desc=f'sum a list of numbers and add {n}; an empty list returns {n}'
            xs=[[[]],[[0]],[[1,-1]],[[2,3]]];ys=[n,n,n,n+5]
        tests=[dict(args=x,expected=y) for x,y in zip(xs,ys)]
        yield dict(group=f'037:python-family:{family}',bucket='python',component='python',
            messages=pair(f'Write {fn}({args}) in Python to {desc}. Return one Python code block only.',f'```python\ndef {fn}({args}):\n    return {expr}\n```'),
            verification=dict(kind='python'),functional_assertions=tests,
            python_contract=dict(function=fn,code_only=True,docstring=False),
            provenance=dict(repo='local/verified-extension',revision='v1',license='CC0-1.0',split='authored',family=family,parameter=n,method='Authored contract; four separately specified ordinary and boundary tests'))


def build(config):
    plan=read_json(config); folder=local(FOLDER); folder.mkdir(parents=True,exist_ok=True)
    if (folder/'catalog.jsonl').exists(): raise ValueError('Preserve existing frozen catalog')
    tok=Tokenizer(local(plan['tokenizer'])); blocked,bench=blocked_questions(local('pretraining/data/.sft_hf_cache'))
    from .behavior import cases
    for suite in ('instruction-v6','instruction-v7','instruction-v8','conversation-v5'):
      for split in ('dev','test'):
        blocked.update(digest(m['content']) for c in cases(split,suite) for m in c['messages'] if m['role']=='user')
    protocols=['reports/sft/SFT-020-balanced-instructions-20261008/fresh-diagnostics/protocol.json','reports/sft/chat-search-round2-20261008/new-panel.json','reports/sft/sft035-continuation-20261008/new-panel.json','reports/sft/sft036-extension-diagnostics-20261008/protocol.json']
    for path in protocols:
      blocked.update(digest(p) for c in read_json(local(path))['cases'] for p in c['prompts'])
    # Eight-word benchmark sequences, across all ARC splits and IFEval, catch
    # question text embedded inside transformed prompts rather than exact only.
    from datasets import Dataset
    grams=set()
    for repo,field in [('allenai___ai2_arc','question'),('google___ifeval','prompt')]:
      for path in local('pretraining/data/.sft_hf_cache/'+repo).rglob('*.arrow'):
        for r in Dataset.from_file(str(path)):
          words=normalized(r[field]).split(); grams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    # The pinned IFEval runner also keeps an explicit input snapshot. Use it
    # regardless of the datasets library's cache-directory naming convention.
    ifeval_snapshot=local('reports/benchmarks/ifeval-SFT-036-step96-20261008/inputs.json')
    for r in read_json(ifeval_snapshot):
        words=normalized(r['prompt']).split()
        grams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    dataset,source_receipt=cached(SOURCE); rows=[]; seen=set(); rejected=Counter(); candidates=defaultdict(list)
    raw_by_context=defaultdict(list)
    def accept(r):
      try:
        users=[m['content'] for m in r['messages'] if m['role']=='user']
        if any(digest(p) in blocked for p in users): raise ValueError('reserved_prompt')
        for p in users:
          words=normalized(p).split()
          if any(' '.join(words[i:i+8]) in grams for i in range(len(words)-7)): raise ValueError('benchmark_ngram')
        if not check_target(r,r['messages'][-1]['content']): raise ValueError('target_contract')
        # Check complete serialization/round trip now; never clip a long answer
        # or its evidence during later preparation.
        encode(r['messages'],tok,plan,r['component'],r['group'])
        key=hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if key in seen: raise ValueError('duplicate_history')
        r['split']='train' if r['provenance'].get('replay') else split_group(r['group'],263700,12)
        seen.add(key);candidates[(r['split'],r['bucket'])].append(r)
      except ValueError as e: rejected[str(e)]+=1
    def row(bucket,component,group,messages,verification,provenance,**kw):
      return dict(bucket=bucket,component=component,group=group,messages=messages,verification=verification,provenance=provenance,**kw)
    order=sorted(range(len(dataset)),key=lambda i:digest(dataset[i]['id']))
    for idx in order:
      raw=dataset[idx]; q=raw['question'].strip(); ctx=raw['context']; group='037:article:'+digest(raw['title'])
      if raw['id'] in plan.get('source_excluded_ids',[]):
        rejected['manual_ambiguous_annotation']+=1;continue
      if '[citation needed]' in ctx:
        rejected['source_citation_needed']+=1;continue
      if digest(q) in blocked:
        rejected['raw_reserved_question']+=1;continue
      if not 6<=len(q.split())<=40 or not 60<=len(ctx.split())<=220 or RESERVED_TOPICS.search(ctx+' '+q):continue
      if any(c in ctx+q for c in ('\u0085','\u2028','\u2029')): continue
      provenance=dict(SOURCE,row=idx,id=raw['id'],title=raw['title'],context_sha256=hashlib.sha256(ctx.encode()).hexdigest(),answer_offsets=raw['answers']['answer_start'],method='Whole source paragraph; annotated answer offsets checked; no fabricated summary')
      if raw['answers']['text']:
        try:a=source_span(raw)
        except ValueError as e: rejected[str(e)]+=1;continue
        r=row('qa','qa',group,pair(f'Use only the passage below. Answer the question with the shortest answer span from the passage, without explanation.\nPassage: {ctx}\nQuestion: {q}',a),dict(kind='span',answers=raw['answers']['text']),provenance,answers=raw['answers']['text'],answerable=True)
        accept(r); raw_by_context[digest(ctx)].append((raw,provenance,group))
      else:
        # Adversarial unanswerable annotations proved ambiguous in spot review.
        # Do not turn those disputed labels into supposedly verified targets.
        rejected['adversarial_unanswerable_not_used']+=1
      # Balanced source-topic sampling: avoid one prolific title dominating.
    # Contrasts use genuine supplied prose and the same question, with distinct
    # verified output contracts. Article group ties all transformations together.
    for records in raw_by_context.values():
      raw,prov,group=records[0]; ctx=raw['context'];a=source_span(raw);q=raw['question']
      base=f'Passage: {ctx}\nQuestion: {q}'
      specifications=[
        (f'{base}\nReturn only JSON with key "answer" holding the shortest answer span.',json.dumps({'answer':a},ensure_ascii=False),dict(kind='json',expected={'answer':a})),
        (f'{base}\nGive the shortest answer span inside double quotation marks, and nothing else.',json.dumps(a,ensure_ascii=False),dict(kind='exact',expected=json.dumps(a,ensure_ascii=False))),
        (f'Please answer this question using my passage: {q}\nI have not supplied the passage yet.','Please paste the passage so I can answer using it.',dict(kind='request',object_pattern=r'\bpassage\b'))]
      for u,t,v in specifications:
        accept(row('contrast','constraints',group,pair(u,t),v,dict(prov,method='Contrasting supplied/missing evidence and JSON/quoted output contracts'),rules=[dict(kind='exact',args=[t])]))
    # Actual correction/state contracts, not copied benchmark facts. Three
    # different operations per record. Split the whole record as one group.
    names=['Ari','Bea','Cal','Dee','Eli','Fia','Gus','Hana','Ike','Jae','Kira','Luca','Mina','Nico','Ola','Pia','Remy','Sia','Tao','Uma']
    objects=['atlas','brush','chalk','drum','envelope','fork','globe','hat','ink','jug','key','ladder','mat','notebook','oar','plate','rug','stamp','tray','umbrella']
    for i in range(750):
      rng=random.Random(263900+i);name=rng.choice(names);item=rng.choice(objects);identifier=f'ENTRY-{i+10000}'
      record=dict(entry=identifier,person=name,item=item,quantity=i%9+1)
      u=f'Use only this record: {json.dumps(record)}\nWhat is the age of {name}? Return the age if the record contains it; otherwise reply exactly "Not recorded."'
      target='Not recorded.'
      accept(row('unanswerable','qa',f'037:missing-record:{i}',pair(u,target),dict(kind='exact',expected=target),dict(repo='local/verified-extension',revision='v1',license='CC0-1.0',split='authored',record=record,requested_field='age',method='Typed record lacks requested field; cannot infer age from entry ID or quantity; correlated templates'),answers=[target],answerable=False))
    for i in range(750):
      rng=random.Random(263700+i); a,b=rng.sample(names,2);w,x,y,z=rng.sample(objects,4);first=100+i;new=first+13;group=f'037:state:{i}'
      prov=dict(repo='local/verified-extension',revision='v1',license='CC0-1.0',split='authored',record=dict(person=a,other=b,first=first,new=new,items=[w,x,y],remove=x,append=z),method='Typed old/new state, entity preservation and ordered-list oracle; correlated template variants')
      history=pair(f'{a} has box {first}. {b} has box {first+2}.',f'{a}: {first}; {b}: {first+2}.')+pair(f'Update only {a}: the box is now {new}.',f'{a} now has box {new}; {b} still has box {first+2}.')
      for j in range(2):
        u=f'Return only the current box number for {b}.' if j else 'Return only JSON with keys "original" and "current" for '+a+'.'
        target=str(first+2) if j else json.dumps({'original':first,'current':new});v=dict(kind='exact',expected=target) if j else dict(kind='json',expected={'original':first,'current':new})
        accept(row('dialogue','chat',group,history+pair(u,target),v,prov))
      history=pair(f'My ordered list is {w}, {x}, {y}.',f'The list is {w}, {x}, {y}.')+pair(f'Delete {x} and append {z}.',f'The updated list is {w}, {y}, {z}.')+pair('Give the updated items separated by comma-space only.',f'{w}, {y}, {z}')
      accept(row('dialogue','chat',group,history,dict(kind='exact',expected=f'{w}, {y}, {z}'),prov))
    for r in python_rows(): accept(r)
    # Reuse only reviewed, nonfactual social TRAIN anchors, explicitly as replay.
    old=local('pretraining/data/sft_036_sft035_checked_continuation_20261008/train.jsonl')
    for line in old.read_text().splitlines():
      r=json.loads(line)
      if r['bucket']!='social':continue
      msgs=[m for m in r['messages'] if m['role']!='system']
      # Retain original train group; never redistribute parent's training into
      # our holdouts, which would inflate independent evaluation.
      for j in range(1,len(msgs),2):
        if 'topic is' in msgs[0]['content'].lower():continue
        sub=msgs[:j+1];target=sub[-1]['content']
        v=dict(kind='exact',expected=target)
        rr=row('social','chat',r['group'],sub,v,dict(r['provenance'],method='Reviewed social TRAIN replay, not new facts',replay=str(old.relative_to(local('.')))))
        accept(rr)
    # Social template data has few base intents. Expand only actual contextual
    # assistance: varied concrete text operations after a natural opener.
    for i in range(340):
      w=objects[i%len(objects)];n=i+3;g=f'037:social:{i}'
      u=['Could we work on a small task?','I have a quick question.','Are you available to help?','Hello, can I ask something?'][i%4]
      history=pair(u,'Of course. What would you like help with?')+pair(f'I have {n} {w} items and use two. How many are left? Reply with the number.',str(n-2))+pair('That answers it, thank you.','You are welcome.')
      accept(row('social','chat',g,history,dict(kind='social',pattern=r'\b(?:welcome|glad|happy)\b'),dict(repo='local/verified-extension',revision='v1',license='CC0-1.0',split='authored',method='Reviewed social intent and computed arithmetic, correlated templates')))
    # Pick exactly the advertised train quotas; no duplicated padding. Cap a
    # source article at 45 QA records and expose group counts honestly.
    for split in ('train','dev','test'):
      for bucket,quota in QUOTAS.items():
        pool=candidates[(split,bucket)];pool.sort(key=lambda r:digest(json.dumps(r['messages'],sort_keys=True)))
        used=Counter();selected=[];cap=quota if split=='train' else 30
        for r in pool:
          if bucket=='qa' and used[r['group']]>=45:continue
          if bucket=='social' and r['provenance'].get('replay') and split!='train':continue
          selected.append(r);used[r['group']]+=1
          if len(selected)==cap:break
        if len(selected)!=cap:raise ValueError(f'Insufficient {split}/{bucket}: {len(selected)} of {cap}; pool={len(pool)}')
        rows.extend(selected)
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','dev','test')}
    if any(groups[a]&groups[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):raise ValueError('Group leakage')
    path=folder/'catalog.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    receipt=dict(format='opensml-verified-extension-catalog-v1',sha256=file_sha256(path),source_receipts=[dict(source=SOURCE,**source_receipt)],benchmark_exclusions=bench,
      frozen_probe_receipts={p:file_sha256(local(p)) for p in protocols},counts={s:dict(conversations=sum(r['split']==s for r in rows),buckets=dict(Counter(r['bucket'] for r in rows if r['split']==s)),groups=len(groups[s])) for s in groups},
      ifeval_snapshot_sha256=file_sha256(ifeval_snapshot),
      rejected=dict(rejected),checks='Every complete history serializes without clipping; every final target passes its typed contract; all answerable source offsets verified; every Python target passes four ordinary/edge assertions; grouped splits and exact/8-gram ARC/IFEval exclusion.',
      limitations=['Span agreement verifies supplied evidence alignment, not universal historical truth.','Annotated source questions can still contain human annotation defects.','Synthetic state/code/social variants are correlated task families, not 10000 independent facts.','QA is mainly passage-grounded; this does not promise equivalent gains in ungrounded factual recall.','Exact/8-gram screening cannot prove absence of semantic benchmark contamination or prior pretraining exposure.'])
    atomic_json(folder/'catalog.json',receipt)
    # Freeze TEST protocol before any training. It is never used to rank checkpoints.
    panel=[]
    for bucket in QUOTAS:
      for r in [x for x in rows if x['split']=='test' and x['bucket']==bucket][:12]:
        panel.append(dict(r,name=bucket+'-'+str(len(panel))))
    atomic_json(folder/'acceptance-panel.json',dict(format='opensml-extension-acceptance-v1',cases=panel,scope='72 source/family-heldout histories; evaluate one dev-selected candidate once; exact requested outputs, not an exhaustive conversational judge.',policy=dict(min_macro_gain=.05,min_qa_gain=.05,max_category_regression=.05,retention_ceiling=2.6814481621980665,require_manual_review=True)))
    plan['catalog_sha256']=receipt['sha256'];atomic_json(config,plan)
    print(json.dumps(receipt['counts'],indent=2),flush=True)


def generated_panel(model,tokenizer,rows,*,system=None,limit=192):
    """Use the model's own previous replies, never gold history continuations."""
    from .evaluate import generate
    output=[]
    for row in rows:
        history=[dict(role='system',content=system)] if system else []
        turns=[]
        for message in row['messages']:
            if message['role']!='user':continue
            history.append(dict(message));response=generate(model,tokenizer,history,limit=limit,temperature=0)
            turns.append(dict(request=[dict(m) for m in history],result=response))
            if not response['text'].strip():break
            history.append(dict(role='assistant',content=response['text']))
        ended=response['stop_reason'] in ('turn_end','document_end')
        output.append(dict(name=row.get('name',row['group']),bucket=row['bucket'],
            verification=row['verification'],turns=turns,passed=check_target(row,response['text']) and ended,
            natural_stop=ended))
    groups=defaultdict(list)
    for row in output:groups[row['bucket']].append(row['passed'])
    rates={key:sum(values)/len(values) for key,values in groups.items()}
    return dict(rates=rates,macro_pass=sum(rates.values())/len(rates),samples=output)


def development(model,tokenizer,prepared,plan):
    rows=[json.loads(x) for x in (Path(prepared)/'dev.jsonl').read_text().splitlines()]
    selected=[]
    for bucket in QUOTAS:
        selected.extend([r for r in rows if r['bucket']==bucket][:plan.get('verified_dev_per_bucket',8)])
    # Development is allowed for selection. Both presentations are equally
    # weighted; the separate TEST acceptance panel is never consulted here.
    panels=[generated_panel(model,tokenizer,selected,system=system,limit=plan.get('verified_generation_limit',192))
            for system in (None,plan['conversation_system'])]
    rates={bucket:sum(p['rates'][bucket] for p in panels)/len(panels) for bucket in QUOTAS}
    return dict(rates=rates,macro_pass=sum(rates.values())/len(rates),samples=[x for p in panels for x in p['samples']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();build(a.config)
    import os,sys
    sys.stdout.flush();os._exit(0)
