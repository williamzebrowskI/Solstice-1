"""Fresh SFT curriculum from cached short chat, checked constraints and train-only QA."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import re
import shutil
import uuid
import numpy as np
from sml_v1.common import atomic_json,file_sha256,fingerprint,read_json,lock
from sml_v1.tokenization import Tokenizer
from .grounded_corpus import FORMAT,clean,text_hash,split_for,replay_pool,verify,pair
from .balanced_corpus import previous,exclusions,knowledge_guard,requests,grams
from .balanced_tasks import panel,check
from .curriculum_rules import reference_ok,OUT_OF_SCOPE,constraints,ROLE_ASSIGNMENT,INVENTORY_CLAIM,TASK_START

TEMPLATE_WORDS=set('your response answer should must contain include be is not no all any of with at least most exactly more less fewer than sentences sentence words word bullet points point use markdown such as this number a an and or do lowercase uppercase letters capital english entire in without put each on separate line the to only follow format following'.split())
REVIEW_REJECTIONS={
    text_hash('You may provide your response without the unnecessarily placed comma at the start.'),
    text_hash('What tool is used to show how data points relate to each other. A scatter plot, or what is the other graph that does the same?')}


def generic_instruction_gram(gram):
    return all(word in TEMPLATE_WORDS or word.isdigit() for word in gram.split())


def overlaps(prompts,blocked,overlap):
    return any(text_hash(q) in blocked or grams(q)&overlap for q in prompts if isinstance(q,str) and q.strip())


def verified_rows():
    nouns=['cups','spoons','plates','books','bottles','tickets','windows','keys','lamps','chairs','flowers','baskets']
    colors=['blue','green','red','gold','silver','white','orange','purple']
    for i in range(3200):
        family=i%8;noun=nouns[(i//8)%len(nouns)];color=colors[(i//96)%len(colors)];code=f'R{i+701}'
        label=color+' '+noun+' '+code
        if family==0:
            q='Return only this label converted to lowercase: '+label.upper();a=label.lower();rules=[dict(kind='exact',answers=[a])]
        elif family==1:
            q='Return only the uppercase version of this label: '+label;a=label.upper();rules=[dict(kind='exact',answers=[a])]
        elif family==2:
            n=i%7+1;q=f'Inventory record {code}: item {noun}; quantity {n}. Output only a JSON object with keys item and quantity.'
            obj=dict(item=noun,quantity=n);a=json.dumps(obj);rules=[dict(kind='json',value=obj)]
        elif family==3:
            q=f'Record {code} lists two supplies: {color} {noun} and paper bags. Return those two items as exactly two bullet points.'
            a=f'- {color} {noun}\n- paper bags';rules=[dict(kind='exact',answers=[a])]
        elif family==4:
            sentence=f'The {color} {noun} is near shelf {code}.';q='Fix the agreement error. Give only the corrected sentence: '+sentence
            a=sentence.replace(' is ',' are ');rules=[dict(kind='exact',answers=[a])]
        elif family==5:
            if (i//8)%2==0:
                q=f'Stock card {code} says the {noun} are {color}. How many are there? If no quantity is given, reply Not provided.';a='Not provided.'
            else:q=f'Stock card {code} says the {noun} are {color}. What color are they? Reply with the color alone.';a=color
            rules=[dict(kind='exact',answers=[a])]
        elif family==6:
            q=f'Write one sentence saying package {code} is ready. Add the exact final sentence Collection is free.'
            a=f'Package {code} is ready. Collection is free.';rules=[dict(kind='exact',answers=[a])]
        else:
            q=f'For record {code}, correct the spelling without adding commentary: Pleese bring the {color} {noun}.'
            a=f'Please bring the {color} {noun}.';rules=[dict(kind='exact',answers=[a])]
        yield dict(component='verified',group='curriculum-skill:'+str(i),family='train-curriculum/'+str(family),messages=pair(q,a),
            verification=dict(checks=rules),provenance=dict(repo='local/short-curriculum-skills',revision='20261102-v1',license='CC0-1.0'))


def upstream(project):
    import pyarrow as pa
    base=project/'pretraining/data/.sft_hf_cache/HuggingFaceTB___smol-smoltalk/default/0.0.0/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'
    for path in sorted(base.glob('smol-smoltalk-train-*.arrow')):
        yield dict(_receipt=dict(path=str(path),sha256=file_sha256(path),repo='HuggingFaceTB/smol-smoltalk',revision=base.name,license='Apache-2.0'))
        with pa.memory_map(str(path),'r') as stream:
            for batch in pa.ipc.open_stream(stream):
                for raw in batch.to_pylist():
                    source=raw['source']
                    if source not in ('smol-magpie-ultra-short','smol-contraints','everyday-conversations'):continue
                    messages=raw['messages'] if source=='everyday-conversations' else raw['messages'][:2]
                    if len(messages)<2 or messages[0]['role']!='user' or messages[1]['role']!='assistant':continue
                    component={'smol-magpie-ultra-short':'chat','smol-contraints':'constraints','everyday-conversations':'dialogue'}[source]
                    verification=None
                    if component=='constraints':
                        rules=constraints(messages[0]['content'],messages[1]['content'])
                        if not rules:continue
                        verification=dict(checks=rules)
                    yield dict(component=component,group='upstream:'+text_hash(messages[0]['content']) if component!='dialogue' else
                        'dialogue:'+text_hash('\n'.join(m['content'] for m in messages if m['role']=='user' and m['content'].strip().lower() not in ('hi','hello','hey!','hi there'))),
                        messages=messages,verification=verification,provenance=dict(repo='HuggingFaceTB/smol-smoltalk',revision=base.name,
                            source=source,license='Apache-2.0',adaptation='Whole conversation' if component=='dialogue' else 'First complete user/assistant exchange; no answer truncation'))


def prepare(config):
    plan=read_json(config);dest=Path(plan['prepared']);project=Path(__file__).resolve().parents[2]
    if dest.exists():return verify(dest,plan)
    tokenizer=Tokenizer(plan['tokenizer']);old,groups,oldkeys=previous(plan)
    protected=groups['dev']|groups['test'];protectedkeys=oldkeys['dev']|oldkeys['test'];allold=set.union(*groups.values());allkeys=set.union(*oldkeys.values())
    blocked,overlap,benchmarks=exclusions(project);panels={s:panel(s) for s in ('dev','test')}
    template_grams=sum(generic_instruction_gram(g) for g in overlap)
    overlap={g for g in overlap if not generic_instruction_gram(g)}
    for bank in panels.values():
        for row in bank:
            for m in row['messages']:
                if m['role']=='user':blocked.add(text_hash(m['content']));overlap.update(grams(m['content']))
    stage=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True)
    pools=defaultdict(list);seen=set();rejects=Counter();receipts=[]
    try:
        knowledge_guard(project,stage,plan,blocked,overlap)
        def accept(row,force_train=False,forced_split=None):
            c=row['component'];split=forced_split or ('train' if force_train or c=='verified' else split_for(row['group'],plan['seed']))
            target=plan['quotas'][c] if split=='train' else plan.get('holdouts',{}).get(c,plan['holdout_per_component'])
            if len(pools[(c,split)])>=2*target:return
            try:
                if any(len(m['content'])>1200 for m in row['messages'] if m['role']=='assistant'):raise ValueError('overlong_reference_text')
                if any(ROLE_ASSIGNMENT.search(m['content']) for m in row['messages'] if m['role']=='user'):raise ValueError('role_assignment')
                if any(INVENTORY_CLAIM.search(m['content']) for m in row['messages'] if m['role']=='assistant'):raise ValueError('unsupported_personal_or_inventory_claim')
                if any(text_hash(m['content']) in REVIEW_REJECTIONS for m in row['messages'] if m['role']=='user'):raise ValueError('reviewed_bad_reference')
                if any(re.search(r'[*#_=\-]{4,}',m['content']) for m in row['messages']):raise ValueError('formatting_artifact')
                if c=='chat' and '?' not in row['messages'][0]['content'] and not TASK_START.search(row['messages'][0]['content'].strip()):raise ValueError('unclear_first_request')
                if OUT_OF_SCOPE.search('\n'.join(m['content'] for m in row['messages'])):raise ValueError('out_of_scope')
                if any(not reference_ok(m['content']) for m in row['messages'] if m['role']=='assistant'):raise ValueError('reference_quality')
                messages,ids,mask=clean(row,tokenizer,plan);keys=requests(messages)
                if not keys:raise ValueError('social_only')
                if split=='train' and (row['group'] in protected or keys&protectedkeys):raise ValueError('part1_holdout_overlap')
                if split!='train' and c!='grounded' and (row['group'] in allold or keys&allkeys):raise ValueError('part1_seen_in_new_holdout')
                if keys&seen:raise ValueError('duplicate_conditioned_request')
                prompts=[m['content'] for m in messages if m['role']=='user']+[row.get('question','')]
                if overlaps(prompts,blocked,overlap):raise ValueError('benchmark_or_panel_overlap')
                if row.get('verification') and not check(messages[-1]['content'],row['verification']['checks']):raise ValueError('invalid_reference_constraint')
                if c=='dialogue' and sum(m['role']=='user' for m in messages)<2:raise ValueError('not_multiturn')
                seen.update(keys);pools[(c,split)].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask))
            except ValueError as exc:rejects[str(exc)]+=1
        for row in upstream(project):
            if '_receipt' in row:receipts.append(row['_receipt']);print('[prepare-curriculum] '+Path(row['_receipt']['path']).name,flush=True)
            else:accept(row)
        for row in verified_rows():accept(row)
        for split in ('train','dev','test'):
            for row in old[split]:
                if row['provenance']['repo']!='rajpurkar/squad_v2' or row['component']!='grounded':continue
                row=dict(row,component='grounded',verification=dict(checks=[dict(kind='answer',answers=row['verification']['answers'])]))
                accept(row,force_train=split=='train',forced_split=split)
        print('[available-curriculum] '+json.dumps({f'{c}/{s}':len(rr) for (c,s),rr in pools.items()}),flush=True)
        print('[rejected-curriculum] '+json.dumps(dict(rejects)),flush=True)
        selected={s:[] for s in ('train','dev','test')}
        for split in selected:
            for c,quota in plan['quotas'].items():
                if c=='verified' and split!='train':continue
                target=quota if split=='train' else plan.get('holdouts',{}).get(c,plan['holdout_per_component']);rr=pools[(c,split)]
                # Grounded evaluation reuses Part 1's protected article holdouts;
                # new upstream instruction holdouts exclude all Part 1 inputs.
                if len(rr)<target:raise ValueError(f'Insufficient {c}/{split}: {len(rr)} < {target}')
                random.Random(str(plan['seed'])+c+split).shuffle(rr);selected[split].extend(rr[:target])
            if split!='train':
                for row in panels[split]:
                    mm,ids,mask=clean(row,tokenizer,plan);selected[split].append(dict(row,messages=mm,split=split,_ids=ids,_mask=mask))
        reviewed=plan.get('post_selection_exclusions',{})
        selected['train']=[row for row in selected['train'] if not any(text_hash(m['content']) in reviewed for m in row['messages'] if m['role']=='user')]
        counts={};splitkeys={};splitgroups={};lengths=[]
        for split,rr in selected.items():
            random.Random(plan['seed']).shuffle(rr);ids=[];masks=[];offsets=[0];splitkeys[split]=set();splitgroups[split]=set()
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in rr:
                    a,b=row.pop('_ids'),row.pop('_mask');ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
                    splitkeys[split].update(requests(row['messages']));splitgroups[split].add(row['group'])
                    if split=='train':lengths.extend(len(tokenizer.encode(m['content'])) for m in row['messages'] if m['role']=='assistant')
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
            np.save(stage/(split+'.tokens.npy'),np.concatenate(ids),allow_pickle=False);np.save(stage/(split+'.mask.npy'),np.concatenate(masks),allow_pickle=False)
            np.save(stage/(split+'.offsets.npy'),np.array(offsets,np.uint64),allow_pickle=False)
            counts[split]=dict(conversations=len(rr),input_tokens=sum(len(a)-1 for a in ids),assistant_tokens=sum(int(b[1:].sum()) for b in masks),components=dict(Counter(row['component'] for row in rr)))
        for a,b in (('train','dev'),('train','test'),('dev','test')):
            if splitkeys[a]&splitkeys[b] or splitgroups[a]&splitgroups[b]:raise ValueError('Curriculum split leakage')
        for split in ('dev','test'):
            bank=list(panels[split])
            for c,n in plan['behavior_examples'].items():
                for i,row in enumerate([r for r in selected[split] if r['component']==c][:n]):
                    bank.append(dict(id=split+'/'+c+'/'+str(i),family=split+'/'+c+'/'+str(i),component=c,messages=row['messages'],
                        limit=256 if c in ('constraints','grounded') else 512,verification=row.get('verification')))
            atomic_json(stage/('behavior_'+split+'.json'),bank)
        replay_pool(project,stage,tokenizer,plan)
        atomic_json(stage/'audit.json',dict(cross_split_groups=0,cross_split_conditioned_requests=0,part1_holdouts_in_train=0,
            source_counts=dict(Counter(row['provenance']['repo'] for row in selected['train'])),
            upstream_sources=dict(Counter(row['provenance'].get('source','retained_or_authored') for row in selected['train'])),
            answer_tokens=dict(median=float(np.median(lengths)),p90=float(np.quantile(lengths,.9)),max=max(lengths)),
            truncation=False,source_split='Only upstream train files; public benchmarks excluded from gradients',
            generic_instruction_ngrams_exempted=template_grams,
            post_selection_review_exclusions=reviewed,
            grounded_evaluation='Reuses protected Part 1 dev/test article groups; never included in new training'))
        manifest=dict(format=FORMAT,strategy='fresh-short-curriculum-v4',plan=fingerprint(plan),tokenizer=tokenizer.fingerprint,counts=counts,
            rejected=dict(rejects),sources=receipts,benchmark_exclusions=benchmarks,
            builder_sources={name:file_sha256(Path(__file__).with_name(name)) for name in ('curriculum_corpus.py','curriculum_rules.py','balanced_tasks.py')},
            limitations=['Synthetic references can contain factual or stylistic errors; filtering and sampled review are not exhaustive semantic certification.',
                'Only recognized simple constraints are automatically checked; natural chat answers receive structural filters and sampled review.',
                'Exact/n-gram exclusions do not prove complete semantic or pretraining benchmark decontamination.',
                'Generic formatting-instruction n-grams are permitted; full benchmark prompts and non-template overlaps are excluded.',
                'Private behavior panels cover measured abilities; unseen prompts can still regress.'])
        manifest['files']={p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in stage.iterdir() if p.is_file()}
        atomic_json(stage/'manifest.json',manifest);verify(stage,plan);stage.rename(dest);return manifest
    finally:
        if stage.exists():shutil.rmtree(stage)
