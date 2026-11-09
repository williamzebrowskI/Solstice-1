"""Local, primarily human-reference SFT with separate behavioral/knowledge guards."""
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
from .grounded_corpus import FORMAT,clean,text_hash,split_for,replay_pool,verify,benchmark_hashes
from .quality import social_prompt
from .quality_gate import looping,unchanged_edit_claim
from .balanced_tasks import training_rows,panel,check


# Exact exclusions found in the documented human review sample.
REVIEW_REJECTIONS={'1aa2078e8b63f2871ecb61c79782cee67c2d57d0f62b8498a2a7aa8799704fd7': 'Multiple spelling errors in the reference explanation.', '206b8bb18868e5b86785ee5f4c71009776bec15c1c667388183a44b8bf11acfe': 'Reference does not answer the requested creation date.', 'db398c8e579b54410082af87cfdc7184d8c34f148733fe80061f0140ee28e63b': 'Reference turns a quoted criticism into an unsupported endorsement.', '13a3660c19e77e1f2d6693ef63e593c48ad768a4f6e92758a06c19990be65d52': 'Reference has malformed and misleading mathematical wording.'}

def grams(text):
    tokens=re.findall(r'\w+',text.casefold())
    return {' '.join(tokens[i:i+13]) for i in range(len(tokens)-12)}


def requests(messages):
    prefix=[];out=set()
    for m in messages:
        prefix.append(m['role']+'\n'+m['content'])
        if m['role']=='user' and not social_prompt(m['content']):out.add(text_hash('\n'.join(prefix)))
    return out


def complete(text):
    value=text.rstrip()
    if value.endswith((':',',',';','-')):return False
    try:
        if isinstance(json.loads(value),(dict,list)):return True
    except (ValueError,TypeError):pass
    if re.search(r'\n(?:best(?: regards| wishes)?|kind regards|warm regards|sincerely|thanks|cheers|regards)[,!]?\s*\n[^\n]{1,60}$',value,re.I):return True
    tail=re.sub(r'["\u201d\u2019\x27\]\)*_]+$','',value).rstrip()
    return tail.endswith(('.', '!', '?')) or len(tail.split())<=7


def previous(plan):
    rows={s:[json.loads(x) for x in (Path(plan['previous_data'])/(s+'.jsonl')).read_text().splitlines()] for s in ('train','dev','test')}
    groups={s:{r['group'] for r in rr} for s,rr in rows.items()}
    keys={s:set().union(*(requests(r['messages']) for r in rr)) for s,rr in rows.items()}
    return rows,groups,keys


def exclusions(project):
    blocked,receipts=benchmark_hashes(project);overlap=set()
    from datasets import load_from_disk
    frozen=project/'pretraining/data/benchmark_inputs_20261102';manifest=read_json(frozen/'manifest.json')
    for name,expected in manifest['files'].items():
        if file_sha256(frozen/name)!=expected:raise ValueError('Frozen benchmark inputs changed: '+name)
    for name in manifest['tasks']:
        data=load_from_disk(str(frozen/name))
        for raw in data:
            for field in ('question','goal','ctx','ctx_a'):
                text=raw.get(field)
                if isinstance(text,str):blocked.add(text_hash(text));overlap.update(grams(text))
        receipts.append(dict(task=name,rows=len(data),revision=manifest['tasks'][name]['revision']))
    for raw in read_json(frozen/'ifeval_inputs.json'):
        blocked.add(text_hash(raw['prompt']));overlap.update(grams(raw['prompt']))
    return blocked,overlap,receipts


def source_rows(project):
    import pyarrow as pa
    import pyarrow.parquet as pq
    base=project/'pretraining/data/.sft_hf_cache'
    path=base/'no_robots/e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/train-00000-of-00001.parquet'
    yield dict(_receipt=dict(path=str(path),sha256=file_sha256(path),license='CC-BY-NC-4.0'))
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for raw in batch.to_pylist():
            if raw['category']=='Coding':continue
            component=('grounded' if raw['category'] in ('Closed QA','Classify','Extract') else
                       'writing' if raw['category'] in ('Summarize','Rewrite') else
                       'dialogue' if raw['category']=='Chat' and len(raw['messages'])>=4 else 'general')
            messages=[dict(m) for m in raw['messages']]
            if not messages or messages[0]['role']!='user':continue
            yield dict(component=component,group='human:'+text_hash(messages[0]['content']),messages=messages,
                provenance=dict(repo='HuggingFaceH4/no_robots',revision='e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b',
                    license='CC-BY-NC-4.0',category=raw['category'],id=raw['prompt_id'],reference='Human-written reference'))
    path=base/'databricks___databricks-dolly-15k/default/0.0.0/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k-train.arrow'
    yield dict(_receipt=dict(path=str(path),sha256=file_sha256(path),license='CC-BY-SA-3.0'))
    with pa.memory_map(str(path),'r') as stream:
        for batch in pa.ipc.open_stream(stream):
            for raw in batch.to_pylist():
                component=('writing' if raw['category'] in ('summarization','creative_writing') else
                           'grounded' if raw['context'].strip() else 'general')
                prompt=raw['instruction']+('\n\nPassage:\n'+raw['context'] if raw['context'] else '')
                yield dict(component=component,group='human:'+text_hash(prompt),messages=[dict(role='user',content=prompt),dict(role='assistant',content=raw['response'])],
                    provenance=dict(repo='databricks/databricks-dolly-15k',revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
                        license='CC-BY-SA-3.0',category=raw['category'],reference='Human-written reference'))
    base=base/'HuggingFaceTB___smol-smoltalk/default/0.0.0/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'
    for path in sorted(base.glob('smol-smoltalk-train-*.arrow')):
        yield dict(_receipt=dict(path=str(path),sha256=file_sha256(path),selected_source='everyday-conversations'))
        with pa.memory_map(str(path),'r') as stream:
            for batch in pa.ipc.open_stream(stream):
                for raw in batch.to_pylist():
                    if raw['source']!='everyday-conversations':continue
                    questions=[m['content'] for m in raw['messages'] if m['role']=='user' and not social_prompt(m['content'])]
                    yield dict(component='dialogue',group='dialogue:'+text_hash('\n'.join(questions)),messages=raw['messages'],
                        provenance=dict(repo='HuggingFaceTB/smol-smoltalk',revision='f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc',
                            license='Apache-2.0',source=raw['source'],reference='Upstream synthetic everyday dialogue; sampled review only'))


def knowledge_guard(project,stage,plan,blocked,overlap):
    from datasets import Dataset
    paths={
      'arc_easy':project/'pretraining/data/.sft_hf_cache/allenai___ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-validation.arrow',
      'piqa':project/'reports/benchmarks/cache/datasets/baber___piqa/default/0.0.0/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1/piqa-train.arrow'}
    if not paths['piqa'].exists():
        choices=list((project/'pretraining/data/.sft_hf_cache').glob('**/piqa-train.arrow'))
        if not choices:choices=list((Path.home()/'.cache/huggingface/datasets').glob('**/piqa-train.arrow'))
        if not choices:raise ValueError('Local PIQA training split is needed for the separate knowledge guard')
        paths['piqa']=choices[0]
    records=[];receipts=[]
    for task,path in paths.items():
        data=Dataset.from_file(str(path));indices=list(range(len(data)));random.Random(str(plan['seed'])+task).shuffle(indices)
        for i in indices[:plan['knowledge_examples_per_task']]:
            raw=data[i]
            if task=='arc_easy':
                question=raw['question'];choices=raw['choices']['text'];gold=raw['choices']['label'].index(raw['answerKey'])
            else:question=raw['goal'];choices=[raw['sol1'],raw['sol2']];gold=int(raw['label'])
            records.append(dict(task=task,index=i,question=question,choices=choices,gold=gold));blocked.add(text_hash(question));overlap.update(grams(question))
        receipts.append(dict(task=task,path=str(path),sha256=file_sha256(path),split='validation' if task=='arc_easy' else 'train',selected=plan['knowledge_examples_per_task']))
    atomic_json(stage/'knowledge_guard.json',dict(records=records,sources=receipts,
        scope='Unused-in-gradients ARC validation and PIQA training questions; official test/validation benchmarks remain final comparisons.'))


def prepare(config):
    plan=read_json(config);dest=Path(plan['prepared']);project=Path(__file__).resolve().parents[2]
    if dest.exists():return verify(dest,plan)
    tokenizer=Tokenizer(plan['tokenizer']);old,groups,old_keys=previous(plan)
    protected=groups['dev']|groups['test'];all_old=set.union(*groups.values())
    protected_keys=old_keys['dev']|old_keys['test'];all_old_keys=set.union(*old_keys.values())
    blocked,overlap,benchmarks=exclusions(project)
    panels={s:panel(s) for s in ('dev','test')}
    panel_keys=set().union(*(requests(r['messages']) for rows in panels.values() for r in rows))
    for rows in panels.values():
        for row in rows:
            for m in row['messages']:
                if m['role']=='user':blocked.add(text_hash(m['content']))
    stage=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True)
    pools=defaultdict(list);seen=set();rejects=Counter();sources=[]
    no_scope=re.compile(r'\b(?:role.?play|pretend|play the (?:part|role)|act as|calculus|lagrange|gradient|differential|cryptocurrency|investment advice|diagnos\w*|prescription|chemotherapy)\b',re.I)
    try:
        knowledge_guard(project,stage,plan,blocked,overlap)
        def accept(row,retained=False):
            component=row['component'];split='train' if retained else split_for(row['group'],plan['seed'])
            if component=='verified':split='train'
            limit=plan['quotas'][component] if split=='train' else plan.get('holdouts',{}).get(component,plan['holdout_per_component'])
            if len(pools[(component,split)])>=limit*2:return
            try:
                if any(text_hash(m['content']) in REVIEW_REJECTIONS for m in row['messages'] if m['role']=='user'):raise ValueError('reviewed_bad_reference')
                message_text='\n'.join(m['content'] for m in row['messages'])
                if no_scope.search(message_text):raise ValueError('out_of_scope')
                messages,ids,mask=clean(row,tokenizer,plan);keys=requests(messages)
                if not keys:raise ValueError('social_only')
                if row['group'] in protected or keys&protected_keys:raise ValueError('old_holdout')
                if split!='train' and (row['group'] in all_old or keys&all_old_keys):raise ValueError('old_seen_in_new_holdout')
                if keys&panel_keys:raise ValueError('behavior_panel_overlap')
                questions=[m['content'] for m in messages if m['role']=='user' and not social_prompt(m['content'])]
                if any(text_hash(q) in blocked or grams(q)&overlap for q in questions):raise ValueError('benchmark_overlap')
                if keys&seen:raise ValueError('duplicate_conditioned_request')
                answers=[m['content'] for m in messages if m['role']=='assistant']
                if any(looping(a) or unchanged_edit_claim(a) for a in answers):raise ValueError('repetitive_reference')
                if any(not complete(a) for a in answers):raise ValueError('unfinished_reference')
                if any(re.search(r'\b(?:I made (?:several|the following|a few) changes|I changed|I replaced)\b',a,re.I) for a in answers):raise ValueError('edit_commentary')
                if component=='general' and max(len(tokenizer.encode(a)) for a in answers)<8:raise ValueError('too_brief_general')
                if component=='dialogue' and len([m for m in messages if m['role']=='user'])<2:raise ValueError('not_multiturn')
                if row.get('verification') and not check(answers[-1],row['verification']['checks']):raise ValueError('invalid_authored_reference')
                seen.update(keys);pools[(component,split)].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask))
            except ValueError as exc:rejects[str(exc)]+=1
        for row in source_rows(project):
            if '_receipt' in row:sources.append(row['_receipt']);print('[prepare-balanced] '+Path(row['_receipt']['path']).name,flush=True)
            else:accept(row)
        for row in training_rows():accept(row)
        candidates=list(old['train']);random.Random(plan['seed']).shuffle(candidates)
        for row in candidates:
            if row['component']=='unanswerable':continue
            row=dict(row,component='retained',original_component=row['component']);row.pop('verification',None);accept(row,True)
        selected={s:[] for s in ('train','dev','test')}
        for split in selected:
            for component,quota in plan['quotas'].items():
                if component in ('verified','retained') and split!='train':continue
                rows=pools[(component,split)];target=quota if split=='train' else plan.get('holdouts',{}).get(component,plan['holdout_per_component'])
                if len(rows)<target:raise ValueError(f'Insufficient {component}/{split}: {len(rows)} < {target}; rejections={dict(rejects)}')
                random.Random(str(plan['seed'])+component+split).shuffle(rows);selected[split].extend(rows[:target])
            if split!='train':
                for row in panels[split]:
                    messages,ids,mask=clean(row,tokenizer,plan);selected[split].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask))
        counts={};split_groups={};split_keys={};answer_lengths=[]
        for split,rows in selected.items():
            random.Random(plan['seed']).shuffle(rows);offsets=[0];ids=[];masks=[];split_groups[split]=set();split_keys[split]=set()
            with (stage/(split+'.jsonl')).open('w') as stream:
                for row in rows:
                    a,b=row.pop('_ids'),row.pop('_mask');ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
                    split_groups[split].add(row['group']);split_keys[split].update(requests(row['messages']))
                    if split=='train':answer_lengths.extend(len(tokenizer.encode(m['content'])) for m in row['messages'] if m['role']=='assistant')
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            np.save(stage/(split+'.tokens.npy'),np.concatenate(ids),allow_pickle=False)
            np.save(stage/(split+'.mask.npy'),np.concatenate(masks),allow_pickle=False)
            np.save(stage/(split+'.offsets.npy'),np.array(offsets,np.uint64),allow_pickle=False)
            counts[split]=dict(conversations=len(rows),input_tokens=sum(len(a)-1 for a in ids),assistant_tokens=sum(int(b[1:].sum()) for b in masks),components=dict(Counter(r['component'] for r in rows)))
        for a,b in (('train','dev'),('train','test'),('dev','test')):
            if split_groups[a]&split_groups[b] or split_keys[a]&split_keys[b]:raise ValueError('Balanced split leakage')
        for split in ('dev','test'):
            behavior=list(panels[split])
            for component in ('general','grounded','dialogue','writing'):
                rows=[r for r in selected[split] if r['component']==component]
                for i,row in enumerate(rows[:plan['behavior_examples_per_component']]):
                    behavior.append(dict(id=split+'/'+component+'/'+str(i),family=split+'/natural/'+component,
                        messages=row['messages'],limit=512,verification=None,component=component))
            atomic_json(stage/('behavior_'+split+'.json'),behavior)
        replay_pool(project,stage,tokenizer,plan)
        audit=dict(cross_split_groups=0,cross_split_conditioned_requests=0,old_holdouts_in_train=0,old_seen_groups_in_new_holdouts=0,
            authored_training_families=sorted({r['family'] for r in selected['train'] if r['component']=='verified'}),
            authored_dev_families=[r['family'] for r in panels['dev']],authored_test_families=[r['family'] for r in panels['test']],
            assistant_lengths=dict(median=float(np.median(answer_lengths)),p90=float(np.quantile(answer_lengths,.9)),max=max(answer_lengths)),
            human_and_source_counts=dict(Counter(r['provenance']['repo'] for r in selected['train'])))
        assert not set(audit['authored_training_families'])&set(audit['authored_dev_families']+audit['authored_test_families'])
        atomic_json(stage/'audit.json',audit)
        manifest=dict(format=FORMAT,strategy='balanced-redo-v3',plan=fingerprint(plan),tokenizer=tokenizer.fingerprint,counts=counts,
            rejected=dict(rejects),sources=sources,benchmark_exclusions=benchmarks,
            builder_sources={name:file_sha256(Path(__file__).with_name(name)) for name in ('balanced_corpus.py','balanced_tasks.py','quality_gate.py')},
            limitations=['Human and synthetic source references can contain errors; filters and sampled review are not complete semantic certification.',
                'No Robots retains CC-BY-NC-4.0 terms and is also present in the preserved first-stage lineage.',
                'Authored panel generators and rule combinations are separated from the small authored training supplement; human sources can naturally exercise related skills.',
                'Literal/n-gram exclusions do not prove full semantic or pretraining benchmark decontamination.',
                'Regression gates cover measured prompts and datasets; unseen abilities can still regress.'])
        manifest['files']={p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in stage.iterdir() if p.is_file()}
        atomic_json(stage/'manifest.json',manifest);verify(stage,plan);stage.rename(dest);print('[prepared-balanced] '+str(dest),flush=True);return manifest
    finally:
        if stage.exists():shutil.rmtree(stage)
