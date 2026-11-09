"""Bounded public streaming selection and fully audited prose-only SFT arrays."""
from collections import Counter
import json
from pathlib import Path
import random
import numpy as np
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint
from sml_v1.stream import SourceCursor,StreamExhausted
from sml_v1.tokenization import Tokenizer
from .text_quality import FORMAT,validate,key,new_public_chat,dolly,squad,science,CODE,PUBLIC_SCOPE,review_row
from .data import Corpus,training_order,serialize
from .text_support import exclusions


def split_for(group,seed):
    value=int(fingerprint(dict(group=group,seed=seed))[:8],16)%100
    return 'dev' if value<10 else 'test' if value<20 else 'train'


def reference_training(path):
    groups=set();questions=set()
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);groups.add(row['group']);questions.add(key(row['question']))
        questions.update(key(m['content']) for m in row['messages'] if m['role']=='user' and len(m['content'].split())>=5)
    return groups,questions


def reference_overlap(row,reference):
    groups,questions=reference
    return (row['group'] in groups or key(row['question']) in questions or
            any(key(m['content']) in questions for m in row['messages'] if m['role']=='user' and len(m['content'].split())>=5))


def verify(path,plan=None):
    path=Path(path);manifest=read_json(path/'manifest.json')
    if manifest['format']!=FORMAT or (plan is not None and manifest['plan']!=fingerprint(plan)):raise ValueError('Data plan differs')
    for name,r in manifest['files'].items():
        f=path/name
        if Path(name).name!=name or f.is_symlink() or f.stat().st_size!=r['bytes'] or file_sha256(f)!=r['sha256']:raise ValueError('Data changed: '+name)
    return manifest


def select(source,plan,tok,here,blocked,questions,chat_groups,group_splits,reference):
    label=source['label'];path=here/'selection'/(label+'.jsonl');path.parent.mkdir(exist_ok=True);saved=path.with_suffix('.cursor.json')
    rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    counts=Counter((r['slot'],r['split']) for r in rows);groups=Counter(r['group'] for r in rows)
    slots=dict(smol=['general_smol','dialogue_smol','constraints_smol','rewrite_smol'],sciq=['knowledge'],social=['social'],squad=['grounded_yes','grounded_no'])[label]
    caps={(s,split):plan['public_quotas'][s] if split=='train' else plan['public_holdouts'][s] for s in slots for split in ['train','dev','test']}
    if all(counts[k]==n for k,n in caps.items()):return rows
    receipt=read_json(saved) if saved.exists() else dict(plan=fingerprint(plan),cursor=None,rejected={})
    if receipt['plan']!=fingerprint(plan):raise ValueError('Source selection plan changed')
    cursor=SourceCursor(source,4,receipt['cursor']);rejected=Counter(receipt['rejected'])
    with path.open('a') as out:
        def save():
            out.flush();atomic_json(saved,dict(plan=fingerprint(plan),cursor=cursor.state_dict(),counts={str(k):v for k,v in counts.items()},rejected=dict(rejected),prior_pass_rows=receipt.get('prior_pass_rows',0)))
        try:
            while not all(counts[k]==n for k,n in caps.items()):
                try:raw=cursor.next()
                except StreamExhausted:break
                candidates=['dialogue','general'] if label=='smol' else [None]
                for mode in candidates:
                    try:
                        row=new_public_chat(raw,label,mode) if label in ('smol','social') else science(raw) if label=='sciq' else squad(raw)
                        if label=='dolly' and PUBLIC_SCOPE.search(row['question']):raise ValueError('public_unverified_scope')
                        slot=(row['component']+'_smol' if label=='smol' else 'knowledge' if label=='sciq' else 'social' if label=='social' else 'grounded_'+('yes' if row['verification']['answerable'] else 'no'))
                        split=split_for(row['group'],plan['seed']);counter=(slot,split)
                        if counts[counter]>=caps[counter]:raise ValueError('quota')
                        if split!='train' and reference_overlap(row,reference):raise ValueError('seen_by_reference_417')
                        if label=='squad':
                            if groups[row['group']] >= (48 if split=='train' else 8):raise ValueError('article_cap')
                        elif row['group'] in chat_groups:raise ValueError('duplicate_group')
                        digest=key(row['question'])
                        if digest in questions:raise ValueError('duplicate_question')
                        if digest in blocked or any(key(m['content']) in blocked for m in row['messages'] if m['role']=='user' and len(m['content'].split())>=5):raise ValueError('known_holdout')
                        review_row(row)
                        validate(row['messages'],tok,plan,row['component'])
                        if split!='train':
                            ids,_=serialize(tok,row['messages'][:-1],generation=True)
                            if len(ids)+plan['answer_max_tokens'][row['component']]+1>2048:raise ValueError('generation_room')
                        if row['group'] in group_splits and group_splits[row['group']]!=split:raise ValueError('group_split')
                        row.update(split=split,slot=slot,provenance=dict(row['provenance'],repo=source['repo'],revision=source['revision'],license=source['license'],source_split='train',source_row=cursor.rows-1))
                        rows.append(row);counts[counter]+=1;groups[row['group']]+=1;questions.add(digest);group_splits[row['group']]=split
                        if label!='squad':chat_groups.add(row['group'])
                        out.write(json.dumps(row,ensure_ascii=True)+'\n');break
                    except (ValueError,KeyError,TypeError,IndexError) as e:rejected[str(e)[:100]]+=1
                if cursor.rows%3000==0:
                    save();print(f'[select] {label} rows={cursor.rows} selected={dict(counts)}',flush=True)
        finally:save();cursor.close()
    if any(counts[k]!=n for k,n in caps.items()):raise ValueError(f'Insufficient clean {label}: {dict(counts)} expected {caps}')
    print(f'[selected] {label}: {dict(counts)}',flush=True)
    return rows


def prose_replay(project,dest,tok,plan,steps):
    from .grounded_corpus import replay_pool
    import hashlib
    required=steps*16
    replay_pool(project,dest,tok,dict(plan,replay_pool_tokens=(required+1024)*plan['replay_length']))
    values=np.load(dest/'replay.npy',mmap_mode='r',allow_pickle=False)
    kept=[];seen=set();rejected=Counter()
    for i,row in enumerate(values):
        if CODE.search(tok.decode(row.tolist())):rejected['window_scope']+=1;continue
        digest=hashlib.sha256(np.asarray(row,dtype='<i4').tobytes()).digest()
        if digest in seen:rejected['duplicate_window']+=1;continue
        seen.add(digest);kept.append(i)
        if len(kept)==required:break
    if len(kept)!=required:raise ValueError('Insufficient unique prose replay windows')
    selected=np.array(values[kept],dtype=np.int32);del values
    meta=read_json(dest/'replay.json');labels=[s for s,n in meta['source_counts'].items() for _ in range(n)]
    # replay_pool builds source blocks in this explicit order, before shuffling.
    labels=[s for s in ('web','dclm','wiki') for _ in range(meta['source_counts'][s])]
    order=np.random.default_rng(plan['seed']).permutation(len(labels))
    meta.update(candidate_rows=meta['rows'],rows=required,source_counts=dict(Counter(labels[order[i]] for i in kept)),
                selected_candidate_indices=kept,window_rejections=dict(rejected),candidate_documents=meta['documents'])
    np.save(dest/'replay.npy',selected,allow_pickle=False);atomic_json(dest/'replay.json',meta)


def prepare(here):
    here=Path(here).resolve();project=here.parents[2];plan=read_json(here/'plan.json');dest=Path(plan['prepared'])
    if dest.exists():return verify(dest,plan)
    reference_path=here.parent/'text_chat_20k_sft_20261105/data/train.jsonl'
    reference=reference_training(reference_path)
    atomic_json(here/'reference_holdout_exclusions.json',dict(source=str(reference_path),sha256=file_sha256(reference_path),
        groups=len(reference[0]),questions=len(reference[1]),policy='New dev/test exclude checkpoint 417 training groups and normalized user questions; new train may reuse prior train.'))
    tok=Tokenizer(plan['tokenizer']);blocked,receipts=exclusions(project);blocked.update(read_json(here/'previous_holdout_hashes.json'))
    if (here/'review_exclusions.json').exists():blocked.update(read_json(here/'review_exclusions.json')['question_keys'])
    benchmark=project/'reports/benchmarks/chat_text_v1_20261105/cases.jsonl'
    for line in benchmark.read_text().splitlines():
        for message in json.loads(line)['messages']:
            if message['role']=='user':blocked.add(key(message['content']))
    atomic_json(here/'exclusions.json',dict(count=len(blocked),source_receipts=receipts,chat_benchmark_sha256=file_sha256(benchmark)))
    for split in ('dev','test'):
        old=here.parent/'text_chat_20k_sft_20261105/data'/(split+'.jsonl')
        for line in old.read_text().splitlines():
            row=json.loads(line);blocked.add(key(row['question']))
            for m in row['messages']:
                if m['role']=='user' and len(m['content'].split())>=5:blocked.add(key(m['content']))
    existing=[]
    for file in (here/'selection').glob('*.jsonl'):existing.extend(json.loads(l) for l in file.read_text().splitlines())
    questions={key(r['question']) for r in existing};chat_groups={r['group'] for r in existing if r['subset']!='squad'};group_splits={r['group']:r['split'] for r in existing}
    rows=[]
    for source in plan['sources']:rows.extend(select(source,plan,tok,here,blocked,questions,chat_groups,group_splits,reference))
    from .verified_tasks import make,recompute,FAMILIES
    for component in plan['authored_quotas']:
        for split in ('train','dev','test'):
            quota=plan['authored_quotas'][component] if split=='train' else plan['authored_holdouts'][component];n=0;family_counts=Counter()
            families=FAMILIES[component]
            caps={str(i):quota//families+int(i<quota%families) for i in range(families)}
            for row in make(split,component,quota*40,plan['seed']):
                family=row['provenance']['family']
                if family_counts[family]>=caps[family]:continue
                q=key(row['question'])
                if q in questions or q in blocked:continue
                if split!='train' and reference_overlap(row,reference):continue
                if any(key(m['content']) in blocked for m in row['messages'] if m['role']=='user' and len(m['content'].split())>=5):continue
                try:validate(row['messages'],tok,plan,component)
                except ValueError:continue
                if component=='skills' and recompute(row['verification'])!=row['messages'][-1]['content']:raise ValueError('Authored oracle mismatch')
                questions.add(q);rows.append(row);n+=1;family_counts[family]+=1
                if n==quota:break
            if n!=quota:raise ValueError(f'Insufficient authored {component}/{split}: {n}')
    stage=here/'data.building';stage.mkdir(exist_ok=True);counts={};stats={};groups={}
    for split in ('train','dev','test'):
        selected=[r for r in rows if r['split']==split];random.Random(str(plan['seed'])+split).shuffle(selected)
        tokens=[];masks=[];offsets=[0];lengths=[];target_count=0;assistant_turns=0
        for row in selected:
            ids,mask=validate(row['messages'],tok,plan,row['component']);tokens.append(ids);masks.append(mask);offsets.append(offsets[-1]+len(ids));lengths.append(len(ids)-1);target_count+=int(mask[1:].sum());assistant_turns+=sum(m['role']=='assistant' for m in row['messages'])
        (stage/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in selected))
        np.save(stage/(split+'.tokens.npy'),np.concatenate(tokens),allow_pickle=False);np.save(stage/(split+'.mask.npy'),np.concatenate(masks),allow_pickle=False);np.save(stage/(split+'.offsets.npy'),np.asarray(offsets,np.uint64),allow_pickle=False)
        counts[split]=dict(conversations=len(selected),components=dict(Counter(r['component'] for r in selected)),subsets=dict(Counter(r['subset'] for r in selected)),input_tokens=sum(lengths),assistant_tokens=target_count,assistant_turns=assistant_turns,multiple_exchanges=sum(sum(m['role']=='assistant' for m in r['messages'])>1 for r in selected))
        stats[split]=dict(max_sequence=max(lengths),p50=float(np.median(lengths)),p95=float(np.quantile(lengths,.95)),truncated=0)
        groups[split]={r['group'] for r in selected}
    assert counts['train']['components']==plan['component_quotas']
    assert all(not groups[a]&groups[b] for a,b in [('train','dev'),('train','test'),('dev','test')])
    train=Corpus(stage,'train');steps=len(training_order(train.lengths,plan['seed'],48,plan['epochs']));prose_replay(project,stage,tok,plan,steps)
    manifest=dict(format=FORMAT,plan=fingerprint(plan),tokenizer=tok.fingerprint,counts=counts,token_stats=stats,total_steps=steps,sources=plan['sources'],
        limitations=['Public synthetic/human answers are filtered and sample-reviewed, not independently verified fact by fact.','SQuAD span checks verify annotations, not every annotation’s semantic correctness.','Exact prompt exclusions and grouped splits do not prove semantic decontamination.','Authored operations overlap across splits; entities and instruction templates are partitioned. Public constraint properties are checked, but semantics are sample-reviewed.'])
    manifest['files']={f.name:dict(bytes=f.stat().st_size,sha256=file_sha256(f)) for f in stage.iterdir() if f.is_file() and f.name!='manifest.json'}
    atomic_json(stage/'manifest.json',manifest);verify(stage,plan);stage.rename(dest)
    print(json.dumps({k:manifest[k] for k in ['counts','token_stats','total_steps']},indent=2),flush=True)
    return manifest
