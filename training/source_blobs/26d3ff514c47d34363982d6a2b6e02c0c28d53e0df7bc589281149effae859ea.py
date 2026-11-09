"""Short annotated/human SFT targets, grouped holdouts and train-only replay.

All upstream inputs are already local. Annotation/span and mechanical checks
have limited scope; they are not a universal factual correctness certificate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import uuid
import numpy as np
from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json, lock
from sml_v1.corpus import content_key, partition
from sml_v1.tokenization import Tokenizer
from .data import Corpus, serialize
from .quality import normalized, repeated, social_prompt

FORMAT='opensml-grounded-sft-corpus-v1'
ABSTAIN='Not stated in the passage.'
CODE=re.compile(r'```|\b(?:python|javascript|typescript|java|programming|pseudocode|sql|html|css|bash|source code|code snippet)\b|\bdef\s+\w+\s*\(',re.I)
DRIFT=re.compile(r'\b(?:role.?play|pretend|impersonate|act as|as an ai|as a language model|'
    r'i am (?:chatgpt|claude)|sorry for (?:the )?(?:confusion|misunderstanding))\b',re.I)


def text_hash(text):return hashlib.sha256(normalized(text).encode()).hexdigest()


def split_for(group,seed):
    bucket=int(fingerprint(dict(group=group,seed=seed))[:8],16)%10000
    return 'test' if bucket<200 else 'dev' if bucket<400 else 'train'


def clean(row,tokenizer,plan):
    messages=[dict(m) for m in row['messages']]
    while messages and messages[-1]['role']=='user':messages.pop()
    if not 2<=len(messages)<=plan['max_messages'] or messages[-1]['role']!='assistant':
        raise ValueError('incomplete_or_long_dialogue')
    for m in messages:
        text=m['content']
        if not isinstance(text,str) or not text.strip() or '\x00' in text or '\ufffd' in text:
            raise ValueError('invalid_text')
        if len(text)>12000 or CODE.search(text) or DRIFT.search(text):raise ValueError('code_or_task_drift')
        if m['role']=='assistant':
            if repeated(text):raise ValueError('repetition')
            if not 1<=len(tokenizer.encode(text))<=plan['assistant_max_tokens']:raise ValueError('answer_length')
    ids,mask=serialize(tokenizer,messages)
    if len(ids)>plan['max_sequence_length']+1:raise ValueError('context_length')
    return messages,ids,mask


def benchmark_hashes(project):
    import pyarrow as pa
    keys,receipts=set(),[]
    cache=project/'pretraining/data/.sft_hf_cache'
    for name in ('allenai___ai2_arc','google___if_eval','openai___gsm8k','cais___mmlu'):
        for path in sorted((cache/name).rglob('*.arrow')):
            if path.name.startswith('cache-'):continue
            with pa.memory_map(str(path),'r') as mm:
                table=pa.ipc.open_stream(mm).read_all()
                for field in ('question','prompt'):
                    if field in table.column_names:keys.update(text_hash(x) for x in table[field].to_pylist() if isinstance(x,str))
            receipts.append(dict(path=str(path),sha256=file_sha256(path)))
    # Previously inspected examples remain regression diagnostics, never targets.
    for prompt in ('Why do plants need sunlight? Answer briefly.','What does the scale on a map tell its reader?',
        'Summarize this in one sentence: The town library will close on Friday for repairs. It will reopen on Monday.',
        'Rewrite this sentence clearly: We was waiting but the bus never come.',
        'Write exactly three short bullet points about keeping a desk tidy.'):
        keys.add(text_hash(prompt))
    return keys,receipts


def pair(user,answer):return [dict(role='user',content=user),dict(role='assistant',content=answer)]


def squad_row(raw):
    texts=raw['answers']['text'];starts=raw['answers']['answer_start']
    if len(texts)!=len(starts):raise ValueError('answer_annotation_mismatch')
    if any(raw['context'][i:i+len(t)]!=t or not t.strip() for t,i in zip(texts,starts)):
        raise ValueError('answer_span_mismatch')
    if texts and any(len(t.split())>12 for t in texts):raise ValueError('long_span')
    answer=texts[0] if texts else ABSTAIN
    prompt='Answer using only the passage. Give only the shortest answer. If the answer is absent, reply "'+ABSTAIN+'"\n\nPassage:\n'+raw['context']+'\n\nQuestion: '+raw['question']
    return dict(component='grounded' if texts else 'unanswerable',group='squad-article:'+raw['title'],
        messages=pair(prompt,answer),question=raw['question'],evidence_hash=content_key(raw['context']),
        verification=dict(kind='span',answers=texts or [ABSTAIN]),
        provenance=dict(repo='rajpurkar/squad_v2',revision='3ffb306f725f7d2ce8394bc1873b24868140c412',split='train',id=raw['id'],license='CC-BY-SA-4.0'))


def authored_rows():
    """Finite task families with explicit references; no teacher generation."""
    rows=[]
    for i in range(1000):
        name=f'Parcel {i+401}';value=f'QN-{(i*37+113)%10007:04d}';other=f'RX-{(i*19+31)%10007:04d}'
        group='authored-record:'+str(i)
        context=f'{name} has tracking code {value}. The reserve shipment has tracking code {other}.'
        rows.append(dict(component='verified',group=group,messages=pair(context+'\nReturn only the tracking code for '+name+'.',value),verification=dict(kind='exact',answers=[value])))
        if i%2==0:
            rows.append(dict(component='verified',group=group,messages=pair(context+'\nWhat is the delivery date? If it is absent, reply "'+ABSTAIN+'"',ABSTAIN),verification=dict(kind='exact',answers=[ABSTAIN])))
    words=['amber','birch','cobalt','daisy','ember','fern','granite','hazel','indigo','jade','kiwi','lavender','maple','nectar','olive','pearl','quartz','ruby','sage','teal','umber','violet','willow','zinc','apricot','basil','cherry','dill','elm','fig','grape','honey','iris','jasmine','kelp','lemon','mint','nutmeg','oak','plum']
    for i in range(1000):
        a,b=words[i%40],words[(i//40+7)%40];label=a+' '+b+str(i+11)
        group='authored-label:'+str(i)
        rows.append(dict(component='verified',group=group,messages=pair('Copy this label exactly, with no extra words: '+label,label),verification=dict(kind='exact',answers=[label])))
    for row in rows:row['provenance']=dict(repo='local/short-grounded-skills',revision='20261102-v1',license='CC0-1.0',method='Authored templates and deterministic references; variants are correlated task families.')
    return rows


def cached_rows(project):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cache=project/'pretraining/data/.sft_hf_cache'
    sources=[
      cache/'rajpurkar___squad_v2/squad_v2/0.0.0/3ffb306f725f7d2ce8394bc1873b24868140c412/squad_v2-train.arrow',
      cache/'databricks___databricks-dolly-15k/default/0.0.0/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k-train.arrow',
      cache/'no_robots/e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/train-00000-of-00001.parquet']
    receipts=[]
    for index,path in enumerate(sources):
        receipts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path)))
        if path.suffix=='.parquet':tables=pq.ParquetFile(path).iter_batches(batch_size=256)
        else:
            # The mapping remains open for the lifetime of this source iterator.
            mm=pa.memory_map(str(path),'r');tables=pa.ipc.open_stream(mm)
        try:
            for batch in tables:
                for n,raw in enumerate(batch.to_pylist()):
                    if index==0:
                        try:yield squad_row(raw)
                        except ValueError:continue
                    elif index==1:
                        if raw['category'] not in ('open_qa','closed_qa','information_extraction','classification'):continue
                        prompt=raw['instruction']+('\n\nPassage:\n'+raw['context'] if raw['context'] else '')
                        yield dict(component='human',group='human:'+text_hash(prompt),messages=pair(prompt,raw['response']),
                            provenance=dict(repo='databricks/databricks-dolly-15k',revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',license='CC-BY-SA-3.0',category=raw['category']))
                    else:
                        if raw['category'] not in ('Open QA','Closed QA','Classify','Extract','Chat'):continue
                        messages=raw['messages']
                        # Keep one complete exchange, never a truncated answer.
                        if len(messages)<2 or [m['role'] for m in messages[:2]]!=['user','assistant']:continue
                        yield dict(component='human',group='human:'+text_hash(messages[0]['content']),messages=messages[:2],
                            provenance=dict(repo='HuggingFaceH4/no_robots',revision='e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b',license='CC-BY-NC-4.0',id=raw['prompt_id']))
        finally:
            if path.suffix!='.parquet':mm.close()
    path=cache/'HuggingFaceTB___smol-smoltalk/default/0.0.0/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'
    for f in sorted(path.glob('smol-smoltalk-train-*.arrow')):
        receipts.append(dict(path=str(f),sha256=file_sha256(f),selected_source='everyday-conversations'))
        with pa.memory_map(str(f),'r') as mm:
            for batch in pa.ipc.open_stream(mm):
                for raw in batch.to_pylist():
                    if raw['source']!='everyday-conversations':continue
                    questions=[m['content'] for m in raw['messages'] if m['role']=='user' and not social_prompt(m['content'])]
                    yield dict(component='dialogue',group='dialogue:'+text_hash('\n'.join(questions)),messages=raw['messages'],
                        provenance=dict(repo='HuggingFaceTB/smol-smoltalk',revision='f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc',source=raw['source'],license='Apache-2.0'))
    yield dict(_receipts=receipts)


def replay_pool(project,stage,tokenizer,plan):
    """Reuse checksum-verified prose candidates, admitting only TRAIN documents."""
    from sml_v1.prose_filter import heuristic_reason
    from .text_quality import CODE
    root=project/'pretraining/data/prose_20b_20261022/audit-v0/candidates'
    length=plan['replay_length'];rows=plan['replay_pool_tokens']//length
    weights=dict(web=70,dclm=20,wiki=10);parts=[];receipts=[];seen=set();sources=[];docs=0
    for source,weight in weights.items():
        count=rows*weight//100
        if source=='web':count+=rows-sum(rows*w//100 for w in weights.values())
        target=count*length+1;segments=[];total=0
        for marker in sorted((root/source).glob('chunk-*.json')):
            m=read_json(marker);prefix=marker.with_suffix('');paths=[Path(str(prefix)+s) for s in ('.bin','.records.npy')]
            for p in paths:
                if file_sha256(p)!=m['files'][p.name]:raise ValueError('Replay source checksum mismatch')
            receipts.append(dict(path=str(marker),sha256=file_sha256(marker),files={p.name:m['files'][p.name] for p in paths}))
            records=np.load(paths[1],mmap_mode='r');tokens=np.memmap(paths[0],mode='r',dtype='<u2')
            for record in records:
                key=bytes(record['key']).hex()
                if key in seen or partition(key)!='train':continue
                start=int(record['start']);ids=np.array(tokens[start:start+int(record['tokens'])],dtype=np.int32)
                if not len(ids) or ids[-1]!=1:raise ValueError('Replay document EOS mismatch')
                text=tokenizer.decode(ids[:-1])
                if content_key(text)!=key:raise ValueError('Replay tokenizer/content identity mismatch')
                if heuristic_reason(text,200,200000) or CODE.search(text):continue
                if source=='dclm' and float(record['score'])<3:continue
                seen.add(key);segments.append(ids);total+=len(ids);docs+=1
                if total>=target:break
            if total>=target:break
        if total<target:raise ValueError('Insufficient checksum-verified train-only replay')
        flat=np.concatenate(segments)
        parts.extend(flat[i*length:i*length+length+1].astype('<u2') for i in range(count));sources.extend([source]*count)
    order=np.random.default_rng(plan['seed']).permutation(len(parts))
    np.save(stage/'replay.npy',np.stack(parts)[order],allow_pickle=False)
    atomic_json(stage/'replay.json',dict(rows=len(parts),length=length,documents=docs,
        partition='train',source_counts=dict(Counter(sources)),source_receipts=receipts,
        selected_document_keys=sorted(seen),policy='No validation/final/tokenizer-dev documents; non-overlapping target segments, no row recycling.'))


def verify(path,plan=None):
    path=Path(path);m=read_json(path/'manifest.json')
    if m['format']!=FORMAT or (plan is not None and m['plan']!=fingerprint(plan)):
        raise ValueError('Grounded SFT data plan differs')
    for name,r in m['files'].items():
        f=path/name
        if Path(name).name!=name or f.is_symlink() or f.stat().st_size!=r['bytes'] or file_sha256(f)!=r['sha256']:
            raise ValueError('Grounded SFT corpus checksum mismatch: '+name)
    for split in ('train','dev','test'):
        c=Corpus(path,split)
        if not len(c) or len(c)!=m['counts'][split]['conversations']:raise ValueError('Invalid grounded SFT split')
    return m


def prepare(config):
    plan=read_json(config);dest=Path(plan['prepared']);project=Path(__file__).resolve().parents[2]
    if dest.exists():return verify(dest,plan)
    tokenizer=Tokenizer(plan['tokenizer']);blocked,benchmarks=benchmark_hashes(project)
    pools=defaultdict(list);rejects=Counter();seen=set();receipts=[]
    def accept(row):
        if '_receipts' in row:receipts.extend(row['_receipts']);return
        try:
            messages,ids,mask=clean(row,tokenizer,plan)
            questions=[m['content'] for m in messages if m['role']=='user']
            if any(text_hash(q) in blocked for q in questions+[row.get('question','')]):raise ValueError('known_benchmark')
            key=text_hash('\n'.join(questions))
            if key in seen:raise ValueError('duplicate_prompt')
            seen.add(key);row=dict(row,messages=messages,_ids=ids,_mask=mask)
            row['split']=split_for(row['group'],plan['seed']);pools[(row['component'],row['split'])].append(row)
        except ValueError as exc:rejects[str(exc)]+=1
    for row in cached_rows(project):accept(row)
    for row in authored_rows():accept(row)
    selected=defaultdict(list)
    for component,quota in plan['quotas'].items():
        for split in ('train','dev','test'):
            rows=pools[(component,split)];random.Random(str(plan['seed'])+component+split).shuffle(rows)
            target=quota if split=='train' else max(32,round(quota/24))
            chosen=rows[:target]
            if not chosen:raise ValueError('Empty component: '+component+'/'+split)
            selected[split].extend(chosen)
    stage=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True)
    try:
        with lock(dest.with_suffix('.lock')):
            counts={}
            for split,rows in selected.items():
                random.Random(plan['seed']).shuffle(rows);offsets=[0];ids=[];masks=[]
                with (stage/(split+'.jsonl')).open('w') as f:
                    for row in rows:
                        a,b=row.pop('_ids'),row.pop('_mask');ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(json.dumps(row,ensure_ascii=False)+'\n')
                np.save(stage/(split+'.tokens.npy'),np.concatenate(ids),allow_pickle=False);np.save(stage/(split+'.mask.npy'),np.concatenate(masks),allow_pickle=False);np.save(stage/(split+'.offsets.npy'),np.array(offsets,np.uint64),allow_pickle=False)
                counts[split]=dict(conversations=len(rows),input_tokens=sum(len(a)-1 for a in ids),assistant_tokens=sum(int(b[1:].sum()) for b in masks),components=dict(Counter(r['component'] for r in rows)))
            replay_pool(project,stage,tokenizer,plan)
            receipt=dict(format=FORMAT,plan=fingerprint(plan),tokenizer=tokenizer.fingerprint,counts=counts,rejected=dict(rejects),sources=receipts,benchmark_exclusions=benchmarks,
                limitations=['Human and source annotations can contain mistakes.','SQuAD span offsets are checked; unanswerability follows source annotations.','Authored variants are correlated task families.','Exact exclusions do not prove semantic decontamination.','No Robots carries CC-BY-NC-4.0; record source terms before any downstream distribution.'])
            receipt['files']={p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in stage.iterdir() if p.is_file()}
            atomic_json(stage/'manifest.json',receipt);verify(stage,plan);stage.rename(dest);return receipt
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.config)['counts'],indent=2),flush=True)
