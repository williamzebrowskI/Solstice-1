"""Non-ARC science SFT search: fixed splits, overlap screening, labeled targets.

ARC questions are read only to reject overlap, never emitted as training rows.
This is an experimental research corpus, not an exhaustively verified truth set.
"""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid
import numpy as np
from datasets import Image, load_dataset
from .curate import local, source_rows, filter_messages, write_split, verify
from .quality import digest, normalized, repeated
from .conversation import prior_exclusions
from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json

POOL = 'pretraining/data/sft_science_search_20261008'
ARC_REVISION = '210d026faf9955653af8916fad021475a3f00453'
SOURCES = [
    dict(bucket='sciq', repo='allenai/sciq', config='default', revision='2c94ad3e1aafab77146f384e23536f97a4849815', split='train', license='CC-BY-NC-3.0; research-only'),
    dict(bucket='qasc', repo='allenai/qasc', config='default', revision='a34ba204eb9a33b919c10cc08f4f1c8dae5ec070', split='train', license='CC-BY-4.0'),
    dict(bucket='commonsense', repo='tau/commonsense_qa', config='default', revision='94630fe30dad47192a8546eb75f094926d47e155', split='train', license='MIT'),
    dict(bucket='nq', repo='google-research-datasets/nq_open', config='nq_open', revision='5dd9790a83002ad084ddeb7c420dc716852c6f28', split='train', license='CC-BY-SA-3.0'),
    dict(bucket='scienceqa', repo='derek-thomas/ScienceQA', config='default', revision='f18b0a70359ebfb41f658fd564208d0355b013f4', split='train', license='CC-BY-SA-4.0'),
]


class OverlapFilter:
    """Exact, eight-word substring, and high-similarity question exclusions."""
    def __init__(self, questions):
        self.questions = sorted({normalized(q) for q in questions if normalized(q)})
        self.exact = set(self.questions)
        self.grams = set()
        self.index = defaultdict(set)
        for i, q in enumerate(self.questions):
            words = q.split()
            self.grams.update(' '.join(words[j:j+8]) for j in range(len(words)-7))
            for word in set(words):
                self.index[word].add(i)

    def reject(self, question, other_text=''):
        q = normalized(question)
        if q in self.exact:
            return 'exact_question'
        for text in (q, normalized(other_text)):
            words = text.split()
            if any(' '.join(words[j:j+8]) in self.grams for j in range(len(words)-7)):
                return 'shared_eight_words'
        words = set(q.split())
        rare = sorted((w for w in words if w in self.index), key=lambda w:len(self.index[w]))[:3]
        candidates = set().union(*(self.index[w] for w in rare)) if rare else set()
        for i in candidates:
            ref = self.questions[i]
            refwords = set(ref.split())
            if len(words & refwords)/max(1,len(words | refwords)) >= .8 and SequenceMatcher(None,q,ref).ratio() >= .85:
                return 'near_question'
        return None


def group_split(question):
    n = int(fingerprint({'seed':260108,'question':normalized(question)})[:8],16)%1000
    return 'test' if n < 40 else 'dev' if n < 80 else 'train'


def science_row(raw, spec, index):
    bucket = spec['bucket']; q = raw['question'].strip()
    context = ''; support = ''; choices = []
    if bucket == 'sciq':
        answer = raw['correct_answer'].strip()
        choices = [answer,*[raw[f'distractor{i}'].strip() for i in (1,2,3)]]
        support = raw['support'].strip()
    elif bucket in ('qasc','commonsense'):
        choices = raw['choices']['text']
        answer = choices[raw['choices']['label'].index(raw['answerKey'])].strip()
        if bucket == 'qasc':
            support = raw['combinedfact'].strip()
    elif bucket == 'nq':
        answer = raw['answer'][0].strip()
    elif bucket == 'scienceqa':
        if raw['image'] is not None:
            raise ValueError('requires_image')
        if raw['subject'] != 'natural science':
            raise ValueError('not_natural_science')
        choices = raw['choices']; answer = choices[int(raw['answer'])].strip()
        context = raw['hint'].strip(); support = raw['solution'].strip()
        if re.search(r'\b(?:image|picture|diagram|map|graph|figure|shown|photograph)\b', q+' '+context+' '+support,re.I):
            raise ValueError('visual_dependency')
    else:
        raise ValueError('unknown_source')
    if not q or not answer or len(answer.split()) > 40 or len(q.split()) > 120 or len(context.split()) > 200:
        raise ValueError('invalid_or_long_question')
    if choices and (len(set(normalized(c) for c in choices)) != len(choices) or answer not in choices):
        raise ValueError('invalid_choices')
    return dict(bucket=bucket, question=q, answer=answer, choices=choices, context=context,
                support=support, group='science:'+digest(q), provenance=dict(spec,row=index))


def build_pool():
    root = local(POOL); root.mkdir(parents=True,exist_ok=True)
    receipt_path = root/'pool.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        for name,sha in receipt['files'].items():
            if file_sha256(root/name)!=sha:raise ValueError('Search pool changed')
        return receipt
    cache=root/'hf-cache'; arc_questions=[]; arc_receipts=[]
    for config in ('ARC-Easy','ARC-Challenge'):
        for split in ('train','validation','test'):
            spec=dict(repo='allenai/ai2_arc',config=config,split=split,revision=ARC_REVISION)
            rows=source_rows(spec,cache)
            arc_questions.extend(row['question'] for row in rows)
            arc_receipts.append(dict(spec,rows=len(rows),purpose='overlap exclusion only; no training targets'))
    overlap=OverlapFilter(arc_questions)
    oldgroups,oldprompts,oldreceipts=prior_exclusions()
    previous=local('pretraining/data/sft_176m_conversation_v2_20261007')
    for split in ('dev','test'):
        for line in (previous/f'{split}.jsonl').read_text().splitlines():
            row=json.loads(line);oldgroups.add(row['group'])
            oldprompts.update(digest(m['content']) for m in row['messages'] if m['role']=='user')
    observed=local('reports/sft/SFT-004-conversation-v2-20261008/playground-diagnostics.json')
    if observed.exists():
        oldprompts.update(digest(r['prompt']) for r in read_json(observed)['rows'])
    selected={s:[] for s in ('train','dev','test')}; rejected=Counter(); seen=set()
    for spec in SOURCES:
        rows=source_rows(spec,cache)
        if spec['bucket']=='scienceqa':rows=rows.cast_column('image',Image(decode=False))
        for i,raw in enumerate(rows):
            try:
                r=science_row(raw,spec,i)
                if digest(r['question']) in oldprompts:raise ValueError('observed_prompt')
                why=overlap.reject(r['question'],r['context']+' '+r['support']+' '+r['answer'])
                if why:raise ValueError('arc_'+why)
                if r['group'] in seen:raise ValueError('cross_source_duplicate')
                seen.add(r['group']);selected[group_split(r['question'])].append(r)
            except ValueError as e:rejected[spec['bucket']+':'+str(e)]+=1
        print('[pool]',spec['bucket'],dict(Counter(r['bucket'] for r in selected['train'])),flush=True)
    anchors={s:[] for s in selected}
    science_prompts={digest(r['question']) for rs in selected.values() for r in rs}
    for split in selected:
        for line in (previous/f'{split}.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r.get('system_variant') or r['bucket'] not in ('dolly','oasst','everyday','qa','constraints','python','skills'):continue
            question='\n'.join(m['content'] for m in r['messages'] if m['role']=='user')
            if any(digest(m['content'].rsplit('Question:',1)[-1]) in science_prompts for m in r['messages'] if m['role']=='user'):
                rejected['anchor:science_question_duplicate']+=1;continue
            if overlap.reject(question,' '.join(m['content'] for m in r['messages'])):
                rejected['anchor:arc_overlap']+=1;continue
            anchors[split].append(r)
    for split in selected:
        with (root/f'pool-{split}.jsonl').open('w') as f:
            for r in selected[split]:f.write(json.dumps(r)+'\n')
        with (root/f'anchors-{split}.jsonl').open('w') as f:
            for r in anchors[split]:f.write(json.dumps(r)+'\n')
    # Frozen non-ARC dev choices are identical across all ten plans.
    choices=[]
    for bucket in ('sciq','qasc','commonsense','scienceqa'):
        subset=[r for r in selected['dev'] if r['bucket']==bucket and r['choices']]
        random.Random(260108).shuffle(subset)
        for r in subset[:16]:choices.append(dict(r, gold=r['choices'].index(r['answer'])))
    if len(choices)<48:raise ValueError('Insufficient independent science choice dev panel')
    atomic_json(root/'science-dev.json',choices)
    files={p.name:file_sha256(p) for p in root.iterdir() if p.is_file()}
    result=dict(format='opensml-science-search-pool-v1',sources=SOURCES,files=files,
                counts={s:dict(Counter(r['bucket'] for r in rs)) for s,rs in selected.items()},
                anchors={s:len(rs) for s,rs in anchors.items()},rejected=dict(rejected),arc_exclusions=arc_receipts,
                prior_exclusions=oldreceipts,limitations=['Question overlap screening cannot prove semantic decontamination.','SciQ is noncommercial; runs using it are research-only.','Labeled/support text can contain source errors.'])
    atomic_json(receipt_path,result);return result


def supporting_sentence(row):
    sentences=re.split(r'(?<=[.!?])\s+',row['support'])
    preferred=[s for s in sentences if normalized(row['answer']) in normalized(s)]
    for sentence in preferred+sentences:
        if 4<=len(sentence.split())<=65 and not repeated(sentence):return sentence
    return ''


def messages_for(row, style, seed):
    rng=random.Random(seed+int(digest(row['question'])[:8],16))
    selected_style=rng.choice(('direct','choice','explain')) if style=='mixed' else style
    question=row['question'];answer=row['answer'];explanation=supporting_sentence(row)
    if row['context']:question=row['context']+'\nQuestion: '+question
    if selected_style=='grounded' and row['support'] and len(row['support'].split())<=160 and rng.random()<.5:
        question='Use this passage to answer briefly:\n'+row['support']+'\nQuestion: '+question
    elif selected_style=='choice' and row['choices']:
        choices=list(row['choices']);rng.shuffle(choices)
        question+='\n'+ '\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(choices))+'\nGive the answer text, not a letter.'
    elif selected_style=='explain' and explanation:
        question+='\nGive a brief answer and explanation.'
        answer=answer.rstrip('.')+'. '+explanation
    return [dict(role='user',content=question),dict(role='assistant',content=answer)]


def native_qa(tokenizer, row, plan):
    """Answer-only document continuation, from the same NON-ARC source label."""
    prompt=(row['context']+'\n' if row['context'] else '')+'Question: '+row['question']+'\nAnswer:'
    prefix=tokenizer.assert_roundtrip(prompt)
    joint=tokenizer.assert_roundtrip(prompt+' '+row['answer'])
    if joint[:len(prefix)]!=list(prefix):raise ValueError('Native QA token boundary')
    ids=np.asarray([*joint,1],np.int32)
    mask=np.zeros(len(ids),np.bool_);mask[len(prefix):]=True
    if len(ids)-1>plan['max_seq_len'] or int(mask.sum())>plan['assistant_max_tokens']:
        raise ValueError('Native QA length')
    return ids,mask,prompt


def prepare(config):
    from sml_v1.tokenization import Tokenizer
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
        result=verify(dest)
        if result['plan_sha256']!=file_sha256(config):raise ValueError('Prepared science plan changed')
        return result
    pool=build_pool();root=local(POOL);tokenizer=Tokenizer(local(plan['tokenizer']))
    selected={s:[] for s in ('train','dev','test')}; rejected=Counter()
    for split in selected:
        rows=[json.loads(l) for l in (root/f'pool-{split}.jsonl').read_text().splitlines()]
        random.Random(plan['seed']).shuffle(rows);counts=Counter()
        for r in rows:
            cap=plan['science_caps'].get(r['bucket'],0) if split=='train' else 24
            if counts[r['bucket']]>=cap:continue
            try:
                msgs=messages_for(r,plan['science_style'] if split=='train' else 'direct',plan['seed'])
                if split=='train' and int(digest(r['group'])[:2],16)%2:
                    msgs=[dict(role='system',content=plan['conversation_system']),*msgs]
                ids,mask=filter_messages(msgs,tokenizer,'qa',plan)
                mode='chat';native_prompt=None
                if split=='train' and int(digest(r['group'])[-4:],16)/65536 < plan.get('native_qa_fraction',0):
                    ids,mask,native_prompt=native_qa(tokenizer,r,plan);mode='plain-qa'
                    msgs=[dict(role='user',content=native_prompt),dict(role='assistant',content=r['answer'])]
                selected[split].append(dict(r,component='qa',messages=msgs,split=split,
                    answers=[r['answer']],answerable=True,serialization_mode=mode,native_prompt=native_prompt,_ids=ids,_mask=mask))
                counts[r['bucket']]+=1
            except ValueError as e:rejected[r['bucket']+':'+str(e)]+=1
        anchors=[json.loads(l) for l in (root/f'anchors-{split}.jsonl').read_text().splitlines()]
        random.Random(plan['seed']).shuffle(anchors)
        # Guarantee small component panels, then sample the requested anchor budget.
        by_component=defaultdict(list)
        for r in anchors:by_component[r['component']].append(r)
        chosen=[]
        per=24 if split!='train' else min(200,plan['anchor_cap']//5)
        for rs in by_component.values():chosen+=rs[:per]
        used={r['group'] for r in chosen}
        if split=='train':chosen += [r for r in anchors if r['group'] not in used][:max(0,plan['anchor_cap']-len(chosen))]
        for r in chosen:
            try:
                msgs=r['messages'];ids,mask=filter_messages(msgs,tokenizer,r['component'],plan)
                selected[split].append(dict(r,split=split,bucket='anchor-'+r['bucket'],_ids=ids,_mask=mask))
            except ValueError as e:rejected['anchor:'+str(e)]+=1
        random.Random(plan['seed']).shuffle(selected[split])
    if len(selected['train'])<3000:raise ValueError('Science corpus too small')
    groups={s:{r['group'] for r in rs} for s,rs in selected.items()}
    if any(groups[a]&groups[b] for a,b in (('train','dev'),('train','test'),('dev','test'))):raise ValueError('Science split leakage')
    stats={s:dict(conversations=len(rs),semantic_conversations=len({r['group'] for r in rs}),
                components=dict(Counter(r['component'] for r in rs)),buckets=dict(Counter(r['bucket'] for r in rs)),
                serialization_modes=dict(Counter(r.get('serialization_mode','chat') for r in rs)),
                input_tokens=sum(len(r['_ids'])-1 for r in rs),assistant_tokens=sum(int(r['_mask'].sum()) for r in rs),
                assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rs if r['component']==c) for c in {r['component'] for r in rs}})
           for s,rs in selected.items()}
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        for s,rs in selected.items():write_split(staging,s,rs)
        shutil.copyfile(root/'science-dev.json',staging/'science-dev.json')
        atomic_json(staging/'curation.json',dict(pool_sha256=file_sha256(root/'pool.json'),pool=pool,counts=stats,rejected=dict(rejected),style=plan['science_style']))
        atomic_json(staging/'review_examples.json',[{k:v for k,v in r.items() if not k.startswith('_')} for r in selected['train'][:40]])
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        result=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,
                    sources=plan['sources'],counts=stats,files=files,seed=plan['seed'],serialization='mixed-chat-and-plain-qa-masks-v1' if plan.get('native_qa_fraction') else 'explicit-ids-assistant-mask-v1',
                    epochs=plan['epochs'],final_quality_version='science-search-v1',total_steps=math.ceil(len(selected['train'])/64)*plan['epochs'])
        atomic_json(staging/'manifest.json',result);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print('[science-prepared]',json.dumps(stats),flush=True);return result


def choice_dev(model, tokenizer, prepared):
    """Frozen NON-ARC question choice likelihood; no benchmark imports needed."""
    import mlx.core as mx
    rows=read_json(Path(prepared)/'science-dev.json');correct=normalized_correct=0;scores_by_source=defaultdict(list)
    for row in rows:
        context=(row['context']+'\n' if row['context'] else '')+'Question: '+row['question']+'\nAnswer:'
        prefix=tokenizer.encode(context);scores=[]
        for choice in row['choices']:
            joint=tokenizer.encode(context+' '+choice)
            if joint[:len(prefix)]!=prefix or len(joint)>model.cfg.max_seq_len:raise ValueError('Dev choice token boundary/context')
            x=mx.array([joint[:-1]],dtype=mx.int32);y=np.full((1,len(joint)-1),-100,np.int32)
            y[0,len(prefix)-1:]=joint[len(prefix):]
            out=model(x,targets=mx.array(y));value=-float((out['token_loss']*(mx.array(y)!=-100)).sum().item());scores.append(value)
        raw=int(np.argmax(scores));norm=int(np.argmax([v/len(c) for v,c in zip(scores,row['choices'])]))
        good=raw==row['gold'];correct+=good;normalized_correct+=norm==row['gold'];scores_by_source[row['bucket']].append(good)
    return dict(evaluated=len(rows),acc=correct/len(rows),acc_norm=normalized_correct/len(rows),
                sources={k:sum(v)/len(v) for k,v in scores_by_source.items()},panel_sha256=file_sha256(Path(prepared)/'science-dev.json'),
                scope='Fixed non-ARC dev choices; checkpoint selection only; small development panel, not ARC test accuracy.')
