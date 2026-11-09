"""Pinned HF sources -> filtered, grouped splits -> immutable token arrays."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import shutil
import time
import uuid
import numpy as np
from datasets import load_dataset, load_from_disk
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sml_v1.tokenization import Tokenizer
from .data import serialize
from .quality import Deduplicator, normalized, digest, repeated, constraint_rules, check_rules, python_training_target, social_prompt, direct_general

PROJECT = Path(__file__).resolve().parents[2]
def local(path): return (PROJECT/path).resolve()


def source_rows(source, cache):
    print(f'[download] {source["repo"]}/{source["config"]}@{source["revision"]} {source["split"]}', flush=True)
    return load_dataset(source['repo'], source['config'], revision=source['revision'],
                        split=source['split'], cache_dir=str(cache))


def blocked_questions(cache):
    # All ARC splits are excluded, not only benchmark test answers.
    source = dict(repo='allenai/ai2_arc', revision='210d026faf9955653af8916fad021475a3f00453')
    hashes = set(); receipts = []
    for config in ('ARC-Easy', 'ARC-Challenge'):
        for split in ('train','validation','test'):
            spec = dict(source, config=config, split=split)
            rows = source_rows(spec, cache)
            for row in rows: hashes.add(digest(row['question']))
            receipts.append(dict(spec, rows=len(rows)))
    # Existing local snapshots preserve the original benchmark revisions.
    for name,field in (('piqa','goal'),('hellaswag','ctx')):
        path=PROJECT/'reports/benchmarks/research-best-20261005/datasets'/name
        rows=load_from_disk(str(path))
        for row in rows: hashes.add(digest(row[field]))
        receipts.append(dict(snapshot=str(path.relative_to(PROJECT)),files={p.name:file_sha256(p) for p in path.iterdir() if p.is_file()},rows=len(rows)))
    sources=[
        ('google/IFEval','default','train','966cd89545d6b6acfd7638bc708b98261ca58e84','prompt'),
        ('openai/gsm8k','main','test','740312add88f781978c0658806c59bc2815b9866','question'),
        ('cais/mmlu','all','test','c30699e8356da336a370243923dbaf21066bb9fe','question'),
        ('openai/openai_humaneval','openai_humaneval','test','7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544','prompt'),
        ('google-research-datasets/mbpp','sanitized','test','4bb6404fdc6cacfda99d4ac4205087b89d32030c','prompt')]
    for repo,config,split,revision,field in sources:
        spec=dict(repo=repo,config=config,split=split,revision=revision)
        rows=source_rows(spec,cache)
        for row in rows: hashes.add(digest(row[field]))
        receipts.append(dict(spec,rows=len(rows)))
    return hashes, receipts


def convert(row, source):
    component = source['component']; extra = {}
    if component == 'qa':
        context, question = row['context'], row['question']
        answerable = bool(row['answers']['text'])
        if answerable:
            answer = row['answers']['text'][0]; start = row['answers']['answer_start'][0]
            if context[start:start+len(answer)] != answer: raise ValueError('answer_span')
        else: answer = 'The passage does not provide enough information to answer.'
        messages = [dict(role='user',content='Answer using only the passage. If it does not contain the answer, say that the passage does not provide enough information to answer.\n\nPassage:\n'+context+'\n\nQuestion: '+question),dict(role='assistant',content=answer)]
        group = 'wikipedia:'+row['title']
        extra.update(answerable=answerable, answers=row['answers']['text'], title=row['title'], id=row['id'])
    elif component == 'python':
        messages = [dict(role='user',content=row['instruction']),dict(role='assistant',content=row['response'])]
        group = 'python:'+digest(str(row.get('seed',row['instruction'])))
        extra['seed'] = row.get('seed'); extra['concepts'] = row.get('concepts')
        if source['revision']=='356bb069eee815daa6e23e9a282eeefe1490ad44' and row['instruction'].startswith('Write a Python function `string_concat_fn`'): raise ValueError('reviewed_semantic_error')
        if not python_training_target(row['instruction'],row['response']): raise ValueError('python_complex_or_invalid')
    else:
        if source.get('select_source') and row.get('source') != source['select_source']: raise ValueError('unselected_source')
        messages = [dict(role=m['role'],content=m['content']) for m in row['messages']]
        if source.get('chat_redo'):
            # Keep all complete everyday turns; discard an unanswered trailing
            # user turn. General task examples train the first direct answer.
            while messages and messages[-1]['role'] != 'assistant': messages.pop()
            if component == 'general': messages = messages[:2]
            if not messages: raise ValueError('no_complete_answer')
            if component=='general' and source.get('general_direct_only') and not direct_general(messages): raise ValueError('not_direct_general_qa')
            users = [m['content'] for m in messages if m['role']=='user']
            if component == 'general':
                prompt=' '.join(users)
                if re.search(r'\b(?:rewrite|rephrase|proofread|role.?play|act as|pretend|impersonate|essay|edit this|imagine you|you are an?|what would your name|your persona)\b|\bplay (?:an?|the)\b',prompt,re.I): raise ValueError('general_task_bias')
                if re.search(r'```|\b(?:python|javascript|typescript|java|sql|html|code|programming|algorithm|latex|equation|calculus)\b',prompt,re.I): raise ValueError('general_specialist_task')
                if len(prompt.split())>120: raise ValueError('complex_general_prompt')
            if component == 'chat':
                if re.search(r'\b(?:currently|right now|latest|medical advice|diagnos|investment|stocks)\b', ' '.join(users),re.I): raise ValueError('time_sensitive_or_specialist_chat')
                group_text = '\n'.join(p for p in users if not social_prompt(p))
                extra['has_opening_greeting'] = bool(users and social_prompt(users[0]))
            else: group_text = '\n'.join(users)
        else: group_text = next((m['content'] for m in messages if m['role']=='user'),'')
        group = 'dialogue:'+digest(group_text)
        if component == 'constraints':
            if len(messages) != 2: raise ValueError('constraint_multi_turn')
            rules = constraint_rules(messages[0]['content'])
            if rules is None: raise ValueError('unsupported_constraint')
            if not check_rules(messages[1]['content'], rules): raise ValueError('failed_constraint')
            extra['rules'] = rules
    return messages, group, extra


def filter_messages(messages, tokenizer, component, plan=None):
    redo = bool(plan and plan.get('chat_redo'))
    if len(messages) > (11 if redo else 7): raise ValueError('too_many_turns')
    for i, message in enumerate(messages):
        text = message['content']
        if not text.strip() or '\x00' in text or '�' in text: raise ValueError('invalid_text')
        if len(text) > 16000: raise ValueError('too_long_text')
        if message['role'] == 'assistant':
            ids = tokenizer.encode(text)
            maximum = 512 if component == 'python' else (plan.get('assistant_max_tokens',256) if redo else 256)
            if len(ids) > maximum or (not redo and component not in ('qa','constraints') and len(ids)<16): raise ValueError('answer_length')
            if repeated(text): raise ValueError('repetition')
            if re_identity(text): raise ValueError('identity_claim')
            if redo and re.search(r"\b(?:i(?: am|'m) (?:alex|sarah|a teacher|your (?:english|grammar) tutor)|as (?:your|an?) (?:english|grammar) (?:teacher|tutor))\b",text,re.I): raise ValueError('invented_specialist_identity')
            if redo and component in ('chat','general') and re.search(r"(?i:\b(?:my name(?:'s| is)|i(?:'m| am)))\s+[A-Z][a-z]{2,}(?=[.,!]| and| the|$)",text): raise ValueError('invented_personal_identity')
            if redo and component in ('general','chat') and re.search(r"(?:here(?:'s| is) a (?:revised|rewritten) version|i (?:changed|rewrote|revised)|first sentence about)",text,re.I): raise ValueError('unwanted_editing_template')
            prompt = normalized(messages[i-1]['content']) if i else ''
            answer = normalized(text)
            if len(answer) >= 80 and answer in prompt: raise ValueError('answer_in_prompt')
    ids, mask = serialize(tokenizer, messages)
    if len(ids)>2048: raise ValueError('context_length')
    return ids, mask


def re_identity(text):
    import re
    return bool(re.search(r'\b(?:as an ai|as a language model|i am (?:chatgpt|claude)|i was (?:created|developed) by)\b',text,re.I))


def write_split(path, name, rows):
    offsets=[0]; ids=[]; masks=[]
    with (path/f'{name}.jsonl').open('w') as f:
        for row in rows:
            tokens, mask = row.pop('_ids'), row.pop('_mask')
            ids.append(tokens); masks.append(mask); offsets.append(offsets[-1]+len(tokens))
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    np.save(path/f'{name}.tokens.npy',np.concatenate(ids) if ids else np.array([],np.int32),allow_pickle=False)
    np.save(path/f'{name}.mask.npy',np.concatenate(masks) if masks else np.array([],np.bool_),allow_pickle=False)
    np.save(path/f'{name}.offsets.npy',np.asarray(offsets,np.uint64),allow_pickle=False)


def prepare(config_path):
    plan=read_json(config_path); destination=local(plan['prepared']); cache=destination.parent/'.sft_hf_cache'
    tokenizer=Tokenizer(local(plan['tokenizer']))
    if destination.exists():
        manifest=verify(destination)
        if manifest['plan_sha256']!=file_sha256(config_path): raise ValueError('Prepared plan changed; use a new destination')
        print(json.dumps(manifest['counts'],indent=2)); return manifest
    with lock(destination.with_suffix('.lock')):
        blocked, benchmark_receipts=blocked_questions(cache)
        if plan.get('behavior_suite'):
            from .behavior import cases
            blocked.update(digest(m['content']) for split in ('dev','test') for case in cases(split)
                           for m in case['messages'] if m['role']=='user' and not social_prompt(m['content']))
        dedup=Deduplicator(blocked); pools=defaultdict(list); stats={}; reviews={}
        for source in plan['sources']:
            rejected=Counter(); component=source['component']; rows=source_rows(source,cache)
            for index,row in enumerate(rows):
                try:
                    messages,group,extra=convert(row,dict(source,chat_redo=plan.get('chat_redo',False),general_direct_only=plan.get('general_direct_only',False)))
                    if plan.get('chat_redo') and component in ('chat','general','qa') and int(digest(group)[:8],16)%2:
                        messages=[dict(role='system',content=plan['conversation_system']),*messages]
                    ids,mask=filter_messages(messages,tokenizer,component,plan)
                    if not dedup.accept(messages,component): raise ValueError('overlap_or_duplicate')
                    split_bucket=int(fingerprint({'seed':plan['seed'],'group':group})[:8],16)%10000
                    split='test' if split_bucket<100 else 'dev' if split_bucket<200 else 'train'
                    pools[component].append(dict(component=component,group=group,split=split,messages=messages,
                        provenance=dict(repo=source['repo'],config=source['config'],revision=source['revision'],split=source['split'],row=index,license=source['license']),
                        **extra,_ids=ids,_mask=mask))
                except ValueError as e: rejected[str(e)]+=1
                if (index+1)%25000==0: print(f'[curate] {component} {index+1}/{len(rows)} retained={len(pools[component])}',flush=True)
            stats[component]=dict(scanned=len(rows),accepted=len(pools[component]),rejected=dict(rejected))
            reviews[component]=[{k:v for k,v in r.items() if not k.startswith('_')} for r in random.Random(plan['seed']).sample(pools[component],min(25,len(pools[component])))]
            print(f'[accepted] {component} {stats[component]}',flush=True)
        # Keep the relative sizes of the main components. Scarce mechanically
        # verifiable constraints are added without duplication or weak targets.
        weights={s['component']:s['weight'] for s in plan['sources']}
        # The cap is a ceiling, never duplicate data to fill a quota.
        selected=defaultdict(list)
        train_available={c:sum(r['split']=='train' for r in pools[c]) for c in weights}
        units=min(plan['max_conversations']*.98/100,
                  *(train_available[c]/weights[c] for c in weights if c!='constraints'))
        for component in weights:
            rng=random.Random(plan['seed']); candidates=pools[component]; rng.shuffle(candidates)
            for split in ('train','dev','test'):
                subset=[r for r in candidates if r['split']==split]
                target=int(units*weights[component]*(1 if split=='train' else .01/.98))
                # Small writing shares must still have enough held-out examples
                # for their diagnostics; increasing dev/test counts never adds
                # training examples or duplicates a source conversation.
                if plan.get('chat_redo') and split != 'train': target=max(8,target)
                if component=='qa':
                    yes=[r for r in subset if r['answerable']]; no=[r for r in subset if not r['answerable']]
                    n=min(target, len(yes)*5//4,len(no)*5)
                    subset=yes[:n*4//5]+no[:n-n*4//5]
                else: subset=subset[:target]
                selected[split].extend(subset)
        held_out={c:sum(r['component']==c for r in selected['dev']) for c in weights}
        if any(n<8 for n in held_out.values()): raise ValueError('Insufficient component hold-out: '+json.dumps(held_out))
        if len(selected['train']) < 10000: raise ValueError('Too few clean training conversations')
        counts={s:dict(conversations=len(rows),components=dict(Counter(r['component'] for r in rows)),
                    input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),
                    assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in weights},
                    multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows)) for s,rows in selected.items()}
        staging=destination.with_name('.'+destination.name+'-'+uuid.uuid4().hex); staging.mkdir(parents=True)
        try:
            for name,rows in selected.items(): write_split(staging,name,rows)
            atomic_json(staging/'review_examples.json',reviews)
            atomic_json(staging/'curation.json',dict(filters=stats,benchmark_exclusions=benchmark_receipts,
                requested_weights=weights,counts=counts,limitations=['Constraint verification accepts only fully supported formatting rules.',
                'Synthetic general/writing factual correctness has not been exhaustively human reviewed.',
                'Pinned benchmark question exclusion and prompt shingle checks cannot establish complete semantic decontamination or undo pre-training overlap.',
                'Shortage of clean examples reduces a component share; examples are never repeated to fill quotas.']))
            files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
            manifest=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config_path),tokenizer=tokenizer.fingerprint,
                sources=plan['sources'],counts=counts,files=files,seed=plan['seed'],serialization='explicit-ids-assistant-mask-v1',
                total_steps=math.ceil(len(selected['train'])/(sum(plan['batches'])*plan['grad_accum']))*plan.get('epochs',1))
            atomic_json(staging/'manifest.json',manifest); staging.rename(destination)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    print(json.dumps(manifest['counts'],indent=2),flush=True); return manifest


def verify(path):
    path=Path(path); manifest=read_json(path/'manifest.json')
    if manifest['format']!='opensml-sft-corpus-v1': raise ValueError('Corpus format')
    for name,info in manifest['files'].items():
        if Path(name).name!=name or (path/name).is_symlink() or (path/name).stat().st_size!=info['bytes'] or file_sha256(path/name)!=info['sha256']: raise ValueError('Corpus integrity: '+name)
    return manifest


def derive_reviewed(config):
    """Further review a frozen HF-derived draft in a NEW destination only.

    Preserve original provenance and split membership; no targets are rewritten
    and no discarded examples are replaced from a different split.
    """
    plan=read_json(config); origin=local(plan['derive_from']); destination=local(plan['prepared'])
    if destination.exists():
        result=verify(destination)
        if result['plan_sha256']!=file_sha256(config): raise ValueError('Derived plan changed')
        return result
    old=verify(origin)
    if old['sources']!=plan['sources']: raise ValueError('Derived source revisions changed')
    tokenizer=Tokenizer(local(plan['tokenizer']))
    if old['tokenizer']!=tokenizer.fingerprint: raise ValueError('Derived tokenizer changed')
    from .data import Corpus
    selected={}; counts={}; removed={}
    for split in ('train','dev','test'):
        corpus=Corpus(origin,split); rows=[]; rejected=Counter(); used=Counter()
        for i,line in enumerate((origin/f'{split}.jsonl').read_text().splitlines()):
            row=json.loads(line); component=row['component']
            if component=='general' and not direct_general(row['messages']): rejected['not_direct_general_qa']+=1; continue
            cap=plan.get('train_component_caps',{}).get(component)
            if split=='train' and cap is not None and used[component]>=cap: rejected[component+'_cap']+=1; continue
            used[component]+=1
            a,b=map(int,corpus.offsets[i:i+2]); row['_ids']=corpus.ids[a:b]; row['_mask']=corpus.mask[a:b]; rows.append(row)
        selected[split]=rows; removed[split]=dict(rejected)
        counts[split]=dict(conversations=len(rows),components=dict(used),input_tokens=sum(len(r['_ids'])-1 for r in rows),
            assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in used},
            multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows))
    if counts['train']['conversations']<10000 or any(counts[s]['components'].get(source['component'],0)<8 for s in ('dev','test') for source in plan['sources']): raise ValueError('Insufficient reviewed corpus: '+json.dumps(counts))
    destination.mkdir(parents=True)
    try:
        reviews={c:[{k:v for k,v in r.items() if not k.startswith('_')} for r in selected['train'] if r['component']==c][:25] for c in counts['train']['components']}
        for split,rows in selected.items(): write_split(destination,split,rows)
        atomic_json(destination/'review_examples.json',reviews)
        report=read_json(origin/'curation.json'); report.update(counts=counts,reviewed_derivation=dict(origin=str(origin),manifest_sha256=file_sha256(origin/'manifest.json'),removed=removed,policy='Direct-question/request general subset; exclude persona/editing/specialist prompts; deterministic small writing/Python caps; unchanged answers and grouped splits.'))
        atomic_json(destination/'curation.json',report)
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in destination.iterdir()}
        result=dict(old,plan_sha256=file_sha256(config),counts=counts,files=files,total_steps=math.ceil(counts['train']['conversations']/64)*plan.get('epochs',1),epochs=plan.get('epochs',1),final_quality_version='sft-chat-review-v3')
        atomic_json(destination/'manifest.json',result)
    except Exception:
        shutil.rmtree(destination); raise
    return result


def refine(directory,config):
    """Final stronger checks on the derived corpus before it is frozen/staged."""
    directory=Path(directory)
    if (directory/'prepared.json').exists(): return verify(directory)
    manifest=verify(directory)
    if manifest.get('final_quality_version') in ('sft-quality-v2','sft-chat-review-v3','balanced-chat-v1','conversation-v2','science-search-v1','short-chat-focus-v1','reliable-chat-v1','instruction-chat-v1','reviewed-instructions-v1','balanced-instructions-v2','contrast-instructions-v1','chat-search-v1'): return manifest
    tokenizer=Tokenizer(local(read_json(config)['tokenizer']))
    rows_by_split={}; removals=Counter(); counts={}
    from .data import Corpus
    for split in ('train','dev','test'):
        corpus=Corpus(directory,split); accepted=[]
        for i,line in enumerate((directory/f'{split}.jsonl').read_text().splitlines()):
            row=json.loads(line); messages=row['messages']; component=row['component']
            if component=='constraints':
                rules=constraint_rules(messages[0]['content'])
                if rules is None or not check_rules(messages[-1]['content'],rules): removals['unsupported_or_failed_constraint']+=1; continue
                row['rules']=rules
            elif component=='python':
                if messages[0]['content'].startswith('Write a Python function `string_concat_fn`') or not python_training_target(messages[0]['content'],messages[-1]['content']): removals['python_complex_or_example_error']+=1; continue
            a,b=map(int,corpus.offsets[i:i+2]); row['_ids']=corpus.ids[a:b]; row['_mask']=corpus.mask[a:b]; accepted.append(row)
        rows_by_split[split]=accepted
        counts[split]=dict(conversations=len(accepted),components=dict(Counter(r['component'] for r in accepted)),
            input_tokens=sum(len(r['_ids'])-1 for r in accepted),assistant_tokens=sum(int(r['_mask'].sum()) for r in accepted),
            assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in accepted if r['component']==c) for c in manifest['counts'][split]['components']},
            multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in accepted))
    if any(counts['dev']['components'].get(s['component'],0)<8 for s in manifest['sources']): raise ValueError('Refinement left too few component dev examples')
    staging=directory.with_name('.'+directory.name+'-refined-'+uuid.uuid4().hex); staging.mkdir()
    try:
        for split,rows in rows_by_split.items(): write_split(staging,split,rows)
        report=read_json(directory/'curation.json'); report.update(final_refinement=dict(version='sft-quality-v2',removed=dict(removals),checks='Complete supported constraint suffix; simple Python AST and literal numeric example checks; no dataset code execution.'),counts=counts)
        atomic_json(staging/'curation.json',report)
        reviews={c:[] for c in counts['train']['components']}
        for line in (staging/'train.jsonl').read_text().splitlines():
            row=json.loads(line)
            if len(reviews[row['component']])<25: reviews[row['component']].append(row)
        atomic_json(staging/'review_examples.json',reviews)
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        manifest.update(counts=counts,files=files,total_steps=math.ceil(counts['train']['conversations']/64)*read_json(config).get('epochs',1),final_quality_version='sft-quality-v2')
        atomic_json(staging/'manifest.json',manifest)
        backup=directory.with_name(directory.name+'_before_refinement')
        if backup.exists(): raise ValueError('Refinement backup already exists')
        directory.rename(backup); staging.rename(directory)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    print('[quality-v2] '+json.dumps(dict(removed=dict(removals),counts=counts)),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    prepare(parser.parse_args().config)
