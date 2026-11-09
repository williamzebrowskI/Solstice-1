"""Component NLL, bounded chat checks, and separate pre-training retention."""
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
import mlx.core as mx
from .data import Corpus, serialize
from .quality import normalized, repeated, check_rules, python_target


def conversation_losses(model, x, y):
    outputs = model(x, targets=y)
    mask = y != -100
    counts = mask.sum(axis=1)
    per_row = (outputs['token_loss']*mask).sum(axis=1)/mx.maximum(counts,1)
    return per_row, counts


def loss_sum(model, x, y, *, token_weighted=False):
    losses, counts = conversation_losses(model,x,y)
    return (losses * counts).sum() if token_weighted else losses.sum()


def generate(model, tokenizer, messages, limit=256, temperature=0., seed=7337):
    if temperature: mx.random.seed(seed)
    ids,_ = serialize(tokenizer,messages,generation=True)
    limit=min(limit,model.cfg.max_seq_len-len(ids))
    if limit < 1: raise ValueError('No generation space')
    inputs=mx.array(ids[None]); caches=None; output=[]; reason='token_limit'; stop_token=None
    for _ in range(limit):
        logits,caches=model.logits(inputs,caches)
        token=int((mx.random.categorical(logits[0,-1].astype(mx.float32)/temperature) if temperature else mx.argmax(logits[0,-1])).item())
        if token in (1,3): reason='turn_end' if token==3 else 'document_end'; stop_token=token; break
        if token in (0,2): reason='unexpected_structure'; stop_token=token; break
        output.append(token); inputs=mx.array([[token]],dtype=mx.int32)
    return dict(text=tokenizer.decode(output),tokens=len(output),stop_reason=reason,token_ids=output,stop_token_id=stop_token)


def evaluate(model, tokenizer, prepared, plan, *, step, output, retention=False):
    corpus=Corpus(prepared,'dev')
    records=[json.loads(line) for line in (Path(prepared)/'dev.jsonl').read_text().splitlines()]
    indices=defaultdict(list)
    for i,row in enumerate(records): indices[row['component']].append(i)
    nll={}; component_tokens={}; behavior=[]; scores=defaultdict(list)
    model.eval()
    try:
        for component,positions in indices.items():
            values=[]; tokens=0
            for i in positions[:plan['eval_examples_per_component']]:
                x,y=corpus.batch([i]); losses,counts=conversation_losses(model,mx.array(x),mx.array(y))
                value=float(losses.item()); tokens+=int(counts.item())
                if not math.isfinite(value): raise RuntimeError('Nonfinite SFT validation')
                values.append(value)
            nll[component]=sum(values)/len(values); component_tokens[component]=tokens
            for i in positions[:plan['behavior_examples_per_component']]:
                row=records[i]; result=generate(model,tokenizer,row['messages'][:-1])
                natural=result['stop_reason'] in ('turn_end','document_end')
                clean=not repeated(result['text']) and bool(result['text'].strip())
                scores['natural_stop'].append(natural); scores['nonrepetitive_nonempty'].append(clean)
                checks=dict(natural_stop=natural,nonrepetitive_nonempty=clean)
                if component=='qa':
                    expected=row['answers'] if row['answerable'] or plan.get('reviewed_instructions') else ['The passage does not provide enough information to answer.']
                    checks['qa_exact']=normalized(result['text']) in {normalized(a) for a in expected}
                    scores['qa_exact'].append(checks['qa_exact'])
                elif component=='coqa':
                    expected=normalized(row['messages'][-1]['content']).split()
                    actual=normalized(result['text']).split()
                    from collections import Counter
                    common=sum((Counter(expected)&Counter(actual)).values())
                    checks['coqa_exact']=actual==expected
                    checks['coqa_token_f1']=2*common/(len(expected)+len(actual)) if expected or actual else 1.
                    scores['coqa_exact'].append(checks['coqa_exact'])
                    scores['coqa_token_f1'].append(checks['coqa_token_f1'])
                elif component=='constraints':
                    checks['constraint_pass']=check_rules(result['text'],row['rules']) and clean
                    scores['constraint_pass'].append(checks['constraint_pass'])
                elif component=='python':
                    checks['python_syntax_only']=python_target(result['text'])
                    if plan.get('reviewed_instructions') and row.get('functional_assertions'):
                        from .behavior_v7 import python_checks
                        if row['messages'][-1]['content'].startswith('```python'):
                            checks.update(python_checks(result['text'],row['functional_assertions'],**row['python_contract']))
                            for k in ('python_functional','python_name','python_format','python_docstring'):
                                scores[k].append(checks[k])
                    elif (plan.get('reliable_chat') or plan.get('instruction_chat')) and row.get('functional_assertions'):
                        from .bounded_python import verify_target
                        # Authored code+follow-up rows have a final text answer;
                        # evaluate code on code-ending component rows only.
                        if row['messages'][-1]['content'].startswith('```python'):
                            checks['python_functional']=verify_target('',result['text'],tests=row['functional_assertions'],strict=True) is not None
                        else:
                            checks['python_functional']=None
                        if checks['python_functional'] is not None:scores['python_functional'].append(checks['python_functional'])
                    elif plan.get('conversation_v2') and row.get('functional_assertions'):
                        from .pure_python import verify_target
                        instruction=next(m['content'] for m in row['messages'] if m['role']=='user')
                        checks['python_functional']=verify_target(instruction,result['text']) is not None
                        scores['python_functional'].append(checks['python_functional'])
                behavior.append(dict(component=component,messages=row['messages'][:-1],reference=row['messages'][-1]['content'],**result,checks=checks))
            print(f'[eval-sft {step}] {component} assistant_nll={nll[component]:.4f}',flush=True)
        rates={k:sum(v)/len(v) for k,v in scores.items()}
        result=dict(step=step,assistant_nll=sum(nll.values())/len(nll),components=nll,supervised_tokens=component_tokens,
                    behavior_rates=rates,selection_score=sum(rates.values())/len(rates),
                    scope='Dev-only automatic formatting/QA/termination checks; not a full chatbot correctness assessment.')
        if plan.get('behavior_suite'):
            from .behavior import run
            result['chat_behavior']=run(model,tokenizer,plan)
            print(f'[eval-chat {step}] macro_pass={result["chat_behavior"]["macro_pass"]:.3f} rates={result["chat_behavior"]["rates"]}',flush=True)
        if plan.get('contrast_instructions'):
            from .contrast_eval import run as contrast_run
            result['contrast_behavior']=contrast_run(model,tokenizer,prepared,'dev')
            panel=result['contrast_behavior']
            print(f'[eval-contrast {step}] macro_pair_pass={panel["macro_pair_pass"]:.3f} pairs={panel["pair_count"]} rates={panel["rates"]}',flush=True)
        if plan.get('science_search'):
            from .science import choice_dev
            result['science_dev']=choice_dev(model,tokenizer,prepared)
            print(f'[eval-science {step}] acc={result["science_dev"]["acc"]:.4f} acc_norm={result["science_dev"]["acc_norm"]:.4f}',flush=True)
        if plan.get('verified_extension'):
            from .verified_extension import development
            result['verified_dev']=development(model,tokenizer,prepared,plan)
            print(f'[eval-verified {step}] macro_pass={result["verified_dev"]["macro_pass"]:.3f} rates={result["verified_dev"]["rates"]}',flush=True)
        if plan.get('evidence_instructions'):
            from .evidence_instructions import development
            result['evidence_dev']=development(model,tokenizer,prepared,plan)
            panel=result['evidence_dev']
            print(f'[eval-whole-exchange {step}] macro_pass={panel["macro_pass"]:.3f} rates={panel["rates"]}',flush=True)
        if plan.get('natural_transfer'):
            from .natural_transfer import development
            result['natural_dev']=development(model,tokenizer,prepared,plan)
            if plan.get('selection_reference'):
                result['selection_reference']=plan['selection_reference']
            print(f'[eval-transfer {step}] macro_pass={result["natural_dev"]["macro_pass"]:.3f}',flush=True)
        if plan.get('continuation_trials'):
            from .continuation_trials import run as trial_run
            result['trial_dev']=trial_run(model,tokenizer,prepared,plan)
            print(f'[eval-trial-dev {step}] macro_pass={result["trial_dev"]["macro_pass"]:.3f}',flush=True)
        if retention:
            receipt=json.loads((Path(prepared)/'retention.json').read_text())
            losses={}
            with np.load(Path(prepared)/'retention.npz',allow_pickle=False) as panel:
                for component in receipt['weights']:
                    values=[]
                    for x,y in zip(panel[component+'_x'],panel[component+'_y']):
                        values.append(float(model(mx.array(x),targets=mx.array(y))['loss'].item()))
                    losses[component]=sum(values)/len(values)
            result['retention']=dict(val_loss=sum(losses[c]*w for c,w in receipt['weights'].items())/sum(receipt['weights'].values()),sources=losses,
                validation_fingerprint=receipt['validation_fingerprint'],scope='Unchanged Stage A document validation; separate from SFT assistant NLL.')
            print(f'[eval-retention {step}] {result["retention"]}',flush=True)
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        if plan.get('behavior_suite'):
            from .behavior import selection, candidate_eligible
            result['selection_key']=selection(result,plan)
            if plan.get('natural_transfer'):
                from .natural_transfer import eligibility
                result['selection_checks']=eligibility(result,plan)
            if plan.get('science_search'):
                rates=result['chat_behavior']['rates']
                result['chat_candidate_eligible']=bool(result['selection_key'][0] and result['chat_behavior']['macro_pass']>=plan['min_chat_pass'] and all(v>=plan['min_category_pass'] for v in rates.values()))
            else:
                result['chat_candidate_eligible']=candidate_eligible(result['selection_key'],plan)
        from sml_v1.common import atomic_json
        atomic_json(output/f'eval_{step:07d}.json',dict(**result,samples=behavior))
        from .behavior import compact_evaluation
        return compact_evaluation(result)
    finally: model.train()
