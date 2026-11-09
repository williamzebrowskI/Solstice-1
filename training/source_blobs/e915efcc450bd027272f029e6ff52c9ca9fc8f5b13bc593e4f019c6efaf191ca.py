"""Small, transparent chat diagnostics. These are not a general factual judge.

Greetings measure intentional training coverage. Other literal prompts are
excluded by curation; dev/test use different wording, facts and context values.
Only dev diagnostics influence checkpoint selection.
"""
import re
from collections import defaultdict
from .quality import normalized, repeated


def cases(split, suite='chat-relevance-v1'):
    if split not in ('dev', 'test'): raise ValueError('Unknown diagnostic split')
    if suite == 'instruction-v8':
        from .behavior_v8 import cases as fresh_cases
        return fresh_cases(split)
    if suite == 'instruction-v7':
        from .behavior_v7 import cases as fresh_cases
        return fresh_cases(split)
    if suite == 'instruction-v6':
        from .behavior_v6 import cases as fresh_cases
        return fresh_cases(split)
    if suite == 'conversation-v5':
        from .behavior_v5 import cases as fresh_cases
        return fresh_cases(split)
    if suite == 'conversation-v4':
        from .behavior_v4 import cases as focused_cases
        return focused_cases(split)
    if suite == 'conversation-v3':
        from .behavior_v3 import cases as fresh_cases
        return fresh_cases(split)
    def case(name, kind, prompt, expected=None, history=None):
        return dict(name=name,kind=kind,messages=[*(history or []),dict(role='user',content=prompt)],expected=expected)
    if suite == 'chat-relevance-v2':
        base=cases(split)
        extra = ([
            case('thanks-2','thanks','That explanation helped me. Thank you.'),
            case('thanks-3','thanks','I appreciate the answer.'),
            case('farewell-2','farewell','I am leaving now. See you later.'),
            case('clarify-2','clarify','Please check my work.'),
            case('clarify-3','clarify','Something is wrong. Can you help?'),
            case('literal-2','exact','Write only the word copper, with no punctuation.',['copper']),
            case('extract-2','exact','The blue box contains a scarf. The green box contains a book. Which box contains the scarf? Answer with only its color.',['blue']),
            case('fact-2','fact','How many wheels does a standard bicycle have?',['two','2']),
        ] if split=='dev' else [
            case('thanks-2','thanks','I appreciate your explanation.'),
            case('thanks-3','thanks','You answered my question. Thanks.'),
            case('farewell-2','farewell','I will go now. See you soon.'),
            case('clarify-2','clarify','I would like help with my draft.'),
            case('clarify-3','clarify','There is a problem with this. Please help.'),
            case('literal-2','exact','Write only the word silver, with no punctuation.',['silver']),
            case('extract-2','exact','The red bag holds a notebook. The yellow bag holds a pencil. Which bag holds the pencil? Answer with only its color.',['yellow']),
            case('fact-2','fact','How many legs does a typical adult insect have?',['six','6']),
        ])
        return [dict(c,suite=suite) for c in base+extra]
    if suite!='chat-relevance-v1': raise ValueError('Unknown diagnostic suite')
    if split == 'dev':
        return [case('greeting-1','greeting','hello'),case('greeting-2','greeting','Hey there!'),
            case('greeting-3','greeting','Good morning!'),
            case('thanks','thanks','Thank you for your help!'),case('farewell','farewell','Goodbye for now!'),
            case('days','fact','How many days are in a week?',['seven','7']),
            case('thermometer','fact','What does a thermometer measure?',['temperature']),
            case('triangle','fact','How many sides does a triangle have?',['three','3']),
            case('context-name','fact',"What is my dog's name?",['cedar'],[
                dict(role='user',content="My dog's name is Cedar."),dict(role='assistant',content="Got it. Your dog's name is Cedar.")]),
            case('context-extract','exact','Read this note: The package arrives on Friday. Reply with only the arrival day.',['friday']),
            case('literal-format','exact',"Reply with only the lowercase word 'ready'.",['ready']),
            case('clarification','clarify','Can you fix it for me?')]
    return [case('greeting-1','greeting','Hi!'),case('greeting-2','greeting','Hello there!'),
        case('greeting-3','greeting','Good evening!'),
        case('thanks','thanks','Thanks, that helped.'),case('farewell','farewell','Bye, see you later.'),
        case('hours','fact','How many hours make up one day?',['twenty four','24']),
        case('ruler','fact','What can you measure with a ruler?',['length','distance']),
        case('square','fact','How many sides does a square have?',['four','4']),
        case('context-name','fact',"What is my cat's name?",['juniper'],[
            dict(role='user',content="My cat's name is Juniper."),dict(role='assistant',content="Got it. Your cat's name is Juniper.")]),
        case('context-extract','exact','Read this note: The meeting is on Tuesday. Reply with only the meeting day.',['tuesday']),
        case('literal-format','exact',"Respond with just the lowercase word 'done'.",['done']),
        case('clarification','clarify','Could you improve this for me?')]


def check(case, result):
    if case.get('suite') == 'instruction-v8':
        from .behavior_v8 import check as fresh_check
        return fresh_check(case,result)
    if case.get('suite') == 'instruction-v7':
        from .behavior_v7 import check as fresh_check
        return fresh_check(case,result)
    if case.get('suite') == 'instruction-v6':
        from .behavior_v6 import check as fresh_check
        return fresh_check(case,result)
    if case.get('suite') == 'conversation-v5':
        from .behavior_v5 import check as fresh_check
        return fresh_check(case,result)
    if case.get('suite') == 'conversation-v4':
        from .behavior_v4 import check as focused_check
        return focused_check(case,result)
    if case.get('suite') == 'conversation-v3':
        from .behavior_v3 import check as strict_check
        return strict_check(case,result)
    text=result['text'].strip(); norm=normalized(text)
    clean=bool(text) and not repeated(text)
    natural=result['stop_reason'] in ('turn_end','document_end')
    bounded=len(norm.split()) <= 45
    specialized=bool(re.search(r"(?:revised version|rewritten version|first sentence|english (?:language|grammar)|your (?:teacher|tutor)|i(?: am|'m) (?:alex|sarah))",text,re.I))
    kind=case['kind']
    if kind=='greeting': relevant=bool(re.search(r'\b(?:hi|hello|hey|greetings|good morning|good evening)\b',text,re.I)) and not specialized and len(norm.split())<=30
    elif kind=='thanks': relevant=bool(re.search(r"(?:welcome|glad|happy to help|any ?time|no problem)",text,re.I)) and not specialized
    elif kind=='farewell': relevant=bool(re.search(r'\b(?:bye|goodbye|see you|take care|later)\b',text,re.I)) and not specialized
    elif kind=='exact': relevant=norm in case['expected']
    elif kind=='fact': relevant=any(re.search(r'\b'+re.escape(a)+r'\b',norm) for a in case['expected']) and bounded
    elif kind=='clarify': relevant='?' in text and bool(re.search(r'\b(?:what|which|share|send|provide|clarify|details|information)\b',text,re.I)) and not specialized
    else: raise ValueError('Unknown diagnostic kind')
    if case.get('suite')=='chat-relevance-v2':
        if kind=='thanks': relevant=relevant and bounded and not re.search(r'\b(?:hello|math problem|what.s your problem)\b',text,re.I)
        if kind=='farewell': relevant=relevant and len(norm.split())<=20 and not re.search(r'\b(?:lesson|japanese|how can i help|what would you like)\b',text,re.I)
        if kind=='fact':
            expected_numbers={'days':7,'triangle':3,'hours':24,'square':4,'fact-2':2 if 'bicycle' in case['messages'][-1]['content'] else 6}
            if case['name'] in expected_numbers:
                numbers=[int(n) for n in re.findall(r'\b\d+\b',norm)]
                relevant=relevant and all(n==expected_numbers[case['name']] for n in numbers)
    return dict(relevant=bool(relevant),natural_stop=natural,nonrepetitive_nonempty=clean,
                passed=bool(relevant and clean and natural))


def run(model, tokenizer, plan, split='dev'):
    if plan.get('behavior_suite') == 'instruction-v8':
        from .behavior_v8 import run as fresh_run
        return fresh_run(model,tokenizer,plan,split)
    if plan.get('behavior_suite') == 'instruction-v7':
        from .behavior_v7 import run as fresh_run
        return fresh_run(model,tokenizer,plan,split)
    if plan.get('behavior_suite') == 'instruction-v6':
        from .behavior_v6 import run as fresh_run
        return fresh_run(model,tokenizer,plan,split)
    if plan.get('behavior_suite') == 'conversation-v5':
        from .behavior_v5 import run as fresh_run
        return fresh_run(model,tokenizer,plan,split)
    from .evaluate import generate
    samples=[]; rates=defaultdict(list)
    for index,case in enumerate(cases(split,plan.get('behavior_suite','chat-relevance-v1'))):
        # Cold start and generic system instruction must both work; a sampled
        # check catches the instability observed in the first SFT run.
        for system,temperature in ((False,0.),(True,0.),(True,.7)):
            messages=case['messages']
            if system: messages=[dict(role='system',content=plan['conversation_system']),*messages]
            result=generate(model,tokenizer,messages,limit=80,temperature=temperature,seed=7337+index)
            checks=check(case,result); rates[case['kind']].append(checks['passed'])
            samples.append(dict(name=case['name'],kind=case['kind'],messages=messages,temperature=temperature,**result,checks=checks))
    means={kind:sum(values)/len(values) for kind,values in rates.items()}
    return dict(split=split,rates=means,macro_pass=sum(means.values())/len(means),samples=samples,
        scope='Constrained whole-answer QA, exact casing, grounded abstention, context correction and bounded pure-function checks; social/clarification checks remain heuristics. Raw replies require review.' if plan.get('behavior_suite') in ('conversation-v3','conversation-v4') else 'Coverage greetings plus excluded-prompt direct QA, context, exact format and clarification diagnostics; heuristic checks require human review of raw replies.')


def selection(result,plan):
    if not plan.get('behavior_suite'): return [result['selection_score'],-result['assistant_nll']]
    chat=result['chat_behavior']['macro_pass']
    retention=result.get('retention')
    if retention is None: raise ValueError('Chat selection requires current checkpoint retention')
    retained=retention['val_loss'] <= plan['parent_val_loss']+plan['max_retention_increase']
    if plan.get('continuation_trials'):
        from .continuation_trials import ranking
        return ranking(result,plan)
    if plan.get('natural_transfer'):
        from .natural_transfer import ranking
        return ranking(result,plan)
    if plan.get('evidence_instructions'):
        panel=result['evidence_dev'];generated=panel['macro_pass']
        covered=all(panel['rates'][b]>=.5 for b in ('evidence','dialogue','format','python'))
        return [int(retained),int(retained and covered),generated,chat,-result['assistant_nll']]
    if plan.get('verified_extension'):
        generated=result['verified_dev']['macro_pass']
        return [int(retained),int(retained and generated>=plan.get('min_verified_dev_pass',.75)),generated,chat,-result['assistant_nll']]
    if plan.get('science_search'):
        # Fixed independent science development panel, never ARC test scores.
        # Chat eligibility is reported separately by the experiment run report.
        return [int(retained), result['science_dev']['acc'], chat, -result['assistant_nll']]
    greeting=result['chat_behavior']['rates'].get('greeting',0.)
    covered=all(rate>=plan.get('min_category_pass',0.) for rate in result['chat_behavior']['rates'].values())
    eligible=retained and covered and chat >= plan['min_chat_pass'] and greeting >= plan.get('min_greeting_pass',.8)
    if plan.get('contrast_instructions'):
        contrast=result['contrast_behavior']
        eligible=eligible and contrast['macro_pair_pass']>=plan['min_contrast_pair_pass'] and all(v>=plan['min_category_pass'] for v in contrast['rates'].values())
        return [int(retained),int(eligible),contrast['macro_pair_pass'],chat,result['selection_score'],-result['assistant_nll']]
    if plan.get('short_chat_focus'):
        return [int(retained),int(eligible),chat,result['selection_score'],-result['assistant_nll']]
    return [int(eligible),chat,result['selection_score'],-result['assistant_nll']]


def candidate_eligible(key,plan):
    """Eligibility is distinct from the retention-first ranking flag."""
    return bool(key[1] if plan.get("short_chat_focus") else key[0])


def compact_evaluation(result):
    """Scores fit the bounded control channel; raw replies stay on disk."""
    summary={k:v for k,v in result.items() if k!='samples'}
    if 'trial_dev' in summary:
        summary['trial_dev']=dict(summary['trial_dev'],metrics={
            v:{k:x for k,x in panel.items() if k!='samples'}
            for v,panel in summary['trial_dev']['metrics'].items()})
    if 'chat_behavior' in summary:
        summary['chat_behavior']={k:v for k,v in summary['chat_behavior'].items() if k!='samples'}
    if 'contrast_behavior' in summary:
        summary['contrast_behavior']={k:v for k,v in summary['contrast_behavior'].items() if k!='samples'}
    if 'verified_dev' in summary:
        summary['verified_dev']={k:v for k,v in summary['verified_dev'].items() if k!='samples'}
    if 'evidence_dev' in summary:
        summary['evidence_dev']={k:v for k,v in summary['evidence_dev'].items() if k!='samples'}
    if 'natural_dev' in summary:
        summary['natural_dev']=dict(summary['natural_dev'],metrics={
            v:{k:x for k,x in panel.items() if k!='samples'}
            for v,panel in summary['natural_dev']['metrics'].items()})
    return summary
