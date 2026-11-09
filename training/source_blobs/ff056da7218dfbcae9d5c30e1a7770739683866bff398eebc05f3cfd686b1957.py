"""Conversation intent regressions plus the prior factual/context/Python panel."""
import re
from .behavior_v3 import cases as prior_cases,check as prior_check
from .quality import normalized,repeated


def cases(split):
    rows=prior_cases(split)
    # User-observed wording is a development regression, not an unseen test.
    prompts=(['whats up?','whats up with you','what can you help me with'] if split=='dev'
             else ['Anything happening with you?','Are you up for a chat?','What sorts of things could you assist me with?'])
    for i,prompt in enumerate(prompts):
        rows.append(dict(name=f'intent-{i}',kind='capability' if i==2 else 'casual',
            messages=[dict(role='user',content='Hello'),dict(role='assistant',content='Hello! How can I help you today?'),dict(role='user',content=prompt)],
            suite='conversation-v4',observed_regression=split=='dev'))
    return rows


def check(case,result):
    if case['kind'] not in ('casual','capability'):return prior_check(case,result)
    text=result['text'].strip();norm=normalized(text)
    clean=bool(text) and not repeated(text);natural=result['stop_reason'] in ('turn_end','document_end')
    bounded=len(norm.split())<=60
    if case['kind']=='casual':
        relevant=bool(re.search(r"\b(?:here|ready|not much|happy to chat|on your mind|talk about|chat with you)\b",text,re.I))
        relevant=relevant and not re.search(r'\b(?:tornado|earthquake|united states|my day|my weekend|my job)\b',text,re.I)
    else:
        relevant=bool(re.search(r'\b(?:i can|help you|assist you)\b',text,re.I)) and bool(re.search(r'\b(?:questions|explain|writing|python|code|summarize|summarise)\b',text,re.I))
        relevant=relevant and not re.search(r'\byou can help me\b',text,re.I)
    return dict(relevant=bool(relevant and bounded),natural_stop=natural,nonrepetitive_nonempty=clean,passed=bool(relevant and bounded and clean and natural))
