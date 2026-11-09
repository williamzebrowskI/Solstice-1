"""SFT-016 diagnostics. Frozen before training; test never ranks checkpoints.

Social checks are heuristics. Exact answers and bounded Python tests are narrow
contracts, not a general judge. Rollouts use the model's actual prior replies.
"""
import re
from .quality import normalized, repeated
from .bounded_python import verify_target


def cases(split):
    if split not in ('dev','test'):raise ValueError('Unknown split')
    rows=[]
    def add(name,kind,prompt,expected=None,history=(),**kw):
        rows.append(dict(name=name,kind=kind,messages=[*history,dict(role='user',content=prompt)],expected=expected,suite='conversation-v5',**kw))
    if split=='dev':
        add('greet','greeting','Hello there, can we talk for a moment?')
        add('casual','casual','How are things with you?')
        add('capability','capability','What sorts of tasks are you able to help me do?')
        add('thanks','thanks','Your explanation made that clearer. Much appreciated.')
        add('bye','farewell','I am going to close this chat. Take care.')
        add('clarify','clarify','I have a problem with my assignment but have not shared it yet. Can you check it?')
        add('animal','fact','Which animal group do frogs belong to? Give only the group name.',['amphibians','amphibian'])
        add('measure','fact','Which instrument measures air pressure? Give only its name.',['barometer','a barometer'])
        add('sum','fact','Calculate 26 plus 59. Give only the result.',['85','eighty five'])
        add('copy','exact','Output only this exact identifier: KESTREL_72',['KESTREL_72'])
        add('ground','grounded','Notice: Imani has locker 83; Pavel has locker 16. Which locker is Imani’s? Output only the number.',['83'])
        add('absent','abstain','Notice: Imani has locker 83. What is Imani’s birthday? Output exactly "Not supplied" if absent.',['Not supplied'])
        add('correct','context','What is the current reservation code? Give only the code.',['RQ-91'],history=[dict(role='user',content='My reservation code is RQ-26.'),dict(role='assistant',content='Your reservation code is RQ-26.'),dict(role='user',content='Please correct it to RQ-91.'),dict(role='assistant',content='The current code is RQ-91.')])
        add('python','python','Write a Python function last_item(items) that returns the last item, or None when the list is empty. Return only a Python code block.',function='last_item',tests=[dict(function='last_item',args=[[3,9]],expected=9),dict(function='last_item',args=[[]],expected=None),dict(function='last_item',args=[['a']],expected='a')])
    else:
        add('greet','greeting','Hey there, have you got time to talk?')
        add('casual','casual','What is happening on your end?')
        add('capability','capability','Could you describe the kinds of help you offer?')
        add('thanks','thanks','That explanation was useful to me. Thank you kindly.')
        add('bye','farewell','We can leave it there. Until next time.')
        add('clarify','clarify','I would like you to review a file, but I have not sent it. What do you need?')
        add('animal','fact','Which animal group do salmon belong to? Return only the group name.',['fish'])
        add('measure','fact','What instrument measures wind speed? Return only its name.',['anemometer','an anemometer'])
        add('sum','fact','Calculate 54 plus 29. Return the result alone.',['83','eighty three'])
        add('copy','exact','Return this identifier exactly, without extra text: LARK_95',['LARK_95'])
        add('ground','grounded','Notice: Soren has seat 57; Leila has seat 92. What is Leila’s seat? Return only the number.',['92'])
        add('absent','abstain','Notice: Soren has seat 57. What is Soren’s phone number? Output exactly "Not recorded" when missing.',['Not recorded'])
        add('correct','context','What is the current ticket code? Return only the code.',['TK-68'],history=[dict(role='user',content='The ticket code is TK-35.'),dict(role='assistant',content='The ticket code is TK-35.'),dict(role='user',content='Replace it with TK-68.'),dict(role='assistant',content='The current ticket code is TK-68.')])
        add('python','python','Write a Python function first_item(items) that returns the first item, or None for an empty list. Return only a Python code block.',function='first_item',tests=[dict(function='first_item',args=[[7,2]],expected=7),dict(function='first_item',args=[[]],expected=None),dict(function='first_item',args=[['b']],expected='b')])
    return rows


def rollouts(split):
    if split=='dev':
        return [dict(name='live-casual',kind='casual',prompts=['Hello, are you available?','Anything interesting going on with you?']),
                dict(name='live-correction',kind='context',prompts=['My parcel tag is VN-42. Please remember that.','Change the parcel tag to VN-87.','What is the current parcel tag? Return the tag alone.'],expected=['VN-87'])]
    return [dict(name='live-casual',kind='casual',prompts=['Hi, can I talk to you?','Anything new on your side?']),
            dict(name='live-correction',kind='context',prompts=['My folder tag is HM-23. Please remember it.','Update the folder tag to HM-76.','What is the folder’s current tag? Return the tag alone.'],expected=['HM-76'])]


def check(case,result):
    text=result['text'].strip();n=normalized(text);kind=case['kind']
    clean=bool(text) and not repeated(text);natural=result['stop_reason'] in ('turn_end','document_end')
    identity=bool(re.search(r'\b(?:my name|i live|my job|my weekend|your tutor|english teacher|you can help me)\b',text,re.I))
    if kind in ('exact','grounded','abstain','context'):ok=text in case['expected']
    elif kind=='fact':ok=n in {normalized(x) for x in case['expected']}
    elif kind=='python':ok=verify_target('',text,tests=case['tests'],strict=True) is not None
    elif kind=='clarify':ok='?' in text and bool(re.search(r'\b(?:share|send|provide|describe|file|assignment|details)\b',text,re.I)) and len(n.split())<=50
    elif kind=='greeting':ok=bool(re.search(r'\b(?:hello|hi|hey|sure|yes)\b',text,re.I)) and len(n.split())<=35
    elif kind=='casual':ok=bool(re.search(r'\b(?:here|ready|chat|talk|help|not much|on your mind)\b',text,re.I)) and len(n.split())<=50
    elif kind=='capability':ok=bool(re.search(r'\b(?:i can|help you|assist you)\b',text,re.I)) and bool(re.search(r'\b(?:questions|explain|writing|python|code|summarize)\b',text,re.I)) and len(n.split())<=60
    elif kind=='thanks':ok=bool(re.search(r'\b(?:welcome|glad|happy|anytime)\b',text,re.I)) and len(n.split())<=25
    elif kind=='farewell':ok=bool(re.search(r'\b(?:bye|goodbye|take care|see you|later)\b',text,re.I)) and len(n.split())<=25 and '?' not in text
    else:raise ValueError('Unknown case')
    return dict(relevant=bool(ok and not identity),natural_stop=natural,nonrepetitive_nonempty=clean,passed=bool(ok and not identity and natural and clean))


def run(model,tokenizer,plan,split='dev'):
    from .evaluate import generate
    from collections import defaultdict
    samples=[];rates=defaultdict(list)
    for case in cases(split)+rollouts(split):
        for with_system in (False,True):
            messages=[dict(role='system',content=plan['conversation_system'])] if with_system else []
            if 'prompts' in case:
                transcript=[]
                for prompt in case['prompts']:
                    messages.append(dict(role='user',content=prompt))
                    result=generate(model,tokenizer,messages,limit=128)
                    transcript.append(dict(request=[dict(m) for m in messages],**result))
                    messages.append(dict(role='assistant',content=result['text'] or '[empty reply]'))
                checks=check(case,result)
                checks['passed']=checks['passed'] and all(t['stop_reason'] in ('turn_end','document_end') and bool(t['text'].strip()) and not repeated(t['text']) for t in transcript)
                record=dict(transcript=transcript,messages=transcript[-1]['request'])
            else:
                messages.extend(case['messages']);result=generate(model,tokenizer,messages,limit=128)
                checks=check(case,result);record=dict(messages=messages)
            rates[case['kind']].append(checks['passed'])
            samples.append(dict(name=case['name'],kind=case['kind'],system_variant=with_system,temperature=0.,**record,**result,checks=checks))
    means={k:sum(v)/len(v) for k,v in rates.items()}
    return dict(split=split,rates=means,macro_pass=sum(means.values())/len(means),samples=samples,
                scope='Frozen conversation-v5: greedy system/no-system, strict atomic contracts and actual-history rollouts. Small diagnostics; social rules are heuristics; inspect raw replies. Prior panels are regression evidence, not independent tests.')
