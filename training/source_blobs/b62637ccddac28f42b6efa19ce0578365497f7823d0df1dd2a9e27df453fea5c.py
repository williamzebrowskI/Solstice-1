"""Instruction-text/Python diagnostics frozen before SFT-018.

Small contracts and heuristic social checks; no general truth judge. Test never
ranks checkpoints. Rollouts retain actual replies and stop on an empty turn.
"""
from collections import defaultdict
from .behavior_v5 import check as social_check
from .quality import repeated


def cases(split):
    if split not in ('dev','test'): raise ValueError('Unknown split')
    rows=[]
    def add(name,kind,prompt,expected=None,**kw):
        rows.append(dict(name=name,kind=kind,messages=[dict(role='user',content=prompt)],expected=expected,suite='instruction-v6',**kw))
    dev=split=='dev'
    add('greet','greeting','Hello, I would like to ask a couple of questions.' if dev else 'Hi there, are you ready for a few questions?')
    add('casual','casual','How is your day going so far?' if dev else 'How have things been going for you?')
    add('capability','capability','Tell me briefly what you can assist me with.' if dev else 'What kinds of questions or tasks can I bring to you?')
    add('thanks','thanks','That cleared it up, thank you for explaining.' if dev else 'Thanks for taking the time to explain that.')
    add('bye','farewell','I will leave the conversation here. Goodbye.' if dev else 'I have to leave the chat now. Bye for today.')
    add('clarify','clarify','Please review my paragraph. I have not pasted it yet.' if dev else 'Can you help debug my program? I have not provided the code yet.')
    code='CEDAR_418' if dev else 'HERON_603'
    add('copy','exact',f'Print this identifier alone, exactly as written: {code}',[code])
    name,other,number,number2=('Elio','Tessa','46','72') if dev else ('Anika','Ravi','31','86')
    add('ground','grounded',f'Record: {name} carries badge {number}; {other} carries badge {number2}. Return only {name}’s badge number.',[number])
    add('missing','abstain',f'Record: {name} carries badge {number}. What is {name}’s birth year? Return exactly "Absent" if the record omits it.',['Absent'])
    add('case','exact','Convert only the word FERN to lowercase.' if dev else 'Convert only the word OTTER to lowercase.',['fern'] if dev else ['otter'])
    fn='count_chars' if dev else 'total_numbers'
    prompt=('Write count_chars(text) in Python, returning the number of characters, including spaces.' if dev else 'Write total_numbers(numbers) in Python, returning the sum of the list, or zero for an empty list.')
    tests=([dict(function=fn,args=[''],expected=0),dict(function=fn,args=['ab cd'],expected=5),dict(function=fn,args=['z'],expected=1)] if dev else [dict(function=fn,args=[[]],expected=0),dict(function=fn,args=[[4,-1,2]],expected=5),dict(function=fn,args=[[-3]],expected=-3)])
    add('python','python',prompt+' Return only a Python code block.',function=fn,tests=tests)
    return rows


def rollouts(split):
    if split=='dev':
        return [dict(name='live-social',kind='casual',prompts=['Hello, could we chat briefly?','How are you getting on?']),
            dict(name='live-reference',kind='context',prompts=['My delivery reference is KL-204. Please note it.','I mistyped it: the reference should be KL-718.','Give me only the corrected delivery reference.'],expected=['KL-718']),
            dict(name='live-python',kind='exact',prompts=['Write triple(n) in Python to return three times n. Include the code.','Give only the result of triple(6).','Give only the result of triple(-4).'],expected=['-12'])]
    return [dict(name='live-social',kind='casual',prompts=['Hello there, shall we talk for a bit?','How are you doing at the moment?']),
        dict(name='live-reference',kind='context',prompts=['Please keep my booking reference, WP-513, in mind.','Correct the booking reference to WP-862.','Return only the updated booking reference.'],expected=['WP-862']),
        dict(name='live-python',kind='exact',prompts=['Write subtract_two(n) in Python to return n minus two. Include the code.','Give only the result of subtract_two(9).','Give only the result of subtract_two(-5).'],expected=['-7'])]


def check(case,result):
    return social_check(case,result)


def run(model,tokenizer,plan,split='dev'):
    from .evaluate import generate
    samples=[];rates=defaultdict(list)
    for case in cases(split)+rollouts(split):
        for with_system in (False,True):
            messages=[dict(role='system',content=plan['conversation_system'])] if with_system else []
            if 'prompts' in case:
                transcript=[];incomplete=False
                for prompt in case['prompts']:
                    messages.append(dict(role='user',content=prompt))
                    result=generate(model,tokenizer,messages,limit=160)
                    transcript.append(dict(request=[dict(m) for m in messages],**result))
                    if not result['text'].strip(): incomplete=True;break
                    messages.append(dict(role='assistant',content=result['text']))
                checks=check(case,result)
                checks['passed']=checks['passed'] and not incomplete and all(t['stop_reason'] in ('turn_end','document_end') and not repeated(t['text']) for t in transcript)
                if case['name']=='live-python':
                    # Both numeric follow-ups must work, not just the last one.
                    expected='18' if split=='dev' else '7'
                    checks['passed']=checks['passed'] and len(transcript)==3 and transcript[1]['text'].strip()==expected
                    from .bounded_python import verify_target
                    fn='triple' if split=='dev' else 'subtract_two'
                    tests=[dict(function=fn,args=[6],expected=18),dict(function=fn,args=[-4],expected=-12)] if split=='dev' else [dict(function=fn,args=[9],expected=7),dict(function=fn,args=[-5],expected=-7)]
                    checks['passed']=checks['passed'] and verify_target('',transcript[0]['text'],tests=tests) is not None
                record=dict(transcript=transcript,messages=transcript[-1]['request'],incomplete=incomplete)
            else:
                messages.extend(case['messages']);result=generate(model,tokenizer,messages,limit=160)
                checks=check(case,result);record=dict(messages=messages)
            rates[case['kind']].append(checks['passed'])
            samples.append(dict(name=case['name'],kind=case['kind'],system_variant=with_system,temperature=0.,**record,**result,checks=checks))
    means={k:sum(v)/len(v) for k,v in rates.items()}
    return dict(split=split,rates=means,macro_pass=sum(means.values())/len(means),samples=samples,
        scope='Instruction-v6: greedy system/no-system atomic instruction, grounding, absent-information, Python and actual-history checks; social checks heuristic; small panel requires raw-response review.')
