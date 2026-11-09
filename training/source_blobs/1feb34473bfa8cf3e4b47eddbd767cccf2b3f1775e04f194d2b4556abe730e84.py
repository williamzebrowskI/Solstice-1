"""Reviewed instruction-v7 contracts. Historical v5/v6 results stay frozen.

Functional Python, requested style and social heuristics are reported separately.
No regex panel is a universal factual judge. Test never selects checkpoints.
"""
import ast
import json
import re
from collections import defaultdict
from .bounded_python import run_function
from .quality import normalized, repeated, check_rules


def typed_equal(actual, expected):
    # Python's True == 1 must not certify an integer as a Boolean answer.
    if type(actual) is bool or type(expected) is bool:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected,(list,tuple)):
        return type(actual) is type(expected) and len(actual)==len(expected) and all(typed_equal(a,e) for a,e in zip(actual,expected))
    if isinstance(expected,dict):
        return type(actual) is dict and actual.keys()==expected.keys() and all(typed_equal(actual[k],v) for k,v in expected.items())
    return actual == expected


def python_checks(response, tests, *, function=None, code_only=True, docstring=False):
    """Resolve a single unnamed function only when no name was requested.

    Arguments are copied, interpreted with the existing bounded AST evaluator;
    downloaded/generated Python is never exec'd. Independent edge tests required.
    """
    answer=dict(python_syntax=False,python_functional=False,python_name=False,
                python_format=False,python_docstring=not docstring)
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
    if len(blocks)!=1:return answer
    try:
        tree=ast.parse(blocks[0]);fn=tree.body[0]
        if len(tree.body)!=1 or not isinstance(fn,ast.FunctionDef):return answer
        answer['python_syntax']=True
        answer['python_name']=function is None or fn.name==function
        answer['python_format']=not code_only or bool(re.fullmatch(r'```(?:python|py)?\s*\n.*?```\s*',response.strip(),re.S))
        answer['python_docstring']=not docstring or bool(ast.get_docstring(fn))
        answer['python_functional']=len(tests)>=3 and all(
            typed_equal(run_function(tree,fn.name,ast.literal_eval(t['args_literal']) if 'args_literal' in t else json.loads(json.dumps(t['args']))),
            ast.literal_eval(t['expected_literal']) if 'expected_literal' in t else t['expected']) for t in tests)
    except (SyntaxError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError,RecursionError,AttributeError):pass
    return answer


def check(case,result):
    text=result['text'].strip();n=normalized(text);kind=case['kind']
    clean=bool(text) and not repeated(text);natural=result['stop_reason'] in ('turn_end','document_end')
    identity=bool(re.search(r"\b(?:my name|i live|my job|my weekend|your tutor|english teacher|you can help me|i(?: am|'m) (?:a doctor|a teacher|alex|sarah)|you(?: are|'re) feeling|glad you(?: are|'re)|you had a (?:good|great)|my family)\b",text,re.I))
    if kind in ('exact','grounded','abstain','context','fact'):
        ok=text in case['expected'] if kind!='fact' else n in {normalized(x) for x in case['expected']}
    elif kind=='clarify':
        # An imperative request is as valid as a question. Merely promising a
        # review is not a request for the missing input.
        request=bool(re.search(r'(?:^|[.!?]\s+)(?:please\s+)?(?:(?:go ahead and|feel free to)\s+)?(?:share|send|paste|provide|describe|upload)\b|\b(?:could|can|would|will) you (?:please )?(?:share|send|paste|provide|describe|upload)\b',text,re.I))
        detail=bool(re.search(case.get('missing_pattern',r'\b(?:paragraph|text|code|error|details|question|file|list)\b'),text,re.I))
        ok=request and detail and len(n.split())<=50 and not re.search(r'\b(?:looks good|well written|error is|fixed it|your paragraph says)\b',text,re.I)
    elif kind=='python':
        pc=python_checks(text,case['tests'],function=case.get('function'),code_only=case.get('code_only',True),docstring=case.get('docstring',False))
        return dict(**pc,relevant=pc['python_functional'],natural_stop=natural,nonrepetitive_nonempty=clean,
                    passed=bool(all(pc.values()) and natural and clean))
    elif kind=='format':ok=check_rules(text,case['rules']) and all(re.search(pattern,text,re.I) for pattern in case.get('semantic_patterns',[]))
    else:
        patterns={'greeting':r'\b(?:hello|hi|hey|sure|yes)\b','casual':r'\b(?:here|ready|chat|talk|help|not much|on your mind)\b',
                  'capability':r'\b(?:i can|help you|assist you)\b','thanks':r'\b(?:welcome|glad|happy|anytime)\b',
                  'farewell':r'\b(?:bye|goodbye|take care|see you|later)\b'}
        ok=bool(re.search(patterns[kind],text,re.I)) and len(n.split())<=case.get('max_words',40)
        if kind=='capability':ok=ok and bool(re.search(r'\b(?:questions|explain|writing|python|code|summarize)\b',text,re.I))
        if kind=='farewell':ok=ok and '?' not in text
        if kind in ('greeting','casual','capability'):ok=ok and not re.search(r'\b(?:grammar|homework|tutoring|first sentence|revision|revised version|your health)\b',text,re.I)
    return dict(relevant=bool(ok and not identity),unsupported_persona_or_context=identity,
                natural_stop=natural,nonrepetitive_nonempty=clean,passed=bool(ok and not identity and natural and clean))


def cases(split):
    if split not in ('dev','test'):raise ValueError('Unknown split')
    d=split=='dev';rows=[]
    def add(name,kind,prompt,expected=None,**kw):
        rows.append(dict(name=name,kind=kind,messages=[dict(role='user',content=prompt)],expected=expected,suite='instruction-v7',**kw))
    add('welcome','greeting','Hello, can we talk for a few minutes?' if d else 'Hey, may I ask you a few things?')
    add('casual','casual','What is happening on your side of the chat?' if d else 'Anything going on at your end?')
    add('capability','capability','Give a brief overview of the help you offer.' if d else 'What sorts of things could you help me do?')
    add('thanks','thanks','Thank you, that resolved my question.' if d else 'That makes it clearer. Thanks for helping.')
    add('bye','farewell','I am heading out now. Goodbye for today.' if d else 'Time for me to go. See you another day.')
    add('clarify-writing','clarify','I want feedback on a passage but have not included it.' if d else 'Check a sentence for me; I have not sent the sentence.',missing_pattern=r'\b(?:passage|sentence|text)\b')
    add('clarify-code','clarify','My script fails, but I have not shared the code or error.' if d else 'Help find a bug. I have not supplied the program or its output.',missing_pattern=r'\b(?:code|script|program|error|output)\b')
    add('clarify-data','clarify','Sort my items; I have not given you the list.' if d else 'I need a summary, but the text is not included.',missing_pattern=r'\b(?:list|items|text)\b')
    code='LYNX_731' if d else 'SWIFT_926'
    add('copy','exact',f'Reply with only this identifier, preserving case and punctuation: {code}',[code])
    a,b,num,num2=('Nori','Asha','58','29') if d else ('Oren','Kira','63','17')
    add('record','grounded',f'Roster: {a} is assigned cabin {num}; {b} is assigned cabin {num2}. Give only {b}’s cabin number.',[num2])
    add('missing','abstain',f'Roster: {a} is assigned cabin {num}. The roster gives no arrival date. Reply with only "Unlisted" if the arrival date is missing.',['Unlisted'])
    add('lowercase','exact','Reply only with the lowercase form of SPRUCE.' if d else 'Reply only with the lowercase form of BADGER.',['spruce'] if d else ['badger'])
    add('factual','fact','Give only the name of the instrument used to measure temperature.' if d else 'Give only the number of sides of a hexagon.',['thermometer','a thermometer'] if d else ['6','six'])
    add('two-sentences','format','Context: A bookmark keeps your place in a book. Explain this in exactly two sentences.' if d else 'Context: An envelope holds a letter. Explain this in exactly two sentences.',rules=[dict(kind='count',args=['exactly','2','sentences'])],semantic_patterns=[r'\b(?:book|page)\b',r'\b(?:place|mark|reading)\b'] if d else [r'\benvelope\b',r'\bletter\b'])
    add('three-bullets','format','List only these flowers in three bullet points: poppy, lupin, aster.' if d else 'List only these fabrics in three bullet points: velvet, denim, satin.',rules=[dict(kind='count',args=['exactly','3','bullet points'])],semantic_patterns=[r'\bpoppy\b',r'\blupin\b',r'\baster\b'] if d else [r'\bvelvet\b',r'\bdenim\b',r'\bsatin\b'])
    add('word-limit','format','In at most twelve words, restate: A bookmark keeps your place in a book.' if d else 'In at most twelve words, restate: An envelope holds a letter.',rules=[dict(kind='count',args=['at most','12','words'])],semantic_patterns=[r'\b(?:book|page)\b',r'\b(?:place|mark|reading)\b'] if d else [r'\benvelope\b',r'\bletter\b'])
    recipes=([
        ('unnamed-double','Write a Python function that doubles every number in a list. Return only a Python code block.',None,[[[]],[[2,-3,0]],[[1,1]]],[[],[4,-6,0],[2,2]],False),
        ('unnamed-repeat','Write a Python function taking n and char, returning char repeated n times for nonnegative n. Include a docstring. Return only a Python code block.',None,[[0,'x'],[3,'a'],[2,'']],[ '', 'aaa',''],True),
        ('named-whitespace','Write nonblank(text) in Python. Return True only if the string contains a non-whitespace character. Return only a Python code block.','nonblank',[[''],[' \t\n'],[' a ']], [False,False,True],False),
        ('named-filter','Write keep_even(numbers) in Python, preserving only even integers in their original order. Return only a Python code block.','keep_even',[[[]],[[1,2,-4,0,3]],[[7,7]]],[[],[2,-4,0],[]],False)
    ] if d else [
        ('unnamed-absolute','Write a Python function returning the absolute value of each number in a list. Return only a Python code block.',None,[[[]],[[-4,0,3]],[[-1,-1]]],[[],[4,0,3],[1,1]],False),
        ('unnamed-prefix','Write a Python function taking text and prefix, returning whether text starts with prefix. Include a docstring. Return only a Python code block.',None,[['alpha','al'],['alpha','z'],['','']],[True,False,True],True),
        ('named-space','Write join_words(words) in Python, joining strings with exactly one space. Return only a Python code block.','join_words',[[[]],[['red','blue']],[['one']]],[ '', 'red blue','one'],False),
        ('named-odd','Write count_odd(numbers) in Python, returning the count of odd integers. Return only a Python code block.','count_odd',[[[]],[[1,-3,2,0,5]],[[2,4]]],[0,3,0],False)
    ])
    for name,prompt,fn,args,expected,doc in recipes:
        add(name,'python',prompt,function=fn,docstring=doc,tests=[dict(args=a,expected=e) for a,e in zip(args,expected)])
    return rows


def rollouts(split):
    d=split=='dev'
    return [dict(name='live-social',kind='casual',prompts=['Hi, shall we chat?' if d else 'Hello, can we have a chat?','What are you up to over there?' if d else 'What is new at your end?']),
        dict(name='live-correction',kind='context',prompts=['Keep track of my meeting: room 26 at noon.' if d else 'Keep track of my delivery: shelf 43 on Friday.',
            'Correction: room 81 at noon.' if d else 'Correction: shelf 67 on Friday.',
            'Give only the latest room number.' if d else 'Give only the latest shelf number.'],expected=['81'] if d else ['67']),
        dict(name='live-list',kind='context',prompts=['My list is plum, oat, fig. Remember it.' if d else 'My list is silk, clay, ink. Remember it.',
            'Remove oat from that list.' if d else 'Remove clay from that list.',
            'Return the remaining items separated by a comma and a space, and nothing else.'],expected=['plum, fig'] if d else ['silk, ink']),
        dict(name='live-python',kind='exact',prompts=['Write plus_four(n) in Python, returning n plus four. Return only a Python code block.' if d else 'Write minus_five(n) in Python, returning n minus five. Return only a Python code block.',
            'Return only the result for input 7.' if d else 'Return only the result for input 8.',
            'Return only the result for input -2.' if d else 'Return only the result for input -3.'],expected=['2'] if d else ['-8'],middle=['11'] if d else ['3'],
            function='plus_four' if d else 'minus_five',tests=[dict(args=[a],expected=a+4 if d else a-5) for a in (0,7,-2)])]


def run(model,tokenizer,plan,split='dev'):
    from .evaluate import generate
    samples=[];rates=defaultdict(list)
    for case in cases(split)+rollouts(split):
        for with_system in (False,True):
            messages=[dict(role='system',content=plan['conversation_system'])] if with_system else []
            if 'prompts' in case:
                transcript=[];incomplete=False
                for prompt in case['prompts']:
                    messages.append(dict(role='user',content=prompt));result=generate(model,tokenizer,messages,limit=160)
                    transcript.append(dict(request=[dict(m) for m in messages],**result))
                    if not result['text'].strip():incomplete=True;break
                    messages.append(dict(role='assistant',content=result['text']))
                checks=check(case,result)
                checks['passed']=checks['passed'] and not incomplete and all(t['stop_reason'] in ('turn_end','document_end') and not repeated(t['text']) for t in transcript)
                if case['name']=='live-python':
                    checks['passed']=checks['passed'] and len(transcript)==3 and transcript[1]['text'].strip() in case['middle'] and all(python_checks(transcript[0]['text'],case['tests'],function=case['function']).values())
                record=dict(transcript=transcript,messages=transcript[-1]['request'],incomplete=incomplete)
            else:
                messages.extend(case['messages']);result=generate(model,tokenizer,messages,limit=160);checks=check(case,result);record=dict(messages=messages)
            rates[case['kind']].append(checks['passed'])
            samples.append(dict(name=case['name'],kind=case['kind'],system_variant=with_system,temperature=0.,**record,**result,checks=checks))
    means={k:sum(v)/len(v) for k,v in rates.items()}
    return dict(split=split,rates=means,macro_pass=sum(means.values())/len(means),samples=samples,
        scope='Instruction-v7: exact contracts, bounded Python functions with independent edge tests and separately reported naming/format/docstring; imperative clarification accepted; actual-history rollouts; social/fact coverage is narrow and needs raw-response review.')
