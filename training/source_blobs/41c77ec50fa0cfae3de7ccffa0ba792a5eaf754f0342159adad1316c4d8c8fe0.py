"""Fresh constrained diagnostic panel; strict whole-answer checks, not a judge.

Atomic fact cases explicitly request concise answers. Matching accepted complete
answers avoids rewarding a correct keyword inside contradictory prose. Valid
unlisted paraphrases can fail; raw replies remain available for human review.
"""
import ast
import operator
import re
from .quality import normalized, repeated


def cases(split):
    if split not in ('dev','test'):raise ValueError('Unknown split')
    result=[]
    def add(name,kind,prompt,accepted=None,history=None,**extra):
        result.append(dict(name=name,kind=kind,messages=[*(history or []),dict(role='user',content=prompt)],expected=accepted,suite='conversation-v3',**extra))
    if split=='dev':
        add('greet','greeting','Hello, good to meet you.')
        add('thanks','thanks','That clears up my question; thank you.')
        add('bye','farewell','I will end our chat here. Goodbye.')
        add('planet','fact','Which planet is known as the Red Planet? Answer with its name only.',['Mars'])
        add('shape','fact','How many sides does a pentagon have? Give only the number.',['5','five'])
        add('sum','fact','What is 37 plus 46? Answer with the number only.',['83','eighty-three'])
        add('exact','exact','Output exactly ZEPHYR, with no other text.',['ZEPHYR'])
        add('case','exact','Convert MARIGOLD to lowercase and output only the result.',['marigold'])
        add('grounded','grounded','Record: visitor Liora; room 28; arrival Sunday. What room is assigned? Answer with only the room number.',['28'])
        add('abstain','abstain','Record: visitor Liora; room 28. What is the visitor’s age? Reply exactly "Not stated" if missing.',['Not stated'])
        add('context','context','What is the current delivery location? Reply with only the location.',['East Gate'],history=[dict(role='user',content='The delivery location is West Gate.'),dict(role='assistant',content='Understood: West Gate.'),dict(role='user',content='Correction: the delivery goes to East Gate.'),dict(role='assistant',content='The current delivery location is East Gate.')])
        add('clarify','clarify','Can you help me change this?')
        add('python','python','Write a Python function increment(value) that returns the input plus one. Output only a Python code block.',function='increment',tests=[([0],1),([-4],-3),([21],22)])
    else:
        add('greet','greeting','Hi, nice to chat with you.')
        add('thanks','thanks','Thank you, that cleared things up for me.')
        add('bye','farewell','That is everything. I am leaving the chat now.')
        add('planet','fact','Which planet is famous for its prominent rings? Give its name only.',['Saturn'])
        add('shape','fact','How many sides are on an octagon? Reply with the number only.',['8','eight'])
        add('sum','fact','What is 43 plus 38? Give only the number.',['81','eighty-one'])
        add('exact','exact','Reply exactly ORCHID without punctuation or any other words.',['ORCHID'])
        add('case','exact','Output the lowercase form of TANGERINE, and nothing else.',['tangerine'])
        add('grounded','grounded','Record: guest Nera; room 64; arrival Wednesday. Which room does the guest have? Give the room number alone.',['64'])
        add('abstain','abstain','Record: guest Nera; room 64. What is the guest’s phone number? Reply exactly "Not given" if absent.',['Not given'])
        add('context','context','Which desk is now assigned? Give only its name.',['Desk B'],history=[dict(role='user',content='I was assigned Desk A.'),dict(role='assistant',content='Your assigned desk is Desk A.'),dict(role='user',content='That changed. I am now assigned Desk B.'),dict(role='assistant',content='Your current assigned desk is Desk B.')])
        add('clarify','clarify','I need assistance with the issue I mentioned.')
        add('python','python','Write a Python function double_value(value) returning twice the input. Output only a Python code block.',function='double_value',tests=[([0],0),([-3],-6),([17],34)])
    return result


def functional_python(text,name,tests):
    """Interpret a bounded pure return expression; never execute generated code."""
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    if len(blocks)!=1:return False
    try:
        tree=ast.parse(blocks[0])
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return False
        fn=tree.body[0]
        body=[n for n in fn.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        if fn.name!=name or fn.decorator_list or fn.args.defaults or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or len(body)!=1 or not isinstance(body[0],ast.Return):return False
        if len(list(ast.walk(tree)))>50:return False
        operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul}
        def value(node,env):
            if isinstance(node,ast.Name):return env[node.id]
            if isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<1000:return node.value
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -value(node.operand,env)
            if isinstance(node,ast.BinOp) and type(node.op) in operations:
                a,b=value(node.left,env),value(node.right,env)
                if abs(a)>100000 or abs(b)>100000:raise ValueError('Bound')
                return operations[type(node.op)](a,b)
            raise ValueError('Unsupported expression')
        return all(len(args)==len(fn.args.args) and value(body[0].value,dict(zip([a.arg for a in fn.args.args],args)))==expected for args,expected in tests)
    except (SyntaxError,ValueError,KeyError,TypeError,RecursionError):return False


def check(case,result):
    text=result['text'].strip();kind=case['kind'];norm=normalized(text)
    clean=bool(text) and not repeated(text);natural=result['stop_reason'] in ('turn_end','document_end')
    if kind in ('exact','grounded','abstain','context'):relevant=text in case['expected']
    elif kind=='fact':relevant=norm in {normalized(a) for a in case['expected']}
    elif kind=='python':relevant=functional_python(text,case['function'],case['tests'])
    elif kind=='clarify':relevant='?' in text and len(norm.split())<=40 and bool(re.search(r'\b(?:details|share|describe|which|what|information|issue)\b',text,re.I)) and not re.search(r'\b(?:download|restart|price|diagnos|medical|grammar|teacher|rewrite)\b',text,re.I)
    else:
        unwanted=bool(re.search(r'\b(?:teacher|tutor|grammar|english language|revised|rewrite|lesson|my name|i am alex)\b',text,re.I))
        if kind=='greeting':relevant=bool(re.search(r'\b(?:hi|hello|hey|greetings)\b',text,re.I)) and len(norm.split())<=20
        elif kind=='thanks':relevant=bool(re.search(r'\b(?:welcome|glad|anytime|happy to help)\b',text,re.I)) and len(norm.split())<=20
        elif kind=='farewell':relevant=bool(re.search(r'\b(?:bye|goodbye|take care|see you)\b',text,re.I)) and len(norm.split())<=15 and '?' not in text
        else:raise ValueError('Unknown diagnostic kind')
        relevant=relevant and not unwanted
    return dict(relevant=bool(relevant),natural_stop=natural,nonrepetitive_nonempty=clean,passed=bool(relevant and clean and natural))
