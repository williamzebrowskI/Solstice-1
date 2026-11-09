"""Conservative filters and mechanically verifiable formatting constraints.

Passing these checks is not a factual correctness certificate.
"""
import ast
import hashlib
import re


def normalized(text): return ' '.join(re.findall(r'\w+', text.casefold()))
def digest(text): return hashlib.sha256(normalized(text).encode()).hexdigest()


def social_prompt(text):
    return normalized(text) in {'hi','hello','hey','hi there','hello there','hey there','thanks','thank you','okay','ok','bye','goodbye','good morning','good evening'}


def direct_general(messages):
    prompt=next(m['content'] for m in messages if m['role']=='user').strip()
    answer=messages[-1]['content']
    starts=r'^(?:what\b|why\b|how\b|when\b|where\b|which\b|who\b|can you\b|could you\b|please explain\b|explain\b|describe\b|tell me\b|give me\b|list\b|name\b|is\b|are\b|do\b|does\b|will\b|should\b|would\b)'
    unwanted=r'\b(?:role|character|persona|pretend|impersonate|rewrite|rephrase|proofread|edit\w*|essay|poem|story|prompt|dialogue|conversation|kidney|disease|diagnos\w*|cancer|treatment|medication|prescrib\w*|legal|tax|stocks|financial)\b|\b(?:your name|who are you|talk to a|act as)\b'
    return bool(re.search(starts,prompt,re.I) and not re.search(unwanted,prompt,re.I)
                and not re.search(r"\bmy name(?:['’]s| is)\b|\b(?:revised|rewritten|corrected) (?:sentence|version|text)\b",answer,re.I))


def repeated(text):
    words = normalized(text).split()
    if len(words) < 30: return False
    grams = [' '.join(words[i:i+8]) for i in range(len(words)-7)]
    return len(set(grams))/len(grams) < .75


def constraint_rules(prompt):
    # Accept a subset whose entire constraint suffix we can parse. Unknown
    # instructions are rejected, rather than silently calling them verified.
    match = re.search(r'Your (?:ENTIRE )?(?:response|answer)\b', prompt, re.I)
    if not match or len(prompt[:match.start()].strip()) < 25: return None
    # Do not skip an earlier unsupported rule and verify only the last one.
    if not prompt[:match.start()].strip().endswith('?'): return None
    tail = prompt[match.start():]
    rules = []
    number = r'(\d+)'
    patterns = [
        (r'Your (?:response|answer) should contain (exactly|at least|less than|at most|more than) '+number+r' (words|sentences|bullet points)\.?', 'count'),
        (r'Your (?:response|answer) should be (?:in )?all (lowercase|uppercase)(?: letters)?\.?', 'case'),
        (r'(?:Include|Use) a title wrapped in double angular brackets, (?:such as|i\.e\.) <<title>>\.?', 'title'),
        (r'(?:Include|Add) a postscript (?:to your response )?starting with ["\[]?(P\.S\.|P\.P\.S\.)["\]]?\.?', 'postscript'),
    ]
    rest = tail
    for pattern, kind in patterns:
        for m in re.finditer(pattern, tail, re.I):
            rules.append({'kind': kind, 'args': list(m.groups())})
        rest = re.sub(pattern, '', rest, flags=re.I)
    rest = re.sub(r'Use the markdown bullet points such as:\s*\* This is point\s*1\.?', '', rest, flags=re.I)
    if re.search(r'[a-zA-Z0-9]', rest) or not rules: return None
    return rules


def check_rules(text, rules):
    for rule in rules:
        kind, args = rule['kind'], rule['args']
        if kind == 'exact':
            if text.strip() != args[0]: return False
        elif kind == 'count':
            relation, number, unit = args; number = int(number)
            if unit.lower() == 'words': count = len(text.split())
            elif unit.lower() == 'sentences': count = len(re.findall(r'[^.!?]+[.!?](?:\s|$)', text))
            else: count = len(re.findall(r'^\s*[-*+]\s+\S', text, re.M))
            passed = {'exactly': count == number, 'at least': count >= number,
                      'less than': count < number, 'at most': count <= number,
                      'more than': count > number}[relation.lower()]
            if not passed: return False
        elif kind == 'case':
            if text != (text.lower() if args[0].lower() == 'lowercase' else text.upper()): return False
        elif kind == 'title':
            if not re.search(r'<<[^<>\n]+>>', text): return False
        elif kind == 'postscript':
            if not re.search(r'(?:^|\n)\s*'+re.escape(args[0]), text): return False
        else: raise ValueError('Unknown constraint rule')
    return True


def python_target(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    if not blocks: return False
    imports = []
    try:
        for block in blocks:
            tree = ast.parse(block)
            if len(list(ast.walk(tree))) > 350: return False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): imports += [a.name.split('.')[0] for a in node.names]
                if isinstance(node, ast.ImportFrom): imports.append((node.module or '').split('.')[0])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec', 'compile', '__import__'): return False
    except SyntaxError: return False
    allowed = {'math','collections','itertools','functools','re','json','datetime','statistics','string','typing','random','decimal','fractions','heapq','bisect'}
    return all(name in allowed for name in imports)


def python_training_target(instruction, response):
    if not python_target(response) or len(instruction.split())>120: return False
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
    trees=[ast.parse(block) for block in blocks]
    if sum(len(list(ast.walk(t))) for t in trees)>80: return False
    forbidden=(ast.ClassDef,ast.AsyncFunctionDef,ast.Try,ast.With,ast.Lambda,ast.Yield)
    if any(isinstance(n,forbidden) for t in trees for n in ast.walk(t)): return False
    # Validate literal numeric examples through a restricted arithmetic AST
    # interpreter. No downloaded Python code is executed.
    examples=re.findall(r'([a-zA-Z_]\w*)\((\d+(?:\s*,\s*\d+)*)\)\s*(?:,?\s*(?:would|should|must) return|==|=)\s*(-?\d+)\b',instruction)
    def arithmetic(node,bindings):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float): return node.value
        if isinstance(node,ast.Name): return bindings[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub): return -arithmetic(node.operand,bindings)
        if isinstance(node,ast.BinOp):
            left,right=arithmetic(node.left,bindings),arithmetic(node.right,bindings)
            if abs(left)>1e8 or abs(right)>100: raise ValueError('Arithmetic bound')
            ops={ast.Add:lambda:left+right,ast.Sub:lambda:left-right,ast.Mult:lambda:left*right,
                 ast.Div:lambda:left/right,ast.FloorDiv:lambda:left//right,ast.Mod:lambda:left%right}
            return ops[type(node.op)]()
        raise ValueError('Unsupported numeric example')
    for name,args,expected in examples:
        try:
            fn=next(n for t in trees for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
            body=[n for n in fn.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
            values=[int(v) for v in args.split(',')]
            if len(body)!=1 or not isinstance(body[0],ast.Return) or len(values)!=len(fn.args.args): return False
            if arithmetic(body[0].value,dict(zip([a.arg for a in fn.args.args],values)))!=int(expected): return False
        except (StopIteration,KeyError,ValueError,ZeroDivisionError): return False
    return True


class Deduplicator:
    def __init__(self, blocked=()):
        self.exact = set(blocked); self.responses = set(); self.shingles = set()

    def accept(self, messages, component):
        prompts = [m['content'] for m in messages if m['role'] == 'user']
        answers = [m['content'] for m in messages if m['role'] == 'assistant']
        # QA passage is shared intentionally, so use the question for overlap.
        # Generic greetings legitimately recur in different conversations.
        # Preserve the original conversation group for splitting, and still
        # check every substantive question for benchmark/cross-source overlap.
        keys = [digest(p.rsplit('Question:', 1)[-1]) if component == 'qa' else digest(p) for p in prompts
                if not (component == 'chat' and social_prompt(p))]
        if component == 'chat': keys.append(digest('\n'.join(prompts)))
        for prompt in prompts:
            match = re.search(r'(?:^|\n)Question:\s*(.*?)(?:\n|$)', prompt)
            if match: keys.append(digest(match[1]))
        response_keys = [digest(a) for a in answers if len(a.split()) >= 40]
        chunks = set()
        for prompt in prompts:
            text = prompt.rsplit('Question:', 1)[-1] if component == 'qa' else prompt
            words = normalized(text).split()
            if len(words) >= 20:
                chunks.update(digest(' '.join(words[i:i+20])) for i in range(0,len(words)-19,10))
        if any(k in self.exact for k in keys) or any(k in self.responses for k in response_keys) or chunks & self.shingles: return False
        self.exact.update(keys); self.responses.update(response_keys); self.shingles.update(chunks)
        return True
