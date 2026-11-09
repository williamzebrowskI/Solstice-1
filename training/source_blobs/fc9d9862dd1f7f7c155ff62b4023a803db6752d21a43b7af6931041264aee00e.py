"""Conservative source-code contract checks without executing downloaded code.

Accept a single bounded pure function plus at least two literal assertions in
its instruction. A deliberately small AST interpreter evaluates the assertions;
unsupported programs are rejected rather than called functionally verified.
"""
import ast
import json
import operator
import re


def run_function(tree,name,args,kwargs=None):
    if len(list(ast.walk(tree)))>100:raise ValueError('AST bound')
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1 or len(tree.body)!=1:raise ValueError('Single function required')
    fn=functions[0]
    if fn.name!=name or fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs:raise ValueError('Unsupported signature')
    if len(args)!=len(fn.args.args) or kwargs:raise ValueError('Argument count')
    env=dict(zip([a.arg for a in fn.args.args],args))
    operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.FloorDiv:operator.floordiv,ast.Div:operator.truediv,ast.Mod:operator.mod}
    comparisons={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
    def bounded(v):
        if type(v) in (int,float) and abs(v)>1000000:raise ValueError('Numeric bound')
        if type(v) in (str,list,tuple,dict,set) and len(v)>1000:raise ValueError('Container bound')
        return v
    def value(n):
        if isinstance(n,ast.Constant) and type(n.value) in (str,int,float,bool,type(None)):return bounded(n.value)
        if isinstance(n,ast.Name):return env[n.id]
        if isinstance(n,(ast.List,ast.Tuple,ast.Set)):
            xs=[value(x) for x in n.elts];return bounded(tuple(xs) if isinstance(n,ast.Tuple) else set(xs) if isinstance(n,ast.Set) else xs)
        if isinstance(n,ast.Dict):return bounded(dict(zip([value(x) for x in n.keys],[value(x) for x in n.values])))
        if isinstance(n,ast.BinOp) and type(n.op) in operations:
            a,b=value(n.left),value(n.right)
            if isinstance(n.op,ast.Mult) and ((type(a) is int and abs(a)>1000) or (type(b) is int and abs(b)>1000)):raise ValueError('Multiplication bound')
            return bounded(operations[type(n.op)](a,b))
        if isinstance(n,ast.UnaryOp):
            v=value(n.operand)
            if isinstance(n.op,ast.USub):return bounded(-v)
            if isinstance(n.op,ast.Not):return not v
        if isinstance(n,ast.Compare) and len(n.ops)==1 and type(n.ops[0]) in comparisons:return comparisons[type(n.ops[0])](value(n.left),value(n.comparators[0]))
        if isinstance(n,ast.IfExp):return value(n.body if value(n.test) else n.orelse)
        if isinstance(n,ast.Subscript):
            index=slice(value(n.slice.lower) if n.slice.lower else None,value(n.slice.upper) if n.slice.upper else None,value(n.slice.step) if n.slice.step else None) if isinstance(n.slice,ast.Slice) else value(n.slice)
            return bounded(value(n.value)[index])
        if isinstance(n,ast.Call) and not n.keywords:
            xs=[value(x) for x in n.args]
            if isinstance(n.func,ast.Name):
                functions={'len':len,'sum':sum,'min':min,'max':max,'abs':abs,'sorted':sorted,'tuple':tuple,'list':list,'set':set,'str':str,'int':int,'bool':bool}
                if n.func.id in functions:return bounded(functions[n.func.id](*xs))
            elif isinstance(n.func,ast.Attribute):
                obj=value(n.func.value)
                if type(obj) is str and n.func.attr in ('lower','upper','strip','split','join','startswith','endswith','count','replace'):
                    return bounded(getattr(obj,n.func.attr)(*xs))
        raise ValueError('Unsupported expression')
    for n in fn.body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):continue
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):env[n.targets[0].id]=value(n.value)
        elif isinstance(n,ast.Return):return value(n.value)
        else:raise ValueError('Unsupported statement')
    raise ValueError('Return required')


def assertions(instruction):
    tests=[]
    for block in re.findall(r'```(?:python|py)?\s*\n(.*?)```',instruction,re.S):
        try:tree=ast.parse(block)
        except SyntaxError:continue
        for n in tree.body:
            if not isinstance(n,ast.Assert):continue
            c=n.test
            if not isinstance(c,ast.Compare) or len(c.ops)!=1 or not isinstance(c.ops[0],ast.Eq) or not isinstance(c.left,ast.Call) or not isinstance(c.left.func,ast.Name) or c.left.keywords:continue
            try:
                args=[ast.literal_eval(x) for x in c.left.args];expected=ast.literal_eval(c.comparators[0])
                if len(repr((args,expected)))>3000:continue
                tests.append(dict(function=c.left.func.id,args=args,expected=expected))
            except (ValueError,TypeError):continue
    return tests


def verify_target(instruction,response):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
    tests=assertions(instruction)
    if len(blocks)!=1 or len(tests)<2:return None
    try:
        json.dumps(tests,allow_nan=False)  # Reject nonportable bytes/set contracts.
        tree=ast.parse(blocks[0])
        if all(run_function(tree,t['function'],t['args'])==t['expected'] for t in tests):return tests
    except (SyntaxError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError):pass
    return None
