"""Versioned, bounded interpreter for simple SFT code contracts; no exec/eval.

This deliberately supports only one pure function and literal test arguments.
Unsupported syntax is a rejection, not a successful functional check.
"""
import ast
import json
import operator
import re
from .pure_python import assertions


def run_function(tree, name, args):
    if len(list(ast.walk(tree))) > 250 or len(tree.body) != 1:
        raise ValueError('AST bound/single function')
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.name != name or fn.decorator_list:
        raise ValueError('Function identity')
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.defaults:
        raise ValueError('Unsupported signature')
    if len(args) != len(fn.args.args): raise ValueError('Argument count')
    env = dict(zip([a.arg for a in fn.args.args], args))
    budget = [10000]
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
    comparisons = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                   ast.In: lambda a,b:a in b, ast.NotIn:lambda a,b:a not in b,
                   ast.Is:operator.is_, ast.IsNot:operator.is_not}
    def tick():
        budget[0] -= 1
        if budget[0] < 0: raise ValueError('Operation budget')
    def bounded(v, depth=0):
        if depth > 10: raise ValueError('Container depth')
        if type(v) in (int, float) and abs(v) > 1000000: raise ValueError('Number bound')
        if type(v) in (str, list, tuple, dict, set):
            if len(v) > 1000: raise ValueError('Container bound')
            if not isinstance(v,str):
                for x in (v.items() if isinstance(v,dict) else v): bounded(x,depth+1)
        elif type(v) not in (int,float,bool,type(None)): raise ValueError('Value type')
        return v
    for v in args: bounded(v)
    def bind(target, v, scope):
        if isinstance(target, ast.Name): scope[target.id] = bounded(v)
        elif isinstance(target,(ast.Tuple,ast.List)) and len(target.elts)==len(v):
            for t,x in zip(target.elts,v): bind(t,x,scope)
        else: raise ValueError('Assignment target')
    def value(n, scope):
        tick()
        if isinstance(n,ast.Constant): return bounded(n.value)
        if isinstance(n,ast.Name): return scope[n.id]
        if isinstance(n,(ast.List,ast.Tuple,ast.Set)):
            xs=[value(x,scope) for x in n.elts]
            return bounded(tuple(xs) if isinstance(n,ast.Tuple) else set(xs) if isinstance(n,ast.Set) else xs)
        if isinstance(n,ast.Dict): return bounded({value(k,scope):value(v,scope) for k,v in zip(n.keys,n.values)})
        if isinstance(n,ast.BinOp) and type(n.op) in ops:
            a,b=value(n.left,scope),value(n.right,scope)
            if isinstance(n.op,ast.Mult) and ((type(a) is int and abs(a)>1000) or (type(b) is int and abs(b)>1000)): raise ValueError('Multiply bound')
            return bounded(ops[type(n.op)](a,b))
        if isinstance(n,ast.UnaryOp):
            v=value(n.operand,scope)
            if isinstance(n.op,ast.USub): return bounded(-v)
            if isinstance(n.op,ast.Not): return not v
            if isinstance(n.op,ast.UAdd): return bounded(+v)
        if isinstance(n,ast.BoolOp):
            result=value(n.values[0],scope)
            for x in n.values[1:]:
                if (isinstance(n.op,ast.And) and not result) or (isinstance(n.op,ast.Or) and result):break
                result=value(x,scope)
            return result
        if isinstance(n,ast.Compare):
            left=value(n.left,scope)
            for op,right_node in zip(n.ops,n.comparators):
                if type(op) not in comparisons:raise ValueError('Comparison')
                right=value(right_node,scope)
                if not comparisons[type(op)](left,right):return False
                left=right
            return True
        if isinstance(n,ast.IfExp):return value(n.body if value(n.test,scope) else n.orelse,scope)
        if isinstance(n,ast.Subscript):
            s=n.slice
            index=slice(value(s.lower,scope) if s.lower else None,value(s.upper,scope) if s.upper else None,value(s.step,scope) if s.step else None) if isinstance(s,ast.Slice) else value(s,scope)
            return bounded(value(n.value,scope)[index])
        if isinstance(n,(ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)):
            if len(n.generators)!=1 or n.generators[0].is_async:raise ValueError('Comprehension bound')
            g=n.generators[0]; xs=[]; local=dict(scope)
            iterable=value(g.iter,scope)
            if isinstance(iterable,set):raise ValueError('Unordered iteration')
            for item in iterable:
                tick();bind(g.target,item,local)
                if all(value(c,local) for c in g.ifs):
                    xs.append((value(n.key,local),value(n.value,local)) if isinstance(n,ast.DictComp) else value(n.elt,local))
                if len(xs)>1000:raise ValueError('Comprehension length')
            return bounded(dict(xs) if isinstance(n,ast.DictComp) else set(xs) if isinstance(n,ast.SetComp) else xs)
        if isinstance(n,ast.Call):
            xs=[value(x,scope) for x in n.args];kw={k.arg:value(k.value,scope) for k in n.keywords}
            if None in kw:raise ValueError('Expanded kwargs')
            if isinstance(n.func,ast.Name):
                if n.func.id in ('list','tuple','str','enumerate','zip') and any(isinstance(x,set) for x in xs):
                    raise ValueError('Unordered conversion')
                functions={'len':len,'sum':sum,'min':min,'max':max,'abs':abs,'sorted':sorted,
                           'tuple':tuple,'list':list,'set':set,'str':str,'int':int,'bool':bool,'round':round,
                           'all':all,'any':any,'reversed':lambda x:list(reversed(x)),
                           'enumerate':lambda x:list(enumerate(x)), 'zip':lambda *x:list(zip(*x))}
                if n.func.id=='range':
                    r=range(*xs)
                    if len(r)>1000:raise ValueError('Range bound')
                    return list(r)
                if n.func.id in functions and (not kw or (n.func.id=='sorted' and set(kw)<= {'reverse'})):
                    return bounded(functions[n.func.id](*xs,**kw))
            if isinstance(n.func,ast.Attribute) and not kw:
                obj=value(n.func.value,scope);method=n.func.attr
                allowed={str:{'lower','upper','strip','lstrip','rstrip','split','join','startswith','endswith','count','replace','isalpha','isdigit','capitalize','title'},
                         list:{'count','index','copy'},dict:{'get','keys','values','items','copy'}}
                if method in allowed.get(type(obj),set()):
                    if method=='join' and xs and isinstance(xs[0],set):raise ValueError('Unordered join')
                    v=getattr(obj,method)(*xs)
                    return bounded(list(v) if type(obj) is dict and method in ('keys','values','items') else v)
            raise ValueError('Unsupported call')
        raise ValueError('Unsupported expression')
    class Returned(Exception):
        def __init__(self,v): self.v=v
    def statements(body):
        for n in body:
            tick()
            if isinstance(n,ast.Return):raise Returned(value(n.value,env) if n.value else None)
            if isinstance(n,ast.Assign) and len(n.targets)==1:bind(n.targets[0],value(n.value,env),env)
            elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and type(n.op) in ops:
                left,right=env[n.target.id],value(n.value,env)
                if isinstance(n.op,ast.Mult) and ((isinstance(left,(str,list,tuple)) and type(right) is int and len(left)*right>1000)
                                                or (isinstance(right,(str,list,tuple)) and type(left) is int and len(right)*left>1000)):
                    raise ValueError('Multiply bound')
                bind(n.target,bounded(ops[type(n.op)](left,right)),env)
            elif isinstance(n,ast.If):statements(n.body if value(n.test,env) else n.orelse)
            elif isinstance(n,ast.For):
                iterable=value(n.iter,env)
                if isinstance(iterable,set):raise ValueError('Unordered iteration')
                for item in iterable:tick();bind(n.target,item,env);statements(n.body)
                statements(n.orelse)
            elif isinstance(n,ast.Expr):
                if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):continue
                c=n.value
                if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and not c.keywords:
                    obj=value(c.func.value,env);xs=[value(x,env) for x in c.args]
                    if type(obj) is list and c.func.attr in ('append','extend','reverse'):
                        getattr(obj,c.func.attr)(*xs);bounded(obj);continue
                raise ValueError('Unsupported side effect')
            else:raise ValueError('Unsupported statement')
    try:statements(fn.body)
    except Returned as result:return bounded(result.v)
    raise ValueError('Return required')


def verify_target(instruction,response, tests=None, strict=False):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
    tests=assertions(instruction) if tests is None else tests
    if len(blocks)!=1 or len(tests)<2:return None
    if strict and not re.fullmatch(r'```(?:python|py)?\s*\n.*?```\s*',response.strip(),re.S):return None
    try:
        json.dumps(tests,allow_nan=False)
        tree=ast.parse(blocks[0])
        # Copy arguments: list operations must never mutate the saved tests.
        if all(run_function(tree,t['function'],ast.literal_eval(t['args_literal']) if 'args_literal' in t else json.loads(json.dumps(t['args']))) == (ast.literal_eval(t['expected_literal']) if 'expected_literal' in t else t['expected']) for t in tests):return tests
    except (SyntaxError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError,RecursionError,AttributeError):pass
    return None
