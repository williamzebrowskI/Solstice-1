"""Check fresh targets against source facts, separately from their generator."""
import math,re
from collections import Counter

def check(row):
    try:return _check(row)
    except (ValueError,KeyError,TypeError,IndexError,AttributeError):return False

def _check(row):
    spec=row['spec'];answer=row['messages'][-1]['content'];users='\n'.join(m['content'] for m in row['messages'] if m['role']=='user');kind=spec['kind']
    if kind=='operation':
        x=spec['values'];op=spec['op']
        if op in ('add','subtract','multiply'):
            expected=math.fsum(x) if op=='add' else x[0]+(-x[1]) if op=='subtract' else math.prod(x)
            return answer==str(int(expected)) and all(str(v) in users or (v==2 and 'two items' in users) for v in x)
        if op in ('sort','descending'):
            try:values=[int(v) for v in answer.split(', ')]
            except ValueError:return False
            ordered=all((a<=b if op=='sort' else a>=b) for a,b in zip(values,values[1:]))
            return Counter(values)==Counter(x) and ordered
        if op in ('even','odd'):
            values=[] if answer=='none' else [int(v) for v in answer.split(', ')]
            parity=0 if op=='even' else 1
            return values==[v for v in x if v%2==parity]
        if op=='reverse':return answer.split(', ')[::-1]==x
        if op=='alphabetical':
            values=answer.split(', ');return Counter(values)==Counter(x) and all(a<=b for a,b in zip(values,values[1:]))
        if op=='first':return answer.split(', ')==x[:2]
        if op=='last':return answer.split(', ')==x[len(x)-2:]
        if op=='count':return answer==str(sum(1 for _ in x))
        if op=='unique':return len(set(answer.split(', ')))==len(answer.split(', ')) and answer.split(', ')==[v for i,v in enumerate(x) if v not in x[:i]]
        if op=='upper':return answer.casefold()==x[0].casefold() and answer==answer.upper()
        if op=='lower':return answer.casefold()==x[0].casefold() and answer==answer.lower()
        if op=='replace':return answer==x[2].join(x[0].split(x[1]))
        raise ValueError(op)
    if kind=='record':
        records={person:dict(quantity=quantity,item=item,shelf=shelf) for person,quantity,item,shelf in re.findall(r'(\w+) stored (\d+) ([\w-]+) items on shelf (\d+)\.',users)}
        if len(records)!=2:return False
        return answer==records[spec['person']].get(spec['field'],'The passage does not say.')
    if kind=='latest':
        changes=re.findall(r'(?:Update the reference to|moved to shelf|Change that to) ([\d-]+)',users)
        return bool(changes) and answer==changes[-1]
    if kind=='lookup':
        values=spec['mapping'];key=spec['key']
        if key=='design':return answer==re.search(r'I want the ([\w-]+) design\.',users).group(1)
        choice=re.search(re.escape(key)+r' chose the ([\w-]+) item',users)
        if choice:return answer==choice.group(1)
        amount=re.search(re.escape(key)+r' owns (\d+)',users)
        if amount:return answer==amount.group(1)
        raise ValueError('Unknown dialogue lookup')
    if kind=='missing':return 'No collection time has been specified.' in users and answer=='The collection time has not been provided.'
    if kind=='format':
        source=spec['sentences'];family=spec['family']
        if not all(s in users for s in source):return False
        if family==0:return answer.split('\n')==['- '+s for s in source]
        if family==1:return answer==' '.join(source)
        if family==2:return answer.split('\n\n')==source
        if family in (3,4):return answer.casefold()==' '.join(source).casefold() and answer==(answer.lower() if family==3 else answer.upper())
        if family==5:return answer.startswith('"') and answer.endswith('"') and answer[1:-1]==source[0]
        if family==6:return answer.split('\n\n')==[spec['title'],source[0]]
        if family==7:return answer==source[0]+' '+spec['ending']
        if family==8:return answer.split('\n')==[str(i+1)+'. '+s for i,s in enumerate(source)]
        if family==9:return answer.split('\n')==source[:2]
    raise ValueError(kind)
