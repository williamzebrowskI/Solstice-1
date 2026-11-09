"""Check labels against the actual rendered prompts, not just generator metadata."""
import json
import re
from collections import Counter
from .lora_repair_common import SHARED, REPORT
from sml_v1.common import atomic_json, fingerprint

def check(row):
    spec=row['spec'];text='\n'.join(m['content'] for m in row['messages'][:-1] if m['role']=='user')
    answer=row['messages'][-1]['content'];kind=spec['kind']
    if kind=='facts':
        patterns=dict(contact=r'The contact is ([A-Za-z]+)\.', time=r'Pickup time is (\d+:\d+)\.',
                      quantity=r'There are (\d+) items\.', label=r'The label is ([a-z]+)\.',
                      location=r'The collection point is the ([a-z ]+)\.')
        parsed={k:re.search(p,text).group(1) for k,p in patterns.items() if re.search(p,text)}
        assert parsed==spec['values'], 'Rendered facts differ'
        missing=re.search(r'reply exactly "([^"]+)"',text).group(1)
        expected=parsed.get(spec['field'],missing)
        assert (row['component']=='abstention')==(spec['field'] not in parsed)
    elif kind=='latest':
        field=re.escape(spec['field'])
        patterns=[rf'the {field} is now ([^.]+)\.',rf'replace the old {field} with ([^.]+)\.',rf'The current {field} is ([^.]+)\.']
        matches=[m.group(1) for p in patterns for m in re.finditer(p,text)]
        assert len(matches)==1 and matches[0]!=spec['initial'], 'Ambiguous correction'
        expected=matches[0]
    elif kind=='layout':
        items=text.split(': ',1)[1].split('.\n',1)[0].split(', ')
        assert items==spec['values']
        f=spec['family']
        if f==0:expected='\n'.join('- '+x for x in sorted(items))
        elif f==1:expected='\n'.join(x.upper() for x in items)
        elif f==2:expected=', '.join(x.lower() for x in items)
        elif f==3:expected='\n'.join(str(i)+'. '+x for i,x in enumerate(items[::-1],1))
        elif f==4:expected=', '.join(items[:2])
        elif f==5:expected='"'+', '.join(sorted(items))+'"'
        elif f==6:expected=', '.join(x.upper() for x in items)
        else:expected='\n'.join(items)+'\nDone.'
    elif kind=='reasoning':
        values=list(map(int,re.findall(r'\b\d+\b',text)));assert len(values)==2
        a,b=values;f=spec['family']
        if f==0:expected=str(a+b)
        elif f==1:expected=str(a-b)
        elif f==2:expected=str(a*b)
        elif f==3:
            names=re.findall(r'([A-Z][a-z]+) (?:has|holds|owns) \d+ points',text)
            assert len(names)==2;expected=names[0] if a>b else names[1]
        elif f==4:assert a%b==0;expected=str(a//b)
        else:expected=str(max(a,b)-min(a,b))
    else:raise AssertionError(kind)
    assert answer==expected, 'Wrong training answer'
    assert row['verification']['answers']==[expected]
    return True


def audit():
    seen=set();counts={};families={}
    for split in ('train','dev','test'):
        rows=[json.loads(l) for l in (SHARED/'data'/(split+'.jsonl')).read_text().splitlines()]
        for row in rows:
            check(row);key=fingerprint(row['messages'][:-1])
            assert key not in seen, 'Duplicate prompt between splits'
            seen.add(key)
            assert not re.search(r'\b(?:python|sql|json)\b|```', ' '.join(m['content'] for m in row['messages']),re.I)
        counts[split]=len(rows);families[split]=dict(Counter(r['component']+':'+r['family'] for r in rows))
    result=dict(checked=counts,families=families,wrong_targets=0,duplicate_prompts=0,text_only=True)
    atomic_json(REPORT/'data_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='families'}))


if __name__=='__main__':audit()
