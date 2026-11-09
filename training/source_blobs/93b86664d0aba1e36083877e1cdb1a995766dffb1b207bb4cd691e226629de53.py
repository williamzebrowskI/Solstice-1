"""Conservative reference filters for short, untruncated upstream messages."""
import re
from .balanced_corpus import complete
from .quality_gate import looping,unchanged_edit_claim

OUT_OF_SCOPE=re.compile(r'\b(?:you (?:are|own|run|work as|teach as) (?:a|an|the)|role.?play|pretend|impersonat\w*|act as|calculus|lagrange|differential|cryptocurrency|diagnos\w*|prescription|chemotherapy|self.harm|suicid\w*)\b|\\(?:frac|sum|begin)\b',re.I)
ROLE_ASSIGNMENT=re.compile(r"\byou(?: are|'re|\u2019re)\b|\bas (?:a|an) (?:expert|doctor|teacher|lawyer|salesman|shopkeeper)\b",re.I)
INVENTORY_CLAIM=re.compile(r"\b(?:we (?:have|sell|offer|stock)|our (?:shop|store|inventory)|in my (?:shop|store)|i(?:'ve| have) (?:spent years|worked|lived)|i (?:work as|own|run))\b",re.I)
TASK_START=re.compile(r'^(?:what|why|who|where|when|how|can|could|would|please|write|explain|summari[sz]e|rewrite|correct|list|give|describe|define|compare|name|suggest|provide|help|calculate|find|identify|classify|extract|return|convert|translate|make|tell|sort|answer)\b',re.I)
PERSONA=re.compile(r"\b(?:my (?:wife|husband|children|grandmother|grandfather)|i live in|i was born|i am (?:a|an) (?:doctor|teacher|lawyer|student)|i(?:'m| am) (?:chatgpt|claude)|i (?:changed|replaced)|i made (?:several|a few|the following) changes)\b",re.I)
UNSUPPORTED=re.compile(r'\b(?:placeholders?|highlights?|postscript|keywords?|paragraphs?|quotation|frequency|repeat|exact phrase|no other words|letter [a-z] should|word .* should appear|include the word|include keywords|do not include|without using|finish your response|end your response|start your response)\b|P\.S\.',re.I)


def reference_ok(text):
    if looping(text) or unchanged_edit_claim(text) or PERSONA.search(text):return False
    if complete(text):return True
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    return len(lines)>=2 and all(re.match(r'^(?:[-*+]\s+|\d+[.)]\s+)\S',s) and not s.endswith((':',',',';','-')) for s in lines)


def constraints(prompt,answer):
    """Accept simple source constraints only; reject unsupported combinations."""
    if UNSUPPORTED.search(prompt):return None
    checks=[]
    for mode,amount,unit in re.findall(r'(exactly|at least|at most|less than|fewer than|more than)\s+(\d+)\s+(bullet points?|sentences?|words?)',prompt,re.I):
        mode=mode.lower();n=int(amount);unit=unit.lower()
        bounds=({'min':n,'max':n} if mode=='exactly' else {'min':n} if mode=='at least' else
                {'max':n} if mode=='at most' else {'min':n+1} if mode=='more than' else {'max':n-1})
        checks.append(dict(kind='length',unit='bullets' if unit.startswith('bullet') else 'words' if unit.startswith('word') else 'sentences',**bounds))
    if re.search(r'all (?:lowercase|lower case)|no capital letters',prompt,re.I):checks.append(dict(kind='case',mode='lower'))
    if re.search(r'all (?:uppercase|upper case|capital letters)|no lowercase letters',prompt,re.I):checks.append(dict(kind='case',mode='upper'))
    # Upstream tasks can embed their own reference in the request; skip these.
    if len(answer.split())>12 and ' '.join(answer.split()).casefold() in ' '.join(prompt.split()).casefold():return None
    return checks or None
