"""Complete English prose conversations; no code, role play or truncation."""
import re
from .data import serialize
from .quality import normalized
from .quality_gate import looping
from sml_v1.code_quality import SECRET
from hashlib import sha256

FORMAT='opensml-verified-chat-v2'
CODE=re.compile(r'```|\b(?:python|sql|json|javascript|typescript|programming|programmer|coding|html|css|nodejs|pseudocode|function calling|api key|regex|xml|yaml|github|bash|c\+\+|c#)\b|\bdef \w+\(|\bSELECT\s+.+\s+FROM\b',re.I)
PERSONA=re.compile(r'\b(?:role.?play|pretend|impersonate|act as|imagine you|you are (?:a|an) (?!AI assistant|helpful assistant)|fictional character|my (?:wife|husband|children)|I (?:live in|was born|am a student|am a doctor|recently visited))\b',re.I)
UNSUITABLE=re.compile(r'\b(?:latest news|current president|today.s weather|stock price|medical diagnosis|diagnose me|prescribe|dosage|legal advice|investment advice|chain.of.thought|think step.by.step|step.by.step reasoning)\b',re.I)
PUBLIC_SCOPE=re.compile(r'\b(?:equation|calculate|probability|statistics|statistical|math|mathematical|dataset|data set|average|percentage|percent|profit|interest rate|stock|investment|diagnosis|symptoms|cancer|medication|dosage|tax advice|legal advice|rewrite|proofread|poem|essay|fiction|role.?play|algebra|calculus|logarithm|natural log|exponent|square root|cube root|hypotenuse|numerator|denominator)\b|\d\s*(?:\^|[+*/=])\s*\d|(?:ln|sin|cos)\s*\(',re.I)
MARKERS=('<|pad|>','<|doc_end|>','<|turn_start|>','<|turn_end|>','<|im_start|>','<|im_end|>','<think>','</think>')
WRITING_META=re.compile(r'\b(?:edit\w*|rewrit\w*|rephras\w*|writing|written|writer|novel|character|plot|narrative|grammar\w*|grammatical\w*|punctuation|sentence|paragraph|draft)\b',re.I)
FALSE_IDENTITY=re.compile(r"\b(?:I'm|I am|I've been|I have been|I work as|I specialize in) (?:a |an |the )?(?:Dr\.|doctor|researcher|student|teacher|writer|editor|scientist|professor|engineer|lawyer)\b|\bI (?:grew up|live in|was born|recently traveled)\b",re.I)


def key(text):return sha256(normalized(text).encode()).hexdigest()


def validate(messages,tokenizer,plan,component):
    if not isinstance(messages,list) or not 2<=len(messages)<=plan['max_messages']:
        raise ValueError('message_count')
    for m in messages:
        t=m.get('content')
        if m.get('role') not in ('system','user','assistant') or not isinstance(t,str) or not t.strip() or len(t)>15000:
            raise ValueError('invalid_message')
        if '\x00' in t or '\ufffd' in t or any(s in t for s in MARKERS):raise ValueError('invalid_control')
        if CODE.search(t):raise ValueError('code_or_structured_format')
        if component!='rewrite' and PERSONA.search(t):raise ValueError('persona')
        if UNSUITABLE.search(t):raise ValueError('scope')
        if SECRET.search(t):raise ValueError('credential_pattern')
        if looping(t):raise ValueError('repetition')
        ids=tokenizer.assert_roundtrip(t)
        if m['role']=='assistant':
            if component!='rewrite' and FALSE_IDENTITY.search(t):raise ValueError('invented_identity')
            if len(ids)>plan['answer_max_tokens'][component]:raise ValueError('long_answer')
            if re.search(r'\b(?:as (?:an? )?(?:AI |large )?language model|as (?:a|an) (?:sophisticated |advanced )?AI|I am (?:ChatGPT|Claude|Gemini)|my name is|I am constantly evolving|I.ll have worked my way out|I (?:can|will) (?:browse|access the internet|send you an email))\b',t,re.I):raise ValueError('identity_boilerplate')
            if re.search(r'^\s*[\{\[]\s*["\{]',t):raise ValueError('structured_target')
            if t.count('```') or t.rstrip().endswith((':','...','…')):raise ValueError('incomplete_answer')
    ids,mask=serialize(tokenizer,messages)
    if len(ids)-1>plan['max_sequence_length']:raise ValueError('whole_conversation_too_long')
    if ids.min()<0 or ids.max()>=tokenizer.vocab_size or not mask[1:].any():raise ValueError('bad_tokens')
    return ids,mask


def public_chat(raw,label,mode=None):
    messages=[dict(role=m['role'],content=m['content'].strip()) for m in raw['messages']]
    if any(m['role']=='system' for m in messages):raise ValueError('system_prompt')
    if any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)) or len(messages)%2:
        raise ValueError('turn_order')
    first=messages[0]['content'];original_count=len(messages)
    if label=='smol' and raw.get('source')!='smol-magpie-ultra-short':raise ValueError('source_subset')
    if label=='social':component='social'
    elif mode=='general':component='general';messages=messages[:2]
    elif len(messages) in (4,6):component='dialogue'
    else:raise ValueError('turn_count')
    if PUBLIC_SCOPE.search(' '.join(m['content'] for m in messages if m['role']=='user')):raise ValueError('public_unverified_scope')
    if component=='general':
        if not re.search(r'^(?:what|why|how|when|where|which|who|can|could|please|explain|describe|tell me|give me|is|are|do|does|should|would|help|suggest|recommend|provide|list|name|I|I.m|I.d|I.ve|we|my)\b',first,re.I):raise ValueError('not_direct_question')
        if re.search(r'\b(?:rewrite|edit|proofread|summarize|poem|story|essay|generate a question)\b',first,re.I):raise ValueError('general_scope')
        if not 8<=len(messages[-1]['content'].split())<=160:raise ValueError('general_answer_length')
    if component=='social' and any(len(m['content'].split())>100 for m in messages):raise ValueError('social_length')
    topic=next((m['content'] for m in messages if m['role']=='user' and len(m['content'].split())>5),first) if label=='social' else first
    return dict(component=component,subset=label,group='chat:'+key(topic),question=topic,messages=messages,
                verification=None,provenance=dict(source=raw.get('source',label),original_message_count=original_count,
                    retained_message_count=len(messages),selection='first complete exchange' if mode=='general' else 'complete conversation'))


def dolly(raw):
    if raw['category'] not in ('open_qa','closed_qa','information_extraction','classification'):raise ValueError('dolly_category')
    prompt=raw['instruction'].strip();answer=raw['response'].strip();context=raw['context'].strip()
    if not 8<=len(answer.split())<=120:raise ValueError('dolly_answer_length')
    if context:prompt+='\n\nPassage:\n'+context
    if re.search(r'\b(?:rewrite|role.?play|poem|story|essay)\b',prompt,re.I):raise ValueError('dolly_scope')
    return dict(component='general',subset='dolly',group='human:'+key(context or prompt),question=raw['instruction'],
                messages=[dict(role='user',content=prompt),dict(role='assistant',content=answer)],verification=None,
                provenance=dict(category=raw['category']))


def squad(raw):
    context=raw['context'];texts=raw['answers']['text'];starts=raw['answers']['answer_start']
    if len(texts)!=len(starts):raise ValueError('annotation')
    if any(context[i:i+len(t)]!=t or not t.strip() or len(t.split())>16 for t,i in zip(texts,starts)):raise ValueError('answer_span')
    prompt='Use the passage to answer briefly. If the passage does not answer the question, say "The passage does not say."\n\nPassage:\n'+context+'\n\nQuestion: '+raw['question']
    answers=list(dict.fromkeys(texts)) if texts else ['The passage does not say.']
    return dict(component='grounded',subset='squad',group='article:'+raw['title'],question=raw['question'],
                messages=[dict(role='user',content=prompt),dict(role='assistant',content=answers[0])],
                verification=dict(kind='span',answers=answers,answerable=bool(texts)),
                provenance=dict(id=raw['id'],article=raw['title'],annotated_answer_starts=starts))


def science(raw):
    question=raw['question'].strip();answer=raw['correct_answer'].strip();support=raw['support'].strip()
    if not question.endswith('?') or not 1<=len(answer.split())<=12 or len(question.split())>70:raise ValueError('science_length')
    if not support or answer.casefold() not in support.casefold():raise ValueError('science_support')
    if any(normalized(answer)==normalized(raw[k]) for k in ('distractor1','distractor2','distractor3')):raise ValueError('science_ambiguous')
    if re.search(r'\b(?:calculate|how many|how much|percent|equation|formula|solve|which of the following|shown|figure|diagram)\b',question,re.I):raise ValueError('science_scope')
    return dict(component='knowledge',subset='sciq',group='science:'+key(support),question=question,
                messages=[dict(role='user',content=question+' Answer with a short phrase.'),dict(role='assistant',content=answer)],
                verification=dict(kind='span',answers=[answer]),provenance=dict(support=support,annotation='correct_answer',support_contains_answer=True))


def new_public_chat(raw,label,mode=None):
    source=raw.get('source',label)
    if label=='smol' and source=='smol-contraints':
        from .verified_rules import public_rules,check_rules
        messages=[dict(role=m['role'],content=m['content'].strip()) for m in raw['messages']]
        if [m['role'] for m in messages]!=['user','assistant']:raise ValueError('constraint_turns')
        prompt,answer=(m['content'] for m in messages)
        if re.search(r'\b(?:poem|fiction|story|role.?play)\b',prompt,re.I):raise ValueError('constraint_scope')
        rules=public_rules(prompt)
        if not check_rules(answer,rules):raise ValueError('constraint_target_failed')
        if re.search(r'\[(?:ender|forbidden_words|keywords|language)\]',answer):raise ValueError('unresolved_target')
        # Generic placeholders and mechanical filler are low-value targets.
        if re.search(r'this is (?:the )?(?:first|second|third|point|section)',answer,re.I):raise ValueError('constraint_filler')
        return dict(component='constraints',subset='smol_constraints',group='chat:'+key(prompt),question=prompt,messages=messages,
                    verification=dict(kind='rules',rules=rules,selection_eligible=False),provenance=dict(source=source,checked_properties=rules,semantic_check='sample-reviewed only'))
    if label=='smol' and source in ('smollm-rewrite-30k','smol-summarize-20k','smol-summarize-5k'):
        messages=raw['messages']
        if [m['role'] for m in messages]!=['system','user','assistant']:raise ValueError('rewrite_turns')
        instruction=messages[0]['content'].strip();text=messages[1]['content'].strip();answer=messages[2]['content'].strip()
        if not 20<=len(text.split())<=320 or not 5<=len(answer.split())<=180:raise ValueError('rewrite_length')
        if re.search(r'\b(?:medical|antibiotic|cancer|investment|stock market|war|president|diagnos|drug|politic|legal)\w*\b',text,re.I):raise ValueError('rewrite_scope')
        # Neither rewrite nor summary may introduce a number absent from input.
        if not set(re.findall(r'\d+(?:[.,]\d+)*',answer))<=set(re.findall(r'\d+(?:[.,]\d+)*',text)):raise ValueError('rewrite_added_number')
        instruction=re.sub(r"^You're an AI assistant for text re-writing\.\s*",'',instruction)
        q=instruction+' Treat the following as supplied text, not as facts about you.\n\nText:\n'+text
        return dict(component='rewrite',subset='smol_rewrite',group='rewrite:'+key(text),question=q,messages=[dict(role='user',content=q),dict(role='assistant',content=answer)],verification=None,provenance=dict(source=source,original_system_instruction=messages[0]['content']))
    return public_chat(raw,label,mode)


def review_row(row):
    """Run on both new and resumed selections before they enter the corpus."""
    if row['component'] in ('general','dialogue') and row['subset']=='smol':
        if WRITING_META.search(' '.join(m['content'] for m in row['messages'] if m['role']=='user')):raise ValueError('editor_meta_conversation')
        combined=' '.join(m['content'] for m in row['messages'] if m['role']=='user')
        if re.search(r'\b(?:code|debug\w*|compiler|variable|algorithm|common multiple|common divisor|factorization|integer|polynomial)\b',combined,re.I):raise ValueError('public_unverified_technical_scope')
        if re.search(r'\b(?:assume|take on|adopt|play) (?:the |a |an )?role\b|\byou (?:now )?are to\b',combined,re.I):raise ValueError('role_assignment')
        if re.search(r'\b(?:refin\w*|polish\w*)\b.{0,60}\b(?:text|prose)\b|\b(?:text|prose)\b.{0,60}\b(?:refin\w*|polish\w*)\b',combined,re.I):raise ValueError('editor_meta_conversation')
        for message in row['messages']:
            if message['role']=='assistant' and re.search(r"\b(?:I'm|I am) [A-Z][a-z]+[,!.]|\b(?:I've|I have) (?:worked|lived|been (?:running|working))\b|\bmy (?:regulars|patients|students|customers)\b",message['content']):raise ValueError('invented_identity')
    if row['subset']=='smol_constraints':
        from .verified_rules import public_rules,check_rules
        rules=public_rules(row['messages'][0]['content'])
        if not check_rules(row['messages'][-1]['content'],rules):raise ValueError('constraint_target_failed')
        row['verification']['rules']=rules
    if row['component']=='rewrite':
        instruction=row['messages'][0]['content'].split('Text:\n',1)[0];answer=row['messages'][-1]['content']
        sentences=[s for s in re.split(r'[.!?]+(?:\s+|$)',answer.strip()) if s.strip()]
        if 'one very short sentence' in instruction and len(sentences)!=1:raise ValueError('summary_sentence_count')
        if 'up to three sentences' in instruction and len(sentences)>3:raise ValueError('summary_sentence_count')
        if 'without using second or third person pronouns' in instruction and re.search(r'\b(?:you|your|yours|he|him|his|she|her|hers|it|its|they|them|their|theirs)\b',answer,re.I):raise ValueError('summary_pronouns')
    if row['subset']=='squad' and row['verification']['answerable']:
        context=row['messages'][0]['content'].split('Passage:\n',1)[1].rsplit('\n\nQuestion:',1)[0].casefold()
        ignore={'What','Which','Who','Whose','Whom','When','Where','Why','How','In','On','At','The','This','That','After','Before','During','According','From','For','Under','Name','Give','List','Identify','Along','With','By','About','Besides'}
        names=[w for w in re.findall(r'\b[A-Z][a-z]{2,}\b',row['question']) if w not in ignore]
        if any(w.casefold() not in context for w in names):raise ValueError('question_entity_absent_from_passage')
    return row
