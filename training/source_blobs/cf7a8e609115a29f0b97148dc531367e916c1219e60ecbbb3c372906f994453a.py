"""Frozen, versioned generated-answer diagnostics for SFT-040..054.

Social checks accept several natural phrasings, reject role inversion/personas,
and remain heuristics. Closed tasks and Python have explicit bounded oracles.
Every actual generated turn is checked. Development selects; test does not.
"""
from collections import defaultdict
import re
from pathlib import Path
from sml_v1.common import read_json
from .quality import repeated
from .natural_transfer import short_check
from .evidence_instructions import check_contract, value, exact, jcontract

VERSION='continuation-dialogue-v1'
SYSTEM='You are a helpful assistant. Answer the user directly and briefly. Ask a question when you need more information.'

def social(kind):
    return dict(kind='intent',intent=kind,max_words=45)

def check(c,text):
    if c['kind']!='intent': return short_check(c,text)
    if not text.strip() or repeated(text) or len(text.split())>c['max_words']:return False
    if re.search(r"\b(?:tornadoes|earthquake|revised version|grammar tutor|my name is|I live|my family)\b",text,re.I):return False
    if re.search(r"\byou (?:can|could) help me\b|\byou are useful\b",text,re.I):return False
    kind=c['intent']
    if kind=='greeting':
        return bool(re.search(r'\b(?:hello|hi|hey|welcome|sure|of course)\b',text,re.I)) and not re.search(r'\b(?:goodbye|bye)\b',text,re.I)
    if kind=='casual':
        return bool(re.search(r"\b(?:here|ready|chat|talk|help|questions|mind|discuss)\b",text,re.I)) and not re.search(r'\b(?:goodbye|bye|your friend|a friend)\b',text,re.I)
    if kind=='capability':
        return bool(re.search(r"\b(?:I can|I help|I am here|I'm here)\b",text,re.I)) and bool(re.search(r'\b(?:questions|writing|text|explain|Python|code)\b',text,re.I))
    if kind=='thanks':return bool(re.search(r"welcome|glad|happy to help|no problem|anytime",text,re.I))
    if kind=='farewell':return bool(re.search(r'\b(?:bye|goodbye|see you|take care|later|good night)\b',text,re.I)) and not re.search(r'\b(?:what would you|how can I help)\b',text,re.I)
    raise ValueError(kind)

def cases(split):
    """Distinct concrete records and wordings; recurring skill families disclosed."""
    assert split in ('dev','test')
    dev=split=='dev';rows=[]
    def add(bucket,name,turns):
        rows.append(dict(group=f'{VERSION}:{split}:{name}',bucket=bucket,turns=[dict(prompt=q,gold=a,contract=c) for q,a,c in turns]))
    social_q=[
      ('Good afternoon, assistant.','Hello! How can I help?', 'greeting'),
      ('Got time for a light conversation?','Yes, I am here to chat.','casual'),
      ('What is going on with you in this conversation?','I am here and ready to help with your questions.','casual'),
      ('Tell me briefly what you can assist with.','I can answer questions, explain ideas, help with text and simple Python.','capability'),
      ('Thanks for spending time on that.','You are welcome!','thanks'),
      ('I am heading out. Until next time.','See you later!','farewell')
    ] if dev else [
      ('Hi assistant, nice to meet you.','Hello! How can I help?','greeting'),
      ('Would you like to talk for a moment?','Sure, I am here to chat.','casual'),
      ('Anything happening on your end?','I am here and ready to help.','casual'),
      ('Give me a quick description of how you help people.','I can explain topics, answer questions and help with text and Python.','capability'),
      ('Much appreciated, that was useful.','Glad it helped!','thanks'),
      ('I will leave the chat now. Catch you later.','Goodbye! Take care.','farewell')]
    for i,(q,a,k) in enumerate(social_q):add('social',str(i),[(q,a,social(k))])
    clarify=[('Can you proofread a letter I wrote?','Please share the letter.','letter'),('Something in my Python program crashes.','Please share the code and error message.','code'),('Please explain this paragraph for me.','Please paste the paragraph.','paragraph'),('Turn my notes into a shorter message.','Please share the notes.','notes')] if dev else [('Can you improve my draft email?','Please share the email.','email'),('Help debug a script that will not run.','Please share the script and error.','script'),('Summarize the article I am reading.','Please share the article.','article'),('I want the announcement made clearer.','Please paste the announcement.','announcement')]
    for i,(q,a,obj) in enumerate(clarify):add('clarification',str(i),[(q,a,dict(kind='request',objects=[obj]))])
    person,other,item,place,num=('Demi','Pavel','parcel','shelf',47) if dev else ('Zola','Enzo','folder','drawer',63)
    qa=[(f'{person} stores a {item} on the {place}. {other} has an empty bag. Where is the {item}?',place),
        (f'{person} has code {num}. No collection time is recorded. When is collection?', 'Unknown'),
        (f'The file assigns code {num} to {person}. Who has code {num}?',person),
        (f'The {item} is in the {place}, not the bag. Which location contains it?',place)]
    for i,(q,a) in enumerate(qa):add('grounding',str(i),[(q+' Give only the requested value; say Unknown if it is not given.',a,value(a))])
    word,word2=('saffron','maple') if dev else ('indigo','cedar')
    add('format','upper',[(f'Write only the uppercase version of {word}.',word.upper(),exact(word.upper()))])
    add('format','lower',[(f'Write only the lowercase version of {word.upper()}.',word,exact(word))])
    add('format','lines',[(f'Put {word} then {word2} on two lines, no bullets.',word+'\n'+word2,exact(word+'\n'+word2))])
    add('format','json',[(f'Give just a JSON object with key "color" and value "{word}".', '{"color": "'+word+'"}',jcontract(dict(color=word)))])
    project,day1,day2=('Orchid','Monday','Wednesday') if dev else ('Juniper','Saturday','Tuesday')
    add('context','retain',[(f'Our project is {project}. State only its name.',project,value(project)),('What was the project name? Give only that name.',project,value(project))])
    add('context','correct',[(f'{person} will arrive {day1}. Give only the day.',day1,value(day1)),(f'Correction: {person} arrives {day2}, not {day1}. What is the new day? Give only the day.',day2,value(day2))])
    add('context','case',[(f'Reply only with {word.upper()}.',word.upper(),exact(word.upper())),('Now give that word in lowercase, with no other text.',word,exact(word))])
    add('context','clarify',[(('Can you lowercase a sentence for me?' if dev else 'Would you lowercase a sentence I have?'),'Please share the sentence.',dict(kind='request',objects=['sentence'])),(f'The sentence is: {project.upper()} IS HERE. Give only its lowercase form.',project.lower()+' is here.',exact(project.lower()+' is here.'))])
    # Explicit outputs prevent keyword-matching a correct first clause followed
    # by false explanations. Independent facts, not the user's inspection set.
    facts=[('How many minutes are in one hour? Answer with only the number.','60'),('What organ do people use to breathe? Answer with only its plural name.','lungs'),('Is salt dissolving in water a physical or chemical change? Give only physical or chemical.','physical'),('What gas do plants take in for photosynthesis? Give only the gas name.','carbon dioxide')] if dev else [('How many months are in one year? Answer with only the number.','12'),('Which planet is closest to the Sun? Give only its name.','Mercury'),('How many sides does a hexagon have? Give only the number.','6'),('What gas makes up most of Earth\'s atmosphere? Give only the gas name.','nitrogen')]
    for i,(q,a) in enumerate(facts):add('facts',str(i),[(q,a,value(a))])
    fn='resolve' if dev else 'transform'
    specs=[('return text in uppercase, preserving all spaces','text','text.upper()',[dict(args=[' aB '],expected=' AB '),dict(args=[''],expected=''),dict(args=['xy'],expected='XY')]),
      ('return numbers strictly greater than threshold, in original order','values, threshold','[x for x in values if x > threshold]',[dict(args=[[1,2,2,3],2],expected=[3]),dict(args=[[],2],expected=[]),dict(args=[[-2,0,-1],-1],expected=[0])]),
      ('remove leading and trailing spaces without altering case','text','text.strip()',[dict(args=[' Ab '],expected='Ab'),dict(args=[''],expected=''),dict(args=['Xy'],expected='Xy')]),
      ('return the last item or None for an empty list','items','items[-1] if items else None',[dict(args=[[0,3]],expected=3),dict(args=[[]],expected=None),dict(args=[[None]],expected=None)])] if dev else [
      ('return text in lowercase, preserving all spaces','text','text.lower()',[dict(args=[' aB '],expected=' ab '),dict(args=[''],expected=''),dict(args=['XY'],expected='xy')]),
      ('return numbers at least threshold, in original order','values, threshold','[x for x in values if x >= threshold]',[dict(args=[[1,2,2,3],2],expected=[2,2,3]),dict(args=[[],2],expected=[]),dict(args=[[-2,0,-1],-1],expected=[0,-1])]),
      ('remove trailing spaces while preserving leading spaces and case','text','text.rstrip()',[dict(args=[' Ab '],expected=' Ab'),dict(args=[''],expected=''),dict(args=[' Xy'],expected=' Xy')]),
      ('return the first item or None for an empty list','items','items[0] if items else None',[dict(args=[[0,3]],expected=0),dict(args=[[]],expected=None),dict(args=[[None]],expected=None)])]
    for i,(spec,args,body,tests) in enumerate(specs):
        a=f'```python\ndef {fn}({args}):\n    return {body}\n```'
        add('python',str(i),[(f'Implement Python {fn}({args}) to {spec}. Return just a Python code block.',a,dict(kind='python',function=fn,tests=tests))])
    assert all(check(t['contract'],t['gold']) for r in rows for t in r['turns'])
    return rows

def panel(model,tok,rows,system=None):
    from .evaluate import generate
    scores=defaultdict(list);samples=[]
    for row in rows:
        history=[dict(role='system',content=system)] if system else [];turns=[]
        for t in row['turns']:
            history.append(dict(role='user',content=t['prompt']))
            result=generate(model,tok,history,limit=160)
            ok=check(t['contract'],result['text']) and result['stop_reason'] in ('turn_end','document_end')
            turns.append(dict(prompt=history.copy(),contract=t['contract'],result=result,passed=bool(ok)))
            history.append(dict(role='assistant',content=result['text']))
        passed=all(t['passed'] for t in turns);scores[row['bucket']].append(passed)
        samples.append(dict(group=row['group'],bucket=row['bucket'],turns=turns,passed=passed))
    rates={k:sum(v)/len(v) for k,v in scores.items()}
    return dict(rates=rates,macro_pass=sum(rates.values())/len(rates),samples=samples)

def run(model,tok,prepared,plan,split='dev'):
    rows=read_json(Path(prepared)/('trial-'+split+'.json'))['cases']
    results={v:panel(model,tok,rows,system) for v,system in [('no-system',None),('system',SYSTEM)]}
    return dict(metrics=results,macro_pass=sum(p['macro_pass'] for p in results.values())/2,scope='Small category-balanced strict generated-answer diagnostic; heuristic social scoring; actual previous replies; not benchmark/general-chat accuracy.')

def ranking(result,plan):
    p=result['trial_dev'];retained=result['retention']['val_loss']<=plan['parent_val_loss']+plan['max_retention_increase']
    # Floor is provisional, not final acceptance. Independent test + raw review
    # decide whether weights are retained after each attempt.
    covered=all(v['rates']['social']>=.5 and v['rates']['python']>=.25 for v in p['metrics'].values())
    return [int(retained),int(retained and covered),p['macro_pass'],-result['assistant_nll']]
