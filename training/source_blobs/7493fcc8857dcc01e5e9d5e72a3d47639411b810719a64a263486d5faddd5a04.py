"""Checked prose tasks with separate entities and wording across splits."""
import random
import string


def pair(prompt,answer):return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]


def make(split,component,count,seed):
    rng=random.Random(f'{seed}:{split}:{component}')
    banks={
        'train':('Ada Ben Cara Deni Elio Faye Gail Hari Iona Jori Kaia Levi Mena Nico Opal Pia Remi Sara Tali Uri Veda Wren Xavi Yara Zane'.split(),
                 'acorn birch candle daisy ember fabric garden harbor ink jasmine kettle lemon marble nutmeg olive pebble quartz robin stone tulip velvet willow yarn zebra'.split()),
        'dev':('Adele Bastian Corin Daria Enzo Freya Gideon Hilda Ilya Joelle'.split(),
               'anchor beacon copper driftwood elm fern granite holly iris juniper'.split()),
        'test':('Aisha Belen Casper Dalia Eamon Flora Goran Helga Idris Jacinta'.split(),
                'apricot boulder coral dune estuary flint glacier heather inlet jade'.split())}
    names,words=banks[split];style={'train':0,'dev':1,'test':2}[split]
    for i in range(count):
        name,other=rng.sample(names,2);word,second,third=rng.sample(words,3);a=rng.randrange(2,31);b=rng.randrange(1,a)
        prefix=('Please do this task. ','I need this result: ','Help with the following task: ')[style]
        if component=='skills':
            family=i%12
            if family==0:
                label=''.join(rng.choices(string.ascii_letters,k=rng.randrange(4,9)))+'-'+str(rng.randrange(100,9999))
                question=('Copy this label exactly: ','Repeat only the following identifier: ','Write this label unchanged, with no added words: ')[style]+label
                answer=label;operation='copy';inputs=[label]
            elif family in (1,2):
                value=''.join(rng.choice((c,c.upper())) for c in word+' '+second);upper=family==1
                question=prefix+f'Write "{value}" in {"uppercase" if upper else "lowercase"}. Return only the converted words.'
                answer=value.upper() if upper else value.lower();operation='upper' if upper else 'lower';inputs=[value]
            elif family==3:
                values=rng.sample(range(1,100),4);question=prefix+'Sort these numbers from smallest to largest; give only the comma-separated result: '+', '.join(map(str,values))
                answer=', '.join(map(str,sorted(values)));operation='sort';inputs=values
            elif family==4:
                values=[word,second,third];question=prefix+'Reverse this list; give only the comma-separated result: '+', '.join(values)
                answer=', '.join(values[::-1]);operation='reverse';inputs=values
            elif family==5:
                old,new=rng.sample(['round','square','wide','narrow','bright','dark','heavy','light'],2)
                sentence=f'{name} has a {old} basket.';question=prefix+f'Replace "{old}" with "{new}" in "{sentence}" Return only the changed sentence.'
                answer=sentence.replace(old,new);operation='replace';inputs=[sentence,old,new]
            elif family==6:
                question=prefix+f'{name} has {a} beads and receives {b} more. How many beads now? Give the number.'
                answer=str(a+b);operation='add';inputs=[a,b]
            elif family==7:
                question=prefix+f'There are {a} shells. {b} are taken away. How many shells remain? Give the number.'
                answer=str(a-b);operation='subtract';inputs=[a,b]
            elif family==8:
                a=rng.randrange(2,10);b=rng.randrange(2,7);question=prefix+f'{name} has {a} bags holding {b} coins each. How many coins in total? Give the number.'
                answer=str(a*b);operation='multiply';inputs=[a,b]
            elif family==9:
                question=prefix+f'{name} has {a} tokens and {other} has {b}. Who has more? Give only the name.'
                answer=name;operation='more';inputs=[name,a,other,b]
            elif family==10:
                values=rng.sample(range(1,40),5);question=prefix+'Keep only the even numbers, in their original order. Give a comma-separated result: '+', '.join(map(str,values))
                if not any(n%2==0 for n in values):values[0]=2;question=prefix+'Keep only the even numbers, in their original order. Give a comma-separated result: '+', '.join(map(str,values))
                answer=', '.join(str(n) for n in values if n%2==0);operation='even';inputs=values
            else:
                values=rng.sample(words,rng.randrange(3,8));question=prefix+'How many words are in this list? Give only the number: '+' '.join(values)
                answer=str(len(values));operation='word_count';inputs=values
            messages=pair(question,answer);verification=dict(kind='exact',answers=[answer],operation=operation,inputs=inputs,case_sensitive=True)
        else:
            family=i%5;number=rng.randrange(100,9999);new=rng.randrange(100,9999)
            if family==0:
                question=f'My collection number is {number}. Please remember it.'
                messages=pair(question,f'Your collection number is {number}.')+pair(f'Update: use {new} instead of {number}.',f'Your collection number is now {new}.')
                last=('Which collection number should you use now?','Tell me the updated collection number.','What collection number is currently correct?')[style]
                answer=f'The current collection number is {new}.';aliases=[answer,str(new)]
            elif family==1:
                question=f'{name} owns the {word} badge. {other} owns the {second} badge.'
                messages=pair(question,f'{name} has {word}; {other} has {second}.')
                last=f'Which badge belongs to {name}?';answer=f'{name} owns the {word} badge.';aliases=[answer,word,word+' badge']
            elif family==2:
                question=f'My name is {name}. Please remember that my preferred meeting room is {word}.'
                messages=pair(question,f'Your preferred meeting room is {word}.')+pair(f'I also need {a} chairs.','I have noted the number of chairs.')
                last='Which meeting room did I prefer?';answer=f'You preferred the {word} room.';aliases=[answer,word,word+' room']
            elif family==3:
                place,newplace=rng.sample(['locker','basket','drawer','suitcase','cabinet','crate','pouch','cupboard'],2)
                question=f'{name} put the {word} tag in the {place}.';messages=pair(question,f'The tag is in the {place}.')+pair(f'I moved that tag to the {newplace}.',f'The tag is now in the {newplace}.')
                last='Where is the tag now?';answer=f'The tag is in the {newplace}.';aliases=[answer,newplace,'the '+newplace]
            else:
                question=f'My name is {name}. I prefer the {word} design, while {other} prefers {second}.'
                messages=pair(question,f'You prefer {word}; {other} prefers {second}.')
                last='Which design do I prefer?';answer=f'You prefer the {word} design.';aliases=[answer,word,word+' design']
            messages+=pair(last,answer);verification=dict(kind='exact',answers=aliases,case_sensitive=False)
        yield dict(component=component,subset='verified_'+component,group=f'authored:{split}:{component}:{i}',question=messages[0]['content'],
                   messages=messages,split=split,verification=verification,provenance=dict(license='CC0-1.0',family=str(family),split_entities=split))


def recompute(v):
    op=v['operation'];x=v['inputs']
    if op=='copy':return x[0]
    if op=='upper':return x[0].upper()
    if op=='lower':return x[0].lower()
    if op=='sort':return ', '.join(map(str,sorted(x)))
    if op=='reverse':return ', '.join(x[::-1])
    if op=='replace':return x[0].replace(x[1],x[2])
    if op=='add':return str(x[0]+x[1])
    if op=='subtract':return str(x[0]-x[1])
    if op=='multiply':return str(x[0]*x[1])
    if op=='more':return x[0] if x[1]>x[3] else x[2]
    if op=='even':return ', '.join(str(n) for n in x if n%2==0)
    if op=='word_count':return str(len(x))
    raise ValueError(op)
