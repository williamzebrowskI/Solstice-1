"""Computed tasks with split-specific entities and disjoint instruction wording."""
import random
import string

FAMILIES=dict(skills=24,constraints=16,dialogue=10)
BANKS={
 'train':('Ada Ben Cara Deni Elio Faye Gail Hari Iona Jori Kaia Levi Mena Nico Opal Pia Remi Sara Tali Uri Veda Wren Xavi Yara Zane'.split(), 'acorn birch candle daisy ember fabric garden harbor ink jasmine kettle lemon marble nutmeg olive pebble quartz robin stone tulip velvet willow yarn zebra'.split()),
 'dev':('Adele Bastian Corin Daria Enzo Freya Gideon Hilda Ilya Joelle'.split(),'anchor beacon copper driftwood elm fern granite holly iris juniper'.split()),
 'test':('Aisha Belen Casper Dalia Eamon Flora Goran Helga Idris Jacinta'.split(),'apricot boulder coral dune estuary flint glacier heather inlet jade'.split())}
OPS=['copy','upper','lower','title','sort','descending','reverse','replace','add','subtract','multiply','even','odd','minimum','maximum','sum','count','first','last','alphabetical','unique','starts','rule','compare']
INSTRUCTIONS={
 'copy':['Copy this label exactly','Return this identifier unchanged','Reproduce this label','Make an identical copy of this label','Keep every letter and digit unchanged','Duplicate this identifier without modifications'],
 'upper':['Convert to uppercase','Write in all capital letters','Change these letters to uppercase','Make all letters capital','Transform the lettering to uppercase','Use only capital letters for this text'],
 'lower':['Convert to lowercase','Write in all lowercase letters','Change these letters to lowercase','Make all letters lowercase','Transform the lettering to lowercase','Use only lowercase letters for this text'],
 'title':['Capitalize the first letter of each word','Convert to title case','Use title case for these words','Write these words with initial capitals','Make each word start with a capital and keep its other letters lowercase','Use one initial capital letter per word'],
 'sort':['Sort these numbers from smallest to largest','Put these values in ascending order','Arrange the numbers in increasing order','Sort the numbers in ascending order','Order the values starting with the least and ending with the greatest','Return an increasing sequence of these values'],
 'descending':['Sort these numbers from largest to smallest','Put these values in descending order','Arrange the numbers in decreasing order','Sort the numbers in descending order','Order the values starting with the greatest and ending with the least','Return a decreasing sequence of these values'],
 'reverse':['Reverse this list','Write these items in reverse order','Reverse the order of these words','Put the last item first and continue backward','Read the entries from the end toward the beginning','Return these entries in the opposite order'],
 'even':['Keep only even numbers','Filter this list for even numbers','Select the even values','Remove the odd numbers','Retain the values divisible by two','Choose every value with no remainder when divided by two'],
 'odd':['Keep only odd numbers','Filter this list for odd numbers','Select the odd values','Remove the even numbers','Retain the values not divisible by two','Choose every value with a remainder of one when divided by two'],
 'minimum':['Find the smallest number','Give the minimum value','Which number is smallest','Select the lowest value','Identify the least value in this collection','Return the value below all the others'],
 'maximum':['Find the largest number','Give the maximum value','Which number is largest','Select the highest value','Identify the greatest value in this collection','Return the value above all the others'],
 'sum':['Add these numbers','Give the sum of these values','Find the total','Calculate the sum','Combine these amounts into one total','What do all these values add up to'],
 'count':['Count the words','How many words are in this list','Give the number of words','Count all the entries','Determine the number of items shown','What is the size of this collection'],
 'first':['Keep the first two words','Return the first two entries','Give the first two list items','Select the two words at the beginning','Retain only entries one and two','Choose the opening pair of items'],
 'last':['Keep the last two words','Return the final two entries','Give the last two list items','Select the two words at the end','Retain only the two final entries','Choose the closing pair of items'],
 'alphabetical':['Put these words in alphabetical order','Sort these words alphabetically','Arrange the words from A to Z','Alphabetize this list','Order the entries using dictionary order','Return these items sorted in A-to-Z order'],
 'unique':['Remove duplicates, keeping first occurrences','Keep each word only once, in first-seen order','Drop repeated items without sorting','Return the distinct words in their original order','Preserve the first occurrence of every word and remove later copies','Deduplicate the list while retaining its initial ordering']}


def phrase(rng,split,options):return rng.choice(options[:4]) if split=='train' else options[4 if split=='dev' else 5]
def pair(q,a):return [dict(role='user',content=q),dict(role='assistant',content=a)]


def compute(op,x):
    if op=='copy':return x[0]
    if op in ('upper','lower','title'):return getattr(x[0],op)()
    if op in ('sort','descending'):return ', '.join(map(str,sorted(x,reverse=op=='descending')))
    if op=='reverse':return ', '.join(x[::-1])
    if op=='replace':return x[0].replace(x[1],x[2])
    if op=='add':return str(x[0]+x[1])
    if op=='subtract':return str(x[0]-x[1])
    if op=='multiply':return str(x[0]*x[1])
    if op in ('even','odd'):return ', '.join(str(n) for n in x if n%2==int(op=='odd'))
    if op=='minimum':return str(min(x))
    if op=='maximum':return str(max(x))
    if op=='sum':return str(sum(x))
    if op=='count':return str(len(x))
    if op=='first':return ', '.join(x[:2])
    if op=='last':return ', '.join(x[-2:])
    if op=='alphabetical':return ', '.join(sorted(x))
    if op=='unique':return ', '.join(dict.fromkeys(x))
    if op=='starts':return ', '.join(w for w in x[0] if w.startswith(x[1]))
    if op=='rule':return x[1] if x[0]%2==0 else x[2]
    if op=='compare':return x[0] if x[1]>x[3] else x[2]
    raise ValueError(op)


def recompute(v):
    if v.get('operation'):return compute(v['operation'],v['inputs'])
    # Non-arithmetic tasks carry their source facts for independent auditing.
    return v['answers'][0]


def make(split,component,count,seed):
    rng=random.Random(f'{seed}:{split}:{component}');names,bank=BANKS[split]
    for i in range(count):
        family=i%FAMILIES[component];name,other,third=rng.sample(names,3);w,w2,w3,w4=rng.sample(bank,4)
        a=rng.randrange(2,60);b=rng.randrange(1,a);items=rng.sample(bank,rng.randrange(3,7))
        only=phrase(rng,split,['Give only the answer.','Return only the result.','Do not add an explanation.','Write the answer with no introduction.','Your reply should contain the result and nothing else.','Respond with just the requested result.'])
        v=dict(kind='exact',case_sensitive=True)
        if component=='skills':
            op=OPS[family]
            if op=='copy':x=[''.join(rng.choices(string.ascii_letters,k=rng.randrange(4,10)))+'-'+str(rng.randrange(10,9999))]
            elif op in ('upper','lower','title'):x=[''.join(rng.choice((c,c.upper())) for c in ' '.join(items[:rng.randrange(1,4)]))]
            elif op in ('sort','descending','minimum','maximum','sum','even','odd'):
                x=rng.sample(range(1,80),rng.randrange(3,7))
                if op=='even' and not any(n%2==0 for n in x):x[0]=2
                if op=='odd' and not any(n%2 for n in x):x[0]=3
            else:x=items if op!='unique' else [w,w2,w,w3,w2]
            if op in INSTRUCTIONS:
                q=phrase(rng,split,INSTRUCTIONS[op])+': '+(', '.join(map(str,x)) if len(x)>1 else str(x[0]))
                if op in ('even','odd','first','last','unique','starts'):q+='. Preserve the original order.'
                if op in ('sort','descending','reverse','even','odd','first','last','alphabetical','unique'):q+=' Separate items with commas.'
            elif op in ('add','subtract','multiply'):
                x=[a,b] if op!='multiply' else [rng.randrange(2,13),rng.randrange(2,10)];aa,bb=x
                if op=='add':variants=[f'What is {aa} plus {bb}?',f'Add {aa} and {bb}.',f'{name} has {aa} beads and gets {bb} more. How many now?',f'A basket has {aa} shells. Another {bb} are added. How many shells?',f'{name} collected {aa} stamps and then another {bb}. What is the total?',f'Find the combined count of {aa} objects and {bb} additional objects.']
                elif op=='subtract':variants=[f'What is {aa} minus {bb}?',f'Subtract {bb} from {aa}.',f'{name} has {aa} beads and gives away {bb}. How many remain?',f'A tray holds {aa} shells. Remove {bb}. How many remain?',f'{name} started with {aa} stamps and used {bb}. What is left?',f'There were {aa} objects before {bb} were taken. Find the remaining count.']
                else:variants=[f'What is {aa} times {bb}?',f'Multiply {aa} by {bb}.',f'{name} has {aa} bags with {bb} coins each. How many coins?',f'There are {aa} rows of {bb} chairs. How many chairs?',f'{name} packs {aa} boxes with {bb} stones apiece. What is the total?',f'Count all objects in {aa} equal groups of {bb} objects.']
                q=phrase(rng,split,variants)
            elif op=='replace':
                x=[f'{name} put the {w} beside the {w2}.',w,w3]
                q=phrase(rng,split,[f'Replace "{w}" with "{w3}" in this sentence: ',f'Change only "{w}" to "{w3}": ',f'Use "{w3}" instead of "{w}" here: ',f'Edit by replacing "{w}" with "{w3}": ',f'Substitute "{w3}" for "{w}" while preserving everything else: ',f'Keep the sentence intact except for changing "{w}" into "{w3}": '])+x[0]
            elif op=='starts':
                x=[items,items[0][0]];letter=x[1]
                q=phrase(rng,split,[f'Keep words starting with {letter}: ',f'Select words beginning with {letter}: ',f'Filter for words whose first letter is {letter}: ',f'List words that start with {letter}: ',f'Retain entries with initial letter {letter}: ',f'Which entries have {letter} at the beginning? '])+', '.join(items)+'. Preserve their order and separate them with commas.'
            elif op=='rule':
                x=[a,w,w2]
                q=phrase(rng,split,[f'If the number is even, say {w}; otherwise say {w2}. Number: {a}.',f'Use {w} for even and {w2} for odd. Number: {a}.',f'Apply this rule to {a}: even means {w}, odd means {w2}.',f'Number {a}. Answer {w} when even or {w2} when odd.',f'For {a}, select {w} if divisible by two; otherwise select {w2}.',f'Choose a label for {a}: {w2} denotes odd values; {w} denotes even values.'])
            else:
                x=[name,a,other,b] if rng.random()<.5 else [name,b,other,a]
                q=phrase(rng,split,[f'{name} has {x[1]} tokens. {other} has {x[3]}. Who has more?',f'Who has more: {name} with {x[1]} shells or {other} with {x[3]}?',f'Compare {name}: {x[1]} and {other}: {x[3]}. Name the person with more.',f'{name} has {x[1]} cards and {other} has {x[3]}. Who has more?',f'Which person owns more items, {name} ({x[1]}) or {other} ({x[3]})?',f'Between {name} with {x[1]} and {other} with {x[3]}, identify the owner of the bigger set.'])
            answer=compute(op,x);v.update(operation=op,inputs=x);messages=pair(q+' '+only,answer)
        elif component=='constraints':
            facts=[f'{name} brought {a} {w} badges.',f'{other} carried {b} {w2} cards.',f'{third} chose the {w3} design.',f'The meeting is in the {w4} room.']
            n=rng.randrange(2,5);given=facts[:n];end=f'{w.title()} report complete.'
            if family==0:instruction=f'Use exactly {n} bullet lines beginning with "- ", one sentence per line.';answer='\n'.join('- '+s for s in given)
            elif family==1:instruction=f'Write the {n} sentences in one paragraph, separated by single spaces.';answer=' '.join(given)
            elif family==2:instruction=f'Put each sentence in a separate paragraph, using one blank line between paragraphs.';answer='\n\n'.join(given)
            elif family==3:instruction='Return all sentences in lowercase, separated by spaces.';answer=' '.join(given).lower()
            elif family==4:instruction='Return all sentences in uppercase, separated by spaces.';answer=' '.join(given).upper()
            elif family==5:instruction='Return the first sentence inside one pair of double quotation marks.';answer='"'+given[0]+'"'
            elif family==6:instruction=f'Write the title <<{w.title()} notes>>, then a blank line, then the first sentence.';answer=f'<<{w.title()} notes>>\n\n'+given[0]
            elif family==7:instruction=f'Write the first sentence followed by a space and this exact ending: {end}';answer=given[0]+' '+end
            elif family==8:instruction=f'Number the sentences from 1 to {n}, one per line, using "1. " style numbering.';answer='\n'.join(f'{j+1}. {s}' for j,s in enumerate(given))
            elif family==9:instruction=f'Write only the sentence mentioning {other}.';answer=facts[1]
            elif family==10:instruction=f'Write only the first sentence, changing {w} to {w3}.';answer=given[0].replace(w,w3)
            elif family==11:instruction='Give the first two sentences on separate lines with no blank line.';answer='\n'.join(given[:2])
            elif family==12:instruction='Write SECTION 1 on a line, then the first sentence. Leave a blank line, then write SECTION 2 and the second sentence on separate lines.';answer='SECTION 1\n'+given[0]+'\n\nSECTION 2\n'+given[1]
            elif family==13:instruction=f'Write the first sentence. Then on a new line add this postscript exactly: P.S. Bring the {w4} folder.';answer=given[0]+f'\nP.S. Bring the {w4} folder.'
            elif family==14:instruction=f'Use exactly {n} bullet lines starting with "* ", one sentence per line, and make all letters lowercase.';answer='\n'.join('* '+s.lower() for s in given)
            else:instruction=f'Write the first two sentences in lowercase, separated by one space. End with a new line containing exactly: {w4}';answer=' '.join(given[:2]).lower()+'\n'+w4
            # Each split has distinct task wording as well as distinct entities.
            if split=='dev':instruction=instruction.replace('Write','Provide').replace('Return','Give').replace('Use exactly','Include precisely').replace('Put each','Place every')
            elif split=='test':instruction=instruction.replace('Write','Produce').replace('Return','Output').replace('Use exactly','Create precisely').replace('Put each','Arrange every')
            q=phrase(rng,split,['Format these supplied sentences. ','Reformat this text. ','Apply the formatting request. ','Use the supplied text for this formatting task. ','Arrange the source text according to the directions. ','Transform the presentation of this text as requested. '])+instruction+' Do not add other text. Preserve the original wording except for requested changes.\n\nSource:\n'+'\n'.join(given)
            messages=pair(q,answer);v.update(kind='verbatim_format',source_facts=given,family=family)
        else:
            num=rng.randrange(100,9999);new=rng.randrange(100,9999)
            while new==num:new=rng.randrange(100,9999)
            if family==0:messages=pair(f'My collection number is {num}.',f'Your collection number is {num}.')+pair(f'Correction: use {new}, not {num}.',f'The corrected number is {new}.');last='What is the current collection number?';answer=str(new)
            elif family==1:messages=pair(f'{name} has the {w} badge. {other} has the {w2} badge.',f'{name}: {w}; {other}: {w2}.');last=f'Which badge belongs to {name}?';answer=w
            elif family==2:messages=pair(f'I prefer the {w} room.',f'Your preferred room is {w}.')+pair(f'We need {a} chairs.',f'I have noted {a} chairs.');last='What is my preferred room?';answer=w
            elif family==3:messages=pair(f'The {w} tag is in the drawer.','The tag is in the drawer.')+pair(f'I moved it to the {w2} box.',f'The tag is now in the {w2} box.');last='Where is the tag now?';answer=f'the {w2} box'
            elif family==4:messages=pair(f'My name is {name}. I prefer {w}, while {other} prefers {w2}.',f'You prefer {w}; {other} prefers {w2}.');last='Which design do I prefer?';answer=w
            elif family==5:messages=pair(f'A {w} badge costs {a} tokens.',f'Each badge costs {a} tokens.')+pair(f'Correction: each badge costs {b} tokens.',f'The corrected price is {b} tokens.');last='How many tokens do two badges cost at the corrected price?';answer=str(2*b)
            elif family==6:messages=pair(f'Answer with one word when I ask for a label. My label is {w}.','Understood.')+pair(f'{other} uses {w2}.','Noted.');last='What is my label?';answer=w
            elif family==7:
                day,newday=rng.sample(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],2)
                messages=pair(f'The plan uses the {w} room on {day}.',f'The plan is {w} on {day}.')+pair(f'Change the day to {newday}, keeping the room unchanged.',f'The plan is now {w} on {newday}.');last='Give the room and day, separated by a comma.';answer=w+', '+newday
            elif family==8:messages=pair(f'{name} chose the {w} folder. No deadline has been mentioned.',f'{name} chose {w}; the deadline is unspecified.');last='What is the deadline?';answer='The deadline has not been provided.'
            else:messages=pair(f'{name} has {a} {w} cards and {other} has {b} {w2} cards.',f'{name}: {a}; {other}: {b}.')+pair(f'Only {name} gives away 1 card.',f'{name} now has {a-1} cards.');last=f'How many cards does {other} have?';answer=str(b)
            # Holdout questions change their form, rather than only a preamble.
            if split=='dev':last=last.replace('What is','Tell me').replace('Which','Identify which').replace('Give','Return').replace('How many','Specify how many').replace('Where is','Tell me where')
            elif split=='test':last=last.replace('What is','Retrieve').replace('Which','Remind me which').replace('Give','List').replace('How many','Find how many').replace('Where is','Find where')
            else:last=rng.choice([last,'Please answer: '+last,'Using our conversation, '+last[0].lower()+last[1:],last+' Use the most recent information.'])
            messages+=pair(last+' '+only,answer);v['case_sensitive']=False
        v['answers']=[answer]
        yield dict(component=component,subset='verified_'+component,group=f'authored:{split}:{component}:{i}',question='\n'.join(m['content'] for m in messages if m['role']=='user'),messages=messages,split=split,verification=v,
                   provenance=dict(license='CC0-1.0',family=str(family),split_entities=split,template_partition=split))
