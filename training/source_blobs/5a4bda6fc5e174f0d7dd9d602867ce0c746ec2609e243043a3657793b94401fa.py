"""Short complete dialogue selection and a modest authored conversational bridge."""
from .balanced import pair
from .conversation import curriculum
from .quality import digest


def complete_prefix(messages, tokenizer, plan, component='chat'):
    """Keep complete initial exchanges; never shorten an assistant's answer."""
    body=[dict(m) for m in messages if m['role']!='system']
    limit=plan.get('python_max_tokens',256) if component=='python' else plan['assistant_max_tokens']
    result=[]
    for i in range(0,min(len(body),2*plan.get('max_dialogue_turns',4)),2):
        if i+1>=len(body):break
        user,answer=body[i:i+2]
        if user['role']!='user' or answer['role']!='assistant':raise ValueError('nonalternating_dialogue')
        if len(tokenizer.encode(answer['content']))>limit:break
        if len(user['content'].split())>140:break
        result.extend((user,answer))
    if not result:raise ValueError('no_complete_short_exchange')
    return result


def presentation(messages, system, group):
    """One presentation per conversation, deterministically with/without system."""
    if int(digest(group)[:8],16)%2:
        return [[dict(role='system',content=system),*messages]]
    return [messages]


def focused_curriculum():
    rows=[r for r in curriculum() if r['foundation_kind']!='arithmetic']
    # These are authored task-oriented exchanges, not human anecdotes or an
    # attempted augmentation of the ARC test. Related topics share one split.
    topics=[
        ('reading','I want to read more.','Start with ten minutes a day and choose a book you enjoy.'),
        ('walk','I would like a short break.','A short walk or a few minutes away from your screen can help.'),
        ('desk','My desk is cluttered.','Clear one small area first, then put away the items you do not need.'),
        ('schedule','I need to organize my afternoon.','List your tasks, choose the most important one, and leave time for a break.'),
        ('learning','I want to learn something new.','Choose one small topic and practice it regularly.'),
        ('packing','I need to pack for a day out.','Bring water, your keys, and anything needed for your plans.'),
        ('notes','I want to remember what I read.','Write a short note in your own words after each section.'),
        ('explanation','I have a question but I am unsure how to ask it.','Tell me what you understand so far and which part is confusing.'),
        ('python','I am learning Python.','We can start with variables, lists, or a small function. Which would you like to try?'),
        ('planning','I have a project in mind.','Describe the goal and we can break it into smaller steps.'),
        ('writing','I need help with a message.','Share the message and tell me what you want it to say.'),
        ('rest','I am taking a break.','Sounds good. We can chat or pick up your task when you are ready.'),
    ]
    openers=[('Are you around?','Yes, I am here. What would you like to talk about?'),
             ('How is it going?','I am here and ready to help. How are you doing?'),
             ('Anything going on?','I am ready to chat. What is on your mind?'),
             ('Hey, got a minute?','Of course. What would you like to talk about?'),
             ('Can we chat for a little while?','Sure. What is on your mind?'),
             ('What kinds of tasks can you do?','I can answer questions, explain ideas, help with writing, and work through simple Python code.')]
    for topic,prompt,answer in topics:
        for first,reply in openers:
            rows.append(dict(component='chat',group='local-short-chat:'+topic,
                foundation_kind='social-context',messages=pair(first,reply)+pair(prompt,answer)+pair('Thanks for chatting.',"You're welcome!"),
                provenance=dict(repo='local/short-chat-bridge',revision='short-chat-bridge-v1',split='authored',license='CC0-1.0',method='Authored short task-oriented dialogue, not a Hugging Face dataset')))
    return rows


def check_focus(messages):
    """Exclude tutor/role-play/editing task drift from this conversation attempt."""
    import re
    prompts=' '.join(m['content'] for m in messages if m['role']=='user')
    answers=' '.join(m['content'] for m in messages if m['role']=='assistant')
    if re.search(r'\b(?:edit\w*|revis\w*|proofread|rephrase|sonnet|poetry|lyrics|roleplay|role.?play|pretend|impersonate)\b|\b(?:you are (?:a|an|the)|act as|playing the role|play the role)\b',prompts,re.I):raise ValueError('editing_or_roleplay_drift')
    if re.search(r'\b(?:your (?:revised )?sentence|revised text|corrected sentence|grammar lesson|english tutor)\b',answers,re.I):raise ValueError('editing_response_drift')
    if re.search(r'\b(?:internships?|interships?|job openings?|job vacancies|past 24|last 24|breaking news|last (?:one |two |three |four |five |six |seven |eight |nine |ten |[0-9]+ )?(?:years?|months?|weeks?|days?)|p.?value|p value|null hypothesis|statistically significant|retire|retirement|401k)\b',prompts+' '+answers,re.I):raise ValueError('time_sensitive_listing')
