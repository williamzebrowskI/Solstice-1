"""Authored English chatbot diagnostic v1, frozen before seeing model outputs.

30 cases per objective category, plus 12 qualitative conversations. Variants
within a family are correlated; this is not a representative population sample.
"""


def build():
    rows = []

    def add(category, family, prompt, answers, *, messages=None, case_sensitive=False,
            answerable=True, qualitative=False):
        number = sum(r['category'] == category for r in rows)
        rows.append(dict(id=f'{category}-{number:02d}', category=category, family=family,
                         messages=messages or [dict(role='user', content=prompt)],
                         answers=answers, case_sensitive=case_sensitive,
                         answerable=answerable, qualitative=qualitative,
                         max_tokens=160 if qualitative else 128))

    facts = [
        ('What is the capital of Japan?', ['Tokyo']),
        ('Which planet is closest to the Sun?', ['Mercury']),
        ('What is the chemical symbol for gold?', ['Au']),
        ('How many sides does a hexagon have?', ['six', '6']),
        ('Which ocean is the largest on Earth?', ['Pacific', 'Pacific Ocean', 'the Pacific Ocean']),
        ('Who wrote Pride and Prejudice?', ['Jane Austen']),
        ('On which continent is Egypt?', ['Africa']),
        ('What force pulls objects toward Earth?', ['gravity', 'gravitational force']),
        ('What gas do humans need to breathe to survive?', ['oxygen']),
        ('What is the process by which plants use sunlight to make food called?', ['photosynthesis']),
        ('How many minutes are in one hour?', ['60', 'sixty']),
        ('At standard atmospheric pressure, at what Celsius temperature does pure water freeze?', ['0', 'zero', '0 degrees Celsius', '0°C', 'zero degrees Celsius']),
        ('Which instrument is used to measure temperature?', ['thermometer', 'a thermometer']),
        ('Which organ pumps blood around the human body?', ['heart', 'the heart']),
        ('What is the largest living land animal?', ['African elephant', 'African bush elephant', 'African savanna elephant', 'elephant', 'the African elephant']),
        ('What is the opposite of north?', ['south']),
        ('What is the name of the natural satellite that orbits Earth?', ['Moon', 'the Moon']),
        ('Who painted the Mona Lisa?', ['Leonardo da Vinci', 'da Vinci']),
        ('In which country is the city of Barcelona?', ['Spain']),
        ('How many days does a leap year have?', ['366', 'three hundred sixty-six', 'three hundred and sixty-six']),
        ('What is the freezing of a liquid into a solid called?', ['freezing', 'solidification']),
        ('What is the square root of 81?', ['9', 'nine']),
        ('Which language is primarily spoken in Brazil?', ['Portuguese', 'Brazilian Portuguese']),
        ('Which animal is known for building dams?', ['beaver', 'beavers', 'a beaver']),
        ('What part of a plant typically absorbs water from the soil?', ['roots', 'the roots', 'root']),
        ('What is the main gas in Earth’s atmosphere?', ['nitrogen']),
        ('How many degrees are in a right angle?', ['90', 'ninety', '90 degrees']),
        ('Which metal is liquid at ordinary room temperature, around 20 degrees Celsius?', ['mercury']),
        ('What is the name of the galaxy that contains our solar system?', ['Milky Way', 'the Milky Way']),
        ('Which sense uses the ears?', ['hearing']),
    ]
    for q, answers in facts:
        add('knowledge', 'stable_fact', q + ' Give a short answer.', answers)

    names = ['Anika', 'Bram', 'Celia', 'Dante', 'Esme', 'Farid']
    other = ['Greta', 'Hector', 'Ines', 'Jules', 'Kira', 'Lucio']
    colors = ['teal', 'maroon', 'ivory', 'indigo', 'ochre', 'turquoise']
    items = ['notebook', 'umbrella', 'camera', 'scarf', 'compass', 'lantern']
    cities = ['Kyoto', 'Porto', 'Bergen', 'Ghent', 'Bari', 'Lyon']
    days = ['Tuesday', 'Friday', 'Sunday', 'Monday', 'Thursday', 'Wednesday']
    foods = ['rice', 'pasta', 'lentils', 'potatoes', 'noodles', 'beans']
    for i, name in enumerate(names):
        lead = 'Use only the information below. Give a short answer.\n\n'
        add('reading', 'contrast', lead + f'{name} brought a {colors[i]} {items[i]}. {other[i]} brought a black {items[i]}.\nWhat color was {name}’s {items[i]}?', [colors[i]])
        add('reading', 'schedule_exception', lead + f'The club normally meets on Saturday. This week its meeting has been moved to {days[i]}.\nOn which day is this week’s meeting?', [days[i]])
        add('reading', 'relation', lead + f'{name} visited {cities[i]}. {other[i]} stayed in Naples.\nWhich city did {name} visit?', [cities[i]])
        add('reading', 'negation', lead + f'For dinner, {name} chose {foods[i]}, not bread.\nWhat did {name} choose for dinner?', [foods[i]])
        add('reading', 'quantity', lead + f'The green shelf has {11+i} books. The orange shelf has {23+i} books.\nHow many books are on the orange shelf?', [str(23+i)])

        a, b = 7+i, 3+i
        add('reasoning', 'addition', f'{name} has {a} postcards and receives {b} more. How many postcards does {name} have now? Give only the number.', [str(a+b)])
        add('reasoning', 'subtraction', f'A tray contains {20+2*i} cups. Someone removes {b} cups. How many remain? Give only the number.', [str(20+2*i-b)])
        add('reasoning', 'multiplication', f'There are {b} packets, each containing 4 stickers. How many stickers are there altogether? Give only the number.', [str(b*4)])
        add('reasoning', 'ordering', f'{name} is taller than {other[i]}. {other[i]} is taller than Zola. Who is tallest? Give only the name.', [name])
        add('reasoning', 'state_change', f'{name} puts a coin in a drawer, then moves it into a cup. The drawer is closed. Where is the coin now? Give only the container.', ['cup', 'the cup'])

        add('instructions', 'case', f'Convert this word to uppercase and return only the result: {items[i]}', [items[i].upper()], case_sensitive=True)
        words = [colors[i], items[i], cities[i].lower()]
        add('instructions', 'reverse_list', 'Reverse the order of these three items. Return the items separated by commas, with no extra words: ' + ', '.join(words), [', '.join(reversed(words))])
        values = [10+i*2, 3+i*2, 8+i*2]
        add('instructions', 'sort_numbers', 'Put these numbers in ascending order. Return only the comma-separated numbers: ' + ', '.join(map(str, values)), [', '.join(map(str, sorted(values)))])
        add('instructions', 'replacement', f'Replace "small" with "large" in this sentence and return only the revised sentence: The small {items[i]} is here.', [f'The large {items[i]} is here.'], case_sensitive=True)
        add('instructions', 'copy', f'Repeat this label exactly, without a prefix or explanation: QZ-{417+i}-R', [f'QZ-{417+i}-R'], case_sensitive=True)

        def dialogue(first, last, middle=None):
            turns = [dict(role='user', content=first), dict(role='assistant', content='Understood.')]
            if middle:
                turns += [dict(role='user', content=middle), dict(role='assistant', content='Understood.')]
            return turns + [dict(role='user', content=last)]
        add('conversation', 'correction', '', [str(317+i)], messages=dialogue(f'My reservation number is {217+i}.', 'What is my current reservation number? Give only the number.', f'Correction: my reservation number is {317+i}.'))
        add('conversation', 'preference', '', [foods[i]], messages=dialogue(f'My favorite dinner is {foods[i]}.', 'What did I say my favorite dinner was? Give only the food.'))
        add('conversation', 'entity_tracking', '', [colors[i]], messages=dialogue(f'My name is {name}. My bag is {colors[i]}. {other[i]} has a black bag.', 'What color is my bag? Give only the color.'))
        add('conversation', 'distraction', '', [cities[i]], messages=dialogue(f'My next trip is to {cities[i]}.', 'Which city did I say I am visiting? Give only the city.', 'I am packing a coat and two books.'))
        add('conversation', 'updated_plan', '', [days[i]], messages=dialogue('I planned to visit the museum on Saturday.', 'On which day do I now plan to visit the museum? Give only the day.', f'I changed my plans. I will visit the museum on {days[i]} instead.'))

        available = i % 2 == 0
        suffix = ' Use only these facts. If the answer is missing, reply "Not provided." Otherwise give a short answer.'
        info = [
            ('missing_time', f'{name} booked a train ticket.' + (f' Departure is at {8+i}:30.' if available else ''), 'What time does the train leave?', [f'{8+i}:30']),
            ('missing_quantity', f'{name} bought apples.' + (f' The bag contains {8+i} apples.' if available else ''), 'How many apples did they buy?', [str(8+i)]),
            ('missing_identity', f'A {items[i]} was found in the park.' + (f' It belongs to {name}.' if available else ''), 'Who owns it?', [name]),
            ('missing_location', f'{name} mailed a parcel.' + (f' The destination is {cities[i]}.' if available else ''), 'To which city was the parcel sent?', [cities[i]]),
            ('missing_attribute', f'{name} has a bicycle.' + (f' The bicycle is {colors[i]}.' if available else ''), 'What color is the bicycle?', [colors[i]]),
        ]
        for family, passage, question, answer in info:
            add('answerability', family, passage + '\n' + question + suffix,
                answer if available else ['Not provided.', 'Not stated.', 'Not specified.', 'Unknown.', 'I don’t know.', 'I do not know.', 'Not enough information.', 'The information is not provided.'],
                answerable=available)

    qualitative = [
        ('greeting', 'Hi! How are you?', 'A brief friendly greeting; no invented personal history.'),
        ('thanks', 'Thanks, that helped.', 'A brief relevant acknowledgement.'),
        ('explanation', 'Why does an ice cube melt in a warm room? Explain simply in one or two sentences.', 'Heat from the surroundings makes solid ice turn into liquid water.'),
        ('explanation', 'Why is a bicycle easier to pedal in a lower gear when going uphill?', 'Lower gearing reduces the pedal force needed at the cost of more pedal turns or lower speed.'),
        ('practical', 'Give me two simple ways to keep my desk organized.', 'Two distinct useful organization suggestions, without irrelevant filler.'),
        ('practical', 'I keep forgetting to bring my reusable bag to the store. Suggest one practical habit.', 'One relevant concrete reminder or preparation habit.'),
        ('writing', 'Make this request more polite without changing its meaning: Send me the notes today.', 'A polite request for the notes today; preserve the deadline.'),
        ('comparison', 'What is the difference between borrowing and buying a book?', 'Borrowing is temporary access with return; buying confers ownership.'),
        ('explanation', 'What does the word "fragile" mean? Give a simple example.', 'Easily broken or damaged, with a suitable example.'),
        ('planning', 'I have ten minutes to tidy my room. What should I do first?', 'A feasible brief starting action, not an elaborate or unrelated plan.'),
        ('uncertainty', 'What color is the mug on my desk right now?', 'Acknowledge lack of visual/contextual information; do not invent a color.'),
        ('followup', None, 'Use the previously stated no-spicy-food preference and rice/vegetables ingredients.'),
    ]
    for family, prompt, rubric in qualitative:
        messages = None
        if prompt is None:
            messages = [dict(role='user',content='I do not like spicy food. I have rice and vegetables.'),
                        dict(role='assistant',content='Understood.'),
                        dict(role='user',content='Suggest a simple dinner using those ingredients.')]
        add('open_chat', family, prompt, [], messages=messages, qualitative=True)
        rows[-1]['review_rubric'] = rubric

    assert len(rows) == 192
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({str(r['messages']) for r in rows}) == len(rows)
    return rows
