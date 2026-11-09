# Recorded inference examples

Greedy FP32, no system message, no repetition penalty. Reproduced in the clean release environment. These include a factual error so that successful loading is not confused with correctness.

## knowledge-00

**User:** What is the capital of Japan? Give a short answer.

**Generated answer:** Tokyo

Stop: `turn_end`. Saved answer matcher: `True`.

## knowledge-01

**User:** Which planet is closest to the Sun? Give a short answer.

**Generated answer:** venus

Stop: `turn_end`. Saved answer matcher: `False`.

## reading-00

**User:** Use only the information below. Give a short answer.

Anika brought a teal notebook. Greta brought a black notebook.
What color was Anika’s notebook?

**Generated answer:** teal

Stop: `turn_end`. Saved answer matcher: `True`.

## conversation-00

**User:** My reservation number is 217.

**Assistant:** Understood.

**User:** Correction: my reservation number is 317.

**Assistant:** Understood.

**User:** What is my current reservation number? Give only the number.

**Generated answer:** 317

Stop: `turn_end`. Saved answer matcher: `True`.

