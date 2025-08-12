BASE = """
For the following SENTENCE
1) Generate a classification for the SENTENCE into one or more Sustainable Development Goals as a list.
2) Classify as True or False based on if the SENTENCE talks about Artificial Intelligence or related terms and append to the list.
3) Classify the sentiment of the text as Positive or Negative and append to the list.

Provide the answer strictly in the following format as a single List:
[SDGA, SDGB, ..., True/False, Positive/Negative]

If SENTENCE doesn't fall into any SDG, just return 0 instead of a goal

example1 - [1, 5, True, Negative]
example2 - [11, 12, False, Negative]
example3 - [0, False, Positive]

Here is the SENTENCE - {sentence}
"""