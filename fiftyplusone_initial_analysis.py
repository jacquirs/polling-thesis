import pandas as pd
import numpy as np

# THIS FILE ANALYZES 50+1 data provided generously for 2024 polling

# load CCES 2024 data from site
df = pd.read_csv("data/president_2024_general.csv")

######## Jacqui's Notes on How to Know What Is a Useful Question 

# each poll has its own poll_id
# each row is one answer in one question in one poll
# each question has a question_id
# the answers to a question can be accessed through a shared question_id x poll_id
# the answers are in answer


######## find unique sets of answers and the number of times they occur

# group answers by (question_id, poll_id) and collect the set of answer names
answer_sets = (
    df.groupby(['question_id', 'poll_id'])['answer']
    .apply(lambda x: frozenset(x.dropna()))
)

# count occurrences of each unique answer-set combination
answer_set_counts = answer_sets.value_counts().reset_index()
answer_set_counts.columns = ['answer_set', 'count']

print(f"Found {len(answer_set_counts)} unique answer set(s):\n")
for _, row in answer_set_counts.iterrows():
    print(f"Count: {row['count']} | {sorted(row['answer_set'])}")