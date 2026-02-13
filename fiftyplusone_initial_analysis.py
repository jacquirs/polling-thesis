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

########################################################################################
##################################### Unique Answer Sets ###############################
########################################################################################

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

######## find unique sets of answers and the number of times they occur
# split based on start date before or after dropout

# Group answers by (question_id, poll_id) and collect the set of answer names + min start_date
answer_sets = (
    df.groupby(['question_id', 'poll_id'])
    .agg(answer=('answer', lambda x: frozenset(x.dropna())),
         start_date=('start_date', 'min'))
    .reset_index()
)

# convert start_date to datetime
answer_sets['start_date'] = pd.to_datetime(answer_sets['start_date'])

# check everything converted correctly
print("NaT count:", answer_sets['start_date'].isna().sum())

# define cutoff date
cutoff = pd.Timestamp('2024-07-21')

# count occurrences split by date
before = answer_sets[answer_sets['start_date'] < cutoff].groupby('answer')['answer'].count()
after  = answer_sets[answer_sets['start_date'] >= cutoff].groupby('answer')['answer'].count()

# combine into a summary df
summary = (
    pd.DataFrame({'before_7_21': before, 'after_7_21': after})
    .fillna(0)
    .astype(int)
    .assign(total=lambda x: x['before_7_21'] + x['after_7_21'])
    .sort_values('total', ascending=False)
    .reset_index()
)
summary.columns = ['answer_set', 'before_7_21', 'after_7_21', 'total']

print(f"Found {len(summary)} unique answer set(s):\n")
for _, row in summary.iterrows():
    print(f"Before: {row['before_7_21']} | After: {row['after_7_21']} | Total: {row['total']} | {sorted(row['answer_set'])}")