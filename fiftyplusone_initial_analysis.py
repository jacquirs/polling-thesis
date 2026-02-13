import pandas as pd
import numpy as np

# THIS FILE ANALYZES 50+1 data provided generously for 2024 polling

# load CCES 2024 data from site
df = pd.read_csv("data/president_2024_general.csv")

# normalize answer to reflect candidate_name distinctions where trump jr and trump aren't the same person
df.loc[df['candidate_name'].str.lower().str.contains('trump jr', na=False), 'answer'] = 'Trump Jr.'

######## Jacqui's Notes on How to Know What Is a Useful Question 

# each poll has its own poll_id
# each row is one answer in one question in one poll
# each question has a question_id
# the answers to a question can be accessed through a shared question_id x poll_id
# the answers are in answer

########################################################################################
##################################### Unique Answer Sets ###############################
########################################################################################

# group answers by (question_id, poll_id), collect answer sets and min start_date
question_answer_sets = (
    df.groupby(['question_id', 'poll_id'])
    .agg(
        answer_set=('answer', lambda x: frozenset(x.dropna())),
        start_date=('start_date', 'min')
    )
    .reset_index()
)

# convert start_date to datetime and verify
question_answer_sets['start_date'] = pd.to_datetime(question_answer_sets['start_date'])
print("NaT count:", question_answer_sets['start_date'].isna().sum())

# define cutoff date (Biden dropout)
dropout_cutoff = pd.Timestamp('2024-07-21')

######## count of each unique answer set, split before/after dropout

before_dropout = question_answer_sets[question_answer_sets['start_date'] <  dropout_cutoff]
after_dropout  = question_answer_sets[question_answer_sets['start_date'] >= dropout_cutoff]

answer_set_counts = (
    pd.DataFrame({
        'before_dropout': before_dropout.groupby('answer_set')['answer_set'].count(),
        'after_dropout':  after_dropout.groupby('answer_set')['answer_set'].count(),
    })
    .fillna(0)
    .astype(int)
    .assign(total=lambda x: x['before_dropout'] + x['after_dropout'])
    .sort_values('total', ascending=False)
    .reset_index()
)

print(f"\nFound {len(answer_set_counts)} unique answer set(s):\n")
for _, row in answer_set_counts.iterrows():
    print(f"Before: {row['before_dropout']} | After: {row['after_dropout']} | Total: {row['total']} | {sorted(row['answer_set'])}")

######## count of unique answer sets by trump/harris inclusion, split before/after dropout

def classify_trump_harris(answer_set):
    names = {a.lower() for a in answer_set}
    has_trump  = any('trump' in a and 'jr' not in a for a in names)
    has_harris = any('harris' in a for a in names)
    if has_trump and has_harris:
        return 'both Trump and Harris'
    elif has_trump:
        return 'Trump only (no Harris)'
    elif has_harris:
        return 'Harris only (no Trump)'
    else:
        return 'neither Trump nor Harris'

question_answer_sets['trump_harris_classification'] = question_answer_sets['answer_set'].apply(classify_trump_harris)

trump_harris_counts = (
    question_answer_sets.groupby('trump_harris_classification')
    .apply(lambda x: pd.Series({
        'before_dropout': (x['start_date'] <  dropout_cutoff).sum(),
        'after_dropout':  (x['start_date'] >= dropout_cutoff).sum(),
        'total':          len(x)
    }))
    .sort_values('total', ascending=False)
    .reset_index()
)

print(f"\nTrump/Harris classification:\n")
print(trump_harris_counts.to_string(index=False))

# results were
# trump_harris_classification  before_dropout  after_dropout  total
#     Trump only (no Harris)            3412             16   3428
#      both Trump and Harris             277           2337   2614
#   neither Trump nor Harris             712              0    712
#     Harris only (no Trump)              48              4     52
# should at least drop all of the polls with neither from the final dataset
# likely need to also drop the single candidate polls, at least for Harris, but can we impute the biden levels pre dropout for harris? perhaps as overall democrat
# these are all general election so having harris and biden would likely be a problem, but maybe we use biden x trump pre groupout?

######## count of unique answer sets by biden/harris/trump inclusion, split before/after dropout

def classify_biden_harris_trump(answer_set):
    names = {a.lower() for a in answer_set}
    has_biden  = any('biden'  in a for a in names)
    has_harris = any('harris' in a for a in names)
    has_trump  = any('trump'  in a and 'jr' not in a for a in names)

    present = []
    if has_biden:  present.append('Biden')
    if has_harris: present.append('Harris')
    if has_trump:  present.append('Trump')

    return ' + '.join(present) if present else 'none of Biden/Harris/Trump'

# all 8 possible combinations explicitly defined
all_classifications = [
    'Biden + Harris + Trump',
    'Biden + Harris',
    'Biden + Trump',
    'Harris + Trump',
    'Biden',
    'Harris',
    'Trump',
    'none of Biden/Harris/Trump'
]

question_answer_sets['biden_harris_trump_classification'] = question_answer_sets['answer_set'].apply(classify_biden_harris_trump)

biden_harris_trump_counts = (
    question_answer_sets.groupby('biden_harris_trump_classification')
    .apply(lambda x: pd.Series({
        'before_dropout': (x['start_date'] <  dropout_cutoff).sum(),
        'after_dropout':  (x['start_date'] >= dropout_cutoff).sum(),
        'total':          len(x)
    }))
    .reindex(all_classifications)   # ensures all 8 rows always appear
    .fillna(0)
    .astype(int)
    .reset_index()
)

print(f"\nBiden/Harris/Trump classification:\n")
print(biden_harris_trump_counts.to_string(index=False))