import pandas as pd
import numpy as np
import sys

# redirect all print output to a log file
log_file = open('output/fiftyplusone_analysis_log.txt', 'w')
sys.stdout = log_file

# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure

# load cleaned harris+trump questions dataset (output from fiftyplusone_initial_analysis.py)
harris_trump_full_df = pd.read_csv("data/fiftyplusone_cleaned_harris_trump_questions.csv")

# load actual 2024 results
true_votes = pd.read_csv("data/true_votes_by_state_mengrep.csv")

# define cutoff date (Biden dropout)
dropout_cutoff = pd.Timestamp('2024-07-21')

########################################################################################
############################# General Accuracy Analysis ################################
########################################################################################

######## pivot to get one row per question with trump and harris pct side by side

harris_trump_pivot = (
    harris_trump_full_df[harris_trump_full_df['answer'].isin(['Trump', 'Harris'])]
    .pivot_table(
        index=['question_id', 'poll_id', 'state', 'start_date', 'mode'],
        columns='answer',
        values='pct',
        aggfunc='mean'
    )
    .reset_index()
    .rename(columns={'Trump': 'pct_trump_poll', 'Harris': 'pct_harris_poll'})
)

# drop questions missing either estimate
n_before_drop = harris_trump_pivot['question_id'].nunique()
harris_trump_pivot = harris_trump_pivot.dropna(subset=['pct_trump_poll', 'pct_harris_poll'])
n_after_drop = harris_trump_pivot['question_id'].nunique()

# convert start_date to datetime after pivot
harris_trump_pivot['start_date'] = pd.to_datetime(harris_trump_pivot['start_date'])

print(f"Questions with both Trump and Harris pct: {n_after_drop}")
print(f"Questions dropped due to missing pct:     {n_before_drop - n_after_drop}")

######## compute national true vote shares as weighted average of state results

national_true = pd.Series({
    'p_trump_true':  np.average(true_votes['p_trump_true'],  weights=true_votes['N_state']),
    'p_harris_true': np.average(true_votes['p_harris_true'], weights=true_votes['N_state']),
})

print(f"\nDerived national true vote shares:")
print(f"  Trump:  {national_true['p_trump_true']:.4f}")
print(f"  Harris: {national_true['p_harris_true']:.4f}")

# add national row to true_votes for merging
true_votes_with_national = pd.concat([
    true_votes[['state_name', 'p_trump_true', 'p_harris_true']],
    pd.DataFrame([{
        'state_name':    'National',
        'p_trump_true':  national_true['p_trump_true'],
        'p_harris_true': national_true['p_harris_true']
    }])
], ignore_index=True)

######## merge in actual state + national results

harris_trump_pivot = harris_trump_pivot.merge(
    true_votes_with_national,
    left_on='state',
    right_on='state_name',
    how='left'
).drop(columns='state_name')

# report any remaining unmatched states
unmatched = harris_trump_pivot[harris_trump_pivot['p_trump_true'].isna()]
print(f"\nQuestions unmatched after state + national merge: {len(unmatched)}")
if len(unmatched) > 0:
    print(f"Unmatched states:\n{unmatched['state'].value_counts().to_string()}")

# drop any remaining unmatched
harris_trump_pivot = harris_trump_pivot.dropna(subset=['p_trump_true', 'p_harris_true'])
print(f"\nQuestions remaining for accuracy analysis: {len(harris_trump_pivot)}")

######## compute Method A accuracy measure
# A = ln((poll_trump / poll_harris) / (true_trump / true_harris))
# A = 0: perfect accuracy
# A > 0: Republican bias (poll overestimates Trump relative to Harris)
# A < 0: Democratic bias (poll overestimates Harris relative to Trump)

harris_trump_pivot['A'] = np.log(
    (harris_trump_pivot['pct_trump_poll']  / harris_trump_pivot['pct_harris_poll']) /
    (harris_trump_pivot['p_trump_true'] / harris_trump_pivot['p_harris_true'])
)

######## overall accuracy

print(f"\nOverall Method A accuracy (all Harris+Trump questions):")
print(f"  Mean A:   {harris_trump_pivot['A'].mean():.4f}  (+ = Republican bias, - = Democratic bias)")
print(f"  Median A: {harris_trump_pivot['A'].median():.4f}")
print(f"  Std A:    {harris_trump_pivot['A'].std():.4f}")
print(f"  N:        {len(harris_trump_pivot)}")

######## accuracy split before/after dropout

harris_trump_pivot['period'] = np.where(
    harris_trump_pivot['start_date'] < dropout_cutoff, 'before_dropout', 'after_dropout'
)

accuracy_by_period = (
    harris_trump_pivot.groupby('period')['A']
    .agg(mean='mean', median='median', std='std', n='count')
    .reset_index()
)

print(f"\nMethod A accuracy by period:\n")
print(accuracy_by_period.to_string(index=False))

######## accuracy split by state vs national

harris_trump_pivot['poll_level'] = np.where(
    harris_trump_pivot['state'] == 'National', 'national', 'state'
)

accuracy_by_level = (
    harris_trump_pivot.groupby('poll_level')['A']
    .agg(mean='mean', median='median', std='std', n='count')
    .reset_index()
)

print(f"\nMethod A accuracy by poll level (state vs national):\n")
print(accuracy_by_level.to_string(index=False))

######## accuracy by state vs national, split before/after dropout

accuracy_by_level_period = (
    harris_trump_pivot.groupby(['poll_level', 'period'])['A']
    .agg(mean='mean', median='median', n='count')
    .reset_index()
    .sort_values(['poll_level', 'period'])
)

print(f"\nMethod A accuracy by poll level and period:\n")
print(accuracy_by_level_period.to_string(index=False))

########################################################################################
##################################### Mode Analysis ####################################
########################################################################################

######## fix typo and explode slash-separated modes into base modes

harris_trump_pivot['mode'] = harris_trump_pivot['mode'].str.replace('LIve Phone', 'Live Phone', regex=False)

# each mode with a slash will count in both listed
harris_trump_modes = harris_trump_pivot.copy()
harris_trump_modes['base_mode'] = harris_trump_modes['mode'].str.split('/')
harris_trump_modes = harris_trump_modes.explode('base_mode')
harris_trump_modes['base_mode'] = harris_trump_modes['base_mode'].str.strip()

######## mode counts

mode_counts = (
    harris_trump_modes.groupby('base_mode')['question_id']
    .nunique()
    .reset_index()
    .rename(columns={'question_id': 'unique_questions'})
    .sort_values('unique_questions', ascending=False)
    .reset_index(drop=True)
)

print(f"\nBase mode breakdown (unique questions per mode):\n")
print(mode_counts.to_string(index=False))

######## accuracy by base mode

accuracy_by_mode = (
    harris_trump_modes.groupby('base_mode')['A']
    .agg(mean='mean', median='median', std='std', n='count')
    .reset_index()
    .sort_values('mean', ascending=False)
)

print(f"\nMethod A accuracy by base mode (+ = Republican bias, - = Democratic bias):\n")
print(accuracy_by_mode.to_string(index=False))

######## accuracy by base mode split before/after dropout

accuracy_by_mode_period = (
    harris_trump_modes.groupby(['base_mode', 'period'])['A']
    .agg(mean='mean', median='median', n='count')
    .reset_index()
    .sort_values(['base_mode', 'period'])
)

print(f"\nMethod A accuracy by base mode and period:\n")
print(accuracy_by_mode_period.to_string(index=False))


########################################################################################
###################### Multivariate Regression Analysis ################################
########################################################################################

# do OLS regression mtulivariate predicting method A value 
# state and national polls are in different regressions 
# multivariate analysis to udnerstand impact of different decisions on accuracy
# use method A as dependent variable 
# report the regression coefficient, standard error, signifacnce stars for each factor 
# adjusted r square at bottom, constant at bottom, N at bottom
# Controls for survey methods, forecasting, election dynamics
# factors relating to survey methods
# factors related to election forecasts
# Include duration the poll was in the field, which is an implicit measure of nonresponse 
# include days before election (from last day of poll? first?)
# controls for electoral context: total statewide turnout and final margin of victory
# date of poll based on final day conducted 
# restricting to competitive states versus all states
# who is the sampling frame (ex: lv)
# percent of don't know (100 - total for all candidates in a question)
# absolute margin of victory for winner (state and national)
# split into time periods ( analysis using three different time frame, 90, 30, and 7 days before the elections)
# unit of analysis is a single survey
# variables to include: days in field, days before election, percent don't know, turnout, final margin, and more)
# To control for correlated errors due to the clustering of the surveys within individual statewide studies, robust standard errors are calculated in the manner suggested by Huber (1967) and White (1980) for all but the overall national population surveys
# mode of poll indiactors 

# save question-level accuracy dataset for further analysis
harris_trump_pivot.to_csv('data/harris_trump_accuracy.csv', index=False)

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("Accuracy analysis complete — see output/fiftyplusone_analysis_log.txt")