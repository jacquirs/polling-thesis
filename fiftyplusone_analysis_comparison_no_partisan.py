import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt


# redirect all print output to a log file
log_file = open('output/fiftyplusone_analysis_comparison_no_partisan_log.txt', 'w')
sys.stdout = log_file

# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS FOR NON PARTISAN POLLS ONLY
################### NON PARTISAN ONLY #######################################
# IT IS NOT NOT NOT LIMITED TO THE TIME AFTER BIDEN DROPPED OUT
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy

# load cleaned harris+trump questions dataset (output from fiftyplusone_initial_analysis.py) FOR NON PARTISAN
harris_trump_full_df = pd.read_csv("data/fiftyplusone_cleaned_harris_trump_questions_no_partisan.csv")

# load actual 2024 results
true_votes = pd.read_csv("data/true_votes_by_state_mengrep.csv")

# election day 2024
election_date  = pd.Timestamp('2024-11-05')   


########################################################################################
############################# New fields ################################
########################################################################################
# pct_dk captures the share of respondents not supporting any named candidate
# it equals 100 minus the sum of all named candidates' percentages per question
# this picks up undecided voters, third-party supporters, and refusals combined
# compute this before pivoting because the pivot only keeps trump and harris rows with pct values (maybe need to cahnge later)

pct_total_by_question = (
    harris_trump_full_df
    .groupby('question_id')['pct']
    .sum()
    .reset_index()
    .rename(columns={'pct': 'pct_total'})
)
pct_total_by_question['pct_dk'] = 100 - pct_total_by_question['pct_total']

########################################################################################
##################################### Pivoting #########################################
########################################################################################

######## pivot to get one row per question with trump and harris pct side by side
# only pivot on question_id because of metadata handling dropping 200 rows if not
harris_trump_pivot = (
    harris_trump_full_df[harris_trump_full_df['answer'].isin(['Trump', 'Harris'])]
    .pivot_table(
        index='question_id',
        columns='answer',
        values='pct',
        aggfunc='mean'
    )
    .reset_index()
    .rename(columns={'Trump': 'pct_trump_poll', 'Harris': 'pct_harris_poll'})
)

# merge metadata from one of the rows
metadata_to_merge = (
    harris_trump_full_df[harris_trump_full_df['answer'] == 'Trump']
    [['question_id', 'poll_id', 'pollster', 'state', 'start_date', 'end_date', 
      'mode', 'population', 'sample_size']]
    .drop_duplicates('question_id')
)

harris_trump_pivot = harris_trump_pivot.merge(metadata_to_merge, on='question_id', how='left')

# pivot_table leaves a residual answer name on the columns axis
harris_trump_pivot.columns.name = None

# re-cast start_date and end_date to datetime after pivot
harris_trump_pivot['end_date'] = pd.to_datetime(harris_trump_pivot['end_date'])
harris_trump_pivot['start_date'] = pd.to_datetime(harris_trump_pivot['start_date'])

# drop questions missing either estimate
n_before_drop = harris_trump_pivot['question_id'].nunique()
harris_trump_pivot = harris_trump_pivot.dropna(subset=['pct_trump_poll', 'pct_harris_poll'])
n_after_drop = harris_trump_pivot['question_id'].nunique()

print(f"Questions with both Trump and Harris pct: {n_after_drop}")
print(f"Questions dropped due to missing pct:     {n_before_drop - n_after_drop}")

# merge in pct_dk
harris_trump_pivot = harris_trump_pivot.merge(
    pct_total_by_question[['question_id', 'pct_dk']],
    on='question_id',
    how='left'
)

######## compute national true vote shares as weighted average of state results
national_true = pd.Series({
    'p_trump_true':  np.average(true_votes['p_trump_true'],  weights=true_votes['N_state']),
    'p_harris_true': np.average(true_votes['p_harris_true'], weights=true_votes['N_state']),
})

print(f"\nDerived national true vote shares (confirmed against results yay!!):")
print(f"  Trump:  {national_true['p_trump_true']:.4f}")
print(f"  Harris: {national_true['p_harris_true']:.4f}")

# compute absolute margin of victory for each state (used in regression)
true_votes['abs_margin'] = (true_votes['p_trump_true'] - true_votes['p_harris_true']).abs()

# compute national absolute margin from the weighted averages
national_abs_margin = abs(national_true['p_trump_true'] - national_true['p_harris_true'])

# add a national row so we can merge both state and national polls in one pass
true_votes_with_national = pd.concat([
    true_votes[['state_name', 'p_trump_true', 'p_harris_true', 'abs_margin']],
    pd.DataFrame([{
        'state_name':    'national',
        'p_trump_true':  national_true['p_trump_true'],
        'p_harris_true': national_true['p_harris_true'],
        'abs_margin':    national_abs_margin
    }])
], ignore_index=True)

true_votes_with_national['state_name'] = true_votes_with_national['state_name'].str.strip().str.lower()
harris_trump_pivot['state']            = harris_trump_pivot['state'].str.strip().str.lower()

######## merge in actual state + national results (on state name becuase national now a state name)
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
print(f"Total number of polls: {harris_trump_pivot['poll_id'].nunique()}")

######## compute Method A accuracy measure
# A = ln((poll_trump / poll_harris) / (true_trump / true_harris))
# A = 0: perfect accuracy
# A > 0: Republican bias (poll overestimates Trump relative to Harris)
# A < 0: Democratic bias (poll overestimates Harris relative to Trump)
# the log-odds ratio form is symmetric and scale-invariant, making it more appropriate than simple margin error for comparing polls across states with different swing landscapes

harris_trump_pivot['A'] = np.log(
    ((harris_trump_pivot['pct_trump_poll'])  / (harris_trump_pivot['pct_harris_poll'])) /
    (harris_trump_pivot['p_trump_true'] / harris_trump_pivot['p_harris_true'])
)

# decompose into trump and harris parts
harris_trump_pivot['trump_part_A'] = np.log(harris_trump_pivot['pct_trump_poll']/100) - np.log(harris_trump_pivot['p_trump_true'])
harris_trump_pivot['harris_part_A'] = np.log(harris_trump_pivot['pct_harris_poll']/100) - np.log(harris_trump_pivot['p_harris_true'])

# flag poll level (state vs national) used to split regressions
harris_trump_pivot['poll_level'] = np.where(
    harris_trump_pivot['state'] == 'national', 'national', 'state'
)

# diagnostic: confirm how many questions were flagged as national
print(f"\npoll_level counts after flagging:")
print(harris_trump_pivot['poll_level'].value_counts(dropna=False).to_string())
print(f"\nstate values flagged as national:")
print(harris_trump_pivot[harris_trump_pivot['poll_level'] == 'national']['state'].value_counts(dropna=False).to_string())

########################################################################################
############################# General Accuracy Analysis ################################
########################################################################################
# include SE
def compute_clustered_se(df, value_col, cluster_col):
    """
    compute cluster-robust standard error of the mean, accounts for multiple questions per poll
    """
    # remove any rows with missing values
    df_clean = df[[value_col, cluster_col]].dropna()
    
    if len(df_clean) == 0:
        return np.nan, 0
    
    # get cluster means
    cluster_means = df_clean.groupby(cluster_col)[value_col].mean()
    n_clusters = len(cluster_means)
    
    # need at least 2 clusters to compute cluster-robust SE
    if n_clusters < 2:
        return np.nan, n_clusters
    
    # grand mean
    grand_mean = df_clean[value_col].mean()
    
    # cluster-robust variance of the mean
    cluster_var = ((cluster_means - grand_mean) ** 2).sum() / (n_clusters - 1)
    
    # standard error of grand mean
    se_robust = np.sqrt(cluster_var / n_clusters)
    
    return se_robust, n_clusters

def sig_stars(p):
    if p < 0.01:   return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    else:          return ''


######## overall accuracy
mean_A = harris_trump_pivot['A'].mean()
se_A_robust, n_polls = compute_clustered_se(harris_trump_pivot, 'A', 'poll_id')
t_stat = mean_A / se_A_robust
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls-1))

mean_trump_part_A = harris_trump_pivot['trump_part_A'].mean()
mean_harris_part_A = harris_trump_pivot['harris_part_A'].mean()

print(f"\nOverall Method A accuracy (all Harris+Trump questions) NON PARTISAN:")
print(f"  Mean:   {mean_A:.4f}")
print(f"  SE:       {se_A_robust:.4f}")
print(f"  p-value:  {p_value:.4f} {sig_stars(p_value)}")
print(f"  Median: {harris_trump_pivot['A'].median():.4f}")
print(f"  SD:    {harris_trump_pivot['A'].std():.4f}")
print(f"  N:        {len(harris_trump_pivot)}")
print(f"  Mean Harris Part:   {mean_harris_part_A:.4f}")
print(f"  Mean Trump Part:   {mean_trump_part_A:.4f}")

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__