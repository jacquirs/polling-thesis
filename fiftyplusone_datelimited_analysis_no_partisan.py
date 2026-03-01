import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt


# redirect all print output to a log file
log_file = open('output/fiftyplusone_analysis_datelimited_no_partisan_log.txt', 'w')
sys.stdout = log_file

# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS FOR NON PARTISAN POLLS ONLY
################### NON PARTISAN ONLY #######################################
################### THIS FILE IS FOR COMPARISON #############################
# IT IS LIMITED TO THE TIME AFTER BIDEN DROPPED OUT
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy

# load cleaned harris+trump questions dataset (output from fiftyplusone_initial_analysis.py) FOR NON PARTISAN
harris_trump_full_df = pd.read_csv("data/fiftyplusone_cleaned_harris_trump_questions_no_partisan.csv")

# load actual 2024 results
true_votes = pd.read_csv("data/true_votes_by_state_mengrep.csv")

# define cutoff date (Biden dropout)
dropout_cutoff = pd.Timestamp('2024-07-21')

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

# convert end_date to datetime after pivot
harris_trump_pivot['end_date'] = pd.to_datetime(harris_trump_pivot['end_date'])

print(f"Questions with both Trump and Harris pct: {n_after_drop}")
print(f"Questions dropped due to missing pct:     {n_before_drop - n_after_drop}")

# merge in pct_dk
harris_trump_pivot = harris_trump_pivot.merge(
    pct_total_by_question[['question_id', 'pct_dk']],
    on='question_id',
    how='left'
)


##### LIMIT THE DATES TO PERIOD WHEN HARRIS WAS NOMINEE
harris_trump_pivot = harris_trump_pivot[harris_trump_pivot['end_date'] >=  dropout_cutoff] #maybe reeval this

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


### graph of trump and harris parts, all polls
fig, axes = plt.subplots(2, 3, figsize=(18, 10))


# histograms of components vs method A
axes[0, 0].hist(harris_trump_pivot['trump_part_A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(harris_trump_pivot['trump_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {harris_trump_pivot["trump_part_A"].mean():.4f}')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Trump Component')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trump Component, Non Partisan: ln(poll_trump) - ln(true_trump)')
axes[0, 0].legend()


axes[0, 1].hist(harris_trump_pivot['harris_part_A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(harris_trump_pivot['harris_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {harris_trump_pivot["harris_part_A"].mean():.4f}')
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Harris Component')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Harris Component, Non Partisan: ln(poll_harris) - ln(true_harris)')
axes[0, 1].legend()


axes[0, 2].hist(harris_trump_pivot['A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(harris_trump_pivot['A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {harris_trump_pivot["A"].mean():.4f}')
axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 2].set_xlabel('Method A')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Method A, Non Partisan: trump_part - harris_part')
axes[0, 2].legend()


# poll vs true scatter plots
axes[1, 0].scatter(harris_trump_pivot['p_trump_true']*100, harris_trump_pivot['pct_trump_poll'], alpha=0.3, s=10)
axes[1, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect accuracy')
axes[1, 0].set_xlabel('True Trump %')
axes[1, 0].set_ylabel('Poll Trump %')
axes[1, 0].set_title('Trump, Non Partisan: Poll vs True')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].scatter(harris_trump_pivot['p_harris_true']*100, harris_trump_pivot['pct_harris_poll'], alpha=0.3, s=10)
axes[1, 1].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect accuracy')
axes[1, 1].set_xlabel('True Harris %')
axes[1, 1].set_ylabel('Poll Harris %')
axes[1, 1].set_title('Harris, Non Partisan: Poll vs True')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)


# simple errors
axes[1, 2].hist(harris_trump_pivot['pct_trump_poll'] - harris_trump_pivot['p_trump_true']*100,
                bins=50, alpha=0.5, label='Trump Error', edgecolor='black')
axes[1, 2].hist(harris_trump_pivot['pct_harris_poll'] - harris_trump_pivot['p_harris_true']*100,
                bins=50, alpha=0.5, label='Harris Error', edgecolor='black')
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_xlabel('Simple Error (Poll - True)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Simple Errors, Non Partisan: Poll % - True %')
axes[1, 2].legend()


plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_overall_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### graphs of parts for NATIONAL polls only
national_polls = harris_trump_pivot[harris_trump_pivot['state'] == 'national'].copy()
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

# histograms of components vs method A
axes[0, 0].hist(national_polls['trump_part_A'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(national_polls['trump_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {national_polls["trump_part_A"].mean():.4f}')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Trump Component')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trump Component: ln(poll_trump) - ln(true_trump)')
axes[0, 0].legend()


axes[0, 1].hist(national_polls['harris_part_A'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(national_polls['harris_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {national_polls["harris_part_A"].mean():.4f}')
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Harris Component')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Harris Component: ln(poll_harris) - ln(true_harris)')
axes[0, 1].legend()

axes[0, 2].hist(national_polls['A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(national_polls['A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {national_polls["A"].mean():.4f}')
axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 2].set_xlabel('Method A')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Method A: trump_part - harris_part')
axes[0, 2].legend()

# Rpoll value histograms with true value line
true_trump_national = national_polls['p_trump_true'].iloc[0] * 100
true_harris_national = national_polls['p_harris_true'].iloc[0] * 100

axes[1, 0].hist(national_polls['pct_trump_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(true_trump_national, color='red', linestyle='--', linewidth=2, label=f'True: {true_trump_national:.2f}%')
axes[1, 0].axvline(national_polls['pct_trump_poll'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Poll Mean: {national_polls["pct_trump_poll"].mean():.2f}%')
axes[1, 0].set_xlabel('Trump Poll %')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Trump Poll Values (National)')
axes[1, 0].legend()

axes[1, 1].hist(national_polls['pct_harris_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(true_harris_national, color='red', linestyle='--', linewidth=2, label=f'True: {true_harris_national:.2f}%')
axes[1, 1].axvline(national_polls['pct_harris_poll'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Poll Mean: {national_polls["pct_harris_poll"].mean():.2f}%')
axes[1, 1].set_xlabel('Harris Poll %')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Harris Poll Values (National)')
axes[1, 1].legend()

axes[1, 2].hist(national_polls['pct_trump_poll'] - national_polls['p_trump_true']*100,
                bins=50, alpha=0.5, label='Trump Error', edgecolor='black')
axes[1, 2].hist(national_polls['pct_harris_poll'] - national_polls['p_harris_true']*100,
                bins=50, alpha=0.5, label='Harris Error', edgecolor='black')
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_xlabel('Simple Error (Poll - True)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Simple Errors (National): Poll % - True %')
axes[1, 2].legend()

plt.suptitle('National Polls Only, Non Partisan', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_national_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


print(f"\nNational Polls, Non Partisan (N={len(national_polls)}):")
print(f"Trump - True:, Non Partisan {true_trump_national:.2f}%, Poll Mean: {national_polls['pct_trump_poll'].mean():.2f}%, Error: {national_polls['pct_trump_poll'].mean() - true_trump_national:.2f}")
print(f"Harris - True, Non Partisan: {true_harris_national:.2f}%, Poll Mean: {national_polls['pct_harris_poll'].mean():.2f}%, Error: {national_polls['pct_harris_poll'].mean() - true_harris_national:.2f}")


### battleground states combined treating all as one group
battleground_states = ['arizona', 'georgia', 'michigan', 'nevada', 'north carolina', 'pennsylvania', 'wisconsin']
bg_polls = harris_trump_pivot[harris_trump_pivot['state'].isin(battleground_states)].copy()
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# trump component all battleground polls combined
axes[0, 0].hist(bg_polls['trump_part_A'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 0].axvline(bg_polls['trump_part_A'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {bg_polls["trump_part_A"].mean():.4f}')
axes[0, 0].set_xlabel('Trump Component')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trump Component: ln(poll_trump) - ln(true_trump)')
axes[0, 0].legend()

# harris component all battleground polls combined
axes[0, 1].hist(bg_polls['harris_part_A'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 1].axvline(bg_polls['harris_part_A'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {bg_polls["harris_part_A"].mean():.4f}')
axes[0, 1].set_xlabel('Harris Component')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Harris Component: ln(poll_harris) - ln(true_harris)')
axes[0, 1].legend()

# method a all battleground polls combined
axes[0, 2].hist(bg_polls['A'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 2].axvline(bg_polls['A'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {bg_polls["A"].mean():.4f}')
axes[0, 2].set_xlabel('Method A')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Method A: trump_part - harris_part')
axes[0, 2].legend()

# harris poll distribution mean true value across all battleground states
mean_true_harris = (bg_polls['p_harris_true'] * 100).mean()
axes[1, 0].hist(bg_polls['pct_harris_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(mean_true_harris, color='red', linestyle='--', linewidth=2,
                   label=f'Mean True Value: {mean_true_harris:.2f}%')
axes[1, 0].axvline(bg_polls['pct_harris_poll'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Poll Mean: {bg_polls["pct_harris_poll"].mean():.2f}%')
axes[1, 0].set_xlabel('Harris Poll %')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Harris Poll Distribution')
axes[1, 0].legend()

# trump poll distribution mean true value across all battleground states
mean_true_trump = (bg_polls['p_trump_true'] * 100).mean()
axes[1, 1].hist(bg_polls['pct_trump_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 1].axvline(mean_true_trump, color='red', linestyle='--', linewidth=2,
                   label=f'Mean True Value: {mean_true_trump:.2f}%')
axes[1, 1].axvline(bg_polls['pct_trump_poll'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Poll Mean: {bg_polls["pct_trump_poll"].mean():.2f}%')
axes[1, 1].set_xlabel('Trump Poll %')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Trump Poll Distribution')
axes[1, 1].legend()

# simple errors both candidates combined
trump_error_all = bg_polls['pct_trump_poll'] - bg_polls['p_trump_true'] * 100
harris_error_all = bg_polls['pct_harris_poll'] - bg_polls['p_harris_true'] * 100

axes[1, 2].hist(trump_error_all.dropna(), bins=30, alpha=0.5, label=f'Trump (mean={trump_error_all.mean():.2f})',
               color='red', edgecolor='black')
axes[1, 2].hist(harris_error_all.dropna(), bins=30, alpha=0.5, label=f'Harris (mean={harris_error_all.mean():.2f})',
               color='blue', edgecolor='black')
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=2)
axes[1, 2].set_xlabel('Simple Error (Poll - True %)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Simple Errors')
axes[1, 2].legend()

plt.suptitle('All Battleground States Combined (AZ, GA, MI, NV, NC, PA, WI), Non Partisan', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundcombined_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### trump component, seven panels for each individual state
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# loop through each battleground state
for i, state in enumerate(battleground_states):
    state_data = bg_polls[bg_polls['state'] == state]
    axes[i].hist(state_data['trump_part_A'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[i].axvline(state_data['trump_part_A'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {state_data["trump_part_A"].mean():.4f}')
    axes[i].set_xlabel('Trump Component')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{state.title()} (N={len(state_data)})')
    axes[i].legend()

# hide the extra subplot
axes[7].axis('off')

plt.suptitle('Trump Component by State: ln(poll_trump) - ln(true_trump), Non Partisan', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_trumpcomp_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### harris component, seven panels for each individual state
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# loop through each battleground state
for i, state in enumerate(battleground_states):
    state_data = bg_polls[bg_polls['state'] == state]
    axes[i].hist(state_data['harris_part_A'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[i].axvline(state_data['harris_part_A'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {state_data["harris_part_A"].mean():.4f}')
    axes[i].set_xlabel('Harris Component')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{state.title()} (N={len(state_data)})')
    axes[i].legend()

# hide the extra subplot
axes[7].axis('off')

plt.suptitle('Harris Component by State, Non Partisan: ln(poll_harris) - ln(true_harris)', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_harriscomp_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### method a, seven panels for each individual state
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# loop through each battleground state
for i, state in enumerate(battleground_states):
    state_data = bg_polls[bg_polls['state'] == state]
    axes[i].hist(state_data['A'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[i].axvline(state_data['A'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {state_data["A"].mean():.4f}')
    axes[i].set_xlabel('Method A')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{state.title()} (N={len(state_data)})')
    axes[i].legend()


# hide the extra subplot
axes[7].axis('off')

plt.suptitle('Method A by State, Non Partisan: trump_part - harris_part', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_methoda_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### harris poll distribution, seven panels for each individual state
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# loop through each battleground state
for i, state in enumerate(battleground_states):
    state_data = bg_polls[bg_polls['state'] == state]
    true_harris = state_data['p_harris_true'].iloc[0] * 100
    axes[i].hist(state_data['pct_harris_poll'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(true_harris, color='red', linestyle='--', linewidth=2,
                   label=f'True: {true_harris:.2f}%')
    axes[i].axvline(state_data['pct_harris_poll'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Poll Mean: {state_data["pct_harris_poll"].mean():.2f}%')
    axes[i].set_xlabel('Harris Poll %')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{state.title()} (N={len(state_data)})')
    axes[i].legend()

# hide the extra subplot
axes[7].axis('off')

plt.suptitle('Harris Poll Distribution by State, Non Partisan', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_harrispoll_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### trump poll distribution, seven panels for each individual state
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# loop through each battleground state
for i, state in enumerate(battleground_states):
    state_data = bg_polls[bg_polls['state'] == state]
    true_trump = state_data['p_trump_true'].iloc[0] * 100
    axes[i].hist(state_data['pct_trump_poll'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(true_trump, color='red', linestyle='--', linewidth=2,
                   label=f'True: {true_trump:.2f}%')
    axes[i].axvline(state_data['pct_trump_poll'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Poll Mean: {state_data["pct_trump_poll"].mean():.2f}%')
    axes[i].set_xlabel('Trump Poll %')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{state.title()} (N={len(state_data)})')
    axes[i].legend()

# hide the extra subplot
axes[7].axis('off')

plt.suptitle('Trump Poll Distribution by State, Non Partisan', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_trumppoll_methoda_errors_non_partisan.png", dpi=300)
#plt.show()


### simple errors, seven panels for each individual state
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# loop through each battleground state
for i, state in enumerate(battleground_states):
    state_data = bg_polls[bg_polls['state'] == state]
    trump_error = state_data['pct_trump_poll'] - state_data['p_trump_true'] * 100
    harris_error = state_data['pct_harris_poll'] - state_data['p_harris_true'] * 100
   
    axes[i].hist(trump_error.dropna(), bins=20, alpha=0.5, label=f'Trump (mean={trump_error.mean():.2f})',
                color='red', edgecolor='black')
    axes[i].hist(harris_error.dropna(), bins=20, alpha=0.5, label=f'Harris (mean={harris_error.mean():.2f})',
                color='blue', edgecolor='black')
    axes[i].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[i].set_xlabel('Simple Error (Poll - True %)')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{state.title()} (N={len(state_data)})')
    axes[i].legend()

# hide the extra subplot
axes[7].axis('off')

plt.suptitle('Simple Errors by State, Non Partisan', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_simpleerr_methoda_errors_non_partisan.png", dpi=300)
#plt.show()

### battleground summary stats
# battleground state summary statistics combined
print(f"\nAll Battleground States Combined (N={len(bg_polls)}), Non Partisan:")
print(f"  Trump Component Mean: {bg_polls['trump_part_A'].mean():.4f}")
print(f"  Harris Component Mean: {bg_polls['harris_part_A'].mean():.4f}")
print(f"  Method A Mean: {bg_polls['A'].mean():.4f}")
print(f"  Trump Error Mean: {trump_error_all.mean():.2f}")
print(f"  Harris Error Mean: {harris_error_all.mean():.2f}")


# battleground state summary statistics by state
print(f"\nBy State, Non Partisan:")
for state in battleground_states:
    state_data = bg_polls[bg_polls['state'] == state]
    true_trump = state_data['p_trump_true'].iloc[0] * 100
    true_harris = state_data['p_harris_true'].iloc[0] * 100
    poll_trump = state_data['pct_trump_poll'].mean()
    poll_harris = state_data['pct_harris_poll'].mean()
    trump_error = poll_trump - true_trump
    harris_error = poll_harris - true_harris
   
    print(f"\n{state} (N={len(state_data)}):")
    print(f"  Trump True: {true_trump:.2f}%, Poll Mean: {poll_trump:.2f}%, Error: {trump_error:.2f}")
    print(f"  Harris True: {true_harris:.2f}%, Poll Mean: {poll_harris:.2f}%, Error: {harris_error:.2f}")
    print(f"  Trump Component Mean: {state_data['trump_part_A'].mean():.4f}")
    print(f"  Harris Component Mean: {state_data['harris_part_A'].mean():.4f}")
    print(f"  Method A Mean: {state_data['A'].mean():.4f}")


######## accuracy split by state vs national
print(f"\nMethod A accuracy by poll level (state vs national), Non Partisan:")
results = []
for level in harris_trump_pivot['poll_level'].unique():
    subdf = harris_trump_pivot[harris_trump_pivot['poll_level'] == level]
    mean_A = subdf['A'].mean()
    se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')
    t_stat = mean_A / se_robust

    # only compute p-value if SE is valid
    if pd.notna(se_robust) and se_robust > 0:
        t_stat = mean_A / se_robust
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls-1))
    else:
        p_value = np.nan

    results.append({
        'poll_level': level,
        'mean': mean_A,
        'median': subdf['A'].median(),
        'se': se_robust,
        'p_value': p_value,
        'sd': subdf['A'].std(),
        'n': len(subdf)
    })

results_df = pd.DataFrame(results)
results_df['sig'] = results_df['p_value'].apply(sig_stars)
print(results_df.to_string(index=False))


########################################################################################
################################# SIMPLE MODE ANALSYIS PREP ############################
########################################################################################

# fix typo
harris_trump_pivot['mode'] = harris_trump_pivot['mode'].str.replace('LIve Phone', 'Live Phone', regex=False)

# classify modes as self-administered vs interviewer-administered
self_admin_modes = [
    'Online Opt-In Panel', 'Text-to-Web', 'Online Matched Sample', 
    'Online Ad', 'Probability Panel', 'Text', 'App Panel', 
    'Email', 'Mail-to-Web'
]

interviewer_modes = ['Live Phone', 'IVR']

# mixed modes that combine self-admin and interviewer
explicit_mixed_modes = ['Mail-to-Phone']

# create mode component indicators in the non exploded dataset
harris_trump_pivot['has_self_admin'] = harris_trump_pivot['mode'].apply(
    lambda x: any(mode.strip() in self_admin_modes for mode in str(x).split('/'))
)

harris_trump_pivot['has_interviewer'] = harris_trump_pivot['mode'].apply(
    lambda x: any(mode.strip() in interviewer_modes for mode in str(x).split('/'))
)

harris_trump_pivot['has_explicit_mixed'] = harris_trump_pivot['mode'].apply(
    lambda x: any(mode.strip() in explicit_mixed_modes for mode in str(x).split('/'))
)

# check for modes that don't fall into any category (the "Other" cases)
harris_trump_pivot['has_other'] = ~(
    harris_trump_pivot['has_self_admin'] | 
    harris_trump_pivot['has_interviewer'] | 
    harris_trump_pivot['has_explicit_mixed']
)

# print others
other_examples = harris_trump_pivot[harris_trump_pivot['has_other']]['mode'].value_counts()

print(f"\nOther category (N = {len(harris_trump_pivot[harris_trump_pivot['has_other']])} questions):")
print("  All modes in this category:")
for mode, count in other_examples.items():
    print(f"    '{mode}': {count} questions")

# exclude other (these all don't have a mode), keep everything else
harris_trump_simple_mode_analysis = harris_trump_pivot[
    ~harris_trump_pivot['has_other']
].copy()

# create three mutually exclusive binary indicators
# mixed = either has both components or is the explicit mixed mode Mail-to-Phone, others are solely one type
harris_trump_simple_mode_analysis['interviewer_only'] = (
    (harris_trump_simple_mode_analysis['has_interviewer']) & 
    (~harris_trump_simple_mode_analysis['has_self_admin']) &
    (~harris_trump_simple_mode_analysis['has_explicit_mixed'])
).astype(int)

harris_trump_simple_mode_analysis['self_admin_only'] = (
    (harris_trump_simple_mode_analysis['has_self_admin']) & 
    (~harris_trump_simple_mode_analysis['has_interviewer']) &
    (~harris_trump_simple_mode_analysis['has_explicit_mixed'])
).astype(int)

harris_trump_simple_mode_analysis['mixed_mode'] = (
    ((harris_trump_simple_mode_analysis['has_self_admin']) & 
     (harris_trump_simple_mode_analysis['has_interviewer'])) |
    harris_trump_simple_mode_analysis['has_explicit_mixed']
).astype(int)

# verify they sum to 1 for each row 
mode_sum = (harris_trump_simple_mode_analysis['interviewer_only'] + 
            harris_trump_simple_mode_analysis['self_admin_only'] + 
            harris_trump_simple_mode_analysis['mixed_mode'])

# was not necessary
if not (mode_sum == 1).all():
    print("\nWARNING: Mode categories are not mutually exclusive!")

print("MODE ANALYSIS DATAFRAME")
print(f"\nTotal questions: {len(harris_trump_simple_mode_analysis)}")
print(f"Excluded from original: {len(harris_trump_pivot) - len(harris_trump_simple_mode_analysis)}")
print(f"  Interviewer-only: {harris_trump_simple_mode_analysis['interviewer_only'].sum()}")
print(f"  Self-admin-only: {harris_trump_simple_mode_analysis['self_admin_only'].sum()}")
print(f"  Mixed mode: {harris_trump_simple_mode_analysis['mixed_mode'].sum()}")

print(f"\nMixed mode composition:")
mixed_polls = harris_trump_simple_mode_analysis[harris_trump_simple_mode_analysis['mixed_mode'] == 1]
slash_mixed = ((mixed_polls['has_self_admin']) & (mixed_polls['has_interviewer'])).sum()
explicit_mixed = mixed_polls['has_explicit_mixed'].sum()
print(f"  Slash-separated (e.g., 'Live Phone/Online'): {slash_mixed}")
print(f"  Explicit mixed (Mail-to-Phone): {explicit_mixed}")

# subset for pure binary analysis (excludes mixed)
harris_trump_simple_mode_analysis_pure = harris_trump_simple_mode_analysis[
    harris_trump_simple_mode_analysis['mixed_mode'] == 0
].copy()

print(f"\nPure mode subset (excludes mixed):")
print(f"  Total questions: {len(harris_trump_simple_mode_analysis_pure)}")
print(f"  Self-admin-only: {harris_trump_simple_mode_analysis_pure['self_admin_only'].sum()}")
print(f"  Interviewer-only: {harris_trump_simple_mode_analysis_pure['interviewer_only'].sum()}")

# save the full three-way dataset (interviewer-only, self-admin-only, mixed)
harris_trump_simple_mode_analysis.to_csv('data/harris_trump_datelimited_no_partisan_simple_mode_analysis_threeway.csv', index=False)

# save the pure binary dataset (excludes mixed)
harris_trump_simple_mode_analysis_pure.to_csv('data/harris_trump_datelimited_no_partisan_simple_mode_analysis_pure.csv', index=False)


########################################################################################
##################################### Mode Analysis ####################################
########################################################################################

######## explode slash-separated modes into base modes

# each mode with a slash will count in both listed
# explode slash-separated mode strings so a question with 'live phone/online' appears in counts and accuracy stats for both modes
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

print(f"\nBase mode breakdown (unique questions per mode), Non Partisan:\n")
print(mode_counts.to_string(index=False))

######## accuracy by base mode
print(f"\nMethod A accuracy by base mode")
results = []
for mode in harris_trump_modes['base_mode'].unique():
    subdf = harris_trump_modes[harris_trump_modes['base_mode'] == mode]
    mean_A = subdf['A'].mean()
    se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')
    t_stat = mean_A / se_robust

    # only compute p-value if SE is valid
    if pd.notna(se_robust) and se_robust > 0:
        t_stat = mean_A / se_robust
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls-1))
    else:
        p_value = np.nan

    results.append({
        'base_mode': mode,
        'mean': mean_A,
        'median': subdf['A'].median(),
        'sd': subdf['A'].std(),
        'se': se_robust,
        'p_value': p_value,
        'n': len(subdf)
    })

results_df = pd.DataFrame(results).sort_values('mean', ascending=False)
results_df['sig'] = results_df['p_value'].apply(sig_stars)
print(results_df.to_string(index=False))


########################################################################################
########################## Accuracy by mode and target population ######################
########################################################################################

# these two tables report mean, median, std, and n for method a broken out by (1) polling mode and (2) target population, each split by state vs national
# mode and population are reported separately from the regression because they are categorical design choices better understood descriptively, and the multi-hot nature of mode makes regression coefficients hard to interpret cleanly (but i may go back to this later so i can say something like mode X is more biased controlling for days before election and swingness)

# table print function
def print_accuracy_table(df, group_col, label):
    """
    groups df by group_col and poll_level, computes mean/median/std/se/p-value/n of method a,
    and prints a formatted table with state and national columns side by side
    """
    results = []
    
    for (group_val, level), subdf in df.groupby([group_col, 'poll_level']):
        mean_A = subdf['A'].mean()
        se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')
        t_stat = mean_A / se_robust

        # only compute p-value if SE is valid
        if pd.notna(se_robust) and se_robust > 0:
            t_stat = mean_A / se_robust
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls-1))
        else:
            p_value = np.nan

        results.append({
            group_col: group_val,
            'poll_level': level,
            'mean': mean_A,
            'se': se_robust,
            'p_value': p_value,
            'median': subdf['A'].median(),
            'std': subdf['A'].std(),
            'n': len(subdf)
        })
    
    results_df = pd.DataFrame(results)
    tbl_wide = results_df.pivot(index=group_col, columns='poll_level', 
                                  values=['mean', 'se', 'p_value', 'median', 'std', 'n'])
    tbl_wide.columns = [f"{stat}_{level}" for stat, level in tbl_wide.columns]
    tbl_wide = tbl_wide.reset_index().sort_values('mean_state', ascending=False, na_position='last')

    print(f"\n{'='*110}")
    print(f"  method a accuracy by {label}, Non Partisan")
    print(f"  (+ = republican bias, - = democratic bias)")
    print(f"{'='*110}")
    print(f"  {'':30} {'-------------- state --------------':>42} {'------------ national ------------':>42}")
    print(f"  {group_col:<30} {'mean':>8} {'se':>8} {'p-val':>8} {'median':>8} {'std':>8} {'n':>5}   "
          f"{'mean':>8} {'se':>8} {'p-val':>8} {'median':>8} {'std':>8} {'n':>5}")
    print(f"  {'-'*108}")

    for _, row in tbl_wide.iterrows():
        def fmt_val(val):
            return f"{val:.4f}" if pd.notna(val) else '   --'
        def fmt_pval(val):
            return f"{val:.3f}{sig_stars(val)}" if pd.notna(val) else '   --'
        def fmt_n(val):
            return f"{int(val)}" if pd.notna(val) and val > 0 else '--'

        print(
            f"  {str(row[group_col]):<30} "
            f"{fmt_val(row.get('mean_state')):>8} "
            f"{fmt_val(row.get('se_state')):>8} "
            f"{fmt_pval(row.get('p_value_state')):>8} "
            f"{fmt_val(row.get('median_state')):>8} "
            f"{fmt_val(row.get('std_state')):>8} "
            f"{fmt_n(row.get('n_state')):>5}   "
            f"{fmt_val(row.get('mean_national')):>8} "
            f"{fmt_val(row.get('se_national')):>8} "
            f"{fmt_pval(row.get('p_value_national')):>8} "
            f"{fmt_val(row.get('median_national')):>8} "
            f"{fmt_val(row.get('std_national')):>8} "
            f"{fmt_n(row.get('n_national')):>5}"
        )

    print(f"{'='*110}\n")

# Table: accuracy by polling mode
print_accuracy_table(harris_trump_modes, 'base_mode', 'polling mode')


# Table: accuracy by target population
# shows whether polls targeting different populations are systematically more or less accurate, without controlling for other factors
print_accuracy_table(harris_trump_pivot, 'population', 'target population')


########################################################################################
###################### Multivariate Regression Analysis Set Up #########################
########################################################################################

# unit of analysis: one row per question (question_id)
# dependent variable: method a (continuous, centered at 0)
# two separate regressions: state level polls and national polls
# standard errors: clustered by poll_id in both regressions (clustering on poll_id is robust to both heteroskedasticity and within-poll correlation, as multiple questions from the same poll have correlated errors. this is the huber-white sandwich estimator extended to clusters, satisfying the requirement from martin, traugott & kennedy for correlated errors)

# psuedocode for this section (removing once done)
# unit of analysis is a single survey
# do OLS regression mtulivariate predicting method A value 
# state and national polls are in different regressions 
# To control for correlated errors due to the clustering of the surveys within individual statewide studies, robust standard errors should be calculated in the manner suggested by Huber (1967) and White (1980) for all state but the overall national population surveys
# multivariate analysis to understand impact of different decisions on accuracy
# use method A as dependent variable 
# report the regression coefficient, standard error, signifacnce stars for each factor 
# adjusted r square at bottom, constant at bottom, N at bottom
# variables to include: base_mode / mode indicators, population indicators (have a, lv, rv)
# variables to create: duration in field (difference between start_date and end_date), days before election (end_date to november 5 2025), final absolute margin of victory (state or national depending on regression), percent of don't know (100 - total for all candidates in a question), total statewide turnout percent (i will need to find you a dataset with the number of people registered to vote in each state)

# function to run ols with clustered ses and print a formatted table
def run_ols_clustered(df, y_col, x_cols, cluster_col, label):
    """
    fits ols on df using x_cols to predict y_col.
    standard errors are clustered on cluster_col (huber-white sandwich).
    prints a formatted regression table with stars, adj-r2, constant, and n.
    returns the fitted statsmodels results object.
    """
    # drop rows with any missing values in the variables used
    df_reg = df[x_cols + [y_col, cluster_col]].dropna()

    # add intercept column with has_constant='add' to force it even if data
    # appears to already contain a constant — sm.add_constant names it 'const'
    X      = sm.add_constant(df_reg[x_cols], has_constant='add')
    y      = df_reg[y_col]
    groups = df_reg[cluster_col]

    # fit ols with cluster-robust covariance
    model  = sm.OLS(y, X)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})

    # significance stars based on two-tailed p-values
    def stars(p):
        if p < 0.01:   return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else:          return ''

    # print formatted table
    print(f"\n{'='*70}")
    print(f"  ols regression: {label}")
    print(f"  dependent variable: method a  (+ = republican bias)")
    print(f"  standard errors: clustered by {cluster_col}")
    print(f"{'='*70}")
    print(f"  {'variable':<35} {'coef':>10} {'se':>10} {'sig':>6}")
    print(f"  {'-'*63}")

    params  = result.params
    bse     = result.bse
    pvalues = result.pvalues

    # identify the intercept name defensively — add_constant uses 'const' by
    # default but older statsmodels versions may use 'Intercept' or 'intercept'
    intercept_name = next((v for v in params.index if v.lower() in ('const', 'intercept')), None)

    # print all covariates first, then intercept at the bottom
    var_order = [v for v in params.index if v != intercept_name] + ([intercept_name] if intercept_name else [])
    for var in var_order:
        print(f"  {var:<35} {params[var]:>10.4f} {bse[var]:>10.4f} {stars(pvalues[var]):>6}")

    print(f"  {'-'*63}")
    print(f"  adjusted r2:  {result.rsquared_adj:.4f}")
    print(f"  n:            {int(result.nobs)}")
    print(f"{'='*70}\n")

    return result

# build regression dataset from from the pivoted question-level accuracy dataset
reg_df = harris_trump_pivot.copy()

# VAR: duration in field
# number of days the poll was in the field (inclusive of ends)
# longer polls may average across more days of opinion movement
reg_df['duration_days'] = (reg_df['end_date'] - reg_df['start_date']).dt.days + 1

# VAR: days before election
# how far in advance of election day the poll ended
# polls closer to election day should in theory be more accurate
reg_df['days_before_election'] = (election_date - reg_df['end_date']).dt.days

# flag any polls that ended after election day (negative values)
# these would have negative days_before_election and may warrant exclusion
n_post_election = (reg_df['days_before_election'] < 0).sum()
print(f"\ntime variable diagnostics:")
print(f"  duration_days        -- mean: {reg_df['duration_days'].mean():.1f}, "
      f"min: {reg_df['duration_days'].min()}, max: {reg_df['duration_days'].max()}")
print(f"  days_before_election -- mean: {reg_df['days_before_election'].mean():.1f}, "
      f"min: {reg_df['days_before_election'].min()}, max: {reg_df['days_before_election'].max()}")
# print(f"  polls ending after election day (days_before_election < 0): {n_post_election}") # none yay! so commented out

# VAR: pct_dk
# already computed pre-pivot and merged into harris_trump_pivot above
# higher dk share may indicate less crystallized opinion

# VAR: abs_margin
# the absolute margin of victory (|trump_share - harris_share|) in the state, 
# or nationally captures swingness polling dynamics than lopsided ones, and pollsters may expend more effort in swing states

# VAR: statewide turnout
# load turnout data
turnout_data = pd.read_csv("data/Turnout_2024G_v0.3.csv")

# standardize state names and select needed columns
turnout_clean = turnout_data[['STATE', 'VEP_TURNOUT_RATE']].copy()
turnout_clean['state'] = turnout_clean['STATE'].str.strip().str.lower()
turnout_clean['turnout_pct'] = turnout_clean['VEP_TURNOUT_RATE'].str.rstrip('%').astype(float)

# rename united states to national
turnout_clean['state'] = turnout_clean['state'].replace('united states', 'national')

# merge turnout into reg_df
reg_df = reg_df.merge(
    turnout_clean[['state', 'turnout_pct']],
    on='state',
    how='left'
)


# covariates for both regressions
time_vars  = ['duration_days', 'days_before_election']
state_vars = ['pct_dk', 'abs_margin','turnout_pct']
national_vars = ['pct_dk', 'abs_margin']

# final covariate lists per regression
state_x_vars    = time_vars + state_vars
all_x_vars = time_vars + national_vars

# split into statelevel and national samples
reg_state    = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()

# define swing states and create subset
swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 
                      'north carolina', 'pennsylvania', 'wisconsin']
reg_state_swing = reg_state[reg_state['state'].isin(swing_states)].copy()

print(f"\nregression sample sizes, Non Partisan:")
print(f"  national-level questions: {len(reg_national)}")
print(f"  state-level questions:    {len(reg_state)}")
print(f"  swing state questions: {len(reg_state_swing)}")

########################################################################################
######## BASE REGRESSIONS (NO TIME WINDOWS, NO MODE, SWING/NATIONAL/STATES) ############
########################################################################################

# national regression: also clustered by poll_id for the same reason, though with fewer polls clustering matters less
results_national = run_ols_clustered(
    df          = reg_national,
    y_col       = 'A',
    x_cols      = all_x_vars,
    cluster_col = 'poll_id',
    label       = 'national polls, Non Partisan'
)

# state regression: clustered ses by poll_id to account for the fact that multiple questions from the same poll share correlated errors
results_state = run_ols_clustered(
    df          = reg_state,
    y_col       = 'A',
    x_cols      = state_x_vars,
    cluster_col = 'poll_id',
    label       = 'state-level polls, Non Partisan'
)

# swing state regression
results_swing = run_ols_clustered(
    df          = reg_state_swing,
    y_col       = 'A',
    x_cols      = state_x_vars,
    cluster_col = 'poll_id',
    label       = 'swing states, Non Partisan'
)

########################################################################################
#################### PREPARE MODE FOR REGRESSIONS (LIVE PHONE REFERENCE) ###############
########################################################################################

# explode mode into base_mode (each mixed mode poll becomes multiple rows)
reg_df['base_mode'] = reg_df['mode'].str.split('/')
reg_df = reg_df.explode('base_mode')
reg_df['base_mode'] = reg_df['base_mode'].str.strip()

print("\nUnique base modes after exploding:")
print(reg_df['base_mode'].value_counts())

# set live phone as reference category (following polling literature conventions and gold standard of live phone)
reference_mode = 'Live Phone'

# create dummy variables for all modes
mode_dummies = pd.get_dummies(reg_df['base_mode'], prefix='mode', drop_first=False)

# drop live phone to use as reference (if it exists)
if f'mode_{reference_mode}' in mode_dummies.columns:
    mode_dummies = mode_dummies.drop(f'mode_{reference_mode}', axis=1)
    print(f"\nreference category set to: {reference_mode}")


# convert boolean to int and clean column names by replace hyphens with underscores
mode_dummies = mode_dummies.astype(int)
mode_dummies.columns = mode_dummies.columns.str.replace('-', '_')

# add mode dummies to reg_df
reg_df = pd.concat([reg_df, mode_dummies], axis=1)
mode_vars = [col for col in reg_df.columns if col.startswith('mode_')]

print(f"mode dummy variables created: {mode_vars}")
print(f"\nall coefficients will be interpreted relative to {reference_mode} polls")

# resplit into state and swing state and national after exploding and adding mode dummies
reg_state = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()
reg_state_swing = reg_state[reg_state['state'].isin(swing_states)].copy()

print(f"\nsample sizes after exploding:")
print(f"  all state polls: {len(reg_state)}")
print(f"  swing states only: {len(reg_state_swing)}")
print(f"  national polls: {len(reg_national)}")

print(f"\nswing states breakdown:")
print(reg_state_swing['state'].value_counts().sort_index())

print(f"\nmode distribution in swing states:")
mode_dist = reg_state_swing['base_mode'].value_counts()
for mode, count in mode_dist.items():
    pct = 100 * count / len(reg_state_swing)
    marker = " (REFERENCE)" if mode == reference_mode else ""
    print(f"  {mode}: {count} ({pct:.1f}%){marker}")

# update covariate lists
# without mode
state_x_vars_no_mode = time_vars + state_vars
national_x_vars_no_mode = time_vars + national_vars

# with mode
state_x_vars_with_mode = time_vars + state_vars + mode_vars
national_x_vars_with_mode = time_vars + national_vars + mode_vars

########################################################################################
#################### BASE REGRESSIONS (NO TIME, WITH MODE, SWING/STATES/NATIONAL) ####################################
########################################################################################

print("REGRESSIONS WITH MODE CONTROLS")

# natioanl questions
results_national_mode = run_ols_clustered(
    df          = reg_national,
    y_col       = 'A',
    x_cols      = national_x_vars_with_mode,
    cluster_col = 'poll_id',
    label       = 'national polls with mode controls, Non Partisan'
)

# all state questions
results_all_states_mode = run_ols_clustered(
    df          = reg_state,
    y_col       = 'A',
    x_cols      = state_x_vars_with_mode,
    cluster_col = 'poll_id',
    label       = 'all state polls with mode controls, Non Partisan'
)

# swing state questions
results_swing_mode = run_ols_clustered(
    df          = reg_state_swing,
    y_col       = 'A',
    x_cols      = state_x_vars_with_mode,
    cluster_col = 'poll_id',
    label       = 'swing states with mode controls, Non Partisan'
)


########################################################################################
#################### TIME WINDOW REGRESSIONS (NO MODE, SWING/STATES/NATIONAL) ##########
########################################################################################

# re-run the same regression separately for five time windows prior to the election, where each window includes only polls whose end_date falls within that many days of election day
# ex: the 30 day window includes polls ending 30 or fewer days before election day
# lets us see whether the predictors of accuracy change as we get closer to election day, loses significance in shorter windows as found by harrison and desart/holbrook
# windows chosen: 107, 90, 60, 30, and 7 days.

time_windows = [107, 90, 60, 30, 7]

# swing states by time window (no mode)
swing_window_results_no_mode = {}

for window in time_windows:
    # filter swing states to this time window
    state_w = reg_state_swing[reg_state_swing['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  swing state questions in window: {len(state_w)}")
    
    # check if we have enough complete cases
    state_complete = state_w[state_x_vars_no_mode + ['A', 'poll_id']].dropna()
    
    if len(state_complete) < 10:
        print(f"  swing state regression skipped, only {len(state_complete)} complete cases")
        swing_window_results_no_mode[window] = None
    else:
        res_swing = run_ols_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_no_mode,
            cluster_col = 'poll_id',
            label       = f'swing states {window} days before election (no mode), Non Partisan'
        )
        swing_window_results_no_mode[window] = res_swing

# all states by time window (no mode)
all_states_window_results_no_mode = {}

for window in time_windows:
    # filter all states to this time window
    state_w = reg_state[reg_state['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  all state questions in window: {len(state_w)}")
    
    # check if we have enough complete cases
    state_complete = state_w[state_x_vars_no_mode + ['A', 'poll_id']].dropna()
    
    if len(state_complete) < 10:
        print(f"  all state regression skipped, only {len(state_complete)} complete cases")
        all_states_window_results_no_mode[window] = None
    else:
        res_state = run_ols_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_no_mode,
            cluster_col = 'poll_id',
            label       = f'all states {window} days before election (no mode), Non Partisan'
        )
        all_states_window_results_no_mode[window] = res_state

# national by time window (no mode)
national_window_results_no_mode = {}

for window in time_windows:
    # filter national to this time window
    national_w = reg_national[reg_national['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  national questions in window: {len(national_w)}")
    
    # check if we have enough complete cases
    national_complete = national_w[national_x_vars_no_mode + ['A', 'poll_id']].dropna()
    
    if len(national_complete) < 10:
        print(f"  national regression skipped, only {len(national_complete)} complete cases")
        national_window_results_no_mode[window] = None
    else:
        res_national = run_ols_clustered(
            df          = national_w,
            y_col       = 'A',
            x_cols      = national_x_vars_no_mode,
            cluster_col = 'poll_id',
            label       = f'national {window} days before election (no mode), Non Partisan'
        )
        national_window_results_no_mode[window] = res_national


########################################################################################
#################### TIME WINDOW REGRESSIONS (WITH MODE, SWING/STATES/NATIONAL) ##########
########################################################################################

# swing states by time window (with mode)
swing_window_results_with_mode = {}

for window in time_windows:
    # filter swing states to this time window
    state_w = reg_state_swing[reg_state_swing['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  swing state questions in window: {len(state_w)}")
    
    # check if we have enough complete cases
    state_complete = state_w[state_x_vars_with_mode + ['A', 'poll_id']].dropna()
    
    if len(state_complete) < 10:
        print(f"  swing state regression skipped, only {len(state_complete)} complete cases")
        swing_window_results_with_mode[window] = None
    else:
        res_swing = run_ols_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_with_mode,
            cluster_col = 'poll_id',
            label       = f'swing states {window} days before election (with mode), Non Partisan'
        )
        swing_window_results_with_mode[window] = res_swing

# all states by time window (with mode)
all_states_window_results_with_mode = {}

for window in time_windows:
    # filter all states to this time window
    state_w = reg_state[reg_state['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  all state questions in window: {len(state_w)}")
    
    # check if we have enough complete cases
    state_complete = state_w[state_x_vars_with_mode + ['A', 'poll_id']].dropna()
    
    if len(state_complete) < 10:
        print(f"  all state regression skipped, only {len(state_complete)} complete cases")
        all_states_window_results_with_mode[window] = None
    else:
        res_state = run_ols_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_with_mode,
            cluster_col = 'poll_id',
            label       = f'all states {window} days before election (with mode), Non Partisan'
        )
        all_states_window_results_with_mode[window] = res_state

# national by time window (with mode)
national_window_results_with_mode = {}

for window in time_windows:
    # filter national to this time window
    national_w = reg_national[reg_national['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  national questions in window: {len(national_w)}")
    
    # check if we have enough complete cases
    national_complete = national_w[national_x_vars_with_mode + ['A', 'poll_id']].dropna()
    
    if len(national_complete) < 10:
        print(f"  national regression skipped, only {len(national_complete)} complete cases")
        national_window_results_with_mode[window] = None
    else:
        res_national = run_ols_clustered(
            df          = national_w,
            y_col       = 'A',
            x_cols      = national_x_vars_with_mode,
            cluster_col = 'poll_id',
            label       = f'national {window} days before election (with mode), Non Partisan'
        )
        national_window_results_with_mode[window] = res_national

###############################################################################


### SUMMARY TABLES


######################################################################################

# redfine because before was in loops
def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    else: return ''

########################################################################################
#################### NATIONAL, ACROSS TIME WINDOWS, NO MODE #######################
########################################################################################

# compare across columns to see how predictors change over time for national polls
print("\n" + "="*110)
print("NATIONAL, ACROSS TIME WINDOWS, NO MODE, NON PARTISAN")
print("="*110)

print(f"\n{'Variable':<30}", end='')
for window in time_windows:
    print(f"{'Coef.':>10} {'Std. Err.':>10} {'Sig.':>5}", end='')
print()
print(f"{'':30}", end='')
for window in time_windows:
    print(f"{window:>10}d{' '*15}", end='')
print()
print("." * 110)

national_vars_ordered = ['duration_days', 'days_before_election', 'pct_dk', 'abs_margin']

for var in national_vars_ordered:
    print(f"{var:<30}", end='')
    
    for window in time_windows:
        result = national_window_results_no_mode[window]
        if result is not None and var in result.params:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = stars(pval)
            print(f"{coef:>10.3f} {se:>10.3f} {sig:>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

print()
print(f"{'Constant':<30}", end='')
for window in time_windows:
    result = national_window_results_no_mode[window]
    if result is not None:
        intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
        if intercept_name:
            coef = result.params[intercept_name]
            se = result.bse[intercept_name]
            print(f"{coef:>10.3f} {se:>10.3f} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    else:
        print(f"{'':>10} {'':>10} {'':>5}", end='')
print()

print()
print(f"{'Adjusted R Square':<30}", end='')
for window in time_windows:
    result = national_window_results_no_mode[window]
    if result is not None:
        print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print(f"{'N':<30}", end='')
for window in time_windows:
    result = national_window_results_no_mode[window]
    if result is not None:
        print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print("\nNote: Robust Standard Errors Reported")
print("Sig: *<.10; **<.05; ***<.01")


########################################################################################
#################### NATIONAL, ACROSS TIME WINDOWS, WITH MODE ################
########################################################################################

print("\n" + "="*110)
print("NATIONAL, ACROSS TIME WINDOWS, WITH MODE, NON PARTISAN")
print(f"reference mode: {reference_mode}")
print("="*110)

print(f"\n{'Variable':<30}", end='')
for window in time_windows:
    print(f"{'Coef.':>10} {'Std. Err.':>10} {'Sig.':>5}", end='')
print()
print(f"{'':30}", end='')
for window in time_windows:
    print(f"{window:>10}d{' '*15}", end='')
print()
print("." * 110)

national_vars_with_mode = national_vars_ordered + sorted(mode_vars)

for var in national_vars_with_mode:
    print(f"{var:<30}", end='')
    
    for window in time_windows:
        result = national_window_results_with_mode[window]
        if result is not None and var in result.params:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = stars(pval)
            print(f"{coef:>10.3f} {se:>10.3f} {sig:>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

print()
print(f"{'Constant':<30}", end='')
for window in time_windows:
    result = national_window_results_with_mode[window]
    if result is not None:
        intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
        if intercept_name:
            coef = result.params[intercept_name]
            se = result.bse[intercept_name]
            print(f"{coef:>10.3f} {se:>10.3f} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    else:
        print(f"{'':>10} {'':>10} {'':>5}", end='')
print()

print()
print(f"{'Adjusted R Square':<30}", end='')
for window in time_windows:
    result = national_window_results_with_mode[window]
    if result is not None:
        print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print(f"{'N':<30}", end='')
for window in time_windows:
    result = national_window_results_with_mode[window]
    if result is not None:
        print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print("\nNote: Robust Standard Errors Reported")
print("Sig: *<.10; **<.05; ***<.01")
print(f"all coefficients relative to {reference_mode} (reference category)")


########################################################################################
#################### ALL STATES, ACROSS TIME WINDOWS, NO MODE ##################################
########################################################################################

print("\n" + "="*110)
print("ALL STATES, ACROSS TIME WINDOWS, NO MODE, NON PARTISAN")
print("="*110)

print(f"\n{'Variable':<30}", end='')
for window in time_windows:
    print(f"{'Coef.':>10} {'Std. Err.':>10} {'Sig.':>5}", end='')
print()
print(f"{'':30}", end='')
for window in time_windows:
    print(f"{window:>10}d{' '*15}", end='')
print()
print("." * 110)

swing_vars_ordered = ['duration_days', 'days_before_election', 'pct_dk', 'abs_margin', 'turnout_pct']

for var in swing_vars_ordered:
    print(f"{var:<30}", end='')
    
    for window in time_windows:
        result = all_states_window_results_no_mode[window]
        if result is not None and var in result.params:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = stars(pval)
            print(f"{coef:>10.3f} {se:>10.3f} {sig:>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

print()
print(f"{'Constant':<30}", end='')
for window in time_windows:
    result = all_states_window_results_no_mode[window]
    if result is not None:
        intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
        if intercept_name:
            coef = result.params[intercept_name]
            se = result.bse[intercept_name]
            print(f"{coef:>10.3f} {se:>10.3f} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    else:
        print(f"{'':>10} {'':>10} {'':>5}", end='')
print()

print()
print(f"{'Adjusted R Square':<30}", end='')
for window in time_windows:
    result = all_states_window_results_no_mode[window]
    if result is not None:
        print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print(f"{'N':<30}", end='')
for window in time_windows:
    result = all_states_window_results_no_mode[window]
    if result is not None:
        print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print("\nNote: Robust Standard Errors Reported")
print("Sig: *<.10; **<.05; ***<.01")

########################################################################################
#################### ALL STATES, ACROSS TIME WINDOWS, WITH MODE ##################################
########################################################################################

print("\n" + "="*110)
print("ALL STATES, ACROSS TIME WINDOWS, WITH MODE, NON PARTISAN")
print(f"reference mode: {reference_mode}")
print("="*110)

print(f"\n{'Variable':<30}", end='')
for window in time_windows:
    print(f"{'Coef.':>10} {'Std. Err.':>10} {'Sig.':>5}", end='')
print()
print(f"{'':30}", end='')
for window in time_windows:
    print(f"{window:>10}d{' '*15}", end='')
print()
print("." * 110)

swing_vars_with_mode = swing_vars_ordered + sorted(mode_vars)

for var in swing_vars_with_mode:
    print(f"{var:<30}", end='')
    
    for window in time_windows:
        result = all_states_window_results_with_mode[window]
        if result is not None and var in result.params:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = stars(pval)
            print(f"{coef:>10.3f} {se:>10.3f} {sig:>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

print()
print(f"{'Constant':<30}", end='')
for window in time_windows:
    result = all_states_window_results_with_mode[window]
    if result is not None:
        intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
        if intercept_name:
            coef = result.params[intercept_name]
            se = result.bse[intercept_name]
            print(f"{coef:>10.3f} {se:>10.3f} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    else:
        print(f"{'':>10} {'':>10} {'':>5}", end='')
print()

print()
print(f"{'Adjusted R Square':<30}", end='')
for window in time_windows:
    result = all_states_window_results_with_mode[window]
    if result is not None:
        print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print(f"{'N':<30}", end='')
for window in time_windows:
    result = all_states_window_results_with_mode[window]
    if result is not None:
        print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print("\nNote: Robust Standard Errors Reported")
print("Sig: *<.10; **<.05; ***<.01")
print(f"all coefficients relative to {reference_mode} (reference category)")


########################################################################################
#################### SWING, ACROSS TIME WINDOWS, NO MODE ##################################
########################################################################################

# compare across columns to see how predictors change over time, similar to harrison (2009) analysis of temporal dynamics
print("\n" + "="*110)
print("SWING, ACROSS TIME WINDOWS, NO MODE, NON PARTISAN")
print("="*110)

print(f"\n{'Variable':<30}", end='')
for window in time_windows:
    print(f"{'Coef.':>10} {'Std. Err.':>10} {'Sig.':>5}", end='')
print()
print(f"{'':30}", end='')
for window in time_windows:
    print(f"{window:>10}d{' '*15}", end='')
print()
print("." * 110)

for var in swing_vars_ordered:
    print(f"{var:<30}", end='')
    
    for window in time_windows:
        result = swing_window_results_no_mode[window]
        if result is not None and var in result.params:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = stars(pval)
            print(f"{coef:>10.3f} {se:>10.3f} {sig:>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

print()
print(f"{'Constant':<30}", end='')
for window in time_windows:
    result = swing_window_results_no_mode[window]
    if result is not None:
        intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
        if intercept_name:
            coef = result.params[intercept_name]
            se = result.bse[intercept_name]
            print(f"{coef:>10.3f} {se:>10.3f} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    else:
        print(f"{'':>10} {'':>10} {'':>5}", end='')
print()

print()
print(f"{'Adjusted R Square':<30}", end='')
for window in time_windows:
    result = swing_window_results_no_mode[window]
    if result is not None:
        print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print(f"{'N':<30}", end='')
for window in time_windows:
    result = swing_window_results_no_mode[window]
    if result is not None:
        print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print("\nNote: Robust Standard Errors Reported")
print("Sig: *<.10; **<.05; ***<.01")

########################################################################################
#################### SWING, ACROSS TIME WINDOWS< WITH MODE #############################
########################################################################################

print("\n" + "="*110)
print("SWING, ACROSS TIME WINDOWS< WITH MODE, NON PARTISAN")
print(f"reference mode: {reference_mode}")
print("="*110)

print(f"\n{'Variable':<30}", end='')
for window in time_windows:
    print(f"{'Coef.':>10} {'Std. Err.':>10} {'Sig.':>5}", end='')
print()
print(f"{'':30}", end='')
for window in time_windows:
    print(f"{window:>10}d{' '*15}", end='')
print()
print("." * 110)

for var in swing_vars_with_mode:
    print(f"{var:<30}", end='')
    
    for window in time_windows:
        result = swing_window_results_with_mode[window]
        if result is not None and var in result.params:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = stars(pval)
            print(f"{coef:>10.3f} {se:>10.3f} {sig:>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

print()
print(f"{'Constant':<30}", end='')
for window in time_windows:
    result = swing_window_results_with_mode[window]
    if result is not None:
        intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
        if intercept_name:
            coef = result.params[intercept_name]
            se = result.bse[intercept_name]
            print(f"{coef:>10.3f} {se:>10.3f} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    else:
        print(f"{'':>10} {'':>10} {'':>5}", end='')
print()

print()
print(f"{'Adjusted R Square':<30}", end='')
for window in time_windows:
    result = swing_window_results_with_mode[window]
    if result is not None:
        print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print(f"{'N':<30}", end='')
for window in time_windows:
    result = swing_window_results_with_mode[window]
    if result is not None:
        print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
    else:
        print(f"{'':>25}", end='')
print()

print("\nNote: Robust Standard Errors Reported")
print("Sig: *<.10; **<.05; ***<.01")


########################################################################################
###### NO TIME WINDOWS, MODE ONLY, ALL SUBSETS ######
########################################################################################

# positive coefficients indicate more republican bias, compare across columns to see if mode effects differ by sample
print("MODE COEFFICIENTS ACROSS SAMPLES, NON PARTISAN")

# create comparison table across the three mode regressions
print(f"\n{'Mode':<20} {'National':>15} {'Swing':>15} {'All States':>15}")
print("." * 70)

for mode_var in sorted(mode_vars):
    mode_name = mode_var.replace('mode_', '')
    print(f"{mode_name:<20}", end='')
    
    # national
    if results_national_mode is not None and mode_var in results_national_mode.params:
        coef = results_national_mode.params[mode_var]
        pval = results_national_mode.pvalues[mode_var]
        sig = stars(pval)
        print(f"{coef:>12.2f}{sig:<3}", end='')
    else:
        print(f"{'..':>15}", end='')
    
    # all states
    if results_all_states_mode is not None and mode_var in results_all_states_mode.params:
        coef = results_all_states_mode.params[mode_var]
        pval = results_all_states_mode.pvalues[mode_var]
        sig = stars(pval)
        print(f"{coef:>12.2f}{sig:<3}", end='')
    else:
        print(f"{'..':>15}", end='')

    # swing
    if results_swing_mode is not None and mode_var in results_swing_mode.params:
        coef = results_swing_mode.params[mode_var]
        pval = results_swing_mode.pvalues[mode_var]
        sig = stars(pval)
        print(f"{coef:>12.2f}{sig:<3}", end='')
    else:
        print(f"{'..':>15}", end='')
    
    print()

print("\nnote: *** p<0.01, ** p<0.05, * p<0.10")
print(f"all coefficients relative to {reference_mode} (reference category)")


######## save outputs
# save question-level accuracy dataset for further analysis
harris_trump_pivot.to_csv('data/harris_trump_datelimted_accuracy_non_partisan.csv', index=False)

# save regression-ready dataset withs constructed covariates
reg_df.to_csv('data/harris_trump_datelimted_regression_non_partisan.csv', index=False)

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("Accuracy analysis complete — see output/fiftyplusone_analysis_datelimited_no_partisan_log.txt")