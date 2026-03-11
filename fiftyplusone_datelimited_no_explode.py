import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# redirect all print output to a log file
log_file = open('output/fiftyplusone_analysis_datelimited_log_no_exploded.txt', 'w')
sys.stdout = log_file

# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS
# IT IS LIMITED TO THE TIME AFTER BIDEN DROPPED OUT, for full time see fiftyplusone_analysis.py
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy

# MODE STRATEGY: indicator (no explosion)
# each poll retains its original row; multi-mode polls get a 1 on every active mode indicator
# Text and Text_to_Web are recoded into mutually exclusive categories:
#   mode_Text_only    = uses Text but NOT Text_to_Web (avoids nesting collinearity)
#   mode_Text_to_Web  = uses Text_to_Web (always involves Text recruitment)
# reference category: Live Phone

# load cleaned harris+trump questions dataset (output from fiftyplusone_initial_analysis.py)
harris_trump_full_df = pd.read_csv("data/fiftyplusone_cleaned_harris_trump_questions.csv")

# load actual 2024 results
true_votes = pd.read_csv("data/true_votes_by_state_mengrep.csv")

# define cutoff date (Biden dropout)
dropout_cutoff = pd.Timestamp('2024-07-21')

# election day 2024
election_date = pd.Timestamp('2024-11-05')

# fix one dataset error
harris_trump_full_df['mode'] = harris_trump_full_df['mode'].str.replace('LIve Phone', 'Live Phone', regex=False)


########################################################################################
############################# New fields ################################
########################################################################################
# pct_dk captures the share of respondents not supporting any named candidate
# it equals 100 minus the sum of all named candidates' percentages per question
# this picks up undecided voters, third-party supporters, and refusals combined
# compute this before pivoting because the pivot only keeps trump and harris rows with pct values

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
    [['question_id', 'poll_id', 'pollster', 'partisan', 'state', 'start_date', 'end_date',
      'mode', 'population', 'sample_size']]
    .drop_duplicates('question_id')
)

harris_trump_pivot = harris_trump_pivot.merge(metadata_to_merge, on='question_id', how='left')

# pivot_table leaves a residual answer name on the columns axis
harris_trump_pivot.columns.name = None

# re-cast start_date and end_date to datetime after pivot
harris_trump_pivot['end_date']   = pd.to_datetime(harris_trump_pivot['end_date'])
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


##### LIMIT THE DATES TO PERIOD WHEN HARRIS WAS NOMINEE
harris_trump_pivot = harris_trump_pivot[harris_trump_pivot['start_date'] >= dropout_cutoff]

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
print(f"Total number of polls: {harris_trump_pivot['poll_id'].nunique()}")

######## compute Method A accuracy measure
# A = ln((poll_trump / poll_harris) / (true_trump / true_harris))
# A = 0: perfect accuracy
# A > 0: Republican bias (poll overestimates Trump relative to Harris)
# A < 0: Democratic bias (poll overestimates Harris relative to Trump)
# the log-odds ratio form is symmetric and scale-invariant

harris_trump_pivot['A'] = np.log(
    (harris_trump_pivot['pct_trump_poll'] / harris_trump_pivot['pct_harris_poll']) /
    (harris_trump_pivot['p_trump_true']   / harris_trump_pivot['p_harris_true'])
)

# decompose into trump and harris parts
harris_trump_pivot['trump_part_A']  = np.log(harris_trump_pivot['pct_trump_poll'] / 100)  - np.log(harris_trump_pivot['p_trump_true'])
harris_trump_pivot['harris_part_A'] = np.log(harris_trump_pivot['pct_harris_poll'] / 100) - np.log(harris_trump_pivot['p_harris_true'])

# flag poll level (state vs national)
harris_trump_pivot['poll_level'] = np.where(
    harris_trump_pivot['state'] == 'national', 'national', 'state'
)

print(f"\npoll_level counts after flagging:")
print(harris_trump_pivot['poll_level'].value_counts(dropna=False).to_string())
print(f"\nstate values flagged as national:")
print(harris_trump_pivot[harris_trump_pivot['poll_level'] == 'national']['state'].value_counts(dropna=False).to_string())


########################################################################################
############################# Create partisan flag ################################
########################################################################################
harris_trump_pivot["partisan_flag"] = (
    harris_trump_pivot["partisan"].notna() &
    (harris_trump_pivot["partisan"].str.strip() != "")
).astype(int)

flagged   = (harris_trump_pivot["partisan_flag"] == 1).sum()
unflagged = (harris_trump_pivot["partisan_flag"] == 0).sum()
print("Partisan flagged:",   flagged)
print("Partisan unflagged:", unflagged)


########################################################################################
############################# General Accuracy Analysis ################################
########################################################################################

def compute_clustered_se(df, value_col, cluster_col):
    """compute cluster-robust standard error of the mean"""
    df_clean = df[[value_col, cluster_col]].dropna()
    if len(df_clean) == 0:
        return np.nan, 0
    cluster_means = df_clean.groupby(cluster_col)[value_col].mean()
    n_clusters    = len(cluster_means)
    if n_clusters < 2:
        return np.nan, n_clusters
    grand_mean   = df_clean[value_col].mean()
    cluster_var  = ((cluster_means - grand_mean) ** 2).sum() / (n_clusters - 1)
    se_robust    = np.sqrt(cluster_var / n_clusters)
    return se_robust, n_clusters

def sig_stars(p):
    if p < 0.01:   return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    else:          return ''


######## overall accuracy
mean_A          = harris_trump_pivot['A'].mean()
se_A_robust, n_polls = compute_clustered_se(harris_trump_pivot, 'A', 'poll_id')
t_stat  = mean_A / se_A_robust
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls - 1))

mean_trump_part_A  = harris_trump_pivot['trump_part_A'].mean()
mean_harris_part_A = harris_trump_pivot['harris_part_A'].mean()

print(f"\nOverall Method A accuracy (all Harris+Trump questions):")
print(f"  Mean:   {mean_A:.4f}")
print(f"  SE:     {se_A_robust:.4f}")
print(f"  p-value:{p_value:.4f} {sig_stars(p_value)}")
print(f"  Median: {harris_trump_pivot['A'].median():.4f}")
print(f"  SD:     {harris_trump_pivot['A'].std():.4f}")
print(f"  N:      {len(harris_trump_pivot)}")
print(f"  Mean Harris Part: {mean_harris_part_A:.4f}")
print(f"  Mean Trump Part:  {mean_trump_part_A:.4f}")


### graph of trump and harris parts, all polls
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].hist(harris_trump_pivot['trump_part_A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(harris_trump_pivot['trump_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {harris_trump_pivot["trump_part_A"].mean():.4f}')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Trump Component')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trump Component: ln(poll_trump) - ln(true_trump)')
axes[0, 0].legend()

axes[0, 1].hist(harris_trump_pivot['harris_part_A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(harris_trump_pivot['harris_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {harris_trump_pivot["harris_part_A"].mean():.4f}')
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Harris Component')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Harris Component: ln(poll_harris) - ln(true_harris)')
axes[0, 1].legend()

axes[0, 2].hist(harris_trump_pivot['A'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(harris_trump_pivot['A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {harris_trump_pivot["A"].mean():.4f}')
axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0, 2].set_xlabel('Method A')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Method A: trump_part - harris_part')
axes[0, 2].legend()

axes[1, 0].scatter(harris_trump_pivot['p_trump_true'] * 100, harris_trump_pivot['pct_trump_poll'], alpha=0.3, s=10)
axes[1, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect accuracy')
axes[1, 0].set_xlabel('True Trump %')
axes[1, 0].set_ylabel('Poll Trump %')
axes[1, 0].set_title('Trump: Poll vs True')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(harris_trump_pivot['p_harris_true'] * 100, harris_trump_pivot['pct_harris_poll'], alpha=0.3, s=10)
axes[1, 1].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect accuracy')
axes[1, 1].set_xlabel('True Harris %')
axes[1, 1].set_ylabel('Poll Harris %')
axes[1, 1].set_title('Harris: Poll vs True')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(harris_trump_pivot['pct_trump_poll']  - harris_trump_pivot['p_trump_true']  * 100, bins=50, alpha=0.5, label='Trump Error',  edgecolor='black')
axes[1, 2].hist(harris_trump_pivot['pct_harris_poll'] - harris_trump_pivot['p_harris_true'] * 100, bins=50, alpha=0.5, label='Harris Error', edgecolor='black')
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_xlabel('Simple Error (Poll - True)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Simple Errors: Poll % - True %')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_overall_methoda_errors_no_explode.png", dpi=300)


### graphs of parts for NATIONAL polls only
national_polls = harris_trump_pivot[harris_trump_pivot['state'] == 'national'].copy()
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

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

true_trump_national  = national_polls['p_trump_true'].iloc[0]  * 100
true_harris_national = national_polls['p_harris_true'].iloc[0] * 100

axes[1, 0].hist(national_polls['pct_trump_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(true_trump_national, color='red',  linestyle='--', linewidth=2, label=f'True: {true_trump_national:.2f}%')
axes[1, 0].axvline(national_polls['pct_trump_poll'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Poll Mean: {national_polls["pct_trump_poll"].mean():.2f}%')
axes[1, 0].set_xlabel('Trump Poll %')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Trump Poll Values (National)')
axes[1, 0].legend()

axes[1, 1].hist(national_polls['pct_harris_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(true_harris_national, color='red',  linestyle='--', linewidth=2, label=f'True: {true_harris_national:.2f}%')
axes[1, 1].axvline(national_polls['pct_harris_poll'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Poll Mean: {national_polls["pct_harris_poll"].mean():.2f}%')
axes[1, 1].set_xlabel('Harris Poll %')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Harris Poll Values (National)')
axes[1, 1].legend()

axes[1, 2].hist(national_polls['pct_trump_poll']  - national_polls['p_trump_true']  * 100, bins=50, alpha=0.5, label='Trump Error',  edgecolor='black')
axes[1, 2].hist(national_polls['pct_harris_poll'] - national_polls['p_harris_true'] * 100, bins=50, alpha=0.5, label='Harris Error', edgecolor='black')
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1, 2].set_xlabel('Simple Error (Poll - True)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Simple Errors (National): Poll % - True %')
axes[1, 2].legend()

plt.suptitle('National Polls Only', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_national_methoda_errors_no_explode.png", dpi=300)

print(f"\nNational Polls (N={len(national_polls)}):")
print(f"Trump  - True: {true_trump_national:.2f}%,  Poll Mean: {national_polls['pct_trump_poll'].mean():.2f}%,  Error: {national_polls['pct_trump_poll'].mean() - true_trump_national:.2f}")
print(f"Harris - True: {true_harris_national:.2f}%, Poll Mean: {national_polls['pct_harris_poll'].mean():.2f}%, Error: {national_polls['pct_harris_poll'].mean() - true_harris_national:.2f}")


### battleground states combined
battleground_states = ['arizona', 'georgia', 'michigan', 'nevada', 'north carolina', 'pennsylvania', 'wisconsin']
bg_polls = harris_trump_pivot[harris_trump_pivot['state'].isin(battleground_states)].copy()
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].hist(bg_polls['trump_part_A'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 0].axvline(bg_polls['trump_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {bg_polls["trump_part_A"].mean():.4f}')
axes[0, 0].set_xlabel('Trump Component')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Trump Component: ln(poll_trump) - ln(true_trump)')
axes[0, 0].legend()

axes[0, 1].hist(bg_polls['harris_part_A'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 1].axvline(bg_polls['harris_part_A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {bg_polls["harris_part_A"].mean():.4f}')
axes[0, 1].set_xlabel('Harris Component')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Harris Component: ln(poll_harris) - ln(true_harris)')
axes[0, 1].legend()

axes[0, 2].hist(bg_polls['A'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 2].axvline(bg_polls['A'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {bg_polls["A"].mean():.4f}')
axes[0, 2].set_xlabel('Method A')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Method A: trump_part - harris_part')
axes[0, 2].legend()

mean_true_harris = (bg_polls['p_harris_true'] * 100).mean()
axes[1, 0].hist(bg_polls['pct_harris_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(mean_true_harris, color='red', linestyle='--', linewidth=2, label=f'Mean True Value: {mean_true_harris:.2f}%')
axes[1, 0].axvline(bg_polls['pct_harris_poll'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Poll Mean: {bg_polls["pct_harris_poll"].mean():.2f}%')
axes[1, 0].set_xlabel('Harris Poll %')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Harris Poll Distribution')
axes[1, 0].legend()

mean_true_trump = (bg_polls['p_trump_true'] * 100).mean()
axes[1, 1].hist(bg_polls['pct_trump_poll'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 1].axvline(mean_true_trump, color='red', linestyle='--', linewidth=2, label=f'Mean True Value: {mean_true_trump:.2f}%')
axes[1, 1].axvline(bg_polls['pct_trump_poll'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Poll Mean: {bg_polls["pct_trump_poll"].mean():.2f}%')
axes[1, 1].set_xlabel('Trump Poll %')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Trump Poll Distribution')
axes[1, 1].legend()

trump_error_all  = bg_polls['pct_trump_poll']  - bg_polls['p_trump_true']  * 100
harris_error_all = bg_polls['pct_harris_poll'] - bg_polls['p_harris_true'] * 100

axes[1, 2].hist(trump_error_all.dropna(),  bins=30, alpha=0.5, label=f'Trump (mean={trump_error_all.mean():.2f})',  color='red',  edgecolor='black')
axes[1, 2].hist(harris_error_all.dropna(), bins=30, alpha=0.5, label=f'Harris (mean={harris_error_all.mean():.2f})', color='blue', edgecolor='black')
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=2)
axes[1, 2].set_xlabel('Simple Error (Poll - True %)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Simple Errors')
axes[1, 2].legend()

plt.suptitle('All Battleground States Combined (AZ, GA, MI, NV, NC, PA, WI)', fontsize=16)
plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundcombined_methoda_errors_no_explode.png", dpi=300)


### seven-panel plots per state (trump component, harris component, method a, polls, simple errors)
for component, col, suptitle in [
    ('trump_part_A',  'Trump Component',  'Trump Component by State: ln(poll_trump) - ln(true_trump)'),
    ('harris_part_A', 'Harris Component', 'Harris Component by State: ln(poll_harris) - ln(true_harris)'),
    ('A',             'Method A',         'Method A by State: trump_part - harris_part'),
]:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, state in enumerate(battleground_states):
        state_data = bg_polls[bg_polls['state'] == state]
        axes[i].hist(state_data[component].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[i].axvline(0, color='black', linestyle='-', linewidth=2)
        axes[i].axvline(state_data[component].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {state_data[component].mean():.4f}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{state.title()} (N={len(state_data)})')
        axes[i].legend()
    axes[7].axis('off')
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    fname = col.lower().replace(' ', '')
    plt.savefig(f"figures/fiftyplusonePY/datelimited/fiftyplusone_datelimited_battlegroundsplit_{fname}_methoda_errors_no_explode.png", dpi=300)

### battleground summary stats
print(f"\nAll Battleground States Combined (N={len(bg_polls)}):")
print(f"  Trump Component Mean:  {bg_polls['trump_part_A'].mean():.4f}")
print(f"  Harris Component Mean: {bg_polls['harris_part_A'].mean():.4f}")
print(f"  Method A Mean:         {bg_polls['A'].mean():.4f}")
print(f"  Trump Error Mean:      {trump_error_all.mean():.2f}")
print(f"  Harris Error Mean:     {harris_error_all.mean():.2f}")

print(f"\nBy State:")
for state in battleground_states:
    state_data   = bg_polls[bg_polls['state'] == state]
    true_trump   = state_data['p_trump_true'].iloc[0]  * 100
    true_harris  = state_data['p_harris_true'].iloc[0] * 100
    poll_trump   = state_data['pct_trump_poll'].mean()
    poll_harris  = state_data['pct_harris_poll'].mean()
    trump_error  = poll_trump  - true_trump
    harris_error = poll_harris - true_harris
    print(f"\n{state} (N={len(state_data)}):")
    print(f"  Trump  True: {true_trump:.2f}%,  Poll Mean: {poll_trump:.2f}%,  Error: {trump_error:.2f}")
    print(f"  Harris True: {true_harris:.2f}%, Poll Mean: {poll_harris:.2f}%, Error: {harris_error:.2f}")
    print(f"  Trump Component Mean:  {state_data['trump_part_A'].mean():.4f}")
    print(f"  Harris Component Mean: {state_data['harris_part_A'].mean():.4f}")
    print(f"  Method A Mean:         {state_data['A'].mean():.4f}")


######## accuracy split by state vs national
print(f"\nMethod A accuracy by poll level (state vs national):")
results = []
for level in harris_trump_pivot['poll_level'].unique():
    subdf  = harris_trump_pivot[harris_trump_pivot['poll_level'] == level]
    mean_A = subdf['A'].mean()
    se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')
    if pd.notna(se_robust) and se_robust > 0:
        t_stat  = mean_A / se_robust
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls - 1))
    else:
        p_value = np.nan
    results.append({'poll_level': level, 'mean': mean_A, 'median': subdf['A'].median(),
                    'se': se_robust, 'p_value': p_value, 'sd': subdf['A'].std(), 'n': len(subdf)})

results_df = pd.DataFrame(results)
results_df['sig'] = results_df['p_value'].apply(sig_stars)
print(results_df.to_string(index=False))


########################################################################################
################################# SIMPLE MODE ANALYSIS PREP ############################
########################################################################################

# classify modes as self-administered vs interviewer-administered
self_admin_modes     = ['Online Opt-In Panel', 'Text-to-Web', 'Online Matched Sample',
                        'Online Ad', 'Probability Panel', 'Text', 'App Panel', 'Email', 'Mail-to-Web']
interviewer_modes    = ['Live Phone', 'IVR']
explicit_mixed_modes = ['Mail-to-Phone']

harris_trump_pivot['has_self_admin']    = harris_trump_pivot['mode'].apply(lambda x: any(m.strip() in self_admin_modes     for m in str(x).split('/')))
harris_trump_pivot['has_interviewer']   = harris_trump_pivot['mode'].apply(lambda x: any(m.strip() in interviewer_modes    for m in str(x).split('/')))
harris_trump_pivot['has_explicit_mixed'] = harris_trump_pivot['mode'].apply(lambda x: any(m.strip() in explicit_mixed_modes for m in str(x).split('/')))
harris_trump_pivot['has_other']         = ~(harris_trump_pivot['has_self_admin'] | harris_trump_pivot['has_interviewer'] | harris_trump_pivot['has_explicit_mixed'])

other_examples = harris_trump_pivot[harris_trump_pivot['has_other']]['mode'].value_counts()
print(f"\nOther category (N = {len(harris_trump_pivot[harris_trump_pivot['has_other']])} questions):")
for mode, count in other_examples.items():
    print(f"    '{mode}': {count} questions")

harris_trump_simple_mode_analysis = harris_trump_pivot[~harris_trump_pivot['has_other']].copy()

harris_trump_simple_mode_analysis['interviewer_only'] = (
    harris_trump_simple_mode_analysis['has_interviewer'] &
    ~harris_trump_simple_mode_analysis['has_self_admin'] &
    ~harris_trump_simple_mode_analysis['has_explicit_mixed']
).astype(int)

harris_trump_simple_mode_analysis['self_admin_only'] = (
    harris_trump_simple_mode_analysis['has_self_admin'] &
    ~harris_trump_simple_mode_analysis['has_interviewer'] &
    ~harris_trump_simple_mode_analysis['has_explicit_mixed']
).astype(int)

harris_trump_simple_mode_analysis['mixed_mode'] = (
    (harris_trump_simple_mode_analysis['has_self_admin'] & harris_trump_simple_mode_analysis['has_interviewer']) |
    harris_trump_simple_mode_analysis['has_explicit_mixed']
).astype(int)

mode_sum = (harris_trump_simple_mode_analysis['interviewer_only'] +
            harris_trump_simple_mode_analysis['self_admin_only'] +
            harris_trump_simple_mode_analysis['mixed_mode'])
if not (mode_sum == 1).all():
    print("\nWARNING: Mode categories are not mutually exclusive!")

print("MODE ANALYSIS DATAFRAME")
print(f"\nTotal questions: {len(harris_trump_simple_mode_analysis)}")
print(f"Excluded from original: {len(harris_trump_pivot) - len(harris_trump_simple_mode_analysis)}")
print(f"  Interviewer-only: {harris_trump_simple_mode_analysis['interviewer_only'].sum()}")
print(f"  Self-admin-only:  {harris_trump_simple_mode_analysis['self_admin_only'].sum()}")
print(f"  Mixed mode:       {harris_trump_simple_mode_analysis['mixed_mode'].sum()}")

harris_trump_simple_mode_analysis_pure = harris_trump_simple_mode_analysis[harris_trump_simple_mode_analysis['mixed_mode'] == 0].copy()

print(f"\nPure mode subset (excludes mixed):")
print(f"  Total questions:  {len(harris_trump_simple_mode_analysis_pure)}")
print(f"  Self-admin-only:  {harris_trump_simple_mode_analysis_pure['self_admin_only'].sum()}")
print(f"  Interviewer-only: {harris_trump_simple_mode_analysis_pure['interviewer_only'].sum()}")

harris_trump_simple_mode_analysis.to_csv('data/harris_trump_datelimited_simple_mode_analysis_threeway_no_explode.csv', index=False)
harris_trump_simple_mode_analysis_pure.to_csv('data/harris_trump_datelimited_simple_mode_analysis_pure_no_explode.csv', index=False)


########################################################################################
##################################### MANY Mode Analysis ####################################
########################################################################################

# descriptive mode-level accuracy table using indicators (no explosion)
# for each mode indicator column, subset to polls where that mode is active (indicator == 1)
# a multi-mode poll contributes to every mode it uses, which mirrors the explode approach
# but without duplicating rows — counts and means reflect the question-level dataset

# build indicator columns directly on harris_trump_pivot
# (mode indicators are created later for reg_df_original; replicate the same logic here
#  so the descriptive table is available before the regression setup block)
_mode_indicator_df = harris_trump_pivot.copy()
_all_modes_desc = set()
for mode_str in _mode_indicator_df['mode'].dropna():
    _all_modes_desc.update(m.strip() for m in mode_str.split('/'))

for _mode in sorted(_all_modes_desc):
    _var = f'mode_{_mode.replace(" ", "_").replace("-", "_")}'
    _mode_indicator_df[_var] = _mode_indicator_df['mode'].str.contains(
        _mode, case=False, na=False, regex=False
    ).astype(int)

# mode_Text_only recode to match regression spec
_mode_indicator_df['mode_Text_only'] = (
    (_mode_indicator_df['mode_Text'] == 1) &
    (_mode_indicator_df['mode_Text_to_Web'] == 0)
).astype(int)
_mode_indicator_df = _mode_indicator_df.drop(columns=['mode_Text'])

# collect all indicator column names (include Live Phone for descriptive purposes)
_desc_mode_vars = sorted([c for c in _mode_indicator_df.columns if c.startswith('mode_')])

# mode counts: number of questions with each mode active
print(f"\nBase mode breakdown (unique questions per mode indicator):\n")
mode_counts_rows = []
for var in _desc_mode_vars:
    mode_name = var.replace('mode_', '').replace('_', ' ')
    n = int(_mode_indicator_df[var].sum())
    mode_counts_rows.append({'mode': mode_name, 'unique_questions': n})
mode_counts_df = pd.DataFrame(mode_counts_rows).sort_values('unique_questions', ascending=False).reset_index(drop=True)
print(mode_counts_df.to_string(index=False))

# accuracy by mode indicator
print(f"\nMethod A accuracy by mode indicator")
results = []
for var in _desc_mode_vars:
    mode_name = var.replace('mode_', '').replace('_', ' ')
    subdf  = _mode_indicator_df[_mode_indicator_df[var] == 1]
    if len(subdf) == 0:
        continue
    mean_A = subdf['A'].mean()
    se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')
    if pd.notna(se_robust) and se_robust > 0:
        t_stat  = mean_A / se_robust
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls - 1))
    else:
        p_value = np.nan
    results.append({'base_mode': mode_name, 'mean': mean_A, 'median': subdf['A'].median(),
                    'sd': subdf['A'].std(), 'se': se_robust, 'p_value': p_value, 'n': len(subdf)})

results_df = pd.DataFrame(results).sort_values('mean', ascending=False)
results_df['sig'] = results_df['p_value'].apply(sig_stars)
print(results_df.to_string(index=False))
print("\nnote: a poll with multiple modes contributes to each mode's row")
print("      mode_Text_only = Text recruitment without Text-to-Web; mode_Text_to_Web = Text-to-Web response")


########################################################################################
########################## Accuracy by mode and target population ######################
########################################################################################

def print_accuracy_table(df, group_col, label):
    results = []
    for (group_val, level), subdf in df.groupby([group_col, 'poll_level']):
        mean_A = subdf['A'].mean()
        se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')
        if pd.notna(se_robust) and se_robust > 0:
            t_stat  = mean_A / se_robust
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_polls - 1))
        else:
            p_value = np.nan
        results.append({group_col: group_val, 'poll_level': level, 'mean': mean_A,
                        'se': se_robust, 'p_value': p_value, 'median': subdf['A'].median(),
                        'std': subdf['A'].std(), 'n': len(subdf)})

    results_df = pd.DataFrame(results)
    tbl_wide   = results_df.pivot(index=group_col, columns='poll_level',
                                  values=['mean', 'se', 'p_value', 'median', 'std', 'n'])
    tbl_wide.columns = [f"{stat}_{level}" for stat, level in tbl_wide.columns]
    tbl_wide = tbl_wide.reset_index().sort_values('mean_state', ascending=False, na_position='last')

    print(f"\n{'='*110}")
    print(f"  method a accuracy by {label}")
    print(f"  (+ = republican bias, - = democratic bias)")
    print(f"{'='*110}")
    print(f"  {'':30} {'-------------- state --------------':>42} {'------------ national ------------':>42}")
    print(f"  {group_col:<30} {'mean':>8} {'se':>8} {'p-val':>8} {'median':>8} {'std':>8} {'n':>5}   "
          f"{'mean':>8} {'se':>8} {'p-val':>8} {'median':>8} {'std':>8} {'n':>5}")
    print(f"  {'-'*108}")

    for _, row in tbl_wide.iterrows():
        def fmt_val(val):  return f"{val:.4f}" if pd.notna(val) else '   --'
        def fmt_pval(val): return f"{val:.3f}{sig_stars(val)}" if pd.notna(val) else '   --'
        def fmt_n(val):    return f"{int(val)}" if pd.notna(val) and val > 0 else '--'

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

print_accuracy_table(harris_trump_pivot,  'population', 'target population')

# accuracy by mode indicator - split by state/national
mode_long_rows = []
for var in _desc_mode_vars:
    mode_name = var.replace('mode_', '').replace('_', ' ')
    subdf = _mode_indicator_df[_mode_indicator_df[var] == 1].copy()
    subdf['base_mode'] = mode_name
    mode_long_rows.append(subdf)

mode_long_df = pd.concat(mode_long_rows, ignore_index=True)

print_accuracy_table(mode_long_df, 'base_mode', 'polling mode')
print("\nnote: a poll with multiple modes contributes to each mode's row")
print("      mode_Text_only = Text recruitment without Text-to-Web; mode_Text_to_Web = Text-to-Web response")

########################################################################################
###################### Multivariate Regression Analysis Set Up #########################
########################################################################################

# unit of analysis: one row per question (question_id)
# dependent variable: method a (continuous, centered at 0)
# two separate regressions: state level polls and national polls
# standard errors: clustered by poll_id
# mode strategy: indicator (no explosion) — active modes filtered per sample/window

def run_ols_clustered(df, y_col, x_cols, cluster_col, label, min_obs_threshold=10):
    """
    fits ols on df using x_cols to predict y_col.
    mode indicator variables (starting with 'mode_') are automatically filtered
    to only those with at least one observation in df, so zero-observation modes
    are never included regardless of the global x_cols list.
    standard errors are clustered on cluster_col (huber-white sandwich).
    """
    # separate mode vars from non-mode vars; filter modes to active ones in this sample
    non_mode_x    = [v for v in x_cols if not v.startswith('mode_')]
    all_mode_x    = [v for v in x_cols if v.startswith('mode_')]
    active_mode_x = [v for v in all_mode_x if v in df.columns and df[v].sum() > 0]
    dropped_modes = [v for v in all_mode_x if v not in active_mode_x]

    if dropped_modes:
        print(f"  [{label}] dropping zero-observation mode indicators: {dropped_modes}")

    x_cols_active = non_mode_x + active_mode_x

    df_reg = df[x_cols_active + [y_col, cluster_col]].dropna()

    # check for low-variance variables
    low_variance_vars = []
    for col in x_cols_active:
        if col in df_reg.columns and df_reg[col].nunique() == 2:
            if df_reg[col].value_counts().min() < min_obs_threshold:
                low_variance_vars.append(col)

    X      = sm.add_constant(df_reg[x_cols_active], has_constant='add')
    y      = df_reg[y_col]
    groups = df_reg[cluster_col]

    model  = sm.OLS(y, X)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})

    def stars_local(p):
        if p < 0.01:   return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else:          return ''

    print(f"\n{'='*70}")
    print(f"  ols regression: {label}")
    print(f"  dependent variable: method a  (+ = republican bias)")
    print(f"  standard errors: clustered by {cluster_col}")
    if dropped_modes:
        print(f"  dropped zero-obs modes: {dropped_modes}")
    print(f"{'='*70}")
    print(f"  {'variable':<35} {'coef':>10} {'se':>10} {'sig':>6}")
    print(f"  {'-'*63}")

    params  = result.params
    bse     = result.bse
    pvalues = result.pvalues

    intercept_name = next((v for v in params.index if v.lower() in ('const', 'intercept')), None)
    var_order = [v for v in params.index if v != intercept_name] + ([intercept_name] if intercept_name else [])

    for var in var_order:
        if var in low_variance_vars:
            print(f"  {var:<35} {'----':>10} {'----':>10} {'':>6}")
        else:
            print(f"  {var:<35} {params[var]:>10.4f} {bse[var]:>10.4f} {stars_local(pvalues[var]):>6}")

    print(f"  {'-'*63}")
    print(f"  adjusted r2:  {result.rsquared_adj:.4f}")
    print(f"  n:            {int(result.nobs)}")
    if low_variance_vars:
        print(f"  note:         ---- indicates <{min_obs_threshold} observations in category")
    print(f"{'='*70}\n")

    result.low_variance_vars = low_variance_vars
    return result


# build regression dataset
reg_df = harris_trump_pivot.copy()

# VAR: duration in field
reg_df['duration_days'] = (reg_df['end_date'] - reg_df['start_date']).dt.days + 1

# VAR: days before election
reg_df['days_before_election'] = (election_date - reg_df['end_date']).dt.days

print(f"\ntime variable diagnostics:")
print(f"  duration_days        -- mean: {reg_df['duration_days'].mean():.1f}, "
      f"min: {reg_df['duration_days'].min()}, max: {reg_df['duration_days'].max()}")
print(f"  days_before_election -- mean: {reg_df['days_before_election'].mean():.1f}, "
      f"min: {reg_df['days_before_election'].min()}, max: {reg_df['days_before_election'].max()}")

# VAR: statewide turnout
turnout_data  = pd.read_csv("data/Turnout_2024G_v0.3.csv")
turnout_clean = turnout_data[['STATE', 'VEP_TURNOUT_RATE']].copy()
turnout_clean['state']       = turnout_clean['STATE'].str.strip().str.lower()
turnout_clean['turnout_pct'] = turnout_clean['VEP_TURNOUT_RATE'].str.rstrip('%').astype(float)
turnout_clean['state']       = turnout_clean['state'].replace('united states', 'national')

reg_df = reg_df.merge(turnout_clean[['state', 'turnout_pct']], on='state', how='left')

# single dataset — no explosion needed
reg_df_original = reg_df.copy()


########################################################################################
#################### CREATE MODE INDICATORS (NO EXPLOSION) #############################
########################################################################################

print("\n" + "="*110)
print("CREATING MODE INDICATORS WITHOUT EXPLODING")
print("="*110)

reference_mode     = 'Live Phone'
reference_mode_var = 'mode_Live_Phone'

# get all unique mode types from slash-separated strings
all_modes = set()
for mode_str in reg_df_original['mode'].dropna():
    all_modes.update(m.strip() for m in mode_str.split('/'))

print(f"\nUnique mode types found: {sorted(all_modes)}")

# create a binary indicator for each mode type
for mode in sorted(all_modes):
    var_name = f'mode_{mode.replace(" ", "_").replace("-", "_")}'
    reg_df_original[var_name] = reg_df_original['mode'].str.contains(mode, case=False, na=False, regex=False).astype(int)
    n   = reg_df_original[var_name].sum()
    pct = 100 * n / len(reg_df_original)
    print(f"  {var_name}: {n} questions ({pct:.1f}%)")

# ── Text / Text_to_Web recode ────────────────────────────────────────────────────────
# Text_to_Web is nested inside Text (every Text_to_Web poll also has Text=1).
# Including both causes near-perfect collinearity.
# Fix: replace raw mode_Text with mode_Text_only (Text=1 AND Text_to_Web=0).
# mode_Text_to_Web remains unchanged.

reg_df_original['mode_Text_only'] = (
    (reg_df_original['mode_Text'] == 1) &
    (reg_df_original['mode_Text_to_Web'] == 0)
).astype(int)

reg_df_original = reg_df_original.drop(columns=['mode_Text'])

print(f"\nText recode:")
print(f"  mode_Text_only    (Text=1, Text_to_Web=0): {reg_df_original['mode_Text_only'].sum()}")
print(f"  mode_Text_to_Web  (Text_to_Web=1):         {reg_df_original['mode_Text_to_Web'].sum()}")

# drop reference category (Live Phone) from regression vars
all_mode_vars = sorted([c for c in reg_df_original.columns if c.startswith('mode_')])
mode_vars     = [v for v in all_mode_vars if v != reference_mode_var]

print(f"\nReference category: {reference_mode} ({reference_mode_var} excluded)")
print(f"Mode indicators for regression: {mode_vars}")

# ── Population dummies ───────────────────────────────────────────────────────────────
reg_df_original['population'] = reg_df_original['population'].str.lower().str.strip()

# drop 'v' population
n_before = len(reg_df_original)
reg_df_original = reg_df_original[reg_df_original['population'] != 'v'].copy()
print(f"\nDropped {n_before - len(reg_df_original)} rows with population='v'")
print(reg_df_original['population'].value_counts())

pop_dummies = pd.get_dummies(reg_df_original['population'], prefix='pop', drop_first=False)
pop_dummies = pop_dummies.drop('pop_lv', axis=1)   # reference: lv
pop_dummies = pop_dummies.astype(int)
reg_df_original = pd.concat([reg_df_original, pop_dummies], axis=1)

pop_vars = [col for col in reg_df_original.columns if col.startswith('pop_')]
print(f"\nPopulation reference category: lv")
print(f"Population dummy variables: {pop_vars}")


########################################################################################
#################### DEFINE VARIABLE LISTS #############################################
########################################################################################

time_vars     = ['duration_days', 'days_before_election', 'partisan_flag']
state_vars    = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk']

# without mode
state_x_vars_no_mode    = time_vars + state_vars + pop_vars
national_x_vars_no_mode = time_vars + national_vars + pop_vars

# with mode (full list passed; run_ols_clustered filters active ones per call)
state_x_vars_with_mode    = time_vars + state_vars + mode_vars + pop_vars
national_x_vars_with_mode = time_vars + national_vars + mode_vars + pop_vars

# define swing states
swing_states = ['arizona', 'georgia', 'michigan', 'nevada',
                'north carolina', 'pennsylvania', 'wisconsin']

# split datasets (single set — same rows for both mode and no-mode regressions)
reg_state_original       = reg_df_original[reg_df_original['poll_level'] == 'state'].copy()
reg_national_original    = reg_df_original[reg_df_original['poll_level'] == 'national'].copy()
reg_state_swing_original = reg_state_original[reg_state_original['state'].isin(swing_states)].copy()

print(f"\nregression sample sizes:")
print(f"  national-level questions: {len(reg_national_original)}")
print(f"  state-level questions:    {len(reg_state_original)}")
print(f"  swing state questions:    {len(reg_state_swing_original)}")

print(f"\noriginal swing states breakdown:")
print(reg_state_swing_original['state'].value_counts().sort_index())

print("\nOriginal Mode Strings (Top 15):")
print(f"{'Mode':<50} {'N':>10} {'%':>10}")
print("-" * 70)
mode_original = reg_df_original['mode'].value_counts()
for mode, count in mode_original.head(15).items():
    pct = 100 * count / len(reg_df_original)
    print(f"{mode:<50} {count:>10} {pct:>9.1f}%")
print(f"\nTotal unique mode combinations: {reg_df_original['mode'].nunique()}")


########################################################################################
######## BASE REGRESSIONS (NO TIME WINDOWS, NO MODE) ###################################
########################################################################################

results_national = run_ols_clustered(
    df=reg_national_original, y_col='A', x_cols=national_x_vars_no_mode,
    cluster_col='poll_id', label='national polls'
)

results_state = run_ols_clustered(
    df=reg_state_original, y_col='A', x_cols=state_x_vars_no_mode,
    cluster_col='poll_id', label='state-level polls'
)

results_swing = run_ols_clustered(
    df=reg_state_swing_original, y_col='A', x_cols=state_x_vars_no_mode,
    cluster_col='poll_id', label='swing states'
)


########################################################################################
#################### BASE REGRESSIONS (NO TIME, WITH MODE) #############################
########################################################################################

print("REGRESSIONS WITH MODE CONTROLS")

results_national_mode = run_ols_clustered(
    df=reg_national_original, y_col='A', x_cols=national_x_vars_with_mode,
    cluster_col='poll_id', label='national polls with mode controls'
)

results_all_states_mode = run_ols_clustered(
    df=reg_state_original, y_col='A', x_cols=state_x_vars_with_mode,
    cluster_col='poll_id', label='all state polls with mode controls'
)

results_swing_mode = run_ols_clustered(
    df=reg_state_swing_original, y_col='A', x_cols=state_x_vars_with_mode,
    cluster_col='poll_id', label='swing states with mode controls'
)


########################################################################################
#################### TIME WINDOW REGRESSIONS (NO MODE) #################################
########################################################################################

time_windows = [107, 90, 60, 30, 7]

swing_window_results_no_mode      = {}
all_states_window_results_no_mode = {}
national_window_results_no_mode   = {}

for window in time_windows:
    state_w    = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window].copy()
    state_all_w = reg_state_original[reg_state_original['days_before_election'] <= window].copy()
    national_w = reg_national_original[reg_national_original['days_before_election'] <= window].copy()

    print(f"\n  window: {window} days before election")
    print(f"  swing state questions:  {len(state_w)}")
    print(f"  all state questions:    {len(state_all_w)}")
    print(f"  national questions:     {len(national_w)}")

    if len(state_w[state_x_vars_no_mode + ['A', 'poll_id']].dropna()) < 10:
        print(f"  swing skipped")
        swing_window_results_no_mode[window] = None
    else:
        swing_window_results_no_mode[window] = run_ols_clustered(
            df=state_w, y_col='A', x_cols=state_x_vars_no_mode,
            cluster_col='poll_id', label=f'swing states {window}d (no mode)'
        )

    if len(state_all_w[state_x_vars_no_mode + ['A', 'poll_id']].dropna()) < 10:
        print(f"  all states skipped")
        all_states_window_results_no_mode[window] = None
    else:
        all_states_window_results_no_mode[window] = run_ols_clustered(
            df=state_all_w, y_col='A', x_cols=state_x_vars_no_mode,
            cluster_col='poll_id', label=f'all states {window}d (no mode)'
        )

    if len(national_w[national_x_vars_no_mode + ['A', 'poll_id']].dropna()) < 10:
        print(f"  national skipped")
        national_window_results_no_mode[window] = None
    else:
        national_window_results_no_mode[window] = run_ols_clustered(
            df=national_w, y_col='A', x_cols=national_x_vars_no_mode,
            cluster_col='poll_id', label=f'national {window}d (no mode)'
        )


########################################################################################
#################### TIME WINDOW REGRESSIONS (WITH MODE) ###############################
########################################################################################

swing_window_results_with_mode      = {}
all_states_window_results_with_mode = {}
national_window_results_with_mode   = {}

for window in time_windows:
    state_w     = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window].copy()
    state_all_w = reg_state_original[reg_state_original['days_before_election'] <= window].copy()
    national_w  = reg_national_original[reg_national_original['days_before_election'] <= window].copy()

    print(f"\n  window: {window} days before election")

    if len(state_w[state_x_vars_with_mode + ['A', 'poll_id']].dropna()) < 10:
        print(f"  swing skipped")
        swing_window_results_with_mode[window] = None
    else:
        swing_window_results_with_mode[window] = run_ols_clustered(
            df=state_w, y_col='A', x_cols=state_x_vars_with_mode,
            cluster_col='poll_id', label=f'swing states {window}d (with mode)'
        )

    if len(state_all_w[state_x_vars_with_mode + ['A', 'poll_id']].dropna()) < 10:
        print(f"  all states skipped")
        all_states_window_results_with_mode[window] = None
    else:
        all_states_window_results_with_mode[window] = run_ols_clustered(
            df=state_all_w, y_col='A', x_cols=state_x_vars_with_mode,
            cluster_col='poll_id', label=f'all states {window}d (with mode)'
        )

    if len(national_w[national_x_vars_with_mode + ['A', 'poll_id']].dropna()) < 10:
        print(f"  national skipped")
        national_window_results_with_mode[window] = None
    else:
        national_window_results_with_mode[window] = run_ols_clustered(
            df=national_w, y_col='A', x_cols=national_x_vars_with_mode,
            cluster_col='poll_id', label=f'national {window}d (with mode)'
        )


########################################################################################
### SUMMARY TABLES
########################################################################################

def stars(p):
    if p < 0.01:   return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    else:          return ''

def print_time_window_table(window_results_dict, vars_ordered, title, note=''):
    print("\n" + "="*110)
    print(title)
    if note:
        print(note)
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

    for var in vars_ordered:
        print(f"{var:<30}", end='')
        for window in time_windows:
            result = window_results_dict[window]
            if result is not None and var in result.params:
                coef = result.params[var]
                se   = result.bse[var]
                pval = result.pvalues[var]
                print(f"{coef:>10.3f} {se:>10.3f} {stars(pval):>5}", end='')
            else:
                print(f"{'':>10} {'':>10} {'':>5}", end='')
        print()

    print()
    print(f"{'Constant':<30}", end='')
    for window in time_windows:
        result = window_results_dict[window]
        if result is not None:
            intercept_name = next((v for v in result.params.index if v.lower() in ('const', 'intercept')), None)
            if intercept_name:
                print(f"{result.params[intercept_name]:>10.3f} {result.bse[intercept_name]:>10.3f} {'':>5}", end='')
            else:
                print(f"{'':>10} {'':>10} {'':>5}", end='')
        else:
            print(f"{'':>10} {'':>10} {'':>5}", end='')
    print()

    print()
    print(f"{'Adjusted R Square':<30}", end='')
    for window in time_windows:
        result = window_results_dict[window]
        if result is not None:
            print(f"{result.rsquared_adj:>10.2f} {'':>15}", end='')
        else:
            print(f"{'':>25}", end='')
    print()

    print(f"{'N':<30}", end='')
    for window in time_windows:
        result = window_results_dict[window]
        if result is not None:
            print(f"{int(result.nobs):>10.0f} {'':>15}", end='')
        else:
            print(f"{'':>25}", end='')
    print()

    print("\nNote: Robust Standard Errors Reported")
    print("Sig: *<.10; **<.05; ***<.01")
    if note:
        print(note)


# ordered variable lists for summary tables
national_vars_ordered   = ['duration_days', 'days_before_election', 'partisan_flag', 'pct_dk'] + pop_vars
state_vars_ordered      = ['duration_days', 'days_before_election', 'partisan_flag', 'pct_dk', 'abs_margin', 'turnout_pct'] + pop_vars
national_vars_with_mode = national_vars_ordered + sorted(mode_vars)
state_vars_with_mode    = state_vars_ordered    + sorted(mode_vars)

print_time_window_table(national_window_results_no_mode,   national_vars_ordered,   "NATIONAL, ACROSS TIME WINDOWS, NO MODE")
print_time_window_table(national_window_results_with_mode, national_vars_with_mode, "NATIONAL, ACROSS TIME WINDOWS, WITH MODE", f"reference mode: {reference_mode}")
print_time_window_table(all_states_window_results_no_mode,   state_vars_ordered,   "ALL STATES, ACROSS TIME WINDOWS, NO MODE")
print_time_window_table(all_states_window_results_with_mode, state_vars_with_mode, "ALL STATES, ACROSS TIME WINDOWS, WITH MODE", f"reference mode: {reference_mode}")
print_time_window_table(swing_window_results_no_mode,   state_vars_ordered,   "SWING, ACROSS TIME WINDOWS, NO MODE")
print_time_window_table(swing_window_results_with_mode, state_vars_with_mode, "SWING, ACROSS TIME WINDOWS, WITH MODE", f"reference mode: {reference_mode}")


########################################################################################
###### MODE COEFFICIENTS COMPARISON ACROSS SAMPLES ######
########################################################################################

print("MODE COEFFICIENTS ACROSS SAMPLES")
print(f"\n{'Mode':<30} {'National':>15} {'All States':>15} {'Swing':>15}")
print("." * 75)

for mode_var in sorted(mode_vars):
    mode_name = mode_var.replace('mode_', '')
    print(f"{mode_name:<30}", end='')
    for result, label in [
        (results_national_mode,   'national'),
        (results_all_states_mode, 'all states'),
        (results_swing_mode,      'swing'),
    ]:
        if result is not None and mode_var in result.params:
            coef = result.params[mode_var]
            pval = result.pvalues[mode_var]
            print(f"{coef:>12.2f}{stars(pval):<3}", end='')
        else:
            print(f"{'--':>15}", end='')
    print()

print("\nnote: *** p<0.01, ** p<0.05, * p<0.10")
print(f"all coefficients relative to {reference_mode} (reference category)")
print("mode_Text_only = Text recruitment without Text-to-Web response")
print("mode_Text_to_Web = Text recruitment with web survey link (always involves Text)")


########################################################################################
#################### SAMPLE SIZES ACROSS TIME WINDOWS ##################################
########################################################################################

print("\n" + "="*110)
print("SAMPLE SIZE CHANGES ACROSS TIME WINDOWS")
print("="*110)

sample_tracking = []
for window in time_windows:
    sample_tracking.append({
        'window':     window,
        'swing':      len(reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window]),
        'all_states': len(reg_state_original[reg_state_original['days_before_election'] <= window]),
        'national':   len(reg_national_original[reg_national_original['days_before_election'] <= window]),
    })

sample_df = pd.DataFrame(sample_tracking)

print("\nCumulative Sample Sizes (polls within X days of election):")
print(f"{'Window (days)':<15} {'Swing States':>15} {'All States':>15} {'National':>15}")
print("-" * 60)
for _, row in sample_df.iterrows():
    print(f"{int(row['window']):<15} {int(row['swing']):>15} {int(row['all_states']):>15} {int(row['national']):>15}")

print("\n\nPolls Excluded When Moving to Narrower Window:")
print(f"{'Window Change':<20} {'Swing States':>20} {'All States':>20} {'National':>20}")
print("-" * 80)
for i in range(len(sample_df) - 1):
    from_w = sample_df.iloc[i]['window']
    to_w   = sample_df.iloc[i + 1]['window']
    for col in ['swing', 'all_states', 'national']:
        drop = sample_df.iloc[i][col] - sample_df.iloc[i + 1][col]
        pct  = 100 * drop / sample_df.iloc[i][col] if sample_df.iloc[i][col] > 0 else 0
    swing_drop = sample_df.iloc[i]['swing']      - sample_df.iloc[i + 1]['swing']
    all_drop   = sample_df.iloc[i]['all_states'] - sample_df.iloc[i + 1]['all_states']
    nat_drop   = sample_df.iloc[i]['national']   - sample_df.iloc[i + 1]['national']
    swing_pct  = 100 * swing_drop / sample_df.iloc[i]['swing']      if sample_df.iloc[i]['swing']      > 0 else 0
    all_pct    = 100 * all_drop   / sample_df.iloc[i]['all_states'] if sample_df.iloc[i]['all_states'] > 0 else 0
    nat_pct    = 100 * nat_drop   / sample_df.iloc[i]['national']   if sample_df.iloc[i]['national']   > 0 else 0
    print(f"{int(from_w)} to {int(to_w):<15} {swing_drop:>6} ({swing_pct:>5.1f}%){' ':>8} "
          f"{all_drop:>6} ({all_pct:>5.1f}%){' ':>8} "
          f"{nat_drop:>6} ({nat_pct:>5.1f}%)")

print("="*110 + "\n")


######## save outputs
harris_trump_pivot.to_csv('data/harris_trump_datelimted_accuracy_no_explode.csv', index=False)
reg_df_original.to_csv('data/harris_trump_mode_indicators_with_partisan_and_lv_regression_no_explode.csv', index=False)

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("Accuracy analysis complete — see output/fiftyplusone_analysis_datelimited_log_no_exploded.txt")