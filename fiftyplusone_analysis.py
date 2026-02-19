import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats

# redirect all print output to a log file
log_file = open('output/fiftyplusone_analysis_log.txt', 'w')
sys.stdout = log_file

# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy

# load cleaned harris+trump questions dataset (output from fiftyplusone_initial_analysis.py)
harris_trump_full_df = pd.read_csv("data/fiftyplusone_cleaned_harris_trump_questions.csv")

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
# the log-odds ratio form is symmetric and scale-invariant, making it more appropriate than simple margin error for comparing polls across states with different competitive landscapes

harris_trump_pivot['A'] = np.log(
    (harris_trump_pivot['pct_trump_poll']  / harris_trump_pivot['pct_harris_poll']) /
    (harris_trump_pivot['p_trump_true'] / harris_trump_pivot['p_harris_true'])
)

# flag poll level (state vs national) used to split regressions
harris_trump_pivot['poll_level'] = np.where(
    harris_trump_pivot['state'] == 'national', 'national', 'state'
)

# diagnostic: confirm how many questions were flagged as national
print(f"\npoll_level counts after flagging:")
print(harris_trump_pivot['poll_level'].value_counts(dropna=False).to_string())
print(f"\nstate values flagged as national:")
print(harris_trump_pivot[harris_trump_pivot['poll_level'] == 'national']['state'].value_counts(dropna=False).to_string())

# flag period (before/after biden dropout) used for descriptive splits
harris_trump_pivot['period'] = np.where(
    harris_trump_pivot['end_date'] < dropout_cutoff, 'before_dropout', 'after_dropout'
)

########################################################################################
############################# General Accuracy Analysis ################################
########################################################################################
# include SE
def compute_clustered_se(df, value_col, cluster_col):
    """
    compute cluster-robust standard error of the mean, accounts for multiple questions per poll
    """
    df_clean = df[[value_col, cluster_col]].dropna()
    cluster_means = df_clean.groupby(cluster_col)[value_col].mean()
    n_clusters = len(cluster_means)
    grand_mean = df_clean[value_col].mean()
    cluster_var = ((cluster_means - grand_mean) ** 2).sum() / (n_clusters - 1)
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

print(f"\nOverall Method A accuracy (all Harris+Trump questions):")
print(f"  Mean:   {mean_A:.4f}")
print(f"  SE:       {se_A_robust:.4f}")
print(f"  p-value:  {p_value:.4f} {sig_stars(p_value)}")
print(f"  Median: {harris_trump_pivot['A'].median():.4f}")
print(f"  SD:    {harris_trump_pivot['A'].std():.4f}")
print(f"  N:        {len(harris_trump_pivot)}")

######## accuracy split before/after dropout
accuracy_by_period = (
    harris_trump_pivot.groupby('period')['A']
    .agg(mean='mean', median='median', std='std', n='count')
    .reset_index()
)

print(f"\nMethod A accuracy by period:\n")
print(accuracy_by_period.to_string(index=False))

######## accuracy split by state vs national
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
# or nationally captures competitiveness polling dynamics than lopsided ones, and pollsters may expend more effort in competitive states


# VAR: statewide turnout
# TODO: load a dataset with registered voters per state
# merge on state, compute: turnout_pct = total_votes_cast / registered_voters.
# then merge turnout_pct into reg_df on 'state'.
# note: this variable only makes sense for state-level regression; national
# polls should use the national turnout figure.
# reg_df = reg_df.merge(turnout_df[['state', 'turnout_pct']], on='state', how='left')


# covariates for both regressions
time_vars  = ['duration_days', 'days_before_election']
other_vars = ['pct_dk', 'abs_margin'] # once turnout is ready, add to other_vars

all_x_vars = time_vars + other_vars

# split into statelevel and national samples
reg_state    = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()

print(f"\nregression sample sizes:")
print(f"  state-level questions:    {len(reg_state)}")
print(f"  national-level questions: {len(reg_national)}")

# these polls likely skewing the results given how far from election they are, therefore rely on ones run below split by time to election
# print the top 10 polls by days_before_election to identify outliers
print(f"\ntop questions by days before election:")
top_early = (
    reg_df[['poll_id', 'pollster', 'state', 'start_date', 'end_date', 'days_before_election']]
    .sort_values('days_before_election', ascending=False)
    .head(20)
)
print(top_early.to_string(index=False))

# final covariate lists per regression
state_x_vars    = time_vars + other_vars
national_x_vars = time_vars + other_vars

# state regression: clustered ses by poll_id to account for the fact that multiple questions from the same poll share correlated errors
results_state = run_ols_clustered(
    df          = reg_state,
    y_col       = 'A',
    x_cols      = state_x_vars,
    cluster_col = 'poll_id',
    label       = 'state-level polls'
)

# national regression: also clustered by poll_id for the same reason, though with fewer polls clustering matters less
results_national = run_ols_clustered(
    df          = reg_national,
    y_col       = 'A',
    x_cols      = all_x_vars,
    cluster_col = 'poll_id',
    label       = 'national polls'
)

# later additions after build out
# restricting to competitive states versus all states
# split into time periods ( analysis using three different time frame, 90, 30, and 7 days before the elections)
# how to say "mode X is more biased controlling for days before election and competitiveness"


########################################################################################
########################## Accuracy by mode and target population ######################
########################################################################################

# these two tables report mean, median, std, and n for method a broken out by (1) polling mode and (2) target population, each split by state vs national
# mode and population are reported separately from the regression because they are categorical design choices better understood descriptively, and the multi-hot nature of mode makes regression coefficients hard to interpret cleanly (but i may go back to this later so i can say something like mode X is more biased controlling for days before election and competitiveness)

# table print function
def print_accuracy_table(df, group_col, label):
    """
    groups df by group_col and poll_level, computes mean/median/std/n of method a,
    and prints a formatted table with state and national columns side by side
    rows are sorted by state mean accuracy descending (national-only rows go last)
    """
    tbl = (
        df.groupby([group_col, 'poll_level'])['A']
        .agg(mean='mean', median='median', std='std', n='count')
        .reset_index()
    )

    # pivot so state and national appear as side-by-side column groups
    tbl_wide = tbl.pivot(index=group_col, columns='poll_level', values=['mean', 'median', 'std', 'n'])
    tbl_wide.columns = [f"{stat}_{level}" for stat, level in tbl_wide.columns]
    tbl_wide = tbl_wide.reset_index().sort_values('mean_state', ascending=False, na_position='last')

    print(f"\n{'='*85}")
    print(f"  method a accuracy by {label}")
    print(f"  (+ = republican bias, - = democratic bias)")
    print(f"{'='*85}")
    print(f"  {'':30} {'-------- state --------':>34} {'------- national -------':>34}")
    print(f"  {group_col:<30} {'mean':>8} {'median':>8} {'std':>8} {'n':>5}   "
          f"{'mean':>8} {'median':>8} {'std':>8} {'n':>5}")
    print(f"  {'-'*85}")

    for _, row in tbl_wide.iterrows():
        # fmt_val: format a continuous value (mean/median/std) to 4dp, or dash if missing
        def fmt_val(val):
            return f"{val:.4f}" if pd.notna(val) else '   --'

        # fmt_n: format n as integer, or dash if missing/zero
        # n=0 shouldn't appear (groupby only produces rows with data) but guard anyway
        def fmt_n(val):
            if pd.isna(val) or val == 0:
                return '   --'
            return f"{int(val)}"

        print(
            f"  {str(row[group_col]):<30} "
            f"{fmt_val(row.get('mean_state')):>8} "
            f"{fmt_val(row.get('median_state')):>8} "
            f"{fmt_val(row.get('std_state')):>8} "
            f"{fmt_n(row.get('n_state')):>5}   "
            f"{fmt_val(row.get('mean_national')):>8} "
            f"{fmt_val(row.get('median_national')):>8} "
            f"{fmt_val(row.get('std_national')):>8} "
            f"{fmt_n(row.get('n_national')):>5}"
        )

    print(f"{'='*85}\n")

# Table: accuracy by polling mode
print_accuracy_table(harris_trump_modes, 'base_mode', 'polling mode')


# Table: accuracy by target population
# shows whether polls targeting different populations are systematically more or less accurate, without controlling for other factors
print_accuracy_table(reg_df, 'population', 'target population')


########################################################################################
#################### Regression by time window (like in harrison 2009) #################
########################################################################################

# re-run the same regression separately for five time windows prior to the election, where each window includes only polls whose end_date falls within that many days of election day
# ex: the 30 day window includes polls ending 30 or fewer days before election day
# lets us see whether the predictors of accuracy change as we get closer to election day, loses significance in shorter windows as found by harrison and desart/holbrook
# windows chosen: 200, 107, 90, 60, 30, and 7 days.

time_windows = [200, 107, 90, 60, 30, 7]

def run_window_regressions(reg_state_df, reg_national_df, x_vars, window_days):
    # filter each sample to polls ending within window_days of election day
    state_w    = reg_state_df[reg_state_df['days_before_election'] <= window_days].copy()
    national_w = reg_national_df[reg_national_df['days_before_election'] <= window_days].copy()

    print(f"\n  window: {window_days} days before election")
    print(f"  state questions in window:    {len(state_w)}")
    print(f"  national questions in window: {len(national_w)}")

    # state regression
    # skip if fewer than 10 complete cases to avoid degenerate models
    state_complete = state_w[x_vars + ['A', 'poll_id']].dropna()
    if len(state_complete) < 10:
        print(f"  state regression skipped - only {len(state_complete)} complete cases")
        res_state = None
    else:
        res_state = run_ols_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = x_vars,
            cluster_col = 'poll_id',
            label       = f'state-level polls - {window_days} days before election'
        )

    # national regression 
    # skip if fewer than 10 complete cases
    national_complete = national_w[x_vars + ['A', 'poll_id']].dropna()
    if len(national_complete) < 10:
        print(f"  national regression skipped - only {len(national_complete)} complete cases")
        res_national = None
    else:
        res_national = run_ols_clustered(
            df          = national_w,
            y_col       = 'A',
            x_cols      = x_vars,
            cluster_col = 'poll_id',
            label       = f'national polls - {window_days} days before election'
        )

    return res_state, res_national

# run and store results for each window
window_results = {}
for window in time_windows:
    res_s, res_n = run_window_regressions(reg_state, reg_national, all_x_vars, window)
    window_results[window] = {'state': res_s, 'national': res_n}


######## save outputs
# save question-level accuracy dataset for further analysis
harris_trump_pivot.to_csv('data/harris_trump_accuracy.csv', index=False)

# save regression-ready dataset withs constructed covariates
reg_df.to_csv('data/harris_trump_regression.csv', index=False)

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("Accuracy analysis complete — see output/fiftyplusone_analysis_log.txt")