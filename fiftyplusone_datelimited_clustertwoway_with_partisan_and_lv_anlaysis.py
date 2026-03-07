import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.stattools import durbin_watson


# redirect all print output to a log file
log_file = open('output/fiftyplusone_analysis_datelimited_clustertwoway_with_partisan_and_lv_log.txt', 'w')
sys.stdout = log_file

# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS
# IT IS LIMITED TO THE TIME AFTER BIDEN DROPPED OUT, for full time see fiftyplusone_analysis.py
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy

##### WITH PARTISAN CONTROL
print("WITH PARTISAN CONTROL")
print("WITH LV CONTROL")


# load cleaned harris+trump questions dataset (output from fiftyplusone_initial_analysis.py)
harris_trump_full_df = pd.read_csv("data/fiftyplusone_cleaned_harris_trump_questions.csv")

# load actual 2024 results
true_votes = pd.read_csv("data/true_votes_by_state_mengrep.csv")

# define cutoff date (Biden dropout)
dropout_cutoff = pd.Timestamp('2024-07-21')

# election day 2024
election_date  = pd.Timestamp('2024-11-05')   

# fix one dataset error
harris_trump_full_df['mode'] = harris_trump_full_df['mode'].str.replace('LIve Phone', 'Live Phone', regex=False)

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
    [['question_id', 'poll_id', 'pollster', 'partisan', 'state', 'start_date', 'end_date', 
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


##### LIMIT THE DATES TO PERIOD WHEN HARRIS WAS NOMINEE
harris_trump_pivot = harris_trump_pivot[harris_trump_pivot['start_date'] >=  dropout_cutoff] #maybe reeval this

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
############################# Create partisan flag ################################
########################################################################################
harris_trump_pivot["partisan_flag"] = (
    harris_trump_pivot["partisan"].notna() &
    (harris_trump_pivot["partisan"].str.strip() != "")
).astype(int)

flagged = (harris_trump_pivot["partisan_flag"] == 1).sum()
unflagged = (harris_trump_pivot["partisan_flag"] == 0).sum()

print("Partisan flagged:", flagged)
print("Partiasan unflagged:", unflagged)


########################################################################################
###################### Multivariate Regression Analysis Set Up #########################
########################################################################################

# unit of analysis: one row per question (question_id)
# dependent variable: method a (continuous, centered at 0)
# two separate regressions: state level polls and national polls
# standard errors: clustered by poll_id in both regressions (clustering on poll_id is robust to both heteroskedasticity and within-poll correlation, as multiple questions from the same poll have correlated errors. this is the huber-white sandwich estimator extended to clusters, satisfying the requirement from martin, traugott & kennedy for correlated errors)

# psuedocode for this section 
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

# function to run ols with two way clustered ses and print a formatted table
def run_ols_twoway_clustered(df, y_col, x_cols, cluster_col1, cluster_col2, label, min_obs_threshold=10):
    """
    fits ols on df using x_cols to predict y_col.
    standard errors are two-way clustered on cluster_col1 and cluster_col2.
    prints a formatted regression table with stars, adj-r2, constant, and n.
    returns the fitted statsmodels results object.
    
    if any variable has fewer than min_obs_threshold observations with variation,
    it will be displayed as '----' instead of a coefficient.
    """
    # drop rows with any missing values in the variables used
    df_reg = df[x_cols + [y_col, cluster_col1, cluster_col2]].dropna()

    # add intercept column
    X = sm.add_constant(df_reg[x_cols], has_constant='add')
    y = df_reg[y_col]
    
    # check for low-variance variables (modes with very few observations)
    # these will cause numerical issues
    low_variance_vars = []
    for col in x_cols:
        if col in df_reg.columns:
            # for dummy variables, check if we have enough observations in each category
            if df_reg[col].nunique() == 2:  # binary variable
                value_counts = df_reg[col].value_counts()
                if value_counts.min() < min_obs_threshold:
                    low_variance_vars.append(col)
    
    # fit ols first (without clustering)
    model = sm.OLS(y, X)
    result = model.fit()
    
    # convert cluster variables to categorical codes to get integer cluster ids
    # statsmodels requires integer cluster identifiers for two-way clustering
    cluster1_cat = pd.Categorical(df_reg[cluster_col1])
    cluster2_cat = pd.Categorical(df_reg[cluster_col2])
    
    # extract integer codes and ensure proper dtype
    cluster1 = cluster1_cat.codes.astype(np.int64)
    cluster2 = cluster2_cat.codes.astype(np.int64)
    
    # stack cluster ids into 2d array format required by cov_cluster_2groups
    groups = np.column_stack([cluster1, cluster2])
    
    # compute two-way clustered covariance matrix using statsmodels function
    # cov_cluster_2groups returns a tuple (cov_matrix, ...), so extract first element
    cov_result = cov_cluster_2groups(result, groups)
    
    # check if it's a tuple and extract the covariance matrix
    if isinstance(cov_result, tuple):
        cov_twoway = cov_result[0]
    else:
        cov_twoway = cov_result
    
    # extract standard errors from diagonal of covariance matrix
    bse_twoway = np.sqrt(np.diag(cov_twoway))
    
    # compute t-statistics and p-values using two-way clustered standard errors
    tvalues = result.params / bse_twoway
    
    # use conservative degrees of freedom: minimum cluster count minus 1
    n_cluster1 = len(cluster1_cat.categories)
    n_cluster2 = len(cluster2_cat.categories)
    df_resid = min(n_cluster1, n_cluster2) - 1
    
    # compute two-tailed p-values
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df_resid)

    # significance stars based on two-tailed p-values
    def stars_local(p):
        if p < 0.01:   return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else:          return ''

    # print formatted regression table
    print(f"\n{'='*75}")
    print(f"  ols regression: {label}")
    print(f"  dependent variable: method a  (+ = republican bias)")
    print(f"  standard errors: two-way clustered ({cluster_col1} & {cluster_col2})")
    print(f"{'='*75}")
    print(f"  {'variable':<35} {'coef':>10} {'se':>10} {'sig':>6}")
    print(f"  {'-'*68}")

    params = result.params
    
    # identify the intercept name (could be 'const', 'Intercept', or 'intercept')
    intercept_name = next((v for v in params.index if v.lower() in ('const', 'intercept')), None)

    # print all covariates first, then intercept at the bottom
    var_order = [v for v in params.index if v != intercept_name] + ([intercept_name] if intercept_name else [])
    
    # loop through variables and print coefficients with two-way clustered standard errors
    for i, var in enumerate(var_order):
        # get correct index position for this variable in the arrays
        idx = result.params.index.get_loc(var)
        
        # check if this variable should be suppressed due to low observations
        if var in low_variance_vars:
            print(f"  {var:<35} {'----':>10} {'----':>10} {'':>6}")
        else:
            print(f"  {var:<35} {params[var]:>10.4f} {bse_twoway[idx]:>10.4f} {stars_local(pvalues[idx]):>6}")

    print(f"  {'-'*68}")
    print(f"  adjusted r2:  {result.rsquared_adj:.4f}")
    print(f"  n:            {int(result.nobs)}")
    print(f"  clusters:     {n_cluster1} {cluster_col1}, {n_cluster2} {cluster_col2}")
    if low_variance_vars:
        print(f"  note:         ---- indicates <{min_obs_threshold} observations in category")
    print(f"{'='*75}\n")

    # store two-way clustered results in the result object for later use
    result.bse_twoway = bse_twoway
    result.pvalues_twoway = pvalues
    result.cov_params_twoway = cov_twoway
    result.low_variance_vars = low_variance_vars  # store for later reference
    
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
time_vars  = ['duration_days', 'days_before_election','partisan_flag']
state_vars = ['pct_dk', 'abs_margin','turnout_pct']
national_vars = ['pct_dk']


# split into statelevel and national samples
reg_state    = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()

# define swing states and create subset
swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 
                      'north carolina', 'pennsylvania', 'wisconsin']
reg_state_swing = reg_state[reg_state['state'].isin(swing_states)].copy()

print(f"\nregression sample sizes:")
print(f"  national-level questions: {len(reg_national)}")
print(f"  state-level questions:    {len(reg_state)}")
print(f"  swing state questions: {len(reg_state_swing)}")


########################################################################################
#################### PREPARE MODE FOR REGRESSIONS (LIVE PHONE REFERENCE) ###############
########################################################################################

# Save a copy of the original un-exploded data for non-mode regressions
reg_df_original = reg_df.copy()


# Create population dummies from the CLEANED data
reg_df['population'] = reg_df['population'].str.lower().str.strip()
reg_df_original['population'] = reg_df_original['population'].str.lower().str.strip()

# Drop observations with 'v' population (lowercase to match your data)
n_before = len(reg_df_original)
reg_df_original = reg_df_original[reg_df_original['population'] != 'v'].copy()
n_dropped = n_before - len(reg_df_original)

print(reg_df_original['population'].value_counts())

pop_dummies = pd.get_dummies(reg_df_original['population'], prefix='pop', drop_first=False)

# Set reference category and drop it
pop_dummies = pop_dummies.drop('pop_lv', axis=1)
reference_pop = 'lv'

print(f"\nPopulation reference category: {reference_pop}")

# Add pop_dummies to reg_df_original
pop_dummies = pop_dummies.astype(int)
reg_df_original = pd.concat([reg_df_original, pop_dummies], axis=1)

# NOW define pop_vars (after adding dummies to the dataframe)
pop_vars = [col for col in reg_df_original.columns if col.startswith('pop_')]

print(f"Population dummy variables created: {pop_vars}")

# Save if want to do VIF
reg_df_original.to_csv('data/harris_trump_regression_original_with_partisan_and_lv.csv', index=False)

# need to also drop 'v' from reg_df before exploding
reg_df = reg_df[reg_df['population'] != 'v'].copy()

# Add pop_dummies to reg_df as well (before exploding)
pop_dummies_for_reg_df = pd.get_dummies(reg_df['population'], prefix='pop', drop_first=False)
if f'pop_{reference_pop}' in pop_dummies_for_reg_df.columns:
    pop_dummies_for_reg_df = pop_dummies_for_reg_df.drop(f'pop_{reference_pop}', axis=1)
pop_dummies_for_reg_df = pop_dummies_for_reg_df.astype(int)
reg_df = pd.concat([reg_df, pop_dummies_for_reg_df], axis=1)

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
# EXPLODED versions (for mode regressions)
reg_state = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()
reg_state_swing = reg_state[reg_state['state'].isin(swing_states)].copy()

# ORIGINAL versions (for non-mode regressions)
reg_state_original = reg_df_original[reg_df_original['poll_level'] == 'state'].copy()
reg_national_original = reg_df_original[reg_df_original['poll_level'] == 'national'].copy()
reg_state_swing_original = reg_state_original[reg_state_original['state'].isin(swing_states)].copy()

print(f"\nexploded sample sizes after exploding:")
print(f"  all state polls: {len(reg_state)}")
print(f"  swing states only: {len(reg_state_swing)}")
print(f"  national polls: {len(reg_national)}")

print(f"\original sample sizes after exploding:")
print(f"  all state polls: {len(reg_state_original)}")
print(f"  swing states only: {len(reg_state_swing_original)}")
print(f"  national polls: {len(reg_national_original)}")

print(f"\original swing states breakdown:")
print(reg_state_swing_original['state'].value_counts().sort_index())

print(f"\n exploded swing states breakdown:")
print(reg_state_swing['state'].value_counts().sort_index())

# understand the exploded modes overall
print(f"\nexploded mode distribution in swing states:")
mode_dist = reg_state_swing['base_mode'].value_counts()
for mode, count in mode_dist.items():
    pct = 100 * count / len(reg_state_swing)
    marker = " (REFERENCE)" if mode == reference_mode else ""
    print(f"  {mode}: {count} ({pct:.1f}%){marker}")

# understand the exploded modes by poll level
print("BASE MODE DISTRIBUTION ACROSS SAMPLES")

base_mode_summary = []

for level in ['national', 'state', 'swing']:
    if level == 'national':
        df = reg_national
    elif level == 'state':
        df = reg_state
    else:
        df = reg_state_swing
    
    base_mode_dist = df['base_mode'].value_counts()
    
    for base_mode, count in base_mode_dist.items():
        pct = 100 * count / len(df)
        base_mode_summary.append({
            'sample': level,
            'mode': mode,
            'n': count,
            'pct': pct
        })

mode_df = pd.DataFrame(base_mode_summary)

# understand the original modes overall
print("\nOriginal Mode Strings:")
print(f"{'Mode':<50} {'N':>10} {'%':>10}")
print("-" * 70)

mode_original = reg_df_original['mode'].value_counts()
for mode, count in mode_original.items():
    pct = 100 * count / len(reg_df_original)
    print(f"{mode:<50} {count:>10} {pct:>9.1f}%")

print(f"\nTotal unique mode combinations: {reg_df_original['mode'].nunique()}")

# understand the original modes by poll level
print("ORIGINAL MODE DISTRIBUTION ACROSS SAMPLES")

mode_summary = []

for level in ['national', 'state', 'swing']:
    if level == 'national':
        df = reg_national_original
    elif level == 'state':
        df = reg_state_original
    else:
        df = reg_state_swing_original
    
    mode_dist = df['mode'].value_counts()
    
    for mode, count in mode_dist.items():
        pct = 100 * count / len(df)
        mode_summary.append({
            'sample': level,
            'mode': mode,
            'n': count,
            'pct': pct
        })

mode_df = pd.DataFrame(mode_summary)

for level in ['national', 'swing', 'state']:
    print(f"\n{level.upper()}:")
    subset = mode_df[mode_df['sample'] == level].sort_values('n', ascending=False)
    print(f"{'Mode':<40} {'N':>10} {'%':>10}")
    print("-" * 60)
    for _, row in subset.iterrows():
        print(f"{row['mode']:<40} {int(row['n']):>10} {row['pct']:>9.1f}%")

print("="*110 + "\n")


# update covariate lists
# without mode
state_x_vars_no_mode = state_vars + time_vars + pop_vars
national_x_vars_no_mode = national_vars + time_vars + pop_vars

# with mode
state_x_vars_with_mode = time_vars + state_vars + mode_vars + pop_vars
national_x_vars_with_mode = time_vars + national_vars + mode_vars + pop_vars


########################################################################################
######## BASE REGRESSIONS (NO TIME WINDOWS, NO MODE, SWING/NATIONAL/STATES) ############
########################################################################################

# national regression: also clustered by poll_id and pollster for the same reason, though with fewer polls clustering matters less
results_national = run_ols_twoway_clustered(
    df          = reg_national_original,
    y_col       = 'A',
    x_cols      = national_x_vars_no_mode,
    cluster_col1 = 'poll_id',
    cluster_col2 = 'pollster',
    label       = 'national polls'
)

# state regression: clustered ses by poll_id and pollster to account for the fact that multiple questions from the same poll share correlated errors and pollsters use same method
results_state = run_ols_twoway_clustered(
    df          = reg_state_original,
    y_col       = 'A',
    x_cols      = state_x_vars_no_mode,
    cluster_col1 = 'poll_id',
    cluster_col2 = 'pollster',
    label       = 'state-level polls'
)

# swing state regression
results_swing = run_ols_twoway_clustered(
    df          = reg_state_swing_original,
    y_col       = 'A',
    x_cols      = state_x_vars_no_mode,
    cluster_col1 = 'poll_id',
    cluster_col2 = 'pollster',
    label       = 'swing states'
)


########################################################################################
#################### BASE REGRESSIONS (NO TIME, WITH MODE, SWING/STATES/NATIONAL) ####################################
########################################################################################

print("REGRESSIONS WITH MODE CONTROLS")

# natioanl questions
results_national_mode = run_ols_twoway_clustered(
    df          = reg_national,
    y_col       = 'A',
    x_cols      = national_x_vars_with_mode,
    cluster_col1 = 'poll_id',
    cluster_col2 = 'pollster',
    label       = 'national polls with mode controls'
)

# all state questions
results_all_states_mode = run_ols_twoway_clustered(
    df          = reg_state,
    y_col       = 'A',
    x_cols      = state_x_vars_with_mode,
    cluster_col1 = 'poll_id',
    cluster_col2 = 'pollster',
    label       = 'all state polls with mode controls'
)

# swing state questions
results_swing_mode = run_ols_twoway_clustered(
    df          = reg_state_swing,
    y_col       = 'A',
    x_cols      = state_x_vars_with_mode,
    cluster_col1 = 'poll_id',
    cluster_col2 = 'pollster',
    label       = 'swing states with mode controls'
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
    state_w = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  swing state questions in window: {len(state_w)}")
    
    # check if we have enough complete cases
    state_complete = state_w[state_x_vars_no_mode + ['A', 'poll_id']].dropna() #there are no na to drop
    
    if len(state_complete) < 10:
        print(f"  swing state regression skipped, only {len(state_complete)} complete cases")
        swing_window_results_no_mode[window] = None
    else:
        res_swing = run_ols_twoway_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_no_mode,
            cluster_col1 = 'poll_id',
            cluster_col2 = 'pollster',
            label       = f'swing states {window} days before election (no mode)'
        )
        swing_window_results_no_mode[window] = res_swing

# all states by time window (no mode)
all_states_window_results_no_mode = {}

for window in time_windows:
    # filter all states to this time window
    state_w = reg_state_original[reg_state_original['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  all state questions in window: {len(state_w)}")
    
    # check if we have enough complete cases
    state_complete = state_w[state_x_vars_no_mode + ['A', 'poll_id']].dropna() #there are no na to drop
    
    if len(state_complete) < 10:
        print(f"  all state regression skipped, only {len(state_complete)} complete cases")
        all_states_window_results_no_mode[window] = None
    else:
        res_state = run_ols_twoway_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_no_mode,
            cluster_col1 = 'poll_id',
            cluster_col2 = 'pollster',
            label       = f'all states {window} days before election (no mode)'
        )
        all_states_window_results_no_mode[window] = res_state

# national by time window (no mode)
national_window_results_no_mode = {}

for window in time_windows:
    # filter national to this time window
    national_w = reg_national_original[reg_national_original['days_before_election'] <= window].copy()
    
    print(f"\n  window: {window} days before election")
    print(f"  national questions in window: {len(national_w)}")
    
    # check if we have enough complete cases
    national_complete = national_w[national_x_vars_no_mode + ['A', 'poll_id']].dropna() #there are no na to drop
    
    if len(national_complete) < 10:
        print(f"  national regression skipped, only {len(national_complete)} complete cases")
        national_window_results_no_mode[window] = None
    else:
        res_national = run_ols_twoway_clustered(
            df          = national_w,
            y_col       = 'A',
            x_cols      = national_x_vars_no_mode,
            cluster_col1 = 'poll_id',
            cluster_col2 = 'pollster',
            label       = f'national {window} days before election (no mode)'
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
    state_complete = state_w[state_x_vars_with_mode + ['A', 'poll_id']].dropna() #there are no na to drop
    
    if len(state_complete) < 10:
        print(f"  swing state regression skipped, only {len(state_complete)} complete cases")
        swing_window_results_with_mode[window] = None
    else:
        res_swing = run_ols_twoway_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_with_mode,
            cluster_col1 = 'poll_id',
            cluster_col2 = 'pollster',
            label       = f'swing states {window} days before election (with mode)'
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
    state_complete = state_w[state_x_vars_with_mode + ['A', 'poll_id']].dropna() #there are no na to drop
    
    if len(state_complete) < 10:
        print(f"  all state regression skipped, only {len(state_complete)} complete cases")
        all_states_window_results_with_mode[window] = None
    else:
        res_state = run_ols_twoway_clustered(
            df          = state_w,
            y_col       = 'A',
            x_cols      = state_x_vars_with_mode,
            cluster_col1 = 'poll_id',
            cluster_col2 = 'pollster',
            label       = f'all states {window} days before election (with mode)'
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
    national_complete = national_w[national_x_vars_with_mode + ['A', 'poll_id']].dropna() #there are no na to drop
    
    if len(national_complete) < 10:
        print(f"  national regression skipped, only {len(national_complete)} complete cases")
        national_window_results_with_mode[window] = None
    else:
        res_national = run_ols_twoway_clustered(
            df          = national_w,
            y_col       = 'A',
            x_cols      = national_x_vars_with_mode,
            cluster_col1 = 'poll_id',
            cluster_col2 = 'pollster',
            label       = f'national {window} days before election (with mode)'
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
print("NATIONAL, ACROSS TIME WINDOWS, NO MODE")
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
print("NATIONAL, ACROSS TIME WINDOWS, WITH MODE")
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
print("ALL STATES, ACROSS TIME WINDOWS, NO MODE")
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
print("ALL STATES, ACROSS TIME WINDOWS, WITH MODE")
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
print("SWING, ACROSS TIME WINDOWS, NO MODE")
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
print("SWING, ACROSS TIME WINDOWS< WITH MODE")
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
print("MODE COEFFICIENTS ACROSS SAMPLES")

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


########################################################################################
#################### SAMPLE SIZES ACROSS TIME WINDOWS ##############################
########################################################################################

print("\n" + "="*110)
print("SAMPLE SIZE CHANGES ACROSS TIME WINDOWS")
print("="*110)

# create sample size tracking
sample_tracking = []

for window in time_windows:
    # swing states
    state_w = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window].copy()
    n_swing = len(state_w)
    
    # all states
    state_w_all = reg_state_original[reg_state_original['days_before_election'] <= window].copy()
    n_all = len(state_w_all)
    
    # national
    national_w = reg_national_original[reg_national_original['days_before_election'] <= window].copy()
    n_national = len(national_w)
    
    sample_tracking.append({
        'window': window,
        'swing': n_swing,
        'all_states': n_all,
        'national': n_national
    })

# convert to dataframe
sample_df = pd.DataFrame(sample_tracking)

# print cumulative table
print("\nCumulative Sample Sizes (polls within X days of election):")
print(f"{'Window (days)':<15} {'Swing States':>15} {'All States':>15} {'National':>15}")
print("-" * 60)
for _, row in sample_df.iterrows():
    print(f"{int(row['window']):<15} {int(row['swing']):>15} {int(row['all_states']):>15} {int(row['national']):>15}")

# calculate and print marginal changes (polls added between windows)
print("\n\nPolls Excluded When Moving to Narrower Window:")
print(f"{'Window Change':<20} {'Swing States':>20} {'All States':>20} {'National':>20}")
print(f"{'(from to)':<20} {'N (% of prior)':>20} {'N (% of prior)':>20} {'N (% of prior)':>20}")
print("-" * 80)

for i in range(len(sample_df) - 1):
    from_window = sample_df.iloc[i]['window']
    to_window = sample_df.iloc[i + 1]['window']
    
    # calculate drops
    swing_drop = sample_df.iloc[i]['swing'] - sample_df.iloc[i + 1]['swing']
    swing_pct = 100 * swing_drop / sample_df.iloc[i]['swing'] if sample_df.iloc[i]['swing'] > 0 else 0
    
    all_drop = sample_df.iloc[i]['all_states'] - sample_df.iloc[i + 1]['all_states']
    all_pct = 100 * all_drop / sample_df.iloc[i]['all_states'] if sample_df.iloc[i]['all_states'] > 0 else 0
    
    nat_drop = sample_df.iloc[i]['national'] - sample_df.iloc[i + 1]['national']
    nat_pct = 100 * nat_drop / sample_df.iloc[i]['national'] if sample_df.iloc[i]['national'] > 0 else 0
    
    window_label = f"{int(from_window)} to {int(to_window)}"
    
    print(f"{window_label:<20} {swing_drop:>6} ({swing_pct:>5.1f}%){' ':>8} "
          f"{all_drop:>6} ({all_pct:>5.1f}%){' ':>8} "
          f"{nat_drop:>6} ({nat_pct:>5.1f}%)")

# print overall reduction from widest to narrowest
print("\n\nOverall Reduction (107 days to 7 days):")
print(f"{'Sample':<20} {'From':>10} {'To':>10} {'Drop':>10} {'% Reduction':>15}")
print("-" * 65)

swing_overall = ((sample_df.iloc[0]['swing'] - sample_df.iloc[-1]['swing']) / 
                 sample_df.iloc[0]['swing'] * 100)
all_overall = ((sample_df.iloc[0]['all_states'] - sample_df.iloc[-1]['all_states']) / 
               sample_df.iloc[0]['all_states'] * 100)
nat_overall = ((sample_df.iloc[0]['national'] - sample_df.iloc[-1]['national']) / 
               sample_df.iloc[0]['national'] * 100)

print(f"{'Swing States':<20} {int(sample_df.iloc[0]['swing']):>10} "
      f"{int(sample_df.iloc[-1]['swing']):>10} "
      f"{int(sample_df.iloc[0]['swing'] - sample_df.iloc[-1]['swing']):>10} "
      f"{swing_overall:>14.1f}%")

print(f"{'All States':<20} {int(sample_df.iloc[0]['all_states']):>10} "
      f"{int(sample_df.iloc[-1]['all_states']):>10} "
      f"{int(sample_df.iloc[0]['all_states'] - sample_df.iloc[-1]['all_states']):>10} "
      f"{all_overall:>14.1f}%")

print(f"{'National':<20} {int(sample_df.iloc[0]['national']):>10} "
      f"{int(sample_df.iloc[-1]['national']):>10} "
      f"{int(sample_df.iloc[0]['national'] - sample_df.iloc[-1]['national']):>10} "
      f"{nat_overall:>14.1f}%")

print("="*110 + "\n")


########################################################################################
#################### DESCRIPTIVE STATISTICS FOR REGRESSION VARIABLES ###################
########################################################################################

print("\n" + "="*110)
print("DESCRIPTIVE STATISTICS - REGRESSION VARIABLES")
print("="*110)

# All questions
desc_all = reg_df_original[['A', 'duration_days', 'days_before_election', 'pct_dk', 'abs_margin', 'turnout_pct','partisan_flag','pop_a','pop_rv']].describe()

print("\nAll Questions (N={})".format(len(reg_df_original)))
print(desc_all.to_string())

# National only
desc_nat = reg_national_original[['A', 'duration_days', 'days_before_election', 'pct_dk','partisan_flag','pop_a','pop_rv']].describe()

print("\n\nNational Questions (N={})".format(len(reg_national_original)))
print(desc_nat.to_string())

# Swing states only
desc_swing = reg_state_swing_original[['A', 'duration_days', 'days_before_election', 'pct_dk', 'abs_margin', 'turnout_pct','partisan_flag','pop_a','pop_rv']].describe()

print("\n\nSwing State Questions (N={})".format(len(reg_state_swing_original)))
print(desc_swing.to_string())

print("="*110 + "\n")

# All states only
desc_states = reg_state_original[['A', 'duration_days', 'days_before_election', 'pct_dk', 'abs_margin', 'turnout_pct','partisan_flag']].describe()

print("\n\n All State Questions (N={})".format(len(reg_state_original)))
print(desc_states.to_string())

print("="*110 + "\n")


########################################################################################
#################### CORRELATION MATRIX FOR INDEPENDENT VARIABLES ######################
########################################################################################

print("\n" + "="*110)
print("CORRELATION MATRIX INDEPENDENT VARIABLES, NO MODE")
print("="*110)

# State variables
print("\nSwing state-level polls:")
corr_state_swing = reg_state_swing_original[state_x_vars_no_mode].corr()
print(corr_state_swing.to_string())

# State variables
print("\nState-level polls:")
corr_state = reg_state_original[state_x_vars_no_mode].corr()
print(corr_state.to_string())

# National variables
print("\n\nNational polls:")
corr_national = reg_national_original[national_x_vars_no_mode].corr()
print(corr_national.to_string())

print("="*110 + "\n")

print("\n" + "="*110)
print("CORRELATION MATRIX INDEPENDENT VARIABLES, WITH MODE")
print("="*110)

# State variables
print("\nSwing state-level polls:")
corr_state_swing = reg_state_swing[state_x_vars_with_mode].corr()
print(corr_state_swing.to_string())

# State variables
print("\nState-level polls:")
corr_state = reg_state[state_x_vars_with_mode].corr()
print(corr_state.to_string())

# National variables
print("\n\nNational polls:")
corr_national = reg_national[national_x_vars_with_mode].corr()
print(corr_national.to_string())

print("="*110 + "\n")


########################################################################################
#################### POLLSTER DIST ######################
########################################################################################
print("\nPollster distribution swing state:")
pollster_counts = reg_state_swing_original.groupby('pollster').agg({
    'poll_id': 'nunique',
    'question_id': 'count'
}).rename(columns={'poll_id': 'n_polls', 'question_id': 'n_questions'})
print(pollster_counts.sort_values('n_questions', ascending=False).to_string())

print("\nPollster distribution all state:")
pollster_counts = reg_state_original.groupby('pollster').agg({
    'poll_id': 'nunique',
    'question_id': 'count'
}).rename(columns={'poll_id': 'n_polls', 'question_id': 'n_questions'})
print(pollster_counts.sort_values('n_questions', ascending=False).to_string())

print("\nPollster distribution national:")
pollster_counts = reg_national_original.groupby('pollster').agg({
    'poll_id': 'nunique',
    'question_id': 'count'
}).rename(columns={'poll_id': 'n_polls', 'question_id': 'n_questions'})
print(pollster_counts.sort_values('n_questions', ascending=False).to_string())


########################################################################################
#################### SAMPLE CHARACTERISTICS OVER TIME ######################################
########################################################################################

print("\n" + "="*110)
print("SAMPLE COMPOSITION OVER TIME")
print("="*110)

time_windows = [107, 90, 60, 30, 7]

print("\nSample characteristics by time window (Swing States):")
print(f"{'Window':<10} {'N':>8} {'Mean A':>12} {'Mean Days':>12} {'Mean Duration':>15}")
print("-" * 60)

for window in time_windows:
    state_w = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window].copy()
    print(f"{window:<10} {len(state_w):>8} {state_w['A'].mean():>12.4f} "
          f"{state_w['days_before_election'].mean():>12.1f} "
          f"{state_w['duration_days'].mean():>15.1f}")

print("\nSample characteristics by time window (All States):")
print(f"{'Window':<10} {'N':>8} {'Mean A':>12} {'Mean Days':>12} {'Mean Duration':>15}")
print("-" * 60)

for window in time_windows:
    state_w = reg_state_original[reg_state_original['days_before_election'] <= window].copy()
    print(f"{window:<10} {len(state_w):>8} {state_w['A'].mean():>12.4f} "
          f"{state_w['days_before_election'].mean():>12.1f} "
          f"{state_w['duration_days'].mean():>15.1f}")

print("\nSample characteristics by time window (National):")
print(f"{'Window':<10} {'N':>8} {'Mean A':>12} {'Mean Days':>12} {'Mean Duration':>15}")
print("-" * 60)

for window in time_windows:
    nat_w = reg_national_original[reg_national_original['days_before_election'] <= window].copy()
    print(f"{window:<10} {len(nat_w):>8} {nat_w['A'].mean():>12.4f} "
          f"{nat_w['days_before_election'].mean():>12.1f} "
          f"{nat_w['duration_days'].mean():>15.1f}")
    

########################################################################################
#################### RESIDUAL AUTOCORRELATION TEST #####################################
########################################################################################

print("\n" + "="*110)
print("RESIDUAL AUTOCORRELATION DIAGNOSTICS")
print("="*110)

def test_autocorrelation(df, x_vars, label, states_list=None):
    """Test residual autocorrelation for a given sample"""
    
    print(f"\n{label}:")
    print("-" * 70)
    
    # Prepare regression data
    df_reg = df[x_vars + ['A', 'poll_id', 'end_date']].dropna()
    X = sm.add_constant(df_reg[x_vars], has_constant='add')
    y = df_reg['A']
    
    # Fit OLS WITHOUT clustering to get residuals
    model = sm.OLS(y, X)
    result_ols = model.fit()
    
    # Add residuals to dataframe
    df_with_resid = df.copy()
    df_with_resid['residuals'] = np.nan
    df_with_resid.loc[df_reg.index, 'residuals'] = result_ols.resid
    df_with_resid = df_with_resid.dropna(subset=['residuals'])
    
    # Sort by date and compute Durbin-Watson
    df_sorted = df_with_resid.sort_values('end_date')
    dw_stat = durbin_watson(df_sorted['residuals'])
    
    print(f"  Durbin-Watson statistic: {dw_stat:.3f}")
    print(f"  N observations: {len(df_sorted)}")
    print(f"  Interpretation:")
    if dw_stat < 1.5:
        print(f"     Strong positive autocorrelation (DW < 1.5)")
        print(f"      Consider time clustering")
    elif 1.5 <= dw_stat < 1.8:
        print(f"      Mild positive autocorrelation (1.5 <= DW < 1.8)")
        print(f"      Two-way clustering should be adequate")
    else:
        print(f"      No substantial autocorrelation (DW >= 1.8)")
        print(f"      Two-way clustering is sufficient")
    
    # Check weekly variance
    df_with_resid['week'] = df_with_resid['end_date'].dt.isocalendar().week
    weekly_var = df_with_resid.groupby('week')['residuals'].var()
    
    print(f"\n  Weekly variance:")
    print(f"    Min: {weekly_var.min():.6f}")
    print(f"    Max: {weekly_var.max():.6f}")
    print(f"    Ratio (max/min): {weekly_var.max() / weekly_var.min():.2f}x")
    
    if weekly_var.max() / weekly_var.min() > 10:
        print(f"      Large variance changes (ratio > 10)")
    else:
        print(f"      Stable variance across time")
    
    # If state-level data, check by state
    if states_list is not None:
        print(f"\n  Residual patterns by state:")
        print(f"  {'State':<15} {'N':>8} {'Mean':>10} {'SD':>10} {'Mean|Resid|':>12}")
        print("  " + "-" * 60)
        
        for state in states_list:
            state_data = df_with_resid[df_with_resid['state'] == state]
            if len(state_data) > 0:
                resid_mean = state_data['residuals'].mean()
                resid_sd = state_data['residuals'].std()
                resid_mean_abs = state_data['residuals'].abs().mean()
                print(f"  {state.title():<15} {len(state_data):>8} {resid_mean:>10.4f} {resid_sd:>10.4f} {resid_mean_abs:>12.4f}")
    
    return dw_stat, weekly_var

# Test all three samples
print("\n" + "="*70)
print("SWING STATES")
print("="*70)
dw_swing, weekly_swing = test_autocorrelation(
    df=reg_state_swing_original,
    x_vars=state_x_vars_no_mode,
    label="Swing States (Non-Mode Regression)",
    states_list=swing_states
)

print("\n" + "="*70)
print("ALL STATES")
print("="*70)
dw_all, weekly_all = test_autocorrelation(
    df=reg_state_original,
    x_vars=state_x_vars_no_mode,
    label="All States (Non-Mode Regression)",
    states_list=None  # Too many states to list individually
)

print("\n" + "="*70)
print("NATIONAL")
print("="*70)
dw_national, weekly_national = test_autocorrelation(
    df=reg_national_original,
    x_vars=national_x_vars_no_mode,
    label="National Polls (Non-Mode Regression)",
    states_list=None
)

# Summary table
print("\n" + "="*110)
print("SUMMARY: DURBIN-WATSON STATISTICS")
print("="*110)
print(f"\n{'Sample':<20} {'DW Statistic':>15} {'N':>10} {'Interpretation':>40}")
print("-" * 85)
print(f"{'Swing States':<20} {dw_swing:>15.3f} {len(reg_state_swing_original):>10} {'Adequate' if dw_swing >= 1.5 else 'Time clustering needed':>40}")
print(f"{'All States':<20} {dw_all:>15.3f} {len(reg_state_original):>10} {'Adequate' if dw_all >= 1.5 else 'Time clustering needed':>40}")
print(f"{'National':<20} {dw_national:>15.3f} {len(reg_national_original):>10} {'Adequate' if dw_national >= 1.5 else 'Time clustering needed':>40}")

print("\n" + "="*110 + "\n")


########################################################################################
#################### PARTISAN FLAG VARIANCE DECOMPOSITION ##############################
########################################################################################
### determine why partisan sees an se decrease with poll_id clustering

print("\n" + "="*110)
print("PARTISAN FLAG CONSISTENCY CHECK - DO POLLS HAVE MULTIPLE PARTISAN VALUES?")
print("="*110)

# Check if any poll_id has multiple different partisan_flag values
partisan_check = reg_df_original.groupby('poll_id')['partisan_flag'].nunique()

# Count how many polls have multiple partisan values
polls_with_multiple = (partisan_check > 1).sum()
polls_with_single = (partisan_check == 1).sum()

print(f"\nTotal polls: {len(partisan_check)}")
print(f"Polls with single partisan value: {polls_with_single}")
print(f"Polls with multiple partisan values: {polls_with_multiple}")

if polls_with_multiple > 0:
    print(f"\n WARNING: {polls_with_multiple} polls have multiple partisan flag values!")
    print(f"This violates the assumption that partisan flag is poll-level.")
    
    # Show examples
    print(f"\nExamples of polls with multiple partisan values:")
    problematic_polls = partisan_check[partisan_check > 1].head(10)
    
    for poll_id, n_values in problematic_polls.items():
        poll_data = reg_df_original[reg_df_original['poll_id'] == poll_id]
        print(f"\n  Poll ID: {poll_id}")
        print(f"    Number of partisan values: {n_values}")
        print(f"    Number of questions: {len(poll_data)}")
        print(f"    Partisan values: {poll_data['partisan_flag'].unique()}")
        print(f"    Pollster: {poll_data['pollster'].iloc[0]}")
else:
    print(f"\n CONFIRMED: All polls have consistent partisan flag values")
    print(f"Partisan flag is correctly applied at poll level.")

# Additional check: Show distribution of partisan values
print(f"\n" + "-"*110)
print("PARTISAN FLAG DISTRIBUTION:")
print(f"\nBy questions:")
print(reg_df_original['partisan_flag'].value_counts().to_string())

print(f"\nBy polls (unique poll_id):")
poll_partisan = reg_df_original.groupby('poll_id')['partisan_flag'].first()
print(poll_partisan.value_counts().to_string())

print("="*110 + "\n")

print("\n" + "="*110)
print("PARTISAN FLAG VARIANCE ANALYSIS - WHY DEFF < 1.0?")
print("="*110)

# Calculate variance at different levels
print("\nSwing States:")

# Total variance (treating all questions as independent)
total_var = reg_state_swing_original['partisan_flag'].var()
print(f"  Total variance (all questions): {total_var:.6f}")

# Between-poll variance
poll_means = reg_state_swing_original.groupby('poll_id')['partisan_flag'].mean()
between_poll_var = poll_means.var()
print(f"  Between-poll variance: {between_poll_var:.6f}")

# Within-poll variance
within_poll_var = 0.0  # Should be zero since partisan is poll-level
for poll_id in reg_state_swing_original['poll_id'].unique():
    poll_data = reg_state_swing_original[reg_state_swing_original['poll_id'] == poll_id]
    if len(poll_data) > 1:
        within_poll_var += poll_data['partisan_flag'].var() * (len(poll_data) - 1)

n_total = len(reg_state_swing_original)
n_polls = reg_state_swing_original['poll_id'].nunique()
within_poll_var = within_poll_var / (n_total - n_polls)

print(f"  Within-poll variance: {within_poll_var:.6f}")

# ICC for partisan flag
if between_poll_var + within_poll_var > 0:
    icc_partisan = between_poll_var / (between_poll_var + within_poll_var)
else:
    icc_partisan = 1.0
    
print(f"  ICC (partisan flag): {icc_partisan:.6f}")

# Check effective sample size
print(f"\n  Total questions: {n_total}")
print(f"  Total polls: {n_polls}")
print(f"  Questions per poll (avg): {n_total / n_polls:.2f}")

# Distribution check
print(f"\nPartisan flag distribution in swing states:")
partisan_dist = reg_state_swing_original['partisan_flag'].value_counts()
for val, count in partisan_dist.items():
    pct = 100 * count / len(reg_state_swing_original)
    print(f"  partisan_flag={val}: {count} questions ({pct:.1f}%)")

# At poll level
poll_partisan_dist = poll_means.value_counts()
print(f"\nPartisan flag distribution at poll level:")
for val, count in poll_partisan_dist.items():
    pct = 100 * count / len(poll_means)
    print(f"  partisan_flag={val}: {count} polls ({pct:.1f}%)")

# Compare to other variables
print(f"\n" + "-"*110)
print("Comparison to other variables:")

for var in ['days_before_election', 'pct_dk', 'partisan_flag']:
    # Total variance
    total_v = reg_state_swing_original[var].var()
    
    # Between-poll variance
    poll_m = reg_state_swing_original.groupby('poll_id')[var].mean()
    between_v = poll_m.var()
    
    # ICC
    # Simplified calculation
    if total_v > 0:
        approx_icc = between_v / total_v
    else:
        approx_icc = 0
    
    print(f"\n  {var}:")
    print(f"    Total variance: {total_v:.6f}")
    print(f"    Between-poll variance: {between_v:.6f}")
    print(f"    Approx ICC: {approx_icc:.4f}")

print("="*110 + "\n")

print("\n" + "="*110)
print("PARTISAN FLAG COMPOSITION EFFECT - WHY BETWEEN-POLL VARIANCE > TOTAL VARIANCE")
print("="*110)

# Check questions per poll by partisan status
print("\nSwing States - Questions per poll by partisan status:")

partisan_polls = reg_state_swing_original[reg_state_swing_original['partisan_flag'] == 1]['poll_id'].unique()
nonpartisan_polls = reg_state_swing_original[reg_state_swing_original['partisan_flag'] == 0]['poll_id'].unique()

# Count questions per partisan poll
partisan_questions_per_poll = []
for poll_id in partisan_polls:
    n_questions = len(reg_state_swing_original[reg_state_swing_original['poll_id'] == poll_id])
    partisan_questions_per_poll.append(n_questions)

# Count questions per non-partisan poll
nonpartisan_questions_per_poll = []
for poll_id in nonpartisan_polls:
    n_questions = len(reg_state_swing_original[reg_state_swing_original['poll_id'] == poll_id])
    nonpartisan_questions_per_poll.append(n_questions)

print(f"\nPartisan polls (partisan_flag=1):")
print(f"  Number of polls: {len(partisan_polls)}")
print(f"  Total questions: {sum(partisan_questions_per_poll)}")
print(f"  Questions per poll (mean): {np.mean(partisan_questions_per_poll):.2f}")
print(f"  Questions per poll (median): {np.median(partisan_questions_per_poll):.1f}")
print(f"  Distribution:")
from collections import Counter
partisan_dist = Counter(partisan_questions_per_poll)
for n_q in sorted(partisan_dist.keys()):
    count = partisan_dist[n_q]
    pct = 100 * count / len(partisan_polls)
    print(f"    {n_q} question(s): {count} polls ({pct:.1f}%)")

print(f"\nNon-partisan polls (partisan_flag=0):")
print(f"  Number of polls: {len(nonpartisan_polls)}")
print(f"  Total questions: {sum(nonpartisan_questions_per_poll)}")
print(f"  Questions per poll (mean): {np.mean(nonpartisan_questions_per_poll):.2f}")
print(f"  Questions per poll (median): {np.median(nonpartisan_questions_per_poll):.1f}")
print(f"  Distribution:")
nonpartisan_dist = Counter(nonpartisan_questions_per_poll)
for n_q in sorted(nonpartisan_dist.keys()):
    count = nonpartisan_dist[n_q]
    pct = 100 * count / len(nonpartisan_polls)
    print(f"    {n_q} question(s): {count} polls ({pct:.1f}%)")

print("\n" + "="*110)
print("PARTISAN FLAG COMPOSITION EFFECT - NATIONAL LEVEL")
print("="*110)

# Check questions per poll by partisan status - NATIONAL
print("\nNational - Questions per poll by partisan status:")

partisan_polls_nat = reg_national_original[reg_national_original['partisan_flag'] == 1]['poll_id'].unique()
nonpartisan_polls_nat = reg_national_original[reg_national_original['partisan_flag'] == 0]['poll_id'].unique()

# Count questions per partisan poll
partisan_questions_per_poll_nat = []
for poll_id in partisan_polls_nat:
    n_questions = len(reg_national_original[reg_national_original['poll_id'] == poll_id])
    partisan_questions_per_poll_nat.append(n_questions)

# Count questions per non-partisan poll
nonpartisan_questions_per_poll_nat = []
for poll_id in nonpartisan_polls_nat:
    n_questions = len(reg_national_original[reg_national_original['poll_id'] == poll_id])
    nonpartisan_questions_per_poll_nat.append(n_questions)

print(f"\nPartisan polls (partisan_flag=1):")
print(f"  Number of polls: {len(partisan_polls_nat)}")
print(f"  Total questions: {sum(partisan_questions_per_poll_nat)}")
print(f"  Questions per poll (mean): {np.mean(partisan_questions_per_poll_nat):.2f}")
print(f"  Questions per poll (median): {np.median(partisan_questions_per_poll_nat):.1f}")
print(f"  Distribution:")
from collections import Counter
partisan_dist_nat = Counter(partisan_questions_per_poll_nat)
for n_q in sorted(partisan_dist_nat.keys()):
    count = partisan_dist_nat[n_q]
    pct = 100 * count / len(partisan_polls_nat)
    print(f"    {n_q} question(s): {count} polls ({pct:.1f}%)")

print(f"\nNon-partisan polls (partisan_flag=0):")
print(f"  Number of polls: {len(nonpartisan_polls_nat)}")
print(f"  Total questions: {sum(nonpartisan_questions_per_poll_nat)}")
print(f"  Questions per poll (mean): {np.mean(nonpartisan_questions_per_poll_nat):.2f}")
print(f"  Questions per poll (median): {np.median(nonpartisan_questions_per_poll_nat):.1f}")
print(f"  Distribution:")
nonpartisan_dist_nat = Counter(nonpartisan_questions_per_poll_nat)
for n_q in sorted(nonpartisan_dist_nat.keys()):
    count = nonpartisan_dist_nat[n_q]
    pct = 100 * count / len(nonpartisan_polls_nat)
    print(f"    {n_q} question(s): {count} polls ({pct:.1f}%)")


print("\n" + "="*110)
print("PARTISAN FLAG COMPOSITION EFFECT - ALL STATES LEVEL")
print("="*110)

# Check questions per poll by partisan status - ALL STATES
print("\nAll States - Questions per poll by partisan status:")

partisan_polls_all = reg_state_original[reg_state_original['partisan_flag'] == 1]['poll_id'].unique()
nonpartisan_polls_all = reg_state_original[reg_state_original['partisan_flag'] == 0]['poll_id'].unique()

# Count questions per partisan poll
partisan_questions_per_poll_all = []
for poll_id in partisan_polls_all:
    n_questions = len(reg_state_original[reg_state_original['poll_id'] == poll_id])
    partisan_questions_per_poll_all.append(n_questions)

# Count questions per non-partisan poll
nonpartisan_questions_per_poll_all = []
for poll_id in nonpartisan_polls_all:
    n_questions = len(reg_state_original[reg_state_original['poll_id'] == poll_id])
    nonpartisan_questions_per_poll_all.append(n_questions)

print(f"\nPartisan polls (partisan_flag=1):")
print(f"  Number of polls: {len(partisan_polls_all)}")
print(f"  Total questions: {sum(partisan_questions_per_poll_all)}")
print(f"  Questions per poll (mean): {np.mean(partisan_questions_per_poll_all):.2f}")
print(f"  Questions per poll (median): {np.median(partisan_questions_per_poll_all):.1f}")
print(f"  Distribution:")
partisan_dist_all = Counter(partisan_questions_per_poll_all)
for n_q in sorted(partisan_dist_all.keys()):
    count = partisan_dist_all[n_q]
    pct = 100 * count / len(partisan_polls_all)
    print(f"    {n_q} question(s): {count} polls ({pct:.1f}%)")

print(f"\nNon-partisan polls (partisan_flag=0):")
print(f"  Number of polls: {len(nonpartisan_polls_all)}")
print(f"  Total questions: {sum(nonpartisan_questions_per_poll_all)}")
print(f"  Questions per poll (mean): {np.mean(nonpartisan_questions_per_poll_all):.2f}")
print(f"  Questions per poll (median): {np.median(nonpartisan_questions_per_poll_all):.1f}")
print(f"  Distribution:")
nonpartisan_dist_all = Counter(nonpartisan_questions_per_poll_all)
for n_q in sorted(nonpartisan_dist_all.keys()):
    count = nonpartisan_dist_all[n_q]
    pct = 100 * count / len(nonpartisan_polls_all)
    print(f"    {n_q} question(s): {count} polls ({pct:.1f}%)")

# Calculate weighted proportion
print(f"\n" + "-"*110)
print("Weighted vs Unweighted Proportions:")

# Unweighted (at poll level)
unweighted_partisan_pct = 100 * len(partisan_polls) / (len(partisan_polls) + len(nonpartisan_polls))
print(f"  Unweighted (poll-level): {unweighted_partisan_pct:.1f}% partisan")

# Weighted (at question level)
total_questions = sum(partisan_questions_per_poll) + sum(nonpartisan_questions_per_poll)
weighted_partisan_pct = 100 * sum(partisan_questions_per_poll) / total_questions
print(f"  Weighted (question-level): {weighted_partisan_pct:.1f}% partisan")

print(f"\nDifference: {unweighted_partisan_pct - weighted_partisan_pct:.1f} percentage points")

if unweighted_partisan_pct > weighted_partisan_pct:
    print(f"     Partisan polls have FEWER questions per poll on average")
    print(f"     Multi-question polls are disproportionately non-partisan")
    print(f"     This increases between-poll variance relative to total variance")
else:
    print(f"     Partisan polls have MORE questions per poll on average")
    print(f"     Multi-question polls are disproportionately partisan")

print("="*110 + "\n")


######## save outputs
# save regression-ready dataset withs constructed covariates
reg_df.to_csv('data/harris_trump_datelimted_clustertwoway_with_partisan_and_lv_regression.csv', index=False)

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("Accuracy analysis complete — see output/fiftyplusone_analysis_datelimited_clustertwoway_with_partisan_and_lv_log.txt")