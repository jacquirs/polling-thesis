import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats


# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS USING THE SIMPLE MODES OUTPUT IN fiftyplusone_datelimited_analysis
# IT IS LIMITED TO THE TIME AFTER BIDEN DROPPED OUT
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy


# redirect all print output to a log file
log_file = open('output/fiftyplusone_datelimited_simple_mode_analysis_log.txt', 'w')
sys.stdout = log_file

########################################################################################
#################### LOAD DATASETS #####################################################
########################################################################################

# load the three-way mode dataset
df_threeway = pd.read_csv('data/harris_trump_datelimited_simple_mode_analysis_threeway.csv')

# load the pure binary dataset
df_pure = pd.read_csv('data/harris_trump_datelimited_simple_mode_analysis_pure.csv')

# election date for time window analysis
election_date = pd.Timestamp('2024-11-05')

# convert dates
for df in [df_threeway, df_pure]:
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['start_date'] = pd.to_datetime(df['start_date'])

# swing states
swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 'north carolina', 'pennsylvania', 'wisconsin']

# understand the questions we have, will be less than the exploded ones because those split much more
print(f"\nDatasets loaded:")
print(f"  Three-way (includes mixed): {len(df_threeway)} questions")
print(f"  Pure binary (excludes mixed): {len(df_pure)} questions")

########################################################################################
#################### HELPER FUNCTIONS FROM ORIGINAL FILE ###############################
########################################################################################

# from fiftyplusone_datelimited_analysis.py and fiftyplusone_analysis.py

def compute_clustered_se(df, value_col, cluster_col):
    """compute cluster-robust standard error of the mean"""
    df_clean = df[[value_col, cluster_col]].dropna()
    
    if len(df_clean) == 0:
        return np.nan, 0
    
    cluster_means = df_clean.groupby(cluster_col)[value_col].mean()
    n_clusters = len(cluster_means)
    
    if n_clusters < 2:
        return np.nan, n_clusters
    
    grand_mean = df_clean[value_col].mean()
    cluster_var = ((cluster_means - grand_mean) ** 2).sum() / (n_clusters - 1)
    se_robust = np.sqrt(cluster_var / n_clusters)
    
    return se_robust, n_clusters

def sig_stars(p):
    if p < 0.01:   return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    else:          return ''

def print_accuracy_table(df, group_col, label):
    """
    groups df by group_col and poll_level, computes mean/median/std/se/p-value/n of method a,
    and prints a formatted table with state and national columns side by side
    """
    results = []
    
    for (group_val, level), subdf in df.groupby([group_col, 'poll_level']):
        mean_A = subdf['A'].mean()
        se_robust, n_polls = compute_clustered_se(subdf, 'A', 'poll_id')

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
    print(f"  method a accuracy by {label}")
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

def run_ols_clustered(df, y_col, x_cols, cluster_col, label):
    """
    fits ols on df using x_cols to predict y_col.
    standard errors are clustered on cluster_col (huber-white sandwich).
    prints a formatted regression table with stars, adj-r2, constant, and n.
    returns the fitted statsmodels results object.
    """
    df_reg = df[x_cols + [y_col, cluster_col]].dropna()

    X      = sm.add_constant(df_reg[x_cols], has_constant='add')
    y      = df_reg[y_col]
    groups = df_reg[cluster_col]

    model  = sm.OLS(y, X)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})

    def stars(p):
        if p < 0.01:   return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else:          return ''

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

    intercept_name = next((v for v in params.index if v.lower() in ('const', 'intercept')), None)

    var_order = [v for v in params.index if v != intercept_name] + ([intercept_name] if intercept_name else [])
    for var in var_order:
        print(f"  {var:<35} {params[var]:>10.4f} {bse[var]:>10.4f} {stars(pvalues[var]):>6}")

    print(f"  {'-'*63}")
    print(f"  adjusted r2:  {result.rsquared_adj:.4f}")
    print(f"  n:            {int(result.nobs)}")
    print(f"{'='*70}\n")

    return result

##### add new function to name cateogories in tables 
def get_mode_label(row):
    if row['interviewer_only']:
        return 'Interviewer-Only'
    elif row['self_admin_only']:
        return 'Self-Admin-Only'
    elif row['mixed_mode']:
        return 'Mixed Mode'

df_threeway['mode_category'] = df_threeway.apply(get_mode_label, axis=1)
df_pure['mode_category'] = df_pure.apply(get_mode_label, axis=1)

########################################################################################
#################### DESCRIPTIVE ACCURACY TABLES #######################################
########################################################################################

# three-way comparison (includes mixed)
print_accuracy_table(df_threeway, 'mode_category', 'mode (three-way: interviewer-only vs self-admin-only vs mixed), datelimmited, with partisan')

# pure binary comparison (excludes mixed)
print_accuracy_table(df_pure, 'mode_category', 'mode (pure: self-admin-only vs interviewer-only, excludes mixed), datelimmited, with partisan')

########################################################################################
#################### PREPARE VARIABLES FOR REGRESSIONS #################################
########################################################################################

# VAR: statewide turnout
# load turnout data
turnout_data = pd.read_csv("data/Turnout_2024G_v0.3.csv")

# standardize state names and select needed columns
turnout_clean = turnout_data[['STATE', 'VEP_TURNOUT_RATE']].copy()
turnout_clean['state'] = turnout_clean['STATE'].str.strip().str.lower()
turnout_clean['turnout_pct'] = turnout_clean['VEP_TURNOUT_RATE'].str.rstrip('%').astype(float)

# rename united states to national
turnout_clean['state'] = turnout_clean['state'].replace('united states', 'national')

# merge turnout into df_threeway
df_threeway = df_threeway.merge(
    turnout_clean[['state', 'turnout_pct']],
    on='state',
    how='left'
)

# merge turnout into df_pure
df_pure = df_pure.merge(
    turnout_clean[['state', 'turnout_pct']],
    on='state',
    how='left'
)

# time variables
for df in [df_threeway, df_pure]:
    df['duration_days'] = (df['end_date'] - df['start_date']).dt.days + 1
    df['days_before_election'] = (election_date - df['end_date']).dt.days

# variable lists
time_vars = ['duration_days', 'days_before_election']
state_vars = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk', 'abs_margin']

# mode variables for each analysis type
# for three-way: use two dummies (drop interviewer_only as reference because that is the way polls traditionally were)
# CONSIDER COMING BACK AND CHANGING THIS
threeway_mode_vars = ['self_admin_only', 'mixed_mode']

# for pure binary: use one dummy (self_admin_only, with interviewer_only as reference)
pure_mode_var = ['self_admin_only']

# full covariate lists
state_x_threeway = time_vars + state_vars + threeway_mode_vars
national_x_threeway = time_vars + national_vars + threeway_mode_vars

state_x_pure = time_vars + state_vars + pure_mode_var
national_x_pure = time_vars + national_vars + pure_mode_var

# split by poll level
df_threeway_national = df_threeway[df_threeway['poll_level'] == 'national'].copy()
df_threeway_state = df_threeway[df_threeway['poll_level'] == 'state'].copy()
df_threeway_swing = df_threeway_state[df_threeway_state['state'].isin(swing_states)].copy()

df_pure_national = df_pure[df_pure['poll_level'] == 'national'].copy()
df_pure_state = df_pure[df_pure['poll_level'] == 'state'].copy()
df_pure_swing = df_pure_state[df_pure_state['state'].isin(swing_states)].copy()

print(f"\nSample sizes for regressions:")
print(f"\nThree-way (includes mixed):")
print(f"  National: {len(df_threeway_national)}")
print(f"  All states: {len(df_threeway_state)}")
print(f"  Swing states: {len(df_threeway_swing)}")

print(f"\nPure binary (excludes mixed):")
print(f"  National: {len(df_pure_national)}")
print(f"  All states: {len(df_pure_state)}")
print(f"  Swing states: {len(df_pure_swing)}")


########################################################################################
#################### BASE REGRESSIONS PURE BINARY ######################################
########################################################################################

print("BASE REGRESSIONS, PURE BINARY, DATELIMITED, WITH PARTISAN")

results_pure_national = run_ols_clustered(
    df=df_pure_national, y_col='A', x_cols=national_x_pure,
    cluster_col='poll_id', label='National pure binary mode'
)

results_pure_state = run_ols_clustered(
    df=df_pure_state, y_col='A', x_cols=state_x_pure,
    cluster_col='poll_id', label='All states pure binary mode'
)

results_pure_swing = run_ols_clustered(
    df=df_pure_swing, y_col='A', x_cols=state_x_pure,
    cluster_col='poll_id', label='Swing states pure binary mode'
)


########################################################################################
#################### BASE REGRESSIONS THREE WAY ########################################
########################################################################################

print("BASE REGRESSIONS, THREE WAY, DATELIMITED, WITH PARTISAN")
print("Reference category: Interviewer-Only")

results_threeway_national = run_ols_clustered(
    df=df_threeway_national, y_col='A', x_cols=national_x_threeway,
    cluster_col='poll_id', label='National three way mode'
)

results_threeway_state = run_ols_clustered(
    df=df_threeway_state, y_col='A', x_cols=state_x_threeway,
    cluster_col='poll_id', label='All states three way mode'
)

results_threeway_swing = run_ols_clustered(
    df=df_threeway_swing, y_col='A', x_cols=state_x_threeway,
    cluster_col='poll_id', label='Swing states three way mode'
)


########################################################################################
#################### TIME WINDOW REGRESSIONS - PURE BINARY #############################
########################################################################################
time_windows = [107, 90, 60, 30, 7]

print("TIME WINDOW REGRESSIONS, PURE BINARY MODE, DATELIMITED,WITH PARTISAN")

# national
pure_national_windows = {}
for window in time_windows:
    df_w = df_pure_national[df_pure_national['days_before_election'] <= window].copy()
    print(f"\nWindow: {window} days, National, N={len(df_w)}")
    
    pure_national_windows[window] = run_ols_clustered(
        df=df_w, y_col='A', x_cols=national_x_pure,
        cluster_col='poll_id', label=f'National pure binary mode, {window} dya window'
    )
  
# all states
pure_state_windows = {}
for window in time_windows:
    df_w = df_pure_state[df_pure_state['days_before_election'] <= window].copy()
    print(f"\nWindow: {window} days, All states, N={len(df_w)}")
    
    pure_state_windows[window] = run_ols_clustered(
        df=df_w, y_col='A', x_cols=state_x_pure,
        cluster_col='poll_id', label=f'All states pure binary mode, {window} day window'
    )
   
# swing states
pure_swing_windows = {}
for window in time_windows:
    df_w = df_pure_swing[df_pure_swing['days_before_election'] <= window].copy()
    print(f"\nWindow: {window} days, Swing States, N={len(df_w)}")
    
    pure_swing_windows[window] = run_ols_clustered(
        df=df_w, y_col='A', x_cols=state_x_pure,
        cluster_col='poll_id', label=f'Swing states pure binary mode, {window} day window'
    )


########################################################################################
#################### TIME WINDOW REGRESSIONS THREEWAY ##################################
########################################################################################

print("TIME WINDOW REGRESSIONS, THREE WAY MODE, DATELIMITED,WITH PARTISAN")

# national
threeway_national_windows = {}
for window in time_windows:
    df_w = df_threeway_national[df_threeway_national['days_before_election'] <= window].copy()
    print(f"\nWindow: {window} days, National, N={len(df_w)}")
    
    threeway_national_windows[window] = run_ols_clustered(
        df=df_w, y_col='A', x_cols=national_x_threeway,
        cluster_col='poll_id', label=f'National three way mode, {window} day window'
    )
  

# all states
threeway_state_windows = {}
for window in time_windows:
    df_w = df_threeway_state[df_threeway_state['days_before_election'] <= window].copy()
    print(f"\nWindow: {window} days, All states, N={len(df_w)}")
    
    threeway_state_windows[window] = run_ols_clustered(
        df=df_w, y_col='A', x_cols=state_x_threeway,
        cluster_col='poll_id', label=f'All states three way, {window} day window'
    )
  

# swing states
threeway_swing_windows = {}
for window in time_windows:
    df_w = df_threeway_swing[df_threeway_swing['days_before_election'] <= window].copy()
    print(f"\nWindow: {window} days, Swing States, N={len(df_w)}")
    
    threeway_swing_windows[window] = run_ols_clustered(
        df=df_w, y_col='A', x_cols=state_x_threeway,
        cluster_col='poll_id', label=f'Swing states three way, {window} day window'
    )
  

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__