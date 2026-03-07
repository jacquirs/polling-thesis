import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats

# redirect all print output to a log file
log_file = open('output/fiftyplusone_clustering_diagnostics_log.txt', 'w')
sys.stdout = log_file

print("="*110)
print("CLUSTERING DIAGNOSTICS FOR POLL ACCURACY REGRESSIONS")
print("="*110)

########################################################################################
#################### LOAD DATA #########################################################
########################################################################################

# load the regression-ready dataset
# this should already have ALL variables including turnout_pct
reg_df = pd.read_csv('data/harris_trump_datelimted_check_for_clusering.csv')

# convert date columns to datetime
reg_df['end_date'] = pd.to_datetime(reg_df['end_date'])
reg_df['start_date'] = pd.to_datetime(reg_df['start_date'])

# define variable lists
time_vars = ['duration_days', 'days_before_election','partisan_flag']
state_vars = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk']
state_x_vars = time_vars + state_vars
national_x_vars = time_vars + national_vars

# create subsets
reg_state = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()

swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 
                'north carolina', 'pennsylvania', 'wisconsin']
reg_state_swing = reg_state[reg_state['state'].isin(swing_states)].copy()

print(f"\nData loaded successfully")
print(f"  Total questions: {len(reg_df)}")
print(f"  State questions: {len(reg_state)}")
print(f"  Swing state questions: {len(reg_state_swing)}")
print(f"  National questions: {len(reg_national)}")


########################################################################################
#################### CLUSTERING DIAGNOSTICS - POLL_ID ##################################
########################################################################################

print("\n" + "="*110)
print("PART 1: CLUSTER SIZE DISTRIBUTION (POLL_ID)")
print("="*110)

def analyze_clusters(df, cluster_col, label):
    """analyze the distribution of observations per cluster"""
    cluster_sizes = df.groupby(cluster_col).size()
    
    print(f"\n{label}:")
    print(f"  Total observations: {len(df)}")
    print(f"  Total clusters ({cluster_col}): {len(cluster_sizes)}")
    print(f"  Mean observations per cluster: {cluster_sizes.mean():.2f}")
    print(f"  Median observations per cluster: {cluster_sizes.median():.1f}")
    print(f"  Min observations per cluster: {cluster_sizes.min()}")
    print(f"  Max observations per cluster: {cluster_sizes.max()}")
    print(f"  Std dev of cluster sizes: {cluster_sizes.std():.2f}")
    
    # show distribution
    print(f"\n  Distribution of cluster sizes:")
    print(f"    {'Cluster size':<20} {'N clusters':>15} {'% of clusters':>15}")
    print("    " + "-" * 50)
    
    size_dist = cluster_sizes.value_counts().sort_index()
    for size, count in size_dist.items():
        pct = 100 * count / len(cluster_sizes)
        print(f"    {size:<20} {count:>15} {pct:>14.1f}%")
    
    # identify largest clusters
    print(f"\n  Top 10 largest clusters:")
    print(f"    {'Cluster ID':<30} {'N observations':>15}")
    print("    " + "-" * 45)
    largest = cluster_sizes.nlargest(10)
    for cluster_id, size in largest.items():
        print(f"    {str(cluster_id)[:30]:<30} {size:>15}")
    
    return cluster_sizes

# analyze for each sample
swing_cluster_sizes = analyze_clusters(reg_state_swing, 'poll_id', 'Swing States (poll_id)')
all_cluster_sizes = analyze_clusters(reg_state, 'poll_id', 'All States (poll_id)')
national_cluster_sizes = analyze_clusters(reg_national, 'poll_id', 'National (poll_id)')


########################################################################################
#################### CLUSTERING DIAGNOSTICS - POLLSTER #################################
########################################################################################

print("\n" + "="*110)
print("PART 2: CLUSTER SIZE DISTRIBUTION (POLLSTER)")
print("="*110)

# analyze pollster clusters
swing_pollster_sizes = analyze_clusters(reg_state_swing, 'pollster', 'Swing States (pollster)')
all_pollster_sizes = analyze_clusters(reg_state, 'pollster', 'All States (pollster)')
national_pollster_sizes = analyze_clusters(reg_national, 'pollster', 'National (pollster)')


########################################################################################
#################### INTRA-CLUSTER CORRELATION (ICC) ###################################
########################################################################################

print("\n" + "="*110)
print("PART 3: INTRA-CLUSTER CORRELATION (ICC) - JUSTIFICATION FOR CLUSTERING")
print("="*110)

def compute_icc(df, y_col, cluster_col, label):
    """
    compute intra-cluster correlation coefficient
    ICC near 0 = no clustering needed
    ICC > 0.05 = clustering probably needed
    ICC > 0.10 = clustering definitely needed
    """
    # remove missing values
    df_clean = df[[y_col, cluster_col]].dropna()
    
    # grand mean
    grand_mean = df_clean[y_col].mean()
    
    # between-cluster variance
    cluster_means = df_clean.groupby(cluster_col)[y_col].mean()
    n_clusters = len(cluster_means)
    
    # observations per cluster
    cluster_sizes = df_clean.groupby(cluster_col).size()
    
    # total variance
    ss_total = ((df_clean[y_col] - grand_mean) ** 2).sum()
    
    # between-cluster sum of squares
    ss_between = sum(cluster_sizes * ((cluster_means - grand_mean) ** 2))
    
    # within-cluster sum of squares
    ss_within = ss_total - ss_between
    
    # degrees of freedom
    df_between = n_clusters - 1
    df_within = len(df_clean) - n_clusters
    
    # mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # average cluster size (for ICC calculation)
    n_bar = len(df_clean) / n_clusters
    
    # ICC formula (one-way random effects)
    icc = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within)
    
    print(f"\n{label}:")
    print(f"  ICC (Intra-cluster correlation): {icc:.4f}")
    print(f"  Interpretation:")
    if icc < 0:
        print(f"    Negative ICC - unusual, may indicate model misspecification")
    elif icc < 0.05:
        print(f"    Small - clustering may not be critical")
    elif icc < 0.10:
        print(f"    Moderate - clustering is advisable")
    else:
        print(f"    Large - clustering is necessary")
    
    print(f"  Variance decomposition:")
    print(f"    Between-cluster variance: {ms_between:.6f}")
    print(f"    Within-cluster variance: {ms_within:.6f}")
    print(f"    Proportion of variance between clusters: {100 * icc:.2f}%")
    
    return icc

# compute ICC for poll_id clustering
print("\nICC for POLL_ID clustering:")
icc_swing_poll = compute_icc(reg_state_swing, 'A', 'poll_id', 'Swing States (poll_id)')
icc_all_poll = compute_icc(reg_state, 'A', 'poll_id', 'All States (poll_id)')
icc_national_poll = compute_icc(reg_national, 'A', 'poll_id', 'National (poll_id)')

# compute ICC for pollster clustering
print("\n" + "-"*110)
print("\nICC for POLLSTER clustering:")
icc_swing_pollster = compute_icc(reg_state_swing, 'A', 'pollster', 'Swing States (pollster)')
icc_all_pollster = compute_icc(reg_state, 'A', 'pollster', 'All States (pollster)')
icc_national_pollster = compute_icc(reg_national, 'A', 'pollster', 'National (pollster)')


########################################################################################
#################### COMPARISON: CLUSTERED VS NON-CLUSTERED SEs ########################
########################################################################################

print("\n" + "="*110)
print("PART 4: STANDARD ERROR COMPARISON - CLUSTERED VS NON-CLUSTERED")
print("="*110)

def compare_clustering(df, y_col, x_cols, cluster_col, label):
    """run regression with and without clustering to show the difference"""
    
    # prepare data
    df_reg = df[x_cols + [y_col, cluster_col]].dropna()
    X = sm.add_constant(df_reg[x_cols], has_constant='add')
    y = df_reg[y_col]
    
    # non-clustered (standard OLS)
    model = sm.OLS(y, X)
    result_ols = model.fit()
    
    # clustered
    result_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_col]})
    
    # compare
    print(f"\n{label}:")
    print(f"{'Variable':<25} {'OLS SE':>12} {'Clustered SE':>15} {'Ratio':>10} {'% Increase':>12}")
    print("-" * 74)
    
    for var in x_cols:
        if var in result_ols.params.index:
            se_ols = result_ols.bse[var]
            se_cluster = result_cluster.bse[var]
            ratio = se_cluster / se_ols
            pct_increase = 100 * (se_cluster - se_ols) / se_ols
            
            print(f"{var:<25} {se_ols:>12.6f} {se_cluster:>15.6f} {ratio:>10.3f} {pct_increase:>11.1f}%")
    
    print(f"\n  Interpretation:")
    print(f"    Ratio > 1: Clustering increases SEs (accounting for within-cluster correlation)")
    print(f"    Larger ratios = more important to cluster")
    print(f"    If all ratios approx 1.0, clustering doesn't matter much")
    
    return result_ols, result_cluster

# compare for poll_id clustering
print("\nCLUSTERING BY POLL_ID:")
swing_ols_poll, swing_cluster_poll = compare_clustering(
    reg_state_swing, 'A', state_x_vars, 'poll_id', 'Swing States (poll_id)')

national_ols_poll, national_cluster_poll = compare_clustering(
    reg_national, 'A', national_x_vars, 'poll_id', 'National (poll_id)')

# compare for pollster clustering
print("\n" + "-"*110)
print("\nCLUSTERING BY POLLSTER:")
swing_ols_pollster, swing_cluster_pollster = compare_clustering(
    reg_state_swing, 'A', state_x_vars, 'pollster', 'Swing States (pollster)')

national_ols_pollster, national_cluster_pollster = compare_clustering(
    reg_national, 'A', national_x_vars, 'pollster', 'National (pollster)')


########################################################################################
#################### DESIGN EFFECT FROM CLUSTERING #####################################
########################################################################################

print("\n" + "="*110)
print("PART 5: DESIGN EFFECT (DEFF) FROM CLUSTERING")
print("="*110)

def compute_design_effect(result_cluster, result_ols, label):
    """
    Design effect = (Var(clustered) / Var(OLS))
    DEFF > 1 means clustering increases variance (good - accounting for correlation)
    DEFF >> 1 means strong clustering effects
    """
    print(f"\n{label}:")
    print(f"{'Variable':<25} {'DEFF':>10} {'Interpretation':>30}")
    print("-" * 65)
    
    for var in result_ols.params.index:
        if var.lower() not in ('const', 'intercept'):
            var_cluster = result_cluster.bse[var] ** 2
            var_ols = result_ols.bse[var] ** 2
            deff = var_cluster / var_ols
            
            if deff < 1.1:
                interp = "Minimal clustering effect"
            elif deff < 1.5:
                interp = "Moderate clustering effect"
            else:
                interp = "Strong clustering effect"
            
            print(f"{var:<25} {deff:>10.3f} {interp:>30}")

# compute design effects for poll_id
print("\nDESIGN EFFECTS - POLL_ID CLUSTERING:")
compute_design_effect(swing_cluster_poll, swing_ols_poll, 'Swing States (poll_id)')
compute_design_effect(national_cluster_poll, national_ols_poll, 'National (poll_id)')

# compute design effects for pollster
print("\n" + "-"*110)
print("\nDESIGN EFFECTS - POLLSTER CLUSTERING:")
compute_design_effect(swing_cluster_pollster, swing_ols_pollster, 'Swing States (pollster)')
compute_design_effect(national_cluster_pollster, national_ols_pollster, 'National (pollster)')


########################################################################################
#################### CROSS-TABULATION: POLLS PER POLLSTER ##############################
########################################################################################

print("\n" + "="*110)
print("PART 6: CROSS-TABULATION - POLLS NESTED WITHIN POLLSTERS")
print("="*110)

def analyze_nesting(df, label):
    """analyze how polls are nested within pollsters"""
    
    # count polls per pollster
    polls_per_pollster = df.groupby('pollster')['poll_id'].nunique()
    
    # count questions per poll
    questions_per_poll = df.groupby('poll_id').size()
    
    print(f"\n{label}:")
    print(f"  Total pollsters: {len(polls_per_pollster)}")
    print(f"  Total polls: {len(questions_per_poll)}")
    print(f"  Total questions: {len(df)}")
    print(f"\n  Polls per pollster:")
    print(f"    Mean: {polls_per_pollster.mean():.2f}")
    print(f"    Median: {polls_per_pollster.median():.1f}")
    print(f"    Min: {polls_per_pollster.min()}")
    print(f"    Max: {polls_per_pollster.max()}")
    
    print(f"\n  Top 10 pollsters by number of polls:")
    print(f"    {'Pollster':<40} {'N polls':>10} {'N questions':>15}")
    print("    " + "-" * 65)
    
    top_pollsters = polls_per_pollster.nlargest(10)
    for pollster, n_polls in top_pollsters.items():
        n_questions = len(df[df['pollster'] == pollster])
        print(f"    {str(pollster)[:40]:<40} {n_polls:>10} {n_questions:>15}")
    
    # check if nesting is perfect (each poll belongs to exactly one pollster)
    poll_pollster_counts = df.groupby('poll_id')['pollster'].nunique()
    if (poll_pollster_counts == 1).all():
        print(f"\n  Nesting structure: Perfect (each poll belongs to exactly one pollster)")
    else:
        print(f"\n  WARNING: Imperfect nesting - some polls associated with multiple pollsters")
        print(f"    Polls with >1 pollster: {(poll_pollster_counts > 1).sum()}")

analyze_nesting(reg_state_swing, 'Swing States')
analyze_nesting(reg_state, 'All States')
analyze_nesting(reg_national, 'National')


########################################################################################
#################### SUMMARY AND RECOMMENDATIONS #######################################
########################################################################################

print("\n" + "="*110)
print("SUMMARY AND RECOMMENDATIONS")
print("="*110)

print("\n1. CLUSTER SIZE SUMMARY:")
print(f"   Poll_id clusters:")
print(f"     Swing: {len(swing_cluster_sizes)} polls, avg {swing_cluster_sizes.mean():.1f} questions/poll")
print(f"     All States: {len(all_cluster_sizes)} polls, avg {all_cluster_sizes.mean():.1f} questions/poll")
print(f"     National: {len(national_cluster_sizes)} polls, avg {national_cluster_sizes.mean():.1f} questions/poll")
print(f"\n   Pollster clusters:")
print(f"     Swing: {len(swing_pollster_sizes)} pollsters, avg {swing_pollster_sizes.mean():.1f} questions/pollster")
print(f"     All States: {len(all_pollster_sizes)} pollsters, avg {all_pollster_sizes.mean():.1f} questions/pollster")
print(f"     National: {len(national_pollster_sizes)} pollsters, avg {national_pollster_sizes.mean():.1f} questions/pollster")

print("\n2. INTRA-CLUSTER CORRELATION (ICC):")
print(f"   Poll_id clustering:")
print(f"     Swing: {icc_swing_poll:.4f} - {'Critical' if icc_swing_poll > 0.10 else 'Advisable' if icc_swing_poll > 0.05 else 'Modest'}")
print(f"     All States: {icc_all_poll:.4f} - {'Critical' if icc_all_poll > 0.10 else 'Advisable' if icc_all_poll > 0.05 else 'Modest'}")
print(f"     National: {icc_national_poll:.4f} - {'Critical' if icc_national_poll > 0.10 else 'Advisable' if icc_national_poll > 0.05 else 'Modest'}")
print(f"\n   Pollster clustering:")
print(f"     Swing: {icc_swing_pollster:.4f} - {'Critical' if icc_swing_pollster > 0.10 else 'Advisable' if icc_swing_pollster > 0.05 else 'Modest'}")
print(f"     All States: {icc_all_pollster:.4f} - {'Critical' if icc_all_pollster > 0.10 else 'Advisable' if icc_all_pollster > 0.05 else 'Modest'}")
print(f"     National: {icc_national_pollster:.4f} - {'Critical' if icc_national_pollster > 0.10 else 'Advisable' if icc_national_pollster > 0.05 else 'Modest'}")

print("\n3. RECOMMENDATION:")
if icc_swing_poll > 0.05 and icc_swing_pollster > 0.05:
    print("   Two-way clustering (poll_id AND pollster) is RECOMMENDED")
    print("   - Both poll-level and pollster-level ICCs indicate meaningful clustering")
elif icc_swing_poll > 0.05:
    print("   One-way clustering by poll_id is sufficient")
    print("   - Poll-level ICC indicates clustering needed")
    print("   - Pollster-level ICC is modest")
elif icc_swing_pollster > 0.05:
    print("   One-way clustering by pollster is sufficient")
    print("   - Pollster-level ICC indicates clustering needed")
    print("   - Poll-level ICC is modest")
else:
    print("   Clustering may not be critical (both ICCs < 0.05)")
    print("   - However, clustering is good practice and should still be used")


# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("Clustering diagnostics complete — see output/clustering_diagnostics_log.txt")