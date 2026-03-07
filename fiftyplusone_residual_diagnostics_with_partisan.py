import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# Redirect output to log file
log_file = open('output/fiftyplusone_residual_diagnostics_with_partisan_log.txt', 'w')
sys.stdout = log_file

print("="*110)
print("COMPREHENSIVE RESIDUAL DIAGNOSTICS, WITH PARTISAN CONTROL")
print("="*110)

########################################################################################
#################### LOAD DATA #########################################################
########################################################################################

# Load the ORIGINAL (un-exploded) dataset - before mode splitting
reg_df_original = pd.read_csv('data/harris_trump_regression_original_with_partisan.csv')
reg_df_original['end_date'] = pd.to_datetime(reg_df_original['end_date'])
reg_df_original['start_date'] = pd.to_datetime(reg_df_original['start_date'])

# Variable lists
time_vars = ['duration_days', 'days_before_election','partisan_flag']
state_vars = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk']  # no abs_margin (constant), no turnout_pct (state-level)

state_x_vars_no_mode = time_vars + state_vars
national_x_vars_no_mode = time_vars + national_vars

# Create subsets
reg_state_original = reg_df_original[reg_df_original['poll_level'] == 'state'].copy()
reg_national_original = reg_df_original[reg_df_original['poll_level'] == 'national'].copy()

swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 
                'north carolina', 'pennsylvania', 'wisconsin']
reg_state_swing_original = reg_state_original[reg_state_original['state'].isin(swing_states)].copy()

print(f"\ndata loaded (un-exploded dataset):")
print(f"  swing states: {len(reg_state_swing_original)}")
print(f"  all states: {len(reg_state_original)}")
print(f"  national: {len(reg_national_original)}")


########################################################################################
#################### DIAGNOSTIC FUNCTIONS ##############################################
########################################################################################

def compute_residual_diagnostics(residuals, label):
    """compute numerical diagnostics for residuals"""
    
    print(f"\n{label}:")
    print(f"  n observations: {len(residuals)}")
    print(f"  mean: {residuals.mean():.6f}")
    print(f"  std dev: {residuals.std():.6f}")
    print(f"  min: {residuals.min():.6f}")
    print(f"  max: {residuals.max():.6f}")
    print(f"  skewness: {stats.skew(residuals):.4f}")
    print(f"  kurtosis: {stats.kurtosis(residuals):.4f}")
    
    # normality tests
    shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) <= 5000 else (np.nan, np.nan)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    
    print(f"\n  normality tests:")
    if not np.isnan(shapiro_p):
        print(f"    shapiro-wilk: w={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        if shapiro_p < 0.05:
            print(f"        reject normality (p < 0.05)")
        else:
            print(f"        cannot reject normality (p >= 0.05)")
    
    print(f"    jarque-bera: jb={jb_stat:.4f}, p={jb_p:.4f}")
    if jb_p < 0.05:
        print(f"        reject normality (p < 0.05)")
    else:
        print(f"        cannot reject normality (p >= 0.05)")
    
    # outliers
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    outliers = ((residuals < lower_fence) | (residuals > upper_fence)).sum()
    outlier_pct = 100 * outliers / len(residuals)
    
    print(f"\n  outliers (1.5xiqr rule):")
    print(f"    count: {outliers} ({outlier_pct:.1f}%)")
    print(f"    lower fence: {lower_fence:.6f}")
    print(f"    upper fence: {upper_fence:.6f}")
    
    # extreme outliers (3xIQR)
    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr
    extreme_outliers = ((residuals < extreme_lower) | (residuals > extreme_upper)).sum()
    extreme_pct = 100 * extreme_outliers / len(residuals)
    
    print(f"\n  extreme outliers (3xiqr rule):")
    print(f"    count: {extreme_outliers} ({extreme_pct:.1f}%)")
    
    # interpretation guidance
    print(f"\n  interpretation:")
    
    # mean check
    if abs(residuals.mean()) < 0.001:
        print(f"     mean approx 0 (good - no systematic bias)")
    else:
        print(f"      mean far from 0 (potential model bias)")
    
    # skewness check
    skew_val = stats.skew(residuals)
    if abs(skew_val) < 0.5:
        print(f"     skewness {skew_val:.2f} (good - approximately symmetric)")
    elif abs(skew_val) < 1.0:
        print(f"      skewness {skew_val:.2f} (moderate - acceptable)")
    else:
        print(f"      skewness {skew_val:.2f} (high - check for outliers/events)")
    
    # outliers check
    if outlier_pct < 10:
        print(f"     outliers {outlier_pct:.1f}% (good - within typical range)")
    elif outlier_pct < 15:
        print(f"      outliers {outlier_pct:.1f}% (moderate - acceptable)")
    else:
        print(f"      outliers {outlier_pct:.1f}% (high - investigate influential polls)")
    
    # extreme outliers check
    if extreme_pct < 2:
        print(f"     extreme outliers {extreme_pct:.1f}% (good)")
    elif extreme_pct < 5:
        print(f"      extreme outliers {extreme_pct:.1f}% (moderate)")
    else:
        print(f"      extreme outliers {extreme_pct:.1f}% (high - check specific polls)")
    
    # normality note
    print(f"    note: normality tests often reject with large n due to minor deviations")
    print(f"          visual inspection (q-q plots) more informative than formal tests")
    print(f"          clustered standard errors robust to non-normality")


def run_regression_diagnostics(df, x_vars, label, sample_type='state'):
    """run regression and compute diagnostics"""
    
    print(f"\n" + "="*110)
    print(f"{label}")
    print("="*110)
    
    # prepare data
    df_reg = df[x_vars + ['A', 'poll_id']].dropna()
    X = sm.add_constant(df_reg[x_vars], has_constant='add')
    y = df_reg['A']
    
    # run ols (for residual diagnostics)
    model = sm.OLS(y, X)
    result = model.fit()
    
    # compute diagnostics
    compute_residual_diagnostics(result.resid, label)
    
    return result


########################################################################################
#################### MAIN SPECIFICATIONS ###############################################
########################################################################################

print("\n" + "="*110)
print("PART 1: MAIN SPECIFICATIONS (NON-MODE)")
print("="*110)

# swing states
result_swing = run_regression_diagnostics(
    reg_state_swing_original, 
    state_x_vars_no_mode, 
    "swing states (non-mode)",
    'state'
)

# all states
result_all = run_regression_diagnostics(
    reg_state_original, 
    state_x_vars_no_mode, 
    "all states (non-mode)",
    'state'
)

# national
result_national = run_regression_diagnostics(
    reg_national_original, 
    national_x_vars_no_mode, 
    "national (non-mode)",
    'national'
)


########################################################################################
#################### TIME WINDOW REGRESSIONS ###########################################
########################################################################################

print("\n" + "="*110)
print("PART 2: TIME WINDOW REGRESSIONS")
print("="*110)

time_windows = [107, 90, 60, 30, 7]

for window in time_windows:
    print(f"\n{'='*110}")
    print(f"TIME WINDOW: {window} DAYS BEFORE ELECTION")
    print(f"{'='*110}")
    
    # filter data
    df_window = reg_df_original[reg_df_original['days_before_election'] <= window].copy()
    
    swing_window = df_window[
        (df_window['poll_level'] == 'state') & 
        (df_window['state'].isin(swing_states))
    ]
    state_window = df_window[df_window['poll_level'] == 'state']
    national_window = df_window[df_window['poll_level'] == 'national']
    
    # run diagnostics (only if sufficient sample size)
    if len(swing_window) > 50:
        result_swing_w = run_regression_diagnostics(
            swing_window,
            state_x_vars_no_mode,
            f"swing states - {window} days",
            'state'
        )
    else:
        print(f"\nswing states - {window} days: insufficient sample (n={len(swing_window)})")
    
    if len(state_window) > 50:
        result_state_w = run_regression_diagnostics(
            state_window,
            state_x_vars_no_mode,
            f"all states - {window} days",
            'state'
        )
    else:
        print(f"\nall states - {window} days: insufficient sample (n={len(state_window)})")
    
    if len(national_window) > 50:
        result_national_w = run_regression_diagnostics(
            national_window,
            national_x_vars_no_mode,
            f"national - {window} days",
            'national'
        )
    else:
        print(f"\nnational - {window} days: insufficient sample (n={len(national_window)})")


########################################################################################
#################### MODE REGRESSIONS ##################################################
########################################################################################

print("\n" + "="*110)
print("PART 3: MODE REGRESSIONS")
print("="*110)

# explode mode for mode regressions
print("\nexploding mixed-mode polls for mode analysis...")

# load the exploded dataset for mode regressions
reg_df_exploded = pd.read_csv('data/harris_trump_datelimted_clustertwoway_with_partisan_regression.csv')
reg_df_exploded['end_date'] = pd.to_datetime(reg_df_exploded['end_date'])
reg_df_exploded['start_date'] = pd.to_datetime(reg_df_exploded['start_date'])

# create subsets
reg_state_exploded = reg_df_exploded[reg_df_exploded['poll_level'] == 'state'].copy()
reg_national_exploded = reg_df_exploded[reg_df_exploded['poll_level'] == 'national'].copy()
reg_state_swing_exploded = reg_state_exploded[reg_state_exploded['state'].isin(swing_states)].copy()

print(f"  exploded dataset loaded:")
print(f"    swing states: {len(reg_state_swing_exploded)}")
print(f"    all states: {len(reg_state_exploded)}")
print(f"    national: {len(reg_national_exploded)}")

# identify mode dummy columns
mode_cols = [col for col in reg_df_exploded.columns if col.startswith('mode_')]
print(f"\n  mode dummies found: {mode_cols}")

if len(mode_cols) > 0:
    # define mode variable lists (excluding reference category)
    # assume 'mode_live phone' is reference (excluded)
    mode_vars = [col for col in mode_cols if col != 'mode_live phone']
    
    state_x_vars_mode = state_x_vars_no_mode + mode_vars
    national_x_vars_mode = national_x_vars_no_mode + mode_vars
    
    print(f"  mode variables in regression: {mode_vars}")
    print(f"  reference category: live phone (excluded)")
    
    # swing states with mode
    result_swing_mode = run_regression_diagnostics(
        reg_state_swing_exploded,
        state_x_vars_mode,
        "swing states (with mode)",
        'state'
    )
    
    # all states with mode
    result_all_mode = run_regression_diagnostics(
        reg_state_exploded,
        state_x_vars_mode,
        "all states (with mode)",
        'state'
    )
    
    # national with mode
    result_national_mode = run_regression_diagnostics(
        reg_national_exploded,
        national_x_vars_mode,
        "national (with mode)",
        'national'
    )
else:
    print("    no mode dummy variables found - skipping mode regressions")


########################################################################################
#################### SUMMARY ###########################################################
########################################################################################

print("\n" + "="*110)
print("SUMMARY: RESIDUAL DIAGNOSTICS ACROSS ALL SPECIFICATIONS")
print("="*110)

print("\nkey findings to look for:")
print("\ngood signs:")
print("   mean approx 0 (no systematic bias)")
print("   residuals roughly symmetric around zero (|skewness| < 0.5)")
print("   q-q plot points roughly follow diagonal line")
print("   no clear patterns in residuals vs fitted (constant variance)")
print("   outliers <10% using 1.5xiqr rule")
print("   extreme outliers <2% using 3xiqr rule")

print("\nwarning signs:")
print("    mean far from zero (model bias)")
print("    strong skewness (>1 or <-1)")
print("    funnel shape in residuals vs fitted (heteroskedasticity)")
print("    curved pattern in q-q plot (non-normality)")
print("    >15% outliers (potential influential observations)")
print("    >5% extreme outliers (check specific polls/events)")

print("\nnot concerning:")
print("    normality tests reject (common with large n, minor deviations)")
print("    slight deviations in q-q plot tails (typical)")
print("    kurtosis in 2-4 range (acceptable)")
print("    moderate outlier rates 10-15% (still acceptable)")

print("\ninterpretation notes:")
print("  • normality tests (shapiro-wilk, jarque-bera) are very sensitive with large n")
print("  • rejecting normality doesn't invalidate clustered standard errors")
print("  • visual inspection of q-q plots more informative than formal tests")
print("  • clustered ses remain valid under non-normality")
print("  • outliers may reflect real campaign events (debates, scandals, etc.)")
print("  • extreme outliers should be investigated but not automatically removed")
print("  • heteroskedasticity handled by clustering, not a major concern")

print("\n" + "="*110 + "\n")

# close log
log_file.close()
sys.stdout = sys.__stdout__


########################################################################################
#################### VISUAL DIAGNOSTICS (MAIN SPECS ONLY) ##############################
########################################################################################

print("generating visual diagnostics...")

# create figure for main specifications
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# swing states
residuals_swing = result_swing.resid
fitted_swing = result_swing.fittedvalues

axes[0, 0].scatter(fitted_swing, residuals_swing, alpha=0.3, s=10)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('fitted values')
axes[0, 0].set_ylabel('residuals')
axes[0, 0].set_title('swing states: residuals vs fitted')

axes[0, 1].hist(residuals_swing, bins=30, edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('residuals')
axes[0, 1].set_title('swing states: residual distribution')

stats.probplot(residuals_swing, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('swing states: q-q plot')

# all states
residuals_all = result_all.resid
fitted_all = result_all.fittedvalues

axes[1, 0].scatter(fitted_all, residuals_all, alpha=0.3, s=10)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('fitted values')
axes[1, 0].set_ylabel('residuals')
axes[1, 0].set_title('all states: residuals vs fitted')

axes[1, 1].hist(residuals_all, bins=30, edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--')
axes[1, 1].set_xlabel('residuals')
axes[1, 1].set_title('all states: residual distribution')

stats.probplot(residuals_all, dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('all states: q-q plot')

# national
residuals_national = result_national.resid
fitted_national = result_national.fittedvalues

axes[2, 0].scatter(fitted_national, residuals_national, alpha=0.3, s=10)
axes[2, 0].axhline(0, color='red', linestyle='--')
axes[2, 0].set_xlabel('fitted values')
axes[2, 0].set_ylabel('residuals')
axes[2, 0].set_title('national: residuals vs fitted')

axes[2, 1].hist(residuals_national, bins=30, edgecolor='black')
axes[2, 1].axvline(0, color='red', linestyle='--')
axes[2, 1].set_xlabel('residuals')
axes[2, 1].set_title('national: residual distribution')

stats.probplot(residuals_national, dist="norm", plot=axes[2, 2])
axes[2, 2].set_title('national: q-q plot')

plt.tight_layout()
plt.savefig("figures/fiftyplusonePY/datelimited/residual_diagnostics_main_with_partisan.png", dpi=300)
plt.close()

print("visual diagnostics saved to: figures/fiftyplusonePY/datelimited/residual_diagnostics_main_with_partisan.png")
print("numerical diagnostics saved to: output/fiftyplusone_residual_diagnostics_with_partisan_log.txt")