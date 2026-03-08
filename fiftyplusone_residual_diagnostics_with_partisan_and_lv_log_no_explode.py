import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# Redirect output to log file
log_file = open('output/fiftyplusone_residual_diagnostics_with_partisan_and_lv_log_no_explode.txt', 'w')
sys.stdout = log_file

print("="*110)
print("COMPREHENSIVE RESIDUAL DIAGNOSTICS, WITH PARTISAN AND LV CONTROL (MODE INDICATORS - NO EXPLOSION)")
print("="*110)

########################################################################################
#################### LOAD DATA #########################################################
########################################################################################

# Load the mode-indicator dataset (no explosion)
reg_df_original = pd.read_csv('data/harris_trump_mode_indicators_with_partisan_and_lv_regression_no_explode.csv')
reg_df_original['end_date'] = pd.to_datetime(reg_df_original['end_date'])
reg_df_original['start_date'] = pd.to_datetime(reg_df_original['start_date'])

# Variable lists
time_vars  = ['duration_days', 'days_before_election', 'partisan_flag']
state_vars = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk']
pop_vars = [col for col in reg_df_original.columns if col.startswith('pop_')]

state_x_vars_no_mode   = time_vars + state_vars + pop_vars
national_x_vars_no_mode = time_vars + national_vars + pop_vars

# Identify mode indicator variables (binary indicators, no explosion needed)
# Live Phone is the reference category and was dropped before saving
mode_vars = sorted([col for col in reg_df_original.columns if col.startswith('mode_')
                    and col != 'mode_Live_Phone'])

reference_mode = 'Live Phone'

state_x_vars_with_mode   = time_vars + state_vars + mode_vars + pop_vars
national_x_vars_with_mode = time_vars + national_vars + mode_vars + pop_vars

# Create subsets
reg_state_original    = reg_df_original[reg_df_original['poll_level'] == 'state'].copy()
reg_national_original = reg_df_original[reg_df_original['poll_level'] == 'national'].copy()

swing_states = ['arizona', 'georgia', 'michigan', 'nevada',
                'north carolina', 'pennsylvania', 'wisconsin']
reg_state_swing_original = reg_state_original[reg_state_original['state'].isin(swing_states)].copy()

print(f"\ndata loaded (mode-indicator / no-explosion dataset):")
print(f"  swing states: {len(reg_state_swing_original)}")
print(f"  all states:   {len(reg_state_original)}")
print(f"  national:     {len(reg_national_original)}")
print(f"\npopulation dummies: {pop_vars}")
print(f"mode indicators:    {mode_vars}")
print(f"reference mode:     {reference_mode}")


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

    if abs(residuals.mean()) < 0.001:
        print(f"     mean approx 0 (good - no systematic bias)")
    else:
        print(f"      mean far from 0 (potential model bias)")

    skew_val = stats.skew(residuals)
    if abs(skew_val) < 0.5:
        print(f"     skewness {skew_val:.2f} (good - approximately symmetric)")
    elif abs(skew_val) < 1.0:
        print(f"      skewness {skew_val:.2f} (moderate - acceptable)")
    else:
        print(f"      skewness {skew_val:.2f} (high - check for outliers/events)")

    if outlier_pct < 10:
        print(f"     outliers {outlier_pct:.1f}% (good - within typical range)")
    elif outlier_pct < 15:
        print(f"      outliers {outlier_pct:.1f}% (moderate - acceptable)")
    else:
        print(f"      outliers {outlier_pct:.1f}% (high - investigate influential polls)")

    if extreme_pct < 2:
        print(f"     extreme outliers {extreme_pct:.1f}% (good)")
    elif extreme_pct < 5:
        print(f"      extreme outliers {extreme_pct:.1f}% (moderate)")
    else:
        print(f"      extreme outliers {extreme_pct:.1f}% (high - check specific polls)")

    print(f"    note: normality tests often reject with large n due to minor deviations")
    print(f"          visual inspection (q-q plots) more informative than formal tests")
    print(f"          clustered standard errors robust to non-normality")


def run_regression_diagnostics(df, x_vars, label):
    """run OLS and compute residual diagnostics; returns fitted result"""

    print(f"\n" + "="*110)
    print(f"{label}")
    print("="*110)

    # only keep x_vars that actually exist in this dataframe
    available_x = [v for v in x_vars if v in df.columns]
    missing_x   = [v for v in x_vars if v not in df.columns]
    if missing_x:
        print(f"  note: variables not found in df, skipped: {missing_x}")

    df_reg = df[available_x + ['A', 'poll_id']].dropna()
    X = sm.add_constant(df_reg[available_x], has_constant='add')
    y = df_reg['A']

    model  = sm.OLS(y, X)
    result = model.fit()

    compute_residual_diagnostics(result.resid, label)

    return result


########################################################################################
#################### PART 1: MAIN SPECIFICATIONS (NO MODE) ############################
########################################################################################

print("\n" + "="*110)
print("PART 1: MAIN SPECIFICATIONS (NON-MODE)")
print("="*110)

result_swing    = run_regression_diagnostics(reg_state_swing_original, state_x_vars_no_mode,   "swing states (non-mode)")
result_all      = run_regression_diagnostics(reg_state_original,       state_x_vars_no_mode,   "all states (non-mode)")
result_national = run_regression_diagnostics(reg_national_original,    national_x_vars_no_mode, "national (non-mode)")


########################################################################################
#################### PART 2: TIME WINDOW REGRESSIONS (NO MODE) ########################
########################################################################################

print("\n" + "="*110)
print("PART 2: TIME WINDOW REGRESSIONS (NON-MODE)")
print("="*110)

time_windows = [107, 90, 60, 30, 7]

for window in time_windows:
    print(f"\n{'='*110}")
    print(f"TIME WINDOW: {window} DAYS BEFORE ELECTION")
    print(f"{'='*110}")

    df_window = reg_df_original[reg_df_original['days_before_election'] <= window].copy()

    swing_window    = df_window[(df_window['poll_level'] == 'state') & (df_window['state'].isin(swing_states))]
    state_window    = df_window[df_window['poll_level'] == 'state']
    national_window = df_window[df_window['poll_level'] == 'national']

    if len(swing_window) > 50:
        run_regression_diagnostics(swing_window,    state_x_vars_no_mode,    f"swing states - {window} days (non-mode)")
    else:
        print(f"\nswing states - {window} days: insufficient sample (n={len(swing_window)})")

    if len(state_window) > 50:
        run_regression_diagnostics(state_window,    state_x_vars_no_mode,    f"all states - {window} days (non-mode)")
    else:
        print(f"\nall states - {window} days: insufficient sample (n={len(state_window)})")

    if len(national_window) > 50:
        run_regression_diagnostics(national_window, national_x_vars_no_mode, f"national - {window} days (non-mode)")
    else:
        print(f"\nnational - {window} days: insufficient sample (n={len(national_window)})")


########################################################################################
#################### PART 3: MODE REGRESSIONS (INDICATOR STRATEGY, NO EXPLOSION) ######
########################################################################################

print("\n" + "="*110)
print("PART 3: MODE REGRESSIONS (INDICATOR STRATEGY - NO EXPLOSION)")
print(f"reference mode: {reference_mode} (mode_Live_Phone excluded from regressors)")
print("="*110)

print(f"\nmode indicator variables used: {mode_vars}")
print("each poll retains its original row; multi-mode polls get a 1 on every active mode indicator")

# --- main specifications with mode ---

result_swing_mode    = run_regression_diagnostics(reg_state_swing_original, state_x_vars_with_mode,    "swing states (with mode indicators)")
result_all_mode      = run_regression_diagnostics(reg_state_original,       state_x_vars_with_mode,    "all states (with mode indicators)")
result_national_mode = run_regression_diagnostics(reg_national_original,    national_x_vars_with_mode, "national (with mode indicators)")

# --- time window regressions with mode ---

print("\n" + "="*110)
print("PART 3b: TIME WINDOW REGRESSIONS (WITH MODE INDICATORS)")
print("="*110)

for window in time_windows:
    print(f"\n{'='*110}")
    print(f"TIME WINDOW: {window} DAYS BEFORE ELECTION (WITH MODE)")
    print(f"{'='*110}")

    df_window = reg_df_original[reg_df_original['days_before_election'] <= window].copy()

    swing_window    = df_window[(df_window['poll_level'] == 'state') & (df_window['state'].isin(swing_states))]
    state_window    = df_window[df_window['poll_level'] == 'state']
    national_window = df_window[df_window['poll_level'] == 'national']

    if len(swing_window) > 50:
        run_regression_diagnostics(swing_window,    state_x_vars_with_mode,    f"swing states - {window} days (with mode)")
    else:
        print(f"\nswing states - {window} days: insufficient sample (n={len(swing_window)})")

    if len(state_window) > 50:
        run_regression_diagnostics(state_window,    state_x_vars_with_mode,    f"all states - {window} days (with mode)")
    else:
        print(f"\nall states - {window} days: insufficient sample (n={len(state_window)})")

    if len(national_window) > 50:
        run_regression_diagnostics(national_window, national_x_vars_with_mode, f"national - {window} days (with mode)")
    else:
        print(f"\nnational - {window} days: insufficient sample (n={len(national_window)})")


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
print("  • mode indicator strategy: multi-mode polls get 1 on all active indicators;")
print("    coefficients capture marginal contribution of each mode relative to Live Phone")

print("\n" + "="*110 + "\n")

# close log
log_file.close()
sys.stdout = sys.__stdout__


########################################################################################
#################### VISUAL DIAGNOSTICS (MAIN SPECS ONLY) ##############################
########################################################################################

print("generating visual diagnostics...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Residual Diagnostics (Mode Indicator Strategy, No Explosion)', fontsize=14, y=1.01)

specs = [
    (result_swing,    'swing states (no mode)'),
    (result_all,      'all states (no mode)'),
    (result_national, 'national (no mode)'),
]

for row, (result, label) in enumerate(specs):
    residuals = result.resid
    fitted    = result.fittedvalues

    # residuals vs fitted
    axes[row, 0].scatter(fitted, residuals, alpha=0.3, s=10)
    axes[row, 0].axhline(0, color='red', linestyle='--')
    axes[row, 0].set_xlabel('fitted values')
    axes[row, 0].set_ylabel('residuals')
    axes[row, 0].set_title(f'{label}: residuals vs fitted')

    # histogram
    axes[row, 1].hist(residuals, bins=30, edgecolor='black')
    axes[row, 1].axvline(0, color='red', linestyle='--')
    axes[row, 1].set_xlabel('residuals')
    axes[row, 1].set_title(f'{label}: residual distribution')

    # q-q plot
    stats.probplot(residuals, dist="norm", plot=axes[row, 2])
    axes[row, 2].set_title(f'{label}: q-q plot')

plt.tight_layout()
plt.savefig("figures/residual_diagnostics_main_with_partisan_and_lv_no_explode.png", dpi=300, bbox_inches='tight')
plt.close()

print("visual diagnostics saved to: figures/residual_diagnostics_main_with_partisan_and_lv_no_explode.png")
print("numerical diagnostics saved to: output/fiftyplusone_residual_diagnostics_with_partisan_and_lv_log_no_explode.txt")