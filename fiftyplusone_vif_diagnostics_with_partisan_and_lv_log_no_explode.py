import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# redirect all print output to a log file
log_file = open('output/fiftyplusone_vif_diagnostics_with_partisan_and_lv_log_no_explode.txt', 'w')
sys.stdout = log_file

print("="*110)
print("MULTICOLLINEARITY DIAGNOSTICS (VIF), WITH PARTISAN AND LV CONTROL (MODE INDICATORS - NO EXPLOSION)")
print("source data: data/harris_trump_mode_indicators_with_partisan_and_lv_regression_no_explode.csv")
print("="*110)

########################################################################################
#################### LOAD DATA #########################################################
########################################################################################

# load mode-indicator dataset (no explosion) saved by main analysis file
reg_df_original = pd.read_csv('data/harris_trump_mode_indicators_with_partisan_and_lv_regression_no_explode.csv')

reg_df_original['end_date']   = pd.to_datetime(reg_df_original['end_date'])
reg_df_original['start_date'] = pd.to_datetime(reg_df_original['start_date'])

# define time windows and swing states
time_windows = [107, 90, 60, 30, 7]
swing_states = ['arizona', 'georgia', 'michigan', 'nevada',
                'north carolina', 'pennsylvania', 'wisconsin']
reference_mode = 'Live Phone'

print(f"\ndataset size: {len(reg_df_original)}")

########################################################################################
#################### DEFINE VARIABLE LISTS #############################################
########################################################################################

# population dummies (dynamically read from saved dataset)
pop_vars = [col for col in reg_df_original.columns if col.startswith('pop_')]

# mode indicator variables (binary, one per mode type; Live Phone excluded as reference)
mode_vars = sorted([col for col in reg_df_original.columns if col.startswith('mode_')
                    and col != 'mode_Live_Phone'])

print(f"\npopulation dummies: {pop_vars}")
print(f"mode indicators:    {mode_vars}")
print(f"reference mode:     {reference_mode} (mode_Live_Phone excluded)")

# base variable lists
time_vars     = ['duration_days', 'days_before_election', 'partisan_flag']
state_vars    = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk']

# variable sets — mode vars passed in full; compute_vif filters to active ones per sample
state_x_vars_no_mode      = time_vars + state_vars + pop_vars
national_x_vars_no_mode   = time_vars + national_vars + pop_vars
state_x_vars_with_mode    = time_vars + state_vars + mode_vars + pop_vars
national_x_vars_with_mode = time_vars + national_vars + mode_vars + pop_vars

########################################################################################
#################### SPLIT DATASETS ####################################################
########################################################################################

reg_state_original       = reg_df_original[reg_df_original['poll_level'] == 'state'].copy()
reg_national_original    = reg_df_original[reg_df_original['poll_level'] == 'national'].copy()
reg_state_swing_original = reg_state_original[reg_state_original['state'].isin(swing_states)].copy()

print(f"\nsample sizes:")
print(f"  swing states: {len(reg_state_swing_original)}")
print(f"  all states:   {len(reg_state_original)}")
print(f"  national:     {len(reg_national_original)}")

########################################################################################
#################### VIF FUNCTION ######################################################
########################################################################################

def compute_vif(df, x_cols, label):
    """
    compute VIF for each variable in x_cols.
    mode indicator variables (starting with 'mode_') are automatically filtered
    to only those with at least one observation in df, so each call uses the
    correct active set for that specific sample/window combination.
    """

    # separate mode vars from non-mode vars
    non_mode_x  = [v for v in x_cols if not v.startswith('mode_')]
    all_mode_x  = [v for v in x_cols if v.startswith('mode_')]

    # keep only mode indicators with at least one observation in this specific sample
    active_mode_x = [v for v in all_mode_x if v in df.columns and df[v].sum() > 0]
    dropped_modes = [v for v in all_mode_x if v not in active_mode_x]

    if dropped_modes:
        print(f"\n  [{label}] dropping zero-observation mode indicators: {dropped_modes}")

    # check all non-mode vars exist
    missing = [v for v in non_mode_x if v not in df.columns]
    if missing:
        print(f"  [{label}] non-mode variables not found, skipped: {missing}")

    available = [v for v in non_mode_x if v not in missing] + active_mode_x
    df_clean  = df[available].dropna()

    if len(df_clean) < len(available) + 1:
        print(f"\n{label}: insufficient data (n={len(df_clean)})")
        return None

    X    = sm.add_constant(df_clean)
    rank = np.linalg.matrix_rank(X.values)

    if rank < X.shape[1]:
        print(f"\n{label} (N={len(df_clean)}):")
        print(f"  NOTE: perfect collinearity detected even after dropping zero-observation modes")
        print(f"  matrix rank: {rank}, expected: {X.shape[1]}")
        print(f"  remaining variables: {available}")
        return None

    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(np.nan if (np.isinf(vif) or np.isnan(vif)) else vif)
        except Exception:
            vif_values.append(np.nan)

    vif_data = pd.DataFrame({"Variable": X.columns, "VIF": vif_values})
    vif_data  = vif_data[vif_data["Variable"] != "const"]

    print(f"\n{label} (N={len(df_clean)}):")
    if dropped_modes:
        print(f"  active mode indicators ({len(active_mode_x)}): {active_mode_x}")
    print(vif_data.to_string(index=False))

    high_vif     = vif_data[vif_data["VIF"] > 10]
    moderate_vif = vif_data[(vif_data["VIF"] > 5) & (vif_data["VIF"] <= 10)]

    if not high_vif.empty:
        print(f"  WARNING: {len(high_vif)} variable(s) with VIF > 10 (problematic)")
    if not moderate_vif.empty:
        print(f"  WARNING: {len(moderate_vif)} variable(s) with VIF > 5 (caution warranted)")

    return vif_data

########################################################################################
#################### RUN VIF DIAGNOSTICS ###############################################
########################################################################################

vif_issues = []  # collect any concerns for final summary

# ── NO MODE ──────────────────────────────────────────────────────────────────────────

print("\n" + "="*110)
print("WITHOUT MODE VARIABLES")
print("="*110)

for df, label in [
    (reg_state_swing_original, "swing state-level polls (no mode)"),
    (reg_state_original,       "state-level polls (no mode)"),
    (reg_national_original,    "national polls (no mode)"),
]:
    x_vars = national_x_vars_no_mode if 'national' in label else state_x_vars_no_mode
    result = compute_vif(df, x_vars, label)
    if result is not None and (result["VIF"] > 5).any():
        vif_issues.append(f"{label}: VIF concerns")

# ── WITH MODE INDICATORS ─────────────────────────────────────────────────────────────

print("\n" + "="*110)
print("WITH MODE INDICATOR VARIABLES (NO EXPLOSION)")
print(f"reference mode: {reference_mode}")
print("each poll retains its original row; multi-mode polls get 1 on every active mode indicator")
print("zero-observation mode indicators dropped automatically per sample")
print("="*110)

for df, label in [
    (reg_state_swing_original, "swing state-level polls (with mode indicators)"),
    (reg_state_original,       "state-level polls (with mode indicators)"),
    (reg_national_original,    "national polls (with mode indicators)"),
]:
    x_vars = national_x_vars_with_mode if 'national' in label else state_x_vars_with_mode
    result = compute_vif(df, x_vars, label)
    if result is not None and (result["VIF"] > 5).any():
        vif_issues.append(f"{label}: VIF concerns")

# ── TIME WINDOWS — NO MODE ────────────────────────────────────────────────────────────

print("\n" + "="*110)
print("TIME WINDOW ANALYSIS (NO MODE)")
print("="*110)

for window in time_windows:
    print(f"\n{'-'*110}")
    print(f"TIME WINDOW: {window} days before election")
    print(f"{'-'*110}")

    swing_w    = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window]
    state_w    = reg_state_original[reg_state_original['days_before_election'] <= window]
    national_w = reg_national_original[reg_national_original['days_before_election'] <= window]

    for df, label, x_vars in [
        (swing_w,    f"swing states ({window}d window)",  state_x_vars_no_mode),
        (state_w,    f"all states ({window}d window)",    state_x_vars_no_mode),
        (national_w, f"national ({window}d window)",      national_x_vars_no_mode),
    ]:
        if len(df) >= 10:
            result = compute_vif(df, x_vars, label)
            if result is not None and (result["VIF"] > 5).any():
                vif_issues.append(f"{label} (no mode): VIF concerns")
        else:
            print(f"\n{label}: insufficient sample (n={len(df)})")

# ── TIME WINDOWS — WITH MODE ──────────────────────────────────────────────────────────

print("\n" + "="*110)
print("TIME WINDOW ANALYSIS (WITH MODE INDICATORS)")
print("zero-observation mode indicators dropped automatically per window/sample")
print("="*110)

for window in time_windows:
    print(f"\n{'-'*110}")
    print(f"TIME WINDOW: {window} days before election")
    print(f"{'-'*110}")

    swing_w    = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window]
    state_w    = reg_state_original[reg_state_original['days_before_election'] <= window]
    national_w = reg_national_original[reg_national_original['days_before_election'] <= window]

    for df, label, x_vars in [
        (swing_w,    f"swing states ({window}d window, with mode)",  state_x_vars_with_mode),
        (state_w,    f"all states ({window}d window, with mode)",    state_x_vars_with_mode),
        (national_w, f"national ({window}d window, with mode)",      national_x_vars_with_mode),
    ]:
        if len(df) >= 10:
            result = compute_vif(df, x_vars, label)
            if result is not None and (result["VIF"] > 5).any():
                vif_issues.append(f"{label}: VIF concerns")
        else:
            print(f"\n{label}: insufficient sample (n={len(df)})")

########################################################################################
#################### FINAL SUMMARY #####################################################
########################################################################################

print("\n" + "="*110)
print("VIF DIAGNOSTICS SUMMARY")
print("="*110)

if vif_issues:
    print("\nWARNING: MULTICOLLINEARITY CONCERNS DETECTED:")
    print("-" * 110)
    for issue in vif_issues:
        print(f"  * {issue}")
    print("\nreview variables with VIF > 5")
else:
    print("\nno serious multicollinearity issues detected across all specifications")
    print("all VIF values < 5 in all samples and time windows")

print("\nnote: VIF > 10 suggests problematic multicollinearity")
print("      VIF > 5 warrants caution")
print("      mode indicator strategy: multi-mode polls get 1 on all active indicators;")
print("      higher correlation between mode dummies is expected but VIF should still be acceptable")
print("      zero-observation mode indicators are dropped per sample before VIF calculation")
print("="*110 + "\n")


# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("VIF diagnostics complete — see output/fiftyplusone_vif_diagnostics_with_partisan_and_lv_log_no_explode.txt")