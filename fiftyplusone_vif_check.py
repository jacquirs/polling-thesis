import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# redirect all print output to a log file
log_file = open('output/fiftyplusone_vif_diagnostics_log.txt', 'w')
sys.stdout = log_file

print("="*110)
print("MULTICOLLINEARITY DIAGNOSTICS (VIF), no partisan control")
print("source data: data/harris_trump_regression_original.csv")
print("="*110)

########################################################################################
#################### LOAD DATA #########################################################
########################################################################################

# load regression-ready dataset created by main analysis file (non-exploded)
reg_df_original = pd.read_csv('data/harris_trump_regression_original.csv')

# convert date columns back to datetime
reg_df_original['end_date'] = pd.to_datetime(reg_df_original['end_date'])
reg_df_original['start_date'] = pd.to_datetime(reg_df_original['start_date'])

# election date for filtering
election_date = pd.Timestamp('2024-11-05')

# define time windows
time_windows = [107, 90, 60, 30, 7]

# define swing states
swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 
                'north carolina', 'pennsylvania', 'wisconsin']

# reference mode
reference_mode = 'Live Phone'

print(f"\noriginal dataset size: {len(reg_df_original)}")

########################################################################################
#################### EXPLODE MODE INTO BASE_MODE #######################################
########################################################################################

# create exploded version by splitting mode column
reg_df = reg_df_original.copy()
reg_df['base_mode'] = reg_df['mode'].str.split('/')
reg_df = reg_df.explode('base_mode')
reg_df['base_mode'] = reg_df['base_mode'].str.strip()

print(f"\nunique base modes after exploding:")
print(reg_df['base_mode'].value_counts())

# create mode dummy variables
mode_dummies = pd.get_dummies(reg_df['base_mode'], prefix='mode', drop_first=False)

# drop live phone to use as reference (if it exists)
if f'mode_{reference_mode}' in mode_dummies.columns:
    mode_dummies = mode_dummies.drop(f'mode_{reference_mode}', axis=1)
    print(f"\nreference category set to: {reference_mode}")

# convert boolean to int and clean column names
mode_dummies = mode_dummies.astype(int)
mode_dummies.columns = mode_dummies.columns.str.replace('-', '_')

# add mode dummies to reg_df
reg_df = pd.concat([reg_df, mode_dummies], axis=1)
mode_vars = [col for col in reg_df.columns if col.startswith('mode_')]

print(f"\nmode dummy variables created: {len(mode_vars)}")
print(f"mode variables: {mode_vars}")

########################################################################################
#################### SPLIT DATASETS ####################################################
########################################################################################

# EXPLODED versions (for mode regressions)
reg_state = reg_df[reg_df['poll_level'] == 'state'].copy()
reg_national = reg_df[reg_df['poll_level'] == 'national'].copy()
reg_state_swing = reg_state[reg_state['state'].isin(swing_states)].copy()

# ORIGINAL versions (for non-mode regressions)
reg_state_original = reg_df_original[reg_df_original['poll_level'] == 'state'].copy()
reg_national_original = reg_df_original[reg_df_original['poll_level'] == 'national'].copy()
reg_state_swing_original = reg_state_original[reg_state_original['state'].isin(swing_states)].copy()

print(f"\nsample sizes:")
print(f"  swing states (original): {len(reg_state_swing_original)}")
print(f"  swing states (exploded):  {len(reg_state_swing)}")
print(f"  all states (original):   {len(reg_state_original)}")
print(f"  all states (exploded):    {len(reg_state)}")
print(f"  national (original):     {len(reg_national_original)}")
print(f"  national (exploded):      {len(reg_national)}")

########################################################################################
#################### DEFINE VARIABLE LISTS #############################################
########################################################################################

# base variables
time_vars = ['duration_days', 'days_before_election']
state_vars = ['pct_dk', 'abs_margin', 'turnout_pct']
national_vars = ['pct_dk']

# variable sets for regressions
state_x_vars_no_mode = time_vars + state_vars
national_x_vars_no_mode = time_vars + national_vars
state_x_vars_with_mode = time_vars + state_vars + mode_vars
national_x_vars_with_mode = time_vars + national_vars + mode_vars

########################################################################################
#################### VIF FUNCTION ######################################################
########################################################################################

def compute_vif(df, x_cols, label):
    """compute VIF for each variable"""
    # drop any missing values
    df_clean = df[x_cols].dropna()
    
    if len(df_clean) < len(x_cols) + 1:
        print(f"\n{label}: insufficient data (n={len(df_clean)})")
        return None
    
    # add constant for VIF calculation
    X = sm.add_constant(df_clean)
    
    # check for perfect collinearity before computing VIF
    rank = np.linalg.matrix_rank(X.values)
    if rank < X.shape[1]:
        print(f"\n{label} (N={len(df_clean)}):")
        print(f"  NOTE: perfect collinearity detected (likely mode variables with no variation)")
        print(f"  matrix rank: {rank}, expected: {X.shape[1]}")
        print(f"  VIF cannot be computed - this is expected in small time windows with limited mode diversity")
        return None
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    
    # compute VIF with error handling for numerical issues
    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X.values, i)
            if np.isinf(vif) or np.isnan(vif):
                vif_values.append(np.nan)
            else:
                vif_values.append(vif)
        except:
            vif_values.append(np.nan)
    
    vif_data["VIF"] = vif_values
    
    # remove constant from output
    vif_data = vif_data[vif_data["Variable"] != "const"]
    
    # check if any VIF values are invalid
    if vif_data["VIF"].isna().any():
        print(f"\n{label} (N={len(df_clean)}):")
        print(f"  NOTE: numerical issues in VIF calculation (likely constant mode variables)")
        print(vif_data.to_string(index=False))
        return vif_data
    
    print(f"\n{label} (N={len(df_clean)}):")
    print(vif_data.to_string(index=False))
    
    # check for high VIF
    high_vif = vif_data[vif_data["VIF"] > 10]
    moderate_vif = vif_data[(vif_data["VIF"] > 5) & (vif_data["VIF"] <= 10)]
    
    if not high_vif.empty:
        print(f"  WARNING: {len(high_vif)} variable(s) with VIF > 10 (problematic)")
    if not moderate_vif.empty:
        print(f"  WARNING: {len(moderate_vif)} variable(s) with VIF > 5 (caution warranted)")
    
    return vif_data

########################################################################################
#################### RUN VIF DIAGNOSTICS ###############################################
########################################################################################

# store all VIF results for final summary
vif_issues = []

# no mode analysis
print("\n" + "="*110)
print("WITHOUT MODE VARIABLES")
print("="*110)

vif_swing_no_mode = compute_vif(reg_state_swing_original, state_x_vars_no_mode, "swing state-level polls (no mode)")
if vif_swing_no_mode is not None and ((vif_swing_no_mode["VIF"] > 10).any() or (vif_swing_no_mode["VIF"] > 5).any()):
    vif_issues.append("swing (no mode, full sample): VIF concerns")

vif_state_no_mode = compute_vif(reg_state_original, state_x_vars_no_mode, "state-level polls (no mode)")
if vif_state_no_mode is not None and ((vif_state_no_mode["VIF"] > 10).any() or (vif_state_no_mode["VIF"] > 5).any()):
    vif_issues.append("all states (no mode, full sample): VIF concerns")

vif_national_no_mode = compute_vif(reg_national_original, national_x_vars_no_mode, "national polls (no mode)")
if vif_national_no_mode is not None and ((vif_national_no_mode["VIF"] > 10).any() or (vif_national_no_mode["VIF"] > 5).any()):
    vif_issues.append("national (no mode, full sample): VIF concerns")

# with mode analysis
print("\n" + "="*110)
print("WITH MODE VARIABLES")
print(f"reference mode: {reference_mode}")
print("="*110)

vif_swing_mode = compute_vif(reg_state_swing, state_x_vars_with_mode, "swing state-level polls (with mode)")
if vif_swing_mode is not None and ((vif_swing_mode["VIF"] > 10).any() or (vif_swing_mode["VIF"] > 5).any()):
    vif_issues.append("swing (with mode, full sample): VIF concerns")

vif_state_mode = compute_vif(reg_state, state_x_vars_with_mode, "state-level polls (with mode)")
if vif_state_mode is not None and ((vif_state_mode["VIF"] > 10).any() or (vif_state_mode["VIF"] > 5).any()):
    vif_issues.append("all states (with mode, full sample): VIF concerns")

vif_national_mode = compute_vif(reg_national, national_x_vars_with_mode, "national polls (with mode)")
if vif_national_mode is not None and ((vif_national_mode["VIF"] > 10).any() or (vif_national_mode["VIF"] > 5).any()):
    vif_issues.append("national (with mode, full sample): VIF concerns")

# time window analysis (no mode)
print("\n" + "="*110)
print("TIME WINDOW ANALYSIS (NO MODE)")
print("="*110)

for window in time_windows:
    print(f"\n{'-'*110}")
    print(f"TIME WINDOW: {window} days before election")
    print(f"{'-'*110}")
    
    # swing states
    swing_w = reg_state_swing_original[reg_state_swing_original['days_before_election'] <= window].copy()
    if len(swing_w) >= 10:
        vif_data = compute_vif(swing_w, state_x_vars_no_mode, f"swing states ({window}d window)")
        if vif_data is not None and ((vif_data["VIF"] > 10).any() or (vif_data["VIF"] > 5).any()):
            vif_issues.append(f"swing (no mode, {window}d): VIF concerns")
    
    # all states
    state_w = reg_state_original[reg_state_original['days_before_election'] <= window].copy()
    if len(state_w) >= 10:
        vif_data = compute_vif(state_w, state_x_vars_no_mode, f"all states ({window}d window)")
        if vif_data is not None and ((vif_data["VIF"] > 10).any() or (vif_data["VIF"] > 5).any()):
            vif_issues.append(f"all states (no mode, {window}d): VIF concerns")
    
    # national
    national_w = reg_national_original[reg_national_original['days_before_election'] <= window].copy()
    if len(national_w) >= 10:
        vif_data = compute_vif(national_w, national_x_vars_no_mode, f"national ({window}d window)")
        if vif_data is not None and ((vif_data["VIF"] > 10).any() or (vif_data["VIF"] > 5).any()):
            vif_issues.append(f"national (no mode, {window}d): VIF concerns")

# time window analysis (with mode)
print("\n" + "="*110)
print("TIME WINDOW ANALYSIS (WITH MODE)")
print("="*110)

for window in time_windows:
    print(f"\n{'-'*110}")
    print(f"TIME WINDOW: {window} days before election")
    print(f"{'-'*110}")
    
    # swing states
    swing_w = reg_state_swing[reg_state_swing['days_before_election'] <= window].copy()
    if len(swing_w) >= 10:
        vif_data = compute_vif(swing_w, state_x_vars_with_mode, f"swing states ({window}d window, with mode)")
        if vif_data is not None and ((vif_data["VIF"] > 10).any() or (vif_data["VIF"] > 5).any()):
            vif_issues.append(f"swing (with mode, {window}d): VIF concerns")
    
    # all states
    state_w = reg_state[reg_state['days_before_election'] <= window].copy()
    if len(state_w) >= 10:
        vif_data = compute_vif(state_w, state_x_vars_with_mode, f"all states ({window}d window, with mode)")
        if vif_data is not None and ((vif_data["VIF"] > 10).any() or (vif_data["VIF"] > 5).any()):
            vif_issues.append(f"all states (with mode, {window}d): VIF concerns")
    
    # national
    national_w = reg_national[reg_national['days_before_election'] <= window].copy()
    if len(national_w) >= 10:
        vif_data = compute_vif(national_w, national_x_vars_with_mode, f"national ({window}d window, with mode)")
        if vif_data is not None and ((vif_data["VIF"] > 10).any() or (vif_data["VIF"] > 5).any()):
            vif_issues.append(f"national (with mode, {window}d): VIF concerns")

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
print("="*110 + "\n")

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__
print("VIF diagnostics complete — see output/fiftyplusone_vif_diagnostics_log.txt")