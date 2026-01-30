import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data from pipeline
cces = pd.read_csv("cces2024_meng_replication_set.csv")
truth_raw = pd.read_csv("Meng_true_votes.csv")

# build truth table
truth = truth_raw[["state_name", "p_trump_true", "N_state"]].copy()

# check success for all 51 result areas 
# print("Truth jurisdictions:", len(truth))

########################################################################################
######################## REPLICATION OF FIGURE 4 ON PAGE 711 ###########################
########################################################################################

###### The below first section replicates figure 4 using wald ses for likely voters and binary likely, different from Meng
###### State-level estimators, Meng does this for raw, likely, validated voters
# helper to compute state-level n, p_hat, se, 95% CI
def state_estimates(df, mask=None, value_col="X_trump"):
    """
    Returns df with columns: state_name, n, p_hat, se, ci_lo, ci_hi
    corresponds to Meng's per-state sample mean \hat p_s and its wald SE

    Used only for the left and right panels of Figure 4

    Formula for unweighted sample proportion: \hat p_s = (1/n_s) * sum_{i in s} X_i

    From Meng: "Confidence intervals based on unweighted sample proportions are computed following (3.9)" 
    This gives the wald SE formulas: SE(\hat p_s) = sqrt( \hat p_s (1 - \hat p_s) / n_s ) ;; CI = \hat p_s ± 1.96 * SE(\hat p_s)

    Meng explicitly says SRS variances may be conservative under stratified designs,
    but still do not protect against MSE inflation from nonresponse bias.
    """
    # masks used to limit to validated voters versus just raw sample
    if mask is None:
        sub = df.copy()
    else:
        sub = df[mask].copy()

    # keep only respondents with observed vote for Trump indicator
    sub = sub.dropna(subset=[value_col])

    # group and compute unweighted mean becasue Meng's estimand is the raw sample mean
    out = (
        sub.groupby("state_name")[value_col]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "p_hat"})
    )

    # Meng uses unweighted Wald SE for the sample mean: SE = sqrt(p_hat * (1-p_hat) / n)
    out["se"] = np.sqrt(out["p_hat"] * (1 - out["p_hat"]) / out["n"])

    # 95% wald CIs kept to [0,1]
    out["ci_lo"] = (out["p_hat"] - 1.96 * out["se"]).clip(0, 1)
    out["ci_hi"] = (out["p_hat"] + 1.96 * out["se"]).clip(0, 1)
    return out

# Compute the three estimators used in Meng's figure
# Raw: all respondents with a reported choice
# Likely: those flagged as likely_voter == 1 
# Validated: those with validated_voter == 1 
raw_est = state_estimates(cces, mask=None)
likely_est = state_estimates(cces, mask=(cces["likely_voter"] == 1))
validated_est = state_estimates(cces, mask=(cces["validated_voter"] == 1))

# merge each estimator with truth for comparison
raw_m = raw_est.merge(truth, on="state_name", how="left")
likely_m = likely_est.merge(truth, on="state_name", how="left")
val_m = validated_est.merge(truth, on="state_name", how="left")

###### compute bias per state for the validated estimator 
# bias_s = \hat p_s - p_s (signed), Meng uses this to compute data defect correlation later
val_m["bias"] = val_m["p_hat"] - val_m["p_trump_true"]

# also include absolute bias
val_m["abs_bias"] = val_m["bias"].abs()

# get validated sampling fraction f_s
val_m["f_s"] = val_m["n"] / val_m["N_state"]

# save by state tables for later use
raw_m.to_csv("state_raw_vs_truth.csv", index=False)
likely_m.to_csv("state_likely_vs_truth.csv", index=False)
val_m.to_csv("state_validated_vs_truth.csv", index=False)

###### coloring for plotting of state
# battleground states
purple_states = ["Arizona", "Georgia", "Michigan", "Nevada", "North Carolina", "Pennsylvania", "Wisconsin"]

# Trump won these in 2024 (excluding the purple battlegrounds)
red_states = [
    "Alabama", "Alaska", "Arkansas", "Florida", "Idaho", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Mississippi", "Missouri", "Montana", "Nebraska",
    "North Dakota", "Ohio", "Oklahoma", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "West Virginia", "Wyoming"
]

# Harris won these in 2024 (excluding the purple battlegrounds)
blue_states = [
    "California", "Colorado", "Connecticut", "Delaware", "District Of Columbia",
    "Hawaii", "Illinois", "Maine", "Maryland", "Massachusetts", "Minnesota",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "Oregon",
    "Rhode Island", "Vermont", "Virginia", "Washington"
]

# function to match states to color
def assign_color(state):
    if state in purple_states:
        return "purple"
    elif state in red_states:
        return "red"
    elif state in blue_states:
        return "blue"
    return "black" # in case messed up states
    
# assign colors to each merged dataframe
for df in [raw_m, likely_m, val_m]:
    df["color"] = df["state_name"].apply(assign_color)

###### plot Figure 4 three panels
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

panels = [
    ("Raw (all respondents)", raw_m),
    ("Likely voters", likely_m),
    ("Validated voters", val_m),
]

for ax, (title, dfm) in zip(axes, panels):
    plot_df = dfm.dropna(subset=["p_trump_true", "p_hat"]).copy()

    # errorbars are the 95% CIs
    yerr_lower = plot_df["p_hat"] - plot_df["ci_lo"]
    yerr_upper = plot_df["ci_hi"] - plot_df["p_hat"]

    # loop through each color group to apply the specific color to the markers
    for color_val, group in plot_df.groupby("color"):
        yerr_low = group["p_hat"] - group["ci_lo"]
        yerr_high = group["ci_hi"] - group["p_hat"]
        
        ax.errorbar(group["p_trump_true"], group["p_hat"],
                yerr=[yerr_low, yerr_high],
                fmt="o", ms=6, alpha=0.85, 
                color=color_val,
                capsize=3)
        
    # add 45 degree line
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    # axis labels for figure 4
    ax.set_title(title)
    ax.set_xlabel("True Trump share (state)")
    if ax is axes[0]:
        ax.set_ylabel("Estimated Trump share (CCES)")

plt.suptitle("Figure 4 Replication: State-level CCES estimates vs Official 2024 Results (Trump, Binary Likely)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figure4_cces2024_trump_binarylikely.png", dpi=300)
plt.show()

###### Meng uses different process for panel 2, likely voters, but does not disclose exact formulas, so I have determined my own method from statements
# Meng's caption for Figure 4 says the middle plot uses:
# "estimates weighted to likely voters according to turnout intent" and
# "the turnout adjusted estimate, which is in a ratio form, a delta-method is employed
# to approximate its variance, which is then used to construct confidence intervals."" 

# above previous middle panel used a hard subset (likely_voter == 1) and unweighted Wald SEs, counter to Meng's note

# now will use the ratio estimator weighted mean within each state: \hat p^{LV}_s =  ( sum_i w_i X_i ) / ( sum_i w_i )
# where w_i depends on turnout intent (CCES question CC24_363, how likely are you to vote, cleaned to likely_voter and mapped to a turnout propensity)

# Delta-method / linearization variance for the ratio/weighted mean:
# Var( \hat p^{LV}_s ) ≈  [ sum_i w_i^2 (X_i - \hat p^{LV}_s)^2 ] / (sum_i w_i)^2
# SE = sqrt(Var), CI = \hat p^{LV}_s +/- 1.96*SE

# turnout-intent based weight w_i, treats higher turnout intent as higher propensity to vote
turnout_prop_map = {
    "Yes, definitely": 0.98,
    "Probably": 0.70,
    "I already voted (early or absentee)": 1.00,
    "I plan to vote before November 5th": 0.90,
    "No": 0.05,
    "Undecided": 0.50
}

# normalization of voter intent text
cces["turnout_propensity"] = cces["CC24_363_names"].map(turnout_prop_map)

# turnout weight used in the ratio estimator
cces["lv_weight"] = cces["turnout_propensity"].astype(float)

# state-level turnout weighted ratio estimator with delta method SE
def state_turnout_weighted(df, weight_col="lv_weight", value_col="X_trump"):
    """
    Gives a df with columns as in unweighted function: state_name, n, sum_w, p_hat, se, ci_lo, ci_hi

    Estimator: \hat p^{LV}_s = (Σ w_i X_i) / (Σ w_i)

    Delta-method variance for the weighted mean: Var( \hat p^{LV}_s ) ≈ [ Σ w_i^2 (X_i - \hat p^{LV}_s)^2 ] / (Σ w_i)^2
    """
    sub = df.dropna(subset=[value_col, weight_col]).copy()

    # compute weighted numerator and denominator per state
    sub["_wx"] = sub[weight_col] * sub[value_col]

    agg = sub.groupby("state_name").agg(
        # unweighted count of observed X in the state
        n=(value_col, "count"),
        # Σ w_i
        sum_w=(weight_col, "sum"),
        # Σ w_i X_i
        sum_wx=("_wx", "sum")
    ).reset_index()

    # ratio estimator (is a weighted mean)
    agg["p_hat"] = agg["sum_wx"] / agg["sum_w"]

    # delta-method variance term Σ w_i^2 (X_i - p_hat)^2
    sub = sub.merge(agg[["state_name", "p_hat", "sum_w"]], on="state_name", how="left")
    sub["_w2_dev2"] = (sub[weight_col] ** 2) * ((sub[value_col] - sub["p_hat"]) ** 2)

    num = sub.groupby("state_name")["_w2_dev2"].sum().reset_index().rename(columns={"_w2_dev2": "num_for_var"})
    agg = agg.merge(num, on="state_name", how="left")

    agg["var_delta"] = agg["num_for_var"] / (agg["sum_w"] ** 2)
    agg["se"] = np.sqrt(agg["var_delta"])
    agg["ci_lo"] = (agg["p_hat"] - 1.96 * agg["se"]).clip(0, 1)
    agg["ci_hi"] = (agg["p_hat"] + 1.96 * agg["se"]).clip(0, 1)

    return agg

# calculate the panel as above, but just for the likely, will use the panel 1 and 3 from above

# different from other likely_est
likely_est_weighted = state_turnout_weighted(cces, weight_col="lv_weight", value_col="X_trump")

# merge each estimator with truth for comparison
likely_m_weighted = likely_est_weighted.merge(truth, on="state_name", how="left")

# save by state results
likely_m_weighted.to_csv("state_likely_weighted_vs_truth.csv", index=False)  # now turnout-weighted + delta-method CIs

###### plot Figure 4 three panels, with weighted for panel 2
# assign colors to new likely weighted
for df in [likely_m_weighted]:
    df["color"] = df["state_name"].apply(assign_color)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

panels = [
    ("Raw (all respondents)", raw_m),
    ("Turnout adjusted likely voters", likely_m_weighted),
    ("Validated voters", val_m),
]

for ax, (title, dfm) in zip(axes, panels):
    plot_df = dfm.dropna(subset=["p_trump_true", "p_hat"]).copy()

    # errorbars are the 95% CIs
    yerr_lower = plot_df["p_hat"] - plot_df["ci_lo"]
    yerr_upper = plot_df["ci_hi"] - plot_df["p_hat"]

    # loop through each color group to apply the specific color to the markers
    for color_val, group in plot_df.groupby("color"):
        yerr_low = group["p_hat"] - group["ci_lo"]
        yerr_high = group["ci_hi"] - group["p_hat"]
        
        ax.errorbar(group["p_trump_true"], group["p_hat"],
                yerr=[yerr_low, yerr_high],
                fmt="o", ms=6, alpha=0.85, 
                color=color_val,
                capsize=3)

    # add 45 degree line
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    # axis labels for figure 4
    ax.set_title(title)
    ax.set_xlabel("True Trump share (state)")
    if ax is axes[0]:
        ax.set_ylabel("Estimated Trump share (CCES)")

plt.suptitle("Figure 4 Replication: State-level CCES estimates vs Official 2024 Results (Trump, Weighted Likely)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figure4_cces2024_trump_weighted.png", dpi=300)
plt.show()

##### TO DO: COME BACK TO DO HARRIS PLOTS

########################################################################################
################## STATE LEVEL DATA DEFECT CORRELATIONS, Figures 5 and 8 ###############
########################################################################################

# compute actual bias for each state s
# convert bias to meng 4.7
# make figure 5
# make figure 8

########################################################################################
######################## LAW OF LARGE POPULATIONS, Figures 6 and 7 #####################
########################################################################################

# compute Z score (ish) for each state, equation 3.9
# regress log Z on log N, figure 6
# analyze for selection bias
# figure 7

########################################################################################
######################## Effective Sample Size #########################################
########################################################################################

# get DI and DO
# get neff



###### additional values (NEED TO MAP TO PAGE)
# summary table of bias and n for validated sample
val_summary = val_m[["state_name", "n", "p_hat", "p_trump_true", "bias", "abs_bias"]].sort_values("abs_bias", ascending=False)
print("\nTop 10 states by absolute bias (validated):")
print(val_summary.head(10).to_string(index=False))
val_summary.to_csv("validated_state_bias_summary.csv", index=False)

# mean bias and RMSE
mean_bias = val_m["bias"].mean()
rmse = np.sqrt((val_m["bias"]**2).mean())
print(f"\nValidated sample mean bias: {mean_bias:.4f}, RMSE: {rmse:.4f}")