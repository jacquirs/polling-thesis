import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


# redirect all print output to a log file
log_file = open('output/meng_recreation.txt', 'w')
sys.stdout = log_file

# data from pipeline
CES = pd.read_csv("data/cces2024_cleaned_mengrep.csv")
truth_raw = pd.read_csv("data/true_votes_by_state_mengrep.csv")

# build truth table
truth = truth_raw[["state_name", "p_trump_true", "p_harris_true", "N_state"]].copy()

########################################################################
# Plot style constants
########################################################################
TITLE_FS  = 16
LABEL_FS  = 14
TICK_FS   = 13
LEGEND_FS = 13

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
    but still do not protect against MSE inflation from nonresponse bias
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

# Because there are third-party voters, Harris share is not equal to (1 - Trump share) in either truth or survey; therefore I compute Harris directly from X_harris

# Compute the three estimators used in Meng's figure for each candidate
# Raw: all respondents with a reported choice
# Likely: those flagged as likely_voter == 1 
# Validated: those with validated_voter == 1 

##### Trump estimates
raw_est_T = state_estimates(CES, mask=None, value_col="X_trump")
likely_est_T = state_estimates(CES, mask=(CES["likely_voter"] == 1), value_col="X_trump")
validated_est_T = state_estimates(CES, mask=(CES["validated_voter"] == 1), value_col="X_trump")

# merge each estimator with truth for comparison
raw_mergedtruth_T = raw_est_T.merge(truth, on="state_name", how="left")
likely_mergedtruth_T = likely_est_T.merge(truth, on="state_name", how="left")
val_mergedtruth_T = validated_est_T.merge(truth, on="state_name", how="left")

# Trump bias + abs bias + sampling fraction for validated, used later for DDC
# bias_s = \hat p_s - p_s (signed), Meng uses this to compute data defect correlation
val_mergedtruth_T["bias_trump"] = val_mergedtruth_T["p_hat"] - val_mergedtruth_T["p_trump_true"]

# absolute bias
val_mergedtruth_T["abs_bias_trump"] = val_mergedtruth_T["bias_trump"].abs()

# get validated sampling fraction f_s
val_mergedtruth_T["f_s"] = val_mergedtruth_T["n"] / val_mergedtruth_T["N_state"]


##### Harris estimates
raw_est_H = state_estimates(CES, mask=None, value_col="X_harris")
likely_est_H = state_estimates(CES, mask=(CES["likely_voter"] == 1), value_col="X_harris")
validated_est_H = state_estimates(CES, mask=(CES["validated_voter"] == 1), value_col="X_harris")

raw_mergedtruth_H = raw_est_H.merge(truth, on="state_name", how="left")
likely_mergedtruth_H = likely_est_H.merge(truth, on="state_name", how="left")
val_mergedtruth_H = validated_est_H.merge(truth, on="state_name", how="left")

# Harris bias + abs bias for validated
val_mergedtruth_H["bias_harris"] = val_mergedtruth_H["p_hat"] - val_mergedtruth_H["p_harris_true"]
val_mergedtruth_H["abs_bias_harris"] = val_mergedtruth_H["bias_harris"].abs()
val_mergedtruth_H["f_s"] = val_mergedtruth_H["n"] / val_mergedtruth_H["N_state"]

# save by state tables for later use
raw_mergedtruth_T.to_csv("data/mengrep_fig4_state_estimates_raw_trump_vs_truth.csv", index=False)
likely_mergedtruth_T.to_csv("data/mengrep_fig4_state_estimates_likely_binary_trump_vs_truth.csv", index=False)
val_mergedtruth_T.to_csv("data/mengrep_fig4_state_estimates_validated_trump_vs_truth.csv", index=False)

raw_mergedtruth_H.to_csv("data/mengrep_fig4_state_estimates_raw_harris_vs_truth.csv", index=False)
likely_mergedtruth_H.to_csv("data/mengrep_fig4_state_estimates_likely_binary_harris_vs_truth", index=False)
val_mergedtruth_H.to_csv("data/mengrep_fig4_state_estimates_validated_harris_vs_truth.csv", index=False)


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
    
# assign colors to each merged dataframe
for df_ in [raw_mergedtruth_T, likely_mergedtruth_T, val_mergedtruth_T, raw_mergedtruth_H, likely_mergedtruth_H, val_mergedtruth_H]:
    df_["color"] = df_["state_name"].apply(assign_color)

###### plot Figure 4 three panels for trump
fig, axes = plt.subplots(1, 3, figsize=(20, 6))   # no sharex/sharey — set limits manually

panels_T = [
    ("Raw (All Respondents)", raw_mergedtruth_T),
    ("Likely Voters (Binary)", likely_mergedtruth_T),
    ("Validated Voters", val_mergedtruth_T),
]

for ax, (title, dfm) in zip(axes, panels_T):
    plot_df = dfm.dropna(subset=["p_trump_true", "p_hat"]).copy()

    for color_val, group in plot_df.groupby("color"):
        yerr_low = group["p_hat"] - group["ci_lo"]
        yerr_high = group["ci_hi"] - group["p_hat"]
        ax.errorbar(group["p_trump_true"], group["p_hat"],
                yerr=[yerr_low, yerr_high],
                fmt="o", ms=6, alpha=0.85, color=color_val, capsize=3)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_xlabel("True Trump Share (State)", fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel("Estimated Trump Share (CES)", fontsize=LABEL_FS, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FS)
        label.set_visible(True)
    ax.tick_params(axis='both', labelsize=TICK_FS)

plt.suptitle("CES vs Official 2024 Results (Trump, Binary Likely)",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig4_trump_likely_binary.png", dpi=300)

###### plot Figure 4 three panels for harris
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

panels_H = [
    ("Raw (All Respondents)", raw_mergedtruth_H),
    ("Likely Voters (Binary)", likely_mergedtruth_H),
    ("Validated Voters", val_mergedtruth_H),
]

for ax, (title, dfm) in zip(axes, panels_H):
    plot_df = dfm.dropna(subset=["p_harris_true", "p_hat"]).copy()

    for color_val, group in plot_df.groupby("color"):
        yerr_low = group["p_hat"] - group["ci_lo"]
        yerr_high = group["ci_hi"] - group["p_hat"]
        ax.errorbar(group["p_harris_true"], group["p_hat"],
            yerr=[yerr_low, yerr_high],
            fmt="o", ms=6, alpha=0.85, color=color_val, capsize=3)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_xlabel("True Harris Share (State)", fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel("Estimated Harris Share (CES)", fontsize=LABEL_FS, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FS)
        label.set_visible(True)
    ax.tick_params(axis='both', labelsize=TICK_FS)

plt.suptitle("CES vs Official 2024 Results (Harris, Binary Likely)",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig4_harris_likely_binary.png", dpi=300)


###### Meng uses different process for panel 2, likely voters, but does not disclose exact formulas, so I have determined my own method from statements
# Meng's caption for Figure 4 says the middle plot uses:
# "estimates weighted to likely voters according to turnout intent" and
# "the turnout adjusted estimate, which is in a ratio form, a delta-method is employed
# to approximate its variance, which is then used to construct confidence intervals."" 

# above previous middle panel used a hard subset (likely_voter == 1) and unweighted Wald SEs, counter to Meng's note

# now will use the ratio estimator weighted mean within each state: \hat p^{LV}_s =  ( sum_i w_i X_i ) / ( sum_i w_i )
# where w_i depends on turnout intent (CES question CC24_363, how likely are you to vote, cleaned to likely_voter and mapped to a turnout propensity)

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
CES["turnout_propensity"] = CES["CC24_363_names"].map(turnout_prop_map)

# turnout weight used in the ratio estimator
CES["lv_weight"] = CES["turnout_propensity"].astype(float)

# state-level turnout weighted ratio estimator with delta method SE
def state_turnout_weighted(df, weight_col="lv_weight", value_col="X_trump"):
    """
    Gives a df with columns as in unweighted function: state_name, n, sum_w, p_hat, se, ci_lo, ci_hi

    Estimator: \hat p^{LV}_s = (Σ w_i X_i) / (Σ w_i)

    Delta-method variance for the weighted mean: Var( \hat p^{LV}_s ) ≈ [ Σ w_i^2 (X_i - \hat p^{LV}_s)^2 ] / (Σ w_i)^2
    
    Meng does not publish the exact mapping from turnout intent categories to weight so this mapping is an inference seeimingly consistent with his description
    """

    sub = df.dropna(subset=[value_col, weight_col]).copy()

    # compute weighted numerator and denominator per state
    sub["_wx"] = sub[weight_col] * sub[value_col]

    agg = sub.groupby("state_name").agg(
        n=(value_col, "count"),
        sum_w=(weight_col, "sum"),
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

# Weighted likely panel for Trump 
likely_est_weighted_T = state_turnout_weighted(CES, weight_col="lv_weight", value_col="X_trump")
likely_mergedtruth_weighted_T = likely_est_weighted_T.merge(truth, on="state_name", how="left")
likely_mergedtruth_weighted_T["color"] = likely_mergedtruth_weighted_T["state_name"].apply(assign_color)
likely_mergedtruth_weighted_T.to_csv("data/mengrep_fig4_state_estimates_likely_weighted_trump_vs_truth.csv", index=False)

# Weighted likely panel for Harris
likely_est_weighted_H = state_turnout_weighted(CES, weight_col="lv_weight", value_col="X_harris")
likely_mergedtruth_weighted_H = likely_est_weighted_H.merge(truth, on="state_name", how="left")
likely_mergedtruth_weighted_H["color"] = likely_mergedtruth_weighted_H["state_name"].apply(assign_color)
likely_mergedtruth_weighted_H.to_csv("data/mengrep_fig4_state_estimates_likely_weighted_harris_vs_truth.csv", index=False)


###### plot Figure 4 three panels, with weighted for panel 2

# TRUMP
for df in [likely_mergedtruth_weighted_T, likely_mergedtruth_weighted_H]:
    df["color"] = df["state_name"].apply(assign_color)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

panels_wieghted_T = [
    ("Raw (All Respondents)", raw_mergedtruth_T),
    ("Turnout Adjusted Likely Voters", likely_mergedtruth_weighted_T),
    ("Validated Voters", val_mergedtruth_T),
]

for ax, (title, dfm) in zip(axes, panels_wieghted_T):
    plot_df = dfm.dropna(subset=["p_trump_true", "p_hat"]).copy()

    for color_val, group in plot_df.groupby("color"):
        yerr_low = group["p_hat"] - group["ci_lo"]
        yerr_high = group["ci_hi"] - group["p_hat"]
        ax.errorbar(group["p_trump_true"], group["p_hat"],
                yerr=[yerr_low, yerr_high],
                fmt="o", ms=6, alpha=0.85, color=color_val, capsize=3)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_xlabel("True Trump Share (State)", fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel("Estimated Trump Share (CES)", fontsize=LABEL_FS, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FS)
        label.set_visible(True)
    ax.tick_params(axis='both', labelsize=TICK_FS)

plt.suptitle("CES vs Official 2024 Results (Trump, Weighted Likely)",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig4_trump_likely_weighted.png", dpi=300)

# HARRIS
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

panels_weighted_H = [
    ("Raw (All Respondents)", raw_mergedtruth_H),
    ("Turnout Adjusted Likely Voters", likely_mergedtruth_weighted_H),
    ("Validated Voters", val_mergedtruth_H),
]

for ax, (title, dfm) in zip(axes, panels_weighted_H):
    plot_df = dfm.dropna(subset=["p_harris_true", "p_hat"]).copy()

    for color_val, group in plot_df.groupby("color"):
        yerr_low = group["p_hat"] - group["ci_lo"]
        yerr_high = group["ci_hi"] - group["p_hat"]
        ax.errorbar(
            group["p_harris_true"], group["p_hat"],
            yerr=[yerr_low, yerr_high],
            fmt="o", ms=6, alpha=0.85,
            color=color_val, capsize=3
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_xlabel("True Harris Share (State)", fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel("Estimated Harris Share (CES)", fontsize=LABEL_FS, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FS)
        label.set_visible(True)
    ax.tick_params(axis='both', labelsize=TICK_FS)

plt.suptitle("CES vs Official 2024 Results (Harris, Weighted Likely)",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig4_harris_likely_weighted.png", dpi=300)

########################################################################################
################## STATE LEVEL DATA DEFECT CORRELATIONS, Figures 5 and 8 ###############
########################################################################################

# create table combining trump and harris validated estimates
val_mergedtruth_TH = val_mergedtruth_T[["state_name", "n", "p_hat", "p_trump_true", "p_harris_true", "N_state", "bias_trump", "abs_bias_trump", "f_s"]].copy()

val_mergedtruth_TH = val_mergedtruth_TH.rename(columns={"p_hat": "p_hat_trump"})

val_mergedtruth_TH = val_mergedtruth_TH.merge(
    val_mergedtruth_H[["state_name", "bias_harris", "p_hat"]].rename(columns={"p_hat": "p_hat_harris"}),
    on="state_name",
    how="left"
)

val_mergedtruth_TH.to_csv("data/mengrep_ddc_state_validated_combined_vs_truth.csv", index=False)

# compute (2.4) DO_s = (1 - f_s) / f_s
val_mergedtruth_TH["DO_s"] = (1.0 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["f_s"]

###### Compute per-state DDC estimates (4.7) 
eps = 1e-12 

# Trump
pT = val_mergedtruth_TH["p_trump_true"].clip(eps, 1.0 - eps)
val_mergedtruth_TH["sigma_trump"] = np.sqrt(pT * (1.0 - pT))
val_mergedtruth_TH["rho_hat_trump"] = (val_mergedtruth_TH["bias_trump"] / val_mergedtruth_TH["sigma_trump"]) * np.sqrt(val_mergedtruth_TH["f_s"] / (1.0 - val_mergedtruth_TH["f_s"]))

# Harris
pH = val_mergedtruth_TH["p_harris_true"].clip(eps, 1.0 - eps)
val_mergedtruth_TH["sigma_harris"] = np.sqrt(pH * (1.0 - pH))
val_mergedtruth_TH["rho_hat_harris"] = (val_mergedtruth_TH["bias_harris"] / val_mergedtruth_TH["sigma_harris"]) * np.sqrt(val_mergedtruth_TH["f_s"] / (1.0 - val_mergedtruth_TH["f_s"]))

val_mergedtruth_TH.to_csv("data/mengrep_ddc_state_level_validated_vs_truth.csv", index=False)

###### Figure 5
def histogram_maker_for_figure5(values):
    values_array = np.asarray(values)
    values_array = values_array[~np.isnan(values_array)]
    num_states = len(values_array)
    if num_states <= 1:
        return (np.nan, np.nan, num_states)
    mean_value = float(np.mean(values_array))
    sd_across_states = float(np.std(values_array, ddof=1))
    se_of_mean = sd_across_states / np.sqrt(num_states)
    return (mean_value, 2.0 * se_of_mean, num_states)

rhoH = val_mergedtruth_TH["rho_hat_harris"].values
rhoT = val_mergedtruth_TH["rho_hat_trump"].values

mean_value_of_rhoH, SE_plusminus2_H, number_of_states_used_H = histogram_maker_for_figure5(rhoH)
mean_value_of_rhoT, SE_plusminus2_T, number_of_states_used_T = histogram_maker_for_figure5(rhoT)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Harris panel (left)
axes[0].hist(rhoH, bins=15, edgecolor="black")
axes[0].axvline(0, linestyle="--", linewidth=1, color="red")
axes[0].axvline(mean_value_of_rhoH, linestyle="--", linewidth=1)
axes[0].set_title("Harris",
                  fontsize=TITLE_FS, fontweight='bold')
axes[0].set_xlabel("$\\hat\\rho_N$", fontsize=LABEL_FS, fontweight='bold')
axes[0].set_ylabel("Number of States", fontsize=LABEL_FS, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=TICK_FS)
for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
axes[0].text(
    0.98, 0.95,
    f"mean +/- 2 s.e.\n{mean_value_of_rhoH:.4f} ± {SE_plusminus2_H:.4f}\n(S={number_of_states_used_H})",
    transform=axes[0].transAxes,
    ha="right", va="top", fontsize=LEGEND_FS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

# Trump panel (right)
axes[1].hist(rhoT, bins=15, edgecolor="black")
axes[1].axvline(0, linestyle="--", linewidth=1, color="red")
axes[1].axvline(mean_value_of_rhoT, linestyle="--", linewidth=1)
axes[1].set_title("Trump",
                  fontsize=TITLE_FS, fontweight='bold')
axes[1].set_xlabel("$\\hat\\rho_N$", fontsize=LABEL_FS, fontweight='bold')
axes[1].set_ylabel("Number of States", fontsize=LABEL_FS, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=TICK_FS)
for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
axes[1].text(
    0.98, 0.95,
    f"mean +/- 2 s.e.\n{mean_value_of_rhoT:.4f} ± {SE_plusminus2_T:.4f}\n(S={number_of_states_used_T})",
    transform=axes[1].transAxes,
    ha="right", va="top", fontsize=LEGEND_FS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

plt.suptitle("Distribution of State Data Defect Correlation By Candidate",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig5_ddc_histograms.png", dpi=300)

# figure 5 with color
val_mergedtruth_TH["color"] = val_mergedtruth_TH["state_name"].apply(assign_color)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

color_order = ["blue", "purple", "red"]
color_map_mpl = {"blue": "#4a90d9", "purple": "#9b59b6", "red": "#e25454"}

bins_H = np.linspace(rhoH.min(), rhoH.max(), 16)
bins_T = np.linspace(rhoT.min(), rhoT.max(), 16)

for ax, rho_vals, rho_col, mean_val, se_val, n_used, title, bins in [
    (axes[0], rhoH, "rho_hat_harris", mean_value_of_rhoH, SE_plusminus2_H, number_of_states_used_H,
     "Harris", bins_H),
    (axes[1], rho_vals_T := rhoT, "rho_hat_trump", mean_value_of_rhoT, SE_plusminus2_T, number_of_states_used_T,
     "Trump", bins_T),
]:
    bin_indices = np.digitize(rho_vals, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    color_col = val_mergedtruth_TH["color"].values
    n_bins = len(bins) - 1
    counts = {c: np.zeros(n_bins) for c in color_order}

    for i, (bi, col) in enumerate(zip(bin_indices, color_col)):
        if col in counts:
            counts[col][bi] += 1

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_width = bins[1] - bins[0]
    bottoms = np.zeros(n_bins)

    for col in color_order:
        ax.bar(bin_centers, counts[col], width=bar_width * 0.95,
               bottom=bottoms, color=color_map_mpl[col], edgecolor="none")
        bottoms += counts[col]

    ax.axvline(0, linestyle="--", linewidth=1, color="red")
    ax.axvline(mean_val, linestyle="--", linewidth=1, color="black")
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_xlabel("$\\hat\\rho_N$", fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel("Number of States", fontsize=LABEL_FS, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FS)
        label.set_visible(True)
    ax.text(
        0.98, 0.95,
        f"mean +/- 2 s.e.\n{mean_val:.4f} ± {se_val:.4f}\n(S={n_used})",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=LEGEND_FS,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
    )

plt.suptitle("Distribution of State Data Defect Correlation By Candidate",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig5_ddc_histograms_colors.png", dpi=300)

###### Figure 8
val_mergedtruth_TH["O_trump"] = pT / (1.0 - pT)
val_mergedtruth_TH["O_harris"] = pH / (1.0 - pH)

def meng_bounds_2_9(O_G, DO):
    OG_DO = O_G * DO
    rho_ub = np.minimum(np.sqrt(OG_DO), 1.0 / np.sqrt(OG_DO))
    rho_lb = -np.minimum(np.sqrt(DO / O_G), np.sqrt(O_G / DO))
    return rho_lb, rho_ub

val_mergedtruth_TH["rho_lb_trump"], val_mergedtruth_TH["rho_ub_trump"] = meng_bounds_2_9(val_mergedtruth_TH["O_trump"], val_mergedtruth_TH["DO_s"])
val_mergedtruth_TH["rho_lb_harris"], val_mergedtruth_TH["rho_ub_harris"] = meng_bounds_2_9(val_mergedtruth_TH["O_harris"], val_mergedtruth_TH["DO_s"])

plot_df = val_mergedtruth_TH.copy()
plot_df["color"] = plot_df["state_name"].apply(assign_color)
plot_df["log10_N"] = np.log10(plot_df["N_state"])
plot_df = plot_df.sort_values("log10_N").reset_index(drop=True)
plot_df["x_order"] = np.arange(len(plot_df))

mean_rho_trump = np.nanmean(plot_df["rho_hat_trump"])
mean_rho_harris = np.nanmean(plot_df["rho_hat_harris"])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

line_alpha = 0.6
point_alpha = 0.9
small_point_size = 30
big_point_size = 90

# Harris panel
ax = axes[0]
for _, row in plot_df.iterrows():
    ax.vlines(row["log10_N"], ymin=row["rho_lb_harris"], ymax=row["rho_ub_harris"],
              colors='gray', linestyles='dashed', linewidth=0.9, alpha=line_alpha, zorder=1)

ax.scatter(plot_df["log10_N"], plot_df["rho_hat_harris"],
           s=small_point_size, c=plot_df["color"], alpha=point_alpha,
           edgecolor='none', zorder=3, label=r'$\hat\rho_N$ (empirical)')
ax.scatter(plot_df["log10_N"], plot_df["rho_hat_harris"],
           s=big_point_size, facecolors=plot_df["color"], edgecolors='black',
           linewidths=0.6, alpha=0.95, zorder=4)
ax.axhline(0.0, color='red', linestyle='--', linewidth=1.0, label=r'$\rho=0$ (no bias)')
ax.axhline(mean_rho_harris, color='black', linestyle='--', linewidth=1.0,
           label=f'mean(ρ̂)={mean_rho_harris:.4f}')
ax.set_title("Harris: $\\hat\\rho_N$ with Theoretical Bounds",
             fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel("$\\log_{10}(N_s)$ — Total Voters", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"$\hat\rho_N$", fontsize=LABEL_FS, fontweight='bold')
ax.tick_params(axis='both', labelsize=TICK_FS)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
ax.legend(loc='lower left', fontsize=LEGEND_FS)

# Trump panel
ax = axes[1]
for _, row in plot_df.iterrows():
    ax.vlines(row["log10_N"], ymin=row["rho_lb_trump"], ymax=row["rho_ub_trump"],
              colors='gray', linestyles='dashed', linewidth=0.9, alpha=line_alpha, zorder=1)

ax.scatter(plot_df["log10_N"], plot_df["rho_hat_trump"],
           s=small_point_size, c=plot_df["color"], alpha=point_alpha,
           edgecolor='none', zorder=3)
ax.scatter(plot_df["log10_N"], plot_df["rho_hat_trump"],
           s=big_point_size, facecolors=plot_df["color"], edgecolors='black',
           linewidths=0.6, alpha=0.95, zorder=4)
ax.axhline(0.0, color='red', linestyle='--', linewidth=1.0, label=r'$\rho=0$ (no bias)')
ax.axhline(mean_rho_trump, color='black', linestyle='--', linewidth=1.0,
           label=f'mean(ρ̂)={mean_rho_trump:.4f}')
ax.set_title("Trump: $\\hat\\rho_N$ with Theoretical Bounds",
             fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel("$\\log_{10}(N_s)$ — Total Voters", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"$\hat\rho_N$", fontsize=LABEL_FS, fontweight='bold')
ax.tick_params(axis='both', labelsize=TICK_FS)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
ax.legend(loc='lower left', fontsize=LEGEND_FS)

plt.suptitle("DDC Feasible Bounds",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig8_ddc_bounds.png", dpi=300)

figure8_datacheck = plot_df[[
    "state_name", "N_state", "n", "f_s", "DO_s",
    "p_trump_true", "p_hat_trump", "bias_trump", "rho_hat_trump", "rho_lb_trump", "rho_ub_trump",
    "p_harris_true", "p_hat_harris", "bias_harris", "rho_hat_harris", "rho_lb_harris", "rho_ub_harris"
]].copy()
figure8_datacheck.to_csv("data/mengrep_fig8_ddc_bounds_and_estimates_vs_truth.csv", index=False)


########################################################################################
######################## LAW OF LARGE POPULATIONS, Figures 6 and 7 #####################
########################################################################################

llp_df = val_mergedtruth_TH.copy()

def ols_slope_and_se(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a_hat, b_hat = beta[0], beta[1]
    resid = y - (a_hat + b_hat * x)
    n_obs = len(x)
    rss = np.sum(resid**2)
    s2 = rss / (n_obs - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_b = np.sqrt(s2 * XtX_inv[1, 1])
    return (b_hat, se_b, a_hat)

##### Figure 6
llp_df["Z_nN_trump"]  = np.sqrt(llp_df["N_state"] - 1.0) * llp_df["rho_hat_trump"]
llp_df["Z_nN_harris"] = np.sqrt(llp_df["N_state"] - 1.0) * llp_df["rho_hat_harris"]

llp_df["log10_N"] = np.log10(llp_df["N_state"])
llp_df["log10_absZ_nN_trump"]  = np.log10(np.abs(llp_df["Z_nN_trump"]))
llp_df["log10_absZ_nN_harris"] = np.log10(np.abs(llp_df["Z_nN_harris"]))

beta_T, se_beta_T, alpha_T = ols_slope_and_se(llp_df["log10_N"], llp_df["log10_absZ_nN_trump"])
beta_H, se_beta_H, alpha_H = ols_slope_and_se(llp_df["log10_N"], llp_df["log10_absZ_nN_harris"])

x_line = np.linspace(llp_df["log10_N"].min(), llp_df["log10_N"].max(), 200)
yhat_line_T = alpha_T + beta_T * x_line
yhat_line_H = alpha_H + beta_H * x_line

llp_df["color"] = llp_df["state_name"].apply(assign_color)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Harris panel
axes[0].scatter(llp_df["log10_N"], llp_df["log10_absZ_nN_harris"],
                c=llp_df["color"], alpha=0.85, edgecolors="black", linewidths=0.3)
axes[0].plot(x_line, yhat_line_H, linestyle="--", linewidth=1)
axes[0].set_title("Harris", fontsize=TITLE_FS, fontweight='bold')
axes[0].set_xlabel(r"$\log_{10}(N_s)$  (State Turnout)", fontsize=LABEL_FS, fontweight='bold')
axes[0].set_ylabel(r"$\log_{10}(|Z_{n,N,s}|)$", fontsize=LABEL_FS, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=TICK_FS)
for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
axes[1].text(
    0.98, 0.05,
    f"OLS slope beta = {beta_T:.3f} (SE {se_beta_T:.3f})",
    transform=axes[1].transAxes,
    ha="right", va="bottom", fontsize=LEGEND_FS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)


# Trump panel
axes[1].scatter(llp_df["log10_N"], llp_df["log10_absZ_nN_trump"],
                c=llp_df["color"], alpha=0.85, edgecolors="black", linewidths=0.3)
axes[1].plot(x_line, yhat_line_T, linestyle="--", linewidth=1)
axes[1].set_title("Trump", fontsize=TITLE_FS, fontweight='bold')
axes[1].set_xlabel(r"$\log_{10}(N_s)$  (State Turnout)", fontsize=LABEL_FS, fontweight='bold')
axes[1].set_ylabel(r"$\log_{10}(|Z_{n,N,s}|)$", fontsize=LABEL_FS, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=TICK_FS)
for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
axes[1].text(
    0.98, 0.05,
    f"OLS slope beta = {beta_T:.3f} (SE {se_beta_T:.3f})",
    transform=axes[1].transAxes,
    ha="right", va="bottom", fontsize=LEGEND_FS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

plt.suptitle("The Law of Large Populations Through Nominal Z-Score vs Population",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig6_llp_loglog_regression.png", dpi=300)

##### Figure 7
def regular_zscore_3_9(p_hat, p_true, n):
    p_hat = np.asarray(p_hat, dtype=float)
    p_true = np.asarray(p_true, dtype=float)
    n = np.asarray(n, dtype=float)
    var = (p_hat * (1.0 - p_hat)) / n
    se = np.sqrt(var)
    return (p_hat - p_true) / se

llp_df["Z_n_s_trump"] = regular_zscore_3_9(p_hat=llp_df["p_hat_trump"], p_true=llp_df["p_trump_true"], n=llp_df["n"])
llp_df["Z_n_s_harris"] = regular_zscore_3_9(p_hat=llp_df["p_hat_harris"], p_true=llp_df["p_harris_true"], n=llp_df["n"])

llp_df["cover_rate_H"] = (np.abs(llp_df["Z_n_s_harris"]) <= 2.0)
llp_df["cover_rate_T"] = (np.abs(llp_df["Z_n_s_trump"]) <= 2.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Harris panel
axes[0].scatter(llp_df["log10_N"], llp_df["Z_n_s_harris"],
                c=llp_df["color"], alpha=0.85, edgecolors="black", linewidths=0.3)
axes[0].axhspan(-2, 2, alpha=0.15)
axes[0].axhline(0, linestyle="--", linewidth=1)
axes[0].set_title("Harris", fontsize=TITLE_FS, fontweight='bold')
axes[0].set_xlabel(r"$\log_{10}(N_s)$  (State Turnout)", fontsize=LABEL_FS, fontweight='bold')
axes[0].set_ylabel(r"Regular Z Score $Z_{n,s}$", fontsize=LABEL_FS, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=TICK_FS)
for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
cover_rate_mean_H = llp_df["cover_rate_H"].mean()
axes[1].text(
    0.98, 0.05,
    f"OLS slope beta = {beta_T:.3f} (SE {se_beta_T:.3f})",
    transform=axes[1].transAxes,
    ha="right", va="bottom", fontsize=LEGEND_FS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

# Trump panel
axes[1].scatter(llp_df["log10_N"], llp_df["Z_n_s_trump"],
                c=llp_df["color"], alpha=0.85, edgecolors="black", linewidths=0.3)
axes[1].axhspan(-2, 2, alpha=0.15)
axes[1].axhline(0, linestyle="--", linewidth=1)
axes[1].set_title("Trump", fontsize=TITLE_FS, fontweight='bold')
axes[1].set_xlabel(r"$\log_{10}(N_s)$  (State Turnout)", fontsize=LABEL_FS, fontweight='bold')
axes[1].set_ylabel(r"Regular Z Score $Z_{n,s}$", fontsize=LABEL_FS, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=TICK_FS)
for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
    label.set_fontsize(TICK_FS)
    label.set_visible(True)
cover_rate_mean_T = llp_df["cover_rate_T"].mean()
axes[1].text(
    0.02, 0.95,
    f"Share with |Z_n|<=2: {cover_rate_mean_T:.2%}",
    transform=axes[1].transAxes,
    ha="left", va="top", fontsize=LEGEND_FS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

plt.suptitle("Conventional Z Score vs Population, Log-Log",
             fontsize=TITLE_FS, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig7_llp_zscore_coverage.png", dpi=300)

llp_df.to_csv("data/mengrep_llp_state_level_data_vs_truth.csv", index=False)

# print which states are covered and not, by how much
def coverage_report(df, z_col, cover_col, label):
    df2 = df[["state_name", z_col, cover_col]].copy()
    df2 = df2.sort_values("state_name")

    covered = df2[df2[cover_col]].sort_values(z_col, key=lambda s: np.abs(s))
    not_covered = df2[~df2[cover_col]].copy()
    not_covered["abs_Z"] = np.abs(not_covered[z_col])
    not_covered["exceed_by"] = not_covered["abs_Z"] - 2.0
    not_covered = not_covered.sort_values("exceed_by", ascending=False)

    print(f"{label}: COVERED states (|Z_n| <= 2).  Count = {len(covered)}/{len(df2)}")
    print(covered[["state_name", z_col]].to_string(index=False))

    print(f"{label}: NOT covered states (|Z_n| > 2), with exceed amount (|Z_n|-2).  Count = {len(not_covered)}/{len(df2)}")
    print(not_covered[["state_name", z_col, "abs_Z", "exceed_by"]].to_string(index=False))

coverage_report(llp_df, "Z_n_s_trump", "cover_rate_T", "Trump")
coverage_report(llp_df, "Z_n_s_harris", "cover_rate_H", "Harris")

covered_T = set(llp_df.loc[llp_df["cover_rate_T"], "state_name"])
covered_H = set(llp_df.loc[llp_df["cover_rate_H"], "state_name"])
notcovered_T = set(llp_df.loc[~llp_df["cover_rate_T"], "state_name"])
notcovered_H = set(llp_df.loc[~llp_df["cover_rate_H"], "state_name"])

print("Overlap diagnostics")
print("Covered by both:", sorted(covered_T & covered_H))
print("Covered only Trump:", sorted(covered_T - covered_H))
print("Covered only Harris:", sorted(covered_H - covered_T))
print("Not covered by both:", sorted(notcovered_T & notcovered_H))
print("Not covered only Trump:", sorted(notcovered_T - notcovered_H))
print("Not covered only Harris:", sorted(notcovered_H - notcovered_T))

########################################################################################
################## MSE, Design Effect, and SRS Values ##################################
########################################################################################

val_mergedtruth_TH["S2_trump"] = (
    val_mergedtruth_TH["N_state"] / (val_mergedtruth_TH["N_state"] - 1)
    * val_mergedtruth_TH["sigma_trump"]**2
)
val_mergedtruth_TH["S2_harris"] = (
    val_mergedtruth_TH["N_state"] / (val_mergedtruth_TH["N_state"] - 1)
    * val_mergedtruth_TH["sigma_harris"]**2
)

val_mergedtruth_TH["Var_SRS_trump"] = (
    (1 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["n"]
    * val_mergedtruth_TH["S2_trump"]
)
val_mergedtruth_TH["Var_SRS_harris"] = (
    (1 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["n"]
    * val_mergedtruth_TH["S2_harris"]
)

val_mergedtruth_TH["SE_SRS_trump"] = np.sqrt(val_mergedtruth_TH["Var_SRS_trump"])
val_mergedtruth_TH["SE_SRS_harris"] = np.sqrt(val_mergedtruth_TH["Var_SRS_harris"])

val_mergedtruth_TH["DU_trump"] = val_mergedtruth_TH["sigma_trump"]**2
val_mergedtruth_TH["DU_harris"] = val_mergedtruth_TH["sigma_harris"]**2

val_mergedtruth_TH["MSE_trump"] = (
    val_mergedtruth_TH["rho_hat_trump"]**2
    * val_mergedtruth_TH["DO_s"]
    * val_mergedtruth_TH["DU_trump"]
)
val_mergedtruth_TH["MSE_harris"] = (
    val_mergedtruth_TH["rho_hat_harris"]**2
    * val_mergedtruth_TH["DO_s"]
    * val_mergedtruth_TH["DU_harris"]
)

val_mergedtruth_TH["RMSE_trump"] = np.sqrt(val_mergedtruth_TH["MSE_trump"])
val_mergedtruth_TH["RMSE_harris"] = np.sqrt(val_mergedtruth_TH["MSE_harris"])

val_mergedtruth_TH["Deff_trump"] = (
    (val_mergedtruth_TH["N_state"] - 1) * val_mergedtruth_TH["rho_hat_trump"]**2
)
val_mergedtruth_TH["Deff_harris"] = (
    (val_mergedtruth_TH["N_state"] - 1) * val_mergedtruth_TH["rho_hat_harris"]**2
)

print("TRUMP:")
print(f"Mean DI (data defect index):   {val_mergedtruth_TH['rho_hat_trump'].pow(2).mean():.8f}")
print(f"Mean DO (dropout odds):        {val_mergedtruth_TH['DO_s'].mean():.2f}")
print(f"Mean DU (degree uncertainty):  {val_mergedtruth_TH['DU_trump'].mean():.6f}")
print(f"Mean MSE:                      {val_mergedtruth_TH['MSE_trump'].mean():.8f}")
print(f"Mean RMSE:                     {val_mergedtruth_TH['RMSE_trump'].mean():.4f} ({val_mergedtruth_TH['RMSE_trump'].mean()*100:.2f} pp)")
print(f"Mean Var_SRS (benchmark):      {val_mergedtruth_TH['Var_SRS_trump'].mean():.8f}")
print(f"Mean SE_SRS (benchmark):       {val_mergedtruth_TH['SE_SRS_trump'].mean():.4f} ({val_mergedtruth_TH['SE_SRS_trump'].mean()*100:.2f} pp)")
print(f"Mean Design Effect (Deff):     {val_mergedtruth_TH['Deff_trump'].mean():.2f}")

print("\nHARRIS:")
print(f"Mean DI (data defect index):   {val_mergedtruth_TH['rho_hat_harris'].pow(2).mean():.8f}")
print(f"Mean DO (dropout odds):        {val_mergedtruth_TH['DO_s'].mean():.2f}")
print(f"Mean DU (degree uncertainty):  {val_mergedtruth_TH['DU_harris'].mean():.6f}")
print(f"Mean MSE:                      {val_mergedtruth_TH['MSE_harris'].mean():.8f}")
print(f"Mean RMSE:                     {val_mergedtruth_TH['RMSE_harris'].mean():.4f} ({val_mergedtruth_TH['RMSE_harris'].mean()*100:.2f} pp)")
print(f"Mean Var_SRS (benchmark):      {val_mergedtruth_TH['Var_SRS_harris'].mean():.8f}")
print(f"Mean SE_SRS (benchmark):       {val_mergedtruth_TH['SE_SRS_harris'].mean():.4f} ({val_mergedtruth_TH['SE_SRS_harris'].mean()*100:.2f} pp)")
print(f"Mean Design Effect (Deff):     {val_mergedtruth_TH['Deff_harris'].mean():.2f}")

print("\nTop five by design effect, worst precision loss\n")
print("TRUMP:")
top5_trump = val_mergedtruth_TH.nlargest(5, "Deff_trump")[
    ["state_name", "N_state", "rho_hat_trump", "Deff_trump", "RMSE_trump"]
].copy()
top5_trump["RMSE_pct"] = top5_trump["RMSE_trump"] * 100
print(top5_trump[["state_name", "N_state", "rho_hat_trump", "Deff_trump", "RMSE_pct"]].to_string(index=False))

print("\nHARRIS:")
top5_harris = val_mergedtruth_TH.nlargest(5, "Deff_harris")[
    ["state_name", "N_state", "rho_hat_harris", "Deff_harris", "RMSE_harris"]
].copy()
top5_harris["RMSE_pct"] = top5_harris["RMSE_harris"] * 100
print(top5_harris[["state_name", "N_state", "rho_hat_harris", "Deff_harris", "RMSE_pct"]].to_string(index=False))

print("Trump states with largest MSE")
worst_mse_trump = val_mergedtruth_TH.nlargest(10, "MSE_trump")[["state_name", "MSE_trump", "rho_hat_trump", "f_s", "DO_s","sigma_trump", "p_hat_trump","p_trump_true","bias_trump"]]
print(worst_mse_trump
    .assign(sigma2=lambda df: df["sigma_trump"] ** 2)
    .rename(columns={
        "state_name": "State",
        "MSE_trump": r"$\mathrm{MSE}$",
        "rho_hat_trump": r"$\rho$",
        "f_s": r"$f$",
        "Dropout odds":r"$D_O$",
        "sigma2": r"$\sigma^2$",
        "Estimated Trump Support": r"$\hat p$",
        "Realized Trump Support": r"$p$",
        "Bias":r"bias",
    }).to_string(index=False))

print("\nInterpretation guide:")
print(r"High $|\rho| \;\to\;$ strong selection bias")
print(r"Low $f \;\to\;$ low response rate")
print(r"High $\sigma^2 \;\to\;$ close race ($p \approx 0.5$)")

print("\nHarris states with largest MSE")
worst_mse_harris = val_mergedtruth_TH.nlargest(10, "MSE_harris")[["state_name", "MSE_harris", "rho_hat_harris", "f_s", "sigma_harris","p_hat_harris","p_harris_true","bias_harris"]]
print(
    worst_mse_harris
    .assign(sigma2=lambda df: df["sigma_harris"] ** 2)
    .rename(columns={
        "state_name": "State",
        "MSE_harris": r"$\mathrm{MSE}$",
        "rho_hat_harris": r"$\rho$",
        "f_s": r"$f$",
        "Dropout odds":r"$D_O$",
        "sigma2": r"$\sigma^2$",
        "Estimated Harris Support": r"$\hat p$",
        "Realized Harris Support": r"$p$",
        "Bias":r"bias",
    }).to_string(index=False))

########################################################################################
######################## Effective Sample Size, by state ###############################
########################################################################################

eff_samplesize_df = val_mergedtruth_TH.copy()

eff_samplesize_df["DI_trump"]  = eff_samplesize_df["rho_hat_trump"]**2
eff_samplesize_df["DI_harris"] = eff_samplesize_df["rho_hat_harris"]**2

eff_samplesize_df["n_star_eff_trump"] = 1.0 / (eff_samplesize_df["DO_s"] * eff_samplesize_df["DI_trump"])
eff_samplesize_df["n_star_eff_harris"] = 1.0 / (eff_samplesize_df["DO_s"] * eff_samplesize_df["DI_harris"])

eff_samplesize_df["n_eff_trump"] = eff_samplesize_df["n_star_eff_trump"] / (
    1.0 + (eff_samplesize_df["n_star_eff_trump"] - 1.0) / 
    (eff_samplesize_df["N_state"] - 1.0)
)

eff_samplesize_df["n_eff_harris"] = eff_samplesize_df["n_star_eff_harris"] / (
    1.0 + (eff_samplesize_df["n_star_eff_harris"] - 1.0) / 
    (eff_samplesize_df["N_state"] - 1.0)
) 

eff_samplesize_df["sigma2_trump"]  = eff_samplesize_df["p_trump_true"]  * (1.0 - eff_samplesize_df["p_trump_true"])
eff_samplesize_df["sigma2_harris"] = eff_samplesize_df["p_harris_true"] * (1.0 - eff_samplesize_df["p_harris_true"])

eff_samplesize_df["Me95_star_trump"]  = 2.0 * np.sqrt(eff_samplesize_df["sigma2_trump"]  / eff_samplesize_df["n_star_eff_trump"])
eff_samplesize_df["Me95_star_harris"] = 2.0 * np.sqrt(eff_samplesize_df["sigma2_harris"] / eff_samplesize_df["n_star_eff_harris"])

eff_samplesize_df["Me95_star_upper_trump"]  = 1.0 / np.sqrt(eff_samplesize_df["n_star_eff_trump"])
eff_samplesize_df["Me95_star_upper_harris"] = 1.0 / np.sqrt(eff_samplesize_df["n_star_eff_harris"])

effective_sample_size_outputs = eff_samplesize_df[[
    "state_name", "N_state", "n", "f_s", "DO_s", "rho_hat_trump", "DI_trump", "n_star_eff_trump", "n_eff_trump", "Me95_star_trump", "Me95_star_upper_trump",
    "rho_hat_harris", "DI_harris", "n_star_eff_harris", "n_eff_harris", "Me95_star_harris", "Me95_star_upper_harris",
]].copy()

effective_sample_size_outputs.to_csv("data/mengrep_effss_by_state_vs_truth.csv", index=False)

print("\nBottom 10 states by n*_eff (Trump):")
print(effective_sample_size_outputs.sort_values("n_star_eff_trump").head(10)[
    ["state_name", "n_star_eff_trump", "Me95_star_trump"]
].to_string(index=False))

print("\nBottom 10 states by n*_eff (Harris):")
print(effective_sample_size_outputs.sort_values("n_star_eff_harris").head(10)[
    ["state_name", "n_star_eff_harris", "Me95_star_harris"]
].to_string(index=False))

new_cols = [col for col in eff_samplesize_df.columns if col not in val_mergedtruth_TH.columns]
for col in new_cols:
    val_mergedtruth_TH[col] = eff_samplesize_df[col]

val_mergedtruth_TH.to_csv("data/mengrep_all_vals_state_level.csv", index=False)

########################################################################################
####### Overall Effective Sample Size Pooled Across States, and additional values ######
########################################################################################

n_total = eff_samplesize_df["n"].sum()
N_total = eff_samplesize_df["N_state"].sum()
f_total = n_total / N_total
DO_total = (1.0 - f_total) / f_total

p_hat_trump_total = (eff_samplesize_df["p_hat_trump"] * eff_samplesize_df["n"]).sum() / n_total
p_hat_harris_total = (eff_samplesize_df["p_hat_harris"] * eff_samplesize_df["n"]).sum() / n_total

p_trump_true_total = (eff_samplesize_df["p_trump_true"] * eff_samplesize_df["N_state"]).sum() / N_total
p_harris_true_total = (eff_samplesize_df["p_harris_true"] * eff_samplesize_df["N_state"]).sum() / N_total

bias_trump_total = p_hat_trump_total - p_trump_true_total
bias_harris_total = p_hat_harris_total - p_harris_true_total

sigma_trump_total = np.sqrt(p_trump_true_total * (1.0 - p_trump_true_total))
sigma_harris_total = np.sqrt(p_harris_true_total * (1.0 - p_harris_true_total))

rho_trump_total = (bias_trump_total / sigma_trump_total) * np.sqrt(f_total / (1.0 - f_total))
rho_harris_total = (bias_harris_total / sigma_harris_total) * np.sqrt(f_total / (1.0 - f_total))

DI_trump_total = rho_trump_total**2
DI_harris_total = rho_harris_total**2

n_star_eff_trump_total = 1.0 / (DO_total * DI_trump_total)
n_star_eff_harris_total = 1.0 / (DO_total * DI_harris_total)

n_eff_trump_total = n_star_eff_trump_total / (1.0 + (n_star_eff_trump_total - 1.0) / (N_total - 1.0))
n_eff_harris_total = n_star_eff_harris_total / (1.0 + (n_star_eff_harris_total - 1.0) / (N_total - 1.0))

Me_trump_total = 2.0 * np.sqrt((sigma_trump_total**2) / n_star_eff_trump_total)
Me_harris_total = 2.0 * np.sqrt((sigma_harris_total**2) / n_star_eff_harris_total)

Me_upper_trump_total = 1.0 / np.sqrt(n_star_eff_trump_total)
Me_upper_harris_total = 1.0 / np.sqrt(n_star_eff_harris_total)

sigma2_trump_national = p_trump_true_total * (1.0 - p_trump_true_total)
sigma2_harris_national = p_harris_true_total * (1.0 - p_harris_true_total)

S2_trump_national = (N_total / (N_total - 1)) * sigma2_trump_national
S2_harris_national = (N_total / (N_total - 1)) * sigma2_harris_national

Var_SRS_trump_national = ((1 - f_total) / n_total) * S2_trump_national
Var_SRS_harris_national = ((1 - f_total) / n_total) * S2_harris_national

SE_SRS_trump_national = np.sqrt(Var_SRS_trump_national)
SE_SRS_harris_national = np.sqrt(Var_SRS_harris_national)

MSE_trump_national = DI_trump_total * DO_total * sigma2_trump_national
MSE_harris_national = DI_harris_total * DO_total * sigma2_harris_national

RMSE_trump_national = np.sqrt(MSE_trump_national)
RMSE_harris_national = np.sqrt(MSE_harris_national)

Deff_trump_national = (N_total - 1) * DI_trump_total
Deff_harris_national = (N_total - 1) * DI_harris_total

national_long_table = pd.DataFrame({
    'Metric': [
        '=== population and sample value ===',
        'Population ($N$)', 'Sample size ($n$)', 'Sampling rate ($f$)', 'Dropout odds ($DO$)', '',
        '=== estimates vs truth ===',
        r'Sample proportion ($\hat{p}$)', 'True proportion ($p$)', r'Bias ($\hat{p} - p$)',
        r'Absolute bias ($|\hat{p} - p|$)', r'Standard deviation ($\sigma_G$)',
        r'Variance ($\sigma^2_G = DU$)', '',
        '=== data quality ===',
        r'Data defect correlation ($\rho_{R,G}$)', r'Data defect index ($DI = \rho^2$)', '',
        '=== SRS benchmark ===',
        r'Finite-corrected variance ($S^2_G$)', r'SRS variance ($Var_{SRS}$)',
        r'SRS standard error ($SE_{SRS}$)', r'SRS margin of error (95% CI)', '',
        '=== actual MSE ===',
        'Mean Squared Error (MSE)', 'Root MSE (RMSE)', 'RMSE (percentage points)', '',
        '=== design effect ===',
        'Design Effect (Deff)', 'Deff interpretation', '',
        '=== effective sample size ===',
        r'$n^*_{eff}$ (upper bound)', r'$n_{eff}$ (finite-corrected)',
        'Sample reduction (%)', 'Sample reduction (count)', '',
        '=== margin of error ===',
        'Margin of error (Me)', 'Me (percentage points)', 'Me upper bound',
        'Me upper bound (pp)', 'Me inflation vs SRS', '',
    ],
    'Trump': [
        '', f'{N_total:,.0f}', f'{n_total:,.0f}', f'{f_total:.6f}', f'{DO_total:.4f}', '',
        '', f'{p_hat_trump_total:.6f}', f'{p_trump_true_total:.6f}', f'{bias_trump_total:+.6f}',
        f'{abs(bias_trump_total):.6f}', f'{sigma_trump_total:.6f}', f'{sigma_trump_total**2:.8f}', '',
        '', f'{rho_trump_total:+.8f}', f'{DI_trump_total:.10f}', '',
        '', f'{S2_trump_national:.8f}', f'{Var_SRS_trump_national:.10f}',
        f'{SE_SRS_trump_national:.6f}', f'{SE_SRS_trump_national*2:.6f}', '',
        '', f'{MSE_trump_national:.10f}', f'{RMSE_trump_national:.6f}', f'{RMSE_trump_national*100:.4f}', '',
        '', f'{Deff_trump_national:.4f}', f'{Deff_trump_national:.1f}x worse than SRS', '',
        '', f'{n_star_eff_trump_total:,.0f}', f'{n_eff_trump_total:,.0f}',
        f'{(1 - n_star_eff_trump_total/n_total)*100:.4f}%', f'{n_total - n_star_eff_trump_total:,.0f}', '',
        '', f'{Me_trump_total:.6f}', f'{Me_trump_total*100:.4f}',
        f'{Me_upper_trump_total:.6f}', f'{Me_upper_trump_total*100:.4f}',
        f'{Me_trump_total/(SE_SRS_trump_national*2):.4f}x', '',
    ],
    'Harris': [
        '', f'{N_total:,.0f}', f'{n_total:,.0f}', f'{f_total:.6f}', f'{DO_total:.4f}', '',
        '', f'{p_hat_harris_total:.6f}', f'{p_harris_true_total:.6f}', f'{bias_harris_total:+.6f}',
        f'{abs(bias_harris_total):.6f}', f'{sigma_harris_total:.6f}', f'{sigma_harris_total**2:.8f}', '',
        '', f'{rho_harris_total:+.8f}', f'{DI_harris_total:.10f}', '',
        '', f'{S2_harris_national:.8f}', f'{Var_SRS_harris_national:.10f}',
        f'{SE_SRS_harris_national:.6f}', f'{SE_SRS_harris_national*2:.6f}', '',
        '', f'{MSE_harris_national:.10f}', f'{RMSE_harris_national:.6f}', f'{RMSE_harris_national*100:.4f}', '',
        '', f'{Deff_harris_national:.4f}', f'{Deff_harris_national:.1f}x worse than srs', '',
        '', f'{n_star_eff_harris_total:,.0f}', f'{n_eff_harris_total:,.0f}',
        f'{(1 - n_star_eff_harris_total/n_total)*100:.4f}%', f'{n_total - n_star_eff_harris_total:,.0f}', '',
        '', f'{Me_harris_total:.6f}', f'{Me_harris_total*100:.4f}',
        f'{Me_upper_harris_total:.6f}', f'{Me_upper_harris_total*100:.4f}',
        f'{Me_harris_total/(SE_SRS_harris_national*2):.4f}x', '',
    ]
})

national_wide_table = pd.DataFrame({
    'candidate': ['Trump', 'Harris'],
    'N': [N_total, N_total], 'n': [n_total, n_total], 'f': [f_total, f_total], 'DO': [DO_total, DO_total],
    'p_hat': [p_hat_trump_total, p_hat_harris_total],
    'p_true': [p_trump_true_total, p_harris_true_total],
    'bias': [bias_trump_total, bias_harris_total],
    'abs_bias': [abs(bias_trump_total), abs(bias_harris_total)],
    'sigma_G': [sigma_trump_total, sigma_harris_total],
    'sigma2_G': [sigma_trump_total**2, sigma_harris_total**2],
    'rho_R_G': [rho_trump_total, rho_harris_total],
    'DI': [DI_trump_total, DI_harris_total],
    'S2_G': [S2_trump_national, S2_harris_national],
    'Var_SRS': [Var_SRS_trump_national, Var_SRS_harris_national],
    'SE_SRS': [SE_SRS_trump_national, SE_SRS_harris_national],
    'Me_SRS': [SE_SRS_trump_national*2, SE_SRS_harris_national*2],
    'MSE': [MSE_trump_national, MSE_harris_national],
    'RMSE': [RMSE_trump_national, RMSE_harris_national],
    'RMSE_pp': [RMSE_trump_national*100, RMSE_harris_national*100],
    'Deff': [Deff_trump_national, Deff_harris_national],
    'n_star_eff': [n_star_eff_trump_total, n_star_eff_harris_total],
    'n_eff': [n_eff_trump_total, n_eff_harris_total],
    'reduction_pct': [(1-n_star_eff_trump_total/n_total)*100, (1-n_star_eff_harris_total/n_total)*100],
    'reduction_count': [n_total - n_star_eff_trump_total, n_total - n_star_eff_harris_total],
    'Me': [Me_trump_total, Me_harris_total],
    'Me_pp': [Me_trump_total*100, Me_harris_total*100],
    'Me_upper': [Me_upper_trump_total, Me_upper_harris_total],
    'Me_upper_pp': [Me_upper_trump_total*100, Me_upper_harris_total*100],
    'Me_inflation': [Me_trump_total/(SE_SRS_trump_national*2), Me_harris_total/(SE_SRS_harris_national*2)],
})

national_long_table.to_csv("data/mengrep_national_long.csv", index=False)
print("\n" + national_long_table.to_string(index=False))
national_wide_table.to_csv("data/mengrep_national_wide.csv", index=False)

##### handling NAs
mask = (CES["validated_voter"] == 1)

n_validated_all = mask.sum()
n_validated_trump_nonmissing = (mask & CES["X_trump"].notna()).sum()
n_validated_harris_nonmissing = (mask & CES["X_harris"].notna()).sum()

truth_states = set(truth["state_name"])
n_validated_in_truth_states = (mask & CES["state_name"].isin(truth_states)).sum()
n_validated_trump_nonmissing_in_truth_states = (mask & CES["state_name"].isin(truth_states) & CES["X_trump"].notna()).sum()
n_validated_harris_nonmissing_in_truth_states = (mask & CES["state_name"].isin(truth_states) & CES["X_harris"].notna()).sum()

n_table_total = eff_samplesize_df["n"].sum()

print("Microdata: validated (all)                         =", n_validated_all)
print("Microdata: validated & X_trump notna               =", n_validated_trump_nonmissing)
print("Microdata: validated & X_harris notna              =", n_validated_harris_nonmissing)
print("Microdata: validated & state in truth              =", n_validated_in_truth_states)
print("Microdata: validated & state in truth & trump notna=", n_validated_trump_nonmissing_in_truth_states)
print("Microdata: validated & state in truth & harris notna=", n_validated_harris_nonmissing_in_truth_states)
print("Table: sum eff_samplesize_df['n']                  =", n_table_total)

cces_state_counts = (
    CES.loc[CES["validated_voter"] == 1]
        .groupby("state_name")
        .size()
        .reset_index(name="n_cces_validated")
)

eff_state_counts = (
    eff_samplesize_df[["state_name", "n"]]
        .rename(columns={"n": "n_eff_table"})
)

state_count_diff = (
    cces_state_counts
        .merge(eff_state_counts, on="state_name", how="outer")
        .fillna(0)
)

state_count_diff["difference"] = (
    state_count_diff["n_cces_validated"]
    - state_count_diff["n_eff_table"]
)

print(
    state_count_diff.loc[state_count_diff["difference"] != 0]
        .sort_values("difference", ascending=False)
        .to_string(index=False)
)

print("National rho trump:", rho_trump_total)
print("National rho harris:", rho_harris_total)
print("State rho trump mean:", val_mergedtruth_TH["rho_hat_trump"].mean())
print("State rho trump max abs:", val_mergedtruth_TH["rho_hat_trump"].abs().max())
print("State rho harris mean:", val_mergedtruth_TH["rho_hat_harris"].mean())
print("State rho harris max abs:", val_mergedtruth_TH["rho_hat_harris"].abs().max())
print("N_total:", N_total)
print("Mean N_state:", val_mergedtruth_TH["N_state"].mean())
print("N_total / mean N_state:", N_total / val_mergedtruth_TH["N_state"].mean())

# close log and restore terminal
log_file.close()
sys.stdout = sys.__stdout__