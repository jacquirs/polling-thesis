import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data from pipeline
cces = pd.read_csv("cces2024_meng_replication_set.csv")
truth_raw = pd.read_csv("Meng_true_votes.csv")

# build truth table
truth = truth_raw[["state_name", "p_trump_true", "p_harris_true", "N_state"]].copy()

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

# Because there are third-party voters, Harris share is not equal to (1 - Trump share) in either truth or survey; therefore I compute Harris directly from X_harris

# Compute the three estimators used in Meng's figure for each candidate
# Raw: all respondents with a reported choice
# Likely: those flagged as likely_voter == 1 
# Validated: those with validated_voter == 1 

##### Trump estimates
raw_est_T = state_estimates(cces, mask=None, value_col="X_trump")
likely_est_T = state_estimates(cces, mask=(cces["likely_voter"] == 1), value_col="X_trump")
validated_est_T = state_estimates(cces, mask=(cces["validated_voter"] == 1), value_col="X_trump")

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
raw_est_H = state_estimates(cces, mask=None, value_col="X_harris")
likely_est_H = state_estimates(cces, mask=(cces["likely_voter"] == 1), value_col="X_harris")
validated_est_H = state_estimates(cces, mask=(cces["validated_voter"] == 1), value_col="X_harris")

raw_mergedtruth_H = raw_est_H.merge(truth, on="state_name", how="left")
likely_mergedtruth_H = likely_est_H.merge(truth, on="state_name", how="left")
val_mergedtruth_H = validated_est_H.merge(truth, on="state_name", how="left")

# Harris bias + abs bias for validated
val_mergedtruth_H["bias_harris"] = val_mergedtruth_H["p_hat"] - val_mergedtruth_H["p_harris_true"]
val_mergedtruth_H["abs_bias_harris"] = val_mergedtruth_H["bias_harris"].abs()
val_mergedtruth_H["f_s"] = val_mergedtruth_H["n"] / val_mergedtruth_H["N_state"]

# save by state tables for later use
raw_mergedtruth_T.to_csv("state_raw_vs_truth_trump.csv", index=False)
likely_mergedtruth_T.to_csv("state_likely_vs_truth_trump_binarylikely.csv", index=False)
val_mergedtruth_T.to_csv("state_validated_vs_truth_trump.csv", index=False)

raw_mergedtruth_H.to_csv("state_raw_vs_truth_harris.csv", index=False)
likely_mergedtruth_H.to_csv("state_likely_vs_truth_harris_binarylikely.csv", index=False)
val_mergedtruth_H.to_csv("state_validated_vs_truth_harris.csv", index=False)


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
for df_ in [raw_mergedtruth_T, likely_mergedtruth_T, val_mergedtruth_T, raw_mergedtruth_H, likely_mergedtruth_H, val_mergedtruth_H]:
    df_["color"] = df_["state_name"].apply(assign_color)

###### plot Figure 4 three panels for trump
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

panels_T = [
    ("Raw (all respondents)", raw_mergedtruth_T),
    ("Likely voters (binary)", likely_mergedtruth_T),
    ("Validated voters", val_mergedtruth_T),
]

for ax, (title, dfm) in zip(axes, panels_T):
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

###### plot Figure 4 three panels for harris, same as trump above jsut with harris var
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

panels_H = [
    ("Raw (all respondents)", raw_mergedtruth_H),
    ("Likely voters (binary)", likely_mergedtruth_H),
    ("Validated voters", val_mergedtruth_H),
]

for ax, (title, dfm) in zip(axes, panels_H):
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
    ax.set_title(title)
    ax.set_xlabel("True Harris share (state)")
    if ax is axes[0]:
        ax.set_ylabel("Estimated Harris share (CCES)")

plt.suptitle("Figure 4 Replication: CCES vs Official 2024 Results (Harris, Binary Likely)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figure4_cces2024_harris_binarylikely.png", dpi=300)
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
    
    Meng does not publish the exact mapping from turnout intent categories to weight so this mapping is an inference seeimingly consistent with his description
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

# Weighted likely panel for Trump 
likely_est_weighted_T = state_turnout_weighted(cces, weight_col="lv_weight", value_col="X_trump")
likely_mergedtruth_weighted_T = likely_est_weighted_T.merge(truth, on="state_name", how="left")
likely_mergedtruth_weighted_T["color"] = likely_mergedtruth_weighted_T["state_name"].apply(assign_color)
likely_mergedtruth_weighted_T.to_csv("state_likely_weighted_vs_truth_trump.csv", index=False)

# Weighted likely panel for Harris
likely_est_weighted_H = state_turnout_weighted(cces, weight_col="lv_weight", value_col="X_harris")
likely_mergedtruth_weighted_H = likely_est_weighted_H.merge(truth, on="state_name", how="left")
likely_mergedtruth_weighted_H["color"] = likely_mergedtruth_weighted_H["state_name"].apply(assign_color)
likely_mergedtruth_weighted_H.to_csv("state_likely_weighted_vs_truth_harris.csv", index=False)


###### plot Figure 4 three panels, with weighted for panel 2

# TRUMP
# assign colors to new likely weighted
for df in [likely_mergedtruth_weighted_T, likely_mergedtruth_weighted_H]:
    df["color"] = df["state_name"].apply(assign_color)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

panels_wieghted_T = [
    ("Raw (all respondents)", raw_mergedtruth_T),
    ("Turnout adjusted likely voters", likely_mergedtruth_weighted_T),
    ("Validated voters", val_mergedtruth_T),
]

for ax, (title, dfm) in zip(axes, panels_wieghted_T):
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

# HARRIS, same as trump
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

panels_weighted_H = [
    ("Raw (all respondents)", raw_mergedtruth_H),
    ("Turnout adjusted likely voters", likely_mergedtruth_weighted_H),
    ("Validated voters", val_mergedtruth_H),
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
    ax.set_title(title)
    ax.set_xlabel("True Harris share (state)")
    if ax is axes[0]:
        ax.set_ylabel("Estimated Harris share (CCES)")

plt.suptitle("Figure 4 Replication: CCES vs Official 2024 Results (Harris, Weighted Likely)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figure4_cces2024_harris_weighted.png", dpi=300)
plt.show()

########################################################################################
################## STATE LEVEL DATA DEFECT CORRELATIONS, Figures 5 and 8 ###############
########################################################################################

# create table comabining trump and harris validated esimtates
val_mergedtruth_TH = val_mergedtruth_T[["state_name", "n", "p_hat", "p_trump_true", "p_harris_true", "N_state", "bias_trump", "abs_bias_trump", "f_s"]].copy()

val_mergedtruth_TH = val_mergedtruth_TH.rename(columns={"p_hat": "p_hat_trump"})

val_mergedtruth_TH = val_mergedtruth_TH.merge(
    val_mergedtruth_H[["state_name", "bias_harris", "p_hat"]].rename(columns={"p_hat": "p_hat_harris"}),
    on="state_name",
    how="left"
)

# for later use in case
val_mergedtruth_TH.to_csv("state_validated_trump_harris_vs_truth.csv", index=False)

# all states satisfy the domain requirements for the data defect correlation (non-missing p^s,ps\hat p_s, p_sp^​s​,ps​ and 0<ns<Ns0 < n_s < N_s0<ns​<Ns​)
# therefore no states are excluded from the DDC analysis

# compute (2.4) DO_s = (1 - f_s) / f_s, data over-quantity, amplification factor that makes large N dangerous when rho not 0
val_mergedtruth_TH["DO_s"] = (1.0 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["f_s"]

# Compute per-state DDC estimates (4.7) 
# Meng’s question for each state rho_hat_{N,s} = ((p_hat_s - p_s) / sigma_s) * sqrt( f_s / (1 - f_s) )
# where sigma_s = sqrt( p_s (1 - p_s) ) is the population SD of the Bernoulli outcome
# below computed for both trump and harris

eps = 1e-12 

# before continuing, I verified that bias_trump already equals p_hat_trump − p_trump_true, no recomputation needed

# Let pT​ and pH​ be the true state-level vote shares for Trump and Harris, trimmed away from 0 and 1 to avoid division-by-zero
# pT and pH are the true state-level vote shares (population probabilities)
# for Trump and Harris; they correspond to Meng’s p_G in equations (4.6)–(4.7)
# and are used to compute sigma_G, odds O_G, and the data defect correlation

# Trump
# pT = the true probability that a randomly chosen voter in state s voted for Trump
pT = val_mergedtruth_TH["p_trump_true"].clip(eps, 1.0 - eps)
val_mergedtruth_TH["sigma_trump"] = np.sqrt(pT * (1.0 - pT))
val_mergedtruth_TH["rho_hat_trump"] = (val_mergedtruth_TH["bias_trump"] / val_mergedtruth_TH["sigma_trump"]) * np.sqrt(val_mergedtruth_TH["f_s"] / (1.0 - val_mergedtruth_TH["f_s"]))

# Harris
# pH = the true probability that a randomly chosen voter in state s voted for Harris
pH = val_mergedtruth_TH["p_harris_true"].clip(eps, 1.0 - eps)
val_mergedtruth_TH["sigma_harris"] = np.sqrt(pH * (1.0 - pH))
val_mergedtruth_TH["rho_hat_harris"] = (val_mergedtruth_TH["bias_harris"] / val_mergedtruth_TH["sigma_harris"]) * np.sqrt(val_mergedtruth_TH["f_s"] / (1.0 - val_mergedtruth_TH["f_s"]))

# save per-state DDC outputs
val_mergedtruth_TH.to_csv("state_level_rho_hat_trump_harris_validated.csv", index=False)



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



# ###### additional values (NEED TO MAP TO PAGE)
# # summary table of bias and n for validated sample
# val_summary = val_m[["state_name", "n", "p_hat", "p_trump_true", "bias", "abs_bias"]].sort_values("abs_bias", ascending=False)
# print("\nTop 10 states by absolute bias (validated):")
# print(val_summary.head(10).to_string(index=False))
# val_summary.to_csv("validated_state_bias_summary.csv", index=False)

# # mean bias and RMSE
# mean_bias = val_m["bias"].mean()
# rmse = np.sqrt((val_m["bias"]**2).mean())
# print(f"\nValidated sample mean bias: {mean_bias:.4f}, RMSE: {rmse:.4f}")