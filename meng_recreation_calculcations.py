import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data from pipeline
cces = pd.read_csv("data/cces2024_cleaned_mengrep.csv")
truth_raw = pd.read_csv("data/true_votes_by_state_mengrep.csv")

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
plt.savefig("figures/mengrep_fig4_trump_likely_binary.png", dpi=300)
#plt.show()

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
plt.savefig("figures/mengrep_fig4_harris_likely_binary.png", dpi=300)
#plt.show()


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
likely_mergedtruth_weighted_T.to_csv("data/mengrep_fig4_state_estimates_likely_weighted_trump_vs_truth.csv", index=False)

# Weighted likely panel for Harris
likely_est_weighted_H = state_turnout_weighted(cces, weight_col="lv_weight", value_col="X_harris")
likely_mergedtruth_weighted_H = likely_est_weighted_H.merge(truth, on="state_name", how="left")
likely_mergedtruth_weighted_H["color"] = likely_mergedtruth_weighted_H["state_name"].apply(assign_color)
likely_mergedtruth_weighted_H.to_csv("data/mengrep_fig4_state_estimates_likely_weighted_harris_vs_truth.csv", index=False)


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
plt.savefig("figures/mengrep_fig4_trump_likely_weighted.png", dpi=300)
#plt.show()

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
plt.savefig("figures/mengrep_fig4_harris_likely_weighted.png", dpi=300)
#plt.show()

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
val_mergedtruth_TH.to_csv("data/mengrep_ddc_state_validated_combined_vs_truth.csv", index=False)

# all states satisfy the domain requirements for the data defect correlation (non-missing p^s,ps\hat p_s, p_sp^​s​,ps​ and 0<ns<Ns0 < n_s < N_s0<ns​<Ns​)
# therefore no states are excluded from the DDC analysis

# compute (2.4) DO_s = (1 - f_s) / f_s, data over-quantity, amplification factor that makes large N dangerous when rho not 0
val_mergedtruth_TH["DO_s"] = (1.0 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["f_s"]

###### Compute per-state DDC estimates (4.7) 
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

# save per state DDC outputs
val_mergedtruth_TH.to_csv("data/mengrep_ddc_state_level_validated_vs_truth.csv", index=False)

###### Figure 5 from meng: "Histograms of state-level data defect correlations assessed by using the validated voter
# data: Clinton's supporters (left) versus Trump’s supporters (right). The numbers in boxes show
# "mean ± 2 standard error"

def histogram_maker_for_figure5(values):
    """
    Compute the mean and 2 standard errors of the mean for states

    center = mean across states
    spread indicator = +/- 2 * (SD / sqrt(number of states))
    """

    values_array = np.asarray(values)

    values_array = values_array[~np.isnan(values_array)]

    # Number of states/jurisdictions included in the summary
    num_states = len(values_array)

    # If there are 0 or 1 states, the standard deviation and SE are undefined
    if num_states <= 1:
        return (np.nan, np.nan, num_states)

    # Mean of the quantity across states (e.g., mean of rho_hat_N)
    mean_value = float(np.mean(values_array))

    # sample standard deviation across states (ddof=1 = sample SD, not population SD)
    # matches Meng's use of SD across states when forming the standard error 
    sd_across_states = float(np.std(values_array, ddof=1))

    # standard error of the mean across states: SD / sqrt(number of states)
    se_of_mean = sd_across_states / np.sqrt(num_states)

    # returns mean across states, 2 * standard error of that mean , number of states used
    return (mean_value, 2.0 * se_of_mean, num_states)

# state level data defect correlation estimate
rhoH = val_mergedtruth_TH["rho_hat_harris"].values
rhoT = val_mergedtruth_TH["rho_hat_trump"].values

# mean_value_of_rhoT = mean of the state-level Trump data defect correlations, vertical dashed line

mean_value_of_rhoH, SE_plusminus2_H, number_of_states_used_H = histogram_maker_for_figure5(rhoH)
mean_value_of_rhoT, SE_plusminus2_T, number_of_states_used_T = histogram_maker_for_figure5(rhoT)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Harris panel (left)
axes[0].hist(rhoH, bins=15, edgecolor="black")
axes[0].axvline(0, linestyle="--",linewidth=1, color="red")
axes[0].axvline(mean_value_of_rhoH, linestyle="--", linewidth=1)
axes[0].set_title("Harris: distribution of state-level $\\hat\\rho_N$ (validated voters)")
axes[0].set_xlabel("$\\hat\\rho_N$")
axes[0].set_ylabel("Number of states / jurisdictions")
axes[0].text(
    0.98, 0.95,
    f"mean +/- 2 s.e.\n{mean_value_of_rhoH:.4f} ± {SE_plusminus2_H:.4f}\n(S={number_of_states_used_H})",
    transform=axes[0].transAxes,
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

# Trump panel (right)
axes[1].hist(rhoT, bins=15, edgecolor="black")
axes[1].axvline(0, linestyle="--",linewidth=1, color="red")
axes[1].axvline(mean_value_of_rhoT, linestyle="--", linewidth=1)
axes[1].set_title("Trump: distribution of state-level $\\hat\\rho_N$ (validated voters)")
axes[1].set_xlabel("$\\hat\\rho_N$")
axes[1].text(
    0.98, 0.95,
    f"mean +/- 2 s.e.\n{mean_value_of_rhoT:.4f} ± {SE_plusminus2_T:.4f}\n(S={number_of_states_used_T})",
    transform=axes[1].transAxes,
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

plt.suptitle("Figure 5 Replication (2024): Histograms of state-level data defect correlation $\\hat\\rho_N$")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig5_ddc_histograms.png", dpi=300)
#plt.show()

###### Figure 8
# goal of Figure 8 is to show the state level data defect correlations and overlay the theoretical feasible bounds implied by Meng’s inequality (2.9)

# compute odds O_G used in Meng bound (2.9)
# Meng defines odds for a bernoulli population outcome G O_G = p_G / (1 - p_G)
# where p_G is the true state-level prevalence of the outcome

# compute O_trump and O_harris from true state vote shares, use pT and pH from above
val_mergedtruth_TH["O_trump"] = pT / (1.0 - pT)   # odds that a random voter is Trump vs not Trump
val_mergedtruth_TH["O_harris"] = pH / (1.0 - pH)  # odds that a random voter is Harris vs not Harris

# Meng feasible bounds for rho (Equation (2.9) provides bounds on the data defect correlation \rho_{G,R} or \rho_N in the paper’s shorthand
# in terms of O_G (odds of the outcome G in the population) and DO = (1 - f) / f  (Meng 2.4)

def meng_bounds_2_9(O_G, DO):
    """
    Implements Meng inequality bounds (2.9) for correlation by state

    Inputs:
    O_G : odds p/(1-p) for outcome G in the population per state
    DO  : data over-quantity (1-f)/f (per state), Meng 2.4
    
    Outputs: rho_lb, rho_ub : lower and upper feasible bounds for rho in Meng 2.9
    """

    OG_DO = O_G * DO

    # Upper bound: rho_ub = min(sqrt(O_G*DO), 1/sqrt(O_G*DO))
    rho_ub = np.minimum(np.sqrt(OG_DO), 1.0 / np.sqrt(OG_DO))

    # Lower bound: rho_lb = -min(sqrt(DO/O_G), sqrt(O_G/DO))
    rho_lb = -np.minimum(np.sqrt(DO / O_G), np.sqrt(O_G / DO))

    return rho_lb, rho_ub

# apply bounds per state
val_mergedtruth_TH["rho_lb_trump"], val_mergedtruth_TH["rho_ub_trump"] = meng_bounds_2_9(val_mergedtruth_TH["O_trump"], val_mergedtruth_TH["DO_s"])
val_mergedtruth_TH["rho_lb_harris"], val_mergedtruth_TH["rho_ub_harris"] = meng_bounds_2_9(val_mergedtruth_TH["O_harris"], val_mergedtruth_TH["DO_s"])

# plot Figure 8, each panel shows upper and lower bounds from 2.9 and rho_hat from 4.7
plot_df = val_mergedtruth_TH.copy()
plot_df["color"] = plot_df["state_name"].apply(assign_color)

# match meng's x axis
plot_df["log10_N"] = np.log10(plot_df["N_state"])

# sort by log10_N so points run left-->right in increasing population (like Meng)
plot_df = plot_df.sort_values("log10_N").reset_index(drop=True)

# keep an integer index for ordering reference
plot_df["x_order"] = np.arange(len(plot_df))

# get the overall mean rho
mean_rho_trump = np.nanmean(plot_df["rho_hat_trump"])
mean_rho_harris = np.nanmean(plot_df["rho_hat_harris"])


fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

# matching meng design
line_alpha = 0.6 
point_alpha = 0.9 
small_point_size = 30
big_point_size = 90

# Harris panel
ax = axes[0]

# draw vertical dashed line from lower bound to upper bound
for _, row in plot_df.iterrows():
    x = row["log10_N"]
    lb = row["rho_lb_harris"]
    ub = row["rho_ub_harris"]
    # draw vertical dashed line for interval from 2.9
    ax.vlines(
        x,
        ymin=lb,
        ymax=ub,
        colors='gray',
        linestyles='dashed',
        linewidth=0.9,
        alpha=line_alpha,
        zorder=1
    )

# plot marker for rho_hat at each state's position
ax.scatter(
    plot_df["log10_N"],
    plot_df["rho_hat_harris"],
    s=small_point_size,
    c=plot_df["color"],
    alpha=point_alpha,
    edgecolor='none',
    zorder=3,
    label=r'$\hat\rho_N$ (empirical)'
)

ax.scatter(
    plot_df["log10_N"],
    plot_df["rho_hat_harris"],
    s=big_point_size,
    facecolors=plot_df["color"],
    edgecolors='black',
    linewidths=0.6,
    alpha=0.95,
    zorder=4
)

# horizontal references: rho=0 (no selection bias) and mean rho
ax.axhline(0.0, color='red', linestyle='--', linewidth=1.0, label=r'$\rho=0$ (no bias)')
ax.axhline(mean_rho_harris, color='black', linestyle='--', linewidth=1.0, label=f'mean(ρ̂)={mean_rho_harris:.4f}')

# labels and formatting
ax.set_title("Harris: $\\hat\\rho_N$ with theoretical bounds (Meng eq. (2.9))")
ax.set_xlabel("log10 (Total voters, $N_s$)")
ax.set_ylabel(r"$\hat\rho_N$")
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

ax.legend(loc='lower left', fontsize=8)

# Trump panel
ax = axes[1]

# draw vertical dashed line from lower bound to upper bound
for _, row in plot_df.iterrows():
    x = row["log10_N"]
    lb = row["rho_lb_trump"]
    ub = row["rho_ub_trump"]
    # draw vertical dashed line for interval from 2.9
    ax.vlines(
        x,
        ymin=lb,
        ymax=ub,
        colors='gray',
        linestyles='dashed',
        linewidth=0.9,
        alpha=line_alpha,
        zorder=1
    )

# plot marker for rho_hat at each state's position
ax.scatter(
    plot_df["log10_N"],
    plot_df["rho_hat_trump"],
    s=small_point_size,
    c=plot_df["color"],
    alpha=point_alpha,
    edgecolor='none',
    zorder=3
)

ax.scatter(
    plot_df["log10_N"],
    plot_df["rho_hat_trump"],
    s=big_point_size,
    facecolors=plot_df["color"],
    edgecolors='black',
    linewidths=0.6,
    alpha=0.95,
    zorder=4
)

# horizontal references: rho=0 (no selection bias) and mean rho
ax.axhline(0.0, color='red', linestyle='--', linewidth=1.0, label=r'$\rho=0$ (no bias)')
ax.axhline(mean_rho_trump, color='black', linestyle='--', linewidth=1.0, label=f'mean(ρ̂)={mean_rho_trump:.4f}')

# labels and formatting
ax.set_title("Trump: $\\hat\\rho_N$ with theoretical bounds (Meng eq. (2.9))")
ax.set_xlabel("log10 (Total voters, $N_s$)")
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
ax.legend(loc='lower left', fontsize=8)

plt.suptitle("Figure 8 style: state-level $\\hat\\rho_N$ with Meng feasible bounds (2.9)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig8_ddc_bounds.png", dpi=300)
#plt.show()

# save values in case needed later
figure8_datacheck = plot_df[[
    "state_name", "N_state", "n", "f_s", "DO_s",
    "p_trump_true", "p_hat_trump", "bias_trump", "rho_hat_trump", "rho_lb_trump", "rho_ub_trump",
    "p_harris_true", "p_hat_harris", "bias_harris", "rho_hat_harris", "rho_lb_harris", "rho_ub_harris"
]].copy()

figure8_datacheck.to_csv("data/mengrep_fig8_ddc_bounds_and_estimates_vs_truth.csv", index=False)


########################################################################################
######################## LAW OF LARGE POPULATIONS, Figures 6 and 7 #####################
########################################################################################

# This section matches what Meng does in his section 4.2 and figures 6 adn 7
# Figure 6 uses the "nominal Z-score" from Meng (3.1), denoted Z_{n,N}, and the log–log regression motivated by Meng 4.8 and 4.9
# Figure 7 uses the conventional Z-score Z_n from Meng (3.9) and checks how often the usual |Z_n| <= 2 "95% CI region" contains the truth, and whether the misses grow with N

# make copy of data for law of large population (LLP)
llp_df = val_mergedtruth_TH.copy()

# calculate OLS estimates 
def ols_slope_and_se(x, y):
    """
    b_hat: slope estimate
    se_b: standard error of slope 
    a_hat: intercept

    using y = a+bx form
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # matrix with intercept [1, x]
    X = np.column_stack([np.ones_like(x), x])

    # OLS coefficients (X'X)^{-1} X'y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a_hat, b_hat = beta[0], beta[1]

    # residuals and residual variance, s^2 = RSS/(n-2)
    resid = y - (a_hat + b_hat * x)
    n_obs = len(x)
    rss = np.sum(resid**2)
    s2 = rss / (n_obs - 2)

    # Var(beta) = s^2 (X'X)^{-1}, slope is element [1,1]
    XtX_inv = np.linalg.inv(X.T @ X)
    se_b = np.sqrt(s2 * XtX_inv[1, 1])

    return (b_hat, se_b, a_hat)

##### info needed for figure 6, log|Z_{n,N}| vs log N regression 
# Meng 3.1: Z_{n,N} = sqrt(N - 1) * rho_{R,G}
llp_df["Z_nN_trump"]  = np.sqrt(llp_df["N_state"] - 1.0) * llp_df["rho_hat_trump"]
llp_df["Z_nN_harris"] = np.sqrt(llp_df["N_state"] - 1.0) * llp_df["rho_hat_harris"]

# use log10 to match Meng’s plotting
# log–log variables for Meng 4.9, log|Z_{n,N}| and log N
llp_df["log10_N"] = np.log10(llp_df["N_state"])
llp_df["log10_absZ_nN_trump"]  = np.log10(np.abs(llp_df["Z_nN_trump"]))
llp_df["log10_absZ_nN_harris"] = np.log10(np.abs(llp_df["Z_nN_harris"]))

# fit Meng 4.9 separately for Harris and Trump, log|Z_{n,N}| = alpha + beta log N
beta_T, se_beta_T, alpha_T = ols_slope_and_se(llp_df["log10_N"], llp_df["log10_absZ_nN_trump"])
beta_H, se_beta_H, alpha_H = ols_slope_and_se(llp_df["log10_N"], llp_df["log10_absZ_nN_harris"])

# plot based on log10
x_line = np.linspace(llp_df["log10_N"].min(), llp_df["log10_N"].max(), 200)

##### plotting figure 6, log log plot of log|Z_{n,N}| and log N with OLS line
# predicted values y_hat=alpha+beta*x
yhat_line_T = alpha_T + beta_T * x_line
yhat_line_H = alpha_H + beta_H * x_line

# colors for plotting
llp_df["color"] = llp_df["state_name"].apply(assign_color)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Harris panel
axes[0].scatter(
    llp_df["log10_N"],
    llp_df["log10_absZ_nN_harris"],
    c=llp_df["color"],
    alpha=0.85,
    edgecolors="black",
    linewidths=0.3
)
axes[0].plot(x_line, yhat_line_H, linestyle="--", linewidth=1)
axes[0].set_title("Harris (validated voters)")
axes[0].set_xlabel(r"$\log_{10}(N_s)$  (state turnout)")
axes[0].set_ylabel(r"$\log_{10}(|Z_{n,N,s}|)$")

# slope from beta
axes[0].text(
    0.02, 0.95,
    f"OLS slope beta = {beta_H:.3f} (SE {se_beta_H:.3f})",
    transform=axes[0].transAxes,
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

# Trump panel
axes[1].scatter(
    llp_df["log10_N"],
    llp_df["log10_absZ_nN_trump"],
    c=llp_df["color"],
    alpha=0.85,
    edgecolors="black",
    linewidths=0.3
)
axes[1].plot(x_line, yhat_line_T, linestyle="--", linewidth=1)
axes[1].set_title("Trump (validated voters)")
axes[1].set_xlabel(r"$\log_{10}(N_s)$  (state turnout)")

axes[1].text(
    0.02, 0.95,
    f"OLS slope beta = {beta_T:.3f} (SE {se_beta_T:.3f})",
    transform=axes[1].transAxes,
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

plt.suptitle("Law of Large Populations (Figure 6 replication): log |Z_{n,N}| vs log N")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig6_llp_loglog_regression.png", dpi=300)
#plt.show()

##### figure 7: compares to regular Z score used in SRS
# In each state, for a bernoulli outcome with true proportion p_s and sample mean p_hat_s
# the standard error under SRS is SE_srs = sqrt( p_hat_s (1 - p_hat_s) / n_s )
# which gives the z score Z_n = (p_hat_s - p_s) / SE_srs

# Meng’s point is that even if the confidence interval logic looks fine under SRS assumptions, |Z_n| can blow up with N
# because bias dominates variance under nonresponse, which is his Big Data Paradox

def regular_zscore_3_9(p_hat, p_true, n):
    # Z_n,s = (p_hat_s - p_true_s) / sqrt( p_hat_s (1 - p_hat_s) / n_s )
    p_hat = np.asarray(p_hat, dtype=float)
    p_true = np.asarray(p_true, dtype=float)
    n = np.asarray(n, dtype=float)

    var = (p_hat * (1.0 - p_hat)) / n
    se = np.sqrt(var)
    return (p_hat - p_true) / se

# compute z score by state
llp_df["Z_n_s_trump"] = regular_zscore_3_9(p_hat=llp_df["p_hat_trump"], p_true=llp_df["p_trump_true"], n=llp_df["n"])
llp_df["Z_n_s_harris"] = regular_zscore_3_9(p_hat=llp_df["p_hat_harris"], p_true=llp_df["p_harris_true"], n=llp_df["n"])

# determine is state covered, indidctaor if |Z_n,s| <= 2
llp_df["cover_rate_H"] = (np.abs(llp_df["Z_n_s_harris"]) <= 2.0)
llp_df["cover_rate_T"] = (np.abs(llp_df["Z_n_s_trump"]) <= 2.0)

# plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Harris panel
axes[0].scatter(
    llp_df["log10_N"],
    llp_df["Z_n_s_harris"],
    c=llp_df["color"],
    alpha=0.85, edgecolors="black", linewidths=0.3)
axes[0].axhspan(-2, 2, alpha=0.15)  # the “nominal 95%” band (Meng’s visual point)
axes[0].axhline(0, linestyle="--", linewidth=1)
axes[0].set_title("Harris (validated voters)")
axes[0].set_xlabel(r"$\log_{10}(N_s)$  (state turnout)")
axes[0].set_ylabel(r"Regular Z Score $Z_{n,s}$ ")

# add avg coverage
cover_rate_mean_H = llp_df["cover_rate_H"].mean()
axes[0].text(
    0.02, 0.95,
    f"Share with |Z_n|<=2: {cover_rate_mean_H:.2%}",
    transform=axes[0].transAxes,
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

# Trump panel
axes[1].scatter(
    llp_df["log10_N"],
    llp_df["Z_n_s_trump"],
    c=llp_df["color"],
    alpha=0.85, edgecolors="black", linewidths=0.3)
axes[1].axhspan(-2, 2, alpha=0.15)
axes[1].axhline(0, linestyle="--", linewidth=1)
axes[1].set_title("Trump (validated voters)")
axes[1].set_xlabel(r"$\log_{10}(N_s)$  (state turnout)")

# add avg coverage
cover_rate_mean_T = llp_df["cover_rate_T"].mean()
axes[1].text(
    0.02, 0.95,
    f"Share with |Z_n|<=2: {cover_rate_mean_T:.2%}",
    transform=axes[1].transAxes,
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
)

plt.suptitle("Law of Large Populations (Figure 7 replication): Conventional Z_n vs log N")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/mengrep_fig7_llp_zscore_coverage.png", dpi=300)
#plt.show()

# save data
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

# Trump report
coverage_report(llp_df, "Z_n_s_trump", "cover_rate_T", "Trump")

# Harris report
coverage_report(llp_df, "Z_n_s_harris", "cover_rate_H", "Harris")

# compare sets
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
# MSE calculations (2.4)
# Design effect (3.2)
# SRS variance for comparison

# calculate S^2_G, finite-population corrected variance
# Meng 2.5: S^2_G = [N/(N-1)] x sigma^2_G
#   sigma^2_G = p(1-p) is the population variance
#   the factor N/(N-1) is the finite population correction
#   S^2_G is the unbiased population variance used in SRS formulas
#   for large N, S^2_G approx sigma^2_G
#   in finite population sampling, we need to account for sampling without replacement
#   so this adjustment ensures unbiased variance estimation when sample size is non trivial
#   relative to population size
val_mergedtruth_TH["S2_trump"] = (
    val_mergedtruth_TH["N_state"] / (val_mergedtruth_TH["N_state"] - 1)
    * val_mergedtruth_TH["sigma_trump"]**2
)
val_mergedtruth_TH["S2_harris"] = (
    val_mergedtruth_TH["N_state"] / (val_mergedtruth_TH["N_state"] - 1)
    * val_mergedtruth_TH["sigma_harris"]**2
)

# calculate SRS variance, 2.5: Var_SRS(G_n) = [(1-f)/n] x S^2_G
# what variance would be under perfect random sampling
#   is the variance of the sample mean under SRS
#   (1-f) is the finite population correction
#   when f -> 0 (tiny sample), FPC -> 1, so Var approx sigma^2/n 
#   when f -> 1 (census), FPC _> 0, so Var -> 0, no sampling error
#   sigma²_G/n is the variance of sample mean from 110
#   serves as benchmark, how good would it be with perfect random sampling
#   compare actual MSE to this benchmark to see how much damage the biased R-mechanism (selection bias) causes
#   if MSE >> Var_SRS, we have a serious problem
val_mergedtruth_TH["Var_SRS_trump"] = (
    (1 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["n"]
    * val_mergedtruth_TH["S2_trump"]
)
val_mergedtruth_TH["Var_SRS_harris"] = (
    (1 - val_mergedtruth_TH["f_s"]) / val_mergedtruth_TH["n"]
    * val_mergedtruth_TH["S2_harris"]
)

# standard error, square root of variance
# margin of error if SRS
# SE_SRS = 0.01 means estimate would typically be within +/-1% of truth
val_mergedtruth_TH["SE_SRS_trump"] = np.sqrt(val_mergedtruth_TH["Var_SRS_trump"])
val_mergedtruth_TH["SE_SRS_harris"] = np.sqrt(val_mergedtruth_TH["Var_SRS_harris"])

# calculate DU, degree of Uncertainty = population variance, problem difficulty
# part of 2.4: DU = sigma^2_G
#   DU measures how difficult the estimation problem is
#   for binary outcome, sigma^2_G = p(1-p), maximized at p=0.5
#   higher variance means harder to estimate accurately
#   only factor determined by the population, not by sampling design
#   cannot control DU through better sampling, but we can reduce it with stratification or covariate adjustment (5.1)
#   lower DU means easier problem, 50/50 race is max difficulty and would get 0.25
val_mergedtruth_TH["DU_trump"] = val_mergedtruth_TH["sigma_trump"]**2
val_mergedtruth_TH["DU_harris"] = val_mergedtruth_TH["sigma_harris"]**2

# calculate actual MSE, from 2.4 MSE = DI x DO x DU
# where DI = rho^2_{R,G}, DO = (1-f)/f, DU = sigma^2_G

val_mergedtruth_TH["MSE_trump"] = (
    val_mergedtruth_TH["rho_hat_trump"]**2 # DI
    * val_mergedtruth_TH["DO_s"] # DO
    * val_mergedtruth_TH["DU_trump"] # DU
)
val_mergedtruth_TH["MSE_harris"] = (
    val_mergedtruth_TH["rho_hat_harris"]**2
    * val_mergedtruth_TH["DO_s"]
    * val_mergedtruth_TH["DU_harris"]
)

# RMSE, typical error in our estimate 
val_mergedtruth_TH["RMSE_trump"] = np.sqrt(val_mergedtruth_TH["MSE_trump"])
val_mergedtruth_TH["RMSE_harris"] = np.sqrt(val_mergedtruth_TH["MSE_harris"])

# calculate design effect
# 3.2: Deff = MSE / Var_SRS = (N-1) x DI
# how many times worse the actual sampling is vs perfect SRS
# need deff times the sample size to match SRS precision
# if rho doesn't shrink with N, the deff grows with N
val_mergedtruth_TH["Deff_trump"] = (
    (val_mergedtruth_TH["N_state"] - 1) * val_mergedtruth_TH["rho_hat_trump"]**2
)
val_mergedtruth_TH["Deff_harris"] = (
    (val_mergedtruth_TH["N_state"] - 1) * val_mergedtruth_TH["rho_hat_harris"]**2
)

# summary statistics
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

# worst states by design effect
print("\nTop five by design effect, worst precision loss\n")
print("TRUMP:")
top5_trump = val_mergedtruth_TH.nlargest(5, "Deff_trump")[
    ["state_name", "N_state", "rho_hat_trump", "Deff_trump", "RMSE_trump"]
].copy()
top5_trump["RMSE_pct"] = top5_trump["RMSE_trump"] * 100
print(
    top5_trump[
        ["state_name", "N_state", "rho_hat_trump", "Deff_trump", "RMSE_pct"]
    ].to_string(index=False)
)

print("\nHARRIS:")
top5_harris = val_mergedtruth_TH.nlargest(5, "Deff_harris")[
    ["state_name", "N_state", "rho_hat_harris", "Deff_harris", "RMSE_harris"]
].copy()
top5_harris["RMSE_pct"] = top5_harris["RMSE_harris"] * 100
print(
    top5_harris[
        ["state_name", "N_state", "rho_hat_harris", "Deff_harris", "RMSE_pct"]
    ].to_string(index=False)
)

# trump worst MSE
print("Trump states with largest MSE")

worst_mse_trump = val_mergedtruth_TH.nlargest(5, "MSE_trump")[["state_name", "MSE_trump", "rho_hat_trump", "f_s", "sigma_trump"]]

print(worst_mse_trump
    .assign(sigma2=lambda df: df["sigma_trump"] ** 2)
    .rename(columns={
        "state_name": "State",
        "MSE_trump": r"$\mathrm{MSE}$",
        "rho_hat_trump": r"$\rho$",
        "f_s": r"$f$",
        "sigma2": r"$\sigma^2$",
    }).to_string(index=False))

print("\nInterpretation guide:")
print(r"High $|\rho| \;\to\;$ strong selection bias")
print(r"Low $f \;\to\;$ low response rate")
print(r"High $\sigma^2 \;\to\;$ close race ($p \approx 0.5$)")

# harris worst MSE
print("\nHarris states with largest MSE")

worst_mse_harris = val_mergedtruth_TH.nlargest(5, "MSE_harris")[["state_name", "MSE_harris", "rho_hat_harris", "f_s", "sigma_harris"]]

print(
    worst_mse_harris
    .assign(sigma2=lambda df: df["sigma_harris"] ** 2)
    .rename(columns={
        "state_name": "State",
        "MSE_harris": r"$\mathrm{MSE}$",
        "rho_hat_harris": r"$\rho$",
        "f_s": r"$f$",
        "sigma2": r"$\sigma^2$",
    }).to_string(index=False))

########################################################################################
######################## Effective Sample Size, by state ###############################
########################################################################################

# mapping to Meng
# state_name        → (state identifier)
# N_state           → N (population size - actual voter turnout per state)
# n                 → n (sample size per state)
# f_s               → f (sampling rate = n/N)
# DO_s              → DO (Dropout Odds = (1-f)/f) [meng 2.4]
# p_hat_trump       → G_n (sample proportion) [meng 2.1]
# p_trump_true      → G_N (population/true proportion)
# p_hat_harris      → G_n (sample proportion)
# p_harris_true     → G_N (population/true proportion)
# bias_trump        → G_n - G_N (actual bias)
# bias_harris       → G_n - G_N (actual bias)
# sigma_trump       → sigma_G (population std dev = sqrt(p(1-p))) [Meng 2.3]
# sigma_harris      → sigma_G (population std dev = sqrt(p(1-p)))
# rho_hat_trump     → ρ_R,G (data defect correlation) [Meng 4.7]
# rho_hat_harris    → ρ_R,G (data defect correlation)

# dataset to use
eff_samplesize_df = val_mergedtruth_TH.copy()

# data defect index (DI)
# for each state s and candidate G in {Trump, Harris}, DI_{s,G} = (rho_hat_{s,G})^2 because DI is governed by rho^2 in Meng’s effective sample size identity in 3.5
eff_samplesize_df["DI_trump"]  = eff_samplesize_df["rho_hat_trump"]**2
eff_samplesize_df["DI_harris"] = eff_samplesize_df["rho_hat_harris"]**2

# data quantity term, 2.4: DO = (1-f)/f, already calculated as DO_s for each state

# n*_eff = (DO * DI)^(-1), from inequality 3.6 effective sample size
# this is the upper bound version Meng uses in practice, 3.2
eff_samplesize_df["n_star_eff_trump"] = 1.0 / (eff_samplesize_df["DO_s"] * eff_samplesize_df["DI_trump"])
eff_samplesize_df["n_star_eff_harris"] = 1.0 / (eff_samplesize_df["DO_s"] * eff_samplesize_df["DI_harris"])

# n_eff, from 3.5: n_eff = n*_eff / ( 1 + (n*_eff - 1)/(N - 1) )
# under SRS, n_eff is about n, if DI=0, n_eff = N rather than infinity
# in practice, Meng uses n*_eff but includes n_eff for completeness
eff_samplesize_df["n_eff_trump"] = eff_samplesize_df["n_star_eff_trump"] / (
    1.0 + (eff_samplesize_df["n_star_eff_trump"] - 1.0) / 
    (eff_samplesize_df["N_state"] - 1.0)
)

eff_samplesize_df["n_eff_harris"] = eff_samplesize_df["n_star_eff_harris"] / (
    1.0 + (eff_samplesize_df["n_star_eff_harris"] - 1.0) / 
    (eff_samplesize_df["N_state"] - 1.0)
) 

# 4.5: margin of error for binary outcomes 
# half-width of 95% CI = 2*sqrt(p(1-p)/n_s)  <= 1/sqrt(n_s)
# Me = 2 x sqrt(p(1-p)/n*_eff)
# If we replace n_s by n*_eff, we get an effective margin of error that reflects the selection bias implied by rho
# huge n can still imply an MoE comparable to a much smaller SRS once n_eff collapses

# candidate specific sigma^2 = p(1-p) using TRUE p (same sigma used in rho_hat construction)
# below is same as eff_samplesize_df["sigma_candidate"]**2
eff_samplesize_df["sigma2_trump"]  = eff_samplesize_df["p_trump_true"]  * (1.0 - eff_samplesize_df["p_trump_true"])
eff_samplesize_df["sigma2_harris"] = eff_samplesize_df["p_harris_true"] * (1.0 - eff_samplesize_df["p_harris_true"])

# Meng 4.5 implied half-widths using n*_eff
eff_samplesize_df["Me95_star_trump"]  = 2.0 * np.sqrt(eff_samplesize_df["sigma2_trump"]  / eff_samplesize_df["n_star_eff_trump"])
eff_samplesize_df["Me95_star_harris"] = 2.0 * np.sqrt(eff_samplesize_df["sigma2_harris"] / eff_samplesize_df["n_star_eff_harris"])

# Meng’s simple upper bound in 4.5, margin of error <= 1/sqrt(n_s), Me upper bound
eff_samplesize_df["Me95_star_upper_trump"]  = 1.0 / np.sqrt(eff_samplesize_df["n_star_eff_trump"])
eff_samplesize_df["Me95_star_upper_harris"] = 1.0 / np.sqrt(eff_samplesize_df["n_star_eff_harris"])

# save outmputs
effective_sample_size_outputs = eff_samplesize_df[[
    "state_name", "N_state", "n", "f_s", "DO_s", "rho_hat_trump", "DI_trump", "n_star_eff_trump", "n_eff_trump", "Me95_star_trump", "Me95_star_upper_trump",
    "rho_hat_harris", "DI_harris", "n_star_eff_harris", "n_eff_harris", "Me95_star_harris", "Me95_star_upper_harris",
]].copy()

effective_sample_size_outputs.to_csv("data/mengrep_effss_by_state_vs_truth.csv", index=False)

# interesting values
print("\nBottom 10 states by n*_eff (Trump):")
print(effective_sample_size_outputs.sort_values("n_star_eff_trump").head(10)[
    ["state_name", "n_star_eff_trump", "Me95_star_trump"]
].to_string(index=False))

print("\nBottom 10 states by n*_eff (Harris):")
print(effective_sample_size_outputs.sort_values("n_star_eff_harris").head(10)[
    ["state_name", "n_star_eff_harris", "Me95_star_harris"]
].to_string(index=False))

# get columns from eff_samplesize_df that aren't already in val_mergedtruth_TH
new_cols = [col for col in eff_samplesize_df.columns if col not in val_mergedtruth_TH.columns]

# add them to val_mergedtruth_TH
for col in new_cols:
    val_mergedtruth_TH[col] = eff_samplesize_df[col]

# save comprehensive state level dataset with all calculations
val_mergedtruth_TH.to_csv("data/mengrep_all_vals_state_level.csv", index=False)

########################################################################################
####### Overall Effective Sample Size Pooled Across States, and additional values ######
########################################################################################

# Meng Section 4.1 combines all polls to get overall n*_eff

# aggregate state-level data to get overall quantities
# total sample size across all states
n_total = eff_samplesize_df["n"].sum()

# total actual voter population across all states 
N_total = eff_samplesize_df["N_state"].sum()

# overall sampling rate, Meng assumed this was 0.01
f_total = n_total / N_total

# overall dropout odds, 2.4
DO_total = (1.0 - f_total) / f_total

# overall sample proportions, weighted by state sample size, proportion from pooling all state samples, 2.1 aggregated by state
p_hat_trump_total = (eff_samplesize_df["p_hat_trump"] * eff_samplesize_df["n"]).sum() / n_total
p_hat_harris_total = (eff_samplesize_df["p_hat_harris"] * eff_samplesize_df["n"]).sum() / n_total

# overall popular vote, weighted by state population, true national proportion of votes
p_trump_true_total = (eff_samplesize_df["p_trump_true"] * eff_samplesize_df["N_state"]).sum() / N_total
p_harris_true_total = (eff_samplesize_df["p_harris_true"] * eff_samplesize_df["N_state"]).sum() / N_total

# overall bias, Meng's G_n - G_N, bias of 0.01 overesimtate by 1 percentage point
bias_trump_total = p_hat_trump_total - p_trump_true_total
bias_harris_total = p_hat_harris_total - p_harris_true_total

# overall sigma_G = sqrt(p(1-p)) using true national proportion, sigma_G from Meng 2.3, national sd, 0.5 max uncertainty
sigma_trump_total = np.sqrt(p_trump_true_total * (1.0 - p_trump_true_total))
sigma_harris_total = np.sqrt(p_harris_true_total * (1.0 - p_harris_true_total))

# overall rho using Meng 4.7, ρ_R,G = (G_n - G_N) / (sigma_G x sqrt((1-f)/f)) = bias / (sigma x sqrt(DO)) = (bias / sigma) x sqrt(f / (1-f)), higher more selection bias, corr between response propensity and outcome, if >0 people voting for candidate more likely to respond
rho_trump_total = (bias_trump_total / sigma_trump_total) * np.sqrt(f_total / (1.0 - f_total))
rho_harris_total = (bias_harris_total / sigma_harris_total) * np.sqrt(f_total / (1.0 - f_total))

# overall DI Meng 2.4, DI = rho^2_R,G, smaller means better quality
DI_trump_total = rho_trump_total**2
DI_harris_total = rho_harris_total**2

# overall n*_eff, Meng  3.6, n*_eff = 1/(DO x DI
n_star_eff_trump_total = 1.0 / (DO_total * DI_trump_total)
n_star_eff_harris_total = 1.0 / (DO_total * DI_harris_total)

# overall n_eff, Meng 3.5, n_eff = n*_eff / (1 + (n*_eff - 1)/(N-1))
n_eff_trump_total = n_star_eff_trump_total / (1.0 + (n_star_eff_trump_total - 1.0) / (N_total - 1.0))
n_eff_harris_total = n_star_eff_harris_total / (1.0 + (n_star_eff_harris_total - 1.0) / (N_total - 1.0))

# overall margin of error, Meng equation 4.5, Me = 2x sqrt(sigma^2_G / n*_eff), Me is half-width of 95% confidence interval, Me = 0.05 means estimate +/- 5 percentage points
# maybe to add Me_actual / Me_naive_SRS = sqrt(n / n*_eff) = sqrt(Deff)
Me_trump_total = 2.0 * np.sqrt((sigma_trump_total**2) / n_star_eff_trump_total)
Me_harris_total = 2.0 * np.sqrt((sigma_harris_total**2) / n_star_eff_harris_total)

# upper bound on Me ≤ 1/sqrt(n*_eff), get when rho = 0.5, when close race is near actual Me
Me_upper_trump_total = 1.0 / np.sqrt(n_star_eff_trump_total)
Me_upper_harris_total = 1.0 / np.sqrt(n_star_eff_harris_total)

# finite-population corrected variance S^2
sigma2_trump_national = p_trump_true_total * (1.0 - p_trump_true_total)
sigma2_harris_national = p_harris_true_total * (1.0 - p_harris_true_total)

S2_trump_national = (N_total / (N_total - 1)) * sigma2_trump_national
S2_harris_national = (N_total / (N_total - 1)) * sigma2_harris_national

# what variance would be under perfect random sampling
Var_SRS_trump_national = ((1 - f_total) / n_total) * S2_trump_national
Var_SRS_harris_national = ((1 - f_total) / n_total) * S2_harris_national

# SRS standard error
SE_SRS_trump_national = np.sqrt(Var_SRS_trump_national)
SE_SRS_harris_national = np.sqrt(Var_SRS_harris_national)

# national MSE, 2.4: MSE = DI x DO x DU
MSE_trump_national = DI_trump_total * DO_total * sigma2_trump_national
MSE_harris_national = DI_harris_total * DO_total * sigma2_harris_national

RMSE_trump_national = np.sqrt(MSE_trump_national)
RMSE_harris_national = np.sqrt(MSE_harris_national)

# national design effect, 3.2: Deff = (N-1) xDI
Deff_trump_national = (N_total - 1) * DI_trump_total
Deff_harris_national = (N_total - 1) * DI_harris_total

# national long tale
national_long_table = pd.DataFrame({
    'Metric': [
        '=== population and sample value ===',
        'Population ($N$)',
        'Sample size ($n$)',
        'Sampling rate ($f$)',
        'Dropout odds ($DO$)',
        '',
        
        '=== estimates vs truth ===',
        r'Sample proportion ($\hat{p}$)',
        'True proportion ($p$)',
        r'Bias ($\hat{p} - p$)',
        r'Absolute bias ($|\hat{p} - p|$)',
        r'Standard deviation ($\sigma_G$)',
        r'Variance ($\sigma^2_G = DU$)',
        '',
        
        '=== data quality ===',
        r'Data defect correlation ($\rho_{R,G}$)',
        r'Data defect index ($DI = \rho^2$)',
        '',
        
        '=== SRS benchmark ===',
        r'Finite-corrected variance ($S^2_G$)',
        r'SRS variance ($Var_{SRS}$)',
        r'SRS standard error ($SE_{SRS}$)',
        r'SRS margin of error (95% CI)',
        '',
        
        '=== actual MSE ===',
        'Mean Squared Error (MSE)',
        'Root MSE (RMSE)',
        'RMSE (percentage points)',
        '',
        
        '=== design effect ===',
        'Design Effect (Deff)',
        'Deff interpretation',
        '',
        
        '=== effective sample size ===',
        r'$n^*_{eff}$ (upper bound)',
        r'$n_{eff}$ (finite-corrected)',
        'Sample reduction (%)',
        'Sample reduction (count)',
        '',
        
        '=== margin of error ===',
        'Margin of error (Me)',
        'Me (percentage points)',
        'Me upper bound',
        'Me upper bound (pp)',
        'Me inflation vs SRS',
        '',
    ],
    
    'Trump': [
        # population and sample values
        '',
        f'{N_total:,.0f}',
        f'{n_total:,.0f}',
        f'{f_total:.6f}',
        f'{DO_total:.4f}',
        '',
        
        # estimates vs truth
        '',
        f'{p_hat_trump_total:.6f}',
        f'{p_trump_true_total:.6f}',
        f'{bias_trump_total:+.6f}',
        f'{abs(bias_trump_total):.6f}',
        f'{sigma_trump_total:.6f}',
        f'{sigma_trump_total**2:.8f}',
        '',
        
        # data quality
        '',
        f'{rho_trump_total:+.8f}',
        f'{DI_trump_total:.10f}',
        '',
        
        # SRS benchmark
        '',
        f'{S2_trump_national:.8f}',
        f'{Var_SRS_trump_national:.10f}',
        f'{SE_SRS_trump_national:.6f}',
        f'{SE_SRS_trump_national*2:.6f}',
        '',
        
        # actual MSE
        '',
        f'{MSE_trump_national:.10f}',
        f'{RMSE_trump_national:.6f}',
        f'{RMSE_trump_national*100:.4f}',
        '',
        
        # design effect
        '',
        f'{Deff_trump_national:.4f}',
        f'{Deff_trump_national:.1f}x worse than SRS',
        '',
        
        # effective sample size
        '',
        f'{n_star_eff_trump_total:,.0f}',
        f'{n_eff_trump_total:,.0f}',
        f'{(1 - n_star_eff_trump_total/n_total)*100:.4f}%',
        f'{n_total - n_star_eff_trump_total:,.0f}',
        '',
        
        # margun of error
        '',
        f'{Me_trump_total:.6f}',
        f'{Me_trump_total*100:.4f}',
        f'{Me_upper_trump_total:.6f}',
        f'{Me_upper_trump_total*100:.4f}',
        f'{Me_trump_total/(SE_SRS_trump_national*2):.4f}x',
        '',
    ],
    
    'Harris': [
        # population and sample values
        '',
        f'{N_total:,.0f}',
        f'{n_total:,.0f}',
        f'{f_total:.6f}',
        f'{DO_total:.4f}',
        '',
        
        # estimates vs truth
        '',
        f'{p_hat_harris_total:.6f}',
        f'{p_harris_true_total:.6f}',
        f'{bias_harris_total:+.6f}',
        f'{abs(bias_harris_total):.6f}',
        f'{sigma_harris_total:.6f}',
        f'{sigma_harris_total**2:.8f}',
        '',
        
        # data quality
        '',
        f'{rho_harris_total:+.8f}',
        f'{DI_harris_total:.10f}',
        '',
        
        # SRS benchmark
        '',
        f'{S2_harris_national:.8f}',
        f'{Var_SRS_harris_national:.10f}',
        f'{SE_SRS_harris_national:.6f}',
        f'{SE_SRS_harris_national*2:.6f}',
        '',
        
        # actual MSE
        '',
        f'{MSE_harris_national:.10f}',
        f'{RMSE_harris_national:.6f}',
        f'{RMSE_harris_national*100:.4f}',
        '',
        
        # design effect
        '',
        f'{Deff_harris_national:.4f}',
        f'{Deff_harris_national:.1f}x worse than srs',
        '',
        
        # effective sample size
        '',
        f'{n_star_eff_harris_total:,.0f}',
        f'{n_eff_harris_total:,.0f}',
        f'{(1 - n_star_eff_harris_total/n_total)*100:.4f}%',
        f'{n_total - n_star_eff_harris_total:,.0f}',
        '',
        
        # mmargin of error
        '',
        f'{Me_harris_total:.6f}',
        f'{Me_harris_total*100:.4f}',
        f'{Me_upper_harris_total:.6f}',
        f'{Me_upper_harris_total*100:.4f}',
        f'{Me_harris_total/(SE_SRS_harris_national*2):.4f}x',
        '',
    ]
})

# national wide table
national_wide_table = pd.DataFrame({
    # candidate
    'candidate': ['Trump', 'Harris'],
    
    # population and sample values
    'N': [N_total, N_total],
    'n': [n_total, n_total],
    'f': [f_total, f_total],
    'DO': [DO_total, DO_total],
    
    # estimate values vs true
    'p_hat': [p_hat_trump_total, p_hat_harris_total],
    'p_true': [p_trump_true_total, p_harris_true_total],
    'bias': [bias_trump_total, bias_harris_total],
    'abs_bias': [abs(bias_trump_total), abs(bias_harris_total)],
    'sigma_G': [sigma_trump_total, sigma_harris_total],
    'sigma2_G': [sigma_trump_total**2, sigma_harris_total**2],
    
    # data quality
    'rho_R_G': [rho_trump_total, rho_harris_total],
    'DI': [DI_trump_total, DI_harris_total],
    
    # SRS benchmarks
    'S2_G': [S2_trump_national, S2_harris_national],
    'Var_SRS': [Var_SRS_trump_national, Var_SRS_harris_national],
    'SE_SRS': [SE_SRS_trump_national, SE_SRS_harris_national],
    'Me_SRS': [SE_SRS_trump_national*2, SE_SRS_harris_national*2],
    
    # actual MSE
    'MSE': [MSE_trump_national, MSE_harris_national],
    'RMSE': [RMSE_trump_national, RMSE_harris_national],
    'RMSE_pp': [RMSE_trump_national*100, RMSE_harris_national*100],
    
    # design effect
    'Deff': [Deff_trump_national, Deff_harris_national],
    
    # effective sample size
    'n_star_eff': [n_star_eff_trump_total, n_star_eff_harris_total],
    'n_eff': [n_eff_trump_total, n_eff_harris_total],
    'reduction_pct': [(1-n_star_eff_trump_total/n_total)*100, 
                      (1-n_star_eff_harris_total/n_total)*100],
    'reduction_count': [n_total - n_star_eff_trump_total, 
                        n_total - n_star_eff_harris_total],
    
    # margin of error
    'Me': [Me_trump_total, Me_harris_total],
    'Me_pp': [Me_trump_total*100, Me_harris_total*100],
    'Me_upper': [Me_upper_trump_total, Me_upper_harris_total],
    'Me_upper_pp': [Me_upper_trump_total*100, Me_upper_harris_total*100],
    'Me_inflation': [Me_trump_total/(SE_SRS_trump_national*2),
                     Me_harris_total/(SE_SRS_harris_national*2)],
})

# save national long table
national_long_table.to_csv("data/mengrep_national_long.csv", index=False)
print("\n" + national_long_table.to_string(index=False))

# save national wide table
national_wide_table.to_csv("data/mengrep_national_wide.csv", index=False)

# In the below calculations of the same value of the national effective sample size, I compute n, f, and DO directly from individual-level data,
# defining the observed sample per candidate as validated_voter == 1 and outcome observed
# the two coincide here because the denominators and weighting align, but the below was used to confirm

# # national truth, repreated from above but with different origin
# p_trump_true_total  = (truth["p_trump_true"]  * truth["N_state"]).sum() / N_total
# p_harris_true_total = (truth["p_harris_true"] * truth["N_state"]).sum() / N_total

# # Meng 4.7 -> DI -> n*_eff, computed on the candidate specific observed sample
# def meng_per_candidate(cces, truth_p, x_col):
#     # observed sample for this candidate: validated voters with non-missing x_col
#     d = cces.loc[(cces["validated_voter"] == 1) & (cces[x_col].notna())].copy()
#     n = len(d)

#     # unweighted sample mean Gn
#     Gn = d[x_col].mean()

#     # benchmark GN
#     GN = truth_p

#     # sampling rate and dropout odds
#     f = n / N_total
#     DO = (1.0 - f) / f

#     # sigma_G
#     sigma = np.sqrt(GN * (1.0 - GN))

#     # 4.7: rho = (Gn - GN) / (sigma * sqrt((1-f)/f))
#     bias = Gn - GN
#     rho = bias / (sigma * np.sqrt((1.0 - f) / f))

#     # DI and effective sample sizes
#     DI = rho**2
#     n_star_eff = 1.0 / (DO * DI)
#     n_eff = n_star_eff / (1.0 + (n_star_eff - 1.0) / (N_total - 1.0))

#     # 4.5: Me and upper bound
#     ME = 2.0 * np.sqrt((sigma**2) / n_star_eff)
#     ME_upper = 1.0 / np.sqrt(n_star_eff)

#     return {
#         "n_used": n,
#         "f": f,
#         "DO": DO,
#         "Gn_hat": Gn,
#         "GN_true": GN,
#         "bias": bias,
#         "sigma_G": sigma,
#         "rho": rho,
#         "DI": DI,
#         "n_star_eff": n_star_eff,
#         "n_eff": n_eff,
#         "ME": ME,
#         "ME_upper": ME_upper,
#     }

# # plug in candidates
# checker_results = {
#     "N_total": N_total,
#     "Trump":  meng_per_candidate(cces, p_trump_true_total,  "X_trump"),
#     "Harris": meng_per_candidate(cces, p_harris_true_total, "X_harris"),
# }

# print(checker_results)

##### ADD SOMETHING NEW TO REPLICATION, NEW ANALYSIS TYPE OR CHANGE ASSUMPTION OR NEW OUTPUTS
# # summary table of bias and n for validated sample
# val_summary = val_m[["state_name", "n", "p_hat", "p_trump_true", "bias", "abs_bias"]].sort_values("abs_bias", ascending=False)
# print("\nTop 10 states by absolute bias (validated):")
# print(val_summary.head(10).to_string(index=False))
# val_summary.to_csv("data/validated_state_bias_summary.csv", index=False)

# # mean bias and RMSE
# mean_bias = val_m["bias"].mean()
# rmse = np.sqrt((val_m["bias"]**2).mean())
# print(f"\nValidated sample mean bias: {mean_bias:.4f}, RMSE: {rmse:.4f}")


##### handling NAs so we know how many are being dropped
mask = (cces["validated_voter"] == 1)

n_validated_all = mask.sum()
n_validated_trump_nonmissing = (mask & cces["X_trump"].notna()).sum()
n_validated_harris_nonmissing = (mask & cces["X_harris"].notna()).sum()

truth_states = set(truth["state_name"])
n_validated_in_truth_states = (mask & cces["state_name"].isin(truth_states)).sum()
n_validated_trump_nonmissing_in_truth_states = (mask & cces["state_name"].isin(truth_states) & cces["X_trump"].notna()).sum()
n_validated_harris_nonmissing_in_truth_states = (mask & cces["state_name"].isin(truth_states) & cces["X_harris"].notna()).sum()

n_table_total = eff_samplesize_df["n"].sum()

print("Microdata: validated (all)                         =", n_validated_all)
print("Microdata: validated & X_trump notna               =", n_validated_trump_nonmissing)
print("Microdata: validated & X_harris notna              =", n_validated_harris_nonmissing)
print("Microdata: validated & state in truth              =", n_validated_in_truth_states)
print("Microdata: validated & state in truth & trump notna=", n_validated_trump_nonmissing_in_truth_states)
print("Microdata: validated & state in truth & harris notna=", n_validated_harris_nonmissing_in_truth_states)
print("Table: sum eff_samplesize_df['n']                  =", n_table_total)

###### which states have NAs
# state-level counts from CCES microdata (validated only)
cces_state_counts = (
    cces.loc[cces["validated_voter"] == 1]
        .groupby("state_name")
        .size()
        .reset_index(name="n_cces_validated")
)

# state-level counts from eff_samplesize_df
eff_state_counts = (
    eff_samplesize_df[["state_name", "n"]]
        .rename(columns={"n": "n_eff_table"})
)

# compute differences
state_count_diff = (
    cces_state_counts
        .merge(eff_state_counts, on="state_name", how="outer")
        .fillna(0)
)

state_count_diff["difference"] = (
    state_count_diff["n_cces_validated"]
    - state_count_diff["n_eff_table"]
)

# print only states where counts differ
print(
    state_count_diff.loc[state_count_diff["difference"] != 0]
        .sort_values("difference", ascending=False)
        .to_string(index=False)
)
