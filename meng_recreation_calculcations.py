import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data from pipeline
cces = pd.read_csv("cces2024_meng_replication_set.csv")
truth_raw = pd.read_csv("Meng_true_votes.csv")

# build truth table
truth = truth_raw[["state_name", "p_trump_true", "N_state"]].copy()

# check success for all 51 result areas 
print("Truth jurisdictions:", len(truth))

########################################################################################
######################## REPLICATION OF FIGURE 4 ON PAGE 711 ###########################
########################################################################################

###### State-level estimators, Meng does this for raw, likely, validated voters
# helper to compute state-level n, p_hat, se, 95% CI
def state_estimates(df, mask=None, value_col="X_trump"):
    """
    Returns DataFrame with columns: state_name, n, p_hat, se, ci_lo, ci_hi
    This corresponds to Meng's per-state sample mean \hat p_s and its Wald SE
    """
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

    # 95% Wald CIs (clamped to [0,1])
    out["ci_lo"] = (out["p_hat"] - 1.96 * out["se"]).clip(0, 1)
    out["ci_hi"] = (out["p_hat"] + 1.96 * out["se"]).clip(0, 1)
    return out

# Compute the three estimators used in Meng's figure:
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

# save by state tables
raw_m.to_csv("state_raw_vs_truth.csv", index=False)
likely_m.to_csv("state_likely_vs_truth.csv", index=False)
val_m.to_csv("state_validated_vs_truth.csv", index=False)

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

    ax.errorbar(plot_df["p_trump_true"], plot_df["p_hat"],
                yerr=[yerr_lower, yerr_upper],
                fmt="o", ms=6, alpha=0.85, ecolor="gray", capsize=3)

    # add 45 degree line
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    # axis labels for figure 4
    ax.set_title(title)
    ax.set_xlabel("True Trump share (state)")
    if ax is axes[0]:
        ax.set_ylabel("Estimated Trump share (CCES)")

plt.suptitle("Figure 4 Replication: State-level CCES estimates vs Official 2024 Results (Trump)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figure4_cces2024_trump.png", dpi=300)
plt.show()

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