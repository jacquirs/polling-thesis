import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

########################################################################################
############ State level visualizations for MSE and design effect ###########################
########################################################################################

# this file is made by running meng_mse_deff_calculations.py
df = pd.read_csv("data/mengrep_all_vals_state_level.csv")

# assign colors to states
purple_states = ["Arizona", "Georgia", "Michigan", "Nevada", "North Carolina", "Pennsylvania", "Wisconsin"]
red_states = [
    "Alabama", "Alaska", "Arkansas", "Florida", "Idaho", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Mississippi", "Missouri", "Montana", "Nebraska",
    "North Dakota", "Ohio", "Oklahoma", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "West Virginia", "Wyoming"
]
blue_states = [
    "California", "Colorado", "Connecticut", "Delaware", "District Of Columbia",
    "Hawaii", "Illinois", "Maine", "Maryland", "Massachusetts", "Minnesota",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "Oregon",
    "Rhode Island", "Vermont", "Virginia", "Washington"
]

def assign_color(state):
    if state in purple_states:
        return "purple"
    elif state in red_states:
        return "red"
    elif state in blue_states:
        return "blue"

df["color"] = df["state_name"].apply(assign_color)


# FIGURE 1
# actual MSE vs SRS variance benchmark
# visualizes Meng's big data paradox, bias dominates variance for large datasets
# under unbiased SRS MSE = Var_SRS; with bias and/or variance inflation: MSE > Var_SRS
# # distance above diagonal,  bias-induced error inflation
# larger inflation is consistent with worse data quality (larger |rho_{R,G}|) holding other factors fixed
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trump panel: actual MSE vs SRS benchmark
ax = axes[0]
ax.scatter(df["Var_SRS_trump"], df["MSE_trump"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.plot([0, df["Var_SRS_trump"].max()], [0, df["Var_SRS_trump"].max()], 
        'k--', linewidth=1, label='MSE = Var_SRS (unbiased SRS benchmark)') # diagonal line represents unbiased SRS
ax.set_xlabel(r"SRS variance ($Var_{SRS}$, benchmark)")
ax.set_ylabel(r"actual MSE")
ax.set_title(r"Trump: actual MSE vs SRS benchmark" + "\n" + 
             r"(points above line = bias-inflated error)")
ax.legend()
ax.grid(alpha=0.3)

# Harris panel: actual MSE vs SRS benchmark
ax = axes[1]
ax.scatter(df["Var_SRS_harris"], df["MSE_harris"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.plot([0, df["Var_SRS_harris"].max()], [0, df["Var_SRS_harris"].max()], 
        'k--', linewidth=1, label='MSE = Var_SRS (unbiased SRS benchmark)') # diagonal line represents unbiased SRS
ax.set_xlabel(r"SRS variance ($Var_{SRS}$, benchmark)")
ax.set_ylabel(r"actual MSE")
ax.set_title(r"Harris: actual MSE vs SRS benchmark" + "\n" + 
             r"(points above line = bias-inflated error)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/mengrep_mse_vs_srs_comparison.png", dpi=300, bbox_inches='tight')
plt.show()



# FIGURE 2
# design effect vs population size LLP
# under data defect (nonzero rho_{R,G}), error can grow with N (law of large populations)
# Deff summarizes variance inflation vs SRS, it does not measure bias
# steeper increase with N is consistent with stronger defect effects (e.g., larger |rho_{R,G}|), holding other factors fixed
# Deff = 1 = SRS equivalent variance (no variance inflation relative to SRS)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trump panel: design effect vs population size
ax = axes[0]
ax.scatter(df["N_state"], df["Deff_trump"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"population size ($N$)")
ax.set_ylabel(r"design effect ($Deff$)")
ax.set_title(r"Trump: design effect vs population size" + "\n" + 
             r"(law of large populations: larger $N$ -> larger $Deff$)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')

# Deff = 1 for perfect SRS (horizontal, independent of N)
ax.axhline(1, color='green', linestyle='--', linewidth=1, 
           label=r'$Deff = 1$ (SRS equivalent variance)', alpha=0.5)

ax.legend(fontsize=8)

# Harris panel: design effect vs population size
ax = axes[1]
ax.scatter(df["N_state"], df["Deff_harris"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"population size ($N$)")
ax.set_ylabel(r"design effect ($Deff$)")
ax.set_title(r"Harris: design effect vs population size" + "\n" + 
             r"(law of large populations: larger $N$ -> larger $Deff$)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.axhline(1, color='green', linestyle='--', linewidth=1, 
           label=r'$Deff = 1$ (SRS equivalent variance)', alpha=0.5)


ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/mengrep_deff_vs_population_size.png", dpi=300, bbox_inches='tight')
plt.show()


# FIGURE 4
# MSE decomposition
# relates MSE to DI x DO x DU in Meng's decomposition
# steeper slope suggests stronger relationship (not causal by itself)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trump: DI vs MSE
ax = axes[0, 0]
ax.scatter(df["DI_trump"], df["MSE_trump"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DI$ (data defect index = $\rho^2$)")
ax.set_ylabel(r"$MSE$")
ax.set_title(r"Trump: MSE vs data quality ($DI$)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')

# Trump: DO vs MSE
ax = axes[0, 1]
ax.scatter(df["DO_s"], df["MSE_trump"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DO$ (dropout odds = $(1-f)/f$)")
ax.set_ylabel(r"$MSE$")
ax.set_title(r"Trump: MSE vs data quantity ($DO$)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')

# Harris: DI vs MSE 
ax = axes[1, 0]
ax.scatter(df["DI_harris"], df["MSE_harris"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DI$ (data defect index = $\rho^2$)")
ax.set_ylabel(r"$MSE$")
ax.set_title(r"Harris: MSE vs data quality ($DI$)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')

# Harris: DO vs MSE
ax = axes[1, 1]
ax.scatter(df["DO_s"], df["MSE_harris"], 
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DO$ (dropout odds = $(1-f)/f$)")
ax.set_ylabel(r"$MSE$")
ax.set_title(r"Harris: MSE vs data quantity ($DO$)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig("figures/mengrep_mse_decomposition.png", dpi=300, bbox_inches='tight')
plt.show()


# FIGURE 5
# RMSE vs state population size
# tests whether larger states have larger error (RMSE)
# in log-log slope b means RMSE prop to N^b
# b > 0 indicates larger states tend to have larger typical error, b approx 0 indicates weak dependence on N
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trump panel
ax = axes[0]
ax.scatter(df["N_state"], df["RMSE_trump"]*100,  # convert to percentage points
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"state turnout ($N$)")
ax.set_ylabel(r"root MSE (percentage points)")
ax.set_title(r"Trump: typical error vs state size" + "\n" + 
             r"(does larger population -> larger error?)")
ax.set_xscale('log')
ax.grid(alpha=0.3)

# log log, log(RMSE) = intercept + slope x log(N)
log_N = np.log10(df["N_state"])
log_RMSE_trump = np.log10(df["RMSE_trump"]*100)
slope_t, intercept_t, r_t, p_t, se_t = linregress(log_N, log_RMSE_trump)
N_fit = np.logspace(np.log10(df["N_state"].min()), np.log10(df["N_state"].max()), 100)
RMSE_fit_trump = 10**(intercept_t + slope_t * np.log10(N_fit))
ax.plot(N_fit, RMSE_fit_trump, 'k--', linewidth=1, alpha=0.5,
        label=rf'trend: slope={slope_t:.3f}, $R^2$={r_t**2:.3f}')
ax.legend()

# Harris panel
ax = axes[1]
ax.scatter(df["N_state"], df["RMSE_harris"]*100,
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"state turnout ($N$)")
ax.set_ylabel(r"root MSE (percentage points)")
ax.set_title(r"Harris: typical error vs state size" + "\n" + 
             r"(does larger population -> larger error?)")
ax.set_xscale('log')
ax.grid(alpha=0.3)

log_RMSE_harris = np.log10(df["RMSE_harris"]*100)
slope_h, intercept_h, r_h, p_h, se_h = linregress(log_N, log_RMSE_harris)
RMSE_fit_harris = 10**(intercept_h + slope_h * np.log10(N_fit))
ax.plot(N_fit, RMSE_fit_harris, 'k--', linewidth=1, alpha=0.5,
        label=rf'trend: slope={slope_h:.3f}, $R^2$={r_h**2:.3f}')
ax.legend()

plt.tight_layout()
plt.savefig("figures/mengrep_rmse_vs_population.png", dpi=300, bbox_inches='tight')
plt.show()