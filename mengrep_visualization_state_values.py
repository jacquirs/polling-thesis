import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

########################################################################################
############ State level visualizations for MSE and design effect from CCES ############
########################################################################################

# this file is made by running meng_mse_deff_calculations.py
df = pd.read_csv("data/mengrep_all_vals_state_level.csv")

########################################################################
# Plot style constants
########################################################################
TITLE_FS  = 16
LABEL_FS  = 14
TICK_FS   = 13
LEGEND_FS = 13

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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trump panel
ax = axes[0]
ax.scatter(df["Var_SRS_trump"], df["MSE_trump"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.axline((0, 0), slope=1, linestyle='--', color='k',
          linewidth=1, label='MSE = Var_SRS (Unbiased SRS Benchmark)')
ax.set_xlabel(r"SRS Variance ($Var_{SRS}$, Benchmark)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"Actual MSE", fontsize=LABEL_FS, fontweight='bold')
ax.set_title("Trump: Actual MSE vs SRS Benchmark",
             fontsize=TITLE_FS, fontweight='bold')
ax.legend(fontsize=LEGEND_FS, loc='upper left')
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.grid(alpha=0.3)

# Harris panel
ax = axes[1]
ax.scatter(df["Var_SRS_harris"], df["MSE_harris"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.axline((0, 0), slope=1, linestyle='--', color='k',
          linewidth=1, label='MSE = Var_SRS (Unbiased SRS Benchmark)')
ax.set_xlabel(r"SRS Variance ($Var_{SRS}$, Benchmark)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"Actual MSE", fontsize=LABEL_FS, fontweight='bold')
ax.set_title("Harris: Actual MSE vs SRS Benchmark",
             fontsize=TITLE_FS, fontweight='bold')
ax.legend(fontsize=LEGEND_FS, loc='upper left')
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.grid(alpha=0.3)

plt.tight_layout(pad=3.0)
plt.savefig("figures/mengrep_mse_vs_srs_comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# FIGURE 2
# design effect vs population size LLP
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trump panel
ax = axes[0]
ax.scatter(df["N_state"], df["Deff_trump"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"Population Size ($N$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"Design Effect ($Deff$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_title("Trump: Design Effect vs Population Size",
             fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.axhline(1, color='green', linestyle='--', linewidth=1,
           label=r'$Deff = 1$ (SRS Equivalent Variance)', alpha=0.5)
ax.legend(fontsize=LEGEND_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)

# Harris panel
ax = axes[1]
ax.scatter(df["N_state"], df["Deff_harris"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"Population Size ($N$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"Design Effect ($Deff$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_title("Harris: Design Effect vs Population Size",
             fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.axhline(1, color='green', linestyle='--', linewidth=1,
           label=r'$Deff = 1$ (SRS Equivalent Variance)', alpha=0.5)
ax.legend(fontsize=LEGEND_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)

plt.tight_layout(pad=3.0)
plt.savefig("figures/mengrep_deff_vs_population_size.png", dpi=300, bbox_inches='tight')
plt.show()


# FIGURE 4
# MSE decomposition
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trump: DI vs MSE
ax = axes[0, 0]
ax.scatter(df["DI_trump"], df["MSE_trump"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DI$ (Data Defect Index = $\rho^2$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"$MSE$", fontsize=LABEL_FS, fontweight='bold')
ax.set_title(r"Trump: MSE vs Data Quality ($DI$)", fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.tick_params(axis='both', labelsize=TICK_FS)

# Trump: DO vs MSE
ax = axes[0, 1]
ax.scatter(df["DO_s"], df["MSE_trump"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DO$ (Dropout Odds = $(1-f)/f$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"$MSE$", fontsize=LABEL_FS, fontweight='bold')
ax.set_title(r"Trump: MSE vs Data Quantity ($DO$)", fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.tick_params(axis='both', labelsize=TICK_FS)

# Harris: DI vs MSE
ax = axes[1, 0]
ax.scatter(df["DI_harris"], df["MSE_harris"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DI$ (Data Defect Index = $\rho^2$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"$MSE$", fontsize=LABEL_FS, fontweight='bold')
ax.set_title(r"Harris: MSE vs Data Quality ($DI$)", fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.tick_params(axis='both', labelsize=TICK_FS)

# Harris: DO vs MSE
ax = axes[1, 1]
ax.scatter(df["DO_s"], df["MSE_harris"],
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"$DO$ (Dropout Odds = $(1-f)/f$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel(r"$MSE$", fontsize=LABEL_FS, fontweight='bold')
ax.set_title(r"Harris: MSE vs Data Quantity ($DO$)", fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
ax.tick_params(axis='both', labelsize=TICK_FS)

plt.tight_layout(pad=3.0)
plt.savefig("figures/mengrep_mse_decomposition.png", dpi=300, bbox_inches='tight')
plt.show()


# FIGURE 5
# RMSE vs state population size
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

log_N = np.log10(df["N_state"])
N_fit = np.logspace(np.log10(df["N_state"].min()), np.log10(df["N_state"].max()), 100)

# Trump panel
ax = axes[0]
ax.scatter(df["N_state"], df["RMSE_trump"] * 100,
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"State Turnout ($N$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel("Root MSE (Percentage Points)", fontsize=LABEL_FS, fontweight='bold')
ax.set_title("Trump: Typical Error vs State Size",
             fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.grid(alpha=0.3)
ax.tick_params(axis='both', labelsize=TICK_FS)

log_RMSE_trump = np.log10(df["RMSE_trump"] * 100)
slope_t, intercept_t, r_t, p_t, se_t = linregress(log_N, log_RMSE_trump)
RMSE_fit_trump = 10 ** (intercept_t + slope_t * np.log10(N_fit))
ax.plot(N_fit, RMSE_fit_trump, 'k--', linewidth=1, alpha=0.5,
        label=rf'Trend: slope={slope_t:.3f}, $R^2$={r_t**2:.3f}')
ax.legend(fontsize=LEGEND_FS)

# Harris panel
ax = axes[1]
ax.scatter(df["N_state"], df["RMSE_harris"] * 100,
           c=df["color"], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)
ax.set_xlabel(r"State Turnout ($N$)", fontsize=LABEL_FS, fontweight='bold')
ax.set_ylabel("Root MSE (Percentage Points)", fontsize=LABEL_FS, fontweight='bold')
ax.set_title("Harris: Typical Error vs State Size",
             fontsize=TITLE_FS, fontweight='bold')
ax.set_xscale('log')
ax.grid(alpha=0.3)
ax.tick_params(axis='both', labelsize=TICK_FS)

log_RMSE_harris = np.log10(df["RMSE_harris"] * 100)
slope_h, intercept_h, r_h, p_h, se_h = linregress(log_N, log_RMSE_harris)
RMSE_fit_harris = 10 ** (intercept_h + slope_h * np.log10(N_fit))
ax.plot(N_fit, RMSE_fit_harris, 'k--', linewidth=1, alpha=0.5,
        label=rf'Trend: slope={slope_h:.3f}, $R^2$={r_h**2:.3f}')
ax.legend(fontsize=LEGEND_FS)

plt.tight_layout(pad=3.0)
plt.savefig("figures/mengrep_rmse_vs_population.png", dpi=300, bbox_inches='tight')
plt.show()