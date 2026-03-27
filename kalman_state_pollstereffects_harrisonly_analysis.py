import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys
from datetime import datetime
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# THIS FILE ANALYZES POLLS USING KALMAN FILTERING/SMOOTHING WITH POLLSTER-LEVEL HOUSE EFFECTS
# swing state level (AZ, GA, MI, NV, NC, PA, WI)
# harris v trump only
# includes pollster level effects

# same model as kalman_national_pollstereffects_harrisonly_analysis.py but run separately for each swing state

########################################################################################
##################################### Plot Style Constants #############################
########################################################################################
TITLE_FS  = 16
LABEL_FS  = 14
TICK_FS   = 13
LEGEND_FS = 13

# color scheme (red / blue / purple palette)
COLORS = {
    'raw':      '#c9b3e8',   # purple for raw polls
    'corrected':'#4C72B0',   # blue for house-effect corrected polls
    'filtered': '#888888',   # medium gray for filtered line
    'smoothed': '#DD8452',   # orange for smoothed
    'true':     '#2ca02c',   # green for true margin
    'bias':     '#d62728',   # red for bias / pro-trump
    'pro_dem':  '#4C72B0',   # blue for pro-harris regions
}

########################################################################################
##################################### Logging Setup ####################################
########################################################################################
class Logger:
    # utility class to write terminal output to both console and a log file simultaneously
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


########################################################################################
##################################### Load Data ########################################
########################################################################################
def load_and_prepare(filepath: str, state: str, election_date: str = '2024-11-05', days_before: int = None) -> pd.DataFrame:
    """
    load and prepare polling data for a given swing state.
    - retains the pollster column (needed for house effects)
    - encodes pollsters as integer indices for efficient array operations
    - filters by specified state instead of 'national'
    
    parameters:
        filepath: path to csv file
        state: which state to analyze (e.g., 'Arizona', 'Pennsylvania', 'national')
        election_date: date of the election (for date filtering)
        days_before: if specified, only keep polls from the last N days before election
    """
    df = pd.read_csv(filepath)

    # filter to specified state 
    df = df[df['state'] == state].copy()

    # using end_date as the poll date throughout
    df['end_date'] = pd.to_datetime(df['end_date'])

    # filter by date window if specified
    if days_before is not None:
        election_dt = pd.to_datetime(election_date)
        cutoff_date = election_dt - pd.Timedelta(days=days_before)
        n_before = len(df)
        df = df[df['end_date'] >= cutoff_date].copy()
        n_after = len(df)
        print(f"date filter applied: keeping polls from {cutoff_date.date()} onward")
        print(f"  polls before filter: {n_before}")
        print(f"  polls after filter:  {n_after}")
        print(f"  polls dropped:       {n_before - n_after}")

    # construct poll margin in percentage points (raw difference)
    df['poll_margin'] = df['pct_trump_poll'] - df['pct_harris_poll']

    # construct true margin in percentage points
    df['true_margin'] = (df['p_trump_true'] - df['p_harris_true']) * 100

    # multinomial sampling variance for a margin
    pT = df['pct_trump_poll']  / 100.0
    pH = df['pct_harris_poll'] / 100.0
    df['sampling_var'] = (pT + pH - (pT - pH) ** 2) / df['sample_size'] * 10000

    # drop missing values
    df = df.dropna(subset=['end_date', 'poll_margin', 'sampling_var', 'sample_size', 'pollster'])
    df = df[df['sample_size'] > 0]

    # encode pollsters as integer indices 0, 1, 2, ...
    pollster_categories = pd.Categorical(df['pollster'])
    df['pollster_id'] = pollster_categories.codes
    pollster_names = list(pollster_categories.categories)

    # sort chronologically
    df = df.sort_values('end_date').reset_index(drop=True)

    n_pollsters = df['pollster_id'].nunique()
    n_polls = df['poll_id'].nunique()
    n_questions = len(df)
    print(f"{state} polls loaded: {n_questions} questions from {n_polls} polls, {n_pollsters} pollsters")
    print(f"date range: {df['end_date'].min().date()} to {df['end_date'].max().date()}")
    print(f"true margin (constant across rows): {df['true_margin'].iloc[0]:.3f} pp")
    print(f"poll margin range: {df['poll_margin'].min():.1f} to {df['poll_margin'].max():.1f} pp")
    print(f"mean poll margin: {df['poll_margin'].mean():.3f} pp")

    return df, pollster_names


########################################################################################
######################### Anchor to true result ########################################
########################################################################################
def append_election_result(df: pd.DataFrame, pollster_names: list,
                           election_date: str = '2024-11-05', anchor: bool = True) -> tuple:
    """
    optionally append the certified result as a terminal observation with near-zero variance.
    the election result is assigned pollster_id = -1 so the em algorithm skips it.
    """
    true_margin = df['true_margin'].iloc[0]

    if not anchor:
        print(f"\nskipping election result anchor (unanchored mode)")
        print(f"true margin: {true_margin:.3f} pp (not used as constraint)")
        return df, pollster_names

    anchor_row = {
        'question_id':  -1,
        'poll_id':       -1,
        'pollster':      'ELECTION_RESULT',
        'pollster_id':   -1,
        'state':         df['state'].iloc[0],
        'end_date':       pd.to_datetime(election_date),
        'poll_margin':    true_margin,
        'true_margin':    true_margin,
        'sampling_var':   1e-6,
        'sample_size':    1_000_000_000,
    }

    df = pd.concat([df, pd.DataFrame([anchor_row])], ignore_index=True)
    df = df.sort_values('end_date').reset_index(drop=True)

    print(f"\nelection result appended: margin = {true_margin:.3f} pp on {election_date}")
    return df, pollster_names


########################################################################################
######################### Kalman filter/smoother (inner loop) ##########################
########################################################################################
def kalman_filter_smoother(y: np.ndarray, obs_var: np.ndarray,
                           days: np.ndarray, sigma2_u: float) -> tuple:
    """
    core kalman filter and rts smoother.
    takes house-effect-corrected margins as input (y).
    """
    n = len(y)

    # forward pass: kalman filter
    F  = np.zeros(n)
    P  = np.zeros(n)
    W  = np.zeros(n)

    F[0] = y[0]
    P[0] = obs_var[0] + sigma2_u

    for t in range(1, n):
        days_elapsed = days[t] - days[t - 1]
        P_pred = P[t - 1] + sigma2_u * days_elapsed
        W[t]   = P_pred / (P_pred + obs_var[t])
        F[t]   = W[t] * y[t] + (1 - W[t]) * F[t - 1]
        P[t]   = P_pred * (1 - W[t])

    # backward pass: rts smoother
    S  = np.zeros(n)
    PS = np.zeros(n)

    S[n - 1]  = F[n - 1]
    PS[n - 1] = P[n - 1]

    for t in range(n - 2, -1, -1):
        days_elapsed = days[t + 1] - days[t]
        P_pred = P[t] + sigma2_u * days_elapsed
        G      = P[t] / P_pred
        S[t]   = F[t] + G * (S[t + 1] - F[t])
        PS[t]  = P[t] + G ** 2 * (PS[t + 1] - P_pred)

    return F, P, S, PS, W


########################################################################################
######################### Grid search MLE for sigma2_u #################################
########################################################################################
def estimate_sigma2u(y: np.ndarray, obs_var: np.ndarray, days: np.ndarray,
                     n_coarse: int = 500, n_fine: int = 200) -> float:
    """
    two-stage grid search mle for sigma2_u.
    called inside the em loop after house effects have been subtracted from y.
    """
    def innovations_loglik(s2):
        n = len(y)
        ll  = 0.0
        F_t = y[0]
        P_t = obs_var[0] + s2

        for t in range(1, n):
            days_elapsed = days[t] - days[t - 1]
            P_pred   = P_t + s2 * days_elapsed
            innov    = y[t] - F_t
            innov_var = P_pred + obs_var[t]
            if innov_var <= 0:
                return -np.inf
            ll  += -0.5 * (np.log(2 * np.pi * innov_var) + innov ** 2 / innov_var)
            W_t  = P_pred / innov_var
            F_t  = W_t * y[t] + (1 - W_t) * F_t
            P_t  = P_pred * (1 - W_t)

        return ll

    coarse_grid = np.logspace(-4, 1, n_coarse)
    ll_coarse   = np.array([innovations_loglik(s2) for s2 in coarse_grid])
    best_idx    = np.argmax(ll_coarse)

    lo        = coarse_grid[max(0, best_idx - 1)]
    hi        = coarse_grid[min(len(coarse_grid) - 1, best_idx + 1)]
    fine_grid = np.linspace(lo, hi, n_fine)
    ll_fine   = np.array([innovations_loglik(s2) for s2 in fine_grid])

    return fine_grid[np.argmax(ll_fine)]


########################################################################################
######################### EM Algorithm: joint estimation ###############################
########################################################################################
def em_kalman_house_effects(df: pd.DataFrame, pollster_names: list,
                             sigma2_u_init: float = None,
                             max_iter: int = 50,
                             tol: float = 1e-4) -> tuple:
    """
    em algorithm to jointly estimate:
        (1) the latent true margin trajectory (via kalman filter/smoother)
        (2) a house effect for each pollster (average signed deviation from truth)
        (3) sigma2_u (opinion volatility per day, re-estimated each iteration)

    identification constraint: house effects are mean-zero (weighted by poll count),
    so they represent deviations from the industry average, not from absolute truth.
    """
    # exclude election result row from em (it has pollster_id = -1)
    is_poll = df['pollster_id'] >= 0
    poll_idx = df[is_poll].index.tolist()

    df = df.copy()
    day_0 = df['end_date'].min()
    df['day'] = (df['end_date'] - day_0).dt.days

    obs_var = df['sampling_var'].values
    days    = df['day'].values
    pollster_ids = df.loc[poll_idx, 'pollster_id'].values.astype(int)
    n_pollsters  = len(pollster_names)

    # --- initialization ---
    house_effects = np.zeros(n_pollsters)

    y_full = df['poll_margin'].values.copy()
    if sigma2_u_init is None:
        print("\ninitializing sigma2_u from raw (uncorrected) margins...")
        sigma2_u = estimate_sigma2u(y_full, obs_var, days)
        print(f"  initial sigma2_u: {sigma2_u:.6f}")
    else:
        sigma2_u = sigma2_u_init

    history = []

    print(f"\nstarting em algorithm (max_iter={max_iter}, tol={tol})")
    print(f"{'iter':>5}  {'max_he_change':>14}  {'sigma2_u':>10}  {'mean_|he|':>10}")
    print("-" * 50)

    for iteration in range(max_iter):

        # e-step: subtract current house effects, run kalman smoother
        y_corrected = y_full.copy()
        for idx, pid in zip(poll_idx, pollster_ids):
            y_corrected[idx] -= house_effects[pid]

        F, P, S, PS, W = kalman_filter_smoother(y_corrected, obs_var, days, sigma2_u)

        # m-step: re-estimate house effects as average residual (poll - smoothed)
        new_house_effects = np.zeros(n_pollsters)
        counts = np.zeros(n_pollsters)

        for idx, pid in zip(poll_idx, pollster_ids):
            new_house_effects[pid] += (df.loc[idx, 'poll_margin'] - S[idx])
            counts[pid] += 1

        mask = counts > 0
        new_house_effects[mask] /= counts[mask]

        # apply mean-zero constraint
        weighted_mean = np.average(new_house_effects[mask], weights=counts[mask])
        new_house_effects -= weighted_mean

        # re-estimate sigma2_u on corrected margins
        sigma2_u = estimate_sigma2u(y_corrected, obs_var, days)

        max_change = np.max(np.abs(new_house_effects - house_effects))
        mean_abs_he = np.mean(np.abs(new_house_effects[mask]))
        print(f"{iteration+1:>5}  {max_change:>14.6f}  {sigma2_u:>10.6f}  {mean_abs_he:>10.4f}")

        house_effects = new_house_effects.copy()
        history.append({
            'iteration': iteration + 1,
            'house_effects': house_effects.copy(),
            'sigma2_u': sigma2_u,
            'max_change': max_change,
        })

        if max_change < tol:
            print(f"\nem converged at iteration {iteration + 1} (max change {max_change:.2e} < tol {tol:.2e})")
            break
    else:
        print(f"\nem did not converge after {max_iter} iterations (max change {max_change:.2e})")

    # --- final pass: store results on dataframe ---
    y_corrected_final = y_full.copy()
    for idx, pid in zip(poll_idx, pollster_ids):
        y_corrected_final[idx] -= house_effects[pid]

    F, P, S, PS, W = kalman_filter_smoother(y_corrected_final, obs_var, days, sigma2_u)

    df['corrected_margin']  = y_corrected_final
    df['filtered']          = F
    df['filtered_se']       = np.sqrt(np.maximum(P,  0))
    df['smoothed']          = S
    df['smoothed_se']       = np.sqrt(np.maximum(PS, 0))
    df['weight']            = W

    df['house_effect_assigned'] = 0.0
    for idx, pid in zip(poll_idx, pollster_ids):
        df.loc[idx, 'house_effect_assigned'] = house_effects[pid]

    df['sigma2_u'] = sigma2_u

    # bias decomposition — three components:
    #   total_error              = poll_margin - true_margin
    #   house_effect_assigned    = pollster's estimated systematic offset
    #   sampling_noise           = corrected_margin - smoothed
    #   residual_systematic_bias = smoothed - true_margin
    true_margin = df['true_margin'].iloc[0]
    df['total_error']              = df['poll_margin']      - true_margin
    df['sampling_noise']           = df['corrected_margin'] - df['smoothed']
    df['residual_systematic_bias'] = df['smoothed']         - true_margin

    he_records = []
    for pid, name in enumerate(pollster_names):
        he_records.append({
            'pollster':      name,
            'house_effect':  house_effects[pid],
            'n_polls':       int(counts[pid]) if mask[pid] else 0,
        })
    house_effects_df = pd.DataFrame(he_records).sort_values('house_effect', ascending=False)

    return df, house_effects_df, sigma2_u, history


########################################################################################
######################### Summary stats ################################################
########################################################################################
def summarize_results(df: pd.DataFrame, house_effects_df: pd.DataFrame,
                      sigma2_u: float, state: str, anchored: bool = True,
                      top_n_pollsters: int = 20) -> None:
    """
    print full decomposition including house effects.
    state parameter added for swing state version to label output correctly.
    """
    results = df[df['pollster_id'] != -1].copy()
    true_margin = results['true_margin'].iloc[0]

    print("\n" + "=" * 70)
    if anchored:
        print(f"polling bias decomposition with house effects — {state} (trump margin, anchored)")
    else:
        print(f"polling opinion trajectory with house effects — {state} (trump margin, unanchored)")
    print("=" * 70)
    print(f"\ntrue margin (certified result): {true_margin:.3f} pp")
    print(f"estimated sigma2_u:             {sigma2_u:.6f} per day")
    print(f"interpretation: true opinion can move ~{np.sqrt(sigma2_u):.3f} pp/day (1 sd)")

    print(f"\n--- overall poll error ---")
    print(f"  mean poll margin:              {results['poll_margin'].mean():.3f} pp")
    print(f"  mean total error:              {results['total_error'].mean():.3f} pp")
    print(f"  sd of total error:             {results['total_error'].std():.3f} pp")

    print(f"\n--- three-component decomposition ---")
    print(f"  mean |house effect|:           {results['house_effect_assigned'].abs().mean():.3f} pp")
    print(f"  mean |sampling noise|:         {results['sampling_noise'].abs().mean():.3f} pp")
    if anchored:
        print(f"  mean residual systematic bias: {results['residual_systematic_bias'].mean():.3f} pp")

    if anchored:
        var_total    = results['total_error'].var()
        var_he       = results['house_effect_assigned'].var()
        var_noise    = results['sampling_noise'].var()
        var_residual = results['residual_systematic_bias'].var()
        print(f"\n--- variance decomposition ---")
        print(f"  var(total error):              {var_total:.4f}")
        print(f"  var(house effects):            {var_he:.4f}  ({100 * var_he / var_total:.1f}%)")
        print(f"  var(sampling noise):           {var_noise:.4f}  ({100 * var_noise / var_total:.1f}%)")
        print(f"  var(residual systematic bias): {var_residual:.4f}  ({100 * var_residual / var_total:.1f}%)")

    print(f"\n--- house effects (top {top_n_pollsters} by absolute effect, min 5 polls) ---")
    he_display = house_effects_df[house_effects_df['n_polls'] >= 5].copy()
    he_display['abs_he'] = he_display['house_effect'].abs()
    he_display = he_display.nlargest(top_n_pollsters, 'abs_he').drop(columns='abs_he')
    he_display['house_effect'] = he_display['house_effect'].round(3)
    print(he_display.to_string(index=False))


########################################################################################
######################### Visualization ################################################
########################################################################################
def plot_results(df: pd.DataFrame, house_effects_df: pd.DataFrame,
                 state: str, anchored: bool = True, save_path: str = None) -> None:
    """
    two separate figures:
      figure 1 (three panels): raw polls, corrected polls, smoothed + filtered estimates,
                               standard errors, residual systematic bias
      figure 2 (one panel):   top 20 house effects by absolute size (bar chart),
                               saved as a separate file (same pattern as national version)
    """
    results = df[df['pollster_id'] != -1].copy()
    true_margin = results['true_margin'].iloc[0]
    n_polls = len(results)

    mode_label = "anchored" if anchored else "unanchored"
    colors = COLORS
    dates = results['end_date']

    ########################################################################
    # figure 1: three-panel main figure
    ########################################################################
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=False)
    fig.suptitle(
        f'Kalman Filter with House Effects: {state.title()} Polling 2024 ({mode_label.title()})\n'
        f'(Trump Margin, pp | n={n_polls} polls)',
        fontsize=TITLE_FS, fontweight='bold', y=0.99
    )

    # panel 1: raw polls, corrected polls, filtered, smoothed, true
    ax1 = axes[0]
    ax1.scatter(dates, results['poll_margin'],
                color=colors['raw'], alpha=0.6, s=8, label='Raw Polls',
                edgecolors='none', zorder=1)
    ax1.scatter(dates, results['corrected_margin'],
                color=colors['corrected'], alpha=0.6, s=8,
                label='House-Effect Corrected Polls',
                edgecolors='none', zorder=2)
    ax1.plot(dates, results['filtered'],
             color=colors['filtered'], linewidth=1.5, linestyle='-',
             label='Filtered Estimate', zorder=3)
    ax1.plot(dates, results['smoothed'],
             color=colors['smoothed'], linewidth=2.5,
             label='Smoothed Estimate', zorder=4)
    ax1.axhline(true_margin, color=colors['true'], linewidth=2,
                label=f'True Margin ({true_margin:.2f} pp)', zorder=5)
    ax1.fill_between(dates,
                     results['smoothed'] - 1.96 * results['smoothed_se'],
                     results['smoothed'] + 1.96 * results['smoothed_se'],
                     color=colors['smoothed'], alpha=0.15, label='Smoothed 95% CI')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax1.set_ylabel('Trump Margin (pp)', fontsize=LABEL_FS, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=LEGEND_FS)
    ax1.set_title('Raw Polls vs House-Effect Corrected Polls vs Smoothed Estimate',
                  fontsize=TITLE_FS, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=TICK_FS)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center',
             fontsize=TICK_FS, fontweight='bold')

    # panel 2: standard errors
    ax2 = axes[1]
    conv_se = results['sampling_var'].apply(np.sqrt)
    ax2.plot(dates, conv_se, color=colors['raw'], linewidth=1, alpha=0.8,
             label='Conventional SE (per-poll)')
    ax2.plot(dates, results['filtered_se'],
             color=colors['filtered'], linewidth=1.5, linestyle='-',
             label='Filtered SE')
    ax2.plot(dates, results['smoothed_se'], color=colors['smoothed'], linewidth=2,
             label='Smoothed SE')
    ax2.set_ylabel('Standard Error (pp)', fontsize=LABEL_FS, fontweight='bold')
    ax2.legend(fontsize=LEGEND_FS)
    ax2.set_title('Uncertainty: Conventional vs Filtered vs Smoothed',
                  fontsize=TITLE_FS, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=TICK_FS)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center',
             fontsize=TICK_FS, fontweight='bold')

    # panel 3: residual systematic bias
    ax3 = axes[2]
    ax3.plot(dates, results['residual_systematic_bias'],
             color=colors['bias'], linewidth=2,
             label='Residual Systematic Bias (Smoothed − True)')
    ax3.fill_between(dates, 0, results['residual_systematic_bias'],
                     where=results['residual_systematic_bias'] >  0,
                     color=colors['bias'], alpha=0.15, label='Pro-Trump Region')
    ax3.fill_between(dates, 0, results['residual_systematic_bias'],
                     where=results['residual_systematic_bias'] <= 0,
                     color=colors['pro_dem'], alpha=0.15, label='Pro-Harris Region')
    ax3.axhline(0, color='black', linewidth=1.0)
    ax3.set_ylabel('Residual Bias (pp)', fontsize=LABEL_FS, fontweight='bold')
    ax3.legend(fontsize=LEGEND_FS)
    if anchored:
        ax3.set_title(
            'Residual Industry Bias After Removing House Effects\n'
            '(Positive = Aggregate Industry Overstated Trump Even After Correcting Individual Pollsters)',
            fontsize=TITLE_FS, fontweight='bold'
        )
    else:
        ax3.set_title(
            'Residual Trajectory vs Certified Result After Removing House Effects\n'
            '(Positive = Smoothed Trajectory Overstated Trump; Final Value = Corrected Forecast Error)',
            fontsize=TITLE_FS, fontweight='bold'
        )
    ax3.tick_params(axis='both', labelsize=TICK_FS)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center',
             fontsize=TICK_FS, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nmain figure saved to: {save_path}")

    plt.close()

    ########################################################################
    # figure 2: house effects bar chart (separate file)
    ########################################################################
    he_plot = (house_effects_df[house_effects_df['n_polls'] >= 3]   # min 3 polls for state level (fewer polls than national)
               .copy()
               .assign(abs_he=lambda x: x['house_effect'].abs())
               .nlargest(20, 'abs_he')
               .sort_values('house_effect', ascending=True))
    bar_colors = [colors['bias'] if v > 0 else colors['pro_dem']
                  for v in he_plot['house_effect']]

    fig2, ax4 = plt.subplots(figsize=(10, 8))
    fig2.suptitle(
        f'Pollster House Effects: {state.title()} Polling 2024 ({mode_label.title()})',
        fontsize=TITLE_FS, fontweight='bold'
    )
    ax4.barh(he_plot['pollster'], he_plot['house_effect'], color=bar_colors, alpha=0.8)
    ax4.axvline(0, color='black', linewidth=1.0)
    ax4.set_xlabel('House Effect (pp, Relative to Industry Average)',
                   fontsize=LABEL_FS, fontweight='bold')
    ax4.set_title(
        'Top 20 Pollster House Effects (Min 3 Polls)\n'
        '(Positive = More Pro-Trump than Industry Average; Negative = More Pro-Harris)',
        fontsize=TITLE_FS, fontweight='bold'
    )
    ax4.tick_params(axis='both', labelsize=TICK_FS)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        he_save_path = save_path.replace('.png', '_house_effects.png')
        plt.savefig(he_save_path, dpi=150, bbox_inches='tight')
        print(f"house effects figure saved to: {he_save_path}")

    plt.close()


########################################################################################
######################### Export results ###############################################
########################################################################################
def export_results(df: pd.DataFrame, house_effects_df: pd.DataFrame,
                   out_path_polls: str, out_path_house_effects: str) -> None:
    """
    export two csv files:
    (1) poll-level results with all decomposition columns
    (2) pollster-level house effects summary
    """
    poll_cols = [
        'question_id', 'poll_id', 'pollster', 'end_date', 'sample_size',
        'poll_margin', 'corrected_margin', 'true_margin',
        'filtered', 'filtered_se', 'smoothed', 'smoothed_se',
        'house_effect_assigned', 'total_error', 'sampling_noise',
        'residual_systematic_bias', 'weight', 'sigma2_u',
    ]
    if 'period' in df.columns:
        poll_cols.append('period')

    out = df[df['pollster_id'] != -1][poll_cols].copy()
    out.to_csv(out_path_polls, index=False)
    print(f"poll results exported to: {out_path_polls}")

    house_effects_df.to_csv(out_path_house_effects, index=False)
    print(f"house effects exported to: {out_path_house_effects}")


########################################################################################
######################### Actually run all these functions from here ###################
########################################################################################
if __name__ == '__main__':

    # important values
    DATA_PATH     = 'data/harris_trump_datelimted_accuracy_no_explode.csv'
    ELECTION_DATE = '2024-11-05'
    TIME_WINDOWS  = [107, 60, 30]
    
    # analyze each swing state separately
    SWING_STATES  = ['arizona', 'georgia', 'michigan', 'nevada', 'north carolina', 'pennsylvania', 'wisconsin']

    # set up logging
    log_filename = f'output/kalman/kalman_state_pollstereffects_harrisonly_anlaysis_log.txt'
    logger       = Logger(log_filename)
    sys.stdout   = logger

    print(f"kalman filter with house effects analysis — swing states")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"log file: {log_filename}")
    print(f"states: {SWING_STATES}")
    print(f"time windows: {TIME_WINDOWS}")

    for state in SWING_STATES:

        # state abbreviation for compact filenames
        state_abbrev = {
            'arizona': 'AZ',
            'georgia': 'GA',
            'michigan': 'MI',
            'nevada': 'NV',
            'north carolina': 'NC',
            'pennsylvania': 'PA',
            'wisconsin': 'WI'
        }[state]

        for days_before in TIME_WINDOWS:

            if days_before is None:
                window_label = "all_data"
                window_desc  = "ALL DATA"
            else:
                window_label = f"last_{days_before}_days"
                window_desc  = f"LAST {days_before} DAYS"

            print("\n" + "="*70)
            print("="*70)
            print(f"ANALYZING: {state} ({state_abbrev}) | {window_desc}")
            print("="*70)
            print("="*70)

            for anchor in [True, False]:
                mode_label = "anchored" if anchor else "unanchored"
                mode_desc  = "ANCHORED" if anchor else "UNANCHORED"

                print("\n" + "="*70)
                print(f"{mode_desc} MODE ({state} | {window_desc})")
                print("="*70)

                df, pollster_names = load_and_prepare(DATA_PATH, state, ELECTION_DATE, days_before=days_before)
                print(f"\nsanity check — unique true_margin values: {df['true_margin'].unique()}")

                df, pollster_names = append_election_result(df, pollster_names, ELECTION_DATE, anchor=anchor)

                df, house_effects_df, sigma2u, history = em_kalman_house_effects(
                    df, pollster_names, max_iter=50, tol=1e-4
                )

                summarize_results(df, house_effects_df, sigma2u, state, anchored=anchor)

                # plot_results now saves two files:
                #   figures/kalman_he_{abbrev}_{mode}_{window}.png          (three-panel main)
                #   figures/kalman_he_{abbrev}_{mode}_{window}_house_effects.png  (bar chart)
                plot_results(
                    df, house_effects_df, state, anchored=anchor,
                    save_path=f'figures/kalman_he_{state_abbrev}_{mode_label}_{window_label}.png'
                )

                export_results(
                    df, house_effects_df,
                    out_path_polls=f'data/kalman_he_polls_{state_abbrev}_{mode_label}_{window_label}.csv',
                    out_path_house_effects=f'data/kalman_he_effects_{state_abbrev}_{mode_label}_{window_label}.csv'
                )

    print(f"\n{'='*70}")
    print(f"analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"log saved to: {log_filename}")
    logger.close()
    sys.stdout = logger.terminal