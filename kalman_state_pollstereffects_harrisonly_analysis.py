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
##################################### Logging Setup ####################################
########################################################################################
# green et al. (1999) specify a simple two-component model:
#
#   observation eq:  X_t = xi_t + e_t       (poll = true opinion + sampling error)
#   state eq:        xi_t = xi_{t-1} + u_t  (true opinion follows a random walk)
#
# their model has one observation per time point and one source of observation noise (e_t),
# which they assume is pure random sampling error with known variance p*(1-p)/N.
# they do not model any systematic, persistent differences between pollsters.
# in their context (tracking % republican using aggregated cbs/nyt data from a single
# survey house) this is appropriate. with our data (many pollsters, each with their own
# methodology, likely voter screens, and question wording), it is not.
#
# this file adds a third component: a pollster-specific fixed offset called a "house effect"
# (also called a "house bias" or "pollster effect" in the literature). the model becomes:
#
#   observation eq:  poll_margin[i,t] = true_margin[t] + house_effect[i] + e[i,t]
#   state eq:        true_margin[t]   = true_margin[t-1] + u[t]
#
# where i indexes pollsters. house_effect[i] is the average signed error that pollster i
# makes relative to the true margin, net of sampling noise. a positive house_effect means
# the pollster systematically overstates trump; negative means it overstates harris.
# in english: if pollster A always shows trump +3 when truth is +1, its house effect is +2.
#
# this is a fundamentally different model structure from green et al., not just a parameter
# extension. the original paper cannot accommodate it without modification because it has
# no pollster index — every observation is treated as an equally valid draw from the same
# data generating process.

########## HOW THIS DIFFERS FROM kalman_national_harrisonly_analysis.py 
# our initial implementation (kalman_polling_bias.py) follows green et al. directly,
# with two extensions: (1) correct multinomial sampling variance for a margin rather
# than a single share, and (2) an optional election-result anchor. it does NOT model
# house effects.
#
# the key problem this creates: with 10-15 polls per day from different pollsters,
# our initial implementation treats all same-day polls as sequential independent
# observations of the same latent state. this means:
#   - pollsters with consistent pro-trump bias pull the smoothed trajectory upward
#   - pollsters with consistent pro-harris bias pull it downward
#   - the smoother cannot distinguish "true opinion shifted" from "today happened to
#     have more pro-trump pollsters in the field"
# in other words, house effects contaminate the estimated latent trajectory. the
# systematic_bias we measure in kalman_polling_bias.py is actually a mixture of
# aggregate industry bias AND the weighted average house effect of whichever pollsters
# happened to be active at each point in time.
#
# this file fixes that by jointly estimating:
#   (1) the latent true margin trajectory (kalman filter/smoother, same as before)
#   (2) a fixed offset for each pollster (house effect, estimated via em algorithm)
#
# the em (expectation-maximization) algorithm works as follows:
#   e-step: given current house effect estimates, subtract them from each poll to get
#           "house-effect-corrected" margins, then run the kalman smoother on those
#   m-step: given the smoothed latent trajectory, estimate each pollster's house effect
#           as the average residual (poll margin - smoothed margin) across all their polls
#   repeat until convergence (changes in house effects fall below a threshold)
#
# in plain english: we alternate between (a) "assuming we know the house effects, what
# was true opinion?" and (b) "assuming we know true opinion, what are the house effects?"
# iterating back and forth until the two estimates stop changing.
#
# other differences from kalman_polling_bias.py:
#   - sigma2_u and house effects are estimated jointly via em + grid search
#   - multiple polls per day are handled correctly: same-day polls from different
#     pollsters are now informative about house effects, not just the latent state
#   - the bias decomposition gains a third component:
#       total_error = sampling_noise + house_effect + residual_systematic_bias
#   - anchoring (election result as terminal observation) is retained
#   - time windows and logging infrastructure are retained

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
    load and prepare polling data, identical to kalman_polling_bias.py except:
    - retains the pollster column (needed for house effects)
    - encodes pollsters as integer indices for efficient array operations
    - filters by specified state instead of 'national' (SWING STATE MODIFICATION)
    
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
    # same choice as kalman_polling_bias.py: raw difference is additive, interpretable,
    # and appropriate near 50/50. caveat: pct_dk variation across pollsters is a soft
    # house effect not captured here, but will be partially absorbed into house_effect[i]
    df['poll_margin'] = df['pct_trump_poll'] - df['pct_harris_poll']

    # construct true margin in percentage points
    df['true_margin'] = (df['p_trump_true'] - df['p_harris_true']) * 100

    # multinomial sampling variance for a margin (departure from green et al., same
    # correction as kalman_polling_bias.py — see that file for full derivation)
    pT = df['pct_trump_poll']  / 100.0
    pH = df['pct_harris_poll'] / 100.0
    df['sampling_var'] = (pT + pH - (pT - pH) ** 2) / df['sample_size'] * 10000

    # drop missing values
    df = df.dropna(subset=['end_date', 'poll_margin', 'sampling_var', 'sample_size', 'pollster'])
    df = df[df['sample_size'] > 0]

    # encode pollsters as integer indices 0, 1, 2, ...
    # this makes array indexing fast and unambiguous throughout the em algorithm
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
    same logic as kalman_polling_bias.py — optionally append the certified result
    as a terminal observation with near-zero variance.

    the election result is assigned a special pollster_id of -1 so the em algorithm
    knows not to estimate a house effect for it (the "true result" has no house bias
    by definition — it is the thing we are measuring bias relative to).
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
        'pollster_id':   -1,           # special value: excluded from house effect estimation
        'state':         df['state'].iloc[0],  # preserve state from input (SWING STATE MODIFICATION)
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
    core kalman filter and rts smoother, identical in structure to kalman_polling_bias.py.

    takes house-effect-corrected margins as input (y), so the model it fits is:
        corrected_margin[t] = true_margin[t] + e[t]
    where corrected_margin = poll_margin - house_effect[pollster].

    this function is called inside the em loop on each iteration after house effects
    have been subtracted. it returns filtered and smoothed estimates plus uncertainties.

    see kalman_polling_bias.py for full documentation of the filter/smoother equations.
    the rts smoother is used (not green et al. eq. 8 directly) for the same reasons
    documented there: eq. 8 assumes unit time steps, rts handles uneven spacing correctly.
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
    two-stage grid search mle for sigma2_u, identical to kalman_polling_bias.py.
    called inside the em loop after house effects have been subtracted from y.
    see kalman_polling_bias.py for full documentation.
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
    em (expectation-maximization) algorithm to jointly estimate:
        (1) the latent true margin trajectory (via kalman filter/smoother)
        (2) a house effect for each pollster (average signed deviation from truth)
        (3) sigma2_u (opinion volatility per day, re-estimated each iteration)

    --- what em means in plain english ---
    we have two unknowns that depend on each other in a circle:
      - to estimate house effects, we need to know what true opinion was
        (so we can compute each pollster's deviation from it)
      - to estimate true opinion, we need to know house effects
        (so we can correct each poll before running the smoother)
    em breaks this circle by alternating between the two:
      e-step ("expectation"): assume house effects are known, subtract them
                              from each poll, run the kalman smoother to get
                              the best estimate of the true opinion trajectory
      m-step ("maximization"): assume the smoothed trajectory is truth, compute
                               each pollster's average residual as their house effect
    we repeat until the house effect estimates stop changing (converge).
    under standard conditions, em is guaranteed to improve the likelihood at every
    iteration and converge to a local maximum.

    --- identification constraint ---
    house effects are only identified up to an additive constant: if we add 1pp
    to every house effect and subtract 1pp from the latent state, the model fits
    identically. to pin down a unique solution we constrain house effects to sum
    to zero (mean zero across pollsters weighted by number of observations). this
    means house effects are interpreted as deviations from the industry average,
    not deviations from absolute truth. the aggregate industry bias (industry
    average vs. true result) is captured in the smoothed trajectory itself.
    in plain english: a house effect of +2 means "this pollster is 2pp more
    pro-trump than the average pollster", not necessarily "2pp more pro-trump
    than truth".

    parameters:
        df:             dataframe with poll_margin, sampling_var, pollster_id, end_date
        pollster_names: list mapping pollster_id integers to names
        sigma2_u_init:  starting value for sigma2_u (if None, estimated from raw data)
        max_iter:       maximum em iterations before stopping
        tol:            convergence threshold (max absolute change in any house effect)

    returns:
        df with added columns: house_effect_assigned, corrected_margin,
                               filtered, filtered_se, smoothed, smoothed_se, weight
        house_effects_df: dataframe of pollster name and estimated house effect
        sigma2_u: final estimated opinion volatility
        history: list of dicts recording house effects and sigma2_u at each iteration
    """
    # exclude election result row from em (it has pollster_id = -1)
    is_poll = df['pollster_id'] >= 0
    poll_idx = df[is_poll].index.tolist()

    df = df.copy()
    day_0 = df['end_date'].min()
    df['day'] = (df['end_date'] - day_0).dt.days

    y_raw   = df.loc[poll_idx, 'poll_margin'].values
    obs_var = df['sampling_var'].values
    days    = df['day'].values
    pollster_ids = df.loc[poll_idx, 'pollster_id'].values.astype(int)
    n_pollsters  = len(pollster_names)

    # --- initialization ---
    # start with all house effects at zero (equivalent to the baseline model)
    # sigma2_u initialized via grid search on raw (uncorrected) margins
    house_effects = np.zeros(n_pollsters)

    y_full = df['poll_margin'].values.copy()   # full array including anchor
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

        # -----------------------------------------------------------------------
        # e-step: subtract current house effects from each poll,
        #         run kalman smoother on corrected margins
        # -----------------------------------------------------------------------
        # "corrected margin" = what this poll would have shown if the pollster
        # had no house effect — our best estimate of what an unbiased poll would
        # have reported on that day
        y_corrected = y_full.copy()
        for idx, pid in zip(poll_idx, pollster_ids):
            y_corrected[idx] -= house_effects[pid]

        F, P, S, PS, W = kalman_filter_smoother(y_corrected, obs_var, days, sigma2_u)

        # -----------------------------------------------------------------------
        # m-step: given smoothed trajectory, re-estimate house effects as
        #         each pollster's average residual (poll - smoothed)
        # -----------------------------------------------------------------------
        # residual[i] = poll_margin[i] - smoothed[t(i)]
        # intuitively: "how far above or below the smoothed true opinion was this
        # pollster's reading, on average across all their polls?"
        new_house_effects = np.zeros(n_pollsters)
        counts = np.zeros(n_pollsters)

        for idx, pid in zip(poll_idx, pollster_ids):
            # M-step: estimate house effect as average residual (poll - smoothed)
            # intuitively: "how far above or below the smoothed true opinion was this
            # pollster's reading, on average across all their polls?"
            new_house_effects[pid] += (df.loc[idx, 'poll_margin'] - S[idx])
            counts[pid] += 1

        # average residuals where count > 0
        mask = counts > 0
        new_house_effects[mask] /= counts[mask]

        # apply mean-zero constraint: subtract weighted mean so house effects
        # represent deviations from industry average, not from absolute truth
        # weight by number of polls so high-volume pollsters don't dominate the centering
        weighted_mean = np.average(new_house_effects[mask],
                                   weights=counts[mask])
        new_house_effects -= weighted_mean

        # re-estimate sigma2_u on corrected margins (re-estimates each iteration
        # because correcting for house effects changes the residual variance, which
        # in turn changes how volatile true opinion appears to be)
        sigma2_u = estimate_sigma2u(y_corrected, obs_var, days)

        # check convergence: have house effects stopped changing?
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
    # run one final e-step with converged house effects to get clean estimates
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

    # assign each row's house effect (election result row gets 0)
    df['house_effect_assigned'] = 0.0
    for idx, pid in zip(poll_idx, pollster_ids):
        df.loc[idx, 'house_effect_assigned'] = house_effects[pid]

    # bias decomposition — now three components instead of two:
    #
    # total_error          = poll_margin - true_margin
    #                        (everything wrong with any given poll)
    #
    # house_effect_assigned = the pollster's estimated systematic offset
    #                        (in english: how much this pollster typically over/understates trump)
    #
    # sampling_noise        = corrected_margin - smoothed
    #                        (random sampling fluctuation after house effect removed)
    #
    # residual_systematic_bias = smoothed - true_margin
    #                        (aggregate industry bias that persists even after removing
    #                         individual house effects. this is the bias shared by all
    #                         pollsters — e.g., systematic undercoverage of trump voters
    #                         that no individual house effect can capture)
    #
    # note: total_error = house_effect_assigned + sampling_noise + residual_systematic_bias
    # (approximately — there is a small cross-term from the mean-zero constraint)

    true_margin = df['true_margin'].iloc[0]
    df['total_error']              = df['poll_margin']    - true_margin
    df['sampling_noise']           = df['corrected_margin'] - df['smoothed']
    df['residual_systematic_bias'] = df['smoothed']       - true_margin

    # build house effects summary dataframe
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
    print full decomposition including house effects
    
    state parameter added for swing state version to label output correctly
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
    four-panel figure:
      panel 1: raw polls, corrected polls, smoothed estimate, true margin
      panel 2: standard errors — conventional vs smoothed
      panel 3: residual systematic bias over time (after house effects removed)
      panel 4: top 20 house effects by absolute size (bar chart)
      
    SWING STATE MODIFICATIONS:
    - state parameter added to customize plot title
    - poll count (n) added to subtitle
    - plt.close() added after saving to prevent blocking
    """
    results = df[df['pollster_id'] != -1].copy()
    true_margin = results['true_margin'].iloc[0]
    n_polls = len(results)  # ADDED: count polls for subtitle

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=False)
    mode_label = "anchored" if anchored else "unanchored"
    # title includes state name and poll count
    fig.suptitle(
        f'kalman filter with house effects: {state} polling 2024 ({mode_label})\n(trump margin, pp | n={n_polls} polls)',
        fontsize=14, fontweight='bold', y=0.99
    )

    dates  = results['end_date']
    colors = {
        'raw':      '#cccccc',
        'corrected':'#aec6e8',
        'smoothed': '#DD8452',
        'true':     '#2ca02c',
        'bias':     '#d62728',
    }

    # panel 1: raw polls, corrected polls, smoothed, true
    ax1 = axes[0]
    ax1.scatter(dates, results['poll_margin'],
                color=colors['raw'], alpha=0.2, s=8, label='raw polls', zorder=1)
    ax1.scatter(dates, results['corrected_margin'],
                color=colors['corrected'], alpha=0.3, s=8,
                label='house-effect corrected polls', zorder=2)
    ax1.plot(dates, results['smoothed'],
             color=colors['smoothed'], linewidth=2.5,
             label='smoothed estimate', zorder=4)
    ax1.axhline(true_margin, color=colors['true'], linewidth=2,
                label=f'true margin ({true_margin:.2f} pp)', zorder=5)
    ax1.fill_between(dates,
                     results['smoothed'] - 1.96 * results['smoothed_se'],
                     results['smoothed'] + 1.96 * results['smoothed_se'],
                     color=colors['smoothed'], alpha=0.15, label='smoothed 95% ci')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax1.set_ylabel('trump margin (pp)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('raw polls vs house-effect corrected polls vs smoothed estimate', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # panel 2: standard errors
    ax2 = axes[1]
    conv_se = results['sampling_var'].apply(np.sqrt)
    ax2.plot(dates, conv_se, color=colors['raw'], linewidth=1, alpha=0.6,
             label='conventional se (per-poll)')
    ax2.plot(dates, results['smoothed_se'], color=colors['smoothed'], linewidth=2,
             label='smoothed se')
    ax2.set_ylabel('standard error (pp)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_title('uncertainty: conventional vs kalman smoothed', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # panel 3: residual systematic bias (the industry-wide component after house effects removed)
    # this is the bias that no individual pollster correction can fix — it's shared by all
    ax3 = axes[2]
    ax3.plot(dates, results['residual_systematic_bias'],
             color=colors['bias'], linewidth=2,
             label='residual systematic bias (smoothed - true)')
    ax3.fill_between(dates, 0, results['residual_systematic_bias'],
                     where=results['residual_systematic_bias'] >  0,
                     color=colors['bias'], alpha=0.15, label='pro-trump region')
    ax3.fill_between(dates, 0, results['residual_systematic_bias'],
                     where=results['residual_systematic_bias'] <= 0,
                     color='#4C72B0', alpha=0.15, label='pro-harris region')
    ax3.axhline(0, color='black', linewidth=1.0)
    ax3.set_ylabel('residual bias (pp)', fontsize=10)
    ax3.legend(fontsize=8)
    if anchored:
        ax3.set_title(
            'residual industry bias after removing house effects\n'
            '(positive = aggregate industry overstated trump even after correcting individual pollsters)',
            fontsize=10
        )
    else:
        ax3.set_title(
            'residual trajectory vs certified result after removing house effects\n'
            '(positive = smoothed trajectory overstated trump; final value = corrected forecast error)',
            fontsize=10
        )
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # panel 4: house effects bar chart (top 20 by absolute size, min 5 polls)
    # positive = pollster systematically overstates trump relative to industry average
    # negative = pollster systematically overstates harris relative to industry average
    ax4 = axes[3]
    ax4.sharex = None  # this panel has its own x-axis (pollster names, not dates)
    he_plot = (house_effects_df[house_effects_df['n_polls'] >= 5]
               .copy()
               .assign(abs_he=lambda x: x['house_effect'].abs())
               .nlargest(20, 'abs_he')
               .sort_values('house_effect', ascending=True))
    bar_colors = [colors['bias'] if v > 0 else '#4C72B0' for v in he_plot['house_effect']]
    ax4.barh(he_plot['pollster'], he_plot['house_effect'], color=bar_colors, alpha=0.8)
    ax4.axvline(0, color='black', linewidth=1.0)
    ax4.set_xlabel('house effect (pp, relative to industry average)', fontsize=10)
    ax4.set_title(
        'top 20 pollster house effects (min 5 polls)\n'
        '(positive = more pro-trump than industry average; negative = more pro-harris)',
        fontsize=10
    )
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nfigure saved to: {save_path}")

    # ADDED: close figure to prevent blocking execution (allows automated batch runs)
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
        'residual_systematic_bias', 'weight',
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
    DATA_PATH     = 'data/harris_trump_accuracy.csv'
    ELECTION_DATE = '2024-11-05'
    TIME_WINDOWS  = [None, 200, 107]
    
    # analyze each swing state separately
    SWING_STATES  = ['arizona', 'georgia', 'michigan', 'nevada', 'north carolina', 'pennsylvania', 'wisconsin']

    # set up logging
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'output/kalman_house_effects_swingstates_log_{timestamp}.txt'
    logger       = Logger(log_filename)
    sys.stdout   = logger

    print(f"kalman filter with house effects analysis — swing states")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"log file: {log_filename}")
    print(f"states: {SWING_STATES}")
    print(f"time windows: {TIME_WINDOWS}")

    # SWING STATE MODIFICATION: outer loop over states
    for state in SWING_STATES:

        # state abbreviation for compact filenames
        state_abbrev = {
            'Arizona': 'AZ',
            'Georgia': 'GA',
            'Michigan': 'MI',
            'Nevada': 'NV',
            'North Carolina': 'NC',
            'Pennsylvania': 'PA',
            'Wisconsin': 'WI'
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

                # pass state parameter to load_and_prepare
                df, pollster_names = load_and_prepare(DATA_PATH, state, ELECTION_DATE, days_before=days_before)
                print(f"\nsanity check — unique true_margin values: {df['true_margin'].unique()}")

                df, pollster_names = append_election_result(df, pollster_names, ELECTION_DATE, anchor=anchor)

                df, house_effects_df, sigma2u, history = em_kalman_house_effects(
                    df, pollster_names, max_iter=50, tol=1e-4
                )

                # pass state parameter to summarize_results
                summarize_results(df, house_effects_df, sigma2u, state, anchored=anchor)

                # pass state parameter and use state_abbrev in filename
                plot_results(
                    df, house_effects_df, state, anchored=anchor,
                    save_path=f'figures/kalman_he_{state_abbrev}_{mode_label}_{window_label}.png'
                )

                # use state_abbrev in output filenames
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