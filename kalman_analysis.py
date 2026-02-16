import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# THIS FILE ANALYZES POLLS USING KALMAN FILTERING/SMOOTHING

########################################################################################
##################################### Load Data ########################################
########################################################################################
def load_and_prepare(filepath: str) -> pd.DataFrame:
    # load the polling dataset and prepare it for kalman analysis
    df = pd.read_csv(filepath)

    # filter to national polls only
    df = df[df['state'] == 'national'].copy()

    # using end_date as the poll date throughout
    df['end_date'] = pd.to_datetime(df['end_date'])

    # construct poll margin in percentage points (raw difference)
    # use the raw difference rather than the two party adjusted margin because: (a) presidential races stay near 50/50 so the adjustment is negligible, (b) the additive structure fits the kalman model directly, and (c) results are immediately interpretable in pp
    # caveat: if pct_dk varies systematically across time or pollsters (e.g., some pollsters prompt for a choice, others don't), this introduces a soft form of house effect that we do not correct for here
    df['poll_margin'] = df['pct_trump_poll'] - df['pct_harris_poll']

    # construct true margin in percentage points to match poll_margin scale
    # p_trump_true and p_harris_true are stored as proportions so we multiply by 100
    df['true_margin'] = (df['p_trump_true'] - df['p_harris_true']) * 100

    # sampling variance per observation, in percentage-point units
    
    # departure from the paper: green et al. track a single percentage (e.g., % republican), so their observable is 100*pT and their sampling variance formula is simply pT*(1-pT)/N * 10000 (p.181, with plug-in pT for the unknown true proportion, footnote 3)

    # our observable is the margin 100*(pT - pH), a *difference* of two shares from the same multinomial sample
    # the correct variance is:
    #   var(100*(pT - pH)) = 10000 * var(pT - pH)
    #                      = 10000 * (var(pT) + var(pH) - 2*cov(pT, pH)) / N
    
    # in a multinomial sample: var(pT) = pT*(1-pT), var(pH) = pH*(1-pH), and cov(pT, pH) = -pT*pH  (negative because the counts compete)
    # substituting and simplifying:
    #   var(pT - pH) = (pT*(1-pT) + pH*(1-pH) + 2*pT*pH) / N
    #               = (pT + pH - (pT - pH)^2) / N
    
    # near 50/50 (e.g., pT=0.47, pH=0.46) this is multiple times larger than the paper's single-share formula
    # using the paper's formula here would systematically understate sampling variance, cause the filter to over-weight individual polls, and under-smooth, inflating the sampling noise component and shrinking the systematic bias estimate
    
    # still follow the paper's plug-in convention (footnote 3): observed shares substitute for unknown true shares
    
    # caveat: this still understates variance for polls with stratified sampling, nonresponse weighting, or design effects > 1, since those factors increase effective variance beyond the simple multinomial; the paper notes this limitation explicitly (p.181, footnote 3)
    pT = df['pct_trump_poll']  / 100.0
    pH = df['pct_harris_poll'] / 100.0
    df['sampling_var'] = (pT + pH - (pT - pH) ** 2) / df['sample_size'] * 10000


    # drop rows missing any required value
    df = df.dropna(subset=['end_date', 'poll_margin', 'sampling_var', 'sample_size'])
    df = df[df['sample_size'] > 0]

    # sort chronologically
    df = df.sort_values('end_date').reset_index(drop=True)

    # caveat: some polls contribute multiple rows (one per question wording)
    # these share the same sample and interview dates, so they are not independent draws, so treating them as independent slightly understates sampling variance at dates with multiple questions from the same poll
    # for now this is a reasonable approximation, a cleaner approach would average across questions within a poll before fitting the filter which may try later
    n_polls = df['poll_id'].nunique()
    n_questions = len(df)
    print(f"national polls loaded: {n_questions} questions from {n_polls} polls")
    print(f"date range: {df['end_date'].min().date()} to {df['end_date'].max().date()}")
    print(f"true margin (constant across rows): {df['true_margin'].iloc[0]:.3f} pp")
    print(f"poll margin range: {df['poll_margin'].min():.1f} to {df['poll_margin'].max():.1f} pp")
    print(f"mean poll margin: {df['poll_margin'].mean():.3f} pp")

    return df


########################################################################################
######################### Anchor to true result ########################################
########################################################################################
def append_election_result(df: pd.DataFrame, election_date: str = '2024-11-05', anchor: bool = True) -> pd.DataFrame:
    """
    optionally treat the certified election result as a final synthetic poll with effectively zero sampling variance (via an enormous synthetic sample size)

    this function operates in two modes:

    anchored (anchor=True, default):
    - adds the election result as a terminal observation, forcing the smoother to reconcile all polls with the known true outcome
    - this enables retrospective bias decomposition: systematic_bias measures how far the smoothed trajectory was from truth at each point
    - the question answered is "given that the outcome was X, what latent opinion trajectory best explains the polls?"

    unanchored (anchor=False): 
    - skips the synthetic observation
    - the smoother estimates the latent opinion trajectory based solely on polls and model assumptions, with no constraint to match the final result
    - systematic_bias becomes an out-of-sample diagnostic (final smoothed estimate minus true result), not something the model was conditioned on
    - the question answered is "based on polls alone, what did underlying opinion look like in real time?"

    the anchored version is appropriate for bias decomposition
    the unanchored version is appropriate for understanding real-time poll aggregation and evaluating forecast performance
    """
    true_margin = df['true_margin'].iloc[0]  # same value for all national rows

    if not anchor:
        print(f"\nskipping election result anchor (unanchored mode)")
        print(f"true margin: {true_margin:.3f} pp (not used as constraint)")
        return df

    anchor = {
        'question_id':  -1,
        'poll_id':       -1,
        'pollster':      'ELECTION_RESULT',
        'state':         'national',
        'end_date':       pd.to_datetime(election_date),
        'poll_margin':    true_margin,
        'true_margin':    true_margin,
        'p_hat':          true_margin / 100.0 + 0.5,  # approx trump share
        'sampling_var':   1e-6,                        # effectively zero
        'sample_size':    1_000_000_000,
    }

    df = pd.concat([df, pd.DataFrame([anchor])], ignore_index=True)
    df = df.sort_values('end_date').reset_index(drop=True)

    print(f"\nelection result appended: margin = {true_margin:.3f} pp on {election_date}")
    return df


########################################################################################
######################### estimate sigma2_u with grid search ###########################
########################################################################################
def estimate_sigma2u(y: np.ndarray,
                     obs_var: np.ndarray,
                     days: np.ndarray,
                     n_coarse: int = 500,
                     n_fine: int = 200) -> float:
    """
    estimate sigma2_u (the disturbance variance per day) by maximizing the gaussian log-likelihood of one-step-ahead forecast errors

    this is the standard mle objective for state-space models
    follows green et al. (1999) directly: they state on p.184 that their software "uses a grid-search algorithm, so that users are not required to supply starting values"
    the grid search is therefore the paper's own recommended estimation method, not an approximation

    we use a two-stage grid: a coarse log-space search to find the ballpark, followed by a finer linear search in a narrow window around the coarse optimum
    this improves precision over a single coarse grid while remaining robust to multiple local optima (rare for this model but possible with sparse or highly irregular time series)

    the log-likelihood is: ll = sum_t [ -0.5 * (log(2*pi*F_t) + v_t^2 / F_t) ] where v_t = y_t - y_hat_t is the one-step-ahead forecast error and F_t = P_pred_t + obs_var_t is its variance (hamilton 1994, ch. 13)
    """

    def innovations_loglik(s2):
        # evaluate the innovations log-likelihood for a given sigma2_u
        n = len(y)
        ll = 0.0
        F_t = y[0]
        P_t = obs_var[0] + s2  # initial uncertainty after first observation

        for t in range(1, n):
            days_elapsed = days[t] - days[t - 1]
            P_pred = P_t + s2 * days_elapsed # predicted variance
            innov = y[t] - F_t # forecast error
            innov_var = P_pred + obs_var[t] # its variance

            # guard against numerical issues with near zero variance
            if innov_var <= 0:
                return -np.inf

            ll += -0.5 * (np.log(2 * np.pi * innov_var) + innov ** 2 / innov_var)

            W_t = P_pred / innov_var
            F_t  = W_t * y[t] + (1 - W_t) * F_t
            P_t  = P_pred * (1 - W_t)

        return ll

    # coarse log-space grid
    coarse_grid = np.logspace(-4, 1, n_coarse)
    ll_coarse   = np.array([innovations_loglik(s2) for s2 in coarse_grid])
    best_idx    = np.argmax(ll_coarse)
    best_coarse = coarse_grid[best_idx]

    # fine linear grid around coarse optimum
    lo        = coarse_grid[max(0, best_idx - 1)]
    hi        = coarse_grid[min(len(coarse_grid) - 1, best_idx + 1)]
    fine_grid = np.linspace(lo, hi, n_fine)
    ll_fine   = np.array([innovations_loglik(s2) for s2 in fine_grid])
    best_s2   = fine_grid[np.argmax(ll_fine)]

    print(f"  sigma2_u estimated (two-stage grid mle): {best_s2:.6f} per day")
    print(f"  interpretation: true opinion can move ~{np.sqrt(best_s2):.3f} pp/day (1 sd)")

    return best_s2


########################################################################################
######################### kalman filtering and smoothing ###############################
########################################################################################

def kalman_filter_smoother(df: pd.DataFrame,sigma2_u_per_day: float = None) -> tuple:
    """
    manual implementation of the green et al. (1999) kalman filter and smoother

    this manual implementation is used rather than a packaged state-space routine (e.g., statsmodels.UnobservedComponents) because packaged routines
    assume a single fixed observation variance, whereas each poll here has a distinct sampling variance determined by its sample size n
    
    the manual implementation passes per-observation variances directly into the kalman gain calculation at each step

    state-space model:
        observation eq:  poll_margin_t = true_margin_t + e_t
                         e_t ~ N(0, sampling_var_t)   [known, varies by poll]
        state eq:        true_margin_t = true_margin_{t-1} + u_t
                         u_t ~ N(0, sigma2_u * days_elapsed_t)   [estimated]

    the random walk (gamma = 1) assumption follows the paper's baseline model

    green et al. note that for slowly evolving series like partisanship, gamma = 1 "typically provides a good approximation" (p.190)
    
    for more volatile series that re-equilibrate quickly, one could estimate gamma as a free parameter, this is a potential extension
    """

    df = df.copy().sort_values('end_date').reset_index(drop=True)
    day_0 = df['end_date'].min()
    df['day'] = (df['end_date'] - day_0).dt.days

    y       = df['poll_margin'].values
    obs_var = df['sampling_var'].values
    days    = df['day'].values
    n       = len(y)

    # estimate sigma2_u if not supplied
    if sigma2_u_per_day is None:
        print("\nestimating sigma2_u via grid search mle...")
        sigma2_u_per_day = estimate_sigma2u(y, obs_var, days)

    ######## forward pass: kalman filter
    # produces filtered estimates F[t] = E[true_margin_t | polls 1..t] and associated uncertainty P[t] = Var[true_margin_t | polls 1..t]
    # filtered estimates use only past and current polls, making them appropriate for real-time forecasting but not retrospective analysis

    F = np.zeros(n) # filtered estimates
    P = np.zeros(n) # filtered uncertainty (variance)
    W = np.zeros(n) # kalman gain (weight on new observation), eq 5

    # diffuse prior: first filtered estimate = first observation at face value
    # the paper initializes with "an (extremely diffuse) variance of 1,000" (table 2 note)
    # we use obs_var[0] + sigma2_u as a data-driven proxy since a hardcoded 1,000 is arbitrary and scale-dependent
    F[0] = y[0]
    P[0] = obs_var[0] + sigma2_u_per_day

    for t in range(1, n):
        days_elapsed = days[t] - days[t - 1]

        # predicted variance: uncertainty grows with time since last poll (signal noise accumulates as opinion can drift between observations)
        P_pred = P[t - 1] + sigma2_u_per_day * days_elapsed

        # kalman gain: weight on new observation vs. prior (green et al. eq 5)
        # higher sampling error in the new poll -> lower weight on it
        W[t] = P_pred / (P_pred + obs_var[t])

        # updated filtered estimate (green et al. eq. 6)
        F[t] = W[t] * y[t] + (1 - W[t]) * F[t - 1]

        # updated uncertainty (green et al. eq 7)
        P[t] = P_pred * (1 - W[t])

    ###### backward pass: kalman smoother
    # produces smoothed estimates S[t] = E[true_margin_t | all polls]
    # smoother is more precise than the filter because it incorporates future information

    # green et al. (pg 184): "smoothed estimates are more precise than filtered estimates, because the latter are based solely on information up to and including the current time period"
    
    # the smoother has the smallest mse of any linear weighting scheme applied to the poll sequence, assuming correct model specification and normally distributed errors (hamilton 1994, cited by paper pg 184)

    # departure from the paper: rts smoother vs eq 8
    # green et al. eq. (8) gives the smoothed point estimate as: S_{T-1} = F_{T-1} + (S_T - F_{T-1}) * (1 - sigma2_u / P_T)
    # this is written for unit time steps (one period = one poll interval)
    # the smoothing factor (1 - sigma2_u / P_T) implicitly assumes that P_pred = P_{T-1} + sigma2_u, i.e., exactly one unit of time elapsed
    # we instead use the standard rauch-tung-striebel (rts) smoother:
    # G = P[t] / P_pred (smoothing gain)
    # S[t] = F[t] + G * (S[t+1] - F[t])
    # where P_pred = P[t] + sigma2_u * days_elapsed
    # rts is derived in hamilton (1994)

    # this is the correct general form that handles uneven time spacing when polls are days or weeks apart, the predicted variance must grow in proportion to the gap, not by a fixed unit amount
    # with days_elapsed=1 and integerspaced polls, rts and eq 8 are algebraically identical
    # for unevenly spaced data like here, eq 8 taken literally would under-smooth across long gaps and over-smooth across short ones

    # for smoothed variances (PS): the paper provides no variance recursion alongside eq 8, it presents only the point estimate
    # we use the standard rts variance recursion from hamilton (1994): PS[t] = P[t] + G^2 * (PS[t+1] - P_pred)
    # an alternative conservative fallback would be PS[t] = P[t] (filtered variance), which would not show the precision gain from smoothing that the paper highlights (won't try this here)

    S  = np.zeros(n) # smoothed estimates
    PS = np.zeros(n) # smoothed uncertainty (variance)

    # last smoothed estimate = last filtered estimate (no future information)
    S[n - 1]  = F[n - 1]
    PS[n - 1] = P[n - 1]

    for t in range(n - 2, -1, -1):
        days_elapsed = days[t + 1] - days[t]

        # predicted variance for period t+1 given information through t
        P_pred = P[t] + sigma2_u_per_day * days_elapsed

        # rts smoothing gain: G = P[t] / P_pred
        # generalises eq 8 to uneven spacing
        # the more true opinion changes over time (large sigma2_u * days_elapsed), the larger P_pred relative to P[t], the smaller G, and the less future information revises the estimate of opinion at the current period
        G = P[t] / P_pred

        # smoothed estimate and variance (green et al. eq 8)
        S[t]  = F[t] + G * (S[t + 1] - F[t])
        PS[t] = P[t] + G ** 2 * (PS[t + 1] - P_pred)

    # store results back on dataframe
    df['filtered']    = F
    df['filtered_se'] = np.sqrt(np.maximum(P,  0))
    df['smoothed']    = S
    df['smoothed_se'] = np.sqrt(np.maximum(PS, 0))
    df['weight']      = W

    ######## bias decomposition
    # total_error = observed poll margin minus certified true margin (everything wrong with any given poll)
    # sampling_noise = observed poll margin minus smoothed estimate (the part the filter attributed to random sampling fluctuation, averages to zero across many polls if the model is correctly specified)
    # systematic_bias = smoothed estimate minus certified true margin (the part that survived smoothing, is persistent, industry-wide directional error shared across polls)

    # caveat: systematic_bias as defined here is aggregate industry bias
    # it does not separate house effects (pollster-specific offsets) from mode effects, likely voter screen differences, or herding
    # extending the model to estimate per-pollster fixed offsets is a natural next step (will do in extension)

    # caveat: if the random walk assumption is wrong (e.g., opinion was actually mean-reverting), sigma2_u will be mis-estimated and the smoother will over- or under-smooth, distorting the decomposition

    true_margin = df['true_margin'].iloc[0]
    df['total_error']     = df['poll_margin'] - true_margin
    df['sampling_noise']  = df['poll_margin'] - df['smoothed']
    df['systematic_bias'] = df['smoothed']    - true_margin

    return df, sigma2_u_per_day


########################################################################################
######################### Summary stats @###############################################
########################################################################################

def summarize_bias(df: pd.DataFrame) -> pd.DataFrame:
    # print a decomposition of polling error into systematic vs random components
    # exclude the synthetic election result row from summary statistics
    results = df[df['pollster'] != 'ELECTION_RESULT'].copy()
    true_margin = results['true_margin'].iloc[0]

    print("\n" + "=" * 60)
    print("polling bias decomposition — national (trump margin)")
    print("=" * 60)
    print(f"\ntrue margin (certified result): {true_margin:.3f} pp")

    print(f"\n--- overall poll error ---")
    print(f"  mean poll margin:          {results['poll_margin'].mean():.3f} pp")
    print(f"  mean total error:          {results['total_error'].mean():.3f} pp")
    print(f"  sd of total error:         {results['total_error'].std():.3f} pp")

    print(f"\n--- decomposition (means) ---")
    print(f"  mean systematic bias:      {results['systematic_bias'].mean():.3f} pp")
    print(f"  mean |sampling noise|:     {results['sampling_noise'].abs().mean():.3f} pp")

    # variance decomposition: what share of total poll error variance
    # is systematic (survives smoothing) vs. random (filtered out)?
    # caveat: this decomposition assumes sampling_noise and systematic_bias are orthogonal, which holds approximately but not exactly in practice
    var_total      = results['total_error'].var()
    var_systematic = results['systematic_bias'].var()
    var_noise      = results['sampling_noise'].var()
    print(f"\n--- variance decomposition ---")
    print(f"  var(total error):          {var_total:.4f}")
    print(f"  var(systematic bias):      {var_systematic:.4f}  ({100 * var_systematic / var_total:.1f}%)")
    print(f"  var(sampling noise):       {var_noise:.4f}  ({100 * var_noise / var_total:.1f}%)")

    return results


########################################################################################
######################### Visualization ################################################
########################################################################################
def plot_results(df: pd.DataFrame, save_path: str = None):
    """
    panel 1: raw polls, filtered estimate, smoothed estimate, true margin
    panel 2: standard errors, conventional vs filtered vs smoothed
    panel 3: systematic bias trajectory over time
    """
    results = df[df['pollster'] != 'ELECTION_RESULT'].copy()
    true_margin = results['true_margin'].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        'kalman filtering & smoothing: national polling bias 2021-2024\n(trump margin, pp)',
        fontsize=14, fontweight='bold', y=0.98
    )

    dates  = results['end_date']
    colors = {
        'raw':      '#aaaaaa',
        'filtered': '#4C72B0',
        'smoothed': '#DD8452',
        'true':     '#2ca02c',
        'bias':     '#d62728',
    }

    # panel 1: margins over time
    ax1 = axes[0]
    ax1.scatter(dates, results['poll_margin'],
                color=colors['raw'], alpha=0.25, s=10,
                label='raw polls', zorder=1)
    ax1.plot(dates, results['filtered'],
             color=colors['filtered'], linewidth=1.5, linestyle='--',
             label='filtered estimate', zorder=3)
    ax1.plot(dates, results['smoothed'],
             color=colors['smoothed'], linewidth=2.5,
             label='smoothed estimate', zorder=4)
    ax1.axhline(true_margin,
                color=colors['true'], linewidth=2,
                label=f'true margin ({true_margin:.2f} pp)', zorder=5)
    ax1.fill_between(
        dates,
        results['smoothed'] - 1.96 * results['smoothed_se'],
        results['smoothed'] + 1.96 * results['smoothed_se'],
        color=colors['smoothed'], alpha=0.15, label='smoothed 95% ci'
    )
    ax1.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax1.set_ylabel('trump margin (pp)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title('poll margins, filtered & smoothed estimates vs. true result', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # panel 2: standard errors
    # the dramatic reduction in se from conventional to smoothed reflects the gain from pooling information across polls (green et al. p.188: smoothed se is typically less than one-third of per-poll se)
    ax2 = axes[1]
    conv_se = results['sampling_var'].apply(np.sqrt)
    ax2.plot(dates, conv_se,
             color=colors['raw'], linewidth=1, alpha=0.6,
             label='conventional se (per-poll)')
    ax2.plot(dates, results['filtered_se'],
             color=colors['filtered'], linewidth=1.5, linestyle='--',
             label='filtered se')
    ax2.plot(dates, results['smoothed_se'],
             color=colors['smoothed'], linewidth=2,
             label='smoothed se')
    ax2.set_ylabel('standard error (pp)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('uncertainty reduction: conventional vs. kalman standard errors', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # panel 3: systematic bias trajectory
    # positive values = polls overstated trump relative to the true result;
    # negative values = polls overstated harris.
    # this is aggregate industry bias only, house effects and other pollster-specific components are not separated out here
    ax3 = axes[2]
    ax3.plot(dates, results['systematic_bias'],
             color=colors['bias'], linewidth=2,
             label='systematic bias (smoothed - true)')
    ax3.fill_between(dates, 0, results['systematic_bias'],
                     where=results['systematic_bias'] >  0,
                     color=colors['bias'],     alpha=0.15, label='pro-trump bias region')
    ax3.fill_between(dates, 0, results['systematic_bias'],
                     where=results['systematic_bias'] <= 0,
                     color=colors['filtered'], alpha=0.15, label='pro-harris bias region')
    ax3.axhline(0, color='black', linewidth=1.0)
    ax3.set_ylabel('systematic bias (pp)', fontsize=11)
    ax3.set_xlabel('date', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_title(
        'systematic industry bias over time\n'
        '(positive = polls overstated trump relative to certified result)',
        fontsize=11
    )
    ax3.grid(True, alpha=0.3)

    # format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nfigure saved to: {save_path}")

    plt.show()
    return fig


########################################################################################
######################### Output of results ############################################
########################################################################################

def export_results(df: pd.DataFrame, out_path: str = 'kalman_results.csv') -> pd.DataFrame:
    """
    export the full results dataframe for further analysis
    excludes the synthetic election result anchor row
    """
    cols = [
        'question_id', 'poll_id', 'pollster', 'end_date', 'sample_size',
        'poll_margin', 'true_margin',
        'filtered',    'filtered_se',
        'smoothed',    'smoothed_se',
        'total_error', 'sampling_noise', 'systematic_bias',
        'weight',
    ]
    if 'period' in df.columns:
        cols.append('period')

    out = df[df['pollster'] != 'ELECTION_RESULT'][cols].copy()
    out.to_csv(out_path, index=False)
    print(f"results exported to: {out_path}")
    return out



########################################################################################
######################### Actually run all these functions from here ###################
########################################################################################

if __name__ == '__main__':

    # important values
    DATA_PATH     = 'data/harris_trump_accuracy.csv'
    ELECTION_DATE = '2024-11-05'
    FIG_PATH      = 'figures/kalman_polling_bias.png'
    RESULTS_PATH  = 'output/kalman_results.csv'

    # run pipeline
    df = load_and_prepare(DATA_PATH)

    # sanity check: true_margin should be near +1.5 pp
    print(f"\nsanity check — unique true_margin values: {df['true_margin'].unique()}")

    df = append_election_result(df, election_date=ELECTION_DATE)

    df, sigma2u = kalman_filter_smoother(df)

    summarize_bias(df)

    plot_results(df, save_path=FIG_PATH)

    export_results(df, out_path=RESULTS_PATH)




# TODO
# edit to have per pollster fixed effects
# state level instead of just national (by each swing state)
# cut the dates to start like 200 days before election
# clean figures
# anchoring vs not