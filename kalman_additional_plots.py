import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

########################################################################
# figure 2: national - residual systematic bias comparison (all four implementations)
########################################################################
# USED
def create_figure2_national_bias_comparison():
    """
    compare residual/systematic bias trajectories across all four implementations
    to test robustness of findings to methodological choices.
    """
    # load data from all four implementations (107 day window)
    em_anch = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    em_unanch = pd.read_csv('data/kalman_he_polls_unanchored_last_107_days.csv')
    agg_anch = pd.read_csv('data/kalman_agg_results_anchored_last_107_days.csv')
    agg_unanch = pd.read_csv('data/kalman_agg_results_unanchored_last_107_days.csv')
    
    # convert dates
    em_anch['end_date'] = pd.to_datetime(em_anch['end_date'])
    em_unanch['end_date'] = pd.to_datetime(em_unanch['end_date'])
    agg_anch['end_date'] = pd.to_datetime(agg_anch['end_date'])
    agg_unanch['end_date'] = pd.to_datetime(agg_unanch['end_date'])
    
    # create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # plot all four trajectories
    ax.plot(em_anch['end_date'], em_anch['residual_systematic_bias'],
            color='#d62728', linewidth=2.5, label='Pollster-Adjusted Anchored', zorder=4)
    ax.plot(em_unanch['end_date'], em_unanch['residual_systematic_bias'],
            color='#d62728', linewidth=2.5, linestyle='--', label='Pollster-Adjusted Unanchored', zorder=3)
    ax.plot(agg_anch['end_date'], agg_anch['systematic_bias'],
            color='#1f77b4', linewidth=2, label='Daily-Aggregated Anchored', zorder=2)
    ax.plot(agg_unanch['end_date'], agg_unanch['systematic_bias'],
            color='#1f77b4', linewidth=2, linestyle='--', label='Daily-Aggregated Unanchored', zorder=1)
    
    # reference line at zero
    ax.axhline(0, color='black', linewidth=1, linestyle=':')
    
    # formatting
    ax.set_xlabel(r'$\bf{Date}$', fontsize=12)
    ax.set_ylabel(r'$\bf{Systematic\ Bias\ (pp)}$', fontsize=12)
    ax.set_title(r'$\bf{National\ Systematic\ Bias\ Trajectories\ Across\ Four\ Implementations}$' + '\n Last 107 Days', 
                 fontsize=14)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    plt.savefig('figures/kalman_national_bias_comparison_four_implementations.png', dpi=300, bbox_inches='tight')
    plt.close()


########################################################################
# figure 3: swing states - composite residual bias trajectories
########################################################################
## USED
def create_figure3_swing_states_bias_trajectories():
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan', 
        'NV': 'Nevada', 'NC': 'North Carolina', 
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    legend_handles = None

    df_nat_he = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    df_nat_he['end_date'] = pd.to_datetime(df_nat_he['end_date'])

    df_nat_agg = pd.read_csv('data/kalman_agg_results_anchored_last_107_days.csv')
    df_nat_agg['end_date'] = pd.to_datetime(df_nat_agg['end_date'])

    for idx, state in enumerate(states):
        ax = axes[idx]
        
        df = pd.read_csv(f'data/kalman_he_polls_{state}_anchored_last_107_days.csv')
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        df_agg = pd.read_csv(f'data/kalman_agg_{state}_results_anchored_last_107_days.csv')
        df_agg['end_date'] = pd.to_datetime(df_agg['end_date'])
        
        line_1 = ax.plot(df['end_date'], df['residual_systematic_bias'],
               color='#d62728', linewidth=2, label='State bias')[0]
        
        ax.fill_between(df['end_date'], 0, df['residual_systematic_bias'],
                      where=df['residual_systematic_bias'] > 0,
                      color='#d62728', alpha=0.15)
        ax.fill_between(df['end_date'], 0, df['residual_systematic_bias'],
                      where=df['residual_systematic_bias'] <= 0,
                      color='#1f77b4', alpha=0.15)
        
        line_2 = ax.plot(df_agg['end_date'], df_agg['systematic_bias'],
               color='#2ca02c', linewidth=1.5, linestyle='--', label='Aggregated anchored')[0]
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        
        true_margin = df['true_margin'].iloc[0]
        ax.set_title(f"{state_names[state]}\n(True: Trump{true_margin:+.1f})", 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    
        if legend_handles is None:
            legend_handles = [line_1, line_2]

    ax = axes[7]

    ax.plot(df_nat_he['end_date'], df_nat_he['residual_systematic_bias'],
               color='#d62728', linewidth=2)

    ax.fill_between(df_nat_he['end_date'], 0, df_nat_he['residual_systematic_bias'],
                  where=df_nat_he['residual_systematic_bias'] > 0,
                  color='#d62728', alpha=0.15)
    ax.fill_between(df_nat_he['end_date'], 0, df_nat_he['residual_systematic_bias'],
                  where=df_nat_he['residual_systematic_bias'] <= 0,
                  color='#1f77b4', alpha=0.15)

    ax.plot(df_nat_agg['end_date'], df_nat_agg['systematic_bias'],
               color='#2ca02c', linewidth=1.5, linestyle='--')

    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_title('National\n(True: Trump+1.5)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    fig.legend(
        legend_handles,
        ['Pollster-Adjusted Residual Bias',
        'Daily Aggregated Systematic Bias'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    fig.text(0.5, 0.02, r'$\bf{Date}$', ha='center', fontsize=13)
    fig.text(0.02, 0.5, r'$\bf{Smoothed\ Bias\ (pp)}$', va='center', rotation='vertical', fontsize=13)
    fig.suptitle(
        r'$\bf{Swing\ State\ and\ National\ Polling\ Bias\ Trajectories}$' + '\n'
        'Pollster-Adjusted and Daily-Aggregated Models\n'
        'Anchored, Last 107 Days',
        fontsize=14,
        y=0.98
    )        
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.savefig('figures/kalman_swing_states_bias_trajectories_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

########################################################################
# figure 4: swing states - three-component bias decomposition (anchored)
########################################################################
def create_figure4_variance_decomposition():
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan',
        'NV': 'Nevada', 'NC': 'North Carolina',
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }

    color_he    = '#ff7f0e'
    color_noise = '#2ca02c'
    color_resid = '#d62728'

    var_decomp = {}

    for state in states:
        df = pd.read_csv(f'data/kalman_he_polls_{state}_anchored_last_107_days.csv')
        var_he    = df['house_effect_assigned'].var()
        var_noise = df['sampling_noise'].var()
        var_resid = df['residual_systematic_bias'].var()
        total_shares = var_he + var_noise + var_resid
        var_decomp[state] = {
            'house':    100 * var_he    / total_shares,
            'noise':    100 * var_noise / total_shares,
            'residual': 100 * var_resid / total_shares,
        }

    df_nat = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    var_he    = df_nat['house_effect_assigned'].var()
    var_noise = df_nat['sampling_noise'].var()
    var_resid = df_nat['residual_systematic_bias'].var()
    total_shares = var_he + var_noise + var_resid
    var_decomp['NAT'] = {
        'house':    100 * var_he    / total_shares,
        'noise':    100 * var_noise / total_shares,
        'residual': 100 * var_resid / total_shares,
    }

    states_sorted = sorted(states, key=lambda s: var_decomp[s]['residual']) + ['NAT']
    state_labels  = [state_names.get(s, 'National') for s in states_sorted]
    y_pos = np.arange(len(states_sorted))

    house_pcts = [var_decomp[s]['house']    for s in states_sorted]
    noise_pcts = [var_decomp[s]['noise']    for s in states_sorted]
    resid_pcts = [var_decomp[s]['residual'] for s in states_sorted]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(y_pos, house_pcts,
        color=color_he, alpha=0.85, label=r'House Effects $h_i$', height=0.6)
    ax.barh(y_pos, noise_pcts,
        left=house_pcts,
        color=color_noise, alpha=0.85, label=r'Sampling Noise $e_{it}$', height=0.6)
    ax.barh(y_pos, resid_pcts,
        left=np.array(house_pcts) + np.array(noise_pcts),
        color=color_resid, alpha=0.85, label=r'Latent Opinion Trajectory $\xi_t$', height=0.6)


    ax.set_yticks(y_pos)
    ax.set_yticklabels(state_labels, fontsize=11)
    ax.set_xlabel('Scaled Share of Total Poll Variance (%)', fontweight="bold", fontsize=11)
    ax.set_title(
        r'$\bf{Error\ Variance\ Decomposition\ by\ Geography}$' + '\n'
        'Pollster-Adjusted Anchored\nLast 107 Days',
        fontsize=13,
        pad=30
    )
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=10
    )
    ax.set_xlim([0, 100])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i, (h, n, r) in enumerate(zip(house_pcts, noise_pcts, resid_pcts)):
        ax.text(101, y_pos[i], f'{h:.0f} / {n:.0f} / {r:.0f}',
                va='center', ha='left', fontsize=9, color='#444444')   
    
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(
        'figures/kalman_variance_decomposition_by_geography.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()




########################################################################
# figure 5
########################################################################
def create_figure5_national_stability_temporal():
    """
    single-panel house effect stability scatter (107 days vs 30 days).
    tests whether pollster-level biases estimated by the em algorithm
    reflect stable methodological tendencies or window-specific noise.
    points close to the 45-degree line indicate stable house effects.
    """
    he_107 = pd.read_csv('data/kalman_he_effects_anchored_last_107_days.csv')
    he_30  = pd.read_csv('data/kalman_he_effects_anchored_last_30_days.csv')

    he_compare = he_107.merge(he_30, on='pollster', suffixes=('_107', '_30'), how='inner')
    he_compare = he_compare[he_compare['n_polls_107'] >= 5]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        he_compare['house_effect_107'],
        he_compare['house_effect_30'],
        s=he_compare['n_polls_30'] * 3,
        alpha=0.6, color='#1f77b4'
    )

    lims = [
        he_compare[['house_effect_107', 'house_effect_30']].min().min() - 0.5,
        he_compare[['house_effect_107', 'house_effect_30']].max().max() + 0.5
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect stability')

    # label top 5 outliers by distance from 45-degree line
    he_compare['distance'] = abs(
        he_compare['house_effect_107'] - he_compare['house_effect_30']
    )
    outliers = he_compare.nlargest(5, 'distance')
    for _, row in outliers.iterrows():
        ax.annotate(
            row['pollster'],
            xy=(row['house_effect_107'], row['house_effect_30']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.8
        )

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('House Effect — Last 107 Days (pp)', fontsize=12)
    ax.set_ylabel('House Effect — Last 30 Days (pp)', fontsize=12)
    ax.set_title(
        'House Effect Stability Across Time Windows\n'
        '(EM Anchored | Point size = number of polls in 30-day window)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(
        'figures/kalman_national_house_effect_stability_and_temporal_dynamics.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


########################################################################
# figure 6: swing states - bias trajectory overlay with national comparison
########################################################################
# USED
def create_figure6_swing_states_bias_overlay():
    """
    overlay residual systematic bias trajectories from all seven swing states
    on a single plot, with national em anchored trajectory as reference.
    
    shows:
    - whether swing state polls were biased in the same direction as national polls
    - whether bias was uniform across battlegrounds or geographically heterogeneous
    - whether individual states were outliers in magnitude or direction
    - how state-level bias evolved relative to the national trajectory
    """
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan',
        'NV': 'Nevada', 'NC': 'North Carolina',
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }

    # distinct colors for each state
    state_colors = {
        'AZ': '#1f77b4',
        'GA': '#ff7f0e',
        'MI': '#2ca02c',
        'NV': '#d62728',
        'NC': '#9467bd',
        'PA': '#8c564b',
        'WI': '#e377c2',
    }

    # filter to last 107 days
    election_date = pd.Timestamp('2024-11-05')
    cutoff = election_date - pd.Timedelta(days=107)

    fig, ax = plt.subplots(figsize=(14, 7))

    # plot each swing state trajectory
    for state in states:
        df = pd.read_csv(
            f'data/kalman_he_polls_{state}_anchored_last_107_days.csv'
        )
        df['end_date'] = pd.to_datetime(df['end_date'])
        df = df[df['end_date'] >= cutoff]

        ax.plot(
            df['end_date'],
            df['residual_systematic_bias'],
            color=state_colors[state],
            linewidth=1.8,
            alpha=0.75,
            label=state_names[state],
            zorder=3
        )
    
    # overlay national trajectory as bold reference
    nat = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    nat['end_date'] = pd.to_datetime(nat['end_date'])
    nat = nat[nat['end_date'] >= cutoff]

    ax.plot(
        nat['end_date'],
        nat['residual_systematic_bias'],
        color='black',
        linewidth=3.0,
        linestyle='--',
        label='National',
        zorder=5
    )
  
    # reference line at zero
    ax.axhline(0, color='gray', linewidth=1.0, linestyle=':')

    # shading to show pro-trump vs pro-harris regions
    ax.fill_between(
        [cutoff, election_date], -10, 0,
        color='#4C72B0', alpha=0.04, zorder=1
    )
    ax.fill_between(
        [cutoff, election_date], 0, 10,
        color='#d62728', alpha=0.04, zorder=1
    )

    # formatting
    ax.set_xlabel(r'$\bf{Date}$', fontsize=12)
    ax.set_ylabel(
        r'$\bf{Residual\ Systematic\ Bias\ (pp)}$',
        fontsize=11
    )
    ax.set_title(
        r'$\bf{Swing\ State\ and\ National\ Residual\ Systematic\ Bias\ Trajectories}$' +' \n' +
        'Pollster Adjusted, Anchored\n'
        'Last 107 Days)',
        fontsize=14,
    )
    ax.legend(
        loc='upper left', fontsize=10, framealpha=0.95,
        ncol=2
    )
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    plt.savefig(
        'figures/kalman_swing_states_bias_overlay_with_national.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


########################################################################
# figure 7: swing states - bias correlation across final 60 and 30 days
########################################################################

def create_figure7_swing_states_bias_correlation():
    """
    two-panel correlation matrix of residual systematic bias trajectories
    across all seven swing states, for the final 60 days (left) and
    final 30 days (right).

    shows:
    - whether bias accelerated at the same time across states
    - whether the industry failure was uniform (high correlations) or
      state-specific (low or heterogeneous correlations)
    - whether the correlation structure tightened closer to election day,
      consistent with industry-wide herding in the final weeks
    """
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan',
        'NV': 'Nevada', 'NC': 'North Carolina',
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }

    election_date = pd.Timestamp('2024-11-05')
    windows = {
        'Final 60 Days': 60,
        'Final 30 Days': 30
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (window_label, days) in zip(axes, windows.items()):
        cutoff = election_date - pd.Timedelta(days=days)

        # build a common date index and collect bias series per state
        bias_dict = {}
        for state in states:
            df = pd.read_csv(
                f'data/kalman_he_polls_{state}_anchored_last_107_days.csv'
            )
            df['end_date'] = pd.to_datetime(df['end_date'])
            df = df[df['end_date'] >= cutoff].copy()

            # one value per date (take mean if duplicates)
            df = df.groupby('end_date')[
                'residual_systematic_bias'
            ].mean()
            bias_dict[state_names[state]] = df

          
        # align all series to a common date index via outer join
        bias_df = pd.DataFrame(bias_dict)

        # interpolate any missing dates linearly
        # (states may not have polls on every single day)
        bias_df = bias_df.interpolate(method='linear', limit_direction='both')

        # compute correlation matrix
        corr = bias_df.corr()

        # plot heatmap
        im = ax.imshow(
            corr.values,
            cmap='RdBu_r',
            vmin=-1, vmax=1,
            aspect='auto'
        )

        # axis labels
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr.index, fontsize=10)

        # annotate cells with correlation values
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                val = corr.values[i, j]
                text_color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=9, color=text_color,
                    fontweight='bold'
                )

        ax.set_title(
            f'Bias Trajectory Correlations\n{window_label}',
            fontsize=12, fontweight='bold'
        )

        # colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Pearson correlation')

    fig.suptitle(
        'Cross-State Correlation of Residual Systematic Bias Trajectories\n'
        '(EM Anchored — do high correlations indicate industry-wide failure?)',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(
        'figures/kalman_swing_states_bias_correlation_60_30_days.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


########################################################################
# national summary - all four implementations × three time windows
########################################################################

def create_table1_national_summary():
    """
    comprehensive comparison table: 4 implementations x 3 time windows.
    shows how all methodological choices affect core estimates.
    """
    implementations = ['EM_anchored', 'EM_unanchored', 'Agg_anchored', 'Agg_unanchored']
    windows = ['last_107_days', 'last_60_days', 'last_30_days']
    window_labels = ['Last 107 Days', 'Last 60 Days', 'Last 30 Days']
    
    results = []
    
    for window, window_label in zip(windows, window_labels):
        for impl in implementations:
            if impl.startswith('EM'):
                mode = 'anchored' if 'anchored' in impl else 'unanchored'
                try:
                    df = pd.read_csv(f'data/kalman_he_polls_{mode}_{window}.csv')
                    
                    row = {
                        'Window': window_label,
                        'Implementation': impl.replace('_', ' ').title(),
                        'N_obs': len(df),
                        'True_margin': df['true_margin'].iloc[0],
                        'Mean_poll_margin': df['poll_margin'].mean(),
                        'Mean_total_error': df['total_error'].mean(),
                        'SD_total_error': df['total_error'].std(),
                        'Mean_abs_house_effect': df['house_effect_assigned'].abs().mean(),
                        'Mean_residual_bias': df['residual_systematic_bias'].mean(),
                        'Sigma2_u': df['sigma2_u'].iloc[0],
                        'Final_smoothed': df['smoothed'].iloc[-1],
                        'Final_forecast_error': df['smoothed'].iloc[-1] - df['true_margin'].iloc[0] if 'unanchored' in mode else np.nan,
                        'Var_systematic_pct': 100 * df['residual_systematic_bias'].var() / df['total_error'].var()
                    }
                    results.append(row)
                except FileNotFoundError:
                    continue
                    
            else:  # aggregated
                mode = 'anchored' if 'anchored' in impl else 'unanchored'
                try:
                    df = pd.read_csv(f'data/kalman_agg_results_{mode}_{window}.csv')
                    
                    row = {
                        'Window': window_label,
                        'Implementation': impl.replace('_', ' ').title(),
                        'N_obs': len(df),
                        'True_margin': df['true_margin'].iloc[0],
                        'Mean_poll_margin': df['poll_margin'].mean(),
                        'Mean_total_error': df['total_error'].mean(),
                        'SD_total_error': df['total_error'].std(),
                        'Mean_abs_house_effect': np.nan,  # not applicable for aggregated
                        'Mean_residual_bias': df['systematic_bias'].mean(),
                        'Sigma2_u': df['sigma2_u'].iloc[0],
                        'Final_smoothed': df['smoothed'].iloc[-1],
                        'Final_forecast_error': df['smoothed'].iloc[-1] - df['true_margin'].iloc[0] if 'unanchored' in mode else np.nan,
                        'Var_systematic_pct': 100 * df['systematic_bias'].var() / df['total_error'].var()
                    }
                    results.append(row)
                except FileNotFoundError:
                    continue
    
    table1 = pd.DataFrame(results)
    
    # format for display
    table1_formatted = table1.round({
        'True_margin': 3,
        'Mean_poll_margin': 3,
        'Mean_total_error': 3,
        'SD_total_error': 3,
        'Mean_abs_house_effect': 3,
        'Mean_residual_bias': 3,
        'Sigma2_u': 6,
        'Final_smoothed': 3,
        'Final_forecast_error': 3,
        'Var_systematic_pct': 1
    })
    
    # save as csv
    table1_formatted.to_csv('output/kalman/national_summary_all_implementations.csv', index=False)
    
    # save as latex
    latex_str = table1_formatted.to_latex(index=False, escape=False, 
                                          caption='National Summary: All Implementations and Time Windows',
                                          label='tab:national_summary')
    with open('output/kalman/national_summary_all_implementations.tex', 'w') as f:
        f.write(latex_str)
    
    return table1_formatted


########################################################################
# national em anchored - top 20 house effects + variance decomposition
########################################################################

def create_table2_house_effects_variance():
    """
    two sub-tables: (1) top 20 pollster house effects, (2) variance decomposition across time windows.
    the two most-cited results in one table.
    """
    # part a: top 20 house effects (all data)
    he_107 = pd.read_csv('data/kalman_he_effects_anchored_last_107_days.csv')
    he_107 = he_107[he_107['n_polls'] >= 5]  # min 5 polls
    he_107 = he_107.reindex(he_107['house_effect'].abs().nlargest(20).index)  # top 20 by |house effect|
    
    house_effects_table = he_107[['pollster', 'house_effect', 'n_polls']].copy()
    house_effects_table.columns = ['Pollster', 'House Effect (pp)', 'N Polls']
    house_effects_table = house_effects_table.round({'House Effect (pp)': 3})
    
    # part b: variance decomposition across time windows
    windows = ['last_107_days', 'last_60_days', 'last_30_days']
    window_labels = ['Last 107 Days', 'Last 60 Days', 'Last 30 Days']
    
    var_decomp_rows = []
    for window, label in zip(windows, window_labels):
        try:
            df = pd.read_csv(f'data/kalman_he_polls_anchored_{window}.csv')
            var_total = df['total_error'].var()
            var_he = df['house_effect_assigned'].var()
            var_noise = df['sampling_noise'].var()
            var_resid = df['residual_systematic_bias'].var()
            
            var_decomp_rows.append({
                'Time Window': label,
                'var(House Effects) \\%': 100 * var_he / var_total,
                'var(Sampling Noise) \\%': 100 * var_noise / var_total,
                'var(Residual Bias) \\%': 100 * var_resid / var_total
            })
        except FileNotFoundError:
            continue
    
    var_decomp_table = pd.DataFrame(var_decomp_rows).round(1)
    
    # save both parts
    house_effects_table.to_csv('output/kalman/national_top20_house_effects.csv', index=False)
    var_decomp_table.to_csv('output/kalman/national_variance_decomposition_by_window.csv', index=False)
    
    # combined latex
    latex_he = house_effects_table.to_latex(index=False, caption='Top 20 Pollster House Effects (EM Anchored, All Data)')
    latex_vd = var_decomp_table.to_latex(index=False, caption='Variance Decomposition Across Time Windows (EM Anchored)')
    
    with open('output/kalman/national_house_effects_and_variance_decomposition.tex', 'w') as f:
        f.write("% Part A: House Effects\n")
        f.write(latex_he)
        f.write("\n\n% Part B: Variance Decomposition\n")
        f.write(latex_vd)
    
    return house_effects_table, var_decomp_table


########################################################################
# swing states - summary statistics & forecast performance
########################################################################

def create_table3_swing_states_summary():
    """
    comprehensive state-level summary: polling quality, error sources, forecast accuracy for all 7 battlegrounds.
    """
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan', 
        'NV': 'Nevada', 'NC': 'North Carolina', 
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }
    
    results = []
    
    for state in states:
        try:
            # anchored for error decomposition
            df_anch = pd.read_csv(f'data/kalman_he_polls_{state}_anchored_last_107_days.csv')
            
            # unanchored for forecast error
            df_unanch = pd.read_csv(f'data/kalman_he_polls_{state}_unanchored_last_107_days.csv')
            
            var_total = df_anch['total_error'].var()
            var_he = df_anch['house_effect_assigned'].var()
            
            row = {
                'State': state_names[state],
                'N Polls': len(df_anch),
                'True Margin (pp)': df_anch['true_margin'].iloc[0],
                'Mean Poll Margin (pp)': df_anch['poll_margin'].mean(),
                'Mean Total Error (pp)': df_anch['total_error'].mean(),
                'Mean |House Effect| (pp)': df_anch['house_effect_assigned'].abs().mean(),
                'Mean Residual Bias (pp)': df_anch['residual_systematic_bias'].mean(),
                '$\\sigma^2_u$ (per day)': df_anch['sigma2_u'].iloc[0],
                'var(HE)/var(Total) \\%': 100 * var_he / var_total,
            }
            results.append(row)
            
        except FileNotFoundError:
            continue
    
    table3 = pd.DataFrame(results)
    
    # format
    table3_formatted = table3.round({
        'True Margin (pp)': 2,
        'Mean Poll Margin (pp)': 2,
        'Mean Total Error (pp)': 2,
        'Mean |House Effect| (pp)': 2,
        'Mean Residual Bias (pp)': 2,
        '$\\sigma^2_u$ (per day)': 6,
        'var(HE)/var(Total) \\%': 1,
    })
    
    # save
    table3_formatted.to_csv('output/kalman/swing_states_polling_quality_and_forecast_performance.csv', index=False)
    
    latex_str = table3_formatted.to_latex(index=False, escape=False,
                                          caption='Swing State Summary (Poll-Adjusted, Anchored, 107 Days)',
                                          label='tab:swing_states_summary')
    with open('output/kalman/swing_states_polling_quality_and_forecast_performance.tex', 'w') as f:
        f.write(latex_str)
    
    return table3_formatted

########################################################################
# figure A1: national - rolling pollster bias dispersion over time
########################################################################

def create_figureA1_national_rolling_dispersion():
    """
    plots the rolling standard deviation of total errors across pollsters
    over time, within the last 107 days.

    answers: as the election approaches, do pollsters spread apart or
    converge in their individual bias levels? declining rolling SD suggests
    herding/convergence; increasing rolling SD suggests methodological
    divergence.

    computed as: for each day, take the standard deviation of total_error
    across all polls on that day, then smooth with a rolling window to
    reduce day-to-day noise from uneven pollster coverage.
    """
    election_date = pd.Timestamp('2024-11-05')
    cutoff = election_date - pd.Timedelta(days=107)

    df = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[df['end_date'] >= cutoff].copy()

    # compute daily cross-pollster SD of total error
    # only meaningful on days with multiple pollsters
    daily_sd = (
        df.groupby('end_date')['total_error']
        .std()
        .reset_index()
        .rename(columns={'total_error': 'cross_pollster_sd'})
    )
    daily_sd = daily_sd[daily_sd['cross_pollster_sd'].notna()]

    # rolling 7-day mean of the daily SD to smooth day-to-day noise
    daily_sd = daily_sd.sort_values('end_date')
    daily_sd['rolling_sd'] = (
        daily_sd['cross_pollster_sd']
        .rolling(window=7, min_periods=3, center=True)
        .mean()
    )

    # also compute daily poll count for context
    daily_n = df.groupby('end_date').size().reset_index(name='n_polls')
    daily_sd = daily_sd.merge(daily_n, on='end_date', how='left')

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # panel 1: rolling SD over time
    ax1.scatter(
        daily_sd['end_date'], daily_sd['cross_pollster_sd'],
        color='#aaaaaa', alpha=0.4, s=15,
        label='Daily cross-pollster SD', zorder=2
    )
    ax1.plot(
        daily_sd['end_date'], daily_sd['rolling_sd'],
        color='#d62728', linewidth=2.5,
        label='7-day rolling mean SD', zorder=3
    )

    # add trend line
    x_numeric = (
        daily_sd['end_date'] - daily_sd['end_date'].min()
    ).dt.days.values
    valid = daily_sd['rolling_sd'].notna()
    if valid.sum() > 2:
        z = np.polyfit(
            x_numeric[valid], daily_sd['rolling_sd'].values[valid], 1
        )
        p = np.poly1d(z)
        ax1.plot(
            daily_sd['end_date'], p(x_numeric),
            color='#1f77b4', linewidth=1.5, linestyle='--',
            label=f'Linear trend (slope={z[0]:.4f} pp/day)', zorder=4
        )

    ax1.set_ylabel('Cross-Pollster SD of Total Error (pp)', fontsize=11)
    ax1.set_title(
        'National Pollster Bias Dispersion Over Time\n'
        '(Does pollster spread narrow or widen as Election Day approaches?)',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        daily_sd['rolling_sd'].mean(), color='gray',
        linewidth=1, linestyle=':', label='Mean SD'
    )

    # panel 2: poll count per day for context
    ax2.bar(
        daily_n['end_date'], daily_n['n_polls'],
        color='#aaaaaa', alpha=0.6, width=1
    )
    ax2.set_ylabel('N Polls', fontsize=10)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(
        'figures/kalman_national_rolling_pollster_dispersion.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


########################################################################
# figure A2: national - rolling pollster-industry bias correlation
########################################################################
## NOT USED (only 11 fit criteria)
def create_figureA2_national_pollster_industry_correlation():
    """
    for each pollster with sufficient polls, computes the rolling
    correlation between that pollster's daily total error and the
    smoothed residual systematic bias (industry-wide trajectory).
    then plots the mean and distribution of these correlations over time.

    answers: do individual pollsters' errors start moving in sync with
    the industry average as the election approaches? increasing mean
    correlation over time is consistent with herding — pollsters
    increasingly tracking the same biased consensus rather than making
    independent methodological choices.

    note: this is distinct from dispersion (figure a1). high correlation
    with low dispersion means pollsters are close together and moving
    together (strong herding). high correlation with high dispersion means
    pollsters are spread apart but all drifting in the same direction
    simultaneously.
    """
    election_date = pd.Timestamp('2024-11-05')
    cutoff = election_date - pd.Timedelta(days=107)
    window_days = 21  # rolling window for correlation: 3 weeks

    df = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[df['end_date'] >= cutoff].copy()
   

    # get industry residual systematic bias — one value per day
    # (same for all pollsters on a given day, take first)
    industry_bias = (
        df.groupby('end_date')['residual_systematic_bias']
        .first()
        .reset_index()
        .sort_values('end_date')
    )

    # identify pollsters with enough polls for meaningful correlation
    min_polls = 15
    pollster_counts = df['pollster'].value_counts()
    eligible_pollsters = pollster_counts[
        pollster_counts >= min_polls
    ].index.tolist()

    print(
        f"Computing rolling correlations for "
        f"{len(eligible_pollsters)} pollsters "
        f"(min {min_polls} polls each)"
    )

    # for each eligible pollster, compute daily mean error
    # then rolling correlation with industry bias
    all_rolling_corrs = []

    for pollster in eligible_pollsters:
        pdf = (
            df[df['pollster'] == pollster]
            .groupby('end_date')['total_error']
            .mean()
            .reset_index()
            .rename(columns={'total_error': 'pollster_error'})
        )

        # merge with industry bias on date
        merged = industry_bias.merge(pdf, on='end_date', how='inner')
        merged = merged.sort_values('end_date').set_index('end_date')

        # rolling correlation over window_days
        if len(merged) < window_days:
            continue

        rolling_corr = (
            merged['pollster_error']
            .rolling(window=window_days, min_periods=10)
            .corr(merged['residual_systematic_bias'])
            .reset_index()
        )
        rolling_corr['pollster'] = pollster
        all_rolling_corrs.append(rolling_corr)

    corr_df = pd.concat(all_rolling_corrs, ignore_index=True)
    corr_df.columns = ['end_date', 'rolling_corr', 'pollster']
    corr_df = corr_df.dropna(subset=['rolling_corr'])

    # compute daily mean and percentile bands across pollsters
    daily_stats = (
        corr_df.groupby('end_date')['rolling_corr']
        .agg(['mean', 'median',
              lambda x: x.quantile(0.25),
              lambda x: x.quantile(0.75)])
        .reset_index()
    )
    daily_stats.columns = [
        'end_date', 'mean', 'median', 'p25', 'p75'
    ]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 2]}
    )

    # panel 1: mean rolling correlation with IQR band
    ax1.fill_between(
        daily_stats['end_date'],
        daily_stats['p25'],
        daily_stats['p75'],
        color='#1f77b4', alpha=0.2,
        label='IQR across pollsters'
    )
    ax1.plot(
        daily_stats['end_date'], daily_stats['mean'],
        color='#1f77b4', linewidth=2.5,
        label='Mean rolling correlation', zorder=3
    )
    ax1.plot(
        daily_stats['end_date'], daily_stats['median'],
        color='#ff7f0e', linewidth=1.5, linestyle='--',
        label='Median rolling correlation', zorder=3
    )
    ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax1.axhline(
        0.5, color='gray', linewidth=0.8, linestyle=':',
        alpha=0.5
    )
    ax1.set_ylabel(
        f'{window_days}-Day Rolling Correlation\n'
        'Pollster Error vs Industry Bias',
        fontsize=11
    )
    ax1.set_title(
        'Do Individual Pollsters Track the Industry Bias Trajectory?\n'
        '(Increasing correlation over time = consistent with herding)',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1.05, 1.05])

    # panel 2: individual pollster traces (faint) to show heterogeneity
    for pollster in eligible_pollsters:
        pdata = corr_df[corr_df['pollster'] == pollster]
        ax2.plot(
            pdata['end_date'], pdata['rolling_corr'],
            color='#aaaaaa', linewidth=0.8, alpha=0.3, zorder=1
        )

    ax2.plot(
        daily_stats['end_date'], daily_stats['mean'],
        color='#1f77b4', linewidth=2.5,
        label='Mean across pollsters', zorder=3
    )
    ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax2.set_ylabel(
        'Individual Pollster\nRolling Correlations',
        fontsize=10
    )
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1.05, 1.05])

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(
        'figures/kalman_national_pollster_industry_correlation.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


########################################################################
# main execution
########################################################################

if __name__ == '__main__':
    # create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('output/kalman').mkdir(parents=True, exist_ok=True)
    
    # generate all 6 figures
    create_figure2_national_bias_comparison() #USED
    create_figure3_swing_states_bias_trajectories() # USED
    create_figure4_variance_decomposition() # USED
    create_figure5_national_stability_temporal() #USED APPENDIX
    create_figure6_swing_states_bias_overlay() #USED
    create_figure7_swing_states_bias_correlation() # NOT USED
    
    # generate all 3 tables
    create_table1_national_summary() #USED APPENDIX
    create_table2_house_effects_variance() # PARTLY USED
    create_table3_swing_states_summary() #USED APPENDIX

    # appendix figures
    create_figureA1_national_rolling_dispersion()
    create_figureA2_national_pollster_industry_correlation() # NOT USED