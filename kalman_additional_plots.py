import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

########################################################################
# figure 2: national - residual systematic bias comparison (all four implementations)
########################################################################

def create_figure2_national_bias_comparison():
    """
    compare residual/systematic bias trajectories across all four implementations
    to test robustness of findings to methodological choices.
    """
    # load data from all four implementations (all_data window)
    em_anch = pd.read_csv('data/kalman_he_polls_anchored_all_data.csv')
    em_unanch = pd.read_csv('data/kalman_he_polls_unanchored_all_data.csv')
    agg_anch = pd.read_csv('data/kalman_agg_results_anchored_all_data.csv')
    agg_unanch = pd.read_csv('data/kalman_agg_results_unanchored_all_data.csv')
    
    # convert dates
    em_anch['end_date'] = pd.to_datetime(em_anch['end_date'])
    em_unanch['end_date'] = pd.to_datetime(em_unanch['end_date'])
    agg_anch['end_date'] = pd.to_datetime(agg_anch['end_date'])
    agg_unanch['end_date'] = pd.to_datetime(agg_unanch['end_date'])
    
    # create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # plot all four trajectories
    ax.plot(em_anch['end_date'], em_anch['residual_systematic_bias'],
            color='#d62728', linewidth=2.5, label='EM Anchored', zorder=4)
    ax.plot(em_unanch['end_date'], em_unanch['residual_systematic_bias'],
            color='#d62728', linewidth=2.5, linestyle='--', label='EM Unanchored', zorder=3)
    ax.plot(agg_anch['end_date'], agg_anch['systematic_bias'],
            color='#1f77b4', linewidth=2, label='Aggregated Anchored', zorder=2)
    ax.plot(agg_unanch['end_date'], agg_unanch['systematic_bias'],
            color='#1f77b4', linewidth=2, linestyle='--', label='Aggregated Unanchored', zorder=1)
    
    # reference line at zero
    ax.axhline(0, color='black', linewidth=1, linestyle=':')
    
    # formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Systematic Bias (pp, positive = polls overstated Trump)', fontsize=12)
    ax.set_title('National Polling Bias Across Four Implementations\n(Trump margin, 2021-2024)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('figures/national_bias_comparison_four_implementations.png', dpi=300, bbox_inches='tight')
    plt.close()


########################################################################
# figure 3: swing states - composite residual bias trajectories
########################################################################

def create_figure3_swing_states_bias_trajectories():
    """
    small multiples showing residual systematic bias over time for all 7 swing states
    (em anchored, all data) to reveal geographic variation in bias dynamics.
    """
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan', 
        'NV': 'Nevada', 'NC': 'North Carolina', 
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, state in enumerate(states):
        ax = axes[idx]
        
        # load state data
        try:
            df = pd.read_csv(f'data/kalman_he_polls_{state}_anchored_all_data.csv')
            df['end_date'] = pd.to_datetime(df['end_date'])
            
            # plot residual systematic bias
            ax.plot(df['end_date'], df['residual_systematic_bias'],
                   color='#d62728', linewidth=2)
            
            # fill regions
            ax.fill_between(df['end_date'], 0, df['residual_systematic_bias'],
                          where=df['residual_systematic_bias'] > 0,
                          color='#d62728', alpha=0.15, label='Pro-Trump bias')
            ax.fill_between(df['end_date'], 0, df['residual_systematic_bias'],
                          where=df['residual_systematic_bias'] <= 0,
                          color='#1f77b4', alpha=0.15, label='Pro-Harris bias')
            
            # reference line
            ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
            
            # state label and true margin
            true_margin = df['true_margin'].iloc[0]
            ax.set_title(f"{state_names[state]}\n(True: Trump{true_margin:+.1f})", 
                        fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.2)
            
            # format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'{state}\nData not found', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim([pd.Timestamp('2021-01-01'), pd.Timestamp('2024-11-05')])
    
    # remove empty subplot
    fig.delaxes(axes[7])
    
    # common labels
    fig.text(0.5, 0.02, 'Date', ha='center', fontsize=13)
    fig.text(0.02, 0.5, 'Residual Systematic Bias (pp)', va='center', rotation='vertical', fontsize=13)
    fig.suptitle('Swing State Polling Bias Trajectories (EM Anchored, All Data)\nPositive = polls overstated Trump relative to certified result',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.savefig('figures/swing_states_bias_trajectories_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()


########################################################################
# figure 4: swing states - forecast error + variance decomposition (two-panel)
########################################################################

def create_figure4_swing_states_error_decomposition():
    """
    left: final forecast error by state (unanchored). right: variance decomposition by state (anchored).
    shows both how much states missed and what drove the errors.
    """
    states = ['AZ', 'GA', 'MI', 'NV', 'NC', 'PA', 'WI']
    state_names = {
        'AZ': 'Arizona', 'GA': 'Georgia', 'MI': 'Michigan', 
        'NV': 'Nevada', 'NC': 'North Carolina', 
        'PA': 'Pennsylvania', 'WI': 'Wisconsin'
    }
    
    # collect data
    forecast_errors = {}
    var_decomp = {}
    
    for state in states:
        # unanchored for forecast error
        try:
            df_unanch = pd.read_csv(f'data/kalman_he_polls_{state}_unanchored_all_data.csv')
            final_smoothed = df_unanch['smoothed'].iloc[-1]
            true_margin = df_unanch['true_margin'].iloc[0]
            forecast_errors[state] = final_smoothed - true_margin
        except:
            forecast_errors[state] = np.nan
        
        # anchored for variance decomposition
        try:
            df_anch = pd.read_csv(f'data/kalman_he_polls_{state}_anchored_all_data.csv')
            var_total = df_anch['total_error'].var()
            var_he = df_anch['house_effect_assigned'].var()
            var_noise = df_anch['sampling_noise'].var()
            var_resid = df_anch['residual_systematic_bias'].var()
            
            var_decomp[state] = {
                'house': 100 * var_he / var_total,
                'noise': 100 * var_noise / var_total,
                'residual': 100 * var_resid / var_total
            }
        except:
            var_decomp[state] = {'house': np.nan, 'noise': np.nan, 'residual': np.nan}
    
    # create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # left panel: forecast error
    errors = [forecast_errors[s] for s in states]
    state_labels = [state_names[s] for s in states]
    colors = ['#d62728' if e > 0 else '#1f77b4' for e in errors]
    
    y_pos = np.arange(len(states))
    ax1.barh(y_pos, errors, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(state_labels)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_xlabel('Final Forecast Error (pp)\n(Positive = overstated Trump)', fontsize=11)
    ax1.set_title('State-Level Forecast Accuracy\n(EM Unanchored, All Data)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # right panel: variance decomposition
    house_pcts = [var_decomp[s]['house'] for s in states]
    noise_pcts = [var_decomp[s]['noise'] for s in states]
    resid_pcts = [var_decomp[s]['residual'] for s in states]
    
    # sort by residual systematic bias (descending)
    sorted_indices = np.argsort(resid_pcts)[::-1]
    states_sorted = [states[i] for i in sorted_indices]
    state_labels_sorted = [state_names[s] for s in states_sorted]
    house_sorted = [house_pcts[i] for i in sorted_indices]
    noise_sorted = [noise_pcts[i] for i in sorted_indices]
    resid_sorted = [resid_pcts[i] for i in sorted_indices]
    
    y_pos = np.arange(len(states_sorted))
    
    ax2.barh(y_pos, house_sorted, color='#ff7f0e', alpha=0.8, label='House Effects')
    ax2.barh(y_pos, noise_sorted, left=house_sorted, color='#2ca02c', alpha=0.8, label='Sampling Noise')
    ax2.barh(y_pos, resid_sorted, left=np.array(house_sorted) + np.array(noise_sorted), 
            color='#d62728', alpha=0.8, label='Residual Systematic Bias')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(state_labels_sorted)
    ax2.set_xlabel('Percent of Total Error Variance', fontsize=11)
    ax2.set_title('Error Source Decomposition\n(EM Anchored, All Data)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim([0, 100])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/swing_states_forecast_error_and_variance_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()


########################################################################
# figure 5: national - house effect stability & temporal dynamics (two-panel)
########################################################################

def create_figure5_national_stability_temporal():
    """
    left: house effect stability (all data vs last 107 days). right: mean absolute bias across time windows.
    tests whether pollster biases are stable and whether systematic bias accelerated near election day.
    """
    # load house effects from different time windows
    he_all = pd.read_csv('data/kalman_he_effects_anchored_all_data.csv')
    he_107 = pd.read_csv('data/kalman_he_effects_anchored_last_107_days.csv')
    
    # merge on pollster
    he_compare = he_all.merge(he_107, on='pollster', suffixes=('_all', '_107'), how='inner')
    he_compare = he_compare[he_compare['n_polls_all'] >= 5]  # min 5 polls in all data
    
    # load poll-level data for temporal dynamics
    polls_all = pd.read_csv('data/kalman_he_polls_anchored_all_data.csv')
    polls_200 = pd.read_csv('data/kalman_he_polls_anchored_last_200_days.csv')
    polls_107 = pd.read_csv('data/kalman_he_polls_anchored_last_107_days.csv')
    
    # create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # left panel: house effect stability
    ax1.scatter(he_compare['house_effect_all'], he_compare['house_effect_107'],
               s=he_compare['n_polls_all']*3, alpha=0.6, color='#1f77b4')
    
    lims = [he_compare[['house_effect_all', 'house_effect_107']].min().min() - 0.5,
            he_compare[['house_effect_all', 'house_effect_107']].max().max() + 0.5]
    ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect stability')
    
    # label outliers
    he_compare['distance'] = abs(he_compare['house_effect_all'] - he_compare['house_effect_107'])
    outliers = he_compare.nlargest(5, 'distance')
    for _, row in outliers.iterrows():
        ax1.annotate(row['pollster'], 
                    xy=(row['house_effect_all'], row['house_effect_107']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax1.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax1.set_xlabel('House Effect - All Data (pp)', fontsize=11)
    ax1.set_ylabel('House Effect - Last 107 Days (pp)', fontsize=11)
    ax1.set_title('House Effect Stability Across Time Windows\n(Point size = number of polls)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # right panel: temporal dynamics of systematic bias
    # calculate mean absolute residual systematic bias for each window
    windows = ['All Data\n(2021-2024)', 'Last 200 Days\n(~6.5 months)', 'Last 107 Days\n(~3.5 months)']
    mean_abs_bias = [
        polls_all['residual_systematic_bias'].abs().mean(),
        polls_200['residual_systematic_bias'].abs().mean(),
        polls_107['residual_systematic_bias'].abs().mean()
    ]
    
    # also show final bias (last observation in each window)
    final_bias = [
        polls_all['residual_systematic_bias'].iloc[-1],
        polls_200['residual_systematic_bias'].iloc[-1],
        polls_107['residual_systematic_bias'].iloc[-1]
    ]
    
    x = np.arange(len(windows))
    width = 0.35
    
    ax2.bar(x - width/2, mean_abs_bias, width, label='Mean |Residual Bias|', 
           color='#ff7f0e', alpha=0.8)
    ax2.bar(x + width/2, [abs(b) for b in final_bias], width, label='|Final Bias|', 
           color='#d62728', alpha=0.8)
    
    ax2.set_ylabel('Absolute Systematic Bias (pp)', fontsize=11)
    ax2.set_title('Systematic Bias Magnitude Across Time Windows\n(EM Anchored)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # add text annotations
    for i, (mean_val, final_val) in enumerate(zip(mean_abs_bias, final_bias)):
        ax2.text(i - width/2, mean_val + 0.1, f'{mean_val:.2f}', 
                ha='center', fontsize=9, fontweight='bold')
        ax2.text(i + width/2, abs(final_val) + 0.1, f'{abs(final_val):.2f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/national_house_effect_stability_and_temporal_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()


########################################################################
# national summary - all four implementations × three time windows
########################################################################

def create_table1_national_summary():
    """
    comprehensive comparison table: 4 implementations × 3 time windows.
    shows how all methodological choices affect core estimates.
    """
    implementations = ['EM_anchored', 'EM_unanchored', 'Agg_anchored', 'Agg_unanchored']
    windows = ['all_data', 'last_200_days', 'last_107_days']
    window_labels = ['All Data', 'Last 200 Days', 'Last 107 Days']
    
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
    he_all = pd.read_csv('data/kalman_he_effects_anchored_all_data.csv')
    he_all = he_all[he_all['n_polls'] >= 5]  # min 5 polls
    he_all = he_all.nlargest(20, 'house_effect', keep='all')  # top 20 by house effect (most pro-trump first)
    
    house_effects_table = he_all[['pollster', 'house_effect', 'n_polls']].copy()
    house_effects_table.columns = ['Pollster', 'House Effect (pp)', 'N Polls']
    house_effects_table = house_effects_table.round({'House Effect (pp)': 3})
    
    # part b: variance decomposition across time windows
    windows = ['all_data', 'last_200_days', 'last_107_days']
    window_labels = ['All Data', 'Last 200 Days', 'Last 107 Days']
    
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
            df_anch = pd.read_csv(f'data/kalman_he_polls_{state}_anchored_all_data.csv')
            
            # unanchored for forecast error
            df_unanch = pd.read_csv(f'data/kalman_he_polls_{state}_unanchored_all_data.csv')
            
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
                'Forecast Error (pp)': df_unanch['smoothed'].iloc[-1] - df_unanch['true_margin'].iloc[0]
            }
            results.append(row)
            
        except FileNotFoundError:
            continue
    
    table3 = pd.DataFrame(results)
    
    # add rank by forecast error magnitude
    table3['Rank'] = table3['Forecast Error (pp)'].abs().rank(ascending=True).astype(int)
    
    # sort by forecast error magnitude
    table3 = table3.sort_values('Forecast Error (pp)', key=abs, ascending=False)
    
    # format
    table3_formatted = table3.round({
        'True Margin (pp)': 2,
        'Mean Poll Margin (pp)': 2,
        'Mean Total Error (pp)': 2,
        'Mean |House Effect| (pp)': 2,
        'Mean Residual Bias (pp)': 2,
        '$\\sigma^2_u$ (per day)': 6,
        'var(HE)/var(Total) \\%': 1,
        'Forecast Error (pp)': 2
    })
    
    # save
    table3_formatted.to_csv('output/kalman/swing_states_polling_quality_and_forecast_performance.csv', index=False)
    
    latex_str = table3_formatted.to_latex(index=False, escape=False,
                                          caption='Swing State Summary: Polling Quality and Forecast Performance',
                                          label='tab:swing_states_summary')
    with open('output/kalman/swing_states_polling_quality_and_forecast_performance.tex', 'w') as f:
        f.write(latex_str)
    
    return table3_formatted


########################################################################
# main execution
########################################################################

if __name__ == '__main__':
    # create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('output/kalman').mkdir(parents=True, exist_ok=True)
    
    # generate all 4 figures
    create_figure2_national_bias_comparison()
    create_figure3_swing_states_bias_trajectories()
    create_figure4_swing_states_error_decomposition()
    create_figure5_national_stability_temporal()
    
    # generate all 3 tables
    create_table1_national_summary()
    create_table2_house_effects_variance()
    create_table3_swing_states_summary()