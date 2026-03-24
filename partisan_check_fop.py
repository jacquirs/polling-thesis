import pandas as pd
import sys

########################################
# log file
########################################

log_file = open('output/partisan_check_fop.txt', 'w')
sys.stdout = log_file

########################################
# load and clean data
########################################

df = pd.read_csv("data/president_2024_general.csv")
df['partisan'] = df['partisan'].replace('', None)
df['start_date'] = pd.to_datetime(df['start_date'])

########################################
# helper function
########################################

def run_partisan_analysis(data, label):
    poll_level = data[['poll_id', 'pollster', 'partisan', 'sponsors']].drop_duplicates()

    pollster_partisan_analysis = (
        poll_level.groupby('pollster')
        .agg({
            'poll_id': 'count',
            'partisan': lambda x: x.notna().sum()
        })
        .rename(columns={'poll_id': 'total_polls', 'partisan': 'polls_with_partisan_flag'})
        .reset_index()
    )

    pollster_partisan_analysis['all_polls_flagged'] = (
        pollster_partisan_analysis['total_polls'] == pollster_partisan_analysis['polls_with_partisan_flag']
    )

    pollster_partisan_analysis['no_polls_flagged'] = (
        pollster_partisan_analysis['polls_with_partisan_flag'] == 0
    )

    pollster_partisan_analysis = pollster_partisan_analysis.sort_values('total_polls', ascending=False)

    print(f"\n{'=' * 60}")
    print(f"partisan flag consistency by pollster -- {label}")
    print(f"{'=' * 60}")

    total_pollsters = len(pollster_partisan_analysis)
    always_flagged = pollster_partisan_analysis['all_polls_flagged'].sum()
    never_flagged = pollster_partisan_analysis['no_polls_flagged'].sum()
    sometimes_flagged = total_pollsters - always_flagged - never_flagged

    print(f"\ntotal pollsters: {total_pollsters}")
    print(f"always flagged (all polls have partisan value): {always_flagged}")
    print(f"never flagged (no polls have partisan value): {never_flagged}")
    print(f"sometimes flagged (inconsistent): {sometimes_flagged}")

    if sometimes_flagged > 0:
        print("\npollsters with inconsistent partisan flagging:")
        inconsistent = pollster_partisan_analysis[
            ~pollster_partisan_analysis['all_polls_flagged'] &
            ~pollster_partisan_analysis['no_polls_flagged']
        ]
        print(f"\n{len(inconsistent)} pollsters have some polls flagged and some not:\n")
        print(inconsistent.to_string(index=False))

    always_flagged_df = pollster_partisan_analysis[
        pollster_partisan_analysis['all_polls_flagged'] &
        (pollster_partisan_analysis['total_polls'] > 0)
    ]

    if len(always_flagged_df) > 0:
        print("\npollsters always flagged as partisan:")

        always_flagged_with_values = always_flagged_df.merge(
            poll_level[['pollster', 'partisan']].drop_duplicates(),
            on='pollster',
            how='left'
        )

        print("\npollster summary:")
        print(always_flagged_with_values[
            ['pollster', 'total_polls', 'partisan']
        ].to_string(index=False))

        sponsor_breakdown = (
            poll_level[poll_level['pollster'].isin(always_flagged_df['pollster'])]
            .groupby(['pollster', 'sponsors'])
            .size()
            .reset_index(name='num_polls')
            .sort_values(['pollster', 'num_polls'], ascending=[True, False])
        )

        print("\nsponsor breakdown:")
        print(sponsor_breakdown.to_string(index=False))


########################################
# full sample
########################################

run_partisan_analysis(df, "full sample")

########################################
# post-july 21 2024 (start_date >= 2024-07-21)
########################################

df_post = df[df['start_date'] >= '2024-07-21']
run_partisan_analysis(df_post, "start_date >= 2024-07-21")

########################################
# close log
########################################

log_file.close()
sys.stdout = sys.__stdout__