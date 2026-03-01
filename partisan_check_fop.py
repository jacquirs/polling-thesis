import pandas as pd
import sys

# log file
log_file = open('output/partisan_check_fop.txt', 'w')
sys.stdout = log_file

# load the data
df = pd.read_csv("data/president_2024_general.csv")

# clean partisan column
df['partisan'] = df['partisan'].replace('', None)

# get one row per poll with its pollster and partisan value
poll_level = df[['poll_id', 'pollster', 'partisan', 'sponsors']].drop_duplicates()

# for each pollster, check if they have consistent partisan flagging
pollster_partisan_analysis = (
    poll_level.groupby('pollster')
    .agg({
        'poll_id': 'count',  # total polls
        'partisan': lambda x: x.notna().sum()  # polls with partisan flag
    })
    .rename(columns={'poll_id': 'total_polls', 'partisan': 'polls_with_partisan_flag'})
    .reset_index()
)

# add column showing if all polls are flagged
pollster_partisan_analysis['all_polls_flagged'] = (
    pollster_partisan_analysis['total_polls'] == pollster_partisan_analysis['polls_with_partisan_flag']
)

# add column showing if no polls are flagged
pollster_partisan_analysis['no_polls_flagged'] = (
    pollster_partisan_analysis['polls_with_partisan_flag'] == 0
)

# sort by total polls
pollster_partisan_analysis = pollster_partisan_analysis.sort_values('total_polls', ascending=False)


print("partisan flag consistency by pollster")

# summary statistics
total_pollsters = len(pollster_partisan_analysis)
always_flagged = pollster_partisan_analysis['all_polls_flagged'].sum()
never_flagged = pollster_partisan_analysis['no_polls_flagged'].sum()
sometimes_flagged = total_pollsters - always_flagged - never_flagged

print(f"\ntotal pollsters: {total_pollsters}")
print(f"always flagged (all polls have partisan value): {always_flagged}")
print(f"never flagged (no polls have partisan value): {never_flagged}")
print(f"sometimes flagged (inconsistent): {sometimes_flagged}")

# show pollsters with inconsistent flagging
if sometimes_flagged > 0:
    print("\npollsters with inconsistent partisan flagging:")
    
    inconsistent = pollster_partisan_analysis[
        ~pollster_partisan_analysis['all_polls_flagged'] & 
        ~pollster_partisan_analysis['no_polls_flagged']
    ]
    
    print(f"\n{len(inconsistent)} pollsters have some polls flagged and some not:\n")
    print(inconsistent.to_string(index=False))
    
# show always flagged pollsters
always_flagged_df = pollster_partisan_analysis[
    pollster_partisan_analysis['all_polls_flagged'] & 
    (pollster_partisan_analysis['total_polls'] > 0)
]

if len(always_flagged_df) > 0:
    print("\npollsters always flagged as partisan:")
    
    # get the actual partisan values for these pollsters
    always_flagged_with_values = always_flagged_df.merge(
        poll_level[['pollster', 'partisan']].drop_duplicates(),
        on='pollster',
        how='left'
    )
    
    print("\npollster summary:")
    print(always_flagged_with_values[
        ['pollster', 'total_polls', 'partisan']
    ].to_string(index=False))
    
    # get sponsor breakdown for these pollsters
    sponsor_breakdown = (
        poll_level[poll_level['pollster'].isin(always_flagged_df['pollster'])]
        .groupby(['pollster', 'sponsors'])
        .size()
        .reset_index(name='num_polls')
        .sort_values(['pollster', 'num_polls'], ascending=[True, False])
    )
    
    print("\nsponsor breakdown:")
    print(sponsor_breakdown.to_string(index=False))

# close log file and restore terminal output
log_file.close()
sys.stdout = sys.__stdout__