import pandas as pd
import numpy as np

# THIS FILE CLEANS THE COMPARISON SET WITH TRUE OUTCOME 2024 PRESIDENTIAL RESULTS FOR USE IN MENG CALCULATIONS

df = pd.read_csv("2024-president-state.csv")

# sort for 2024 presidential results
df_pres_2024 = df[
    (df["year"] == 2024) &
    (df["office"] == "US PRESIDENT")
    # removed unofficial = false due to five states being unofficial: 'Montana', 'South Dakota', 'Kentucky', 'Washington', 'Arizona'
].copy()

# votes that didn't get counted but appear in the list
drop_categories = [
    "UNDERVOTES",
    "OVERVOTES",
    "VOID",
    "NONE OF THESE CANDIDATES"
]

# count for how much is getting removed
rows_before = len(df_pres_2024)
votes_before = df_pres_2024["votes"].sum()

# drop rows of uncounted votes
df_pres_2024_clean = df_pres_2024[
    ~df_pres_2024["candidate"].isin(drop_categories)
].copy()

# check how much was removed
rows_after = len(df_pres_2024_clean)
votes_after = df_pres_2024_clean["votes"].sum()
#print("Rows dropped:", rows_before - rows_after)
#print("Votes dropped:", votes_before - votes_after)

# state names to match CCES
df_pres_2024_clean["state_name"] = df_pres_2024_clean["state"].str.title()

# full data for later use
df_pres_2024_clean.to_csv("presidential_vote_results.csv", index=False)

# aggregate total votes per state
state_totals = (
    df_pres_2024_clean
    .groupby("state_name", as_index=False)["votes"]
    .sum()
    .rename(columns={"votes": "N_state"})
)

# aggregate Trump votes per state
trump_votes = (
    df_pres_2024_clean[df_pres_2024_clean["candidate"] == "TRUMP, DONALD J."]
    .groupby("state_name")["votes"]
    .sum()
)

# aggregate Harris votes per state
harris_votes = (
    df_pres_2024_clean[df_pres_2024_clean["candidate"] == "HARRIS, KAMALA D."]
    .groupby("state_name")["votes"]
    .sum()
)

state_totals["trump_votes"] = state_totals["state_name"].map(trump_votes)
state_totals["harris_votes"] = state_totals["state_name"].map(harris_votes)


# compute Trump share
state_totals["p_trump_true"] = (
    state_totals["trump_votes"] / state_totals["N_state"]
)

# compute harris share
state_totals["p_harris_true"] = (
    state_totals["harris_votes"] / state_totals["N_state"]
)


# keep only needed columns
for_meng = state_totals[
    ["state_name", "p_trump_true", "p_harris_true", "N_state"]
].copy()

for_meng.to_csv("Meng_true_votes.csv", index=False)