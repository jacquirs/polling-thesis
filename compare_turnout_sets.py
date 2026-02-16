import pandas as pd

########################################################################################
############################ Reclean without dropping votes ############################
########################################################################################
# load truth set from site
true_votes_preclean = pd.read_csv("data/2024-president-state.csv")

# sort for 2024 presidential results
df_pres_2024 = true_votes_preclean[
    (true_votes_preclean["year"] == 2024) &
    (true_votes_preclean["office"] == "US PRESIDENT")
].copy()

# did not drop  "UNDERVOTES", "OVERVOTES", "VOID", "NONE OF THESE CANDIDATES"

# state names to match CCES
df_pres_2024["state_name"] = df_pres_2024["state"].str.title()

# full data for later use
df_pres_2024.to_csv("presidential_vote_results.csv", index=False)

# aggregate total votes per state
state_totals = (
    df_pres_2024
    .groupby("state_name", as_index=False)["votes"]
    .sum()
    .rename(columns={"votes": "N_state"})
)

# pull total_votes from raw data (one row per state, should be constant)
state_total_votes = (
    df_pres_2024.groupby("state_name")["totalvotes"]
    .first()
    .reset_index()
    .rename(columns={"totalvotes": "total_votes"})
)
state_total_votes["state_name"] = state_total_votes["state_name"].str.strip().str.lower()

# aggregate Trump votes per state
trump_votes = (
    df_pres_2024[df_pres_2024["candidate"] == "TRUMP, DONALD J."]
    .groupby("state_name")["votes"]
    .sum()
)

# aggregate Harris votes per state
harris_votes = (
    df_pres_2024[df_pres_2024["candidate"] == "HARRIS, KAMALA D."]
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
true_votes = state_totals[
    ["state_name", "p_trump_true", "p_harris_true", "N_state"]
].copy()

# match case of census
true_votes["state_name"] = true_votes["state_name"].str.strip().str.lower()

########################################################################################
#################################### Compare to census #################################
########################################################################################

# load turnout data from census
census_turnout = pd.read_csv("data/turnout_by_state.csv")

# calculate national row for votes by summing all states
national_row = pd.DataFrame([{
    "state_name": "national",
    "N_state": true_votes["N_state"].sum()
}])
true_votes = pd.concat([true_votes, national_row], ignore_index=True)

# merge and compare
df = census_turnout.merge(true_votes[["state_name", "N_state"]], left_on="state", right_on="state_name", how="left")
df = df.drop(columns="state_name")

# merge in total_votes from raw data
df = df.merge(state_total_votes, left_on="state", right_on="state_name", how="left")
df = df.drop(columns="state_name")

df["diff_Nstate_census"] = df["N_state"] - df["total_voted"]
df["diff_census_pct"] = (df["diff_Nstate_census"] / df["total_voted"] * 100).round(2)

df["diff_Nstate_totalvotes"] = df["N_state"] - df["total_votes"]
df["diff_totalvotes_pct"] = (df["diff_Nstate_totalvotes"] / df["total_votes"] * 100).round(2)

print(df[["state", "total_voted", "total_votes", "N_state", "diff_Nstate_census", "diff_census_pct", "diff_Nstate_totalvotes", "diff_totalvotes_pct"]].to_string(index=False))