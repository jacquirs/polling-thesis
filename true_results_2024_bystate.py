import pandas as pd
import numpy as np

# THIS FILE CLEANS THE COMPARISON SET WITH 2024 PRESIDENTIAL RESULTS

df = pd.read_csv("2024-president-state.csv")

# sort for 2024 presidential results
df_pres_2024 = df[
    (df["year"] == 2024) &
    (df["office"] == "US PRESIDENT") &
    (df["unofficial"] == False)
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
print("Rows dropped:", rows_before - rows_after)
print("Votes dropped:", votes_before - votes_after)

df_pres_2024_clean.to_csv("presidential_vote_results.csv", index=False)