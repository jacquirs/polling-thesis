import pandas as pd
import numpy as np

# THIS FILE AIMS TO CREATE A DATASET FOR MENG (2018) RECREATION BASED ON 2024 CCES DATA FOR VALIDATED VOTERS
# DATAFILE AVAILABLE FROM https://dataverse.harvard.edu/file.xhtml?fileId=13398263&version=2.0

# load CCES 2024 data
df = pd.read_csv("merged_recontact_2024_vv.csv")

# these columns are the only ones needed to recreate the analysis
neededcols = [
    "inputstate_24",
    "CC24_410",
    "TS_g2024",
    "TS_voterstatus_24",
    "commonweight_24"
]

df = df[neededcols].copy()

# rename variables for ease of use
df["state"] = df["inputstate_24"]

# create binary vote indicators for trump and harris from vote choices offered
df["X_trump"] = np.where(df["CC24_410"] == "Donald Trump", 1, 0)
df["X_harris"]   = np.where(df["CC24_410"] == "Kamala Harris", 1, 0)

# for those without a vote choice was not clear give an NA result
df.loc[df["CC24_410"].isin(["Not sure", "N"]), ["X_trump", "X_harris"]] = np.nan

# indicator of self repoted voting is if they said they voted for a condaidate, do not count those reporting "not sure" or "N" as having voted
df["selfvoted_2024"] = np.where(
    df["CC24_410"].isin([
        "Kamala Harris",
        "Donald Trump",
        "Robert F. Kennedy Jr.",
        "Jill Stein",
        "Cornel West",
        "Chase Oliver",
        "Other"
    ]),
    1, 0
)

# indicator for a validated voter
df["validated_voter"] = np.where(
    df["TS_g2024"].isin([1, 2, 3, 4, 5, 6]),
    1, 0
)

# -----------------------------
# 8. Final dataset
# -----------------------------
output_df = df[
    [
        "state",
        "X_trump",
        "X_harris",
        "selfvoted_2024",
        "validated_voter",
        "commonweight_24"
    ]
].copy()

output_df.to_csv("cces2024_meng_replication_set.csv", index=False)

print("Dataset created:", output_df.shape)
