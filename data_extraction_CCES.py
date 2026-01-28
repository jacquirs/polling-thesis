import pandas as pd
import numpy as np

# THIS FILE AIMS TO CREATE A DATASET FOR MENG (2018) RECREATION BASED ON 2024 CCES DATA FOR VALIDATED VOTERS
# DATAFILE AVAILABLE FROM https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/X11EP6

# load CCES 2024 data from site
df = pd.read_csv("CCES24_Common_OUTPUT_vv_topost_final.csv")

# these columns are the only ones needed to recreate the analysis
neededcols = [
    "caseid",
    "inputstate",
    "CC24_410",
    "TS_g2024",
    "TS_voterstatus"
]

df = df[neededcols].copy()

# rename variables for ease of use
df["state"] = df["inputstate"]

# create binary vote indicators for trump and harris from vote choices offered
df["X_trump"] = np.where(df["CC24_410"] == "Donald Trump", 1, 0)
df["X_harris"]   = np.where(df["CC24_410"] == "Kamala Harris", 1, 0)

# for those without a vote choice was not clear give an NA result
df.loc[df["CC24_410"].isin(["Did not vote for President", "N"]), ["X_trump", "X_harris"]] = np.nan

# indicator of self repoted voting is if they said they voted for a candidate, do not count those reporting "Did not vote for President" or "N" as having voted
candidate_list = [
    "Kamala Harris",
    "Donald Trump",
    "Robert F. Kennedy Jr.",
    "Jill Stein",
    "Cornel West",
    "Chase Oliver",
    "Other"
]

df["selfvoted_2024"] = np.where(
    df["CC24_410"].isna(),
    np.nan,
    np.where(df["CC24_410"].isin(candidate_list), 1, 0)
)

# handle voter registration match (only active or empty, so we don't know about those people)
df["active_reg_24"] = np.where(
    df["TS_voterstatus"] == 1,
    1,
    np.nan
)

# indicator for a validated voter
df["validated_voter"] = np.where(
    df["TS_g2024"].isin([1, 2, 3, 4, 5, 6]),
    1, 0
)

# create recreation dataset
output_df = df[
    [
        "caseid",
        "state",
        "X_trump",
        "X_harris",
        "selfvoted_2024",
        "validated_voter"
    ]
].copy()

output_df.to_csv("cces2024_meng_replication_set.csv", index=False)

print(output_df.shape)