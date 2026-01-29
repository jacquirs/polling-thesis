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
    "TS_voterstatus",
    "CC24_363"
]

df = df[neededcols].copy()

# rename variables for ease of use
df["state"] = df["inputstate"]

# map numeric CC24_410 codes to candidate names
cc24_410_map = {
    1: "Kamala Harris",
    2: "Donald Trump",
    3: "Robert F. Kennedy Jr.",
    4: "Jill Stein",
    5: "Cornel West",
    6: "Chase Oliver",
    7: "Other",
    8: "Did not vote for President"
}

df["CC24_410_name"] = df["CC24_410"].map(cc24_410_map)

# create binary vote indicators for trump and harris from vote choices offered
df["X_trump"] = np.where(df["CC24_410_name"] == "Donald Trump", 1, 0)
df["X_harris"]   = np.where(df["CC24_410_name"] == "Kamala Harris", 1, 0)

# for those without a vote choice was not clear give an NA result
df.loc[df["CC24_410_name"].isin(["Did not vote for President", "N","NA",np.nan]), ["X_trump", "X_harris"]] = np.nan

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
    df["CC24_410_name"].isna(),
    np.nan,
    np.where(df["CC24_410_name"].isin(candidate_list), 1, 0)
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

# state names are numbers in input dataset
state_map = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District Of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
}

df["state_name"] = df["state"].map(state_map)

# map numbers to answers for likely voters
cc24_363_map = {
    1: "Yes, definitely",
    2: "Probably",
    3: "I already voted (early or absentee)",
    4: "I plan to vote before November 5th",
    5: "No",
    6: "Undecided"
}

df["CC24_363_names"] = df["CC24_363"].map(cc24_363_map)

# identify likely voters based on response
likely_voter_categories = [
    "Yes, definitely",
    "Probably",
    "I already voted (early or absentee)",
    "I plan to vote before November 5th"
]

df["likely_voter"] = np.where(
    df["CC24_363_names"].isna(),
    np.nan,
    np.where(df["CC24_363_names"].isin(likely_voter_categories), 1, 0)
)


# create recreation dataset
output_df = df[
    [
        "caseid",
        "state_name",
        "X_trump",
        "X_harris",
        "selfvoted_2024",
        "likely_voter",
        "validated_voter"
    ]
].copy()

output_df.to_csv("cces2024_meng_replication_set.csv", index=False)

print(output_df.shape)
