import pandas as pd

# data found at
# https://www.census.gov/data/tables/time-series/demo/voting-and-registration/p20-587.html
df_raw = pd.read_excel("data/census_turnout_originally_vote04a_2024.xlsx", header=4)
print(df_raw.head(3).to_string())

# handle col names
df = df_raw[["Unnamed: 0", "Total registered", "Total voted"]].copy()

# desired output cols
df.columns = ["state", "total_registered", "total_voted"]
df["state"] = df["state"].str.strip().str.lower()
df["state"] = df["state"].replace("united states", "national")

# drop footnotes
df = df.dropna(subset=["total_registered", "total_voted"])


print(df)
df.to_csv("data/turnout_by_state.csv", index=False)