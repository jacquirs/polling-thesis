import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats


# THIS FILE ANALYZES POLL ACCURACY AND MODE EFFECTS FOR HARRIS+TRUMP POLLS USING THE SIMPLE MODES OUTPUT IN fiftyplusone_datelimited_analysis
# IT IS LIMITED TO THE TIME AFTER BIDEN DROPPED OUT
# using Martin, Traugott & Kennedy (2005) Method A accuracy measure
# then runs multivariate ols regressions (state and national separately)
# to understand what poll design factors predict accuracy


# redirect all print output to a log file
log_file = open('output/fiftyplusone_datelimited_simple_mode_analysis_log.txt', 'w')
sys.stdout = log_file

########################################################################################
#################### LOAD DATASETS #####################################################
########################################################################################

# load the three-way mode dataset
df_threeway = pd.read_csv('data/harris_trump_datelimited_simple_mode_analysis_threeway.csv')

# load the pure binary dataset
df_pure = pd.read_csv('data/harris_trump_datelimited_simple_mode_analysis_pure.csv')

# election date for time window analysis
election_date = pd.Timestamp('2024-11-05')

# convert dates
for df in [df_threeway, df_pure]:
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['start_date'] = pd.to_datetime(df['start_date'])

# swing states
swing_states = ['arizona', 'georgia', 'michigan', 'nevada', 'north carolina', 'pennsylvania', 'wisconsin']