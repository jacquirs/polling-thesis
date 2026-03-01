import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy import stats

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