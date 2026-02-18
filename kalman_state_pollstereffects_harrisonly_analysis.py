import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys
from datetime import datetime
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# THIS FILE ANALYZES POLLS USING KALMAN FILTERING/SMOOTHING WITH POLLSTER-LEVEL HOUSE EFFECTS
# swing state level (AZ, GA, MI, NV, NC, PA, WI)
# harris v trump only
# includes pollster level effects

# same model as kalman_national_pollstereffects_harrisonly_analysis.py but run separately for each swing state

########################################################################################
##################################### Logging Setup ####################################
########################################################################################
