# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:31:55 2023

@author: Logan
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress

ts1 = pd.read_csv('C:\\Users\\ltmat\\Dropbox\\Fractical Analysis\\Nov23 Soybean Futures data hurstND.csv')
t = ts1['Date']
p = np.array(ts1.iloc[:,1:2])
#first degree diff
#p = np.array([p[i] - p[i-1] for i in range(1, len(p))])

window_size = 30


def rescaled_range(ts):
    """
    Calculate the rescaled range (R/S) of a time series.
    
    :param ts: Time series data as a list or numpy array.
    :return: The R/S value.
    """
    # Calculate the mean of the time series
    mean_ts = np.mean(ts)

    
    # Create a mean-adjusted series
    mean_adjusted = ts - mean_ts
    
    # Calculate the cumulative deviation series
    cum_dev = np.cumsum(mean_adjusted)
    
    # Calculate the range, R
    R = np.max(cum_dev) - np.min(cum_dev)
    
    # Calculate the standard deviation, S
    S = np.std(ts)
    
    # Return the R/S value
    return R / S if S != 0 else 0

def compute_RS(ts, n=30):
    """Calculate the R/S statistic for a given time series ts and window size n."""
    L = len(ts)
    S = np.std(ts)
    if S == 0:
        return 0  # Avoid zero division
    
    # Split time series into chunks of size n
    splits = [ts[i:i+n] for i in range(0, L, ) if len(ts[i:i+n]) == n]
    
    R_values = []
    for s in splits:
        mean_s = np.mean(s)
        mean_adjusted = s - mean_s
        cum_dev = np.cumsum(mean_adjusted)
        R = max(cum_dev) - min(cum_dev)
        R_values.append(R)
    
    RS = np.mean(R_values) / S
    #print(RS)
    return RS


def hurst_exponent(ts, max_n=30):
    """Calculate the Hurst exponent for a given time series ts."""
    log_ns = []
    log_RS = []
    for n in range(2, max_n+1):
        RS = compute_RS(ts, n)
        if RS > 0:
            log_ns.append(np.log(n))
            log_RS.append(np.log(RS))
    
    # Use linear regression to estimate Hurst exponent
    slope, _, _, _, _ = linregress(log_RS, log_ns)
    
    return slope


def rolling_hurst(ts, window_size=window_size, step=1):
    """Compute the Hurst exponent in a rolling window fashion."""
    num_points = len(ts)
    hurst_values = []
    
    for start in range(0, num_points - window_size + 1, step):
        hurst_values.append(hurst_exponent(ts[start:start + window_size]))
    print('Hurst Exp Computed')
    return hurst_values

def rolling_rs(ts, window_size=window_size, step=1):
    """Compute the Hurst exponent in a rolling window fashion."""
    num_points2 = len(ts)
    rs_values = []
    
    for start1 in range(0, num_points2 - window_size + 1, step):
        rs_values.append(compute_RS(ts[start1:start1 + window_size]))
    print('R/S Computed')
    return rs_values




roll = rolling_rs(p)
log = np.log
roll = pd.DataFrame(roll)

hurst = rolling_hurst(p)
hurst = pd.DataFrame(hurst)

print(np.mean(hurst))
print(np.std(hurst))

pd.DataFrame.to_csv(hurst, "C:\\Users\ltmat\Dropbox\Fractical Analysis\soybean-prices-historical-chart-data-eng.csv")





'''
rescaled_range(p)

x1 = 0
x2= 249
while x2 <= 13750:
    print(rescaled_range(p[x1:x2]))
    x1=x2
    x2+=250
   
print('--------------------------------')

print(hurst_exponent(p, max_n = 13768))


x1 = 0
x2= 249
while x2 <= 13750:
    print(hurst_exponent(p[x1:x2]))
    x1=x2
    x2+=250
'''