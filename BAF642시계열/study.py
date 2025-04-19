# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 금융시계열 수업 내용 스터디 
#
# - 필요한 구현은 노트 코드 쓰거나
# - GPT로 구현

# %%
from pykrx import stock

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


from tqdm import tqdm

from pathlib import Path
import os, sys

# %% [markdown]
# ## Lec 1: 시계열 분석 기초

# %% [markdown]
# ### random walk vs white noise

# %%
samsung = stock.get_market_ohlcv_by_date("20240101", "20241231", "005930")
samsung_adjclose = samsung['종가']

# %%
samsung_adjclose.plot() # random walk

# %%
rets = samsung_adjclose.pct_change()

ll = rets.mean() - 2 * rets.std() # lower limit
ul = rets.mean() + 2 * rets.std() # upper limit

rw = np.random.normal(loc=rets.mean(), scale=rets.std(), size=len(rets))
rw = pd.Series(rw, index=samsung_adjclose.index)

rw.plot()
rets.plot()

plt.legend(['White Noise', 'Samsung Returns'])

plt.axhline(ll, color='r', linestyle='--')
plt.axhline(ul, color='r', linestyle='--')

plt.show()

# %% [markdown]
# ### Time Series Decomposition
#
# - Trend
# - Seasonality
# - residual

# %%
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create synthetic data with periodic movement and arbitrary trend
np.random.seed(42)  # For reproducibility
n = 100  # Length of the time series
x = np.arange(n)

# Periodic function (e.g., sin) to mimic seasonality
seasonal_period = 12
seasonality = 10 * np.sin(2 * np.pi * x / seasonal_period)

# Arbitrary trend (non-linear, not predefined for simplicity)
trend = 0.5 * x + 0.02 * x**1.5

# Random noise
noise = np.random.normal(scale=5, size=n)

# Synthetic time series: Add trend, seasonality, and noise
y = trend + seasonality + noise

# Step 2: Detrend the data using a moving average filter
# Define moving average window based on seasonality period
window_size = seasonal_period

# Compute moving average for trend (centered moving average)
trend_estimate = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

# Align the moving average output with the original time series
# Since moving average shortens the series, pad with NaN
padding = (len(y) - len(trend_estimate)) // 2
trend_estimate_padded = np.pad(trend_estimate, (padding + 1, padding), mode="constant", constant_values=np.nan)

# Remove trend to get detrended series
detrended = y - trend_estimate_padded

# Step 3: Decompose seasonality using grouped averaging
# Drop NaN values in detrended series to ensure correct reshaping
valid_detrended = detrended[~np.isnan(detrended)]
adjusted_length = (len(valid_detrended) // seasonal_period) * seasonal_period
valid_detrended = valid_detrended[:adjusted_length]

# Reshape and compute seasonal component
detrended_matrix = valid_detrended.reshape(-1, seasonal_period)
seasonal_component = np.mean(detrended_matrix, axis=0)

# Extend seasonal component to full length
repeated_seasonality = np.tile(seasonal_component, len(y) // seasonal_period + 1)[:len(y)]

# Calculate residuals (deseasonalized data)
residuals = y - trend_estimate_padded - repeated_seasonality

# Step 4: Plot results
plt.figure(figsize=(12, 8))

# Original data
plt.plot(x, y, label="Original Time Series", color="blue")

# Trend component
plt.plot(x, trend_estimate_padded, label="Trend Component (Moving Average)", color="red", linestyle="--")

# Seasonal component
plt.plot(x, repeated_seasonality, label="Seasonal Component", color="green", linestyle="--")

# Residual component
plt.plot(x, residuals, label="Residual Component (Noise)", color="orange", linestyle="--")

plt.legend()
plt.title("Time Series Decomposition: Trend, Seasonality, and Residuals")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()


# %% [markdown]
# ## Lec 2: ARIMA
#
#

# %%
# Re-import libraries after reset
import numpy as np
import matplotlib.pyplot as plt

# Example time series data
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
time_series = 5 * np.sin(2 * np.pi * x / 10) + np.random.normal(scale=0.5, size=n)  # Sinusoidal data with noise

# ACF function
def acf(series, max_lag):
    n = len(series)
    mean = np.mean(series)
    autocorr = []
    for lag in range(max_lag + 1):
        numerator = np.sum((series[:n - lag] - mean) * (series[lag:] - mean))
        denominator = np.sum((series - mean) ** 2)
        autocorr.append(numerator / denominator)
    return np.array(autocorr)

# PACF function (using a regression approach)
def pacf(series, max_lag):
    from numpy.linalg import inv
    pacf_vals = []
    for lag in range(1, max_lag + 1):
        y = series[lag:]
        X = np.column_stack([series[lag - i:-i] for i in range(1, lag + 1)])
        beta = inv(X.T @ X) @ X.T @ y  # Regression coefficients
        pacf_vals.append(beta[-1])    # Partial correlation for the last lag
    return np.array([1.0] + pacf_vals)  # Add 1.0 for lag 0

# Compute ACF and PACF
max_lag = 20
acf_vals = acf(time_series, max_lag)
pacf_vals = pacf(time_series, max_lag)

# Plot the ACF and PACF
lags = np.arange(max_lag + 1)

plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(1, 2, 1)
plt.bar(lags, acf_vals, width=0.3, color="blue", alpha=0.7, label="ACF")
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.axhline(-1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.title("ACF")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()

# PACF plot
plt.subplot(1, 2, 2)
plt.bar(lags, pacf_vals, width=0.3, color="green", alpha=0.7, label="PACF")
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.axhline(-1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.title("PACF")
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.legend()

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

# Example time series data
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
time_series = 5 * np.sin(2 * np.pi * x / 10) + np.random.normal(scale=0.5, size=n)

# Compute ACF and PACF
max_lag = 20
acf_vals = acf(time_series, nlags=max_lag, fft=True)  # FFT for fast computation
pacf_vals = pacf(time_series, nlags=max_lag, method='ols')  # 'ols' regression-based method

# Plot ACF and PACF
lags = np.arange(max_lag + 1)

plt.figure(figsize=(12, 6))

# ACF Plot
plt.subplot(1, 2, 1)
plt.bar(lags, acf_vals, width=0.3, color="blue", alpha=0.7, label="ACF")
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.axhline(-1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.title("ACF")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()

# PACF Plot
plt.subplot(1, 2, 2)
plt.bar(lags, pacf_vals, width=0.3, color="green", alpha=0.7, label="PACF")
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.axhline(-1.96 / np.sqrt(len(time_series)), color="red", linestyle="--", linewidth=0.8)
plt.title("PACF")
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.legend()

plt.tight_layout()
plt.show()


# %%
