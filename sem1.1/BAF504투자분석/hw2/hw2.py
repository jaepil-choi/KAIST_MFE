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
# # 투자분석 hw2 
#
# 20249433 최재필
#

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

# %% [markdown] vscode={"languageId": "plaintext"}
# ## (a)
#
# - MSFT
# - GOOG
# - KO

# %% [markdown] vscode={"languageId": "plaintext"}
# ## (b)

# %%
df = pd.read_csv('wrds_stock_return.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['TICKER'].isin(['MSFT', 'GOOG', 'KO'])]
df # 5-year monthly data

# %% [markdown]
# - `RET`: return
# - `RETX`: return excluding dividend

# %%
ret_df = df.pivot(index='date', columns='TICKER', values='RET')
retx_df = df.pivot(index='date', columns='TICKER', values='RETX')

# %%
ret_df.head()

# %% [markdown]
# ### Calculate historical arithmetic average & SD for each stock

# %% [markdown]
# #### `RET`

# %%
ret_df.mean(axis=0)

# %%
ret_df.std(axis=0)

# %%
ret_df.cov()

# %%
ret_df.corr()

# %% [markdown]
# #### `RETX`

# %%
retx_df.mean(axis=0)

# %%
retx_df.std(axis=0)

# %%
retx_df.cov()

# %% [markdown]
# ## (c)

# %%

# %% [markdown]
# ## (d)

# %%

# %% [markdown]
# ## (e)

# %%

# %% [markdown]
# ## (f)

# %%
opt_port_mean = 0.0197
opt_port_std = 0.0517

# %%
alpha = 0.05

# %%
# Value at Risk
VaR = opt_port_mean + opt_port_std * stats.norm.ppf(alpha)
VaR

# %%
# Expected Shortfall
ES = opt_port_mean - opt_port_std * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
ES

# %%
stats.norm.pdf(stats.norm.ppf(alpha))

# %%
stats.norm.ppf(alpha)

# %%
stats.norm.pdf(-1.6448536269514729)

# %%
stats.norm.pdf(0)

# %%
stats.norm.cdf(-1.6448536269514729)

# %%
from scipy.stats import norm

# Given mean and standard deviation for the returns
mean = 0.0197
std_dev = 0.0517

# Set the confidence level for VaR and ES
confidence_level = 0.95

# Calculate the z-score for the given confidence level
z_score = norm.ppf(confidence_level)

# Calculate VaR (Value at Risk) for the left-tail risk
VaR = mean - z_score * std_dev

# For a normal distribution, the Expected Shortfall (ES) at a given confidence level can be computed using the formula:
# ES = mean - (pdf(z_score)/(1 - confidence_level)) * std_dev
# where pdf(z_score) is the probability density function of the normal distribution at the z_score
ES = mean - (norm.pdf(z_score) / (1 - confidence_level)) * std_dev

VaR, ES


# %%
z_score

# %%

# %% [markdown]
# ## (g)

# %% [markdown]
#
