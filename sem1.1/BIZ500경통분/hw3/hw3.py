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
# # HW3

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom, norm

import itertools

from math import comb

# %% [markdown]
# ## 2.

# %%
n = 400
p = 0.03

# %%
mu = n*p
sigma = np.sqrt(n*p*(1-p))

mu, sigma**2

# %% [markdown]
# ### (a)

# %%
p_alt = 0.05

n * p_alt

# %%
norm.sf(n*p_alt, mu, sigma)

# %% [markdown]
# ### (b)

# %%
binom.sf(n*p_alt, n, p)

# %% [markdown]
# ## 3.

# %%
mu = 10
sigma = 2


# %%
def to_z(x, mu, sigma):
    return (x - mu) / sigma


# %%
lower = to_z(6, mu, sigma)
upper = to_z(14, mu, sigma)

# %% [markdown]
# ### (a)

# %%
prob_between = norm.cdf(upper) - norm.cdf(lower)
prob_between

# %% [markdown]
# ### (b)

# %%
norm.ppf(0.95, loc=mu, scale=sigma)

# %% [markdown]
# ### (c)
#
# $ X_i \sim N(10, 2^2) , n=4 $
#
# So, 
#
# $ \bar{X} \sim N(10, \frac{2^2}{4}) $

# %%
n = 4
mean = 10
std = 2

x_bar_mean = mean
x_bar_std = std / np.sqrt(n)

# %%
norm.cdf(12, x_bar_mean, x_bar_std)

# %% [markdown]
# 원래 풀었던 대로 아래처럼 풀 수도 있음. 그러나 이럴 필요가 없음. 

# %%
# x1x2x3x4_mean = mean * 4
# x1x2x3x4_var = 4 * std**2
# x1x2x3x4_std = np.sqrt(x1x2x3x4_var)

# x1x2x3x4_mean, x1x2x3x4_std

# %%
# norm.cdf(48, x1x2x3x4_mean, x1x2x3x4_std)

# %% [markdown]
# ## 4. 

# %%
n = 160
p = 0.2

# %%
EX = n*p
Var = n*p*(1-p)
std = np.sqrt(Var)

EX, Var, std

# %% [markdown]
# approximately, X~N(32, 25.6)

# %%
z = (50 - EX)/std
z

# %%
norm.sf(z)

# %% [markdown]
# ## 5.

# %%
pool = [0, 1, 2]

choices = [(X1, X2) for X1 in pool for X2 in pool]
choices

# %%
prob_mapping = {
    0: 0.1,
    1: 0.2,
    2: 0.7,
}


# %%
def size2_prob(tup):
    x1, x2 = tup
    return prob_mapping[x1] * prob_mapping[x2]


# %%
probs_df = pd.DataFrame([(X1, X2, size2_prob((X1, X2))) for X1, X2 in choices], columns=['X1', 'X2', 'P(X1, X2)'])
probs_df

# %%
probs_df['max(X1,X2)'] = probs_df[['X1', 'X2']].max(axis=1)
probs_df['X1+X2'] = probs_df['X1'] + probs_df['X2']

# %%
probs_df

# %%
probs_df[['X1+X2', 'P(X1, X2)']].groupby('X1+X2').sum()

# %%
probs_df[['max(X1,X2)', 'P(X1, X2)']].groupby('max(X1,X2)').sum()

# %%
