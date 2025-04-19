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
# # 경통분 hw4
#
# 20249433 최재필

# %%
from scipy.stats import norm
import numpy as np

import sympy as sp

# %% [markdown]
# ## 1. 

# %%
sigma = 3.5
alpha = 0.10

min_interval_length = 0.5 * 2

# %%
Z_a2 = norm.ppf(1 - alpha / 2)
Z_a2

# %%
n = sp.Symbol('n')

# %%
eq = sp.Eq(2 * Z_a2 * sigma / sp.sqrt(n), min_interval_length)

# %%
sol = sp.solve(eq, n)
n_sol = sol[0]
n_sol

# %% [markdown]
# ## 2. 

# %%
n = 43
x_bar = 8.12
s = 1.78

alpha = 0.10
Z_a2 = norm.ppf(1 - alpha / 2)

# %%
CI = (x_bar - Z_a2 * s / np.sqrt(n), x_bar + Z_a2 * s / np.sqrt(n))
CI

# %% [markdown]
# ## 3. 

# %%
x_bar = 630
s = 35
conf_level = 0.95

alpha = 1 - conf_level
Z_a2 = norm.ppf(1 - alpha / 2)

MoE = Z_a2 * s / np.sqrt(n)

# %%
Z_a2

# %%
MoE

# %%
CI = (x_bar - MoE, x_bar + MoE)
CI
