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
# # 경통분 hw 2
#
# 20249433 최재필

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom, norm

import itertools

from math import comb

# %% [markdown]
# ## 1. 

# %%
p = 0.6
n = 5

# %% [markdown]
# ### (a)

# %%
probs = [binom.pmf(k, n, p) for k in range(n+1)]
probs

# %%
sns.barplot(data=np.array(probs))

# %% [markdown]
# ### (b)

# %%
mean = sum(np.array(probs) * np.array(range(n+1)))
mean

# %%
mean = n * p
mean

# %%
variance = sum(np.array(probs) * (np.array(range(n+1)) - mean)**2)
variance

# %%
variance = n * p * (1-p)
variance


# %%
# Define a function to simulate a single Bernoulli trial
def bernoulli_trial(p):
    """Returns 1 with probability p and 0 with probability 1-p."""
    return np.random.rand() < p

# Simulate n Bernoulli trials and compute the sample variance
def simulate_binomial_variance(n, p, num_simulations=10000):
    variances = []
    for _ in range(num_simulations):
        trials = [bernoulli_trial(p) for _ in range(n)]
        variances.append(np.var(trials, ddof=0))  # Population variance
    return np.mean(variances)

# Parameters for the binomial distribution
n = 10  # number of trials
p = 0.5  # probability of success

# Compute q, the probability of failure
q = 1 - p

# Simulate to compute the average variance from the simulations
simulated_variance = simulate_binomial_variance(n, p)

simulated_variance


# %%
n * p * q

# %% [markdown]
# ## 2. 

# %%
mu = 100
sigma = 5


# %%
def prob_smaller_than_b(b, mu, sigma):
    z = (b - mu) / sigma
    
    return norm.cdf(z)



# %%
def prob_bigger_than_b(b, mu, sigma):
    z = (b - mu) / sigma
    
    return 1 - norm.cdf(z)


# %%
def find_b_given_prob(prob, mu, sigma):
    z = norm.ppf(prob)
    
    return mu + z * sigma


# %%
def find_brange_given_prob(prob, mu, sigma):
    cumprob = 0.5 + prob / 2
    z = norm.ppf(cumprob)

    return (mu - z * sigma, mu + z * sigma)
    


# %% [markdown]
# ### (a)

# %%
find_b_given_prob(0.67, mu, sigma)

# %%
prob_smaller_than_b(0.44, 0, 1)

# %%
0.44*5+100

# %% [markdown]
# ### (b)

# %%
find_b_given_prob(0.011, mu, sigma)

# %%
100 + (100 - find_b_given_prob(0.011, mu, sigma))

# %%
find_b_given_prob((1 - 0.011), mu, sigma)

# %% [markdown]
# ### (c)

# %%
a, b = find_brange_given_prob(0.966, mu, sigma)
a, b

# %%
(b - a) / 2

# %% [markdown]
# ### (d)

# %%
prob_smaller_than_b(110, mu, sigma)

# %% [markdown]
# ### (e)

# %%
prob_bigger_than_b(95, mu, sigma)

# %% [markdown]
# ## 3. 

# %%
mu = 302
sigma = 2

# %% [markdown]
# ### (a)

# %%
p = prob_smaller_than_b(299, mu, sigma)
p

# %%
prob_smaller_than_b(-1.5, 0, 1) 

# %% [markdown]
# ### (b)

# %%
p ** 5

# %% [markdown]
# ## 4. 

# %%
x = np.array([0, 1, 2, 3])
probs = np.array([0.4, 0.3, 0.1, 0.2])

# %% [markdown]
# ### (a)

# %%
expected_value = sum(x * probs)
expected_value

# %%
std = np.sqrt(sum(probs * (x - expected_value)**2))
std

# %% [markdown]
# ### (b)

# %%
n = 100

mean_of_samplemean = expected_value
var_of_samplemean = std**2 / n

std_of_samplemean = np.sqrt(var_of_samplemean)
std_of_samplemean

# %%
prob_bigger_than_b(2, mean_of_samplemean, std_of_samplemean)

# %% [markdown]
# ### (c)

# %%
comb(5, 2) * 0.3**2 * 0.7**3

# %% [markdown]
# ### (d)

# %%
n = 100
p = 0.4

# %%
n * p

# %%
n * p * (1 - p)

# %% [markdown]
# ### (e)

# %%
mu = 40
sigma = np.sqrt(24)

# %%
z = (30 - mu) / sigma
z

# %%
prob_smaller_than_b(30, mu, sigma)

# %%
norm.cdf(z)

# %%
binom.cdf(30, 100, 0.4)

# %% [markdown]
# ## 5. 

# %% [markdown]
# ### (a)

# %%
0.7**3 * 0.3

# %% [markdown]
# ### (b)

# %%
comb(5, 1) * 0.3 * 0.7**4

# %% [markdown]
# ## 6. 

# %% [markdown]
# ### (b)

# %%
0.54 + 0.27 - 0.14

# %% [markdown]
# ## 7. 

# %%
