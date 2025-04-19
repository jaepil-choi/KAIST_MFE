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
# # 경통분 hw5
#
# 20249433 최재필

# %%
import scipy.stats as stats

import numpy as np

# %% [markdown]
# # 1. 

# %% [markdown]
# skip

# %% [markdown]
# # 2. 

# %% [markdown]
# ### a
#
# skip

# %% [markdown]
# ### b

# %%
n = 35
x_bar = 0.235
s = 3.1

H0_mu = 0.25
alpha = 0.05

# %% [markdown]
# use t test

# %%
# Calculate the test statistic
t_statistic = (x_bar - H0_mu) / (s / (n ** 0.5))

# Calculate the p-value for a one-tailed test
p_value = stats.t.cdf(t_statistic, df=n-1)

# Print the results
t_statistic, p_value

# %%
reject_null = p_value < alpha
reject_null

# %% [markdown]
# # 3. 

# %% [markdown]
# ### a

# %%
H0_mu = 15
n = 70
x_bar = 14.6
s = 3.0
alpha = 0.025

# %%

# Degrees of freedom
df = n - 1

# Calculate the critical value for a one-tailed test
t_critical = stats.t.ppf(alpha, df)

# Calculate the test statistic
t_statistic = (x_bar - H0_mu) / (s / (n ** 0.5))

t_critical, t_statistic

# %%
reject_null = t_statistic < t_critical
reject_null

# %% [markdown]
# ### b

# %%
p_value = stats.t.cdf(t_statistic, df)

t_statistic, p_value

# %%
reject_null = p_value < alpha
reject_null

# %% [markdown]
# # 4. 

# %%
n1 = 47
x_bar1 = 7.92
s1 = 3.45

n2 = 38
x_bar2 = 5.80
s2 = 2.87

# %%
# Calculate the test statistic
t_statistic = (x_bar1 - x_bar2) / ((s1**2 / n1 + s2**2 / n2) ** 0.5)

# Calculate the degrees of freedom
df = ((s1**2 / n1 + s2**2 / n2) ** 2) / (((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1)))

# Calculate the p-value for a one-tailed test
p_value = 1 - stats.t.cdf(t_statistic, df)

t_statistic, df, p_value

# %%
reject_null = p_value < alpha
reject_null

# %% [markdown]
# alternatively, assume 2. equal variance 를 하고 
#
# sigma 1 = sigma 2 = population sigma 라고 놓고 푸는 방법도 있다. 
#
# (pulled variance 이용)

# %%
import scipy.stats as stats

# Given data
n1 = 47
x_bar1 = 7.92
s1 = 3.45

n2 = 38
x_bar2 = 5.80
s2 = 2.87

# Calculate the pooled variance
sp_squared = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
sp = sp_squared ** 0.5

# Calculate the test statistic
t_statistic = (x_bar1 - x_bar2) / (sp * ((1 / n1 + 1 / n2) ** 0.5))

# Calculate the degrees of freedom
df = n1 + n2 - 2

# Calculate the p-value for a one-tailed test
p_value = 1 - stats.t.cdf(t_statistic, df)

t_statistic, df, p_value


# %% [markdown]
# # 5. 

# %% [markdown]
# ### (a)

# %%
import numpy as np
import scipy.stats as stats

# Given data
data = {
    'female': [57, 84, 90, 71, 71, 77, 68, 73],
    'male': [71, 93, 101, 84, 88, 117, 86, 86, 93, 86, 106]
}

alpha = 0.05

# Calculate sample means and standard deviations
female_xbar = np.mean(data['female'])
male_xbar = np.mean(data['male'])

female_s = np.std(data['female'], ddof=1)
male_s = np.std(data['male'], ddof=1)

# Sample sizes
n_female = len(data['female'])
n_male = len(data['male'])

# Calculate the test statistic
t_statistic = (female_xbar - male_xbar) / np.sqrt((female_s**2 / n_female) + (male_s**2 / n_male))

# Calculate the degrees of freedom
df = ((female_s**2 / n_female) + (male_s**2 / n_male))**2 / (((female_s**2 / n_female)**2 / (n_female - 1)) + ((male_s**2 / n_male)**2 / (n_male - 1)))

# Calculate the p-value for a two-tailed test
p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))

# Print results
t_statistic, df, p_value


# %%
reject_null = p_value < alpha
reject_null

# %% [markdown]
# ### (b)

# %% [markdown]
# ### (c)

# %%
import numpy as np
import scipy.stats as stats

# Given data
data = {
    'female': [57, 84, 90, 71, 71, 77, 68, 73],
    'male': [71, 93, 101, 84, 88, 117, 86, 86, 93, 86, 106]
}

alpha = 0.05

# Calculate sample variances
female_s2 = np.var(data['female'], ddof=1)
male_s2 = np.var(data['male'], ddof=1)

# Sample sizes
n_female = len(data['female'])
n_male = len(data['male'])

# Calculate the F-statistic
F_statistic = male_s2 / female_s2

# Degrees of freedom
df1 = n_male - 1
df2 = n_female - 1

# Calculate the p-value for a two-tailed test
p_value = 2 * min(stats.f.cdf(F_statistic, df1, df2), 1 - stats.f.cdf(F_statistic, df1, df2))

# Print results
F_statistic, df1, df2, p_value


# %%
n_male, n_female

# %%
reject_null = p_value < alpha
reject_null

# %%
