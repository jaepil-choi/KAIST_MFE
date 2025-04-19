# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: sandbox311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3. Control Flows

# %%
import numpy as np

import seaborn as sns

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### If

# %% [markdown]
# Exercise 1
#
# ![image.png](attachment:image.png)

# %%
a = [6, 2, 3, 8]
b = [1, 2, 3, 4]

# %%

if a == sorted(a, reverse=False):
    print('It is sorted in ascending order')
else:
    print('It is not sorted in ascending order')

# %%

if b == sorted(b, reverse=False):
    print('It is sorted in ascending order')
else:
    print('It is not sorted in ascending order')

# %% [markdown]
# Exercise 2
#
# ![image.png](attachment:image.png)

# %%
import random

n = 10
x = [random.randint(1, 100) for i in range(n)]


# %%
x


# %%
def median(x):
    x = sorted(x, reverse=False)

    if n % 2 == 0:
        idx = int(n/2)
        return (x[idx] + x[idx + 1]) / 2
    else:
        return x[(n+1)/2]


# %%
median(x)


# %% [markdown]
# Exercise 7
#
# ![image.png](attachment:image.png)

# %%
def fibonacci_while(n):
    n1 = 1
    n2 = 1
    fibo = [n1, n2]

    while n1 + n2 <= n:
        fibo.append(n1 + n2)
        n1, n2 = n2, n1 + n2

    return fibo


# %%
fibos = fibonacci_while(100000)
len(fibos)

# %%
fibos

# %% [markdown]
# European Call Option 예시
#
# Monte Carlo simulation

# %%
import math
import random

# %%
S0 = 100. # initial stock price
K= 105. # strike price
T = 1. # time-to-maturity
r = 0.05 # riskless short rate
sigma = 0.2 # volatility
M = 50 # number of time steps
dt = T / M # length of time interval 이산화 하였을 때의 시간 간격
I = 250000 # number of paths


# %%
def generate_path():
    path = []

    for t in range(M+1):
        if t == 0:
            path.append(S0)
        else:
            z = random.gauss(0.0, 1.0)
            St = path[t-1] * math.exp((r-0.5*sigma**2)*dt + sigma * math.sqrt(dt)*z)
            path.append(St)
    
    return path


# %%
# %%time

sum_val = 0.0

S = [generate_path() for i in range(I)]

# %%
sum_val = 0.
for path in S:
    sum_val += max(path[-1] - K, 0)

C0 = math.exp(-r * T)*sum_val/I
C0

# %%
S = np.array(S)

# %%
s = S[:5][:]

sns.lineplot(s.T)

# %%
