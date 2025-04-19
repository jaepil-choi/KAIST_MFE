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
# GPT-4o가 푼 hw1, 참고용

# %%
import numpy as np
import QuantLib as ql

# Parameters given in the problem
S0 = 1195          # Initial stock price
K = 1200           # Strike price
T = 2.5            # Time to maturity (years)
r = 0.01           # Risk-free rate
sigma = 0.25       # Volatility
B = 1300           # Barrier level (adjust as needed)
barrierType = ql.Barrier.UpOut  # Adjust according to the question's part A
n_simulations = 10000           # Number of simulations (replications, as mentioned in point E)
m = int(T * 365)   # Discretization steps (one step per day)
dt = T / m         # Time step

# Monte Carlo Simulation for Barrier Option
np.random.seed(42)  # For reproducibility
discount_factor = np.exp(-r * T)
prices = []

for _ in range(n_simulations):
    path = [S0]
    hit_barrier = False
    for _ in range(m):
        z = np.random.normal(0, 1)
        S_t = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        path.append(S_t)
        # Check barrier condition
        if (barrierType == ql.Barrier.UpOut and S_t >= B):
            hit_barrier = True
            break
    
    # Payoff calculation only if barrier not breached
    if not hit_barrier:
        prices.append(max(K - path[-1], 0))  # Put option payoff

# Average price with discount factor
price_mc = discount_factor * np.mean(prices)
print(f"Barrier Option Price: {price_mc:.2f}")


# %%
