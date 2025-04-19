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
# # BAF627 HW1
#
# 20249433 최재필

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import scipy.optimize as sco

from tqdm import tqdm

# %%
np.random.seed(42)

# %%
CWD = Path.cwd()
OUTPUT_PATH = CWD / 'output'

# %% [markdown]
# ## 0. Import Data

# %% [markdown]
# ### Load data & convert

# %%
msf = pd.read_csv('msf.csv')
msp500_rf = pd.read_csv('msp500_risk_free.csv')

# %%
# datetime index
msf['mdate'] = pd.to_datetime(msf['mdate'])
msp500_rf['mdate'] = pd.to_datetime(msp500_rf['mdate'])

# %%
msf_ret_df = msf[['mdate', 'ticker', 'ret']].pivot(index='mdate', columns='ticker', values='ret')
sp500_ret_s = msp500_rf[['mdate', 'spret']].set_index('mdate')
rf_s = msp500_rf[['mdate', 'rf']].set_index('mdate')

sp500_excess_s = sp500_ret_s['spret'] - rf_s['rf']
msf_excess_df = msf_ret_df.subtract(rf_s['rf'], axis=0)

# %% [markdown]
# ### Basic matrix/vectors

# %%
# Pandas
cov_df = msf_ret_df.cov()
excess_cov_df = msf_excess_df.cov()
mean_ret_s = msf_ret_df.mean()
mean_excess_s = msf_excess_df.mean()
std_s = msf_ret_df.std()

# Numpy
cov_2d = cov_df.values
excess_cov_2d = excess_cov_df.values
mean_ret_v = mean_ret_s.values
mean_excess_v = mean_excess_s.values
std_v = std_s.values

sp500_ret_v = sp500_ret_s.values
sp500_excess_v = sp500_excess_s.values
rf_v = rf_s.values

# %%
sid_list = mean_ret_s.index
date_list = msf_ret_df.index

# %% [markdown]
# ### Basic scalar

# %%
mean_ret = mean_ret_v.mean()
mean_excess = mean_excess_v.mean()

sp500_ret = sp500_ret_v.mean()
sp500_std = sp500_ret_v.std()
sp500_excess = sp500_excess_v.mean()
sp500_excess_std = sp500_excess_v.std()

rf = rf_v.mean()


# %% [markdown]
# ## 1. No risk-free & Short-selling allowed
#
# Case #2

# %% [markdown]
# ### (a)
#
# - Derive the mean-variance frontier using the standard deviation for measuring risk
# - Plot the mean-variance frontier
# - Indicate the global minimum portfolio (GMVP) on the plot

# %%
def get_port_mean(W, mean_v, rf_v=None):
    if rf_v:
        return np.dot(W, mean_v - rf_v)
    else:
        return np.dot(W, mean_v)


# %%
def get_port_var(W, cov_2d):
    return np.dot(W.T, np.dot(cov_2d, W))


# %%
def negative_port_sharpe(W, mean_v, cov_2d, rf_v=None):
    port_mean = get_port_mean(W, mean_v, rf_v)
    port_var = get_port_var(W, cov_2d)
    port_std = np.sqrt(port_var)

    if port_mean > 0:
        return -port_mean / port_std
    else:
        return port_mean / port_std


# %%
def optimize_portfolio(mean_v, cov_2d):
    n = len(mean_v)
    args = (mean_v, cov_2d)
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W) - 1},
        ]
    # bounds = tuple((-1, 1) for i in range(n))

    result = sco.minimize(
        negative_port_sharpe,
        n * [1. / n,],
        args=args,
        method='SLSQP',
        # bounds=bounds,
        constraints=constraints,
    )

    return result


# %%
def optimize_portfolio_given_return(target_return, mean_v, cov_2d, rf_v=None):
    n = len(mean_v)
    args = (mean_v, cov_2d, rf_v)
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W) - 1},
        {'type': 'eq', 'fun': lambda W: target_return - get_port_mean(W, mean_v, rf_v)},
        ]
    # bounds = tuple((-1, 1) for i in range(n))

    result = sco.minimize(
        negative_port_sharpe,
        n * [1. / n,],
        args=args,
        method='SLSQP',
        # bounds=bounds,
        constraints=constraints,
    )

    return result


# %%
def get_efficient_frontier(mean_v, cov_2d, rf_v=None, return_minmax=[-0.1, 0.1], num_portfolios=100):
    weights_record = []

    min_return, max_return = return_minmax
    ret_range = np.linspace(min_return, max_return, num_portfolios)
    frontier_ports = np.zeros((3, len(ret_range))) # mean, std, sharpe

    for i, target_return in enumerate(tqdm(ret_range)):
        result = optimize_portfolio_given_return(target_return, mean_v, cov_2d, rf_v)
        weights = result.x

        port_mean = get_port_mean(weights, mean_v, rf_v)
        port_std = np.sqrt(get_port_var(weights, cov_2d))

        frontier_ports[0, i] = port_mean
        frontier_ports[1, i] = port_std
        frontier_ports[2, i] = port_mean / port_std

        weights_record.append(weights)

    return frontier_ports, weights_record


# %%
Q1_frontier_ports, Q1_weights_record = get_efficient_frontier(mean_ret_v, cov_2d, num_portfolios=100)

# %%
# Global minimum variance portfolio
min_vol_idx = np.argmin(Q1_frontier_ports[1])
gmvp_ret, gmvp_std, gmvp_sharpe = Q1_frontier_ports[:, min_vol_idx]

# Plot efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(Q1_frontier_ports[1, :], Q1_frontier_ports[0, :], c=Q1_frontier_ports[2, :], cmap='cool', marker='o')
plt.colorbar(label='Sharpe Ratio')

plt.scatter(gmvp_std, gmvp_ret, marker='*', color='g', s=100, label='GMVP')

plt.title('Efficient frontier (No riskfree, Yes short sale)')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)

plt.annotate(
    f'Return: {gmvp_ret:.2%}\nStd: {gmvp_std:.2%}\nSharpe: {gmvp_sharpe:.2f}',
    (gmvp_std, gmvp_ret),
    textcoords='offset points',
    xytext=(10, 10),
    ha='center'
)

plt.show()

# %% [markdown]
# ### (b)
#
# - Derive optimal portfolio weights that matches S&P500 BM return
# - Report portfolio weights in an excel file
# - Report portfolio weights on a given set of stocks

# %%
sp500_ret

# %%
Q1_match_sp500_result = optimize_portfolio_given_return(sp500_ret, mean_ret_v, cov_2d)
Q1_match_sp500_weights = Q1_match_sp500_result.x

Q1_match_sp500_weights_df = pd.DataFrame(Q1_match_sp500_weights, index=sid_list, columns=['weight'])
Q1_match_sp500_weights_df

# %%
Q1_match_sp500_weights_df.to_csv(OUTPUT_PATH / 'hw1_1.b_match_sp500_weights.csv')

# %%
WATCH_LIST = ['MMM', 'BAC', 'AMD', 'AAPL', 'MCD']

Q1_match_sp500_weights_df.loc[WATCH_LIST]

# %% [markdown]
# ### (c)
#
# - Compute the annualized excess returns, annualized volatility, and annualized Sharpe ratio of the optimal (matching) portfolio and S&P500 BM

# %%
sp500_excess_annual = sp500_excess * 12
sp500_excess_std_annual = sp500_excess_std * np.sqrt(12)
sp500_sharpe = sp500_excess_annual / sp500_excess_std_annual

sp500_excess_annual, sp500_excess_std_annual, sp500_sharpe

# %%
Q1_optimal_ret_s = msf_ret_df.multiply(Q1_match_sp500_weights, axis=1).sum(axis=1)
Q1_optimal_excess_s = Q1_optimal_ret_s - rf_s['rf']

Q1_optimal_excess = Q1_optimal_excess_s.mean()
Q1_optimal_excess_std = Q1_optimal_excess_s.std()

Q1_optimal_excess_annual = Q1_optimal_excess * 12
Q1_optimal_excess_std_annual = Q1_optimal_excess_std * np.sqrt(12)
Q1_optimal_sharpe = Q1_optimal_excess_annual / Q1_optimal_excess_std_annual

Q1_optimal_excess_annual, Q1_optimal_excess_std_annual, Q1_optimal_sharpe


# %% [markdown]
# ### (d)
#
# - Plot the cumulative return of optimal and S&P500
# - Discuss the difference in the performance of the two

# %%
def get_cumreturn(ret_s, cum_method='sum'):
    if cum_method == 'sum':
        return ret_s.cumsum()
    elif cum_method == 'prod':
        return (1 + ret_s).cumprod() - 1


# %%
sp500_cumret_s = get_cumreturn(sp500_ret_s)
Q1_optimal_cumret_s = get_cumreturn(Q1_optimal_ret_s)

# %%
# Plot S&P 500 cumulative return
plt.plot(sp500_cumret_s, label='S&P 500 BM')

# Plot optimal matching portfolio cumulative return
plt.plot(Q1_optimal_cumret_s, label='case#2')

# Add title and labels
plt.title('Q1. (No risk-free, Yes short sale)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (sum)')

# Add legend
plt.legend()

# Show plot
plt.show()

# %% [markdown]
# ## 2. Yes risk-free & Yes Short-selling
#
# Case #1

# %% [markdown]
# ### (a)
#
# - Derive the MV frontier
# - Report the slope of the MV frontier
# - Plot the MV frontier together with the MV frontier in Q1

# %%
Q2_tangent_point = (None, None) # std, ret
tangent_point_index = None
Q2_tangent_weight = None

current_maximum_sharpe = -np.inf
for i in range(Q1_frontier_ports.shape[1]): # From lowest to the highest return
    ret = Q1_frontier_ports[0, i]
    std = Q1_frontier_ports[1, i]
    
    current_sharpe = (ret - rf) / std
    if current_sharpe > current_maximum_sharpe:
        current_maximum_sharpe = current_sharpe
        Q2_tangent_point = (std, ret)
        Q2_tangent_weight = Q1_weights_record[i]
    
    if current_sharpe < current_maximum_sharpe:
        break

# %%
Q2_tangent_sharpe = current_maximum_sharpe
Q2_tangent_sharpe # Slope

# %%
# Plot efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(Q1_frontier_ports[1, :], Q1_frontier_ports[0, :], c='#D3D3D3', marker='o')
# plt.colorbar(label='Sharpe Ratio')

# Plot risk-free point
plt.scatter(0, rf, marker='o', color='r', s=200, label='risk-free')

# Plot tangent point
plt.scatter(*Q2_tangent_point, marker='*', color='g', s=100, label='Tangent')

# Draw the Capital Market Line (CML)
plt.axline(
    (0, rf), 
    slope=(Q2_tangent_point[1] - rf) / Q2_tangent_point[0], 
    color='black', 
    linestyle='--', 
    linewidth=1.5,
)

# Add title and labels
plt.title('Efficient frontier (Yes risk-free, Yes short sale)')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)
plt.xlim(left=0)

# Annotate risk-free point
plt.annotate(
    f'Return: {rf:.2%}',
    (0, rf),
    textcoords='offset points',
    xytext=(10, -20),  # Adjust position to avoid overlap
    ha='center'
)

# Annotate tangent point
plt.annotate(
    f'Return: {Q2_tangent_point[1]:.2%}\nStd: {Q2_tangent_point[0]:.2%}',
    Q2_tangent_point,
    textcoords='offset points',
    xytext=(10, 10),  # Adjust position to avoid overlap
    ha='center'
)

# Show plot
plt.show()


# %% [markdown]
# ### (b)
#
# - Derive optimal portfolio weights that matches S&P500 BM return
# - Report portfolio weights in an excel file
# - Report portfolio weights on a given set of stocks
# - Report the weight of a risk-free asset

# %%
Q2_tangent_std, Q2_tangent_ret = Q2_tangent_point # 순서 조심. 
Q2_tangent_std, Q2_tangent_ret

# %%
sp500_sharpe

# %%
Q2_match_sp500_std = (sp500_ret - rf) / Q2_tangent_sharpe
Q2_match_sp500_std

# %%
Q2_risky_weight = (sp500_ret - rf) / (Q2_tangent_ret - rf)
Q2_riskfree_weight = 1 - Q2_risky_weight

Q2_risky_weight, Q2_riskfree_weight

# %%
Q2_port_weights = Q2_risky_weight * Q2_tangent_weight

Q2_port_weights_df = pd.DataFrame(Q2_port_weights, index=sid_list, columns=['weight'])
Q2_port_weights_df

# %%
Q2_port_weights_df.to_csv(OUTPUT_PATH / 'hw1_2.b_port_weights.csv')

# %%
Q2_port_weights_df.loc[WATCH_LIST]

# %% [markdown]
# ### (c)
#
# - Compute the annualized excess returns, annualized volatility, and annualized Sharpe ratio of the optimal (matching) portfolio and S&P500 BM
# - Compare the result from Q1

# %%
Q2_optimal_ret_s = msf_ret_df.multiply(Q2_port_weights, axis=1).sum(axis=1)
Q2_optimal_ret = Q2_optimal_ret_s.mean()

Q2_riskfree_s = rf_s['rf'] * Q2_riskfree_weight

Q2_total_ret_s = Q2_optimal_ret_s + Q2_riskfree_s
Q2_total_excess_s = Q2_total_ret_s - rf_s['rf']

Q2_total_excess_std = Q2_total_excess_s.std()
Q2_total_excess_mean = Q2_total_excess_s.mean()

# %%
Q2_total_excess_mean_annual = Q2_total_excess_mean * 12
Q2_total_excess_std_annual = Q2_total_excess_std * np.sqrt(12)
Q2_total_sharpe = Q2_total_excess_mean_annual / Q2_total_excess_std_annual

Q2_total_excess_mean_annual, Q2_total_excess_std_annual, Q2_total_sharpe

# %%
# Results from Q1
Q1_optimal_excess_annual, Q1_optimal_excess_std_annual, Q1_optimal_sharpe

# %%
# S&P 500 Benchmark
sp500_excess_annual, sp500_excess_std_annual, sp500_sharpe

# %% [markdown]
# ### (d)
#
# - Plot the cumulative return of optimal and S&P500 (with previous results)
# - Discuss the difference in the performance of the three portfolios

# %%
sp500_cumret_s = get_cumreturn(sp500_ret_s)
Q2_optimal_cumret_s = get_cumreturn(Q2_total_ret_s)

# %%
# Plot S&P 500 cumulative return
plt.plot(sp500_cumret_s, label='S&P 500 BM')

# Plot optimal matching portfolio cumulative return
plt.plot(Q1_optimal_cumret_s, label='case#2')
plt.plot(Q2_optimal_cumret_s, label='case#1')

# Add title and labels
plt.title('Q2. (Yes risk-free, No short sale)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (sum)')

# Add legend
plt.legend()

# Show plot
plt.show()


# %% [markdown]
# ## 3. Yes risk-free & Short-selling not allowed
#
# Case #3

# %% [markdown]
# ### (a)
#
# - Derive the MV frontier
# - Report the slope of the MV frontier
# - Plot the MV frontier together with the MV frontier in Q1, Q2
# - Report which one has a higher slope, and what its meaning is

# %%
def optimize_portfolio_given_return_noshort(target_return, mean_v, cov_2d, rf_v=None):
    n = len(mean_v)
    args = (mean_v, cov_2d, rf_v)
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W) - 1},
        {'type': 'eq', 'fun': lambda W: target_return - get_port_mean(W, mean_v, rf_v)},
        ]
    bounds = tuple((0, 1) for i in range(n))

    result = sco.minimize(
        negative_port_sharpe,
        n * [1. / n,],
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    return result


# %%
def get_efficient_frontier_noshort(mean_v, cov_2d, rf_v=None, return_minmax=[-0.1, 0.1], num_portfolios=100):
    weights_record = []

    min_return, max_return = return_minmax
    ret_range = np.linspace(min_return, max_return, num_portfolios)
    frontier_ports = np.zeros((3, len(ret_range))) # mean, std, sharpe

    for i, target_return in enumerate(tqdm(ret_range)):
        result = optimize_portfolio_given_return_noshort(target_return, mean_v, cov_2d, rf_v)
        weights = result.x

        port_mean = get_port_mean(weights, mean_v, rf_v)
        port_std = np.sqrt(get_port_var(weights, cov_2d))

        frontier_ports[0, i] = port_mean
        frontier_ports[1, i] = port_std
        frontier_ports[2, i] = port_mean / port_std

        weights_record.append(weights)

    return frontier_ports, weights_record


# %%
Q3_frontier_ports, Q3_weights_record = get_efficient_frontier_noshort(mean_ret_v, cov_2d, num_portfolios=100)

# %%
Q3_tangent_point = (None, None) # std, ret
tangent_point_index = None
Q3_tangent_weight = None

current_maximum_sharpe = -np.inf
for i in range(Q3_frontier_ports.shape[1]): # From lowest to the highest return
    ret = Q3_frontier_ports[0, i]
    std = Q3_frontier_ports[1, i]
    
    current_sharpe = (ret - rf) / std

    if ret - rf < 0:
        continue

    if current_sharpe > current_maximum_sharpe:
        current_maximum_sharpe = current_sharpe
        Q3_tangent_point = (std, ret)
        Q3_tangent_weight = Q3_weights_record[i]


# %%
Q3_tangent_sharpe = current_maximum_sharpe
Q3_tangent_sharpe # Slope

# %%
# Plot efficient frontier # Q3
plt.figure(figsize=(10, 7))
plt.scatter(Q3_frontier_ports[1, :], Q3_frontier_ports[0, :], c='#D3D3D3', marker='o')

# Draw line connecting risk-free point to tangent point
plt.plot(
    [0, Q3_tangent_point[0]], 
    [rf, Q3_tangent_point[1]], 
    color='red', 
    linestyle='--', 
    linewidth=1.5,
)

# Plot efficient frontier # Q1
plt.scatter(Q1_frontier_ports[1, :], Q1_frontier_ports[0, :], color='#D3D3D3', marker='o')

# Draw the Capital Market Line (CML) # Q2
plt.plot(
    [0, Q2_tangent_point[0]], 
    [rf, Q2_tangent_point[1]], 
    color='gray', 
    linestyle='--', 
    linewidth=1.5,
)

# Plot tangent point
plt.scatter(*Q3_tangent_point, marker='*', color='g', s=100, label='Tangent')

# Plot risk-free point
plt.scatter(0, rf, marker='o', color='r', s=200, label='risk-free')

# Annotate risk-free point
plt.annotate(
    f'Return: {rf:.2%}',
    (0, rf),
    textcoords='offset points',
    xytext=(10, -20),  # Adjust position to avoid overlap
    ha='center'
)

# Annotate tangent point
plt.annotate(
    f'Return: {Q3_tangent_point[1]:.2%}\nStd: {Q3_tangent_point[0]:.2%}',
    Q3_tangent_point,
    textcoords='offset points',
    xytext=(10, 10),  # Adjust position to avoid overlap
    ha='center'
)

# Add title and labels
plt.title('Efficient frontier (Yes risk-free, No short sale)')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)
plt.xlim(left=0)

# Show plot
plt.show()


# %% [markdown]
# As seen above, slope is much inferior in Q3 (red linear line, green star is the tangent point) compared to Q1 (gray parabolic curve) and Q2 (gray linear line)
#
# This implys that yes-risk-free asset & no-short-sale constraint is making the efficient frontier worse, lowering the maximum utility for any investor. 

# %% [markdown]
# ### (b)
#
# - Derive optimal portfolio weights that matches S&P500 BM return
# - Report portfolio weights in an excel file
# - Report portfolio weights on a given set of stocks
# - Report the weight of a risk-free asset

# %%
Q3_tangent_std, Q3_tangent_ret = Q3_tangent_point # 순서 조심. 
Q3_tangent_std, Q3_tangent_ret

# %%
sp500_sharpe

# %%
Q3_match_sp500_std = (sp500_ret - rf) / Q3_tangent_sharpe
Q3_match_sp500_std

# %%
Q3_risky_weight = (sp500_ret - rf) / (Q3_tangent_ret - rf)
Q3_riskfree_weight = 1 - Q3_risky_weight

Q3_risky_weight, Q3_riskfree_weight

# %%
Q3_port_weights = Q3_risky_weight * Q3_tangent_weight

Q3_port_weights_df = pd.DataFrame(Q3_port_weights, index=sid_list, columns=['weight'])
Q3_port_weights_df

# %%
Q3_port_weights_df.to_csv(OUTPUT_PATH / 'hw1_3.b_port_weights.csv')

# %%
Q3_port_weights_df.loc[WATCH_LIST]

# %% [markdown]
# ### (c)
#
# - Compute the annualized excess returns, annualized volatility, and annualized Sharpe ratio of the optimal (matching) portfolio and S&P500 BM
# - Compare the result from Q1, Q2

# %%
Q3_optimal_ret_s = msf_ret_df.multiply(Q3_port_weights, axis=1).sum(axis=1)
Q3_optimal_ret = Q3_optimal_ret_s.mean()

Q3_riskfree_s = rf_s['rf'] * Q3_riskfree_weight

Q3_total_ret_s = Q3_optimal_ret_s + Q3_riskfree_s
Q3_total_excess_s = Q3_total_ret_s - rf_s['rf']

Q3_total_excess_std = Q3_total_excess_s.std()
Q3_total_excess_mean = Q3_total_excess_s.mean()

# %%
Q3_total_excess_mean_annual = Q3_total_excess_mean * 12
Q3_total_excess_std_annual = Q3_total_excess_std * np.sqrt(12)
Q3_total_sharpe = Q3_total_excess_mean_annual / Q3_total_excess_std_annual

Q3_total_excess_mean_annual, Q3_total_excess_std_annual, Q3_total_sharpe

# %%
Q2_total_excess_mean_annual, Q2_total_excess_std_annual, Q2_total_sharpe

# %%
# Results from Q1
Q1_optimal_excess_annual, Q1_optimal_excess_std_annual, Q1_optimal_sharpe

# %%
# S&P 500 Benchmark
sp500_excess_annual, sp500_excess_std_annual, sp500_sharpe

# %% [markdown]
# ### (d)
#
# - Plot the cumulative return of optimal and S&P500 (with previous results)
# - Discuss the difference in the performance of the four portfolios

# %%
Q3_optimal_cumret_s = get_cumreturn(Q3_total_ret_s)

# %%
# Plot S&P 500 cumulative return
plt.plot(sp500_cumret_s, label='S&P 500 BM')

# Plot optimal matching portfolio cumulative return
plt.plot(Q1_optimal_cumret_s, label='case#2')
plt.plot(Q2_optimal_cumret_s, label='case#1')
plt.plot(Q3_optimal_cumret_s, label='case#3')

# Add title and labels
plt.title('Q3. (Yes risk-free, No short sale)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (sum)')

# Add legend
plt.legend()

# Show plot
plt.show()

# %% [markdown]
# ## 4. No risk-free & Short-selling not allowed
#
# Case #4

# %% [markdown]
# ### (a)
#
# - Derive the MV frontier
# - Report the slop of the MV frontier
# - Plot the MV frontier together with the MV frontier in Q1, Q2, Q3
# - Explain the differences in the plotted frontiers (Check the efficient frontier's shift)

# %%
Q4_tangent_point = (None, None) # std, ret
tangent_point_index = None
Q4_tangent_weight = None

current_maximum_sharpe = -np.inf
for i in range(Q3_frontier_ports.shape[1]): # From lowest to the highest return
    ret = Q3_frontier_ports[0, i]
    std = Q3_frontier_ports[1, i]
    
    current_sharpe = (ret - rf) / std

    if ret - rf < 0:
        continue

    if current_sharpe > current_maximum_sharpe:
        current_maximum_sharpe = current_sharpe
        Q4_tangent_point = (std, ret)
        Q4_tangent_weight = Q3_weights_record[i]


# %%
Q4_tangent_sharpe = current_maximum_sharpe
Q4_tangent_sharpe # Slope

# %%
# Plot efficient frontier # Q3
plt.figure(figsize=(10, 7))
plt.scatter(Q3_frontier_ports[1, :], Q3_frontier_ports[0, :], c=Q3_frontier_ports[2, :], cmap='cool', marker='o')
plt.colorbar(label='Sharpe Ratio')

plt.plot(
    [0, Q3_tangent_point[0]], 
    [rf, Q3_tangent_point[1]], 
    color='gray', 
    linestyle='--', 
    linewidth=1.5,
)

# Plot efficient frontier # Q1
plt.scatter(Q1_frontier_ports[1, :], Q1_frontier_ports[0, :], color='#D3D3D3', marker='o')

# Draw the Capital Market Line (CML) # Q2
plt.axline(
    (0, rf), 
    slope=(Q2_tangent_point[1] - rf) / Q2_tangent_point[0], 
    color='gray', 
    linestyle='--', 
    linewidth=1.5,
)


# Plot tangent point
plt.scatter(*Q4_tangent_point, marker='*', color='g', s=100, label='Tangent')

# Plot risk-free point
plt.scatter(0, rf, marker='o', color='r', s=200, label='risk-free')

# Annotate risk-free point
plt.annotate(
    f'Return: {rf:.2%}',
    (0, rf),
    textcoords='offset points',
    xytext=(10, -20),  # Adjust position to avoid overlap
    ha='center'
)

# Annotate tangent point
plt.annotate(
    f'Return: {Q4_tangent_point[1]:.2%}\nStd: {Q4_tangent_point[0]:.2%}',
    Q4_tangent_point,
    textcoords='offset points',
    xytext=(10, 10),  # Adjust position to avoid overlap
    ha='center'
)

# Add title and labels
plt.title('Efficient frontier (Yes risk-free, No short sale)')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)
plt.xlim(left=0)

# Show plot
plt.show()


# %% [markdown]
# As seen above, slope is much inferior in Q4 (colorful parabolic curve) compared to Q1 (gray parabolic curve), Q2 (gray linear line) and Q3 (gray linear line that's flatter)
#
# This implys that no-risk-free asset & no-short-sale constraint is making the efficient frontier worse, lowering the maximum utility for any investor. 

# %% [markdown]
# ### (b)
#
# - Derive optimal portfolio weights that matches S&P500 BM return
# - Report portfolio weights in an excel file
# - Report portfolio weights on a given set of stocks

# %%
Q4_match_sp500_result = optimize_portfolio_given_return_noshort(sp500_ret, mean_ret_v, cov_2d)
Q4_match_sp500_weights = Q4_match_sp500_result.x

Q4_match_sp500_weights_df = pd.DataFrame(Q4_match_sp500_weights, index=sid_list, columns=['weight'])
Q4_match_sp500_weights_df

# %%
Q4_match_sp500_weights_df.to_csv(OUTPUT_PATH / 'hw1_4.b_match_sp500_weights.csv')

# %%
WATCH_LIST = ['MMM', 'BAC', 'AMD', 'AAPL', 'MCD']

Q4_match_sp500_weights_df.loc[WATCH_LIST]

# %% [markdown]
# ### (c)
#
# - Compute the annualized excess returns, annualized volatility, and annualized Sharpe ratio of the optimal (matching) portfolio and S&P500 BM
# - Compare the result from Q1, Q2, Q3

# %%
Q4_optimal_ret_s = msf_ret_df.multiply(Q4_match_sp500_weights, axis=1).sum(axis=1)
Q4_optimal_excess_s = Q4_optimal_ret_s - rf_s['rf']

Q4_optimal_excess = Q4_optimal_excess_s.mean()
Q4_optimal_excess_std = Q4_optimal_excess_s.std()

Q4_optimal_excess_annual = Q4_optimal_excess * 12
Q4_optimal_excess_std_annual = Q4_optimal_excess_std * np.sqrt(12)
Q4_optimal_sharpe = Q4_optimal_excess_annual / Q4_optimal_excess_std_annual

Q4_optimal_excess_annual, Q4_optimal_excess_std_annual, Q4_optimal_sharpe

# %%
Q3_total_excess_mean_annual, Q3_total_excess_std_annual, Q3_total_sharpe

# %%
Q2_total_excess_mean_annual, Q2_total_excess_std_annual, Q2_total_sharpe

# %%
# Results from Q1
Q1_optimal_excess_annual, Q1_optimal_excess_std_annual, Q1_optimal_sharpe

# %%
# S&P 500 Benchmark
sp500_excess_annual, sp500_excess_std_annual, sp500_sharpe

# %% [markdown]
# ### (d)
#
# - Plot the cumulative return of optimal and S&P500 (with previous results)
# - Discuss the difference in the performance of the five portfolios

# %%
Q4_optimal_cumret_s = get_cumreturn(Q4_optimal_ret_s)

# %%
# Plot S&P 500 cumulative return
plt.plot(sp500_cumret_s, label='S&P 500 BM')

# Plot optimal matching portfolio cumulative return
plt.plot(Q1_optimal_cumret_s, label='case#2')
plt.plot(Q2_optimal_cumret_s, label='case#1')
plt.plot(Q3_optimal_cumret_s, label='case#3')
plt.plot(Q4_optimal_cumret_s, label='case#4')

# Add title and labels
plt.title('Q4. (No risk-free, No short sale)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (sum)')

# Add legend
plt.legend()

# Show plot
plt.show()

# %% [markdown]
# ## 5. Portfolio performance and the number of stocks

# %% [markdown]
# ### (a) 
#
# - Randomly choose 10 stocks
# - Plot the MV frontier for case #2 (No risk-free, short-sale allowed)
# - Plot the MV frontier for case #1 (Yes risk-free, short-sale allowed)
# - Compare annualized Sharpe with case #2 and case #1

# %%
random_10_sid_list = np.random.choice(sid_list, 10, replace=False)
random_10_sid_list

# %%
random_msf_ret_df = msf_ret_df[random_10_sid_list].copy()
random_msf_excess_df = random_msf_ret_df.subtract(rf_s['rf'], axis=0)

# Pandas
random_cov_df = random_msf_ret_df.cov()
random_excess_cov_df = random_msf_excess_df.cov()
random_mean_ret_s = random_msf_ret_df.mean()
random_mean_excess_s = random_msf_excess_df.mean()
random_std_s = random_msf_ret_df.std()

# Numpy
random_cov_2d = random_cov_df.values
random_excess_cov_2d = random_excess_cov_df.values
random_mean_ret_v = random_mean_ret_s.values
random_mean_excess_v = random_mean_excess_s.values
random_std_v = random_std_s.values

random_sid_list = random_mean_ret_s.index

random_mean_ret = random_mean_ret_v.mean()
random_mean_excess = random_mean_excess_v.mean()

# %% [markdown]
# #### MV frontier for case #2 (No risk-free, Yes short-sale)

# %%
Q5a_frontier_ports, Q5a_weights_record = get_efficient_frontier(random_mean_ret_v, random_cov_2d, num_portfolios=100)

# %%
# Global minimum variance portfolio
min_vol_idx = np.argmin(Q5a_frontier_ports[1])
gmvp_ret, gmvp_std, gmvp_sharpe = Q5a_frontier_ports[:, min_vol_idx]

# Plot efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(Q5a_frontier_ports[1, :], Q5a_frontier_ports[0, :], c=Q5a_frontier_ports[2, :], cmap='cool', marker='o')
plt.colorbar(label='Sharpe Ratio')

plt.scatter(gmvp_std, gmvp_ret, marker='*', color='g', s=100, label='GMVP')

plt.title('Efficient frontier (No riskfree, Yes short sale)')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)

plt.annotate(
    f'Return: {gmvp_ret:.2%}\nStd: {gmvp_std:.2%}\nSharpe: {gmvp_sharpe:.2f}',
    (gmvp_std, gmvp_ret),
    textcoords='offset points',
    xytext=(10, 10),
    ha='center'
)

plt.show()

# %% [markdown]
# #### MV frontier for case #1 (Yes risk-free, Yes short-sale)

# %%
Q5a_tangent_point = (None, None) # std, ret
tangent_point_index = None
Q5a_tangent_weight = None

current_maximum_sharpe = -np.inf
for i in range(Q5a_frontier_ports.shape[1]): # From lowest to the highest return
    ret = Q5a_frontier_ports[0, i]
    std = Q5a_frontier_ports[1, i]
    
    current_sharpe = (ret - rf) / std
    if current_sharpe > current_maximum_sharpe:
        current_maximum_sharpe = current_sharpe
        Q5a_tangent_point = (std, ret)
        Q5a_tangent_weight = Q5a_weights_record[i]

# %%
Q5_tangent_sharpe = current_maximum_sharpe
Q5_tangent_sharpe # Slope

# %%
# Plot efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(Q5a_frontier_ports[1, :], Q5a_frontier_ports[0, :], c='#D3D3D3', marker='o')
# plt.colorbar(label='Sharpe Ratio')

# Plot risk-free point
plt.scatter(0, rf, marker='o', color='r', s=200, label='risk-free')

# Plot tangent point
plt.scatter(*Q5a_tangent_point, marker='*', color='g', s=100, label='Tangent')

# Draw the Capital Market Line (CML)
plt.axline(
    (0, rf), 
    slope=(Q5a_tangent_point[1] - rf) / Q5a_tangent_point[0], 
    color='black', 
    linestyle='--', 
    linewidth=1.5,
)

# Add title and labels
plt.title('10-asset portfolio ')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)
plt.xlim(left=0)

# Annotate risk-free point
plt.annotate(
    f'Return: {rf:.2%}',
    (0, rf),
    textcoords='offset points',
    xytext=(10, -20),  # Adjust position to avoid overlap
    ha='center'
)

# Annotate tangent point
plt.annotate(
    f'Return: {Q5a_tangent_point[1]:.2%}\nStd: {Q5a_tangent_point[0]:.2%}',
    Q5a_tangent_point,
    textcoords='offset points',
    xytext=(10, 10),  # Adjust position to avoid overlap
    ha='center'
)

# Show plot
plt.show()


# %% [markdown]
# #### Annualized Sharpe of case #2 (No risk-free, Yes short-sale)

# %%
Q5a_match_sp500_result = optimize_portfolio_given_return(sp500_ret, random_mean_ret_v, random_cov_2d)
Q5a_match_sp500_weights = Q5a_match_sp500_result.x

# %%
Q5a_optimal_ret_s = random_msf_ret_df.multiply(Q5a_match_sp500_weights, axis=1).sum(axis=1)
Q5a_optimal_excess_s = Q5a_optimal_ret_s - rf_s['rf']

Q5a_optimal_excess = Q5a_optimal_excess_s.mean()
Q5a_optimal_excess_std = Q5a_optimal_excess_s.std()

Q5a_optimal_excess_annual = Q5a_optimal_excess * 12
Q5a_optimal_excess_std_annual = Q5a_optimal_excess_std * np.sqrt(12)
Q5a_optimal_sharpe = Q5a_optimal_excess_annual / Q5a_optimal_excess_std_annual

Q5a_optimal_excess_annual, Q5a_optimal_excess_std_annual, Q5a_optimal_sharpe

# %% [markdown]
# #### Annualized Sharpe of case #1 (Yes risk-free, Yes short-sale)

# %%
Q5a_tangent_std, Q5a_tangent_ret = Q5a_tangent_point # 순서 조심. 
Q5a_tangent_std, Q5a_tangent_ret

# %%
Q5a_risky_weight = (sp500_ret - rf) / (Q5a_tangent_ret - rf)
Q5a_riskfree_weight = 1 - Q5a_risky_weight

Q5a_risky_weight, Q3_riskfree_weight

# %%
Q5a_port_weights = Q5a_risky_weight * Q5a_tangent_weight

# %%
Q5a_optimal_ret_s = random_msf_ret_df.multiply(Q5a_port_weights, axis=1).sum(axis=1)
Q5a_optimal_ret = Q5a_optimal_ret_s.mean()

Q5a_riskfree_s = rf_s['rf'] * Q5a_riskfree_weight

Q5a_total_ret_s = Q5a_optimal_ret_s + Q5a_riskfree_s
Q5a_total_excess_s = Q5a_total_ret_s - rf_s['rf']

Q5a_total_excess_std = Q5a_total_excess_s.std()
Q5a_total_excess_mean = Q5a_total_excess_s.mean()

# %%
Q3_total_excess_mean_annual = Q3_total_excess_mean * 12
Q3_total_excess_std_annual = Q3_total_excess_std * np.sqrt(12)
Q3_total_sharpe = Q3_total_excess_mean_annual / Q3_total_excess_std_annual

Q3_total_excess_mean_annual, Q3_total_excess_std_annual, Q3_total_sharpe

# %% [markdown]
# ### (b)
#
# - Construct money-sector portfolio C (`flag_sector=1`)
# - Construct diverse-industry portfolio D (`flag_sector=0`)
# - Plot MV frontiers of C, D for case #2 (No risk-free, short-sale allowed)
# - Plot MV frontiers of C, D for case #1 (Yes risk-free, short-sale allowed)
# - Report whose Sharpe is higher and explain why
#

# %%
msf_C = msf[msf['flag_sector'] == 1].copy()
msf_D = msf[msf['flag_sector'] == 0].copy()

# %%
len(msf_C['ticker'].unique())

# %%
len(msf_D['ticker'].unique())

# %%
msf_C_ret_df = msf_C[['mdate', 'ticker', 'ret']].pivot(index='mdate', columns='ticker', values='ret')
msf_D_ret_df = msf_D[['mdate', 'ticker', 'ret']].pivot(index='mdate', columns='ticker', values='ret')

# %%
C_msf_excess_df = msf_C_ret_df.subtract(rf_s['rf'], axis=0)

# Pandas
C_cov_df = msf_C_ret_df.cov()
C_excess_cov_df = C_msf_excess_df.cov()
C_mean_ret_s = msf_C_ret_df.mean()
C_mean_excess_s = C_msf_excess_df.mean()
C_std_s = msf_C_ret_df.std()

# Numpy
C_cov_2d = C_cov_df.values
C_excess_cov_2d = C_excess_cov_df.values
C_mean_ret_v = C_mean_ret_s.values
C_mean_excess_v = C_mean_excess_s.values
C_std_v = C_std_s.values

C_sid_list = C_mean_ret_s.index

C_mean_ret = C_mean_ret_v.mean()
C_mean_excess = C_mean_excess_v.mean()

# %%
D_msf_excess_df = msf_C_ret_df.subtract(rf_s['rf'], axis=0)

# Pandas
D_cov_df = msf_C_ret_df.cov()
D_excess_cov_df = D_msf_excess_df.cov()
D_mean_ret_s = msf_C_ret_df.mean()
D_mean_excess_s = D_msf_excess_df.mean()
D_std_s = msf_C_ret_df.std()

# Numpy
D_cov_2d = D_cov_df.values
D_excess_cov_2d = D_excess_cov_df.values
D_mean_ret_v = D_mean_ret_s.values
D_mean_excess_v = D_mean_excess_s.values
D_std_v = D_std_s.values

D_sid_list = D_mean_ret_s.index

D_mean_ret = D_mean_ret_v.mean()
D_mean_excess = D_mean_excess_v.mean()


# %% [markdown]
# #### C
#
# - Plot MV frontiers for case #2 (No risk-free, short-sale allowed)
# - Plot MV frontiers for case #1 (Yes risk-free, short-sale allowed)
#

# %%
Q5bC_frontier_ports, Q5bC_weights_record = get_efficient_frontier(C_mean_ret_v, C_cov_2d, num_portfolios=100)

# %%
Q5bC_tangent_point = (None, None) # std, ret
tangent_point_index = None
Q5bC_tangent_weight = None

current_maximum_sharpe = -np.inf
for i in range(Q5bC_frontier_ports.shape[1]): # From lowest to the highest return
    ret = Q5bC_frontier_ports[0, i]
    std = Q5bC_frontier_ports[1, i]
    
    current_sharpe = (ret - rf) / std
    if current_sharpe > current_maximum_sharpe:
        current_maximum_sharpe = current_sharpe
        Q5bC_tangent_point = (std, ret)
        Q5bC_tangent_weight = Q5bC_weights_record[i]

# %%
Q5bC_tangent_sharpe = current_maximum_sharpe
Q5bC_tangent_sharpe # Slope

# %%
# Plot efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(Q5bC_frontier_ports[1, :], Q5bC_frontier_ports[0, :], c=Q5bC_frontier_ports[2, :], cmap='cool', marker='o')
plt.colorbar(label='Sharpe Ratio')

plt.title('Efficient frontier (No riskfree, Yes short sale)')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)

plt.show()

# %%

# %% [markdown]
# #### D
#
# - Plot MV frontiers for case #2 (No risk-free, short-sale allowed)
# - Plot MV frontiers for case #1 (Yes risk-free, short-sale allowed)
#

# %%
Q5a_frontier_ports, Q5a_weights_record = get_efficient_frontier(random_mean_ret_v, random_cov_2d, num_portfolios=100)

# %%
# Plot efficient frontier
plt.figure(figsize=(10, 7))

plt.scatter(Q5bC_frontier_ports[1, :], Q5bC_frontier_ports[0, :], c=Q5bC_frontier_ports[2, :], cmap='cool', marker='o')
plt.scatter(Q5a_frontier_ports[1, :], Q5a_frontier_ports[0, :], c=Q5a_frontier_ports[2, :], cmap='gray', marker='o')

plt.title('Port C and D')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)

plt.show()

# %%

# %% [markdown]
#
