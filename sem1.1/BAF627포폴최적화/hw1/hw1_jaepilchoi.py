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
# 20249433 MFE 최재필
#

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

# %%
msf = pd.read_csv('msf.csv')
msp500_rf = pd.read_csv('msp500_risk_free.csv')

# %%
msf['mdate'] = pd.to_datetime(msf['mdate'])

# %%
msp500_rf['mdate'] = pd.to_datetime(msp500_rf['mdate'])

# %%
msf_ret = msf[['mdate', 'ticker', 'ret']].pivot(index='mdate', columns='ticker', values='ret')
msf_ret.tail()

# %%
msp500_rf

# %%
sp500_ret = msp500_rf[['mdate', 'spret']].set_index('mdate')

# %%
rf_s = msp500_rf[['mdate', 'rf']].set_index('mdate')

# %%
sp500_excess_ret = sp500_ret['spret'] - rf_s['rf']
msf_excess_ret = msf_ret.subtract(rf_s['rf'], axis=0)

# %% [markdown]
# ## 1. No risk-free asset & Short-selling is allowed
#
# Case #2

# %%
cov_df = msf_ret.cov()
mean_s = msf_ret.mean()

# %%
std_s = msf_ret.std()

# %%
cov_2d = cov_df.values
mean_v = mean_s.values
std_v = std_s.values

# %%
sid_list = mean_s.index
sid_list

# %%
date_list = msf_ret.index
date_list


# %% [markdown]
# ### (a) 
#
# - Derive the mean-variance frontier using the standard deviation for measuring risk
# - Plot the mean-variance frontier
# - Indicate the global minimum portfolio (GMVP) on the plot
#
# Note: Raw returns are used to plot efficient frontier

# %%
def port_mean(W, mean_v, rf=0):
    """Get the mean of the portfolio

    Args:
        W (np.ndarray): 1*n array of weights
        mean_v (np.ndarray): 1*n array of mean returns

    Returns:
        float: weighted mean return of the portfolio. (1, ) scalar
    """
    return np.dot(W, mean_v - rf)


# %%
# Test the function

n = len(mean_v)
W = np.ones((1, n)) / n

# %%
port_mean(W, mean_v) 


# %%
def port_var(W, cov_2d):
    """Get the variance of the portfolio

    Args:
        W (np.ndarray): 1*n array of weights
        cov_2d (np.ndarray): n*n array of covariance matrix

    Returns:
        float: variance of the portfolio. (1, 1) array
    """    
    return np.dot(W, np.dot(cov_2d, W.T))


# %%
port_var(W, cov_2d)


# %%
def negative_port_sharpe(W, mean_v, cov_2d):
    """Get the Sharpe ratio of the portfolio

    Args:
        W (np.ndarray): 1*n array of weights
        mean_v (np.ndarray): 1*n array of mean returns
        cov_2d (np.ndarray): n*n array of covariance matrix

    Returns:
        float: Sharpe ratio of the portfolio. (1, 1) array
    """    

    mean_p = port_mean(W, mean_v)
    std_p = np.sqrt(port_var(W, cov_2d))

    if mean_p > 0:
        return -1 * mean_p / std_p # negative Sharpe ratio
    else:
        return mean_p / std_p # positive Sharpe ratio


# %%
negative_port_sharpe(W, mean_v, cov_2d)


# %%
def optimize_portfolio(mean_v, cov_2d):
    """Optimize the portfolio to get the maximum Sharpe ratio

    Args:
        mean_v (np.ndarray): 1*n array of mean returns
        cov_2d (np.ndarray): n*n array of covariance matrix
        rf (float): risk-free rate

    Returns:
        scipy.optimize.OptimizeResult: Result of the optimization
    """
    n = len(mean_v)
    args = (mean_v, cov_2d)
    constraints = {
        "type": "eq",
        "fun": lambda W: np.sum(W) - 1,
    }
    bounds = tuple((-1, 1) for asset in range(n))

    result = sco.minimize(
        negative_port_sharpe,  # Minimize the negative Sharpe ratio = maximize the Sharpe ratio
        n * [1.0 / n,],  # Initial guess
        args=args,  # asset returns, covariance matrix
        method="SLSQP",
        bounds=bounds,  # weights between -1 and 1
        constraints=constraints,  # weights sum to 1
    )

    return result


# %%
def optimize_portfolio_given_return(ret, mean_v, cov_2d):
    """Optimize the portfolio to get the maximum Sharpe ratio

    Args:
        mean_v (np.ndarray): 1*n array of mean returns
        cov_2d (np.ndarray): n*n array of covariance matrix
        rf (float): risk-free rate

    Returns:
        scipy.optimize.OptimizeResult: Result of the optimization
    """
    n = len(mean_v)
    args = (mean_v, cov_2d)
    constraints = [
        {"type": "eq", "fun": lambda W: np.sum(W) - 1,}, 
        {"type": "eq", "fun": lambda W: port_mean(W, mean_v) - ret,}
        ]
    bounds = tuple((-1, 1) for asset in range(n))

    result = sco.minimize(
        negative_port_sharpe,  # Minimize the negative Sharpe ratio = maximize the Sharpe ratio
        n * [1.0 / n,],  # Initial guess
        args=args,  # asset returns, covariance matrix
        method="SLSQP",
        bounds=bounds,  # weights between -1 and 1
        constraints=constraints,  # weights sum to 1, return target
    )

    return result


# %%
optimize_portfolio(mean_v, cov_2d) # optimized result의 x가 optimal weights


# %%
def get_opportunity_set(mean_v, cov_2d, num_portfolios=100):
    """Get the opportunity set by generating random portfolios

    Args:
        mean_v (np.ndarray): 1*n array of mean returns
        cov_2d (np.ndarray): n*n array of covariance matrix
        num_portfolios (int): number of random portfolios to generate

    Returns:
        np.ndarray: 3 * num_portfolios array of results // (portfolio return, portfolio std, portfolio sharpe ratio)
        list: list of weights
    """

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.uniform(-1, 1, len(mean_v))
        weights /= np.sum(np.abs(weights))
        # TODO: Weights don't sum to 1. If normalized, the resulting efficient frontier breaks.

        p_ret, p_std = port_mean(weights, mean_v), np.sqrt(port_var(weights, cov_2d))
        results[0, i] = p_ret
        results[1, i] = p_std
        results[2, i] = p_ret / p_std

        weights_record.append(weights)
    
    return results, weights_record
    


# %%
def get_efficient_frontier(mean_v, cov_2d, return_range=[-0.1, 0.1], num_portfolios=100):
    """Get the efficient frontier by optimizing the portfolio for each return given the range

    Args:
        mean_v (np.ndarray): 1*n array of mean returns
        cov_2d (np.ndarray): n*n array of covariance matrix
        return_range (list): range of return to optimize the portfolio
        num_portfolios (int): number of portfolios to generate

    Returns:
        np.ndarray: 3 * num_portfolios array of results // (portfolio return, portfolio std, portfolio sharpe ratio)
        list: list of weights
    """
    weights_record = []

    min_ret, max_ret = return_range
    ret_range = np.linspace(min_ret, max_ret, num_portfolios)
    results = np.zeros((3, len(ret_range)))

    for i, ret in enumerate(ret_range):
        result = optimize_portfolio_given_return(ret, mean_v, cov_2d)
        weights = result.x

        p_ret, p_std = port_mean(weights, mean_v), np.sqrt(port_var(weights, cov_2d))
        results[0, i] = p_ret
        results[1, i] = p_std
        results[2, i] = p_ret / p_std

        weights_record.append(weights)
        
    return results, weights_record
    


# %%
from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_optimization(ret, mean_v, cov_2d):
    """Helper function to perform optimization and return results"""
    result = optimize_portfolio_given_return(ret, mean_v, cov_2d)
    weights = result.x
    p_ret = port_mean(weights, mean_v)
    p_std = np.sqrt(port_var(weights, cov_2d))
    sharpe_ratio = p_ret / p_std
    return p_ret, p_std, sharpe_ratio, weights

def get_efficient_frontier_parallel(mean_v, cov_2d, return_range=[-0.1, 0.1], num_portfolios=100):
    """Get the efficient frontier by optimizing the portfolio for each return given the range

    Args:
        mean_v (np.ndarray): 1*n array of mean returns
        cov_2d (np.ndarray): n*n array of covariance matrix
        return_range (list): range of return to optimize the portfolio
        num_portfolios (int): number of portfolios to generate

    Returns:
        np.ndarray: 3 * num_portfolios array of results // (portfolio return, portfolio std, portfolio sharpe ratio)
        list: list of weights
    """
    min_ret, max_ret = return_range
    ret_range = np.linspace(min_ret, max_ret, num_portfolios)
    
    results_array = np.zeros((3, num_portfolios))
    weights_record = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(calculate_optimization, ret, mean_v, cov_2d): ret for ret in ret_range}
        for future in as_completed(futures):
            ret = futures[future]
            try:
                p_ret, p_std, sharpe_ratio, weights = future.result()
                idx = np.where(ret_range == ret)[0][0]
                results_array[0, idx] = p_ret
                results_array[1, idx] = p_std
                results_array[2, idx] = sharpe_ratio
                weights_record.append(weights)
            except Exception as exc:
                print(f'Return {ret} generated an exception: {exc}')
    
    return results_array, weights_record



# %% [markdown]
# We can plot the opportunity set but beware that it's NOT deriving the efficient frontier. 
#
# It's just randomly generating portfolios. 

# %%
results, weights_record = get_opportunity_set(mean_v, cov_2d, num_portfolios=1000)

# Optimal portfolio
max_sharpe_idx = np.argmax(results[2])
tangent_p_std, tangent_p_ret = results[1, max_sharpe_idx], results[0, max_sharpe_idx]

# Global minimum variance portfolio
min_vol_idx = np.argmin(results[1])
gmvp_std, gmvp_ret = results[1, min_vol_idx], results[0, min_vol_idx]



# Plot the efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='cool', marker='o')
plt.colorbar(label='Sharpe ratio')
plt.scatter(gmvp_std, gmvp_ret, marker='*', color='r', s=200, label='GMVP')
plt.title('Efficient Frontier with Short Selling Allowed')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)
plt.show()


# %%
results, weights_record = get_efficient_frontier_parallel(mean_v, cov_2d, num_portfolios=100)

# Global minimum variance portfolio
min_vol_idx = np.argmin(results[1])
gmvp_std, gmvp_ret = results[1, min_vol_idx], results[0, min_vol_idx]

# Plot the efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='cool', marker='o')
plt.colorbar(label='Sharpe ratio')
plt.scatter(gmvp_std, gmvp_ret, marker='*', color='r', s=200, label='GMVP')
plt.title('Efficient Frontier with Short Selling Allowed')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')
plt.legend(labelspacing=0.8)

# Annotate the GMVP point with its x and y values
plt.annotate(f'({gmvp_std:.2f}, {gmvp_ret:.2f})', 
             (gmvp_std, gmvp_ret), 
             textcoords="offset points", 
             xytext=(10,-10), 
             ha='center')

plt.show()


# %% [markdown]
# ### (b)

# %%
sp500_mean = sp500_ret.mean().values[0]

# %%
match_sp500_result = optimize_portfolio_given_return(sp500_mean, mean_v, cov_2d)
match_sp500_weights = match_sp500_result.x

match_sp500_weights_df = pd.DataFrame(match_sp500_weights, index=sid_list, columns=['weight'])
match_sp500_weights_df

# %%
match_sp500_weights_df.to_csv(OUTPUT_PATH / 'hw1_1.b_match_sp500_weights.csv')

# %%
watching = ['MMM', 'BAC', 'AMD', 'AAPL', 'MCD']

match_sp500_weights_df.loc[watching]

# %% [markdown]
# ### (c)

# %%
# Optimal portfolio matching sp500

max_sharpe_idx = np.argmax(results[2])
tangent_p_std, tangent_p_ret = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
tangent_p_weight = weights_record[max_sharpe_idx]

# %%
tangent_excess_ret = msf_excess_ret.multiply(tangent_p_weight, axis=1).sum(axis=1)

tangent_excess_mean = tangent_excess_ret.mean()
tangent_excess_std = tangent_excess_ret.std()


# %%
# Annualize return, std, sharpe ratio

def monthly_to_annual(data_v):
    monthly_mean = data_v.mean()
    monthly_std = data_v.std()

    annual_mean = monthly_mean * 12
    annual_std = monthly_std * np.sqrt(12)

    return annual_mean, annual_std


# %%
sp500_annual_excess_mean, sp500_annual_excess_std = monthly_to_annual(sp500_excess_ret)
sp500_annual_excess_mean, sp500_annual_excess_std

# %%
sp500_annual_excess_sharpe = sp500_annual_excess_mean / sp500_annual_excess_std
sp500_annual_excess_sharpe

# %%
tangent_p_mean_annual = tangent_excess_mean * 12
tangent_p_std_annual = tangent_excess_std * np.sqrt(12)

tangent_p_mean_annual, tangent_p_std_annual

# %%
tangent_annual_sharpe = tangent_p_mean_annual / tangent_p_std_annual
tangent_annual_sharpe

# %% [markdown]
# ### (d)

# %%
tangent_p_weight = weights_record[max_sharpe_idx]

tangent_p_ret = np.dot(tangent_p_weight, msf_ret.to_numpy().T)

# %%
tangent_p_cum_ret = np.cumprod(1 + tangent_p_ret) - 1

# %%
sp500_cum_ret = np.cumprod(1 + sp500_ret.to_numpy()) - 1

# %%
sns.lineplot(y=tangent_p_cum_ret[1:], x=date_list[1:], label='Optimal Portfolio Cum Return')
sns.lineplot(y=sp500_cum_ret, x=date_list[1:], label='S&P 500 Cum Return')


# %% [markdown]
# ## 2. There is risk-free asset & Short-selling is allowed
#
# Case #1
#
# Assume correlation between risk-free treasury and any asset is 0

# %% [markdown]
# ### (a)

# %%
rf = rf_s.mean().values[0]
rf

# %%
results.shape

# %%
tangent_point = (None, None) # std, ret
tangent_point_index = None

current_maximum_sharpe = -np.inf
for i in range(results.shape[1]): # From lowest to the highest return
    ret = results[0, i]
    std = results[1, i]
    
    current_sharpe = (ret - rf) / std
    if current_sharpe > current_maximum_sharpe:
        current_maximum_sharpe = current_sharpe
        tangent_point = (std, ret)
        tangent_point_index = i
    
    if current_sharpe < current_maximum_sharpe:
        break

# %%
tangent_sharpe = current_maximum_sharpe
tangent_sharpe # slope

# %%
tangent_point

# %%
tangent_weight = weights_record[tangent_point_index]

# %%

# Plot the efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='cool', marker='o')
plt.colorbar(label='Sharpe ratio')
plt.title('Efficient Frontier with Short Selling Allowed')
plt.xlabel('Risk (Std. Deviation)')
plt.ylabel('Return')

plt.scatter(0, rf, marker='o', color='r', s=200, label='risk-free')
plt.scatter(*tangent_point, marker='*', color='g', s=200, label='Tangent')

plt.axline((0, rf), slope=(tangent_point[1] - rf) / tangent_point[0], color='black', linestyle='--', linewidth=1.5)



# Set the x-axis to start from 0
plt.xlim(left=0)

plt.legend(labelspacing=0.8)
plt.show()


# %% [markdown]
# ### (b)

# %%
tangent_ret, tangent_std = tangent_point

# %%
sp500_mean 

# %%
sp500_matching_std = (sp500_mean  - rf) / tangent_sharpe
sp500_matching_std

# %%
risky_weight = (sp500_mean - rf) / (tangent_ret - rf)
risk_free_weight = 1 - risky_weight

risky_weight, risk_free_weight

# %%
tangent_weight_mixing_riskfree = tangent_weight * risky_weight

tangent_weight_mixing_riskfree_df = pd.DataFrame(tangent_weight_mixing_riskfree, index=sid_list, columns=['weight'])
tangent_weight_mixing_riskfree_df

# %%
tangent_weight_mixing_riskfree_df.loc[watching]

# %%
tangent_weight_mixing_riskfree_df.to_csv(OUTPUT_PATH / 'hw1_2.b_tangent_weight_mixing_riskfree.csv')

# %% [markdown]
# ### (c)

# %%
annual_tangent_excess_ret = (tangent_ret - rf) * 12
annual_tangent_std = tangent_std * np.sqrt(12)

annual_tangent_excess_ret, annual_tangent_std

# %%
annual_tangent_sharpe = annual_tangent_excess_ret / annual_tangent_std
annual_tangent_sharpe

# %% [markdown]
# ### (d)

# %%
