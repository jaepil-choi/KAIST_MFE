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
# # 투자분석 hw3
#
# SIM (Single Index Model)
#
# 20249433 최재필

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
import statsmodels.api as sm

# %%
import yfinance as yf

# %% [markdown]
# ## (a) 
#
# - Choose 4 stocks, each from different industries. 
# - Collect 60 monthly returns 
# - Collect T bill rates for the same period
# - Collect market index returns for the same period
# - Run regression model with this data
# - Report alpha/beta estimates

# %% [markdown]
# ### Collect data
#
# ![image.png](attachment:image.png)
#
# - Stock selection:
#     - Technology: AAPL (Apple)
#     - Financial: SPGI (S&P Global)
#     - Consumer Cyclical: MCD (McDonald's)
#     - Consumer Defensive: KO (Coca Cola's)
# - Risk-free rate selection:
#     - T-bill 3-month
# - Market Index Selection:
#     - S&P 500 Index: SPY

# %%
AAPL = yf.Ticker("AAPL")
SPGI = yf.Ticker("SPGI")
MCD = yf.Ticker("MCD")
KO = yf.Ticker("KO")

SPY = yf.Ticker("SPY")
Tbill3M = yf.Ticker("^IRX")

# %%
AAPL_df = AAPL.history(period="5y", interval='1mo')
AAPL_ret = AAPL_df['Close'].pct_change().dropna()
AAPL_ret = AAPL_ret.rename('AAPL')

SPGI_df = SPGI.history(period="5y", interval='1mo')
SPGI_ret = SPGI_df['Close'].pct_change().dropna()
SPGI_ret = SPGI_ret.rename('SPGI')

MCD_df = MCD.history(period="5y", interval='1mo')
MCD_ret = MCD_df['Close'].pct_change().dropna()
MCD_ret = MCD_ret.rename('MCD')

KO_df = KO.history(period="5y", interval='1mo')
KO_ret = KO_df['Close'].pct_change().dropna()
KO_ret = KO_ret.rename('KO')

SPY_df = SPY.history(period="5y", interval='1mo')
SPY_ret = SPY_df['Close'].pct_change().dropna()
SPY_ret = SPY_ret.rename('SPY')

Tbill3M_df = Tbill3M.history(period="5y", interval='1mo') # Annualized return
Tbill3M_ret = Tbill3M_df['Close']
Tbill3M_ret = Tbill3M_ret.rename('Tbill3M')

Tbill3M_ret = Tbill3M_ret / 100 / 12 # Monthly return

# %%
AAPL_ret.index = AAPL_ret.index.to_period('M')
SPGI_ret.index = SPGI_ret.index.to_period('M')
MCD_ret.index = MCD_ret.index.to_period('M')
KO_ret.index = KO_ret.index.to_period('M')
SPY_ret.index = SPY_ret.index.to_period('M')

Tbill3M_ret.index = Tbill3M_ret.index.to_period('M')

# %%
len(Tbill3M_df)

# %%
data_df = pd.concat([AAPL_ret, SPGI_ret, MCD_ret, KO_ret, SPY_ret, Tbill3M_ret], axis=1)
data_df.dropna(inplace=True)

data_df.tail()

# %%
data_df.info()

# %%
# Make it excess return
data_df.loc[:, ['AAPL', 'SPGI', 'MCD', 'KO', 'SPY']] = data_df.loc[:, ['AAPL', 'SPGI', 'MCD', 'KO', 'SPY']].subtract(data_df.loc[:, 'Tbill3M'], axis=0)

# %% [markdown]
# ### Run regression 

# %%
stocks = ['AAPL', 'SPGI', 'MCD', 'KO']

def get_SIM_regression(stock_returns, market_returns, print_summary=True):
    X = sm.add_constant(market_returns)
    model = sm.OLS(stock_returns, X)
    results = model.fit()

    if print_summary:
        print(results.summary())

    return results


# %%
results = {}

for stock in stocks:
    print(f'SIM regression result of {stock}')
    results[stock] = get_SIM_regression(data_df[stock], data_df['SPY'], print_summary=True)
    print('\n'*5)

# %% [markdown]
# ## (b)
#
# - Interpret alpha/beta estimates
# - Consider the smallest/largest betas among the four stocks
# - To which industries do the two companies belong? 
# - Is the business consistent with the estimated beta for the two companies?

# %%
estimates = [(stock, results[stock].params.values) for stock in stocks]

# %%
sorted(estimates, key=lambda x: x[1][1]) # ticker, alpha, beta / Sort by beta

# %% [markdown]
# - Lowest beta: KO / Consumer Defensive
# - Largest beta: AAPL / Tech
#
# It is consistent with the economic rationale. 
#
# The betas are also in the reasonable range between 0 ~ 2

# %% [markdown]
# ## (c)
#
# - Use the first 30 months only and run the regression.
# - Report the alpha/beta estimates

# %%
first30 = data_df.iloc[:30]

first30_results = {}

for stock in stocks:
    first30_results[stock] = get_SIM_regression(first30[stock], first30['SPY'], print_summary=False)

first30_estimates = [(stock, first30_results[stock].params.values) for stock in stocks]

sorted(first30_estimates, key=lambda x: x[1][1]) # ticker, alpha, beta / Sort by beta

# %% [markdown]
# ## (d)
#
# - Use the latter 30 months only and run the regression.
# - Report the alpha/beta estimates

# %%
last30 = data_df.iloc[30:]

last30_results = {}

for stock in stocks:
    last30_results[stock] = get_SIM_regression(last30[stock], last30['SPY'], print_summary=False)

last30_estimates = [(stock, last30_results[stock].params.values) for stock in stocks]

sorted(last30_estimates, key=lambda x: x[1][1]) # ticker, alpha, beta / Sort by beta


# %% [markdown]
# ## (e)
#
# - Are the three set of estimates (all/first/latter) identical? 
# - Discuss the result of a), c) and d)

# %%
def plot_first_and_last(stock_ticker):
    # Predicting the lines
    first30_line = first30_results[stock_ticker].predict(sm.add_constant(first30['SPY']))
    last30_line = last30_results[stock_ticker].predict(sm.add_constant(last30['SPY']))

    plt.figure(figsize=(12, 8))
    
    # Scatter and plot for first 30 months
    plt.scatter(first30['SPY'], first30[stock_ticker], color='blue', alpha=0.5, label=f'{stock_ticker} First 30 months')
    plt.plot(first30['SPY'], first30_line, color='blue', linewidth=2, label='First 30 months OLS Line')
    
    # Scatter and plot for last 30 months
    plt.scatter(last30['SPY'], last30[stock_ticker], color='orange', alpha=0.5, label=f'{stock_ticker} Last 30 months')
    plt.plot(last30['SPY'], last30_line, color='orange', linewidth=2, label='Last 30 months OLS Line')
    
    # Labels and legend
    plt.xlabel('Market Return (SPY)')
    plt.ylabel(f'{stock_ticker} Return')
    plt.title(f'{stock_ticker} Return vs Market Return')
    plt.legend()
    plt.show()



# %%
plot_first_and_last('AAPL')

# %%
plot_first_and_last('SPGI')

# %%
plot_first_and_last('MCD')

# %%
plot_first_and_last('KO')

# %%
