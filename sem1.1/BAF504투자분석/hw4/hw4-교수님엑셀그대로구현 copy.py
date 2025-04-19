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
# # 투자분석 hw4
#
# 20249433 최재필
#

# %%

# %%

# %% [markdown]
# alpha랑 market risk premium, 앞에서 구한 결과대로 값만 상수로 넣고 돌려보기. 

# %%

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import statsmodels.api as sm

import yfinance as yf

# %% [markdown]
# ## (a)
#
# - MSFT
# - GOOG
# - KO
#
# Source: Yahoo Finance

# %%
START = '2015-01-01'
END = '2019-12-31'

# %%
MSFT = yf.Ticker("MSFT") # originally MSFT
GOOG = yf.Ticker("GOOG") # originally GOOGL
KO = yf.Ticker("KO") # originally KO

SPY = yf.Ticker("SPY")
Tbill3M = yf.Ticker("^IRX")

# %%
MSFT_df = MSFT.history(start=START, end=END, interval='1mo')
MSFT_ret = MSFT_df['Close'].pct_change().dropna()
MSFT_ret = MSFT_ret.rename('MSFT')

GOOG_df = GOOG.history(start=START, end=END, interval='1mo')
GOOG_ret = GOOG_df['Close'].pct_change().dropna()
GOOG_ret = GOOG_ret.rename('GOOG')

KO_df = KO.history(start=START, end=END, interval='1mo')
KO_ret = KO_df['Close'].pct_change().dropna()
KO_ret = KO_ret.rename('KO')

SPY_df = SPY.history(start=START, end=END, interval='1mo')
SPY_ret = SPY_df['Close'].pct_change().dropna()
SPY_ret = SPY_ret.rename('SPY')

Tbill3M_df = Tbill3M.history(start=START, end=END, interval='1mo')
Tbill3M_ret = Tbill3M_df['Close']
Tbill3M_ret = Tbill3M_ret.rename('Tbill3M')
Tbill3M_ret = Tbill3M_ret / 100 / 12 # convert to monthly rate

# %%
MSFT_ret.index = MSFT_ret.index.to_period('M')
GOOG_ret.index = GOOG_ret.index.to_period('M')
KO_ret.index = KO_ret.index.to_period('M')
SPY_ret.index = SPY_ret.index.to_period('M')
Tbill3M_ret.index = Tbill3M_ret.index.to_period('M')

Tbill3M_ret = Tbill3M_ret.reindex(MSFT_ret.index, method='ffill')

# %%
df = pd.concat([MSFT_ret, GOOG_ret, KO_ret, SPY_ret, Tbill3M_ret], axis=1)
df = df.dropna()

# %%
# Make it excess return
excs_df = df.loc[:, ['MSFT', 'GOOG', 'KO', 'SPY']].subtract(df.loc[:, 'Tbill3M'], axis=0)
excs_df.columns = [f'{ticker}_excs' for ticker in excs_df.columns]
excs_df.tail()

# %% [markdown]
# ## (b)

# %%
stocks = ['MSFT', 'GOOG', 'KO']

def get_SIM_regression(stock_returns, market_returns, print_summary=True):
    X = sm.add_constant(market_returns)
    model = sm.OLS(stock_returns, X)
    results = model.fit()

    if print_summary:
        print(results.summary())

    return results


# %% [markdown]
# 각 종목의 초과수익률을 시장 초과수익률에 대해 regress

# %%
results = {}

for stock in stocks:
    print(f'SIM regression result of {stock}')
    results[stock] = get_SIM_regression(excs_df[f'{stock}_excs'], excs_df['SPY_excs'], print_summary=True)
    print('\n'*5)


# %%
def get_result_values(single_stock_result):
    alpha, beta = single_stock_result.params
    t_value_alpha, t_value_beta = single_stock_result.tvalues
    p_value_alpha, p_value_beta = single_stock_result.pvalues
    r_squared = single_stock_result.rsquared

    # residual std는 따로 계산해야 함
    residuals = single_stock_result.resid
    resid_df = single_stock_result.df_resid
    residual_std = np.sqrt( (residuals**2).sum() / resid_df )

    result_values = {
        'alpha': alpha,
        'beta': beta,
        't_value_alpha': t_value_alpha,
        't_value_beta': t_value_beta,
        'p_value_alpha': p_value_alpha,
        'p_value_beta': p_value_beta,
        'r_squared': r_squared,
        'residual_std': residual_std
    }

    return result_values



# %%
msft_result_values = get_result_values(results['MSFT'])
goog_result_values = get_result_values(results['GOOG'])
ko_result_values = get_result_values(results['KO'])


# %%
msft_result_values

# %%
goog_result_values

# %%
ko_result_values

# %% [markdown]
# ## (c)

# %%
mkt_excs_mean = excs_df['SPY_excs'].mean() # Market expected return
mkt_excs_std = excs_df['SPY_excs'].std() # Market expected volatility

# %%
ann_mkt_excs_mean = mkt_excs_mean * 12
ann_mkt_excs_std = mkt_excs_std * np.sqrt(12)

# %%
betas = np.array([msft_result_values['beta'], goog_result_values['beta'], ko_result_values['beta']])
betas

# %%
msft_excs_std = excs_df['MSFT_excs'].std()
goog_excs_std = excs_df['GOOG_excs'].std()
ko_excs_std = excs_df['KO_excs'].std()

excs_stds = np.array([msft_excs_std, goog_excs_std, ko_excs_std])
excs_stds

# %%
ann_excs_stds = excs_stds * np.sqrt(12)

# %%
sys_stds = betas * ann_mkt_excs_std
sys_stds

# %%
resid_stds = np.sqrt(ann_excs_stds**2 - sys_stds**2)
resid_stds

# %%
# alphas = [0.02, -0.01, 0.01] # already annualized
alphas = [ 0.10675492,  0.10936121, -0.06044151]


# %%
# ANN_MKT_RISKPREMIUM = 0.06
ANN_MKT_RISKPREMIUM = 0.1109

riskpremiums = ANN_MKT_RISKPREMIUM * betas
riskpremiums

# %% [markdown]
# ### 교수님 엑셀처럼 값 정리하여 optimal risky portfolio 구하기

# %% [markdown]
# #### $ \sigma^2(\epsilon_i) $ 

# %%
resid_vars = resid_stds**2
resid_vars

# %% [markdown]
# #### $ \alpha_i / \sigma^2(\epsilon_i) $

# %%
alpha_div_resid_vars = alphas / resid_vars
alpha_div_resid_vars

# %% [markdown]
# #### $ w_i $

# %%
weights = alpha_div_resid_vars / alpha_div_resid_vars.sum()
weights

# %%
np.round(weights.sum(), 10)

# %% [markdown]
# #### $ \alpha_A $

# %%
weighted_alpha = weights @ alphas
weighted_alpha

# %% [markdown]
# #### $ \beta_A $

# %%
weighted_beta = weights @ betas
weighted_beta

# %% [markdown]
# #### $ \sigma^2(\epsilon_A) $ - residual variance

# %%
active_residual_var = weights**2 @ resid_vars
active_residual_var

# %% [markdown]
# #### $ \sigma_A^2 $ - active portfolio variance
#

# %%
active_port_var = weighted_beta**2 * ann_mkt_excs_std**2 + active_residual_var
active_port_var

# %% [markdown]
# #### $ w_A^0 $

# %%
w_A_0 = (weighted_alpha / active_residual_var) / (mkt_excs_mean / mkt_excs_std**2) # 여기서 lookahead std로 넣었다. 원래는 historical 쓰던데.
w_A_0

# %% [markdown]
# #### $ w_A^* $

# %%
w_A_star = w_A_0 / (1 + w_A_0 * (1 - weighted_beta))
w_A_star

# %% [markdown]
# 개별 주식 weight

# %%
final_weights = w_A_star * weights
final_weights

# %% [markdown]
# #### $ w_M^* $

# %%
w_M_star = 1 - w_A_star
w_M_star

# %% [markdown]
# ### 그래프 확인

# %%
final_weights

# %%
((1 + df).cumprod() - 1).plot()

# %%
final_alpha = final_weights @ alphas
final_alpha

# %%
final_std = np.sqrt(final_weights @ resid_vars)
final_std

# %% [markdown]
# ## (d)

# %% [markdown]
# Information ratio

# %%
information_ratio = final_alpha / final_std
information_ratio

# %% [markdown]
# ## (e)

# %%
active_riskpremium = final_weights @ riskpremiums
active_riskpremium

# %%
optimal_riskpremium = ANN_MKT_RISKPREMIUM * w_M_star + active_riskpremium * w_A_star
optimal_riskpremium

# %%
market_sharpe = ANN_MKT_RISKPREMIUM / ann_mkt_excs_std
market_sharpe

# %%
optimal_sharpe = np.sqrt(market_sharpe ** 2 + information_ratio ** 2)
optimal_sharpe

# %%
