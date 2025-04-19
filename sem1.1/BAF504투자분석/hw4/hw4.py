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
END = '2024-05-31'

# START = '2015-01-01'
START = pd.to_datetime(END) - pd.DateOffset(years=5)
START = START.strftime('%Y-%m-%d')


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
pd.DataFrame([msft_result_values, goog_result_values, ko_result_values], index=stocks)

# %% [markdown]
# ## (c)

# %% [markdown]
# 다음 방법을 파트 (C)에 사용할 예정입니다.
#
# 여기서의 문제는 시장 위험 프리미엄 $E[R_m]$, 그 표준 편차, 각 주식의 기대 수익률의 알파 $ \alpha_i $ 등의 예측된 값에 접근할 수 없다는 것입니다.
#
# 타임머신이 있다고 가정해보겠습니다. 우리가 2년 전으로 돌아갈 수 있다고 하면, 데이터가 2019년 8월부터 2024년 5월까지 (총 5년) 존재하므로 2022년 6월로 돌아가게 됩니다.
#
# 이 "미래" 2년치 데이터를 통해 시장 위험 프리미엄의 정확한 미래 값, 그 표준 편차, 각 주식의 기대 수익률의 알파를 추정할 수 있습니다.
#
# 이 데이터를 사용하여 최적의 위험 포트폴리오를 구성할 것입니다. 이 포트폴리오는 지수와 세 개의 주식으로 구성되며, 샤프 비율을 최대화하도록 설계될 것입니다.

# %%
lookahead = 24

# %%
# "미래" 2개년치와 "과거" 3개년치를 나눔

lookahead_excs_df = excs_df.iloc[-lookahead:, :].copy()
past_excs_df = excs_df.iloc[:-lookahead, :].copy()

# %%
# "미래" 기준으로 market excess return의 평균과 표준편차를 구함

mkt_excs_mean = lookahead_excs_df['SPY_excs'].mean() # Market expected return
mkt_excs_std = lookahead_excs_df['SPY_excs'].std() # Market expected volatility

# %%
ann_mkt_excs_mean = mkt_excs_mean * 12
ann_mkt_excs_std = mkt_excs_std * np.sqrt(12)

# %%
# "과거" 기준으로 beta를 구함

past_results = {}

for stock in stocks:
    past_results[stock] = get_SIM_regression(past_excs_df[f'{stock}_excs'], past_excs_df['SPY_excs'], print_summary=False)


# %%
# "과거" 기준 beta
betas = np.array([past_results[stock].params.iloc[1] for stock in stocks])
betas

# %%
# "미래" 기준 excess return의 표준편차

msft_excs_std = lookahead_excs_df['MSFT_excs'].std()
goog_excs_std = lookahead_excs_df['GOOG_excs'].std()
ko_excs_std = lookahead_excs_df['KO_excs'].std()

excs_stds = np.array([msft_excs_std, goog_excs_std, ko_excs_std])
excs_stds

# %%
ann_excs_stds = excs_stds * np.sqrt(12)
ann_excs_stds

# %%
sys_stds = betas * ann_mkt_excs_std
sys_stds

# %%
resid_stds = np.sqrt(ann_excs_stds**2 - sys_stds**2)
resid_stds

# %%
# "미래" 기준으로 alpha를 구함

lookahead_results = {}

for stock in stocks:
    lookahead_results[stock] = get_SIM_regression(lookahead_excs_df[f'{stock}_excs'], lookahead_excs_df['SPY_excs'], print_summary=False)

# %%
print(lookahead_results['GOOG'].summary())

# %%
alphas = np.array([lookahead_results[stock].params.iloc[0] for stock in stocks])
alphas = alphas * 12 # annualize
alphas

# %%
ANN_MKT_RISKPREMIUM = ann_mkt_excs_mean # "미래" 기준으로 구했던 market excess return

riskpremiums = ANN_MKT_RISKPREMIUM * betas
riskpremiums

# %%
ANN_MKT_RISKPREMIUM

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
# weights = alpha_div_resid_vars / alpha_div_resid_vars.sum()
weights = alpha_div_resid_vars / np.abs(alpha_div_resid_vars).sum() # 부호 바뀌는 것 막기 위해 absolute sum으로 normalize
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

# %%
