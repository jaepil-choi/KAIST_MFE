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
# # 투자분석 Assignment 1
#
# Stock evaluation
#
# ## Requirements
# - Data
#     - risk-free rate
#         - 3-month T-bill or 1-month LIBOR
#     - market risk premium
# - Model
#     - constant growth dividend discount model
# - Do NOT hand in the data/program code. 
#
# ## Questions:
# - a. What are the names of the stocks you choose:
#     - KO (코카콜라)
#     - XOM (엑손모빌)
#     - NVDA (엔비디아)
# - b. Specify risk-free rate, risk premium, data source
#     - risk-free
#     - risk premium
#     - 주식: Valley AI
# - c. Required Rate of Return using CAPM. Specify: 
#     - B: Firm's beta
#     - R_m: Expected rate of return of the market index portfolio 
# - D. Calculate PVGO for each stock
# - E. Find V_0, V_1 for each stock
# - F. Find the expected return for each stock
#     - i.e = (V_1-P_0) / P_0 
#     - P_0 is the current market price
# - G. Based on the calculation above, which stock do you buy/sell? Explain briefly. 
#
#
#

# %%
import pandas_datareader.data as web
import yfinance as yf

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# sp500tr = web.DataReader('^SP500TR', 'yahoo', start='2023-03-25', end='2024-03-25')
# sp500tr

# %%
sp500tr = yf.Ticker('^SP500TR')

# %%
sp500tr_df = sp500tr.history(period='1y', interval='1d')
sp500tr_df.tail()

# %%
sp500tr_df['Close'].plot()

# %%
# P0 = sp500tr_df.iloc[0]['Close']
# P1 = sp500tr_df.iloc[-1]['Close']

# %% [markdown]
# HPR로 하면 market timing에 노출된다. 
#
# 그냥 일평균 수익률 내서 이를 annualize하자. 

# %%
returns = sp500tr_df['Close'].pct_change()

# %%
sns.histplot(data=returns)

# %%
rf = 0.040727

# %%
mean = returns.mean()
mean_annualized = (1 + mean)**252 - 1
mean_annualized * 100

# %%
gmean = (1 + returns).prod()**(1/len(returns)) - 1
gmean_annualized = (1 + gmean)**252 - 1
gmean_annualized * 100

# %%
risk_premium = gmean_annualized - rf
risk_premium * 100


# %%
def get_k(beta, rm, rf):
    return rf + beta * (rm - rf)


# %%
KO_k = get_k(0.59, gmean_annualized, rf)
XOM_k = get_k(0.95, gmean_annualized, rf)
AAPL_k = get_k(1.29, gmean_annualized, rf)

# %%
print(f'KO: {KO_k * 100:.4f}%')
print(f'XOM: {XOM_k * 100:.4f}%')
print(f'AAPL: {AAPL_k * 100:.4f}%')

# %% [markdown]
# Dividend 데이터 

# %%
KO = yf.Ticker('KO')
XOM = yf.Ticker('XOM')
AAPL = yf.Ticker('AAPL')

# %%
KO_div = KO.dividends
XOM_div = XOM.dividends
AAPL_div = AAPL.dividends

# %%
KO_div

# %%
KO_div5y = KO_div.loc[KO_div.index > '2019-01-01']
XOM_div5y = XOM_div.loc[XOM_div.index > '2019-01-01']
AAPL_div5y = AAPL_div.loc[AAPL_div.index > '2019-01-01']

# %%
KO_div5y.plot(title='KO Dividends')

# %%
KO_g = (1+KO_div5y.pct_change().mean())**4 - 1
KO_g * 100

# %%
XOM_g = (1+XOM_div5y.pct_change().mean())**4 - 1
XOM_g * 100

# %%
AAPL_g = (1+AAPL_div5y.pct_change().mean())**4 - 1
AAPL_g * 100

# %% [markdown]
# calculate g from yahoo finance

# %%
KO_divy = 0.0320
XOM_divy = 0.0334
AAPL_divy = 0.0057

# %%
KO_plowback = 1 - KO_divy
XOM_plowback = 1 - XOM_divy
AAPL_plowback = 1 - AAPL_divy

# %%
KO_ROE = 0.4016
XOM_ROE = 0.1800
AAPL_ROE = 1.5427

# %%
KO_g_est = KO_ROE * KO_plowback
XOM_g_est = XOM_ROE * XOM_plowback
AAPL_g_est = AAPL_ROE * AAPL_plowback

# %%
print(f'KO: {KO_g_est * 100:.4f}%')
print(f'XOM: {XOM_g_est * 100:.4f}%')
print(f'AAPL: {AAPL_g_est * 100:.4f}%')

# %% [markdown]
# PVGO 계산
#

# %%
KO_eps = 2.47
XOM_eps = 8.89
AAPL_eps = 6.43

# %%
KO_div_usd = 1.94
XOM_div_usd = 3.80
AAPL_div_usd = 0.96


# %%
def calculate_V0(g, k, div):
    return div / (k - g)


# %%
KO_V0 = calculate_V0(KO_g, KO_k, KO_div_usd)
XOM_V0 = calculate_V0(XOM_g, XOM_k, XOM_div_usd)
AAPL_V0 = calculate_V0(AAPL_g, AAPL_k, AAPL_div_usd)

# %%
print(f'KO: {KO_V0:.4f}')
print(f'XOM: {XOM_V0:.4f}')
print(f'AAPL: {AAPL_V0:.4f}')


# %%
def calculate_nogrowth_V(eps, k):
    return eps / k


# %%
KO_nogrowth_V = calculate_nogrowth_V(KO_eps, KO_k)
XOM_nogrowth_V = calculate_nogrowth_V(XOM_eps, XOM_k)
AAPL_nogrowth_V = calculate_nogrowth_V(AAPL_eps, AAPL_k)

# %%
print(f'KO: {KO_nogrowth_V:.4f}')
print(f'XOM: {XOM_nogrowth_V:.4f}')
print(f'AAPL: {AAPL_nogrowth_V:.4f}')

# %%
KO_PVGO = KO_V0 - KO_nogrowth_V
XOM_PVGO = XOM_V0 - XOM_nogrowth_V
AAPL_PVGO = AAPL_V0 - AAPL_nogrowth_V

# %%
print(f'KO: {KO_PVGO:.4f}') 
print(f'XOM: {XOM_PVGO:.4f}')
print(f'AAPL: {AAPL_PVGO:.4f}')

# %% [markdown]
# V1 구하기 
#
# (1+g) 만 곱하면 됨. 

# %%
KO_V1 = KO_V0 * (1 + KO_g)
XOM_V1 = XOM_V0 * (1 + XOM_g)
AAPL_V1 = AAPL_V0 * (1 + AAPL_g)

# %%
print(f'KO: {KO_V1:.4f}')   
print(f'XOM: {XOM_V1:.4f}')
print(f'AAPL: {AAPL_V1:.4f}')

# %% [markdown]
# Earning 구하기 

# %%
KO_now = KO.history(period='1d')['Close'].iloc[-1]
XOM_now = XOM.history(period='1d')['Close'].iloc[-1]
AAPL_now = AAPL.history(period='1d')['Close'].iloc[-1]


# %%
def expected_ret(V1, P0):
    return (V1 - P0) / P0


# %%
KO_expR = expected_ret(KO_V1, KO_now) * 100
XOM_expR = expected_ret(XOM_V1, XOM_now) * 100
AAPL_expR = expected_ret(AAPL_V1, AAPL_now) * 100

# %%
print(f'KO: {KO_expR:.4f}')
print(f'XOM: {XOM_expR:.4f}')
print(f'AAPL: {AAPL_expR:.4f}')

# %%
print(f'KO error: {KO_now/KO_V0}')
print(f'XOM error: {XOM_now/XOM_V0}')
print(f'AAPL error: {AAPL_now/AAPL_V0}')

# %%
