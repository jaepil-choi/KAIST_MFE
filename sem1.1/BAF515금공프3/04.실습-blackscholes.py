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
# # 실습 ch4
#
# 블랙 숄즈 머튼 함수 작성 
#
# Black Scholes Merton function

# %%
def bs_call(S0, K, T, r, sigma):

    import math
    import scipy.stats

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    c = S0 * scipy.stats.norm.cdf(d1) - K * math.exp(-r*T) * scipy.stats.norm.cdf(d2)

    return(round(c, 3))

bs_call(S0=40., K=42., T=0.5, r=0.015, sigma=0.2)


# %%
import os

# os.chdir('D:\\MFE_BAF515')  # MyBS.py 모듈은 'D:\\MFE_BAF515'에 저장되어 있음.

def implied_vol_call(S0, K, T, r, c):
    # from MyBS import bs_call
    
    i = 0; diff = 5
    
    while abs(diff) > 0.01:
        sigma = 0.005 * (i+1)
        diff = c - bs_call(S0, K, T, r, sigma)
        i += 1
    
    return i, sigma, diff

implied_vol_call(S0=40., K=40., T=0.5, r=0.05, c=3.3)


# %%
def vega(S0, K, T, r, sigma):
    import math
    from scipy.stats import norm

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    dval = S0 * norm.pdf(d1) * math.sqrt(T)
    
    return dval

# del vega # MyBS.py 모듈에 저장됨.

def implied_vol_call_byNR(S0, K, T, r, c):
    # import MyBS
    
    newsigma = 1
    diff = 1
    while abs(diff) > 0.001:
        sigma = newsigma
        
        # newsigma = sigma - (MyBS.bs_call(S0, K, T, r, sigma) - c) / MyBS.vega(S0, K, T, r, sigma)
        newsigma = sigma - (bs_call(S0, K, T, r, sigma) - c) / vega(S0, K, T, r, sigma)
        diff = newsigma - sigma
        
    return round(newsigma, 3)

implied_vol_call_byNR(S0=40., K=40., T=0.5, r=0.05, c=3.3)


# %%
