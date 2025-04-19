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
# # 금공프3 Final
#
# 20249433 MFE 최재필

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import math
import random

# %% [markdown]
# ## 1. Mean-Variance Optimization

# %%
asset = pd.read_csv('it.csv')
asset.set_index('Date', inplace=True)
asset.head() # 다행히 데이터에 nan이 없다.

# %%
asset.shape


# %% [markdown]
# ### (1) `MVportfolio` 

# %%
def MVportfolio(asset, mu_p):
    asset_2d = asset.to_numpy()

    X = np.log(asset_2d[1:]/asset_2d[:-1]) # 수익률 (수익률은 log 차분 수익률로 계산)
    Q = np.cov(X.T) # 공분산 행렬
    r = np.nanmean(X, axis=0).reshape(-1, 1) # 기대값 (수익률 평균)
    l = np.ones(r.shape) # 1 벡터
    zero = np.zeros(l.shape) # 0 벡터

    # 라그랑지안 편미분 방정식 행렬
    Q_l_r = np.hstack([Q, l, r]) # 목적함수 편미분 
    l_0_0 = np.hstack([l.T, [[0]], [[0]]]) # 제약조건 1: 가중치 합 = 1
    r_0_0 = np.hstack([r.T, [[0]], [[0]]]) # 제약조건 2: 수익률 = mu_p

    L = np.vstack([Q_l_r, l_0_0, r_0_0]) # 완성된 라그랑지안 

    zero_l_mu = np.vstack([zero, [[1]], [[mu_p]]]) # 우변
    L_inv = np.linalg.inv(L) # 역행렬 계산

    w_lmda1_lmda2 = L_inv @ zero_l_mu # 라그랑지안 해벡터

    w = w_lmda1_lmda2[:-2] # 최적 포트폴리오 가중치
    lmda1 = w_lmda1_lmda2[-2] # 라그랑지안 해벡터 람다1
    lmda2 = w_lmda1_lmda2[-1] # 라그랑지안 해벡터 람다2

    var = w.T @ Q @ w # 최적 포트폴리오 분산

    return w, var


# %% [markdown]
# ### (2) Efficient Frontier

# %%
mu_p_min = -0.001
mu_p_max = 0.001

mu_p_range = np.linspace(mu_p_min, mu_p_max, 100)

# %%
asset_2d = asset.to_numpy()

X = np.log(asset_2d[1:]/asset_2d[:-1]) # 수익률 (수익률은 log 차분 수익률로 계산)
r = np.nanmean(X, axis=0).reshape(-1, 1) # 기대값 (수익률 평균)

# %%
w_var = [MVportfolio(asset, mu_p) for mu_p in mu_p_range]
var_ret = np.array([(var, w.T @ r) for w, var in w_var]).reshape(len(w_var), 2)

# %%
# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(var_ret[:, 0], var_ret[:, 1], marker='o', linestyle='-')

plt.title('Efficient Frontier')
plt.xlabel('Variance (Risk^2)')
plt.ylabel('Expected Return (daily)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Momentum

# %%
# (1) price 파일 불러온 뒤 date 열을 DatetimeIndex로 변경한 뒤 인덱스로 설정
price = pd.read_csv('price.csv')
price['date'] = pd.to_datetime(price['date'])
price.set_index('date', inplace=True)
price.head() 

# 참고: 데이터에 nan이 많음. 
# 대부분은 상장폐지 종목 또는 상장 이전 종목이라고 판단됨.
# 하지만 전 기간 nan인 종목도 있음. 

# %%
price.shape

# %%
# (2) 2019년도 자료만 선택
price_sub = price.loc['2019-01-01':'2019-12-31', :].copy() 

# %%
# (3) 누적곱으로 수익률 계산 (Series 객체로 저장)
cum_ret = price_sub.pct_change(fill_method=None).add(1).prod() - 1

# %%
# (4) 누적 수익률 상위 10개 종목 출력
top10_cumret = cum_ret.sort_values(ascending=False).head(10)
top10_cumret

# %%
price_sub[top10_cumret.index[0]].plot(figsize=(10, 6), title='2019 Top 1 Cumulative Return')

# %%
# (5) 종목별 연율화 변동성 계산 (252일 기준, Series 객체로 저장)
std = price_sub.pct_change(fill_method=None).std() * np.sqrt(252)

# %%
# (6) std가 0인 경우와 nan인 경우를 제외
std = std[std != 0].dropna()

# %%
# (7) 샤프지수 계산
shrp = cum_ret / std

# %%
# (8) 샤프지수가 nan인 경우 shrp 최소값으로 대체
shrp = shrp.fillna(shrp.min())

# %%
# (9) 샤프지수 상위 10개 종목 출력
top10_shrp = shrp.sort_values(ascending=False).head(10)
top10_shrp

# %%
# (10) Top 10 종목의 최종 결과 출력

top10_shrp_stocks = top10_shrp.index

final_result = pd.DataFrame(
    data=zip(
        cum_ret[top10_shrp_stocks],
        std[top10_shrp_stocks],
        shrp[top10_shrp_stocks]
        ),
    index=top10_shrp_stocks,
    columns=['cum_ret', 'std', 'shrp']
)

final_result


# %% [markdown]
# ## 3. Monte-Carlo Simulation

# %%
def ECallSimul_1(S0, K, T, r, sigma, M, l=250000):
    S = []
    dt = T/M
    for i in range(l):
        path = []
        
        for t in range(M+1):
            if t == 0:
                path.append(S0)
            else:
                z = random.gauss(0., 1.)
                St = path[t-1] * math.exp( (r - 0.5*sigma**2)*dt + sigma*math.sqrt(dt)*z )
                path.append(St)
            
        S.append(path)
        
    sum_val = 0.

    for path in S:
        sum_val += max(path[-1] - K, 0)
    
    C0 = math.exp(-r*T) * sum_val / l

    return round(C0, 3)


# %% [markdown]
# ### (1) 가능한 모든 부분을 `numpy`를 활용하는 것으로 수정

# %%
def ECallSimul_2(S0, K, T, r, sigma, M, l=250000):
    dt = T/M

    Z = np.random.randn(l, M)
    S = np.zeros((l, M+1)) # 맨 앞에 S0를 넣기 위해 M+1
    S[:, 0] = S0

    drift = (r - 0.5*sigma**2)*dt
    diffusion = sigma * np.sqrt(dt) * Z

    S[:, 1:] = S0 * np.exp(np.cumsum(drift + diffusion, axis=1))

    payoffs = np.maximum(S[:, -1] - K, 0)
    C0 = np.exp(-r*T) * np.mean(payoffs)

    return round(C0, 3)


# %% [markdown]
# ### (2) 연산시간 비교

# %%
S0 = 100.
K = 105.
T = 1.
r = 0.05
sigma = 0.2
M = 50
l = 250000

# %%
# %time C0_1 = ECallSimul_1(S0, K, T, r, sigma, M, l)

# %%
# %time C0_2 = ECallSimul_2(S0, K, T, r, sigma, M, l)
