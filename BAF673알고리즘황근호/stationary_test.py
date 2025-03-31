#%%
import numpy as np
from ar1_process import ar1_process, random_walk
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


sigma = 1
T = 200

N = 1000
c = 0
for i in range(N):
    y = random_walk(sigma, T)
    adf = adfuller(y)
    pvalue = adf[1]
    if pvalue < 0.05:
        c += 1

## 1000번을 했을 때 50번 정도를 기각함. 유의수준을 5% 줬으니까 당연한 이야기. 
## 이렇게 우연히 기각되는 경우도 있으니 항상 주의해서 통계검정 해야 함. 

print(f"Random Walk임에도 Unit root를 기각하는 비율(유의수준))")
print(f"Reject H0(Random Walk): {c/N: 0.1%}")


#%%
c = 0
phi1 = 0.99
for i in range(N):
    y = ar1_process(0, phi1, sigma, T)
    adf = adfuller(y)
    pvalue = adf[1]
    if pvalue < 0.05:
        c += 1
print(f"Stationary로 판단되는 경우의 비율 Power (AR(1) phi1={phi1}): {c/N: 0.1%}")



#%%
c = 0
phi1 = 0.5
for i in range(N):
    y = ar1_process(0, phi1, sigma, T)
    adf = adfuller(y)
    pvalue = adf[1]
    if pvalue < 0.05:
        c += 1
print(f"Stationary로 판단되는 경우의 비율 Power(AR(1) phi1={phi1}): {c/N: 0.1%}")



#%%
# 데이터 길이와 파라미터에 따른 Power 비교

## 여기서 하신 말이, OLS 문제가 있지만 이걸 데이터 많이 써서 하면 plim으로 확률적 lim 된다고 함. 

import pandas as pd
c = 0
phi1s = [0.8, 0.9, 0.95, 0.97, 0.99]
Ts = [100, 200, 300, 400, 500]
powers = pd.DataFrame(index=Ts, columns=phi1s, dtype=float)

for T in Ts:
    for phi1 in phi1s:
        print(phi1, T)
        c = 0
        for i in range(N):
            y = ar1_process(0, phi1, sigma, T)
            adf = adfuller(y)
            pvalue = adf[1]
            if pvalue < 0.05:
                c += 1
        powers.loc[T,phi1] = (c/N)

#%%
powers_formatted = powers.style.format("{:0.2%}")
powers.plot(marker='o', title='Power of ADF Test', xlabel='Time Series Length', ylabel='Power', grid=True)

powers_formatted

## 계속 time series length를 늘려도 0.99면 거의 기각 못하는거고 다른 신뢰수준들은 그래프로 확인. 
# %%
