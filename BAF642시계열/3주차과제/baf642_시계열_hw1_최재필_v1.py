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
# # BAF642 금융시계열 과제 1
#
# 20249433 최재필

# %%
from pathlib import Path
import os, sys
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

## Time Series Decomposition & Stationarity tests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

## ARIMA, GARCH
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from arch import arch_model

## VAR
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

## Kalman Filter
from scipy import stats, signal
from pykalman import KalmanFilter

## Performance analytics
import quantstats as qs

# %% [markdown]
# ## Import Data

# %%
CWD_PATH = Path('.').resolve()
DATA_PATH = CWD_PATH / 'data'


# %%
def import_BB_data(filename, data_cols, return_col=None):
    data = pd.read_csv(DATA_PATH / filename, index_col=0, parse_dates=True)
    data.index.name = 'date'
    data.columns = data_cols
    
    vol_col = data_cols[4]
    ret_col = data_cols[5]

    try:
        data[vol_col] = data[vol_col].str.replace('K', '000')
        data[vol_col] = data[vol_col].str.replace('M', '000000')
        data[vol_col] = data[vol_col].str.replace('B', '000000000')
        data[vol_col] = data[vol_col].astype(float)
    except:
        pass

    data[ret_col] = data[ret_col].str.replace('%', '').astype(float) / 100

    data = data.astype(float)

    if return_col:
        data = data[return_col]
    
    return data


# %%
data_cols1 = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
data_cols2 = ['종가', '시가', '고가', '저가', '거래량', '변동 %']

# %%
v1x = import_BB_data('VIX.csv', data_cols2, '종가')
v1x.name = 'V1X'

v1x_futures = import_BB_data('VIX_futures.csv', data_cols2, '종가')
v1x_futures.name = 'V1X_futures'

v2x = import_BB_data('V2X.csv', data_cols1, 'Price')
v2x.name = 'V2X'

v2x_futures = import_BB_data('V2X_futures.csv', data_cols1, 'Price')
v2x_futures.name = 'V2X_futures'

# %% [markdown]
# ## 1. 추세/계절성/잡음 분해
# VIX와 V2X의 시계열을 추세, 계절성 및 잡음으로 분해하라.
#

# %%
v1x_decompose = seasonal_decompose(v1x, model='additive', period=252).plot()
v2x_decompose = seasonal_decompose(v2x, model='additive', period=252).plot()

# %% [markdown]
# Seasonality가 있다고 봐야하나? 

# %%
## GPT

# Friedman test for seasonality

# --> seasonality test 아님

# %%
## GPT

# Ljung-Box test for seasonality

# --> seasonality check에 사용되기 어려움

# %% [markdown]
#
# ## 2. 단위근 테스트 및 정상성 확인
# VIX와 V2X 각각에 대해 단위근 테스트를 실행하고 정상성을 살펴보라.
#

# %% [markdown]
# ### Plot ACF, PACF
#
# - ACF는 slow decaying을 보여주고 
# - PACF는 빠른 decay를 보여줌 (short term memory)
#
# 결과: Stationary

# %%
# ACF
plot_acf(v1x, lags=10, title='V1X PACF', )
plot_acf(v2x, lags=10, title='V2X PACF', )
plt.show()

# %%
# PACF
plot_pacf(v1x, lags=10, title='V1X PACF', )
plot_pacf(v2x, lags=10, title='V2X PACF', )
plt.show()


# %% [markdown]
# ### ADF Test
#
# - H0: The series has a unit root (i.e, the series is not stationary)
#     - If rejected --> Stationary. 
# - Use `c` regression (and `ct`) regression because there's no clear trend. 
# - Use AIC autolag rather than setting fixed `maxlags=`. 
#
# 결과: Stationary
#

# %%
def adf_result(data, regression='c', autolag='AIC', alpha=0.05):
    result = adfuller(data, regression=regression, autolag=autolag,)
    adf = result[0]
    print(f'ADF Statistic: {adf:.4f}')
    
    p_value = result[1]
    print(f'p-value: {p_value:.4f}')
    
    critical_values = result[4]
    for key, value in critical_values.items():
        print(f'Critical Value {key}: {value:.4f}')
    
    if p_value < alpha:
        print('Reject the null hypothesis: Stationary')
    else:
        print('Fail to reject the null hypothesis: Non-stationary')



# %%
adf_result(v1x)

# %%
adf_result(v2x)

# %% [markdown]
# What if `regression='ct'` (constant trend) ?

# %%
adf_result(v1x, regression='ct')

# %%
adf_result(v2x, regression='ct')


# %% [markdown]
# ### KPSS Test
#
# - H0: The series is either trend stationary or level stationary
#     - If rejected --> Non-Stationary (반대임)
# - Use `c` regression for constant trend
#
# ADF vs KPSS
# - 일단 ADF가 국룰. KPSS는 보조
# - Power and Size
#     - ADF는 작은 샘플 or near-unit root process 에서 low power일 수 있음. 
#     - KPSS는 autoregressive parameter가 아주 작을 때 잘 작동
# - Autoregressive Parameter Values
#     - 자기상관계수가 0보단 훨씬 크고, 1에 가까울 경우: ADF
#     - 자기상관계수가 아주 작을 경우: KPSS
# - Trend Sensitivity
#     - ADF는 trend-stationary process와 unit root process를 헷갈려할 수 있음
#     - Deterministic trend가 있을 때 KPSS가 더 sensitive 
#         - (좋은 것임. Deterministic trend가 있어도 잘 구분한다는 소리)
#     
#
#

# %%
def kpss_result(data, regression='c', nlags='auto', alpha=0.05):
    # 'auto' (default): Uses a data-dependent method based on Hobijn et al. (1998)
    # 'legacy': Uses int(12 * (n / 100)**(1 / 4)) as in Schwert (1989)

    result = kpss(data, regression=regression, nlags=nlags)
    kpss_stat = result[0]
    print(f'KPSS Statistic: {kpss_stat:.4f}')
    
    p_value = result[1]
    print(f'p-value: {p_value:.4f}')
    
    critical_values = result[3]
    for key, value in critical_values.items():
        print(f'Critical Value {key}: {value:.4f}')
    
    if p_value < alpha:
        print('Fail to reject the null hypothesis: Stationary')
    else:
        print('Reject the null hypothesis: Non-stationary')

# 반대로 Fail to reject 해야 stationary라는 점에 주의


# %%
kpss_result(v1x)

# %%
kpss_result(v2x) # V2X의 경우 여기선 non-stationary라고 나옴 

# %% [markdown]
#
# ## 3. ARIMA GARCH 모델 적용
# VIX와 V2X 각각에 대해 ARIMA GARCH 모델을 적용하라.
#

# %% [markdown]
# ### step 1: auto arima
#
# ARIMA 차수를 auto fitting 해줌
#
# - 작동 원리
#     - AR(p), I(d), MA(q) 계수 grid search
#     - AIC 등의 evaluation criteria로 최적 찾음
#     - Seasonality도 자동 detection 해주도록 할 수 있음
#     - Stepwise search: grid search를 모든 combination에 대해 다 돌리는게 아니라 step step 나가며 greedy하게 찾음
#
# Seasonality in VIX? 
# - Seasonality, 눈으로 봐선 있긴 한데 불명확하다. 
# - Friedman, Ljung-Box 는 Seasonality test 아님
# - 그냥 있는 경우 없는 경우 다 해보자. 
#     - 하지만 `seasonal=True`는 너무 계산이 무거워 skip. 
#
# 결과: 
# - V1X: ARIMA(1, 1, 0)으로 fitting
# - V2X: ARIMA(0, 1, 1)로 fitting

# %%
# 계절성 없다고 칠 경우

# v1x
pmd_model_v1x = auto_arima(
    v1x, 

    seasonal=False, 

    trace=True, 
    error_action='ignore', 
    suppress_warnings=True, # 너무 많이 나옴. 
    stepwise=True, # Reduce computational cost by performing stepwise search
    )

pmd_model_v1x.fit(v1x)

# %%
pmd_model_v1x.summary()

# %%
pmd_model_v2x = auto_arima(
    v2x, 

    seasonal=False, 

    trace=True, 
    error_action='ignore', 
    suppress_warnings=True, # 너무 많이 나옴. 
    stepwise=True, # Reduce computational cost by performing stepwise search
    )

pmd_model_v2x.fit(v2x)


# %%
pmd_model_v2x.summary()

# %% [markdown]
# ### ARIMA-GARCH
#
# - fitting 시킨 ARIMA로 mean structure 부분을 효과적으로 제거. 남은 variance structure (residuals) 만가지고 GARCH를 모델링
# - Best GARCH model selection:
#     - GARCH는 model selection (예를들어 (1,1))을 한 뒤 MLE로 fitting 됨. 
#     - fitting시킨 후 AIC같은 eval metric으로 평가 가능
#     - 여러 model을 AIC 구해 그 중 좋은 것을 select하는 방법 (보배)

# %%
arima_resid_v1x = pmd_model_v1x.resid()
arima_resid_v2x = pmd_model_v2x.resid()

# %%
p_range = range(1, 4)
q_range = range(1, 4)

pq_space = product(p_range, q_range)

best_aic_v1x = np.inf
best_pq_v1x = None

best_aic_v2x = np.inf
best_pq_v2x = None

for p, q in pq_space:
    garch_model_v1x = arch_model(arima_resid_v1x, vol='Garch', p=p, q=q)
    result_v1x = garch_model_v1x.fit(disp='off')

    aic_v1x = result_v1x.aic
    if aic_v1x < best_aic_v1x:
        best_aic_v1x = aic_v1x
        best_pq_v1x = (p, q)
    
    garch_model_v2x = arch_model(arima_resid_v2x, vol='Garch', p=p, q=q)
    result_v2x = garch_model_v2x.fit(disp='off')

    aic_v2x = result_v2x.aic
    if aic_v2x < best_aic_v2x:
        best_aic_v2x = aic_v2x
        best_pq_v2x = (p, q)
    
print(f'Best AIC for V1X: {best_aic_v1x:.4f} with p, q: {best_pq_v1x}')
print(f'Best AIC for V2X: {best_aic_v2x:.4f} with p, q: {best_pq_v2x}')



# %%
result_v1x.summary()

# %%
result_v2x.summary()

# %%

# %%

# %%

# %% [markdown]
#
# ## 4. Multivariate GARCH 모델 적용
# VIX와 V2X에 대해 Multivariate GARCH 모델을 적용해보라.
#

# %% [markdown]
#
# ## 5. VAR 모델 및 분산 분석
# VIX와 V2X를 이용해 VAR 모델을 만들고, Granger causality 테스트를 실행하고, Impulse response 그래프와 분산 분해(Variance Decomposition)를 계산하고 그래프로 보여라.
#

# %% [markdown]
#
# ## 6. VECM 모델 구축
# VIX와 V2X를 이용해 공적분관계를 확인하고, VECM 모델을 구축하라.
#

# %%

# %% [markdown]
#
# ## 7. 페어 트레이딩
# 트레이딩 경계(평균 +/- some 표준편차)를 설정하고 다음 방법으로 페어 트레이딩을 수행하라.
#

# %%

# %% [markdown]
#
# ### 1) 단순 스프레드를 이용한 페어 트레이딩
#

# %%

# %% [markdown]
#
# ### 2) 공적분관계를 이용한 페어 트레이딩
#

# %%

# %% [markdown]
#
# ### 3) 칼만 필터를 이용한 페어 트레이딩
#

# %%

# %% [markdown]
#
# ## 8. 최소 총 이익(MTP) 경계 및 백테스트
# 최소 총 이익(MTP) 경계를 구하고 이를 이용한 백테스트 결과를 제시하라. (논문과 블로그 참조)
#

# %%
