# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="6KuW6t2MywG8" executionInfo={"status": "ok", "timestamp": 1698120037904, "user_tz": -540, "elapsed": 2518, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="bd7f995b-f81c-41f4-c35f-f30b214d316d"
from google.colab import drive
drive.mount('/content/drive')

# %% id="-kpdbP4hzTYP" executionInfo={"status": "ok", "timestamp": 1698120037904, "user_tz": -540, "elapsed": 4, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2023년_카이스트_금융시계열/1주차실습/1. 금융시계열 실습')

# %% id="uz3OQ2c_yswW" executionInfo={"status": "ok", "timestamp": 1698119396914, "user_tz": -540, "elapsed": 304, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# ARIMA 모형을 이용하여 주가을 예측해 본다.
# 예측 결과를 신뢰할 수 있는가? 없다면 그 원인은 무엇인가 ?

# %% id="d0XnAwp-yswa" executionInfo={"status": "ok", "timestamp": 1698120039592, "user_tz": -540, "elapsed": 1691, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')
#
# ------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MyUtil.MyTimeSeries import checkNormality
from statsmodels.tsa.arima.model import ARIMA

# %% id="vKTTgg7pyswb" executionInfo={"status": "ok", "timestamp": 1698119509587, "user_tz": -540, "elapsed": 280, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 주가 데이터를 읽어온다
p = pd.read_csv('StockData/069500.csv', index_col=0, parse_dates=True)[::-1]
p = p.dropna()

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} id="nth9G7bpyswc" executionInfo={"status": "ok", "timestamp": 1698119512634, "user_tz": -540, "elapsed": 1788, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="2ff4c110-518c-42aa-fc26-7b804657de79"
# 종가를 기준으로 일일 수익률을 계산한다.
p['Rtn'] = np.log(p['Close']) - np.log(p['Close'].shift(1))
p = p.dropna()

# 수익률 시계열을 육안으로 확인한다. 이분산성이 있는가?
plt.figure(figsize=(10,6))
plt.plot(p['Rtn'], color='red', linewidth=1)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="y3sDDhojyswc" executionInfo={"status": "ok", "timestamp": 1698119517021, "user_tz": -540, "elapsed": 1723, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="0fcdbb46-09ad-4b34-e993-759944b5a9e2"
# 주가 데이터를 ARIMA(2,1,1) 모형으로 분석한다 (Fitting)
y = np.array(pd.to_numeric(p['Close'], downcast='float'))  # int형이면 float형으로 변환한다
model = ARIMA(y, order=(2,1,1))
model_fit = model.fit()
print(model_fit.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 703} id="ryU9Ei1hyswd" executionInfo={"status": "ok", "timestamp": 1698119522378, "user_tz": -540, "elapsed": 1636, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="ed966201-60fe-4312-f410-5c56de9192eb"
# Fitting이 잘되었는지 확인하기 위해 Residual을 분석한다.
# Residual은 실제 데이터와 추정치의 차이이므로 백색 잡음 (잔차) 이어야 한다.
# 따라서 Residual은 정규분포 특성을 가져야한다. 정규분포 특성을 조사하면
# Fitting이 잘되었는지 확인할 수 있다.
residual = model_fit.resid
checkNormality(residual)  # 육안으로 백색 잡음 형태인지 확인한다

# %% colab={"base_uri": "https://localhost:8080/", "height": 519} id="ZzMvhqrRyswd" executionInfo={"status": "ok", "timestamp": 1698119528777, "user_tz": -540, "elapsed": 890, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="6ccbdd11-2d48-4104-d37c-d853034673bf"
# 향후 10 기간 데이터를 예측한다
forecast = model_fit.forecast(steps=10)[0]
forecast = np.r_[y[-1], forecast]  # y의 마지막 값을 forecast 앞 부분에 넣는다

# 원 시계열과 예측된 시계열을 그린다
ytail = y[len(y)-100:]   # 뒷 부분 100개만 그린다
ax1 = np.arange(1, len(ytail) + 1)
ax2 = np.arange(len(ytail), len(ytail) + len(forecast))
plt.figure(figsize=(10, 6))
plt.plot(ax1, ytail, 'b-o', markersize=3, color='blue', label='Stock Price', linewidth=1)
plt.plot(ax2, forecast, color='red', label='Forecast')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.title("Time Series Forcast")
plt.legend()
plt.show()

# %% [markdown] id="G9fVj86Uyswe"
# # 더 좋은 방법을 찾아보자!!!

# %% id="P-4rJXXVyswf"
