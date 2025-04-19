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

# %% colab={"base_uri": "https://localhost:8080/"} id="vbIdD3LW3zU9" executionInfo={"status": "ok", "timestamp": 1698120603547, "user_tz": -540, "elapsed": 34630, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="ee3fa142-a2fc-4c7b-ba7c-f58f7c6dbf3e"
from google.colab import drive
drive.mount('/content/drive')

# %% id="TcDaAnl_3zRM" executionInfo={"status": "ok", "timestamp": 1698120606270, "user_tz": -540, "elapsed": 452, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2023년_카이스트_금융시계열/1주차실습/1. 금융시계열 실습')

# %% id="xiKDMmMz3rcu"
# AR 모형을 임의로 생성하고, 향후 데이터를 예측한다.

# %% id="Yj-3Psbd3rcw" executionInfo={"status": "ok", "timestamp": 1698120609998, "user_tz": -540, "elapsed": 2359, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')
# -----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from MyUtil.MyTimeSeries import sampleARIMA
from statsmodels.tsa.arima.model import ARIMA

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="AqB1zXNs3rcx" executionInfo={"status": "ok", "timestamp": 1698120614069, "user_tz": -540, "elapsed": 676, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="4e315c1a-63ae-4004-8511-45e79b018994"
# AR(1) 샘플을 생성한다
y = sampleARIMA(ar=[0.8], d=0, ma=[0], n=500)
plt.figure(figsize=(10, 3))
plt.plot(y, color='brown', linewidth=1)
plt.show()

# %% id="pDz32hV73rcy" executionInfo={"status": "ok", "timestamp": 1698120617198, "user_tz": -540, "elapsed": 306, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 샘플링 데이터를 AR(1) 모형으로 분석한다 (Fitting)
model = ARIMA(y, order=(1,0,0))
model_fit = model.fit()

# %% id="XmAtRbjh3rcz" executionInfo={"status": "ok", "timestamp": 1698120619718, "user_tz": -540, "elapsed": 2, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 두 배열을 오른쪽 왼쪽으로 붙이기
#    : np.r_[a, b]
#    : np.hstack([a, b])
#    : np.concatenate((a, b), axis = 0)

# %% id="uiXKmtR-3rcz" executionInfo={"status": "ok", "timestamp": 1698120624929, "user_tz": -540, "elapsed": 328, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 향후 10 기간 데이터를 예측한다
forecast = model_fit.forecast(steps=10)[0]
forecast = np.r_[y[-1], forecast]  # y의 마지막 값을 forecast 앞 부분에 넣는다

# %% colab={"base_uri": "https://localhost:8080/", "height": 352} id="OEYOLXGu3rcz" executionInfo={"status": "ok", "timestamp": 1698120632801, "user_tz": -540, "elapsed": 1207, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="0d460721-1a33-42b1-fd93-bf88219a9017"
# 원 시계열과 예측된 시계열을 그린다
ytail = y[len(y)-100:]   # 뒷 부분 100개만 그린다
ax1 = np.arange(1, len(ytail) + 1)
ax2 = np.arange(len(ytail), len(ytail) + len(forecast))
plt.figure(figsize=(10, 3.5))
plt.plot(ax1, ytail, color='blue', label='Time series', linewidth=1)
plt.plot(ax2, forecast, color='red', label='Forecast')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.title("Time Series Forcast")
plt.legend()
plt.show()

# %% id="QroaghR-3rcz"
