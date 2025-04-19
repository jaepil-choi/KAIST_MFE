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

# %% colab={"base_uri": "https://localhost:8080/"} id="V8ZBh9mf57u-" executionInfo={"status": "ok", "timestamp": 1698121179778, "user_tz": -540, "elapsed": 37667, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="42507aeb-4d67-4979-9eb8-ea0b8a6799bb"
from google.colab import drive
drive.mount('/content/drive')

# %% id="YiwCvn7957ks" executionInfo={"status": "ok", "timestamp": 1698121181306, "user_tz": -540, "elapsed": 1532, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2023년_카이스트_금융시계열/1주차실습/1. 금융시계열 실습')

# %% id="Z_4Ee6Fk5Azr"
# ARIMA 모형을 임의로 생성하고, 향후 데이터를 예측한다.

# %% id="4xwqHx1N5Azs" executionInfo={"status": "ok", "timestamp": 1698121184354, "user_tz": -540, "elapsed": 3052, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from MyUtil.MyTimeSeries import sampleARIMA, checkNormality
from statsmodels.tsa.arima.model import ARIMA

# %% colab={"base_uri": "https://localhost:8080/", "height": 468} id="njuCXP1R5Azt" executionInfo={"status": "ok", "timestamp": 1698121185198, "user_tz": -540, "elapsed": 847, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="6e8e0fcc-6721-43dd-f759-1d366d510b91"
# ARIMA(1,1,1) 샘플을 생성한다
y = sampleARIMA(ar=[0.8], d=1, ma=[0.5], n=500)
d = np.diff(y) # 차분하면 ARMA(1,1)이 된다

# ARIMA 시계열과 차분 시계열을 그린다
fig = plt.figure(figsize=(12, 5))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)
p1.plot(y, color='blue', linewidth=1)
p2.plot(d, color='red', linewidth=1)
p1.set_title("ARIMA(1,1,1)")
p2.set_title("ARMA(1,1)")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="vTW8C0K35Azu" executionInfo={"status": "ok", "timestamp": 1698121185673, "user_tz": -540, "elapsed": 481, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="7de22325-5429-4714-e9f9-9a9745f64816"
# 샘플링 데이터를 ARIMA(1,1,1) 모형으로 분석한다 (Fitting)
y = sampleARIMA(ar=[0.8], d=1, ma=[0.5], n=500)
model = ARIMA(y, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 753} id="XigFllrE5Azu" executionInfo={"status": "ok", "timestamp": 1698121187120, "user_tz": -540, "elapsed": 1453, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="a2f9c86d-46d0-4bd7-a081-0b94fc3f9542"
# Fitting이 잘되었는지 확인하기 위해 Residual을 분석한다.
# Residual은 실제 데이터와 추정치의 차이이므로 백색 잡음 (잔차) 이어야 한다.
# 따라서 Residual은 정규분포 특성을 가져야한다. 정규분포 특성을 조사하면
# Fitting이 잘되었는지 확인할 수 있다.
residual = model_fit.resid
checkNormality(residual)  # 육안으로 백색 잡음 형태인지 확인한다

# %% colab={"base_uri": "https://localhost:8080/", "height": 545} id="yd_JxEIt5Azv" executionInfo={"status": "ok", "timestamp": 1698121187120, "user_tz": -540, "elapsed": 9, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="5964e3e8-33e3-4217-fc81-79e09c4e21ff"
# 향후 10 기간 데이터를 예측한다
forecast = model_fit.forecast(steps=10)[0]
forecast = np.r_[y[-1], forecast]  # y의 마지막 값을 forecast 앞 부분에 넣는다

# 원 시계열과 예측된 시계열을 그린다
ytail = y[len(y)-100:]   # 뒷 부분 100개만 그린다
ax1 = np.arange(1, len(ytail) + 1)
ax2 = np.arange(len(ytail), len(ytail) + len(forecast))
plt.figure(figsize=(10, 6))
plt.plot(ax1, ytail, 'b-o', markersize=3, color='blue', label='Time series', linewidth=1)
plt.plot(ax2, forecast, color='red', label='Forecast')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.title("Time Series Forcast")
plt.legend()
plt.show()

# %% id="uOn3Ur4P5Azv" executionInfo={"status": "ok", "timestamp": 1698121187121, "user_tz": -540, "elapsed": 8, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
