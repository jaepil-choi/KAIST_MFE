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

# %% colab={"base_uri": "https://localhost:8080/"} id="bN0lBcgh4XkT" executionInfo={"status": "ok", "timestamp": 1698120773960, "user_tz": -540, "elapsed": 38933, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="9eefab3a-777e-4301-b24d-76bd6d842643"
from google.colab import drive
drive.mount('/content/drive')

# %% id="iRLxDK-G4aQV" executionInfo={"status": "ok", "timestamp": 1698120774556, "user_tz": -540, "elapsed": 601, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2023년_카이스트_금융시계열/1주차실습/1. 금융시계열 실습')

# %% id="l--Hdx0T4L7W" executionInfo={"status": "ok", "timestamp": 1698120774557, "user_tz": -540, "elapsed": 4, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# ARMA 모형을 임의로 생성하고, ARMA 파라메터가 변할 때 시계열의 모습이 어떻게
# 변하는지 육안으로 확인한다. ARMA 파라메터의 특성을 직관적으로 이해한다.
# 또한, ARMA 모형에 대한 ACF와 PACF 특성을 육안으로 확인한다.
# ACF/PACF는 향후 실제 시계열을 분석할 때 분석 모델을 선정에 참고한다.

# %% id="QhRRwtaH4L7Y" executionInfo={"status": "ok", "timestamp": 1698120818458, "user_tz": -540, "elapsed": 552, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')
# ------------------------------------------------------------------------
import matplotlib.pyplot as plt
from MyUtil.MyTimeSeries import sampleARIMA
from statsmodels.tsa.arima.model import ARIMA

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="bYgjYICr4L7Z" executionInfo={"status": "ok", "timestamp": 1698120820028, "user_tz": -540, "elapsed": 1573, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="31de842f-2ee0-4544-a2cd-4bd83f7a5901"
# MA 모형의 parameter 변화에 따른 그래프 모양을 확인한다.
# a 값을 변화시켜 가면서 그래프의 모양이 어떻게 바뀌는지 확인한다.
y1 = sampleARIMA(ar=[0.9], d=0, ma=[0.5], n=500)
y2 = sampleARIMA(ar=[0.9], d=0, ma=[0.5, 0.9], n=500)

fig = plt.figure(figsize=(10, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)

p1.plot(y1, color='blue', linewidth=1)
p2.plot(y2, color='red', linewidth=1)
p1.set_title("ARMA(1,1) : a=0.9, b = 0.5")
p2.set_title("ARMA(1,2) : a=0.9, b = 0.5, 0.9")
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="_l0NLFXM4L7a" executionInfo={"status": "ok", "timestamp": 1698120820593, "user_tz": -540, "elapsed": 573, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="80d0f05b-bf38-4920-8b90-78b82e21f399"
# 임의로 생성한 ARMA(1,1) 샘플 데이터를 분석하여 b 값을 추정해 본다.
# 생성할 때 지정한 값으로 추정이 잘되는지 확인한다.
y = sampleARIMA(ar=[0.9], d=0, ma=[0.5], n=500)
model = ARIMA(y, order=(1,0,1)).fit()
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 881} id="lDh6KtN14L7a" executionInfo={"status": "ok", "timestamp": 1698120822200, "user_tz": -540, "elapsed": 1612, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="da042163-3fa3-4c92-c163-f378cd581a97"
# ARMA(1,2) 모형도 확인해 본다
y = sampleARIMA(ar=[0.5], d=0, ma=[0.5, -0.1], n=1000)
plt.plot(y, color='blue', linewidth=1)
model = ARIMA(y, order=(1,0,2)).fit()
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 665} id="lTc6EZPy4L7a" executionInfo={"status": "ok", "timestamp": 1698120823526, "user_tz": -540, "elapsed": 1332, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="56f77538-e9e2-4257-b014-124e1f7c94f0"
# ARMA 모형의 ACF와 PACF를 확인해 본다. a,b 값을 변화시켜 가면서 비교해 본다
# ACF와 PACF는 향후 실제 시계열을 어느 모형으로 분석할 지에 대한 단서를 제공한다
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
y = sampleARIMA(ar=[0.9], d=0, ma=[0.5, -0.1], n=500)
fig = plt.figure(figsize=(12, 3))
plt.plot(y, color='red', linewidth=1)
plt.show()
fig = plt.figure(figsize=(12, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)
plot_acf(y, p1, lags=50)
plot_pacf(y, p2, lags=50)
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 665} id="iJ7B-FZn4L7a" executionInfo={"status": "ok", "timestamp": 1698120825244, "user_tz": -540, "elapsed": 1722, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="fa9c4460-44c2-49aa-d8bf-9a76b6512e4c"
# ARMA 모형의 ACF와 PACF를 확인해 본다. a,b 값을 변화시켜 가면서 비교해 본다
# ACF와 PACF는 향후 실제 시계열을 어느 모형으로 분석할 지에 대한 단서를 제공한다
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
y = sampleARIMA(ar=[0.5], d=0, ma=[0.1], n=500)
fig = plt.figure(figsize=(12, 3))
plt.plot(y, color='red', linewidth=1)
plt.show()
fig = plt.figure(figsize=(12, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)
plot_acf(y, p1, lags=50)
plot_pacf(y, p2, lags=50)
plt.show()

# %% id="XIYyblql4L7b" executionInfo={"status": "ok", "timestamp": 1698120825245, "user_tz": -540, "elapsed": 6, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}

# %% id="blb3jv4W4L7b" executionInfo={"status": "ok", "timestamp": 1698120825245, "user_tz": -540, "elapsed": 6, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
