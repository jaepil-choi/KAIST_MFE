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

# %% colab={"base_uri": "https://localhost:8080/"} id="tm-FhHai2zWl" executionInfo={"status": "ok", "timestamp": 1698120339325, "user_tz": -540, "elapsed": 35645, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="e8ff05f3-6231-4094-ba75-3a8388b88f44"
from google.colab import drive
drive.mount('/content/drive')

# %% id="r8H4wEFP2zTG" executionInfo={"status": "ok", "timestamp": 1698120339772, "user_tz": -540, "elapsed": 450, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2023년_카이스트_금융시계열/1주차실습/1. 금융시계열 실습')

# %% id="PTh1rZfW2VG5" executionInfo={"status": "ok", "timestamp": 1698120339772, "user_tz": -540, "elapsed": 3, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# AR 모형을 임의로 생성하고, AR 파라메터가 변할 때 시계열의 모습이 어떻게
# 변하는지 육안으로 확인한다. AR 파라메터의 특성을 직관적으로 이해한다.
# 또한, AR 모형에 대한 ACF와 PACF 특성을 육안으로 확인한다.
# ACF/PACF는 향후 실제 시계열을 분석할 때 분석 모델을 선정에 참고한다.

# %% id="m7FDtuY42VG7" executionInfo={"status": "ok", "timestamp": 1698120377431, "user_tz": -540, "elapsed": 319, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
from MyUtil.MyTimeSeries import sampleARIMA
from statsmodels.tsa.arima.model import ARIMA

# %% colab={"base_uri": "https://localhost:8080/", "height": 707} id="TiIza9sK2VG8" executionInfo={"status": "ok", "timestamp": 1698120381963, "user_tz": -540, "elapsed": 4228, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="5005ecb0-8e50-4263-c0c8-7311e2f71a51"
# AR(1) 모형의 parameter 변화에 따른 그래프 모양을 확인한다.
# a 값을 변화시켜 가면서 그래프의 모양이 어떻게 바뀌는지 확인한다.
y1 = sampleARIMA(ar=[0.1], d=0, ma=[0], n=500)
y2 = sampleARIMA(ar=[0.5], d=0, ma=[0], n=500)
y3 = sampleARIMA(ar=[0.9], d=0, ma=[0], n=500)
y4 = sampleARIMA(ar=[0.99], d=0, ma=[0], n=500)

fig = plt.figure(figsize=(10, 7))
p1 = fig.add_subplot(2,2,1)
p2 = fig.add_subplot(2,2,2)
p3 = fig.add_subplot(2,2,3)
p4 = fig.add_subplot(2,2,4)

p1.plot(y1, color='blue', linewidth=1)
p2.plot(y2, color='red', linewidth=1)
p3.plot(y3, color='purple', linewidth=1)
p4.plot(y4, color='green', linewidth=1)
p1.set_title("a = 0.1")
p2.set_title("a = 0.5")
p3.set_title("a = 0.9")
p4.set_title("a = 0.99")
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="kP7cjA0T2VG9" executionInfo={"status": "ok", "timestamp": 1698120381964, "user_tz": -540, "elapsed": 10, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="fa9087c7-e1ac-4a5a-be17-11babbf210d8"
# 임의로 생성한 AR(1) 샘플 데이터를 분석하여 a 값을 추정해 본다.
# 생성할 때 지정한 값으로 추정이 잘되는지 확인한다.
y = sampleARIMA(ar=[0.5], d=0, ma=[0], n=500)
model = ARIMA(y, order=(1,0,0)).fit()
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 864} id="L0NlGu8C2VG9" executionInfo={"status": "ok", "timestamp": 1698120382480, "user_tz": -540, "elapsed": 522, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="519fa1f4-3114-4484-9bb3-a9f4d395f9d4"
# AR(2) 모형도 확인해 본다
y = sampleARIMA(ar=[0.1, -0.4], d=0, ma=[0], n=500)
plt.plot(y, color='blue', linewidth=1)
model = ARIMA(y, order=(2,0,0)).fit()
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="B1jA8ZGk2VG-" executionInfo={"status": "ok", "timestamp": 1698120383085, "user_tz": -540, "elapsed": 614, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="daad5aea-cc4a-48b6-a1da-8b8436f2893b"
# AR 모형의 ACF와 PACD를 확인해 본다. a값을 변화시켜 가면서 비교해 본다
# ACF와 PACF는 향후 실제 시계열을 어느 모형으로 분석할 지에 대한 단서를 제공한다
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
y = sampleARIMA(ar=[0.9], d=0, ma=[0], n=500)
fig = plt.figure(figsize=(12, 3))
plt.plot(y, color='red', linewidth=1)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="jEPbeLI72VG-" executionInfo={"status": "ok", "timestamp": 1698120383975, "user_tz": -540, "elapsed": 901, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="8bdc7a1f-0f19-4454-9ab2-7f5dede78c0f"
fig = plt.figure(figsize=(12, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)
plot_acf(y, p1, lags=50)
plot_pacf(y, p2, lags=50)
plt.show()


# %% id="m8WHGYNh2VG-" executionInfo={"status": "ok", "timestamp": 1698120383975, "user_tz": -540, "elapsed": 11, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
