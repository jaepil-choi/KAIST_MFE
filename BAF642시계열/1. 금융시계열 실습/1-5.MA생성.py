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

# %% colab={"base_uri": "https://localhost:8080/"} id="2--gCmm_3Rka" executionInfo={"status": "ok", "timestamp": 1698120486838, "user_tz": -540, "elapsed": 39138, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="afc78cfc-2b18-4540-bdc6-0f544f6ae418"
from google.colab import drive
drive.mount('/content/drive')

# %% id="fECdXfQ03Rg3" executionInfo={"status": "ok", "timestamp": 1698120488190, "user_tz": -540, "elapsed": 1356, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2023년_카이스트_금융시계열/1주차실습/1. 금융시계열 실습')

# %% id="iBZOFngm3RBT" executionInfo={"status": "ok", "timestamp": 1698120488191, "user_tz": -540, "elapsed": 3, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# MA 모형을 임의로 생성하고, MA 파라메터가 변할 때 시계열의 모습이 어떻게
# 변하는지 육안으로 확인한다. MA 파라메터의 특성을 직관적으로 이해한다.
# 또한, MA 모형에 대한 ACF와 PACF 특성을 육안으로 확인한다.
# ACF/PACF는 향후 실제 시계열을 분석할 때 분석 모델을 선정에 참고한다.

# %% id="H2jFhdCu3RBV" executionInfo={"status": "ok", "timestamp": 1698120491022, "user_tz": -540, "elapsed": 2834, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
from MyUtil.MyTimeSeries import sampleARIMA
from statsmodels.tsa.arima.model  import ARIMA

# %% colab={"base_uri": "https://localhost:8080/", "height": 707} id="nj_naCub3RBV" executionInfo={"status": "ok", "timestamp": 1698120493329, "user_tz": -540, "elapsed": 2311, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="56e86368-fb43-4a4d-c345-b7354b84346f"
# MA(1) 모형의 parameter 변화에 따른 그래프 모양을 확인한다.
# a 값을 변화시켜 가면서 그래프의 모양이 어떻게 바뀌는지 확인한다.
y1 = sampleARIMA(ar=[0], d=0, ma=[0.1], n=500)
y2 = sampleARIMA(ar=[0], d=0, ma=[0.5], n=500)
y3 = sampleARIMA(ar=[0], d=0, ma=[0.9], n=500)
y4 = sampleARIMA(ar=[0], d=0, ma=[0.99], n=500)

fig = plt.figure(figsize=(10, 7))
p1 = fig.add_subplot(2,2,1)
p2 = fig.add_subplot(2,2,2)
p3 = fig.add_subplot(2,2,3)
p4 = fig.add_subplot(2,2,4)

p1.plot(y1, color='blue', linewidth=1)
p2.plot(y2, color='red', linewidth=1)
p3.plot(y3, color='purple', linewidth=1)
p4.plot(y4, color='green', linewidth=1)
p1.set_title("b = 0.1")
p2.set_title("b = 0.5")
p3.set_title("b = 0.9")
p4.set_title("b = 0.99")
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="-cB7AfOL3RBW" executionInfo={"status": "ok", "timestamp": 1698120493330, "user_tz": -540, "elapsed": 14, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="a355b4e6-0bae-4a1e-f4f7-004598d1291a"
# 임의로 생성한 MA(1) 샘플 데이터를 분석하여 b 값을 추정해 본다.
# 생성할 때 지정한 값으로 추정이 잘되는지 확인한다.
y = sampleARIMA(ar=[0], d=0, ma=[0.5], n=500)
model = ARIMA(y, order=(0,0,1)).fit()
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 864} id="pHgMu4hz3RBX" executionInfo={"status": "ok", "timestamp": 1698120494084, "user_tz": -540, "elapsed": 762, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="32f0e245-c14f-41f0-e48a-8e69bc9e71dc"
# MA(2) 모형도 확인해 본다
y = sampleARIMA(ar=[0], d=0, ma=[0.1, -0.4], n=500)
plt.plot(y, color='blue', linewidth=1)
model = ARIMA(y, order=(0,0,2)).fit()
print(model.summary())

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="7kG0R_Gv3RBX" executionInfo={"status": "ok", "timestamp": 1698120494709, "user_tz": -540, "elapsed": 631, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="14c4a969-ede6-4281-b21b-f6c30ca8654a"
# MA 모형의 ACF와 PACF를 확인해 본다. a값을 변화시켜 가면서 비교해 본다
# ACF와 PACF는 향후 실제 시계열을 어느 모형으로 분석할 지에 대한 단서를 제공한다
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
y = sampleARIMA(ar=[0], d=0, ma=[0.5], n=500)
fig = plt.figure(figsize=(12, 3))
plt.plot(y, color='red', linewidth=1)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 391} id="4PJbpKV43RBX" executionInfo={"status": "ok", "timestamp": 1698120495357, "user_tz": -540, "elapsed": 654, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="84b3dc24-d0e0-4087-c445-5482ea6e2d35"
fig = plt.figure(figsize=(12, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)
plot_acf(y, p1, lags=50)
plot_pacf(y, p2, lags=50)
plt.show()


# %% id="PPFVumou3RBX" executionInfo={"status": "ok", "timestamp": 1698120495357, "user_tz": -540, "elapsed": 7, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
