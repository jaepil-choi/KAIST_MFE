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

# %% [markdown] id="xer-0MC30LIU"
# # 금융 시계열 특성을 알아보자

# %% colab={"base_uri": "https://localhost:8080/"} id="OAoOLGlr1-_e" executionInfo={"status": "ok", "timestamp": 1730278264345, "user_tz": -540, "elapsed": 5928, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="e31636a6-cf11-42ec-a700-dd65542b5b6e"
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf

# %% [markdown] id="OYuFzwskiYun"
# # 세선 다시 시작

# %% colab={"base_uri": "https://localhost:8080/"} id="1Zj08oDZ0QLk" executionInfo={"status": "ok", "timestamp": 1730278271251, "user_tz": -540, "elapsed": 6910, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="983e3683-1ea2-41a2-d92c-3f4a3108e9a4"
from google.colab import drive
drive.mount('/content/drive')

# %% id="_refmnX10Uc_" executionInfo={"status": "ok", "timestamp": 1730278271251, "user_tz": -540, "elapsed": 26, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir('/content/drive/MyDrive/2024년 고급금융시계열 공유/1. 금융시계열 실습')

# %% id="8WEPQUet0LIX" executionInfo={"status": "ok", "timestamp": 1730278271252, "user_tz": -540, "elapsed": 27, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')

# %% id="3rXdizjN0LIY" executionInfo={"status": "ok", "timestamp": 1730278271252, "user_tz": -540, "elapsed": 26, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 기본 라이브러리 불러오기
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MyUtil import TaFeatureSet

# %% id="WeE6lzmX0LIZ" executionInfo={"status": "ok", "timestamp": 1730278271252, "user_tz": -540, "elapsed": 26, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 한글 폰트를 위해
plt.rc('font', family='NanumBarunGothic')
plt.rc('axes', unicode_minus=False)

# %% [markdown] id="67qKYvDH0LIZ"
# # 정상성 대 비정상성

# %% [markdown] id="tHw19WQH0LIZ"
# # 백색잡음과 랜덤워크

# %% id="n7rCPRVa0LIa" executionInfo={"status": "ok", "timestamp": 1730278271252, "user_tz": -540, "elapsed": 25, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 백색잡음 1000개를 만들어라
eps=np.random.randn(1000)

# %% id="kdE4WanI0LIa" executionInfo={"status": "ok", "timestamp": 1730278271253, "user_tz": -540, "elapsed": 26, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 이를 이용해 랜덤워크를 만들어라 (초기값은 1로 놓아라)
y0=1
yt=np.cumsum(np.append(y0, eps))

# %% colab={"base_uri": "https://localhost:8080/", "height": 484} id="lN7wnw5B0LIa" executionInfo={"status": "ok", "timestamp": 1730278271253, "user_tz": -540, "elapsed": 25, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="a97dd4ea-5e48-430e-c577-579d6d6bb646"
# 백색잡음과 랜덤워크 시계열을 그려라
fig=plt.figure(figsize=(12,5))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

ax1.plot(eps)
ax1.set_title('백색잡음')
ax2.plot(yt)
ax2.set_title('랜덤워크')

# %% [markdown] id="ljhSR_IB0LIb"
# # 금융시계열(주가수익률)과 랜덤워크(백색잡음)

# %% id="T3oyTHGX0LIc" executionInfo={"status": "ok", "timestamp": 1730278271253, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# Yahoo 사이트에서 주가 데이터를 수집하여 주가, 거래량, 수익률, MACD 지표를
# 관찰하고, 비정상 시계열 (Non-stationary)과 정상 시계열 (Stationary)의
# 차이점을 관찰한다.

# %% id="5u5_VmU20LIc" executionInfo={"status": "ok", "timestamp": 1730278271253, "user_tz": -540, "elapsed": 15, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 주가 데이터를 읽어온다
df = pd.read_csv('StockData/069500.csv', index_col=0, parse_dates=True)[::-1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 237} id="murhEMXo0LId" executionInfo={"status": "ok", "timestamp": 1730278271253, "user_tz": -540, "elapsed": 14, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="87535ef4-66aa-42fc-ef2d-33c9404f8602"
df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 422} id="XTxMuk750LIe" executionInfo={"status": "ok", "timestamp": 1730278272321, "user_tz": -540, "elapsed": 1081, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="19c56c13-be1f-436c-dbe6-b5227cacceaf"
# 종가를 기준으로 그래프를 그린다.
sse = df['Close'].plot()

# %% id="WwLtAHjw0LIe" executionInfo={"status": "ok", "timestamp": 1730278272321, "user_tz": -540, "elapsed": 28, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 종가를 기준으로 일일 수익률을 계산한다.
sse = np.log(df['Close']) - np.log(df['Close'].shift(1))

# %% colab={"base_uri": "https://localhost:8080/", "height": 438} id="xU65KR3t0LIe" executionInfo={"status": "ok", "timestamp": 1730278272321, "user_tz": -540, "elapsed": 27, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="19d9da73-877c-4241-cd9b-d9e6e2504645"
sse.plot()

# %% id="chrwX6Mf0LIe" executionInfo={"status": "ok", "timestamp": 1730278272321, "user_tz": -540, "elapsed": 24, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 2 표준편차로 상하한선을 설정
ll= sse.mean()-2.*sse.std()
ul=sse.mean()+2.*sse.std()

# %% id="Etb1O_6u0LIe" executionInfo={"status": "ok", "timestamp": 1730278272322, "user_tz": -540, "elapsed": 25, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
rw=np.random.normal(loc=np.mean(sse), scale=np.std(sse), size=len(sse))
rw=pd.Series(rw, index=sse.index)

# %% id="Kk3XfyQg0LIe" executionInfo={"status": "ok", "timestamp": 1730278272322, "user_tz": -540, "elapsed": 24, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 랜덤워크(백색잡음)를 그려보자

# %% colab={"base_uri": "https://localhost:8080/", "height": 540} id="COqG9mmz0LIf" executionInfo={"status": "ok", "timestamp": 1730278272322, "user_tz": -540, "elapsed": 24, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="12482be3-1c1b-4904-e417-f24588207efe"
#sse.plot(figsize=(12, 6), alpha=0.5)
rw.plot(figsize=(12, 6), label='백색잡음', c='r', alpha=0.3)
plt.axhline(ll, c='r', ls='--')
plt.axhline(ul, c='r', ls='--')
plt.title("백색잡음")
plt.legend()

# %% [markdown] id="3nUfnYpF0LIf"
# # 주식의 수익률을 그려보자

# %% colab={"base_uri": "https://localhost:8080/", "height": 540} id="fIVUF7VS0LIf" executionInfo={"status": "ok", "timestamp": 1730278273412, "user_tz": -540, "elapsed": 1107, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="3add480e-9735-444a-8af2-3921fce54d38"
sse.plot(figsize=(12, 6), alpha=0.5)
#rw.plot(label='Random Walk', c='r', alpha=0.3)
plt.axhline(ll, c='r', ls='--')
plt.axhline(ul, c='r', ls='--')
plt.title("SEC 수익률")
plt.legend()

# %% [markdown] id="7FMR1wn80LIf"
# # 주식의 수익률과 랜덤워크를 같이 그려보자

# %% colab={"base_uri": "https://localhost:8080/", "height": 540} id="3VhK3tlZ0LIf" executionInfo={"status": "ok", "timestamp": 1730278273412, "user_tz": -540, "elapsed": 12, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="7197eae7-c1dd-495d-8d9e-ab59ec43d424"
sse.plot(figsize=(12, 6), alpha=0.5)
rw.plot(label='Random Walk', c='r', alpha=0.3)
plt.axhline(ll, c='r', ls='--')
plt.axhline(ul, c='r', ls='--')
plt.title("SEC 수익률 대 랜덤워크")
plt.legend()

# %% [markdown] id="-2XfZ5510LIg"
# # 비정상성과 정상성 시계열의 예

# %% colab={"base_uri": "https://localhost:8080/", "height": 607} id="QZctylwe0LIg" executionInfo={"status": "ok", "timestamp": 1730278274632, "user_tz": -540, "elapsed": 1228, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="d12beeea-c5b3-4809-bc8c-fa1fb12ceeef"
# 종가를 기준으로 일일 수익률을 계산한다.
df['Rtn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

# MACD 기술적 지표를 측정한다
df['macd'] = TaFeatureSet.MACD(df)
df = df.dropna()

# 주가, 거래량, 수익률, MACD를 그린다
fig = plt.figure(figsize=(10, 6))
p1 = fig.add_subplot(2,2,1)
p2 = fig.add_subplot(2,2,2)
p3 = fig.add_subplot(2,2,3)
p4 = fig.add_subplot(2,2,4)

p1.plot(df['Close'], color='blue', linewidth=1)
p2.plot(df['Volume'], color='red', linewidth=1)
p3.plot(df['Rtn'], color='purple', linewidth=1)
p4.plot(df['macd'], color='green', linewidth=1)
p1.set_title("Stock Price")
p2.set_title("Volume")
p3.set_title("Return")
p4.set_title("MACD oscilator")
p3.axhline(y=0, color='black', linewidth=1)
p4.axhline(y=0, color='black', linewidth=1)
plt.tight_layout()
plt.show()

# %% id="m549rrWh0LIg" executionInfo={"status": "ok", "timestamp": 1730278274632, "user_tz": -540, "elapsed": 7, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# df = pd.read_csv("jj.csv")

# %% id="aX54x41Q0LIg" executionInfo={"status": "ok", "timestamp": 1730278274633, "user_tz": -540, "elapsed": 8, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
