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

# %% colab={"base_uri": "https://localhost:8080/"} id="zBzN2_D-0ixp" executionInfo={"status": "ok", "timestamp": 1727060635272, "user_tz": -540, "elapsed": 2603, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="1fc9abdf-1599-4419-e9d3-946f112caf22"
from google.colab import drive
drive.mount('/content/drive')

# %% id="KXeUb5K10lVR" executionInfo={"status": "ok", "timestamp": 1727060708372, "user_tz": -540, "elapsed": 513, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir("/content/drive/MyDrive/2024년 카이스트 고급 금융 계량/2024_BAF634고금계과제1_데이터")

# %% [markdown] id="qjrwj-PJ0hpa"
# # 고금계 과제 1 데이터 사용법
#
# - 과제1 수행에 필요한 데이터를 불러오는 방법을 알아봅니다
#

# %% id="swNf2vso0hpd" executionInfo={"status": "ok", "timestamp": 1727060711952, "user_tz": -540, "elapsed": 607, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# %% [markdown] id="0we4N4Pj0hpe"
# ## 경로 설정
#
# - 과제 데이터 파일의 경로를 설정합니다.
# - 주피터노트북이 있는 폴더의 `data/` 안에 데이터를 두는 것을 권장합니다.

# %% id="9c6bNNi00hpf" executionInfo={"status": "ok", "timestamp": 1727060732677, "user_tz": -540, "elapsed": 453, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %% id="b6_qv5Qj0hpf" executionInfo={"status": "ok", "timestamp": 1727060735740, "user_tz": -540, "elapsed": 304, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fndata_path = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'

# %% [markdown] vscode={"languageId": "plaintext"} id="xy45Q4FS0hpf"
# ## 데이터 불러오기
#
# - 주식
#     - 데이터 기간: 2013-01 ~ 2024-08
#     - 기초 유니버스: KSE(코스피) + KOSDAQ 전체
#     - 기본적인 전처리가 되어있습니다.
#         - 생존편향 제거됨
#         - 데이터 기간 내 존재하지 않은 기업 (2013-01 이전 상장폐지) 제거됨
#         - 월말일 기준 관리종목/거래정지 종목 제거됨
#         - 모든 금액은 '원'단위 (천원 아님)
#         - 모든 %는 1.0 == 100%
#         - 금융 업종 제거됨
#         - 월말일 기준 1개월 수익률이 없는 종목 제거
#         - 날짜 str --> datetime 변환
#     - 다양한 포맷으로 데이터 호출
#         - long-format
#             - 날짜-종목코드를 multi-index로, 여러 항목들(수익률, 이익잉여금 등)을 컬럼으로 하여 한 번에 불러올 수 있습니다.
#         - wide-format
#             - 한 개의 항목을 index는 날짜 columns는 종목코드로 하여 불러올 수 있습니다.
# - 시장수익률
# - 무위험 이자율

# %% [markdown] id="kEFM4GZR0hpg"
# ### 주식

# %% [markdown] id="3DppLXZ50hpg"
# #### 기본 사용법

# %% id="TmsG565S0hpg" executionInfo={"status": "ok", "timestamp": 1727060741232, "user_tz": -540, "elapsed": 811, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
from fndata import FnStockData

# %% id="HKKwmb0k0hph" executionInfo={"status": "ok", "timestamp": 1727060784884, "user_tz": -540, "elapsed": 41327, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다.
fn = FnStockData(fndata_path)

# %% id="Ombs920C0hph" outputId="95cef77e-c4fa-4926-ff00-d6e96aa6a043" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727060788297, "user_tz": -540, "elapsed": 438, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 사용 가능한 데이터를 확인합니다.
fn.get_items()

# %% id="stIF3mqs0hpi" outputId="0ebf0f7a-5449-4a4f-e348-e9ed01ba9493" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727060791191, "user_tz": -540, "elapsed": 439, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 분석 기간의 데이터 유니버스를 확인합니다. (금융업종, 거래정지, 관리종목 제외)
univ_list = fn.get_universe()
univ_list

# %% id="hg3yA-UA0hpi" outputId="d401a73b-1914-4946-fe81-1c27ce62a14b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727060794865, "user_tz": -540, "elapsed": 464, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
len(univ_list)

# %% id="X2bw37xE0hpj" outputId="3742459c-aa71-4752-d669-f4db77ed5db4" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1727060796959, "user_tz": -540, "elapsed": 440, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 이름으로 종목코드를 확인합니다.
fn.name_to_symbol('삼성전자')

# %% id="EFcYNHY70hpj" outputId="79e8b385-1a8e-467b-959b-4e40eb837ce2" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1727060799662, "user_tz": -540, "elapsed": 475, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 종목코드로 이름을 확인합니다.
fn.symbol_to_name('A005930')

# %% [markdown] id="Gn1NErMA0hpk"
# #### long-format으로 불러오기

# %% id="Ts9necdB0hpk" outputId="64c29aa4-26f9-44d5-83a3-9731cad5b6d9" colab={"base_uri": "https://localhost:8080/", "height": 455} executionInfo={"status": "ok", "timestamp": 1727060803681, "user_tz": -540, "elapsed": 326, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 원하는 데이터들을 long-format으로 불러옵니다.

my_data = ['수정주가(원)', '수익률 (1개월)(%)']
df = fn.get_data(my_data) # list가 들어가면 long-format으로 불러옵니다.
df

# %% id="HYWLOp5m0hpl" outputId="fad2af72-8410-439c-94de-1c9d0dc9d9da" colab={"base_uri": "https://localhost:8080/", "height": 735} executionInfo={"status": "ok", "timestamp": 1727060811924, "user_tz": -540, "elapsed": 434, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 모든 데이터를 불러옵니다.

df = fn.get_data()
df

# %% [markdown] id="fXH73yh10hpl"
# #### wide-format으로 불러오기

# %% id="fNCB9yrS0hpl" outputId="b9054b7e-eb08-4d8c-ec51-95c6e2cd900b" colab={"base_uri": "https://localhost:8080/", "height": 648} executionInfo={"status": "ok", "timestamp": 1727060816792, "user_tz": -540, "elapsed": 574, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
my_data = '수익률 (1개월)(%)'
fn.get_data(my_data) # string이 들어가면 wide-format으로 불러옵니다.

# %% [markdown] id="V3SLyW3D0hpl"
# ### 시장수익률
#
# - 기초 유니버스로 KSE+KOSDAQ을 썼기 때문에 팩터 분석을 위해 이에 대응하는 시장수익률을 쓰는 것이 좋습니다.
#     - (기본) [MKF2000](https://www.fnindex.co.kr/multi/detail?menu_type=0&fund_cd=FI00)
#     - (보조) [KRX300](https://ko.wikipedia.org/wiki/KRX_300)
#     - 두 지수 모두 코스피+코스닥을 기초로 시가총액, 거래대금 등을 고려하여 상위 2000/300 종목을 선정해 지수를 만듭니다.
#     - 두 지수 모두 기본적으로 시가총액 가중 방식으로 지수를 산출합니다.
#     - 파마 프랜치의 경우 NYSE, AMEX, NASDAQ에 상장된 모든 주식의 시가총액 가중 평균을 사용하였습니다. ([링크](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html))

# %% [markdown] id="AltvIFk10hpm"
# #### 기본 사용법

# %% id="X-BNGdXw0hpm" executionInfo={"status": "ok", "timestamp": 1727060834834, "user_tz": -540, "elapsed": 469, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
from fndata import FnMarketData

# %% id="Z0BLN_sR0hpm" executionInfo={"status": "ok", "timestamp": 1727060836989, "user_tz": -540, "elapsed": 466, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fnmkt_path = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'

# %% id="DEKBrdpb0hpm" executionInfo={"status": "ok", "timestamp": 1727060838872, "user_tz": -540, "elapsed": 438, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다.
fnmkt = FnMarketData(fnmkt_path)

# %% [markdown] id="ES3ArN5b0hpm"
# #### long-format으로 불러오기

# %% id="aY49hQqe0hpm" outputId="42757cc3-924e-49f7-daa5-49e33271f0a9" colab={"base_uri": "https://localhost:8080/", "height": 455} executionInfo={"status": "ok", "timestamp": 1727060841393, "user_tz": -540, "elapsed": 303, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fnmkt.get_data(format='long', multiindex=True)

# %% [markdown] id="dgGS7uEd0hpn"
# #### wide-format으로 불러오기

# %% id="HjWFZkT-0hpn" outputId="7f72bd69-ec6e-411b-f5a5-0215453a24a3" colab={"base_uri": "https://localhost:8080/", "height": 455} executionInfo={"status": "ok", "timestamp": 1727060845033, "user_tz": -540, "elapsed": 467, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fnmkt.get_data(format='wide')

# %% [markdown] id="ER0t1lml0hpn"
# ### 무위험이자율
#
# - 무위험 이자율의 경우 과제 설명과 같이 [한국은행경제통계 시스템의 통화안정증권 364일물 금리](https://ecos.bok.or.kr/#/Short/7478c5)를 사용하였습니다.
#     - 연율화 되어있으므로 과제 수행 시 월율화 작업이 필요합니다.

# %% [markdown] id="1PuxML-40hpn"
# #### 사용법
#
# - 무위험 이자율의 경우 데이터 가이드 포맷이 아니므로 별도 모듈을 제공하지 않습니다.

# %% id="ugKt8Emb0hpo" executionInfo={"status": "ok", "timestamp": 1727060850936, "user_tz": -540, "elapsed": 426, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
rf_path = DATA_DIR / '통안채1년물_월평균_201301-202408.csv'

# %% id="WiZOcaFg0hpo" outputId="c63db159-3a07-496d-9823-ccd0cf90fe47" colab={"base_uri": "https://localhost:8080/", "height": 423} executionInfo={"status": "ok", "timestamp": 1727060852869, "user_tz": -540, "elapsed": 480, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
df = pd.read_csv(rf_path)
df

# %% id="zdlGmr3C0hpp" outputId="442202e9-e554-4720-b44b-697e112a736e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727060858515, "user_tz": -540, "elapsed": 312, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
df.info()

# %% id="-7ttKysl0hpp"
