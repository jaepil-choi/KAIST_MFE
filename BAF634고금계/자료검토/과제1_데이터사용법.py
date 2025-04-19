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
# # 고금계 과제 1 데이터 사용법
#
# - 과제1 수행에 필요한 데이터를 불러오는 방법을 알아봅니다
#

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# %% [markdown]
# ## 경로 설정
#
# - 과제 데이터 파일의 경로를 설정합니다. 
# - 주피터노트북이 있는 폴더의 `data/` 안에 데이터를 두는 것을 권장합니다. 

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
fndata_path = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'

# %% [markdown] vscode={"languageId": "plaintext"}
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

# %% [markdown]
# ### 주식

# %% [markdown]
# #### 기본 사용법

# %%
from fndata import FnStockData

# %%
# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다. 
fn = FnStockData(fndata_path)

# %%
# 사용 가능한 데이터를 확인합니다.
fn.get_items()

# %%
# 분석 기간의 데이터 유니버스를 확인합니다. (금융업종, 거래정지, 관리종목 제외)
univ_list = fn.get_universe()
univ_list

# %%
len(univ_list)

# %%
# 이름으로 종목코드를 확인합니다.
fn.name_to_symbol('삼성전자')

# %%
# 종목코드로 이름을 확인합니다. 
fn.symbol_to_name('A005930')

# %% [markdown]
# #### long-format으로 불러오기

# %%
# 원하는 데이터들을 long-format으로 불러옵니다.

my_data = ['수정주가(원)', '수익률 (1개월)(%)']
df = fn.get_data(my_data) # list가 들어가면 long-format으로 불러옵니다.
df

# %%
# 모든 데이터를 불러옵니다. 

df = fn.get_data()
df

# %% [markdown]
# #### wide-format으로 불러오기

# %%
my_data = '수익률 (1개월)(%)'
fn.get_data(my_data) # string이 들어가면 wide-format으로 불러옵니다.

# %% [markdown]
# ### 시장수익률
#
# - 기초 유니버스로 KSE+KOSDAQ을 썼기 때문에 팩터 분석을 위해 이에 대응하는 시장수익률을 쓰는 것이 좋습니다. 
#     - (기본) [MKF2000](https://www.fnindex.co.kr/multi/detail?menu_type=0&fund_cd=FI00)
#     - (보조) [KRX300](https://ko.wikipedia.org/wiki/KRX_300)
#     - 두 지수 모두 코스피+코스닥을 기초로 시가총액, 거래대금 등을 고려하여 상위 2000/300 종목을 선정해 지수를 만듭니다. 
#     - 두 지수 모두 기본적으로 시가총액 가중 방식으로 지수를 산출합니다. 
#     - 파마 프랜치의 경우 NYSE, AMEX, NASDAQ에 상장된 모든 주식의 시가총액 가중 평균을 사용하였습니다. ([링크](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html))

# %% [markdown]
# #### 기본 사용법

# %%
from fndata import FnMarketData

# %%
fnmkt_path = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'

# %%
# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다. 
fnmkt = FnMarketData(fnmkt_path)

# %% [markdown]
# #### long-format으로 불러오기

# %%
fnmkt.get_data(format='long', multiindex=True)

# %% [markdown]
# #### wide-format으로 불러오기

# %%
fnmkt.get_data(format='wide')

# %% [markdown]
# ### 무위험이자율
#
# - 무위험 이자율의 경우 과제 설명과 같이 [한국은행경제통계 시스템의 통화안정증권 364일물 금리](https://ecos.bok.or.kr/#/Short/7478c5)를 사용하였습니다. 
#     - 연율화 되어있으므로 과제 수행 시 월율화 작업이 필요합니다. 

# %% [markdown]
# #### 사용법
#
# - 무위험 이자율의 경우 데이터 가이드 포맷이 아니므로 별도 모듈을 제공하지 않습니다. 

# %%
rf_path = DATA_DIR / '통안채1년물_월평균_201301-202408.csv'

# %%
df = pd.read_csv(rf_path)
df

# %%
df.info()

# %%
