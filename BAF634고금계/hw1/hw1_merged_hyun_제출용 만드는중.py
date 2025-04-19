# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="BfDUIK2XfENW" outputId="1a27540b-b296-4400-c70a-99d855750eca" executionInfo={"status": "ok", "timestamp": 1728384032534, "user_tz": -540, "elapsed": 21689, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
from google.colab import drive
drive.mount('/content/drive')

# %% id="3cjNFcszfYk5" executionInfo={"status": "ok", "timestamp": 1728384032535, "user_tz": -540, "elapsed": 3, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
import os
os.chdir("/content/drive/MyDrive/고금계")

# %% id="7efaee34-1c13-45ec-b1c6-d999c3268c20" executionInfo={"status": "ok", "timestamp": 1728384079729, "user_tz": -540, "elapsed": 47197, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fndata import FnStockData
from fndata import FnMarketData
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
from pandas.tseries.offsets import MonthEnd

import warnings

warnings.filterwarnings('ignore')

# 지수표기법<>일반표기법 전환. 6자리인 이유는 rf때문
pd.set_option('display.float_format', '{:.6f}'.format)

CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

fndata_path = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'
fnmkt_path = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'
rf_path = DATA_DIR / '통안채1년물_월평균_201301-202408.csv'

# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다.
fn = FnStockData(fndata_path)

# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다.
fnmkt = FnMarketData(fnmkt_path)

# %% id="7b898adc-1f10-4e70-9b28-f4a3a7af0656" executionInfo={"status": "ok", "timestamp": 1728384083791, "user_tz": -540, "elapsed": 4065, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
# 주식 데이터 로드
stocks_df = fn.get_data()
stocks_df = stocks_df.loc[stocks_df.index.get_level_values('date') < '2024-01-31']

# 시장 데이터 로드
market_df = fnmkt.get_data(format='long', multiindex=True)
market_df = market_df.loc[market_df.index.get_level_values('date') < '2024-01-31']

# 무위험 이자율 데이터 로드
df_rf = pd.read_csv(rf_path)
df_rf.columns = ['date', 'rf']
df_rf['date'] = pd.to_datetime(df_rf['date'], format='%Y/%m') + pd.offsets.MonthEnd(0)  # 말일로 변경
df_rf.set_index('date', inplace=True)
df_rf['rf'] = (1 + (df_rf['rf']/100)) ** (1/12) - 1  # 월별 수익률로 변환
df_rf = df_rf.loc[df_rf.index < '2024-01-31']

# 올해 데이터 제거
CUT_DATE = '2023-12-31'
stocks_df = stocks_df[stocks_df.index.get_level_values('date') <= CUT_DATE]
market_df = market_df[market_df.index.get_level_values('date') <= CUT_DATE]
df_rf = df_rf[df_rf.index <= CUT_DATE]

# 수익률 정상화: 100 -> 1
stocks_df['수익률 (1개월)(%)'] = stocks_df['수익률 (1개월)(%)'] / 100

# 컬럼명 변경: '수익률 (1개월)'
stocks_df.rename(columns={'수익률 (1개월)(%)': '수익률 (1개월)'}, inplace=True)

# 'Symbol'과 'date' 기준으로 데이터프레임 정렬
stocks_df = stocks_df.reset_index()
stocks_df = stocks_df.sort_values(['Symbol', 'date'])

# 각 종목(Symbol)별로 결측값 처리 함수 정의
def set_prior_values_to_nan(group):
    # '수익률 (1개월)' 열에서 결측값 위치 확인
    is_na = group['수익률 (1개월)'].isna()
    if is_na.any():
        # 결측값이 처음 발생한 위치의 인덱스 찾기
        first_nan_index = is_na.idxmax()
        # 결측값 발생 시점 이전의 모든 '수익률 (1개월)' 값을 NaN으로 변경
        group.loc[group.index < first_nan_index, '수익률 (1개월)'] = np.nan
    return group

# 그룹별로 함수 적용하여 결측값 처리
stocks_df = stocks_df.groupby('Symbol').apply(set_prior_values_to_nan)

# 인덱스 재설정 (groupby로 인한 멀티인덱스 제거)
stocks_df.reset_index(drop=True, inplace=True)

# 시프트할 재무 데이터 컬럼 목록
financial_columns = [
    '기말발행주식수 (보통)(주)',
    '보통주자본금(천원)',
    '자본잉여금(천원)',
    '이익잉여금(천원)',
    '자기주식(천원)',
    '이연법인세부채(천원)',
    '영업이익(천원)',
    '매출액(천원)',
    '매출원가(천원)',
    '이자비용(천원)',
    '총자산(천원)'
]

# 1. 인덱스 재설정 및 데이터프레임 정렬
stocks_df = stocks_df.sort_values(['Symbol', 'date'])

# 2. 'Symbol'을 인덱스로 설정
stocks_df.set_index('Symbol', inplace=True)

# 3. 각 재무 데이터 컬럼에 대해 그룹별로 6개월 시프트 적용
for col in financial_columns:
    stocks_df[col + '_lag'] = stocks_df.groupby(level='Symbol')[col].shift(6)

# 4. 인덱스 재설정
stocks_df.reset_index(inplace=True)

# 멀티인덱스를 'date'와 'Symbol' 순서로 설정
stocks_df.set_index(['date', 'Symbol'], inplace=True)
stocks_df = stocks_df.sort_index(level=['date', 'Symbol'])

# %% jupyter={"source_hidden": true} colab={"base_uri": "https://localhost:8080/", "height": 735} id="70797981-c45c-43e1-bb0c-6df5d631968e" executionInfo={"status": "ok", "timestamp": 1728384083792, "user_tz": -540, "elapsed": 6, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="1a50ac1c-d7b5-46d3-f96f-fe059226571d"
stocks_df

# %% jupyter={"source_hidden": true} colab={"base_uri": "https://localhost:8080/"} id="52d6affd-6ae3-462b-85c6-7b9ffef5cbb6" executionInfo={"status": "ok", "timestamp": 1728384084377, "user_tz": -540, "elapsed": 590, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="bfade8cb-57c2-4be0-f867-698600e75836"
# 각 날짜별로 '수익률 (1개월)' 열에서 결측치를 가진 Symbol의 수를 계산합니다.
missing_counts = stocks_df['수익률 (1개월)'].isna().groupby(level='date').sum()

# 각 날짜별 전체 Symbol의 수를 계산합니다.
total_counts = stocks_df.groupby('date').size()

# 결과를 하나의 데이터프레임으로 합칩니다.
result_df = pd.DataFrame({
    '전체 Symbol 수': total_counts,
    '결측값을 가진 Symbol 수': missing_counts
})

# 결측값 비율을 계산합니다.
result_df['결측값 비율'] = result_df['결측값을 가진 Symbol 수'] / result_df['전체 Symbol 수']

# 결과를 출력합니다.
print(result_df)

# %% [markdown] id="ee2134db-4dd8-4f53-92bf-5ce23f53746f"
# # Market cap

# %% id="64b77aaf-da0b-4cb0-8f50-d1aec93a7dea" colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"status": "ok", "timestamp": 1728384085738, "user_tz": -540, "elapsed": 1363, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="4a81b921-8c60-4614-d2db-e7674301364e"
# 발행주식 데이터(보통주) : 시가총액 계산 시 시프트된 발행주식수를 사용합니다.
stocks_df['Market cap'] = stocks_df['종가(원)'] * stocks_df['기말발행주식수 (보통)(주)_lag']

# 장부가치 계산: 시프트된 재무 데이터를 사용하여 선행편향을 방지합니다.
stocks_df['Bookvalue'] = (
    stocks_df['보통주자본금(천원)_lag'] +
    stocks_df['자본잉여금(천원)_lag'].fillna(0) +
    stocks_df['이익잉여금(천원)_lag'].fillna(0) +
    stocks_df['자기주식(천원)_lag'].fillna(0) +
    stocks_df['이연법인세부채(천원)_lag'].fillna(0)
)

# BM (Book-to-Market Ratio) 계산
stocks_df['BM'] = stocks_df['Bookvalue'] / stocks_df['Market cap']

# BM 결측값이 있는 행 제거
stocks_df.dropna(subset=['BM'], inplace=True)

# BM 분위수 계산 함수 수정
def qcut_BM(x):
    if x.dropna().empty:
        return pd.Series(np.nan, index=x.index)
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Growth', 'Neutral', 'Value'])
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=x.index)

# 날짜별로 BM 분위수 계산
stocks_df['bm_quantiles'] = stocks_df.groupby('date')['BM'].transform(qcut_BM)

# OP (Operating Profitability) 계산: 시프트된 영업이익을 사용하여 선행편향을 방지합니다.
stocks_df['OP'] = stocks_df['영업이익(천원)_lag'].fillna(0) / stocks_df['Bookvalue']

# OP 분위수 계산 함수
def qcut_OP(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Weak', 'Neutral', 'Robust'])
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=x.index)

# 날짜별로 OP 분위수 계산
stocks_df['OP_quantiles'] = stocks_df.groupby('date')['OP'].transform(qcut_OP)
stocks_df['OP_quantiles']

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="ce565ce1-5655-4b59-b779-68a44502dd83" outputId="690e3577-b81a-4559-de8a-e697bdae3757" executionInfo={"status": "ok", "timestamp": 1728384085738, "user_tz": -540, "elapsed": 8, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
'''
MKF2000 사용함.
'''
market_df = market_df.xs('MKF2000', level='Symbol Name')
market_df.columns = ['mkt']
market_df= pd.concat([market_df, df_rf], axis=1)
market_df['mkt_rf'] = market_df['mkt'] - market_df['rf']
market_df

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="eda9fa7d-085b-488c-858a-69ce2c53a343" outputId="75fdc7a2-fd88-4b70-9183-dd7ad4af47da" executionInfo={"status": "ok", "timestamp": 1728384086352, "user_tz": -540, "elapsed": 620, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
# B/M에 따른 SMB 계산

# 'size_quantiles' 계산 (시가총액을 기준으로 Small, Big 분류)
stocks_df['size_quantiles'] = stocks_df.groupby('date')['Market cap'].transform(
    lambda x: pd.qcut(x, 2, labels=['Small', 'Big'])
)

# BM 분위수별 평균 수익률 계산
df_smb_bm = stocks_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)'].mean().unstack(['size_quantiles', 'bm_quantiles'])

# 작은 규모 기업의 평균 수익률 계산
small_bm_avg = (
    df_smb_bm[('Small', 'Value')] +
    df_smb_bm[('Small', 'Neutral')] +
    df_smb_bm[('Small', 'Growth')]
)

# 큰 규모 기업의 평균 수익률 계산
big_bm_avg = (
    df_smb_bm[('Big', 'Value')] +
    df_smb_bm[('Big', 'Neutral')] +
    df_smb_bm[('Big', 'Growth')]
)

# BM에 따른 SMB 계산
smb_bm = (small_bm_avg / 3) - (big_bm_avg / 3)

# OP에 따른 SMB 계산
df_smb_op = stocks_df.groupby(['date', 'size_quantiles', 'OP_quantiles'])['수익률 (1개월)'].mean().unstack(['size_quantiles', 'OP_quantiles'])

# 작은 규모 기업의 평균 수익률 계산
small_op_avg = (
    df_smb_op[('Small', 'Robust')] +
    df_smb_op[('Small', 'Neutral')] +
    df_smb_op[('Small', 'Weak')]
)

# 큰 규모 기업의 평균 수익률 계산
big_op_avg = (
    df_smb_op[('Big', 'Robust')] +
    df_smb_op[('Big', 'Neutral')] +
    df_smb_op[('Big', 'Weak')]
)

# OP에 따른 SMB 계산
smb_op = (small_op_avg / 3) - (big_op_avg / 3)

# INV에 따른 SMB 계산 (투자율 기준)

# 총자산 변화율로 INV 계산 (시프트된 총자산 데이터를 사용하여 선행편향 방지)
# YOY 계산이므로 첫 1년의 smb_inv 값은 NaN이 됨
stocks_df['INV'] = stocks_df.groupby('Symbol')['총자산(천원)_lag'].pct_change(12)

# INV에 따라 'Conservative', 'Neutral', 'Aggressive'로 분류
def qcut_INV(x):
    if x.dropna().empty:
        return pd.Series(np.nan, index=x.index)
    try:
        # 투자 증가율이 낮은 기업을 'Conservative', 높은 기업을 'Aggressive'로 분류
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Conservative', 'Neutral', 'Aggressive'])
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=x.index)

stocks_df['inv_quantiles'] = stocks_df.groupby('date')['INV'].transform(qcut_INV)

df_smb_inv = stocks_df.groupby(['date', 'size_quantiles', 'inv_quantiles'])['수익률 (1개월)'].mean().unstack(['size_quantiles', 'inv_quantiles'])

# 작은 규모 기업의 평균 수익률 계산
small_inv_avg = (
    df_smb_inv[('Small', 'Conservative')] +
    df_smb_inv[('Small', 'Neutral')] +
    df_smb_inv[('Small', 'Aggressive')]
)

# 큰 규모 기업의 평균 수익률 계산
big_inv_avg = (
    df_smb_inv[('Big', 'Conservative')] +
    df_smb_inv[('Big', 'Neutral')] +
    df_smb_inv[('Big', 'Aggressive')]
)

# INV에 따른 SMB 계산
smb_inv = (small_inv_avg / 3) - (big_inv_avg / 3)

# 최종 SMB 계산
# smb_inv가 NaN인 기간(초기 1년)은 smb_bm과 smb_op의 평균으로 SMB 계산
# smb_inv가 존재하는 기간은 smb_bm, smb_op, smb_inv의 평균으로 SMB 계산
smb = pd.Series(index=smb_bm.index, dtype='float64')

# smb_inv가 NaN이 아닌 기간에 대해 SMB 계산
smb.loc[smb_inv.notna()] = ((smb_bm + smb_op + smb_inv) / 3).loc[smb_inv.notna()]

# smb_inv가 NaN인 기간에 대해 SMB 계산 (smb_bm과 smb_op의 평균)
smb.loc[smb_inv.isna()] = ((smb_bm + smb_op) / 2).loc[smb_inv.isna()]

# 결과 출력
smb

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="HOiVfLYSnkTT" outputId="7e777995-68ef-4ed4-c3cc-309b201cae87" executionInfo={"status": "ok", "timestamp": 1728384086352, "user_tz": -540, "elapsed": 7, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
# HML (High Minus Low) 계산
# B/M (Book-to-Market) 비율에 따른 수익률을 사용하여 HML 계산

# BM 분위수별로 '수익률 (1개월)'의 평균을 계산하고, 사이즈와 BM 분위수별로 나눔
df_hml = stocks_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)'].mean().unstack(['size_quantiles', 'bm_quantiles'])
# Value (고 BM) 주식의 수익률 계산 (Small과 Big 합산)
high_hml = df_hml[('Small', 'Value')] + df_hml[('Big', 'Value')]
# Growth (저 BM) 주식의 수익률 계산 (Small과 Big 합산)
low_hml = df_hml[('Small', 'Growth')] + df_hml[('Big', 'Growth')]
# HML 계산 (Value 주식 수익률 - Growth 주식 수익률) / 2
hml = (high_hml - low_hml) / 2

# 결과 출력
hml

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="diuV8uRjoo8d" outputId="858d1a23-1da8-443b-d300-2ba3bf0af3db" executionInfo={"status": "ok", "timestamp": 1728384086353, "user_tz": -540, "elapsed": 7, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
df_rmv = stocks_df.groupby(['date', 'size_quantiles', 'OP_quantiles'])['수익률 (1개월)'].mean().unstack(['size_quantiles', 'OP_quantiles'])

high_rmw = df_rmv[('Small', 'Robust')] + df_rmv[('Big', 'Robust')]
low_rmw = df_rmv[('Small', 'Weak')] + df_rmv[('Big', 'Weak')]

rmw = (high_rmw - low_rmw) / 2
rmw

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="URP_0F9fopct" outputId="5d425aa4-b165-409d-a78e-12e6fc966090" executionInfo={"status": "ok", "timestamp": 1728384089381, "user_tz": -540, "elapsed": 3034, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
# 데이터 정렬 (종목(Symbol)과 날짜(date) 기준)
stocks_df = stocks_df.sort_values(['Symbol', 'date'])

# 투자율(invest) 계산: 각 종목별로 전년 대비 변화율 계산
stocks_df['invest'] = stocks_df.groupby('Symbol')['총자산(천원)_lag'].transform(lambda x: (x - x.shift(12)) / x.shift(12))

# NaN 및 무한대 값 처리
stocks_df['invest'].replace([np.inf, -np.inf], np.nan, inplace=True)

# 투자율에 따라 분위수 분류
def qcut_invest(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Conservative', 'Neutral', 'Aggressive'])
    except (ValueError, IndexError):  # ValueError와 IndexError 모두 처리
        return pd.Series(np.nan, index=x.index)

# 투자율 분위수 계산 (날짜별로 그룹화)
stocks_df['invest_quantiles'] = stocks_df.groupby('date')['invest'].transform(qcut_invest)

# CMA 데이터프레임 생성
cma_data = stocks_df.groupby(['date', 'size_quantiles', 'invest_quantiles'])['수익률 (1개월)'].mean().unstack(['size_quantiles', 'invest_quantiles'])

# 필요한 컬럼 존재 여부 확인 및 NaN 처리
expected_columns = [
    ('Small', 'Conservative'), ('Small', 'Aggressive'),
    ('Big', 'Conservative'), ('Big', 'Aggressive')
]

for col in expected_columns:
    if col not in cma_data.columns:
        cma_data[col] = np.nan

# Conservative (낮은 투자율) 주식의 수익률 계산 (Small과 Big 합산)
low_invest = cma_data[('Small', 'Conservative')] + cma_data[('Big', 'Conservative')]

# Aggressive (높은 투자율) 주식의 수익률 계산 (Small과 Big 합산)
high_invest = cma_data[('Small', 'Aggressive')] + cma_data[('Big', 'Aggressive')]

# CMA 계산 (Conservative 주식 수익률 - Aggressive 주식 수익률) / 2
cma = (low_invest - high_invest) / 2

# 결과 출력
cma

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="lBqc7a3OrVRh" outputId="f4d58f8a-4423-4bad-e815-8c984e415541" executionInfo={"status": "ok", "timestamp": 1728384092078, "user_tz": -540, "elapsed": 2700, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
# 모멘텀(Momentum) 계산: 각 종목별로 12개월 전 대비 1개월 전의 가격 변동률 계산
stocks_df['Momentum'] = stocks_df.groupby('Symbol')['수정주가(원)'].transform(lambda x: (x.shift(1) - x.shift(12)) / x.shift(12))

# 모멘텀 분위수 분류 함수 정의
def qcut_momentum(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Loser', 'Middle', 'Winner'])
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=x.index)

# 날짜별로 모멘텀 분위수 계산
stocks_df['Momentum_rank'] = stocks_df.groupby('date')['Momentum'].transform(qcut_momentum)

# '수익률 (1개월)(%)' 컬럼명을 '수익률 (1개월)'로 이미 변경하였으므로, 해당 컬럼명 사용
# 모멘텀 분위수별 평균 수익률 계산
umd = stocks_df.groupby(['date', 'Momentum_rank'])['수익률 (1개월)'].mean().unstack()

# 모멘텀 포트폴리오 수익률 계산 (Winner - Loser)
umd['WML'] = umd['Winner'] - umd['Loser']

# 결과 출력
umd

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="_SIJKXiWsfUr" outputId="d2fddbce-fae1-46a7-f8b2-18b0d641275d" executionInfo={"status": "ok", "timestamp": 1728384096339, "user_tz": -540, "elapsed": 4264, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
# 1개월 수익률 계산: 각 종목별로 월별 수익률 계산
stocks_df['1M_Return'] = stocks_df.groupby('Symbol')['수정주가(원)'].transform(lambda x: x.pct_change())

# 리버설 분위수 분류 함수 정의
def qcut_reversal(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Winner', 'Middle', 'Loser'])
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=x.index)

# 날짜별로 리버설 분위수 계산
stocks_df['Reversal_rank'] = stocks_df.groupby('date')['1M_Return'].transform(qcut_reversal)

# 리버설 분위수별 평균 수익률 계산
str_df = stocks_df.groupby(['date', 'Reversal_rank'])['수익률 (1개월)'].mean().unstack()

# 리버설 포트폴리오 수익률 계산 (Winner - Loser)
str_df['WML'] = str_df['Winner'] - str_df['Loser']

# 데이터 정렬 - 원복
stocks_df = stocks_df.sort_values(['date', 'Symbol'])

# 결과 출력
str_df

# %% [markdown] id="0ab8ff47-53ae-4682-8485-bb346df3eaa4"
# # 5*5 만들기(independent, dependent 택1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="PTaKGh0n4aJ_" outputId="f4cffb08-0c48-4488-db7b-26b6db6cd580" executionInfo={"status": "ok", "timestamp": 1728384096339, "user_tz": -540, "elapsed": 5, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
#indenpendent doublesort
stocks_df['size_quantiles_by5'] = pd.qcut(stocks_df['Market cap'], 5, labels=['Small', '2', '3', '4', 'Big'])
# stocks_df['size_quantiles_by5']
def qcut_BM_by5(x):
    try:
        return pd.qcut(x, 5, labels=['Low', '2', '3', '4', 'High'])
    except (ValueError, IndexError):  # ValueError와 IndexError 모두 처리
        return pd.Series(np.nan, index=x.index)
stocks_df['bm_quantiles_by5'] = stocks_df.groupby('date')['BM'].transform(qcut_BM_by5)
stocks_df['bm_quantiles_by5']

# %% id="HuPmHa6G4tnJ" executionInfo={"status": "ok", "timestamp": 1728384098143, "user_tz": -540, "elapsed": 1808, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
stocks_df['excess_rets'] = stocks_df['수익률 (1개월)'] - df_rf['rf'] # 2024-09-19 빼고는 존재함????
portfolios = stocks_df.groupby(['date', 'size_quantiles_by5', 'bm_quantiles_by5']).apply(
    lambda group: group['excess_rets'].mean(skipna=True)
    ).unstack(level=['size_quantiles_by5', 'bm_quantiles_by5'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="kWOrCZBv4_sb" outputId="b363dc77-b781-4def-f39a-51349213ff52" executionInfo={"status": "ok", "timestamp": 1728384098143, "user_tz": -540, "elapsed": 9, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
_3factors = pd.DataFrame({
    'Mkt_RF': market_df['mkt_rf'],
    'SMB': smb,
    'HML': hml,
    'RF' : df_rf['rf'],
    'UMD': umd['WML']
    })
_3factors.dropna(how='all', inplace=True)
_3factors

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="oXgkQclL7KZJ" outputId="3413f5f4-fbe5-46dd-8030-062c6d351333" executionInfo={"status": "ok", "timestamp": 1728384098143, "user_tz": -540, "elapsed": 7, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
_5factors = pd.DataFrame({
    'Mkt_RF': market_df['mkt_rf'],
    'SMB': smb,
    'HML': hml,
    'RMW': rmw,
    'CMA': cma,
    'RF' : df_rf['rf'],
    'UMD': umd['WML'],
    'STR': str_df['WML']
})
_5factors.dropna(how='all', inplace=True)
_5factors

# %% id="j2s0h_HAxwGs" executionInfo={"status": "ok", "timestamp": 1728385230054, "user_tz": -540, "elapsed": 645, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
_5factors.to_excel ("/content/drive/MyDrive/고금계/5factors.xlsx")

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="LoyAHhSi7Mf4" outputId="89fe1476-d010-4b34-a440-c343dc35c6f8" executionInfo={"status": "ok", "timestamp": 1728384098143, "user_tz": -540, "elapsed": 6, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
_5factors.describe()


# %% id="74ee02a5-217c-4478-9730-f0059bf16103" executionInfo={"status": "ok", "timestamp": 1728384100510, "user_tz": -540, "elapsed": 2373, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
def double_sorting(df, size_col, bm_col, method='independent'):
    if method == 'independent':
        # Independent double sorting: 각 변수를 독립적으로 소팅
        df['size_sorted'] = df.groupby('date')[size_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4, 5]))
        df['bm_sorted'] = df.groupby('date')[bm_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5]))
    elif method == 'dependent':
        # Dependent double sorting: Size로 먼저 소팅 후, BM으로 다시 소팅
        df['size_sorted'] = df.groupby('date')[size_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4, 5]))
        df['bm_sorted'] = df.groupby(['date', 'size_sorted'])[bm_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4, 5]))

        # df['bm_sorted'] = df.groupby('date')[bm_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4, 5]))
        # df['size_sorted'] = df.groupby(['date', 'bm_sorted'])[size_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5]))

    else:
        raise ValueError("method는 'independent' 또는 'dependent' 중 하나여야 합니다.")

    return df

# 사용 예시
stocks_df = double_sorting(stocks_df, 'Market cap', 'BM', method='dependent')

# %% id="Pz-AQ38MHxnL" colab={"base_uri": "https://localhost:8080/", "height": 237} executionInfo={"status": "ok", "timestamp": 1728385482977, "user_tz": -540, "elapsed": 3079, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="d13603b7-dd1b-4451-c455-2eb7fdcb5cda"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# stocks_df와 market_df를 결합하여 mkt_rf를 stocks_df에 추가
def merge_market_and_stocks(stocks_df, market_df):
    # 'date' 열을 기준으로 market_df와 병합
    merged_df = pd.merge(stocks_df, market_df[['mkt_rf']], left_index=True, right_index=True, how='left')
    return merged_df

# 포트폴리오 키 생성 함수
def add_portfolio_key(df):
    df['portfolio_key'] = df['size_sorted'].astype(object)*10 + df['bm_sorted'].astype(object)
    return df

# 멀티인덱스에서 날짜만 추출
def get_unique_dates(df):
    return df.index.get_level_values(0).unique()

# Fama-MacBeth 회귀분석 함수 (결측값 처리 추가)
def fama_macbeth_regression(df, factors):
    betas = []
    t_values = []

    # 시점별로 크로스 섹션 회귀 수행
    for date in df.index.get_level_values(0).unique():
        df_date = df.loc[date]

        # 결측값(NaN) 제거
        df_date = df_date.dropna(subset=factors + ['수익률 (1개월)'])

        X = df_date[factors].values  # 요인 변수들
        y = df_date['수익률 (1개월)'].values  # 개별 자산의 수익률

        if len(y) > 0:  # y가 비어있지 않을 때만 회귀 분석 실행
            # 회귀분석 모델
            reg = LinearRegression().fit(X, y)
            betas.append(reg.coef_)  # 회귀계수 저장

            # 잔차 및 t값 계산
            residuals = y - reg.predict(X)
            sigma = np.sqrt(np.var(residuals))
            t = reg.coef_ / (sigma / np.sqrt(len(X)))  # t값 계산
            t_values.append(t)

    # 각 시점별 회귀계수의 평균 계산
    avg_betas = np.mean(betas, axis=0)
    avg_t_values = np.mean(t_values, axis=0)

    return avg_betas, avg_t_values

# 누적 수익률 계산
def backtest_portfolio(df, rebalancing_period='M'):
    unique_dates = get_unique_dates(df)
    rebalanced_dates = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq=rebalancing_period)

    portfolio_returns = {}
    cumulative_returns = {}

    # Initialize portfolios for all 5x5 combinations
    sizes = [1, 2, 3, 4, 5]
    bms = [1, 2, 3, 4, 5]
    for size in sizes:
        for bm in bms:
            portfolio_key = f'{size}_{bm}'
            portfolio_returns[portfolio_key] = []

    # Calculate portfolio returns over time
    for date in rebalanced_dates:
        df_rebalanced = df.loc[(date,), :]

        for size in sizes:
            for bm in bms:
                portfolio_key = f'{size}_{bm}'
                portfolio_df = df_rebalanced[(df_rebalanced['size_sorted'] == size) & (df_rebalanced['bm_sorted'] == bm)]

                portfolio_return = portfolio_df['수익률 (1개월)'].mean() / 100
                portfolio_returns[portfolio_key].append(portfolio_return)

    # Calculate cumulative returns for each portfolio
    for portfolio_key, returns in portfolio_returns.items():
        returns = np.array(returns)
        cumulative_returns[portfolio_key] = np.cumprod(1 + returns) - 1

    return cumulative_returns

# 누적 수익률 시각화
def visualize_cumulative_returns(cumulative_returns, rebalanced_dates):
    fig = go.Figure()

    for portfolio_key, cum_return in cumulative_returns.items():
        fig.add_trace(go.Scatter(x=rebalanced_dates, y=cum_return, mode='lines', name=portfolio_key))

    fig.update_layout(
        title='Cumulative Returns for 5x5 Size-BM Portfolios',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns'
    )

    fig.show()

# 초과수익률 계산 함수
def calculate_excess_returns(df, rf_df):
    df['rf'] = df.index.get_level_values('date').map(rf_df['rf'])
    df['excess_return'] = df['수익률 (1개월)'] - df['rf']
    return df

# 회귀분석 함수 (Fama-Macbeth 또는 단순 회귀 가능)
def run_regression(df, market_rf):
    results = {}
    for portfolio_key in df['portfolio_key'].unique():
        portfolio_df = df[df['portfolio_key'] == portfolio_key]
        X = sm.add_constant(portfolio_df[market_rf])  # market_rf를 독립 변수로 사용
        y = portfolio_df['excess_return']  # 종속 변수는 초과수익률
        model = sm.OLS(y, X).fit()  # 회귀 분석 실행
        results[portfolio_key] = {'coef': model.params[market_rf], 't_value': model.tvalues[market_rf]}  # 회귀 계수와 t값 저장
    return results

# 테이블 생성 함수 (월별 평균수익률과 t값 포함)
def generate_results_table(df, regression_results):
    table_data = []
    sizes = [1, 2, 3, 4, 5]
    bms = [1, 2, 3, 4, 5]

    for size in sizes:
        row = []
        for bm in bms:
            portfolio_key = f'{size}{bm}'
            avg_return = df[df['portfolio_key'] == int(portfolio_key)]['excess_return'].mean()  # 평균 초과수익률 계산
            if portfolio_key in regression_results:
                t_value = regression_results[portfolio_key]['t_value']
                row.append(f'{avg_return:.6f} ({t_value:.6f})')  # 평균 수익률과 t값을 함께 표기
            else:
                row.append(f'{avg_return:.6f} (N/A)')  # 회귀 결과가 없는 경우 N/A로 표기
        table_data.append(row)

    # High-Low 차이 계산 (각 size별로 High-Low 차이 추가)
    for i, size in enumerate(sizes):
        high_return = df[df['portfolio_key'] == f'{size}5']['excess_return'].mean()  # High
        low_return = df[df['portfolio_key'] == f'{size}1']['excess_return'].mean()  # Low
        high_low_diff = high_return - low_return
        table_data[i].append(f'{high_low_diff:.6f}')

    # Small-Big 차이 계산
    row = []
    for bm in bms:
        small_return = df[df['portfolio_key'] == f'11{bm}']['excess_return'].mean()  # Small
        big_return = df[df['portfolio_key'] == f'51{bm}']['excess_return'].mean()  # Big
        small_big_diff = small_return - big_return
        row.append(f'{small_big_diff:.6f}')
    table_data.append(row)

    # 테이블 열과 행 정의
    columns = ['Low', '2', '3', '4', 'High', 'High-Low']
    index = ['Small', '2', '3', '4', 'Big', 'Small-Big']

    results_df = pd.DataFrame(table_data, columns=columns, index=index)
    # High-Low 차이 계산 및 추가 (소수점 2자리로 포맷팅)
    results_df['High-Low'] = (results_df['High'].apply(lambda x: float(x.split(' ')[0])) - results_df['Low'].apply(lambda x: float(x.split(' ')[0])))
    results_df['High-Low'] = results_df['High-Low'].apply(lambda x: f'{x:.6f}')  # 소수점 2자리로 포맷

    # Small-Big 차이 계산 및 추가
    small_big_diff = []
    columns = ['Low', '2', '3', '4', 'High']
    for col in columns:
        small_return_str = results_df.loc['Small', col]
        big_return_str = results_df.loc['Big', col]

        # 수익률만 추출
        small_return = float(small_return_str.split(' ')[0])
        big_return = float(big_return_str.split(' ')[0])

        small_big_diff.append(f'{small_return - big_return:.6f}')  # 소수점 2자리로 포맷

    # Small-Big 차이를 각 열에 추가, 마지막 열은 None
    results_df.loc['Small-Big'] = small_big_diff + [None]

    return results_df

# 전체 실행 함수
def run_backtest_and_create_table(stocks_df, rf_df, market_df, rebalancing_period='M'):
    # 포트폴리오 키 추가
    stocks_df = add_portfolio_key(stocks_df)

    # stocks_df와 market_df 병합 (mkt_rf 추가)
    stocks_df = merge_market_and_stocks(stocks_df, market_df)

    # 초과수익률 계산
    stocks_df = calculate_excess_returns(stocks_df, rf_df)

    # 회귀분석 수행
    regression_results = run_regression(stocks_df, 'mkt_rf')

    # 결과 테이블 생성
    results_table = generate_results_table(stocks_df, regression_results)

    return results_table

# 최종 실행
results_table = run_backtest_and_create_table(stocks_df, df_rf, market_df, rebalancing_period='M')
results_table


# %% colab={"base_uri": "https://localhost:8080/", "height": 648} id="tt226u6fvdBK" executionInfo={"status": "ok", "timestamp": 1728388701319, "user_tz": -540, "elapsed": 4975, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="e47aa5e4-fa56-4ce4-a3dd-4b95d696a108"

# 누적 수익률 계산 _ 결과제출용
def backtest_portfolio_to_table(df, rebalancing_period='M'):
    unique_dates = get_unique_dates(df)
    rebalanced_dates = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq=rebalancing_period)

    portfolio_returns = {}
    cumulative_returns = {}

    # Initialize portfolios for all 5x5 combinations
    sizes = [1, 2, 3, 4, 5]
    bms = [1, 2, 3, 4, 5]
    for size in sizes:
        for bm in bms:
            portfolio_key = f'{size}_{bm}'
            portfolio_returns[portfolio_key] = []

    # Calculate portfolio returns over time
    for date in rebalanced_dates:
        df_rebalanced = df.loc[(date,), :]

        for size in sizes:

            for bm in bms:
                portfolio_key = f'{size}_{bm}'
                portfolio_df = df_rebalanced[(df_rebalanced['size_sorted'] == size) & (df_rebalanced['bm_sorted'] == bm)]

                portfolio_return = portfolio_df['수익률 (1개월)'].mean() / 100
                portfolio_returns[portfolio_key].append(portfolio_return)

    # Calculate cumulative returns for each portfolio
    for portfolio_key, returns in portfolio_returns.items():
        returns = np.array(returns)
        cumulative_returns[portfolio_key] = np.cumprod(1 + returns) - 1

    # 결과제출용으로 추가한 코드
    FbyF_result_table = pd.DataFrame(index=unique_dates)

    for key in cumulative_returns.keys():
        FbyF_result_table[key] = portfolio_returns[key]

    return FbyF_result_table

FbyF = backtest_portfolio_to_table (stocks_df)
FbyF
# FbyF.to_excel ("/content/drive/MyDrive/고금계/5x5 return.xlsx")


# %% colab={"base_uri": "https://localhost:8080/", "height": 735} id="d4dvkCR-uMmV" executionInfo={"status": "ok", "timestamp": 1728384134680, "user_tz": -540, "elapsed": 1333, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="4e4ced24-ffec-4269-f340-a3580e824b7d"
stocks_df

# %% id="ec6ebaa4-fe74-4587-92f2-b37c7fbeee3f" executionInfo={"status": "ok", "timestamp": 1728384103976, "user_tz": -540, "elapsed": 4, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
import plotly.graph_objects as go
import pandas as pd

# 백테스팅을 위한 데이터 전처리 및 리밸런싱 시뮬레이션

def calculate_returns(df, weight_type='equal'):
    """ 섹터 별 동일가중 / 시가총액 가중 수익률 계산 """
    # 수익률 열만 선택 (예시로 'return'이 수익률 컬럼이라고 가정)
    returns_df = df.select_dtypes(include=[np.number])  # 수치형 데이터만 선택

    if weight_type == 'equal':
        return returns_df.mean(axis=1)  # 동일가중
    elif weight_type == 'market_cap':
        return (returns_df * df['시가총액']).sum(axis=1) / df['시가총액'].sum()  # 시가총액 가중

# 1. 5*5 size-BM 기반 백테스팅
def backtest_size_bm_5x5(df, rebalancing_dates, weight_type='equal'):
    """ Size-BM 5*5 기반 누적수익률 백테스트, rebalancing_dates에 따라 리밸런싱 """

    df = df.copy()
    cumulative_returns = []
    valid_rebalancing_dates = []

    # BM과 Size에 따라 5분위로 나누기
    df['BM_quantile'] = pd.qcut(df['BM'], 5, labels=False)  # BM 5분위 나누기
    df['Size_quantile'] = pd.qcut(df['Market cap'], 5, labels=False)  # Size 5분위 나누기

    # 25개의 포트폴리오 그룹 생성
    df['portfolio_group'] = df['BM_quantile'].astype(str) + "-" + df['Size_quantile'].astype(str)

    for rebalance_date in rebalancing_dates:
        # 리밸런싱 날짜에 맞는 데이터 선택
        df_rebalanced = df.loc[df.index.get_level_values('date') == rebalance_date]

        # 리밸런싱 데이터가 없으면 건너뜀
        if df_rebalanced.empty:
            print(f"No data available for rebalancing date: {rebalance_date}")
            continue

        # NaN 값 처리: NaN을 이전 값으로 대체 (method='ffill'로 결측값을 직전 값으로 채움)
        df_rebalanced.fillna(method='ffill', inplace=True)

        # 포트폴리오 그룹별로 평균 수익률 계산
        group_returns = df_rebalanced.groupby('portfolio_group')['수익률 (1개월)'].mean()

        cumulative_returns.append(group_returns)
        valid_rebalancing_dates.append(rebalance_date)  # 실제로 데이터를 처리한 날짜만 저장

    # 누적 수익률 데이터프레임 생성
    if cumulative_returns:
        cumulative_returns_df = pd.concat(cumulative_returns, axis=1).T
        cumulative_returns_df.index = valid_rebalancing_dates  # 유효한 리밸런싱 날짜로 인덱스 설정
        # 누적 수익률 계산 (1을 더한 후 곱셈 누적 방식으로 진행)
        cumulative_returns_df = (1 + cumulative_returns_df).cumprod()
    else:
        print("No cumulative returns calculated.")
        cumulative_returns_df = pd.DataFrame()

    return cumulative_returns_df

def backtest_factor_portfolios_v2(factors_df, factor_list, momentum_factors, rebalancing_dates):
    """ 팩터 기반 롱-숏 포트폴리오 누적 수익률 백테스트 (전용 함수) """
    cumulative_returns = {}

    for factor in factor_list:
        if factor not in factors_df.columns:
            raise KeyError(f"'{factor}' column not found in dataframe")

        if factor in momentum_factors:
            # 모멘텀/리버설 팩터는 매달 리밸런싱
            monthly_returns = []
            for rebalance_date in pd.date_range(rebalancing_dates[0], rebalancing_dates[-1], freq='M'):
                factor_data = factors_df.loc[factors_df.index == rebalance_date, factor].dropna()
                if not factor_data.empty:
                    monthly_returns.append(factor_data.mean())  # 동일가중 평균
            cumulative_returns[factor] = (1 + pd.Series(monthly_returns, index=pd.date_range(rebalancing_dates[0], rebalancing_dates[-1], freq='M'))).cumprod()
        else:
            # 나머지 팩터는 매해 6월 리밸런싱
            yearly_returns = []
            for rebalance_date in rebalancing_dates:
                factor_data = factors_df.loc[factors_df.index == rebalance_date, factor].dropna()
                if not factor_data.empty:
                    yearly_returns.append(factor_data.mean())  # 동일가중 평균
            cumulative_returns[factor] = (1 + pd.Series(yearly_returns, index=rebalancing_dates)).cumprod()

    return pd.DataFrame(cumulative_returns)

# Plotly로 누적 수익률 그래프 그리기
def plot_cumulative_returns(cumulative_returns_df, title):
    fig = go.Figure()

    # 섹터/그룹별로 수익률 그래프 그리기 (mode='lines'를 사용하여 선 그래프 생성)
    for column in cumulative_returns_df.columns:
        column_name = str(column) if isinstance(column, tuple) else column  # 튜플을 문자열로 변환
        fig.add_trace(go.Scatter(x=cumulative_returns_df.index, y=cumulative_returns_df[column],
                                 mode='lines', name=column_name))  # 선 그래프 생성

    # 레이아웃 설정
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Cumulative Return')
    return fig


# %% colab={"base_uri": "https://localhost:8080/", "height": 560} id="68a125b6-1b8f-46ff-882c-1256ec485223" executionInfo={"status": "ok", "timestamp": 1728384104597, "user_tz": -540, "elapsed": 624, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="3fe5e5dc-fb43-4605-ddd3-21accc4f88a2"
# 백테스팅 코드에 해당 로직들을 적용
# 1. size-BM 기반 백테스트
rebalancing_dates = pd.date_range(start='2014-06-30', end='2024-06-30', freq='12M')
cumulative_returns_5x5 = backtest_size_bm_5x5(stocks_df, rebalancing_dates)
cumulative_returns_5x5
# Plot for size-BM based cumulative returns
fig_5x5 = plot_cumulative_returns(cumulative_returns_5x5, 'Size-BM 5x5 Cumulative Returns')
fig_5x5.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="d171caf6-fc7b-4b43-ac13-931bc959faf6" executionInfo={"status": "ok", "timestamp": 1728384104597, "user_tz": -540, "elapsed": 7, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}} outputId="13e1d435-24bb-4955-ac2a-618bf771441e"
def backtest_factor_portfolios_v3(factors_df, factor_list, momentum_factors, yearly_rebalancing_dates, monthly_rebalancing_dates):
    """ 팩터 기반 롱-숏 포트폴리오 누적 수익률 백테스트 (팩터별 리밸런싱 주기 적용) """
    cumulative_returns = {}

    for factor in factor_list:
        if factor not in factors_df.columns:
            raise KeyError(f"'{factor}' column not found in dataframe")

        if factor in momentum_factors:
            # 모멘텀/리버설 팩터는 매달 리밸런싱
            monthly_returns = []
            for rebalance_date in monthly_rebalancing_dates:
                factor_data = factors_df.loc[factors_df.index == rebalance_date, factor].dropna()
                if not factor_data.empty:
                    monthly_returns.append(factor_data.mean())  # 동일가중 평균
            # 매월 리밸런싱에 맞는 수익률을 누적 곱 계산
            cumulative_returns[factor] = (1 + pd.Series(monthly_returns, index=monthly_rebalancing_dates[:len(monthly_returns)])).cumprod()
        else:
            # Fama-French 팩터는 매해 6월 리밸런싱
            yearly_returns = []
            valid_rebalancing_dates = []
            for rebalance_date in yearly_rebalancing_dates:
                factor_data = factors_df.loc[factors_df.index == rebalance_date, factor].dropna()
                if not factor_data.empty:
                    yearly_returns.append(factor_data.mean())  # 동일가중 평균
                    valid_rebalancing_dates.append(rebalance_date)  # 실제 데이터가 있는 날짜만 추가
            # 매해 6월 리밸런싱에 맞는 수익률을 누적 곱 계산
            cumulative_returns[factor] = (1 + pd.Series(yearly_returns, index=valid_rebalancing_dates)).cumprod()

    return pd.DataFrame(cumulative_returns)

# 팩터 리스트 정의 (Fama-French 5요소 + 모멘텀, 리버설)
factor_list = ['SMB', 'HML', 'RMW', 'CMA', 'UMD', 'STR']
momentum_factors = ['UMD', 'STR']

# 리밸런싱 날짜: 매년 6월과 매월 리밸런싱 날짜를 따로 설정
yearly_rebalancing_dates = pd.date_range(start='2014-06-30', end='2024-06-30', freq='12M')
monthly_rebalancing_dates = pd.date_range(start='2014-06-30', end='2024-06-30', freq='M')

# 팩터 기반 백테스트 실행
cumulative_returns_factors = backtest_factor_portfolios_v3(_5factors, factor_list, momentum_factors, yearly_rebalancing_dates, monthly_rebalancing_dates)

# 팩터 기반 누적 수익률 플롯
fig_factors = plot_cumulative_returns(cumulative_returns_factors, 'Factor Cumulative Returns')
fig_factors.show()

# %% id="759b1c48-c814-446d-bb96-d71e79702c8a" executionInfo={"status": "ok", "timestamp": 1728384104597, "user_tz": -540, "elapsed": 6, "user": {"displayName": "SungKyu Ko", "userId": "10266390755652929266"}}
