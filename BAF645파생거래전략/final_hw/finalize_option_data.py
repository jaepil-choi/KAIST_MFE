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
# # 옵션 데이터 전처리 최종
#
# - equitiy option 종목들 중 주식 sma (simple moving average) vol 30% 미만만 남기기
# - underlying (주식) ohlcv 붙이기 
# - option atm otm itm 여부 label, 거래 안되는 strike option들 날리기
# - option strike를 atm 기준으로 relative 하게 바꾸기

# %% [markdown]
# ## import libs

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import os, sys
import re

import FinanceDataReader as fdr
from pykrx import stock

# %%
CWD_PATH = Path.cwd()
DATA_PATH = CWD_PATH / 'data'
OUTPUT_PATH = CWD_PATH / 'output'

# %%
krx_option_df = pd.read_parquet(OUTPUT_PATH / 'krx_option_data_20220101-20241204.parquet')

# %%
krx_option_df.head()

# %% [markdown]
# underlying이 비어있는 경우 발견. 
#
# "LS ELECTRIC C 202401 120,000(100)" 이 이전 정규식에서 제대로 파싱되지 않아서 발생. 이름 중간에 띄어쓰기 들어갈 것이라 생각 못했음. 
#
# 그냥 얘 하나밖에 없으니까 빼고 가자. 

# %%
krx_option_df = krx_option_df[ krx_option_df['underlying'].notnull() ]

# %%
len(krx_option_df)

# %% [markdown]
# ## stock ohlcv 가져오기
#
# - pykrx
# - fdr

# %%
underlyings = krx_option_df['underlying'].unique()
underlyings

# %%
# pykrx로 stock들 이름부터 조회 

stock_list = stock.get_market_ticker_list(market='ALL')
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in stock_list}
name_to_ticker = {v: k for k, v in ticker_to_name.items()}

# %%
underlying_name_mapping = []

name_to_ticker5 = {k[:5]: v for k, v in name_to_ticker.items()}
name_to_ticker6 = {k[:6]: v for k, v in name_to_ticker.items()}

for underlying_name in underlyings:
    if underlying_name in name_to_ticker5.keys():
        ticker = name_to_ticker5[underlying_name]
        real_name = ticker_to_name[ticker]
        temp = (underlying_name, real_name, ticker)
        underlying_name_mapping.append(temp)
    
    elif underlying_name in name_to_ticker6.keys():
        ticker = name_to_ticker6[underlying_name]
        real_name = ticker_to_name[ticker]
        temp = (underlying_name, real_name, ticker)
        underlying_name_mapping.append(temp)

    else:
        print(f'{underlying_name} not found')

# %%
# manual mapping

underlying_name_mapping.append(('LGD', 'LG디스플레이', '034220'))
underlying_name_mapping.append(('삼성SDS', '삼성에스디에스', '018260'))
underlying_name_mapping.append(('하나지주', '하나금융지주', '086790'))
underlying_name_mapping.append(('한국조선해', 'HD한국조선해양', '009540'))
underlying_name_mapping.append(('현대두산인', 'HD현대인프라코어', '042670'))

# %%
underlyings_df = pd.DataFrame(
    data=underlying_name_mapping,
    columns=['underlying', 'underlying_full', 'ticker']
) 

# %%
trade_dates = krx_option_df['trade_date'].unique()
trade_dates = sorted(trade_dates)

# %%
START = trade_dates[0].strftime('%Y%m%d')
END = trade_dates[-1].strftime('%Y%m%d')

START, END

# %%
underlying_tickers = underlyings_df['ticker'].unique()

ohlcv_df = pd.DataFrame()

for ticker in underlying_tickers:
    df = stock.get_market_ohlcv_by_date(START, END, ticker)
    df['ticker'] = ticker
    ohlcv_df = pd.concat([ohlcv_df, df], axis=0)

# %% [markdown]
# 변동성 등 먼저 계산

# %%
SMA_WINDOW = 21 * 6 # 6 months

ohlcv_df['등락률'] = ohlcv_df['등락률'] / 100
ohlcv_df['SMA'] = ohlcv_df.groupby('ticker')['등락률'].transform(lambda x: x.rolling(window=SMA_WINDOW).mean())

# %%
ohlcv_df['ret_vol_20d'] = ohlcv_df.groupby('ticker')['등락률'].transform(lambda x: x.rolling(window=20).std())
ohlcv_df['sma_vol_20d'] = ohlcv_df.groupby('ticker')['SMA'].transform(lambda x: x.rolling(window=20).std())

ohlcv_df['ret_vol_60d'] = ohlcv_df.groupby('ticker')['등락률'].transform(lambda x: x.rolling(window=60).std())
ohlcv_df['sma_vol_60d'] = ohlcv_df.groupby('ticker')['SMA'].transform(lambda x: x.rolling(window=60).std())

ohlcv_df['ret_vol_120d'] = ohlcv_df.groupby('ticker')['등락률'].transform(lambda x: x.rolling(window=120).std())
ohlcv_df['sma_vol_120d'] = ohlcv_df.groupby('ticker')['SMA'].transform(lambda x: x.rolling(window=120).std())

ohlcv_df['ret_vol_180d'] = ohlcv_df.groupby('ticker')['등락률'].transform(lambda x: x.rolling(window=180).std())
ohlcv_df['sma_vol_180d'] = ohlcv_df.groupby('ticker')['SMA'].transform(lambda x: x.rolling(window=180).std())

# %% [markdown]
# underlying ticker info와 붙이기

# %%
ohlcv_df.reset_index(inplace=True, drop=False)
underlyings_df = underlyings_df.merge(ohlcv_df, left_on='ticker', right_on='ticker', how='right')

# %%
underlyings_df.head()

# %%
len(underlyings_df)

# %%
underlyings_df.rename(
    columns={
        '날짜': 'trade_date',
        '시가': 'udly_open',
        '고가': 'udly_high',
        '저가': 'udly_low',
        '종가': 'udly_close',
        '거래량': 'udly_volume',
        '등락률': 'udly_return',
        }, 
    inplace=True,
    )

# %%
option_data_full = krx_option_df.merge(underlyings_df, left_on=['underlying', 'trade_date'], right_on=['underlying', 'trade_date'], how='left')

# %%
# 옵션이 close_price 도 없고 open_interest_quantity 도 0인 경우 제외. 아예 거래 없는 행사가의 옵션들임. 

option_data_full = option_data_full[option_data_full['close_price'].notnull() & (option_data_full['open_interest_quantity'] > 0)].copy()

# %%
# 현재 underlying의 종가와 행사가의 차이 기준으로 ATM, OTM, ITM 구분

option_data_full['close_strike_diff'] = option_data_full['udly_close'] - option_data_full['strike']

# %%
# 가장 작은 차이를 가지는 것이 ATM

option_data_full['atm'] = option_data_full.groupby(['trade_date', 'underlying', 'call_or_put', 'expiration', ])['close_strike_diff'].transform(lambda x: x.abs().idxmin() == x.index)

# %%
# open interest quantity는 있는데 거래가 안돼서 가격이 없는 경우가 존재함. (deep deep ITM/OTM)
# 이 경우 atm, otm, itm 모두 False로 처리

option_data_full['atm'] = option_data_full['atm'] & option_data_full['close_price'].notnull()

# %%
option_data_full['itm'] = False

option_data_full.loc[option_data_full['call_or_put'] == 'C', 'itm'] = option_data_full.loc[option_data_full['call_or_put'] == 'C', 'close_strike_diff'] > 0
option_data_full.loc[option_data_full['call_or_put'] == 'P', 'itm'] = option_data_full.loc[option_data_full['call_or_put'] == 'P', 'close_strike_diff'] < 0

option_data_full.loc[option_data_full['atm'] == True, 'itm'] = False # ATM은 ITM이 아님

# %%
# 마찬가지로 itm도 close_price가 없으면 모두 False로 처리

option_data_full['itm'] = option_data_full['itm'] & option_data_full['close_price'].notnull()

# %%
option_data_full['otm'] = ~option_data_full['atm'] & ~option_data_full['itm']

# %%
# 마찬가지로 otm도 close_price가 없으면 모두 False로 처리

option_data_full['otm'] = option_data_full['otm'] & option_data_full['close_price'].notnull()

# %%
# verify atm, itm, otm 

is_itm = option_data_full['itm'] == True
is_atm = option_data_full['atm'] == True
is_otm = option_data_full['otm'] == True

assert option_data_full[is_itm & is_atm].shape[0] == 0 # ATM이면서 ITM인 경우 없어야 함
assert option_data_full[is_itm & is_otm].shape[0] == 0 # OTM이면서 ITM인 경우 없어야 함
assert option_data_full[is_atm & is_otm].shape[0] == 0 # ATM이면서 OTM인 경우 없어야 함

# assert option_data_full[~(is_itm | is_atm | is_otm)].shape[0] == 0 # ATM, ITM, OTM이 아닌 경우는 존재함. open interest가 있는데 거래가 안된 경우

# %%
option_data_full['moneyness'] = np.log(option_data_full['strike'] / option_data_full['udly_close'])

# %%
option_data_full.columns

# %% [markdown]
# 정확한 만기일 (만기월 두 번째 목요일) 구하기

# %%
from datetime import datetime, timedelta

def get_second_thursday_from_str(yyyymm: str) -> datetime:
    """
    Calculate the second Thursday of a given year and month from a string input.
    
    :param yyyymm: A string in 'YYYYMM' format representing the year and month.
    :return: A datetime object representing the second Thursday.
    """
    # Parse the year and month from the string
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    
    # Get the first day of the month
    first_day = datetime(year, month, 1)
    
    # Find the first Thursday of the month
    first_thursday = first_day + timedelta(days=(3 - first_day.weekday() + 7) % 7)
    
    # Add 7 days to get the second Thursday
    second_thursday = first_thursday + timedelta(days=7)
    
    return second_thursday

# Example usage
yyyymm = '202201'  # YYYYMM format
second_thursday = get_second_thursday_from_str(yyyymm)
print(f"The second Thursday of {yyyymm} is {second_thursday.date()}")


# %%
option_data_full['expiration_date'] = option_data_full['expiration'].apply(lambda x: get_second_thursday_from_str(x))

# %% [markdown]
# SMA vol 범위 30% 내외인 종목만 남기기
#
# --> sma 로 하면 다 smoothing 되어버려 범위가 전혀 안나옴. 
#
# 그냥 6개월 실현 변동성 연율화 한게 30% 내외인걸로 하자. --> `ret_vol_120d`

# %%
SMA_VOL_LOWER, SMA_VOL_UPPER = (0.25, 0.35)

# %%
option_data_full['ret_vol_120d_ann'] = option_data_full['ret_vol_120d'] * np.sqrt(252)

# %%
TEST_PERIOD = 21 * 6 # 6 months

TEST_START_DATE = trade_dates[-TEST_PERIOD]
TRAIN_LAST_DATE = trade_dates[-TEST_PERIOD - 1]

# %%
last_date_data = option_data_full[option_data_full['trade_date'] == TRAIN_LAST_DATE]

# %%
# 6개월 연율화 변동성이 25% 이상 35% 미만인 종목들
investment_targets = last_date_data[ (SMA_VOL_LOWER < last_date_data['ret_vol_120d_ann']) & (last_date_data['ret_vol_120d_ann'] < SMA_VOL_UPPER) ]['underlying'].unique()
investment_targets

# %%
# 원래 전체 데이터
option_data_full['underlying'].unique()

# %%
option_data_targets = option_data_full[option_data_full['underlying'].isin(investment_targets)].copy()

# %%
option_data_targets.drop(
    columns=[
        'SMA', 
        'ret_vol_20d', 'ret_vol_60d', 'ret_vol_120d', 'ret_vol_180d',
        'sma_vol_20d', 'sma_vol_60d', 'sma_vol_120d', 'sma_vol_180d',
        'underlying'
    ],
    inplace=True,
)

# %%
option_data_targets = option_data_targets[
    [
        # 옵션 기본 정보
        'underlying_full', # underlying 주식명
        'ticker', # underlying 주식 코드
        'trade_date', # 거래일자
        'expiration', # 만기월 (YYYYMM)
        'expiration_date', # 정확한 만기일자 (해당 월 2번째 목요일)
        'call_or_put', # C/P
        'strike', # 행사가

        # 옵션 가격 정보
        'close_price', # 옵션의 종가
        'open_price', # 옵션의 시가
        'high_price', # 옵션의 고가
        'low_price', # 옵션의 저가
        'im_vol', # 옵션의 내재 변동성
        'next_day_base_price', # 다음 거래일의 옵션 기준가 (특별한 일 없으면 오늘 옵션의 종가와 같음)
        'trade_volume', # 옵션 거래량
        'trade_value', # 옵션 거래대금
        'open_interest_quantity', # 옵션 잔존수량
        
        # 주식 가격 정보
        'udly_open', # 주식 시가
        'udly_high', # 주식 고가
        'udly_low', # 주식 저가
        'udly_close', # 주식 종가
        'udly_volume', # 주식 거래량
        'udly_return', # 주식 수익률
        'ret_vol_120d_ann', # 주식 120일 변동성 (연율화) = 6개월 실현변동성 (realized volatility)

        # moneyness 정보
        'close_strike_diff', # 행사가와 주식 종가의 차이
        'atm', # ATM 여부 (True/False)
        'itm', # ITM 여부 (True/False)
        'otm', # OTM 여부 (True/False)
        'moneyness', # moneyness (= log(strike / udly_close))

    ]
]

# %%
option_data_targets.to_parquet(OUTPUT_PATH / 'option_data_targets_20220101-20241204.parquet')

# %%
