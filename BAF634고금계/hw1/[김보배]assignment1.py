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
# # 고금계 과제 1

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from fndata import FnStockData
from fndata import FnMarketData

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
fndata_path = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'
mkt_path = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'
rf_path = DATA_DIR / '통안채1년물_월평균_201301-202408.csv'

# %% [markdown]
# ## 0. 데이터 전처리

# %%
fn = FnStockData(fndata_path)
df = fn.get_data()
df

# %%
fnmkt = FnMarketData(mkt_path)
df_mkt = fnmkt.get_data()
df_mkt

# %%
df_rf = pd.read_csv(rf_path)
df_rf.columns = ['date', 'rf']
df_rf['date'] = pd.to_datetime(df_rf['date'], format='%Y/%m') + pd.offsets.MonthEnd(0) # 말일로 변경
df_rf.set_index('date', inplace=True)
df_rf['rf'] = (1 + (df_rf['rf']/100)) ** (1/12) - 1 # 연율화
df_rf

# %% [markdown]
# # 1. Factor Construction

# %% [markdown]
# ## 1.1 MKT-RF
# - MKT : MKT2000
# - RF : 통안채1년물_월평균

# %%
df_mkt = df_mkt.xs('MKF2000', level='Symbol Name')
df_mkt.columns = ['mkt']
df_mkt

# %%
df_mkt_rf= pd.concat([df_mkt, df_rf], axis=1)
df_mkt_rf['mkt_rf'] = df_mkt_rf['mkt'] - df_mkt_rf['rf']

# %%
df['수익률 (1개월)(%)'] = df['수익률 (1개월)(%)'] * 0.01 # 퍼센트를 소수로 변경
df['excess_rets'] = df['수익률 (1개월)(%)'] - df_rf['rf'] # 2024-09-19 빼고는 존재함

# %% [markdown]
# ## 1.2. SMB
# - 시장가치 : t년 12월 말의 보통주 주가에 발행주식을 곱해 측정한다.
# - 자기자본의 장부가치: t-1년 12월말의 보통주 자본금에 자본잉여금, 이익잉여금, 자기주식, 이연법인세 부채를 더해 측정
# - 장부가치 대 시장가치 비율(B/Mi=Bi/(PiXNi))는 자기자본의 장부가치를 시장가치로 나눈다.
#
# > 주의) 각 시점마다 independent sort

# %%
# 종가가 없으면 거래가 되지 않았다고 판단하여 nan
df['시가총액'] = df['종가(원)'] * df['기말발행주식수 (보통)(주)']

# %%
df['size_quantiles'] = df.groupby('date')['시가총액'].transform(lambda x: pd.qcut(x, 2, labels=['Small', 'Big']))
df['size_quantiles']

# %%
df['Book'] = df['보통주자본금(천원)'].fillna(0) + df['자본잉여금(천원)'].fillna(0) + df['이익잉여금(천원)'].fillna(0) - df['자기주식(천원)'].fillna(0) + df['이연법인세부채(천원)'].fillna(0)
df['BM'] = df['Book'] / df['시가총액']


# %%
def qcut_BM(x):
    try:
        return pd.qcut(x, 3, labels=['Low', 'Mid', 'High'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
df['bm_quantiles'] = df.groupby('date')['BM'].transform(qcut_BM)
df['bm_quantiles']

# %%
df_smb = df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])
small_avg = df_smb[('Small', 'Low')] + df_smb[('Small', 'Mid')] + df_smb[('Small', 'High')]
big_avg = df_smb[('Big', 'Low')] + df_smb[('Big', 'Mid')] + df_smb[('Big', 'High')]

smb = (small_avg / 3) - (big_avg / 3)
smb

# %% [markdown]
# ## 1.3 HML

# %%
df_hml = df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])

high_hml = df_hml[('Small', 'High')] + df_hml[('Big', 'High')]
low_hml = df_hml[('Small', 'Low')] + df_hml[('Big', 'Low')]

hml = (high_hml - low_hml) / 2
hml

# %% [markdown]
# ## 1.4 RMW
# - t-1년 12말의 매출액에서 매출원가, 이자비용, 판매및관리비를 차감한 영업이익을 t-1년 12월 말의 보통주 (자기자본) 장부가치로 나누어 측정

# %%
df['OP'] = df['영업이익(천원)'].fillna(0) / df['Book']


# %%
def qcut_OP(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Weak', 'Neutral', 'Robust'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
df['OP_quantiles'] = df.groupby('date')['OP'].transform(qcut_OP)
df['OP_quantiles']

# %%
df_rmv = df.groupby(['date', 'size_quantiles', 'OP_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'OP_quantiles'])

high_rmw = df_rmv[('Small', 'Robust')] + df_rmv[('Big', 'Robust')]
low_rmw = df_rmv[('Small', 'Weak')] + df_rmv[('Big', 'Weak')]

rmw = (high_rmw - low_rmw) / 2
rmw

# %% [markdown]
# ## 1.5 CMA
# -  t-1년 12월 말의 총자산에서 t-2년 12월말의 총자산을 차감한 총자산증가액을 t-2년 12월 말의 총자산으로 나누어서 측정.

# %%
df['invest'] = df.groupby('date')['총자산(천원)'].transform(lambda x: (x - x.shift(12)) / x.shift(12))


# %%
def qcut_invest(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Conservative', 'Neutral', 'Aggressive'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
df['invest_quantiles'] = df.groupby('date')['invest'].transform(qcut_invest)

# %%
cma_data = df.groupby(['date', 'size_quantiles', 'invest_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'invest_quantiles'])

high_invest = cma_data[('Small', 'Aggressive')] + cma_data[('Big', 'Aggressive')]
low_invest = cma_data[('Small', 'Conservative')] + cma_data[('Big', 'Conservative')]

cma = low_invest - high_invest
cma

# %% [markdown]
# ## 1.6 UMD
# - (전월말 주가 – 1년전 월말 주가) / 1년전 주가
# - 보유기간이 1개월이며, 매월 리밸런싱하며, 상위 30%가 Winner(UP)이며, 하위 30%가 Loser(DOWN)이다.

# %%
df['Momentum'] = df.groupby('date')['수정주가(원)'].transform(lambda x: (x - x.shift(12)) / x.shift(12))
df['Momentum_rank'] = df.groupby('date')['Momentum'].transform(lambda x: pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Loser', 'Middle', 'Winner']))
umd = df.groupby(['date', 'Momentum_rank'])['수익률 (1개월)(%)'].mean().unstack()
umd['WML'] = umd['Winner'] - umd['Loser']
umd

# %% [markdown]
# ## 1.7 STR
# - reversal(최근월의 수익률 기반으로)

# %%
df['1M_Return'] = df.groupby('date')['수정주가(원)'].transform(lambda x: x.pct_change())
df['Reversal_rank'] = df.groupby('date')['1M_Return'].transform(lambda x: pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Winner', 'Middle', 'Loser']))
str = df.groupby(['date', 'Reversal_rank'])['수익률 (1개월)(%)'].mean().unstack()
str['WML'] = str['Winner'] - str['Loser']
str

# %% [markdown]
# # 2. Output
# ## 2.1 output1) 25 size BEME Portfolios

# %%
df['size_quantiles_by5'] = pd.qcut(df['시가총액'], 5, labels=['Small', '2', '3', '4', 'Big'])
df['size_quantiles_by5']


# %%
def qcut_BM_by5(x):
    try:
        return pd.qcut(x, 5, labels=['Low', '2', '3', '4', 'High'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
df['bm_quantiles_by5'] = df.groupby('date')['BM'].transform(qcut_BM_by5)
df['bm_quantiles_by5']

# %%
portfolios = df.groupby(['date', 'size_quantiles_by5', 'bm_quantiles_by5']).apply(
    lambda group: group['excess_rets'].mean(skipna=True)
    ).unstack(level=['size_quantiles_by5', 'bm_quantiles_by5'])

# %%
portfolios  # book value가 2024-06-30까지 존재함

# %% [markdown]
# ## 2.2 output 2) Fama-French 3factors

# %%
_3factors = pd.DataFrame({
    'Mkt_RF': df_mkt_rf['mkt_rf'],
    'SMB': smb,
    'HML': hml,
    'RF' : df_rf['rf'],
    'UMD': umd['WML']
    })
_3factors.dropna(how='all', inplace=True)
_3factors

# %% [markdown]
# ## 2-3 output 3) Fama-French 5Factors

# %%
_5factors = pd.DataFrame({
    'Mkt_RF': df_mkt_rf['mkt_rf'],
    'SMB': smb,
    'HML': hml,
    'RMW': rmw,
    'CMA': cma,
    'RF' : df_rf['rf'],
    'UMD': umd['WML'],
    'STR': str['WML']
})
_5factors.dropna(how='all', inplace=True)
_5factors

# %% [markdown]
# (재필)
#
# 보배 데이터 plot

# %%
cols = [
    'Mkt_RF',
    'SMB',
    'HML',
    'RMW',
    'CMA',
    'UMD',
    'STR',
    'RF'
]

(_5factors[cols] + 1).cumprod().plot()

# %%
