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

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas.tseries.offsets import MonthEnd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'
print(DATA_DIR)
fndata_path = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'

# %%
from fndata import FnStockData
from fndata import FnMarketData

# %%
fn = FnStockData(fndata_path)
#fn은 주식 데이터임
fn.get_items()

# %%
fnmkt_path = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'
# 데이터 모듈을 생성하며 기본 전처리들을 수행합니다.
fnmkt = FnMarketData(fnmkt_path)
# long format 불러오기
fnmkt.get_data(format='long', multiindex=True)

# %%
#천원 단위가 아닌 것은 수정주가, 종가 2개다. 
op=fn.get_data('영업이익(천원)')
#op[op.isna()]
op.head()

# %%
# 무위험이자율 파일을 만든다.
rf_path = DATA_DIR / '통안채1년물_월평균_201301-202408.csv'
rf_df = pd.read_csv(rf_path)
rf_df.head()

# %%
# rf 무위험이자율은 365일 단위이므로, 1개월 단위로 변환해야 한다.
# (1+r/100)의 1/12 승.
rf_df['Rf'] = (1 + (rf_df['원자료']/100)) ** (1/12) - 1
rf_df.head()

# %%
# rf의 날짜가 object이므로, 접근하기 쉽게 datetime으로 변환한다. 일은 월말로 한다.
rf_df['date'] = pd.to_datetime(rf_df['변환'], format='%Y/%m') + MonthEnd(0)
rf_df['date'].head()

# %%
# 필요없는 이전 일자는 없앤다.
rf_df.drop(columns='변환', inplace=True)
rf_df

# %% [markdown]
# #1. 시장가치를 통해 장부가치, B/M구하기

# %%
stock_df=fn.get_data()
stock_df.head()

# %%
# 2023-12-31까지만 데이터로 삼는다. 
stock_df=stock_df[stock_df.index.get_level_values(0)<='2023-12-31']

# %%
stock_df[['자본잉여금(천원)','이익잉여금(천원)','자기주식(천원)','이연법인세부채(천원)']].fillna(0,inplace=True)
stock_df[['자본잉여금(천원)','이익잉여금(천원)','자기주식(천원)','이연법인세부채(천원)']]

# %%
# NA는 없다. 
stock_df[['자본잉여금(천원)','이익잉여금(천원)','자기주식(천원)','이연법인세부채(천원)']].info()

# %%
# 자기자본의 장부가치: t-1년 12월말의 보통주 자본금에 자본잉여금, 이익잉여금, 자기주식, 이연법인세 부채를 더해 측정
stock_df['장부가치 (천원)'] = (stock_df['자본잉여금(천원)'] + stock_df['이익잉여금(천원)'] + stock_df['보통주자본금(천원)']+stock_df['자기주식(천원)']+stock_df['이연법인세부채(천원)'])
#시장가치: 12월 말의 보통주 주가에 발행주식을 곱해 측정한다. 단, 종가는 천원 단위가 아니기에 1000을 나눠준 후 1000단위라고 해준다. 
stock_df['시장가치 (천원)']=(stock_df['종가(원)'] * stock_df['기말발행주식수 (보통)(주)'])/1000
stock_df.head()


# %%
# 장부가치가 na가 나왔다는 것은 보통주 자본금이 NA라는 것.
stock_df['장부가치 (천원)'].isna().sum()

# %%
# 보통주 자본금이 NA인것은 제거한다. 
stock_df=stock_df[stock_df['장부가치 (천원)'].notna()]

# %%
stock_df.head()

# %%
#장부가치 대 시장가치 비율(B/Mi=Bi/(PiXNi))는 자기자본의 장부가치를 시장가치로 나눈다.
stock_df['B/M']=stock_df['장부가치 (천원)']/stock_df['시장가치 (천원)']
stock_df.head()


# %%
'''['종가(원)', '수정계수', '수정주가(원)', '수익률 (1개월)(%)', 'FnGuide Sector',
       '거래정지여부', '관리종목여부', '보통주자본금(천원)', '자본잉여금(천원)', '이익잉여금(천원)',
       '자기주식(천원)', '이연법인세부채(천원)', '매출액(천원)', '매출원가(천원)', '이자비용(천원)',
       '영업이익(천원)', '총자산(천원)', '기말발행주식수 (보통)(주)'], dtype=object)'''

# %%
#일단 B/M 중에서 무한대의 값은 없다. 
# B/M 중에서 np.nan 인것들은 없앤다. 
#stock_df.dropna(subset='B/M', inplace=True)
# B/M의 NaN값 제거 이후 열 별 NaN 개수
stock_df['B/M'].isna().sum()

# %%
stock_df['시장가치 (천원)']

# %%
# 일단 (일자/주식번호)의 멀티인덱스
# 날짜마다 있는 주식들의 중앙값을, 일자별로 groupby를 해서 얻는다
stock_df['size_quantiles'] = stock_df.groupby('date')['시장가치 (천원)'].transform(lambda x: pd.qcut(x, 2, labels=['Small', 'Big']))
stock_df['size_quantiles']

# %%
# B/M이 na인 것 삭제
stock_df=stock_df[stock_df['B/M'].notna()]
stock_df.head()


# %%
def qcut_BM(x):
    try:
        return pd.qcut(x, 3, labels=['Low', 'Mid', 'High'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
stock_df['bm_quantiles'] = stock_df.groupby('date')['B/M'].transform(qcut_BM)
stock_df['bm_quantiles']

# %%
df_smb = stock_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])
small_avg = df_smb[('Small', 'Low')] + df_smb[('Small', 'Mid')] + df_smb[('Small', 'High')]
big_avg = df_smb[('Big', 'Low')] + df_smb[('Big', 'Mid')] + df_smb[('Big', 'High')]

smb = (small_avg / 3) - (big_avg / 3)
smb

# %%
df_hml = stock_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])

high_hml = df_hml[('Small', 'High')] + df_hml[('Big', 'High')]
low_hml = df_hml[('Small', 'Low')] + df_hml[('Big', 'Low')]

hml = (high_hml - low_hml) / 2
hml

# %%
# RMW를 위해 영업이익을 확인해야 하는데, 이 값이 na이면 0처리한다. 
stock_df['영업이익(천원)'].isna().sum()

# %%
# RMW를 위한 수익성 지수 측정. 
stock_df['OP'] = stock_df['영업이익(천원)'] / stock_df['장부가치 (천원)']
stock_df['OP'].isna().sum()


# %%
def qcut_OP(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Weak', 'Neutral', 'Robust'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
stock_df['OP_quantiles'] = stock_df.groupby('date')['OP'].transform(qcut_OP)
stock_df['OP_quantiles']

# %%
df_rmv = stock_df.groupby(['date', 'size_quantiles', 'OP_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'OP_quantiles'])

high_rmw = df_rmv[('Small', 'Robust')] + df_rmv[('Big', 'Robust')]
low_rmw = df_rmv[('Small', 'Weak')] + df_rmv[('Big', 'Weak')]

rmw = (high_rmw - low_rmw) / 2
rmw

# %%
#시장 데이터 전처리
df_mkt=fnmkt.get_data(format='wide')
df_mkt=df_mkt[df_mkt.index.get_level_values(0)<='2023-12-31']
df_mkt=df_mkt['MKF2000']
df_mkt

# %%
rf_df=rf_df[rf_df['date']<='2023-12-31']
rf_df

# %%
mr=pd.merge(rf_df, df_mkt,on='date')
mr


# %%
mr['excess']=mr['MKF2000']-mr['Rf']
mr['excess']


# %%
#지금까지 excess, smb, hml, rmw 제작. 
stock_df['invest'] = stock_df.groupby('date')['총자산(천원)'].transform(lambda x: (x - x.shift(12)) / x.shift(12))


# %%
# 이것만 특별히 3:4:3 구간 간격.
def qcut_invest(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Conservative', 'Neutral', 'Aggressive'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
stock_df['invest_quantiles'] = stock_df.groupby('date')['invest'].transform(qcut_invest)

# %%
cma_data = stock_df.groupby(['date', 'size_quantiles', 'invest_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'invest_quantiles'])

high_invest = cma_data[('Small', 'Aggressive')] + cma_data[('Big', 'Aggressive')]
low_invest = cma_data[('Small', 'Conservative')] + cma_data[('Big', 'Conservative')]

cma = low_invest - high_invest
cma

# %%
# 5팩터 최종
d={'Mkt_RF':mr['excess'].values,'SMB': smb.values,'HML': hml.values,'RMW': rmw.values,'CMA': cma.values}
F_5 = pd.DataFrame(data=d, index=smb.index)
F_5
#_5factors.dropna(how='all', inplace=True)
#_5factors

# %%
F_5.info()

# %%
stock_df['size_quantiles_by5'] = pd.qcut(stock_df['시장가치 (천원)'], 5, labels=['Small', '2', '3', '4', 'Big'])
stock_df['size_quantiles_by5']


# %%
def qcut_BM_by5(x):
    try:
        return pd.qcut(x, 5, labels=['Low', '2', '3', '4', 'High'])
    except ValueError:  # 구간을 나눌 수 없는 경우
        return pd.Series(np.nan, index=x.index)
stock_df['bm_quantiles_by5'] = stock_df.groupby('date')['B/M'].transform(qcut_BM_by5)
stock_df['bm_quantiles_by5']

# %%
stock_df['수익률 (1개월)(%)']

# %%
s=pd.Series(rf_df['Rf'].values,index=smb.index)
s

# %%
stock_df['excess_value'] = stock_df['수익률 (1개월)(%)'] - stock_df.index.get_level_values('date').map(s)

# %%
stock_df.head()

# %%
portfolios = stock_df.groupby(['date', 'size_quantiles_by5', 'bm_quantiles_by5']).apply(
    lambda group: group['excess_value'].mean(skipna=True)
    ).unstack(level=['size_quantiles_by5', 'bm_quantiles_by5'])

# %%
portfolios

# %%
portfolios.columns

# %%
meanss=[portfolios[midx].mean() for midx in portfolios.columns]
final25=pd.DataFrame({'Small':meanss[0:5],'2':meanss[5:10],'3':meanss[10:15],'4':meanss[15:20],'Big':meanss[20:]})
final_25=final25.T
final_25.columns=['Low','2','3','4','High']
final_25

# %%
final_25.loc['Small-Big']=final_25.loc['Small']-final_25.loc['Big']
final_25

# %%
final_25['High-Low']=final_25['High']-final_25['Low']
final_25.loc['Small-Big','High-Low']='.'
final_25

# %% [markdown]
# (재필)
#
# 욱이 데이터 플롯

# %%
(portfolios/100 + 1).cumprod().plot()

# %%
