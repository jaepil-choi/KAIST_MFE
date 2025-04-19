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

# %% colab={"base_uri": "https://localhost:8080/"} id="BfDUIK2XfENW" outputId="5d928d52-3d8b-486e-a6e7-58ad6f90dfbe"
# from google.colab import drive
# drive.mount('/content/drive')

# %% id="3cjNFcszfYk5"
# import os
# os.chdir("/content/drive/MyDrive/고금계")

# %% id="7efaee34-1c13-45ec-b1c6-d999c3268c20"
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

# %% id="7b898adc-1f10-4e70-9b28-f4a3a7af0656"
#주식데이터 소환
stocks_df = fn.get_data()
stocks_df = stocks_df.loc[stocks_df.index.get_level_values('date') < '2024-01-31']
# 시장데이터 소환
market_df = fnmkt.get_data(format='long', multiindex=True)
market_df = market_df.loc[market_df.index.get_level_values('date') < '2024-01-31']

# rf 데이터 소환
df_rf = pd.read_csv(rf_path)
df_rf.columns = ['date', 'rf']
df_rf['date'] = pd.to_datetime(df_rf['date'], format='%Y/%m') + pd.offsets.MonthEnd(0) # 말일로 변경
df_rf.set_index('date', inplace=True)
df_rf['rf'] = (1 + (df_rf['rf']/100)) ** (1/12) - 1 # 연율화
df_rf = df_rf.loc[df_rf.index < '2024-01-31']

# 올해 데이터 전부 절삭시키고 시작.
CUT_DATE = '2023-12-31'
stocks_df = stocks_df[stocks_df.index.get_level_values('date') <= CUT_DATE]
market_df = market_df[market_df.index.get_level_values('date') <= CUT_DATE]
df_rf = df_rf[df_rf.index <= CUT_DATE]

# %% [markdown] id="ee2134db-4dd8-4f53-92bf-5ce23f53746f"
# # Market cap

# %% id="64b77aaf-da0b-4cb0-8f50-d1aec93a7dea"
# 발행주식 데이터(보통주) : 1원단위
stocks_df['Market cap']=(stocks_df['종가(원)'] * stocks_df['기말발행주식수 (보통)(주)'])

#장부가치, 이거 좀 서로 많이 다름. 논의해야 하는 부분
'''
승한이한테 입수한 정보 : 현재 장부 관련 데이터에서 빵꾸난 애들이 있는데,
1.하나라도 빵구나면 그냥 버린다.(종목을 버린다. drop)
2.아래처럼 fillna치고 넘어간다.
3.보간법 쓴다.
중 하나 택일해야 함.
'''
#연말 시점??인거 맞는거 확인해야 함.
stocks_df['Bookvalue'] = stocks_df['보통주자본금(천원)'] \
                            + stocks_df['자본잉여금(천원)'].fillna(0) \
                            + stocks_df['이익잉여금(천원)'].fillna(0) \
                            + stocks_df['자기주식(천원)'].fillna(0) \
                            + stocks_df['이연법인세부채(천원)'].fillna(0)

stocks_df['BM'] = stocks_df['Bookvalue'] / stocks_df['Market cap']
#보통주자본금(천원) dropna!
stocks_df.dropna(subset='BM', inplace=True)

# qcut_BM 함수 수정
def qcut_BM(x):
    if x.dropna().empty:
        return pd.Series(np.nan, index=x.index)
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Growth', 'Neutral', 'Value'])
        # return pd.qcut(x, 3, labels=['Growth', 'Neutral', 'Value'])
    except (ValueError, IndexError):  # ValueError와 IndexError 모두 처리
        return pd.Series(np.nan, index=x.index)

stocks_df['bm_quantiles'] = stocks_df.groupby('date')['BM'].transform(qcut_BM)

###
#일단 영업이익 dropna 하지말고 남겨두자. mom이나 rev에서 팩터값 존재하는 경우 있으니 남겨두자.
###
stocks_df['OP'] = stocks_df['영업이익(천원)'].fillna(0) / stocks_df['Bookvalue']#12월말 보통주 장부가치인가?
def qcut_OP(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Weak', 'Neutral', 'Robust'])
    except (ValueError, IndexError):  # ValueError와 IndexError 모두 처리
        return pd.Series(np.nan, index=x.index)
stocks_df['OP_quantiles'] = stocks_df.groupby('date')['OP'].transform(qcut_OP)
stocks_df['OP_quantiles']

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="ce565ce1-5655-4b59-b779-68a44502dd83" outputId="809ef188-9a7a-40c9-aba1-ca2a6f73a170"
'''
MKF2000 사용함.
'''
market_df = market_df.xs('MKF2000', level='Symbol Name')
market_df.columns = ['mkt']
market_df= pd.concat([market_df, df_rf], axis=1)
market_df['mkt_rf'] = market_df['mkt'] - market_df['rf']
market_df

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="eda9fa7d-085b-488c-858a-69ce2c53a343" outputId="c6b2aeae-ecf5-4f22-a952-6fbdb7a38d01"
# stocks_df['size_quantiles'] = stocks_df.groupby('date')['Market cap'].transform(lambda x: pd.qcut(x, 2, labels=['Small', 'Big']))
# df_smb = stocks_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])
# small_avg = df_smb[('Small', 'Value')] + df_smb[('Small', 'Neutral')] + df_smb[('Small', 'Growth')]
# big_avg = df_smb[('Big', 'Value')] + df_smb[('Big', 'Neutral')] + df_smb[('Big', 'Growth')]
# smb = (small_avg / 3) - (big_avg / 3)
# smb

# B/M에 따른 SMB 계산
stocks_df['size_quantiles'] = stocks_df.groupby('date')['Market cap'].transform(lambda x: pd.qcut(x, 2, labels=['Small', 'Big']))
df_smb_bm = stocks_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])
small_bm_avg = df_smb_bm[('Small', 'Value')] + df_smb_bm[('Small', 'Neutral')] + df_smb_bm[('Small', 'Growth')]
big_bm_avg = df_smb_bm[('Big', 'Value')] + df_smb_bm[('Big', 'Neutral')] + df_smb_bm[('Big', 'Growth')]
smb_bm = (small_bm_avg / 3) - (big_bm_avg / 3)

df_smb_op = stocks_df.groupby(['date', 'size_quantiles', 'OP_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'OP_quantiles'])
small_op_avg = df_smb_op[('Small', 'Robust')] + df_smb_op[('Small', 'Neutral')] + df_smb_op[('Small', 'Weak')]
big_op_avg = df_smb_op[('Big', 'Robust')] + df_smb_op[('Big', 'Neutral')] + df_smb_op[('Big', 'Weak')]
smb_op = (small_op_avg / 3) - (big_op_avg / 3)

# INV에 따른 SMB 계산 (자본투자 기준)
# 총자산 변화율로 INV 계산
stocks_df['INV'] = stocks_df.groupby('date')['총자산(천원)'].transform(lambda x: x / x.shift(1) - 1)

# INV에 따라 'Conservative', 'Neutral', 'Aggressive'로 나누기
def qcut_INV(x):
    if x.dropna().empty:
        return pd.Series(np.nan, index=x.index)
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Aggressive', 'Neutral', 'Conservative'])
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=x.index)

stocks_df['inv_quantiles'] = stocks_df.groupby('date')['INV'].transform(qcut_INV)
df_smb_inv = stocks_df.groupby(['date', 'size_quantiles', 'inv_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'inv_quantiles'])
small_inv_avg = df_smb_inv[('Small', 'Conservative')] + df_smb_inv[('Small', 'Neutral')] + df_smb_inv[('Small', 'Aggressive')]
big_inv_avg = df_smb_inv[('Big', 'Conservative')] + df_smb_inv[('Big', 'Neutral')] + df_smb_inv[('Big', 'Aggressive')]
smb_inv = (small_inv_avg / 3) - (big_inv_avg / 3)

# 최종 SMB는 동일가중 평균으로 결합
smb = (smb_bm + smb_op + smb_inv) / 3
smb

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="HOiVfLYSnkTT" outputId="14e02727-3be7-4a45-eb39-a02583fb2248"
df_hml = stocks_df.groupby(['date', 'size_quantiles', 'bm_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'bm_quantiles'])

high_hml = df_hml[('Small', 'Value')] + df_hml[('Big', 'Value')]
low_hml = df_hml[('Small', 'Growth')] + df_hml[('Big', 'Growth')]

hml = (high_hml - low_hml) / 2
hml

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="diuV8uRjoo8d" outputId="a24fe575-58a8-4698-ce7c-521aa35acc43"
df_rmv = stocks_df.groupby(['date', 'size_quantiles', 'OP_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'OP_quantiles'])

high_rmw = df_rmv[('Small', 'Robust')] + df_rmv[('Big', 'Robust')]
low_rmw = df_rmv[('Small', 'Weak')] + df_rmv[('Big', 'Weak')]

rmw = (high_rmw - low_rmw) / 2
rmw

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="URP_0F9fopct" outputId="7c94e33f-bede-4af7-db37-a2cd3975ec76"
stocks_df['invest'] = stocks_df.groupby('date')['총자산(천원)'].transform(lambda x: (x - x.shift(12)) / x.shift(12))

def qcut_invest(x):
    try:
        return pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Conservative', 'Neutral', 'Aggressive'])
    except (ValueError, IndexError):  # ValueError와 IndexError 모두 처리
        return pd.Series(np.nan, index=x.index)


stocks_df['invest_quantiles'] = stocks_df.groupby('date')['invest'].transform(qcut_invest)

cma_data = stocks_df.groupby(['date', 'size_quantiles', 'invest_quantiles'])['수익률 (1개월)(%)'].mean().unstack(['size_quantiles', 'invest_quantiles'])

high_invest = cma_data[('Small', 'Aggressive')] + cma_data[('Big', 'Aggressive')]
low_invest = cma_data[('Small', 'Conservative')] + cma_data[('Big', 'Conservative')]

cma = (low_invest - high_invest)/2
cma

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="lBqc7a3OrVRh" outputId="8ccd0664-4e2b-43b4-ece3-a50b90483150"
stocks_df['Momentum'] = stocks_df.groupby('date')['수정주가(원)'].transform(lambda x: (x.shift(1) - x.shift(12)) / x.shift(12))
stocks_df['Momentum_rank'] = stocks_df.groupby('date')['Momentum'].transform(lambda x: pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Loser', 'Middle', 'Winner']))
umd = stocks_df.groupby(['date', 'Momentum_rank'])['수익률 (1개월)(%)'].mean().unstack()
umd['WML'] = umd['Winner'] - umd['Loser']
umd

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="_SIJKXiWsfUr" outputId="2c733ecd-c902-42ae-e8c5-71cca7fff198"
stocks_df['1M_Return'] = stocks_df.groupby('date')['수정주가(원)'].transform(lambda x: x.pct_change())
stocks_df['Reversal_rank'] = stocks_df.groupby('date')['1M_Return'].transform(lambda x: pd.qcut(x, [0, 0.3, 0.7, 1.0], labels=['Winner', 'Middle', 'Loser']))
str = stocks_df.groupby(['date', 'Reversal_rank'])['수익률 (1개월)(%)'].mean().unstack()
str['WML'] = str['Winner'] - str['Loser']
str

# %% [markdown]
# # 5*5 만들기(independent, dependent 택1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="PTaKGh0n4aJ_" outputId="83d1b445-4b84-4c18-b362-225177b587ae"
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

# %% id="HuPmHa6G4tnJ"
stocks_df['excess_rets'] = stocks_df['수익률 (1개월)(%)'] - df_rf['rf'] # 2024-09-19 빼고는 존재함????
portfolios = stocks_df.groupby(['date', 'size_quantiles_by5', 'bm_quantiles_by5']).apply(
    lambda group: group['excess_rets'].mean(skipna=True)
    ).unstack(level=['size_quantiles_by5', 'bm_quantiles_by5'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="kWOrCZBv4_sb" outputId="7d9fe6ab-2669-4a51-a0c4-a032291f3a0e"
_3factors = pd.DataFrame({
    'Mkt_RF': market_df['mkt_rf'],
    'SMB': smb,
    'HML': hml,
    'RF' : df_rf['rf'],
    'UMD': umd['WML']
    })
_3factors.dropna(how='all', inplace=True)
_3factors

# %% colab={"base_uri": "https://localhost:8080/", "height": 701} id="oXgkQclL7KZJ" outputId="f1109b26-4b9e-41ca-9cb6-d94793a26fe8"
_5factors = pd.DataFrame({
    'Mkt_RF': market_df['mkt_rf'],
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="LoyAHhSi7Mf4" outputId="0eb682b7-9563-4869-c4f5-76142a640862"
_5factors.describe()


# %%
def double_sorting(df, size_col, bm_col, method='independent'):
    if method == 'independent':
        # Independent double sorting: 각 변수를 독립적으로 소팅
        df['size_sorted'] = df.groupby('date')[size_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4, 5]))
        df['bm_sorted'] = df.groupby('date')[bm_col].transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4, 5]))
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

# %% id="Pz-AQ38MHxnL"
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
        df_date = df_date.dropna(subset=factors + ['수익률 (1개월)(%)'])

        X = df_date[factors].values  # 요인 변수들
        y = df_date['수익률 (1개월)(%)'].values  # 개별 자산의 수익률

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
                portfolio_key = f'{size}{bm}'
                portfolio_df = df_rebalanced[(df_rebalanced['size_sorted'] == size) & (df_rebalanced['bm_sorted'] == bm)]
                
                portfolio_return = portfolio_df['수익률 (1개월)(%)'].mean() / 100
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
    df['excess_return'] = df['수익률 (1개월)(%)'] - df['rf']
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
                row.append(f'{avg_return:.2f} ({t_value:.2f})')  # 평균 수익률과 t값을 함께 표기
            else:
                row.append(f'{avg_return:.2f} (N/A)')  # 회귀 결과가 없는 경우 N/A로 표기
        table_data.append(row)

    # High-Low 차이 계산 (각 size별로 High-Low 차이 추가)
    for i, size in enumerate(sizes):
        high_return = df[df['portfolio_key'] == f'{size}5']['excess_return'].mean()  # High
        low_return = df[df['portfolio_key'] == f'{size}1']['excess_return'].mean()  # Low
        high_low_diff = high_return - low_return
        table_data[i].append(f'{high_low_diff:.2f}')

    # Small-Big 차이 계산
    row = []
    for bm in bms:
        small_return = df[df['portfolio_key'] == f'11{bm}']['excess_return'].mean()  # Small
        big_return = df[df['portfolio_key'] == f'51{bm}']['excess_return'].mean()  # Big
        small_big_diff = small_return - big_return
        row.append(f'{small_big_diff:.2f}')
    table_data.append(row)
    
    # 테이블 열과 행 정의
    columns = ['Low', '2', '3', '4', 'High', 'High-Low']
    index = ['Small', '2', '3', '4', 'Big', 'Small-Big']

    results_df = pd.DataFrame(table_data, columns=columns, index=index)
    # High-Low 차이 계산 및 추가 (소수점 2자리로 포맷팅)
    results_df['High-Low'] = (results_df['High'].apply(lambda x: float(x.split(' ')[0])) - results_df['Low'].apply(lambda x: float(x.split(' ')[0])))
    results_df['High-Low'] = results_df['High-Low'].apply(lambda x: f'{x:.2f}')  # 소수점 2자리로 포맷

    # Small-Big 차이 계산 및 추가
    small_big_diff = []
    columns = ['Low', '2', '3', '4', 'High']
    for col in columns:
        small_return_str = results_df.loc['Small', col]
        big_return_str = results_df.loc['Big', col]
        
        # 수익률만 추출
        small_return = float(small_return_str.split(' ')[0])
        big_return = float(big_return_str.split(' ')[0])
        
        small_big_diff.append(f'{small_return - big_return:.2f}')  # 소수점 2자리로 포맷
    
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

# %%
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
        df_date = df_date.dropna(subset=factors + ['수익률 (1개월)(%)'])

        X = df_date[factors].values  # 요인 변수들
        y = df_date['수익률 (1개월)(%)'].values  # 개별 자산의 수익률

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
                portfolio_key = f'{size}{bm}'
                portfolio_df = df_rebalanced[(df_rebalanced['size_sorted'] == size) & (df_rebalanced['bm_sorted'] == bm)]
                
                portfolio_return = portfolio_df['수익률 (1개월)(%)'].mean() / 100
                portfolio_returns[portfolio_key].append(portfolio_return)
    
    # Calculate cumulative returns for each portfolio
    for portfolio_key, returns in portfolio_returns.items():
        returns = np.array(returns)
        cumulative_returns[portfolio_key] = np.cumprod(1 + returns) - 1

    return cumulative_returns, rebalanced_dates

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
    df['excess_return'] = df['수익률 (1개월)(%)'] - df['rf']
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
                row.append(f'{avg_return:.2f} ({t_value:.2f})')  # 평균 수익률과 t값을 함께 표기
            else:
                row.append(f'{avg_return:.2f} (N/A)')  # 회귀 결과가 없는 경우 N/A로 표기
        table_data.append(row)

    # High-Low 차이 계산 (각 size별로 High-Low 차이 추가)
    for i, size in enumerate(sizes):
        high_return = df[df['portfolio_key'] == f'{size}5']['excess_return'].mean()  # High
        low_return = df[df['portfolio_key'] == f'{size}1']['excess_return'].mean()  # Low
        high_low_diff = high_return - low_return
        table_data[i].append(f'{high_low_diff:.2f}')

    # Small-Big 차이 계산
    row = []
    for bm in bms:
        small_return = df[df['portfolio_key'] == f'11{bm}']['excess_return'].mean()  # Small
        big_return = df[df['portfolio_key'] == f'51{bm}']['excess_return'].mean()  # Big
        small_big_diff = small_return - big_return
        row.append(f'{small_big_diff:.2f}')
    table_data.append(row)
    
    # 테이블 열과 행 정의
    columns = ['Low', '2', '3', '4', 'High', 'High-Low']
    index = ['Small', '2', '3', '4', 'Big', 'Small-Big']

    results_df = pd.DataFrame(table_data, columns=columns, index=index)
    # High-Low 차이 계산 및 추가 (소수점 2자리로 포맷팅)
    results_df['High-Low'] = (results_df['High'].apply(lambda x: float(x.split(' ')[0])) - results_df['Low'].apply(lambda x: float(x.split(' ')[0])))
    results_df['High-Low'] = results_df['High-Low'].apply(lambda x: f'{x:.2f}')  # 소수점 2자리로 포맷

    # Small-Big 차이 계산 및 추가
    small_big_diff = []
    columns = ['Low', '2', '3', '4', 'High']
    for col in columns:
        small_return_str = results_df.loc['Small', col]
        big_return_str = results_df.loc['Big', col]
        
        # 수익률만 추출
        small_return = float(small_return_str.split(' ')[0])
        big_return = float(big_return_str.split(' ')[0])
        
        small_big_diff.append(f'{small_return - big_return:.2f}')  # 소수점 2자리로 포맷
    
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
    
    # 백테스팅 및 누적 수익률 계산
    cumulative_returns, rebalanced_dates = backtest_portfolio(stocks_df, rebalancing_period)
    
    # 누적 수익률 시각화
    visualize_cumulative_returns(cumulative_returns, rebalanced_dates)
    
    # 회귀분석 수행
    regression_results = run_regression(stocks_df, 'mkt_rf')
    
    # 결과 테이블 생성
    results_df = generate_results_table(stocks_df, regression_results)
    
    return results_df

# 최종 실행
results_df = run_backtest_and_create_table(stocks_df, df_rf, market_df, rebalancing_period='M')
results_df


# %%
