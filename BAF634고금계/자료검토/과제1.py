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
# # 고금계 과제 1 검토

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %% [markdown]
# ## Load & Preprocess data

# %% [markdown]
# ### Dataguide 추출 데이터 

# %%
fn1 = DATA_DIR / '고금계과제1_v3.0_201301-202408.csv'


# %%
## 추출한 fnguide 데이터를 dataframe으로 전처리

def preprocess_dataguide_csv(fn_file_path, cols, skiprows=8, encoding="cp949"):
    fn_df = pd.read_csv(fn_file_path, encoding=encoding, skiprows=skiprows, thousands=",")
    fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")

    return fn_df


# %%
# 날짜가 아닌 컬럼들
cols = ['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency',]

# %%
fn1_df = preprocess_dataguide_csv(fn1, cols, encoding='utf-8')

# %%
fn1_df.head(30
            )

# %%
fn1_df[ fn1_df['Frequency'] == 'DAILY' ]['Item Name '].unique()

# %%
fn1_df['Kind'].unique()

# %%
fn1_df[ fn1_df['Kind'].isna() ] # 날짜 빼고 다 NaN으로 나오는 케이스들 있다. 

# %%
# univ_list = fn1_df['Symbol'].unique() # 나중에 기간 중 존재하지 않았던 종목들을 제외하고 다시 만들 것. 

items = fn1_df['Item Name '].unique() # 원래부터 DataGuide에 띄어쓰기가 들어가 있어서 이렇게 되어버림

# %%
items

# %% [markdown]
# ### mapping 생성

# %%
symbol_to_name = fn1_df[['Symbol', 'Symbol Name']].drop_duplicates().set_index('Symbol').to_dict()['Symbol Name']

# %%
name_to_symbol = {v:k for k, v in symbol_to_name.items()}

# %% [markdown]
# ### 존재하지 않았던 기업 처리
#
# Dataguide에서 상장폐지 종목 포함하여 불러오면 주어진 기간에 존재하지 않았던 기업까지 불러옴. (즉, 전체기간 모든 기업을 univ로 불러옴)
#
# 주어진 기간동안의 존재하지 않았던 주식들의 value 값에 대해선 모두 NaN을 줘버림. 

# %%
name_to_symbol['신한은행'] # 신한지주 출범으로 신한 증권과 함께 2001년 8월 30일 상장폐지. 우리의 데이터 기간엔 아예 존재하지 말았어야 함. 

# %%
name_to_symbol['신한지주'] # 동년 9월 상장됨 


# %%
def get_panel_df(df, item_name):
    panel_df = df.loc[df['Item Name '] == item_name].copy()
    panel_df = panel_df.pivot(index='date', columns='Symbol', values='value')
    panel_df = panel_df.reset_index()
    
    panel_df = panel_df.set_index('date', inplace=False)
    panel_df.sort_index(inplace=True)
    
    return panel_df 


# %%
returns_df = get_panel_df(fn1_df, '수익률(%)')
returns_df.head()

# %%
get_panel_df(fn1_df, '수익률 (1개월)(%)').head() # 이걸 쓰는 것이 맞아보임. 위의 수익률은 일별 수익률인데 그냥 마지막날에 맞춘 것일 가능성이 높아보인다. 

# %%
returns_df.shape

# %%
returns_df.dropna(axis=1, how='all').shape 

# DataGuide에서 데이터 뽑아올 때, 비영업일 제외로 선택하면 월말일이 주말/공휴일일 경우 데이터가 누락됨. 

# %%
returns_df.index

# %%
nans = returns_df.isnull().all()
nan_tickers = nans[nans].index.tolist()

[ symbol_to_name[ticker] for ticker in nan_tickers ] # 모든 값이 NaN인 종목들. 즉, 현재 존재하지 않는 종목들.

# %%
returns_df.dropna(axis=1, how='all', inplace=True)

univ_list = returns_df.columns

# %%
univ_list


# %%
def filter_univ(univ_list, panel_df, is_copy=True):
    if is_copy:
        return panel_df[univ_list].copy()
    else:
        return panel_df[univ_list]


# %% [markdown]
# ## 데이터셋 생성

# %%
items

# %% [markdown]
# #### 그룹

# %%
# WICS Groups

sector_df = filter_univ(univ_list, get_panel_df(fn1_df, 'FnGuide Sector') )
industry_group_df = filter_univ(univ_list, get_panel_df(fn1_df, 'FnGuide Industry Group') )
industry_df = filter_univ(univ_list, get_panel_df(fn1_df, 'FnGuide Industry') )

# %%
sector_df.head()

# %% [markdown]
# ### 시장

# %%
close_df = filter_univ(univ_list, get_panel_df(fn1_df, '종가(원)') ) 
adjclose_df = filter_univ(univ_list, get_panel_df(fn1_df, '수정주가(원)') )
adjfactor_df = filter_univ(univ_list, get_panel_df(fn1_df, '수정계수') )
monthly_returns_df = filter_univ(univ_list, get_panel_df(fn1_df, '수익률 (1개월)(%)') ) # 수익률은 %로 되어있어 /100 해줘야 함.

all_mkt_cap_df = filter_univ(univ_list, get_panel_df(fn1_df, '시가총액 (상장예정주식수 포함)(백만원)') )
common_mkt_cap_df = filter_univ(univ_list, get_panel_df(fn1_df, '시가총액 (보통-상장예정주식수 포함)(백만원)') )
common_shares_outstanding_df = filter_univ(univ_list, get_panel_df(fn1_df, '기말발행주식수 (보통)(주)') )

is_under_supervision_df = filter_univ(univ_list, get_panel_df(fn1_df, '관리종목여부') )
is_trading_halt_df = filter_univ(univ_list, get_panel_df(fn1_df, '거래정지여부') )

# %%
adjclose_df.head()

# %% [markdown]
# #### 재무

# %%
common_stock_df = filter_univ(univ_list, get_panel_df(fn1_df, '보통주자본금(천원)') )
capital_surplus_df = filter_univ(univ_list, get_panel_df(fn1_df, '자본잉여금(천원)') )
retained_earnings_df = filter_univ(univ_list, get_panel_df(fn1_df, '이익잉여금(천원)') )
treasury_stock_df = filter_univ(univ_list, get_panel_df(fn1_df, '자기주식(천원)') )
deferred_tax_liabilities_df = filter_univ(univ_list, get_panel_df(fn1_df, '이연법인세부채(천원)') )
sales_revenue_df = filter_univ(univ_list, get_panel_df(fn1_df, '매출액(천원)') )
cost_of_goods_sold_df = filter_univ(univ_list, get_panel_df(fn1_df, '매출원가(천원)') )
interest_expense_df = filter_univ(univ_list, get_panel_df(fn1_df, '이자비용(천원)') )
operating_profit_df = filter_univ(univ_list, get_panel_df(fn1_df, '영업이익(천원)') )
total_assets_df = filter_univ(univ_list, get_panel_df(fn1_df, '총자산(천원)') )

# %%
total_assets_df.head()

# %% [markdown]
# ## 데이터셋 추가 전처리

# %%
numeric_data = [
    close_df, adjclose_df, adjfactor_df, monthly_returns_df, all_mkt_cap_df, common_mkt_cap_df, common_shares_outstanding_df,
    common_stock_df, capital_surplus_df, retained_earnings_df, treasury_stock_df, deferred_tax_liabilities_df,
    sales_revenue_df, cost_of_goods_sold_df, interest_expense_df, operating_profit_df, total_assets_df
]

# %%
for df in numeric_data:
    obj_cols = df.select_dtypes('object').columns
    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce')


# %%
monthly_returns_df = monthly_returns_df / 100

# %%
is_under_supervision_mapping = {
    '정상': 1,
    '관리': 0,
}

# %%
is_trading_halt_mapping = {
    '정상': 1,
    '정지': 0,
}

# %%
is_under_supervision_df = is_under_supervision_df.replace(is_under_supervision_mapping)

# %%
is_trading_halt_df = is_trading_halt_df.replace(is_trading_halt_mapping)

# %%
