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
# # 고금계 과제 1 데이터 정리 

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
# ## 데이터 로드

# %%
fn1 = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'


# %%
## 추출한 fnguide 데이터를 dataframe으로 전처리

def preprocess_dataguide_csv(
        fn_file_path, 
        cols=['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency',], # 날짜가 아닌 컬럼들
        skiprows=8, 
        encoding="cp949",
        ):
    fn_df = pd.read_csv(fn_file_path, encoding=encoding, skiprows=skiprows, thousands=",")
    fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")

    return fn_df


# %%
fn1_df = preprocess_dataguide_csv(fn1, encoding='utf-8')

# %%
fn1_df

# %%
items = fn1_df['Item Name '].unique() # 원래부터 DataGuide에 띄어쓰기가 들어가 있어서 'Item Name '

# %%
# Mappings

symbol_to_name = fn1_df[['Symbol', 'Symbol Name']].drop_duplicates().set_index('Symbol').to_dict()['Symbol Name']
name_to_symbol = {v:k for k, v in symbol_to_name.items()}


# %%
def get_panel_df(molten_df, item_name):
    panel_df = molten_df.loc[molten_df['Item Name '] == item_name]
    panel_df = panel_df.pivot(index='date', columns='Symbol', values='value')
    panel_df = panel_df.reset_index()
    
    panel_df = panel_df.set_index('date', inplace=False)
    panel_df.sort_index(inplace=True)
    
    return panel_df 


# %% [markdown]
# ## 전처리 (1차)

# %% [markdown]
# ### 기간 내 존재하지 않는 기업 제외

# %%
adj_close_temp = get_panel_df(fn1_df, '수정주가(원)')

# %%
adj_close_temp.head()

# %%
adj_close_temp.shape

# %%
adj_close_temp.dropna(axis=1, how='all', inplace=True)

# %%
adj_close_temp.shape

# %%
# 분석 기간 내 존재했던 종목들
univ_list = adj_close_temp.columns


# %% [markdown]
# ### 기타 조건별 제외

# %%
def filter_univ(univ_list, panel_df, is_copy=True):
    if is_copy:
        return panel_df[univ_list].copy()
    else:
        return panel_df[univ_list]


# %% [markdown]
# #### 금융주 제외

# %%

# %%
sector_all_df = get_panel_df(fn1_df, 'FnGuide Sector')
sector_all_df.count(axis=1).plot()

# %%
sector_df = filter_univ(univ_list, get_panel_df(fn1_df, 'FnGuide Sector') )
sector_df.head()

# %%
# 섹터는 고정되어있지 않고 중간에 바뀌기도 함. 
sector_df.nunique()[sector_df.nunique() != 1].sort_values(ascending=False)

# %%
univ_df = ~sector_df.isnull() & (sector_df != '금융')

# %% [markdown]
# #### 관리종목, 거래정지 제외

# %%
is_under_supervision_df = filter_univ(univ_list, get_panel_df(fn1_df, '관리종목여부') )
is_trading_halt_df = filter_univ(univ_list, get_panel_df(fn1_df, '거래정지여부') )

# %%
is_under_supervision_mapping = {
    '정상': True,
    '관리': False,
}
is_trading_halt_mapping = {
    '정상': True,
    '정지': False,
}

# %%
is_under_supervision_df = is_under_supervision_df.replace(is_under_supervision_mapping).infer_objects(copy=False)
is_trading_halt_df = is_trading_halt_df.replace(is_trading_halt_mapping).infer_objects(copy=False)

# %%
univ_df = univ_df & is_under_supervision_df & is_trading_halt_df

# %%
# Update univ_list
univ_list = univ_df.columns

# %% [markdown]
# ## 데이터셋 생성

# %% [markdown]
# ### 시장

# %%
close_df = filter_univ(univ_list, get_panel_df(fn1_df, '종가(원)') ) 
adjclose_df = filter_univ(univ_list, get_panel_df(fn1_df, '수정주가(원)') )
adjfactor_df = filter_univ(univ_list, get_panel_df(fn1_df, '수정계수') )
monthly_returns_df = filter_univ(univ_list, get_panel_df(fn1_df, '수익률 (1개월)(%)') ) # 수익률은 %로 되어있어 뒤에서 /100 해줘야 함.

all_mkt_cap_df = filter_univ(univ_list, get_panel_df(fn1_df, '시가총액 (상장예정주식수 포함)(백만원)') )
common_mkt_cap_df = filter_univ(univ_list, get_panel_df(fn1_df, '시가총액 (보통-상장예정주식수 포함)(백만원)') )
common_shares_outstanding_df = filter_univ(univ_list, get_panel_df(fn1_df, '기말발행주식수 (보통)(주)') )

is_under_supervision_df = filter_univ(univ_list, get_panel_df(fn1_df, '관리종목여부') )
is_trading_halt_df = filter_univ(univ_list, get_panel_df(fn1_df, '거래정지여부') )

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
total_assets_df

# %% [markdown]
# ## 전처리 (2차)

# %% [markdown]
# ### 형변환

# %%
numeric_data = [
    close_df, adjclose_df, adjfactor_df, monthly_returns_df, all_mkt_cap_df, common_mkt_cap_df, common_shares_outstanding_df,
    common_stock_df, capital_surplus_df, retained_earnings_df, treasury_stock_df, deferred_tax_liabilities_df,
    sales_revenue_df, cost_of_goods_sold_df, interest_expense_df, operating_profit_df, total_assets_df
]

# %%
for df in numeric_data:
    obj_cols = df.select_dtypes('object').columns
    df[obj_cols] = df[obj_cols].replace(',', '', regex=True).infer_objects(copy=False) 
    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce')


# %% [markdown]
# ### 단위 통일

# %%
monthly_returns_df = monthly_returns_df / 100 # 수익률은 %로 되어있어 /100

# %%
all_mkt_cap_df = all_mkt_cap_df * 100 # 시가총액은 100만원 단위라 *100하여 천원 단위로 맞춰줌
common_mkt_cap_df = common_mkt_cap_df * 100

# %%
all_mkt_cap_df

# %%
