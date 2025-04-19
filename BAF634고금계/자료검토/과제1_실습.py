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
# # 파마프렌치 모델 만들기

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
## custom libs
from fndata import FNData

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %% [markdown]
# ## 데이터 불러오기

# %%
fn1 = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'

# %%
fnd = FNData(fn1)

# %%
items = fnd.get_items()

# %%
items

# %%
close_df = fnd.get_data('종가(원)')
adjclose_df = fnd.get_data('수정주가(원)')
adjfactor_df = fnd.get_data('수정계수')
monthly_returns_df = fnd.get_data('수익률 (1개월)(%)')


# %%

# common_shares_listed_df = fnd.get_data('상장주식수 (보통)(주)')

# all_mkt_cap_df = fnd.get_data('시가총액 (상장예정주식수 포함)(백만원)')
# common_mkt_cap_df = fnd.get_data('시가총액 (보통-상장예정주식수 포함)(백만원)')
common_shares_outstanding_df = fnd.get_data('기말발행주식수 (보통)(주)')

common_stock_df = fnd.get_data('보통주자본금(천원)')
capital_surplus_df = fnd.get_data('자본잉여금(천원)')
retained_earnings_df = fnd.get_data('이익잉여금(천원)')
treasury_stock_df = fnd.get_data('자기주식(천원)')
deferred_tax_liabilities_df = fnd.get_data('이연법인세부채(천원)')
sales_revenue_df = fnd.get_data('매출액(천원)')
cost_of_goods_sold_df = fnd.get_data('매출원가(천원)')
interest_expense_df = fnd.get_data('이자비용(천원)')
operating_profit_df = fnd.get_data('영업이익(천원)')
total_assets_df = fnd.get_data('총자산(천원)')


# %%
sales_revenue_df.head()

# %%
# common_shares_listed_df * close_df

# %%
common_shares_outstanding_df * close_df

# %%
# common_mkt_cap_df

# %%
adjclose_df.count(axis=1).plot()

# %%
sector_df = fnd.get_data('FnGuide Sector')

# %%
sector_df.count(axis=1).plot()

# %%
last_sector = sector_df.iloc[-1]
last_sector[last_sector != ''].count()

# %%
last_sector.count()

# %% [markdown]
# ## Factor Construction

# %%
