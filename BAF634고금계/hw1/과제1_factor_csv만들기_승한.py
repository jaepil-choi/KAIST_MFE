# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: work
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from portsort import portsort

import matplotlib.pyplot as plt

from pathlib import Path
from fndata import FnStockData
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.offsets import YearEnd
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'
fndata_path = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'
fn = FnStockData(fndata_path)
df = fn.get_data()



# %%
df

# %%
df.to_csv('df.csv')

# %%
df['FnGuide Sector'].value_counts()

# %% [markdown]
# # BM

# %%
df2=df.reset_index().copy()
year_end=sorted(list(set(df2['date']+YearEnd(0))))[:-1]


share_equity=fn.get_data('보통주자본금(천원)')
retained_capital=fn.get_data('자본잉여금(천원)')
retained_earning=fn.get_data('이익잉여금(천원)')
treasury=fn.get_data('자기주식(천원)')
tax=fn.get_data('이연법인세부채(천원)')
price=fn.get_data('수정주가(원)')
listed_stocks=fn.get_data('기말발행주식수 (보통)(주)')

me=listed_stocks*price

be=share_equity+retained_capital.fillna(0)+retained_earning.fillna(0)+treasury+tax.fillna(0)

be=be.loc[year_end]
me=me.loc[year_end]
bm=be/me

# %%
bm

# %%
year_end=sorted(list(set(df2['date']+YearEnd(0))))[:-1]


share_equity=fn.get_data('보통주자본금(천원)')
retained_capital=fn.get_data('자본잉여금(천원)')
retained_earning=fn.get_data('이익잉여금(천원)')
treasury=fn.get_data('자기주식(천원)')
tax=fn.get_data('이연법인세부채(천원)')
price=fn.get_data('종가(원)')
listed_stocks=fn.get_data('기말발행주식수 (보통)(주)')

me=listed_stocks*price

be=share_equity.fillna(0)+retained_capital.fillna(0)+retained_earning.fillna(0)+treasury+tax.fillna(0)

# %%
be

# %% [markdown]
# # OP

# %%
df.columns

# %%
# sales=fn.get_data('매출액(천원)')
# cost=fn.get_data('매출원가(천원)')
# interest_cost=fn.get_data('이자비용(천원)')
# op=(sales-cost-interest_cost)/share_equity
op=fn.get_data('영업이익(천원)')/share_equity
op=op.loc[year_end]





# %% [markdown]
# # INVIT

# %%
invit=fn.get_data('총자산(천원)')
invit=invit.loc[year_end]
invit=invit.pct_change()


# %%
invit

# %% [markdown]
# # MOM

# %%
mom=(price.shift(1)-price.shift(12))/price.shift(12)
mom

# %%
rebalancing_month=sorted(list(set(df2['date']+YearEnd(0)-MonthEnd(6))))


# %% [markdown]
# # size

# %%

# %%
###1.이렇게 하면 원래 파마 프랜치

size=listed_stocks*price
# size=size.loc[rebalancing_month]
# size

###2.#이렇게 하면 기홍햄
# size.loc[rebalancing_month]=np.NaN
# size.ffill(inplace=True)


# %%
size

# %%
df2

# %%
factor_list=[size,bm,op,invit,mom]
name=['size','bm','op','invit','mom']
data=df2.copy()
for i in range(len(factor_list)):
    tmp=factor_list[i].reset_index().melt(id_vars='date', var_name='Symbol', value_name=name[i])
    data=pd.merge(data,tmp,on=['date','Symbol'],how='left')

# %%
##3be.dropna(how='all')

# %%

me=listed_stocks*price
be=share_equity+retained_capital.fillna(0)+retained_earning.fillna(0)+treasury.fillna(0)+tax.fillna(0)

devil_hml=be/me
devil_hml=devil_hml.reset_index().melt(id_vars='date', var_name='Symbol', value_name='devil_hml')
data=pd.merge(data,devil_hml,on=['date','Symbol'],how='left')
data['devil_hml']=data.groupby('Symbol')['devil_hml'].shift(1)

# %%
date=sorted(list(set(data['date'])))[-10:]
data.loc[(data['date'].isin(date))&(data['Symbol']=='A005930')]

# %%
data.loc[(data['date'].isin(date))&(data['Symbol']=='A005930')]

# %%
#### 자본 잠식기업 제외
data['be_test']=data['보통주자본금(천원)'].fillna(0)+data['자본잉여금(천원)'].fillna(0)+data['이익잉여금(천원)'].fillna(0)+data['자기주식(천원)'].fillna(0)+data['이연법인세부채(천원)'].fillna(0)
data2=data.loc[data['be_test']>0].copy()
date=sorted(list(set(data['date'])))[-9:]
data_real=data.loc[data['date'].isin(date)]
data_last=pd.concat([data2,data_real])
data_last

# %%
print(np.min(data_last['be_test']))
data_last.drop(columns='be_test',inplace=True)
data_last

# %%
data_last.to_csv('factor.csv',index=False)

# %%
