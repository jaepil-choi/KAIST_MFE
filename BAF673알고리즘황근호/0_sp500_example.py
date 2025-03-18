#%%
import pandas as pd 
import numpy as np 
import datetime
from dateutil.relativedelta import relativedelta
import vectorbt as vbt

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

sp500.columns = sp500.columns.astype(int)
sp500 = sp500.loc["2015":]
sp500.iloc[0] = 0.0
sp500_prices = (1+sp500).cumprod()
sp500_prices = sp500_prices.dropna(how="all", axis=1)

comnam_map = stock_id[["namedt", "permno", "comnam"]].drop_duplicates().groupby(["permno"])["comnam"].last()

#%%
# Survival Bias 예시
# 2015년 1월 1일부터 2024년 12월 31일까지의 S&P 500 종목 목록
d0 = datetime.datetime(2015, 1, 1)
d1 = datetime.datetime(2024, 12, 31)
d0_list = sp500_list[(sp500_list['start']<=d0) & (sp500_list['ending']>=d0)]['permno']
d1_list = sp500_list[(sp500_list['start']<=d1) & (sp500_list['ending']>=d1)]['permno']
## --> survivorship bias를 조심해야 함을 알 수 있음. 


sp500_prices = sp500_prices.loc[:,pd.concat([d0_list, d1_list]).drop_duplicates()]

print(f"{d0.strftime('%Y-%m-%d')}: # of S&P 500 stocks = {d0_list.size}")
print(f"{d1.strftime('%Y-%m-%d')}: # of S&P 500 stocks = {d1_list.size}")

excluded_ids = d0_list[~d0_list.isin(d1_list)]
excluded_stocks = comnam_map.loc[excluded_ids]
included_ids = d1_list[~d1_list.isin(d0_list)]
included_stocks = comnam_map.loc[included_ids]
print("="*60)
print(f"excluded_ids: {excluded_ids.size}")
print(excluded_stocks)
print("="*60)
print(f"included_ids: {included_ids.size}")
print(included_stocks)


#%%
init_cash = 10000.0

#2015-01-01 기준 포트폴리오
orders = pd.DataFrame(index=sp500_prices.index, columns=sp500_prices.columns, data=0, dtype=float)
orders.loc["2015-01-02", d0_list] = init_cash/d0_list.size
#orders.loc[:, d1_list.astype(str)] = 1.0/d1_list.size

pf0 = vbt.Portfolio.from_orders(
    close=sp500_prices,   #.loc[order.index, order.columns], 
    size=orders,
    init_cash=init_cash,
    cash_sharing=True,
    size_type='value',
    call_seq='auto',
    freq='D')

stat0 = pf0.stats()
order_record0 = pf0.orders.records_readable


#2024-12-31 기준 포트폴리오
orders = pd.DataFrame(index=sp500_prices.index, columns=sp500_prices.columns, data=0, dtype=float)
orders.loc["2015-01-02", d1_list] = init_cash/d1_list.size

pf1 = vbt.Portfolio.from_orders(
    close=sp500_prices,   #.loc[order.index, order.columns], 
    size=orders,
    init_cash=init_cash,
    cash_sharing=True,
    size_type='value',
    call_seq='auto',
    freq='D')

stat1 = pf1.stats()
order_record1 = pf1.orders.records_readable

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize=(12,8))
values = pd.concat([pf0.value(), pf1.value()], axis=1)
values.columns = ['Portfolio 0', 'Portfolio 1']
values.plot(ax=ax)

## 선견편향의 효과를 보여준다. survivorship bias 로 인한 차이. 



# %%
