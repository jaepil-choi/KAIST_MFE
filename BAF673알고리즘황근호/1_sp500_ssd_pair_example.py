#%%
import pandas as pd 
import numpy as np 
import datetime
from dateutil.relativedelta import relativedelta
import vectorbt as vbt
import time

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

sp500 = sp500.loc["1969-01-01":]
sp500.columns = sp500.columns.astype(int)
orders = pd.DataFrame(index=sp500.index, columns=sp500.columns, data=0, dtype=float)
sp500_prices = (1+sp500).cumprod()
comnam_map = stock_id[["namedt", "permno", "comnam"]].drop_duplicates().groupby(["permno"])["comnam"].last()


# %%
n = 20  #pair 개수

d = datetime.datetime(2024, 3, 1)
print(d)

#d 시점에 s&p500 종목 목록
d_list = sp500_list[(sp500_list['start']<=d) & (sp500_list['ending']>=d)]['permno']
permno_list = d_list.to_list()

#d 시점에서 1년 전부터 6개월 후까지의 데이터
data = sp500_prices[permno_list].loc[d-relativedelta(years=1):d+relativedelta(months=6)-relativedelta(days=1)]

#d 시점에서 1년 전부터 1일 전까지의 데이터
formation_prices = data.loc[d-relativedelta(years=1):d-relativedelta(days=1)]

#d 시점부터 6개월 간 데이터
test_prices = data.loc[d:d+relativedelta(months=6)-relativedelta(days=1)]

#nomalized prices
norm_formation_prices = (formation_prices - formation_prices.min()) / (formation_prices.max() - formation_prices.min())
norm_test_prices = (test_prices - formation_prices.min()) / (formation_prices.max() - formation_prices.min())
n_cols = len(norm_formation_prices.columns)

#ssd 계산
ssd = pd.DataFrame(index=np.arange(n_cols*(n_cols-1)/2, dtype=int), columns = ["no1", "no2", "ssd", "std"])
c = 0
for i, no_i in enumerate(norm_formation_prices.columns):
    for j, no_j in enumerate(norm_formation_prices.columns):
        if i < j:            
            diff = norm_formation_prices[no_i] - norm_formation_prices[no_j]
            ssd.loc[c] = [no_i, no_j, (diff ** 2).sum(), diff.std(ddof=1)]
            c += 1



#%%
#ssd 상위 n개 조합
ssd_n = ssd.sort_values(by="ssd").iloc[:n]

df_orders = pd.DataFrame()
i = 4
#각 pair 별 주문 데이터 생성
no1, no2 = ssd_n['no1'].iloc[i], ssd_n['no2'].iloc[i]
stdev = ssd_n['std'].iloc[i]
test_prices12 = test_prices[[no1, no2]]
if test_prices12.isna().any().any():
    idx = test_prices12.isna().any(axis=1).idxmax()
    test_prices12 = test_prices12.loc[:idx].iloc[:-1]

norm_price1, norm_price2 = norm_test_prices[no1].loc[test_prices12.index], norm_test_prices[no2].loc[test_prices12.index]
diff = norm_price1 - norm_price2

position = pd.DataFrame(index=diff.index, columns=[no1, no2], dtype=float)
std2 = 2*stdev
position[diff >= std2] = [-1.0, 1.0]
position[diff <= -std2] = [1.0, -1.0]
position[((diff>=0) & (diff.shift(1)<0))] = [0.0, 0.0]
position[((diff<=0) & (diff.shift(1)>0))] = [0.0, 0.0]

position = position.ffill().fillna(0.0)
position.iloc[-1] = [0.0, 0.0]  #최종 시점에서는 포지션 0으로 종료

entry = (position[no1]!=0.0) & ((position[no1].shift(1)==0.0) | (position[no1].shift(1).isnull()))
exit = (position[no1]==0.0) & (position[no1].shift(1)!=0.0) & (position[no1].shift(1).notnull())
order = position.diff()
order.iloc[0] = position.iloc[0]
order[entry] = order[entry] / test_prices12[entry]
order[exit] = (-1) * order[entry].values

orders.loc[order.index, position.columns] += order

df_orders = pd.concat([df_orders, order], axis=1)


print("Stock 1: ", comnam_map.loc[no1])
print("Stock 2: ", comnam_map.loc[no2])
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize=(12,8))
# ax.plot(norm_formation_prices[no1], label=comnam_map.loc[no1])
# ax.plot(norm_formation_prices[no2], label=comnam_map.loc[no2])
ax.plot(norm_price1, label=comnam_map.loc[no1])
ax.plot(norm_price2, label=comnam_map.loc[no2])
df = position.iloc[:,0]

ax.fill_between(df.index, 0, 1,
                where=(df == 1), color="lightgreen", alpha=0.3, label="Buy Zone")
ax.fill_between(df.index, 0, 1, 
                where=(df == -1), color="lightcoral", alpha=0.3, label="Sell Zone")

ax.legend()





# %%
