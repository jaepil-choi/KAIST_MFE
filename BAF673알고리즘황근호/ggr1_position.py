## 포지션 생성하는 로직을 가진 코드드

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

sp500 = sp500.loc["1964-03-01":]
orders = pd.DataFrame(index=sp500.index, columns=sp500.columns, data=0, dtype=float)
orders_delay = pd.DataFrame(index=sp500.index, columns=sp500.columns, data=0, dtype=float)
sp500_prices = (1+sp500).cumprod()
comnam_map = stock_id[["namedt", "permno", "comnam"]].drop_duplicates().groupby(["permno"])["comnam"].last()


# %%
n = 30  #pair 개수

ds = [datetime.datetime(1965, 3, 1)]
i = 0
while True:
    i += 1
    d = ds[0] + relativedelta(months=i*6)
    if d>datetime.datetime(2024, 6, 1):
        break
    ds.append(d)

for d in ds:
    t0 = time.time()
    print(d.strftime("%Y-%m-%d"))

    #d 시점에 s&p500 종목 목록
    d_list = sp500_list[(sp500_list['start']<=d) & (sp500_list['ending']>=d)]['permno']
    permno_list = d_list.astype(str).to_list()

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

    #ssd 상위 n개 조합
    ssd_n = ssd.dropna().sort_values(by="ssd").iloc[:n]


    df_orders = pd.DataFrame()
    for i in range(len(ssd_n)):
        #각 pair 별 주문 데이터 생성
        no1, no2 = ssd_n['no1'].iloc[i], ssd_n['no2'].iloc[i]
        stdev = ssd_n['std'].iloc[i]
        test_prices12 = test_prices[[no1, no2]]
        #중간에 null 값이 있는 경우 최근 시점까지 데이터 추려냄
        if test_prices12.isna().any().any():
            idx = test_prices12.isna().any(axis=1).idxmax()
            test_prices12 = test_prices12.loc[:idx].iloc[:-1]
        if len(test_prices12) == 0: #데이터가 없는 경우 패스
            continue
        norm_price1, norm_price2 = norm_test_prices[no1].loc[test_prices12.index], norm_test_prices[no2].loc[test_prices12.index]
        diff = norm_price1 - norm_price2


        #### 포지션 만드는 방법 (저번시간에도 했음음)
        position = pd.DataFrame(index=diff.index, columns=[no1, no2], dtype=float)
        std2 = 2*stdev
        position[diff >= std2] = [-1.0, 1.0]
        position[diff <= -std2] = [1.0, -1.0]
        position[((diff>=0) & (diff.shift(1)<0))] = [0.0, 0.0]
        position[((diff<=0) & (diff.shift(1)>0))] = [0.0, 0.0]

        position = position.ffill().fillna(0.0)
        position.iloc[-1] = [0.0, 0.0]  #최종 시점에서는 포지션 0으로 종료

        #당일 매매 주문 데이터 생성
        entry = (position[no1]!=0.0) & ((position[no1].shift(1)==0.0) | (position[no1].shift(1).isnull()))
        exit = (position[no1]==0.0) & (position[no1].shift(1)!=0.0) & (position[no1].shift(1).notnull())
        order = position.diff()
        order.iloc[0] = position.iloc[0]

        ## vectorbt에 전략을 넣는 3가지 방법:
        # 1. 금액의 %를 rebalancing하는 형태로
        # 2. 사고파는 주식의 금액을 기준으로
        # 3. 주식의 수량을 기준으로

        ## 여기선 3번째 방법을 사용

        order[entry] = order[entry] / test_prices12[entry] # 가격으로 나눈만큼 주문을 넣는다.
        order[exit] = (-1) * order[entry].values # exit 할 땐 0으로 돌아갈 수 있도록.
        orders.loc[order.index, position.columns] += order

        #하루 딜레이 주문 데이터 생성
        position = position.shift(1).fillna(0)
        position.iloc[-1] = [0.0, 0.0]  #최종 시점에서는 포지션 0으로 종료
        entry = (position[no1]!=0.0) & ((position[no1].shift(1)==0.0) | (position[no1].shift(1).isnull()))
        exit = (position[no1]==0.0) & (position[no1].shift(1)!=0.0) & (position[no1].shift(1).notnull())
        order = position.diff()
        order.iloc[0] = 0.0
        order[entry] = order[entry] / test_prices12[entry]
        order[exit] = (-1) * order[entry].values
        orders_delay.loc[order.index, position.columns] += order

        df_orders = pd.concat([df_orders, order], axis=1)
    
    t1 = time.time()
    print(f"time: {t1-t0:.3f}s")



#%%
## 참고. 대부분이 0/null인 sparse 한 dataset의 경우 SparseDtype을 사용하면 메모리를 효율적으로 사용할 수 있다.
orders = orders.astype(pd.SparseDtype("float", fill_value=0))
orders_delay = orders_delay.astype(pd.SparseDtype("float", fill_value=0))
orders.to_pickle("./data/orders_top30.pkl")
orders_delay.to_pickle("./data/orders_delay_top30.pkl")

#%%

# 내 질문. pair trading은 기본적으로 long short 아닌가? 
# market neutral 될거 같은데 왜 pnl은 market exposure가 큰 것 처럼 보이는가? 
# 한 번 S&P500과의 correlation을 봐야할 것 같다.

# %%
