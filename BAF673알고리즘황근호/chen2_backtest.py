#%%
import pandas as pd 
from dateutil.relativedelta import relativedelta
import numpy as np 
import vectorbt as vbt

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

sp500.columns = sp500.columns.astype(int)
sp500 = sp500.loc["1965-01-01":"2024-06-30"]
sp500_prices = (1+sp500).cumprod()

orders = pd.DataFrame(index=sp500.index, columns=sp500.columns, dtype=float)
orders_long = pd.DataFrame(index=sp500.index, columns=sp500.columns, dtype=float)
orders_short = pd.DataFrame(index=sp500.index, columns=sp500.columns, dtype=float)
comnam_map = stock_id[["namedt", "permno", "comnam"]].drop_duplicates().groupby(["permno"])["comnam"].last()

short_list = pd.read_csv("./data/short_list.csv", parse_dates=True, index_col=0)
long_list = pd.read_csv("./data/long_list.csv", parse_dates=True, index_col=0)

# 오더를 만든다. 비율을 그냥 1/50, 똑같은 비중으로 2%씩 넣도록 했다. 
for i in range(len(long_list)):
    start = long_list.index[i]
    end = long_list.index[i] + relativedelta(months=1) - relativedelta(days=1)
    start, end = sp500[start:end].index[0], sp500[start:end].index[-1]
    long, short = long_list.iloc[i], short_list.iloc[i]
    orders.loc[start] = 0
    orders_long.loc[start] = 0
    orders_short.loc[start] = 0
    orders.loc[start, long] = 1 / 50
    orders.loc[start, short] = -1 / 50
    orders_long.loc[start, long] = 1/50
    orders_short.loc[start, short] = 1 / 50

# 상위 10%, 하위 10% 전략 수익률을 일단 따로따로 봤다. 
# 즉, 1) 전체, 2) 상위 10%, 3) 하위 10% 의 수익률을 따로따로 봤다.

#%%
start_date = "1965-01-01"
start_date = "1990-01-01" # 똑같은 성과를 90년도부터 본다면? 훨씬 성능 떨어지게 나온다. 
end_date = "2025-12-31"
_prices = sp500_prices.loc[start_date:end_date]
_orders = orders.loc[start_date:end_date]
_orders_long = orders_long.loc[start_date:end_date]
_orders_short = orders_short.loc[start_date:end_date]

num_tests = 3
_prices = _prices.vbt.tile(num_tests, keys=pd.Index(["Long-Short", "Long", "Short"], name='group'))
_orders = pd.concat([_orders, _orders_long, _orders_short], axis=1)


pf = vbt.Portfolio.from_orders(
    close=_prices,
    size=_orders,
    size_type='target_percent',
    fees=0.0, 
    freq='d',
    init_cash=1000,
    cash_sharing=True,
    group_by='group')

# 성과를 그려보면 역시 처음에는 잘되다가 나중에 안된다. 
# 교수님은 S&P500만 가지고 해서 논문에서 말한 뉴스 정보전달이 지연되서 나오는 효과 같은 것이 
# 제대로 발휘되지 못했을 수 있다고 하심. (작은게 있어야 효과 최대)

# 특이한 점: 롱숏 나눠서 찍어보면, 롱 성과가 완전 압도하는 것을 알 수 있다. 

#pf.orders.records_readable
fig = pf.value().vbt.plot()
fig.update_layout(
    yaxis=dict(
        #type="log", # 시작점에 의해 차이 매우 뻥튀기 되어 보이는 것 보완해야. 
                     # log scale로 바꾸면 
        title="Portfolio Value (log scale)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    title="Portfolio Value Over Time (Log Scale)"
)
fig.show()

# 교훈
## 1. 복리로 투자한 결과는 볼 때 유의해야 한다. 시작점에 dependent 한 것은 book이 점점 올라가기 때문. 
### 따라서 매번 같은 금액 투자하는걸로 봐야 path dependent 하지 않게 pnl 비교 가능. 
