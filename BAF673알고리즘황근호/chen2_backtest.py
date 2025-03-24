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

#%%
start_date = "1965-01-01"
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

#pf.orders.records_readable
fig = pf.value().vbt.plot()
fig.update_layout(
    yaxis=dict(
        #type="log",
        title="Portfolio Value (log scale)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    title="Portfolio Value Over Time (Log Scale)"
)
fig.show()


