
#%%
import pandas as pd 
import numpy as np 
import datetime
from dateutil.relativedelta import relativedelta
import vectorbt as vbt

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

sp500 = sp500.loc["1964-03-01":]
sp500_prices = (1+sp500).cumprod()

orders = pd.read_pickle("./data/orders_top30.pkl")
orders_delay = pd.read_pickle("./data/orders_delay_top30.pkl")

num_tests = 2
_price = sp500_prices.vbt.tile(num_tests, keys=pd.Index(["No_Delay", "1D_Delay"], name='group'))
_orders = pd.concat([orders, orders_delay], axis=1)

#%%
pf = vbt.Portfolio.from_orders(
    close=_price,   #.loc[order.index, order.columns], 
    size=_orders,
    size_type='amount',
    fees=0.0, 
    freq='d',
    init_cash=0,
    cash_sharing=True,
    group_by='group')

fig = pf.value().vbt.plot()
fig.update_layout(title="No Transaction Costs")
fig.show()


values = pf.value()["1965-03-01":"2024-08-31"]
profit = values.diff()
stats = profit.apply([np.mean, np.std, np.min, np.max])
stats.loc["mean"] *= 252
stats.loc["std"] *= np.sqrt(252)
stats.loc["mean/std"] = stats.loc["mean"]/stats.loc["std"]
print(stats)

#%%
pf = vbt.Portfolio.from_orders(
    close=_price,   #.loc[order.index, order.columns], 
    size=_orders,
    size_type='amount',
    fees=0.001, 
    freq='d',
    init_cash=0,
    cash_sharing=True,
    group_by='group')

fig = pf.value().vbt.plot()
fig.update_layout(title="10bps Transaction Costs")
fig.show()

values = pf.value()["1965-03-01":"2024-08-31"]
profit = values.diff()
stats = profit.apply([np.mean, np.std, np.min, np.max])
stats.loc["mean"] *= 252
stats.loc["std"] *= np.sqrt(252)
stats.loc["mean/std"] = stats.loc["mean"]/stats.loc["std"]
print(stats)