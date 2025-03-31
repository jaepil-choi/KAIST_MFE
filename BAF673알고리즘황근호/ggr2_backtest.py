## ggr1에서 만든 포지션을 이용하여 백테스트를 진행합니다.

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

## 주의: 1달러씩 넣고 빼고 -1하고 그런거라 복리의 효과가 없다. 돈 많이 벌어도 계속 1달러 배팅만 함. 

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
# %%

# 내 질문. pair trading은 기본적으로 long short 아닌가? 
# market neutral 될거 같은데 왜 pnl은 market exposure가 큰 것 처럼 보이는가? 
# 한 번 S&P500과의 correlation을 봐야할 것 같다.

import yfinance as yf

sp500_index = yf.Ticker("^GSPC").history(start="1964-03-01", end="2024-08-31", actions=False)
# Convert timezone-aware index to timezone-naive to match your strategy data
sp500_index.index = sp500_index.index.tz_localize(None)
sp500_index_return = sp500_index["Close"].pct_change().dropna()

# Compare strategy returns with S&P 500 returns
# Align the dates between strategy returns and S&P 500 index returns
common_dates = profit.index.intersection(sp500_index_return.index)
strategy_returns = profit.loc[common_dates]
sp500_returns = sp500_index_return.loc[common_dates]

# Calculate correlation for each strategy
correlations = {}
for col in strategy_returns.columns:
    correlations[col] = strategy_returns[col].corr(sp500_returns)
print("\nCorrelation with S&P 500:")
print(pd.Series(correlations))

# Calculate beta (market exposure)
betas = {}
market_variance = sp500_returns.var()
for col in strategy_returns.columns:
    covariance = strategy_returns[col].cov(sp500_returns)
    betas[col] = covariance / market_variance
print("\nBeta with S&P 500:")
print(pd.Series(betas))

# Visualize the cumulative returns
cumulative_strategy = (1 + strategy_returns).cumprod()
cumulative_sp500 = (1 + sp500_returns).cumprod()

# Normalize to start at 1
cumulative_strategy = cumulative_strategy / cumulative_strategy.iloc[0]
cumulative_sp500 = cumulative_sp500 / cumulative_sp500.iloc[0]

# Plot cumulative returns comparison
import plotly.graph_objects as go
fig = go.Figure()

for col in cumulative_strategy.columns:
    fig.add_trace(go.Scatter(
        x=cumulative_strategy.index,
        y=cumulative_strategy[col],
        mode='lines',
        name=col
    ))

fig.add_trace(go.Scatter(
    x=cumulative_sp500.index,
    y=cumulative_sp500,
    mode='lines',
    name='S&P 500',
    line=dict(color='black', dash='dash')
))

fig.update_layout(
    title="Cumulative Returns: Strategy vs S&P 500",
    xaxis_title="Date",
    yaxis_title="Cumulative Return (normalized)",
    legend_title="Strategy"
)
fig.show()

# Calculate rolling correlation to see how market neutrality changes over time
window = 252  # 1 year of trading days
rolling_correlations = pd.DataFrame(index=strategy_returns.index)

for col in strategy_returns.columns:
    rolling_correlations[col] = strategy_returns[col].rolling(window).corr(sp500_returns)

# Plot rolling correlations
fig = go.Figure()
for col in rolling_correlations.columns:
    fig.add_trace(go.Scatter(
        x=rolling_correlations.index,
        y=rolling_correlations[col],
        mode='lines',
        name=col
    ))

fig.update_layout(
    title=f"Rolling {window}-day Correlation with S&P 500",
    xaxis_title="Date",
    yaxis_title="Correlation",
    legend_title="Strategy",
    yaxis=dict(range=[-1, 1])  # Set y-axis range to show full correlation range
)
fig.show()

# Calculate monthly returns to check for seasonality or market regime dependency
strategy_monthly = strategy_returns.resample('M').sum()
sp500_monthly = sp500_returns.resample('M').sum()

# Create a scatter plot of strategy returns vs market returns
fig = go.Figure()
for col in strategy_monthly.columns:
    fig.add_trace(go.Scatter(
        x=sp500_monthly,
        y=strategy_monthly[col],
        mode='markers',
        name=col
    ))
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(sp500_monthly, strategy_monthly[col])
    x_range = np.linspace(sp500_monthly.min(), sp500_monthly.max(), 100)
    y_range = slope * x_range + intercept
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name=f'{col} fit (r={r_value:.2f})',
        line=dict(dash='dash')
    ))

fig.update_layout(
    title="Strategy Returns vs S&P 500 Returns (Monthly)",
    xaxis_title="S&P 500 Monthly Return",
    yaxis_title="Strategy Monthly Return"
)
fig.show()
# %%
