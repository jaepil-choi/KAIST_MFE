#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
from ar1_process import ar1_process, random_walk
from tqdm import tqdm

prices1 = ar1_process(phi0=0, phi1=0.3, sigma=1, T=2500) # 대략 10년
prices2 = ar1_process(phi0=0, phi1=0.95, sigma=1, T=2500)
prices3 = random_walk(sigma=1, T=2500)

prices1 = prices1 - prices1.min() + 1 # minimum이 1이 되도록 shift
prices2 = prices2 - prices2.min() + 1
prices3 = prices3 - prices3.min() + 1

formation_period = 500 # 전략 형성 기간
m1, s1 = prices1[:formation_period].mean(), prices1[:formation_period].std(ddof=1) # 평균에 수렴한다 보는 것. 
m2, s2 = prices2[:formation_period].mean(), prices2[:formation_period].std(ddof=1) # 2std를 보통의 기준으로 long/short thres 로 쓴다. 
m3, s3 = prices3[:formation_period].mean(), prices3[:formation_period].std(ddof=1) # 그 때 쓰려고 만들어놓는 것임. 



def simulate_strategy(prices, m, s, fees, delay=True): 
    # delay=True: 다음날 매매
    # delay=False: 당일 매매

    low_threshold = m - 2 * s # 2 std
    high_threshold = m + 2 * s
    position = pd.Series(index=prices.index, dtype=float)
    position[prices >= high_threshold] = -1.0
    position[prices <= low_threshold] = 1.0
    position[((prices>=m) & (prices.shift(1)<m))] = 0.0
    position[((prices<=m) & (prices.shift(1)>m))] = 0.0
    position = position.ffill().fillna(0.0)

    #당일 매매 주문 데이터 생성
    order = position.diff()
    order.iloc[0] = position.iloc[0]

    pf = vbt.Portfolio.from_orders(
        prices,
        order if not delay else order.shift(1), 
        size_type="amount", 
        fees=fees,
        freq="D",
        init_cash=100)
    
    return pf

prices1 = pd.Series(prices1[formation_period:])
prices2 = pd.Series(prices2[formation_period:])
prices3 = pd.Series(prices3[formation_period:])

pf1 = simulate_strategy(prices1, m1, s1, 0.001)
pf1.plot().show()

pf2 = simulate_strategy(prices2, m2, s2, 0.001)
pf2.plot().show()

pf3 = simulate_strategy(prices3, m3, s3, 0.001)
pf3.plot().show()

pd.DataFrame({"value1": pf1.value(), "value2": pf2.value(), "value3": pf3.value()}).plot()

#%%
pf1.orders.records_readable
pf2.orders.records_readable
#pf3.orders.records_readable

## vectorbt 좋은데? trade id, order id 등을 다 tracking 할 수 있게 해준다.

#%%
j = 0
phi1s = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
fees = [0.000, 0.001]
res = pd.DataFrame(columns = ["phi1", "fee", "Value", "Sharpe Ratio"])

for _iter in tqdm(range(200)):
    for k, phi1 in enumerate(phi1s):
        prices = ar1_process(phi0=0, phi1=phi1, sigma=1, T=2500)
        prices = prices - prices.min() + 1
        m, s = prices1[:formation_period].mean(), prices1[:formation_period].std(ddof=1)
        prices = pd.Series(prices[formation_period:])

        for i, f in enumerate(fees):
            j += 1
            pf = simulate_strategy(prices, m, s, f, delay=True)
            res.loc[j] = [phi1, f, pf.final_value(), pf.sharpe_ratio()]
## fee 있을 때 없을 때, 그리고 phi1을 여러 번 바꿔가며 분포를 확인. 
## 적정한 수준의 평균회귀 계수가 있다. 너무 강하게 주면 변동성이 너무 작을 수 있다. 
## 차라리 조금 느릴 때 더 샤프도, 수익도 좋을 수 있다는 것을 보여줌. 

res.groupby(["phi1", "fee"]).mean().swaplevel(0,1).sort_index()

#%%
import seaborn as sns
plot_data = res[(res["fee"] == 0.001) & (res["phi1"].isin([0.2,0.5,0.7,0.9,0.95]))]
sns.displot(plot_data, x="Value", hue="phi1", kind="kde")
sns.displot(plot_data, x="Sharpe Ratio", hue="phi1", kind="kde")
# %%
