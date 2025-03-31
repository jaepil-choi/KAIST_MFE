#%%
import pandas as pd 
import numpy as np 
import datetime
from scipy.stats import linregress
from dateutil.relativedelta import relativedelta
import vectorbt as vbt

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

sp500 = sp500.loc["1960-01-01"]
comnam_map = stock_id[["namedt", "permno", "comnam"]].drop_duplicates().groupby(["permno"])["comnam"].last()

ds = [datetime.datetime(1965, 1, 1)]
i = 0
while True:
    i += 1
    d = ds[0] + relativedelta(months=i)
    if d>datetime.datetime(2024, 6, 1):
        break
    ds.append(d)

short_list = pd.DataFrame(columns=np.arange(1,51))
long_list = pd.DataFrame(columns=np.arange(1,51))
for d in ds:
    print(d.strftime("%Y-%m-%d"))
    #d 시점에 s&p500 종목 목록
    d_list = sp500_list[(sp500_list['start']<=d) & (sp500_list['ending']>=d)]['permno']
    permno_list = d_list.astype(str).to_list()
    data = sp500[permno_list].loc[d-relativedelta(years=5):d-relativedelta(days=1)]
    ret = data.resample("ME").agg(lambda x: (1+x).prod()-1)
    # 일간 데이터를 월간으로 바꿔준다. 매일의 수익률을 월간 수익률 화. 
    corr = ret.corr()
    diff_series = pd.Series(index=corr.columns, dtype=float)
    for i in range(len(corr.columns)):
        high_corr_list = corr.iloc[i].sort_values().iloc[-51:-1].index
        Cret = ret[high_corr_list].mean(1)
        Lret = ret.iloc[:,i]
        var = pd.concat([Cret, Lret], axis=1).dropna()        
        beta = var.cov().iloc[0,1] / var.cov().iloc[1,1]
        diff = beta*Cret.iloc[-1] - Lret.iloc[-1]
        diff_series.iloc[i] = diff
    
    sorted_diff_series = diff_series.sort_values().dropna()
    short_50 = sorted_diff_series.iloc[:50]
    long_50 = sorted_diff_series.iloc[-50:]
    
    short_list.loc[d] = short_50.index.to_list()
    long_list.loc[d] = long_50.index.to_list()


short_list.to_csv("./data/short_list.csv")
long_list.to_csv("./data/long_list.csv")
