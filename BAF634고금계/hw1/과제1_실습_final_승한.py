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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# #### 선견편향 제거를 위해서 사이즈 레깅

# %%
### 매월 6월 말 기준으로 리벨런싱
factor_df=pd.read_csv('factor.csv')
# factor_df['size_lag1']=factor_df.groupby('Symbol')['size'].shift(1)
rebalancing_period=sorted(list(set(pd.to_datetime(factor_df['date'])+YearEnd(0)+MonthEnd(-6))))
factor_df['date']=pd.to_datetime(factor_df['date'])

factor_df_size_ffill=factor_df.copy()
factor_df_size_ffill=factor_df_size_ffill.drop_duplicates(subset=['date','Symbol'])
factor_df_size_ffill=factor_df_size_ffill.loc[~(factor_df_size_ffill['수익률 (1개월)(%)'].isna())]
factor_df_size_ffill=factor_df_size_ffill.loc[~(factor_df_size_ffill['기말발행주식수 (보통)(주)'].isna())]


# %%
factor_df.columns


# %% [markdown]
# # 백테스트 짜기
#
# #### 1. 모멘텀 팩터를 제외한 다른 팩터들은 매년 6월 말 리벨런싱/ 모멘텀은 매달 말 리벨런싱.
# #### 2. 다른 팩터들은 이미 래깅이 되어있지만 사이즈는 안되어 있어서 레깅함.
# #### 3. 포트폴리오 구성할 떄도 6월 말의 시총액 가중평균으로 7월부터 이듬 해 6월까지 리턴을 가중평균 해야함.
# #### 4. Independent Sorting

# %%
class backtest:

    def __init__(self,factor_df,quantile_1,quantile_2,factor_1,factor_2):
        self.factor_df=factor_df
        self.quantile_1=quantile_1
        self.quantile_2=quantile_2
        self.factor_1=factor_1
        self.factor_2=factor_2
        




    def winsorizing(factor_list, q):
        #factor_list=[i+'w' for i in factor_list]
        self.factor_df[factor_list]=self.factor_df.groupby('date')[factor_list].apply(lambda x: x.clip(x.quantile(q, interpolation='lower'), 
                     x.quantile(1-q, interpolation='higher'), axis=0))


    


    def assign_scores(self,x,quantile_list):
        # 각 그룹에 대해 퀀타일을 계산
        result = x.quantile(q=quantile_list)
        score = pd.Series(np.NaN, index=x.index)
        
        for i in range(len(quantile_list)):
            if i == 0:
                score = np.where(x <= result[quantile_list[i]], i + 1, score)
            else:
                score = np.where((x <= result[quantile_list[i]]) & 
                                (x >= result[quantile_list[i-1]]), 
                                i + 1, score)
        
        # 마지막 퀀타일보다 큰 값에 대해 score 할당
        score = np.where(x > result[quantile_list[-1]], len(quantile_list) + 1, score)
        
        return pd.Series(score, index=x.index)


        
    def sorting(self,dependent_sort=True,lagging1=0,lagging2=0):
        self.test=self.factor_df.copy()
        self.test=self.test.loc[self.test['거래정지여부']=='정상']#### 거래되지 않는 종목들 테스트에서 제외
        self.test=self.test.loc[self.test['관리종목여부']=='정상']
        self.test['rtn']=self.test['수익률 (1개월)(%)']/100## 이름 헷갈려서 바꿈
        
        if lagging1!=0:
       
            self.test[self.factor_1]=self.test.groupby('Symbol')[self.factor_1].shift(lagging1)
        if lagging2!=0:
            self.test[self.factor_2]=self.test.groupby('Symbol')[self.factor_2].shift(lagging2)
        
        self.test['size_1']=self.test.groupby('Symbol')['size'].shift(1)
        
        self.test['score']=self.test.groupby('date')[self.factor_1].transform(func=lambda x: self.assign_scores(x,quantile_list=self.quantile_1))
        ###dependent sort
        if dependent_sort:
            self.test['score2']=self.test.groupby('date')[self.factor_2].transform(func=lambda x: self.assign_scores(x,quantile_list=self.quantile_2))
           

        else: ### independent_sort
            self.test['score2']=self.test.groupby(['date','score'])[self.factor_2].transform(func=lambda x: self.assign_scores(x,quantile_list=self.quantile_2))
            




    def run(self,score1,score2,value_weighted=True,not_monthly_rebalancing=True):
        
        self.test['indicator']=np.where((self.test['score']==score1) & (self.test['score2']==score2),1,np.nan)
        #self.result=self.test.loc[self.test['indicator']==1]
        self.test['indicator_1']=self.test.groupby('Symbol')['indicator'].shift(1)
        if not_monthly_rebalancing:
            self.test['indicator_1']=self.test.groupby('Symbol')['indicator_1'].ffill()
        #test_period=sorted((list(set(self.test['date']))))
        #self.rebalancing_period=rebalancing_period

        if value_weighted:
            self.v_weight=self.test.loc[self.test['indicator_1'].notna()]
            self.v_weight['weight']=self.v_weight.groupby(['date','indicator_1'])['size_1'].transform(lambda x: x/x.sum())
            self.port=pd.merge( self.test, self.v_weight[['date','Symbol','weight']],on=['date','Symbol'],how='left')[['date',"Symbol",'weight','rtn','indicator_1']]
        else:
            self.port=self.test.copy()
            self.port['weight']=self.port.groupby(['date'])['indicator_1'].transform(lambda x: x/x.count())

        self.port['port_rtn']=self.port['rtn']*self.port['weight']
        self.result=self.port[['date','Symbol','indicator_1','port_rtn']]
        self.port_rtn=self.result.dropna().groupby('date')['port_rtn'].sum()
        return self.port_rtn
        



        
        

# %%
from tqdm import tqdm
quantile_list=[0.2,0.4,0.6,0.8]
test=backtest(factor_df=factor_df_size_ffill,quantile_1=quantile_list,quantile_2=quantile_list,factor_1='size',factor_2='bm')
test.sorting(lagging2=6)
result=pd.DataFrame()
for i in tqdm(range(1,6)):
    for j in range(1,6):
        tmp=test.run(score1=i,score2=j,value_weighted=True)
        result[f'size_{i}_bm_{6-j}']=tmp

# %%
import plotly.express as px
px.line((1+result).cumprod())

# %%
result=result*100

# %%
result.to_csv('5x5_table.csv')

# %% [markdown]
# ## 팩터 수익률

# %%
factor_df_size_ffill['devil_hml_m']=factor_df_size_ffill['devil_hml'].copy()

# %%
factor_df_size_ffill['devil_hml_m']=factor_df_size_ffill['devil_hml'].copy() ### 매달 리벨런싱하는 것도 만듬
factors=['bm','op','invit','devil_hml','mom','devil_hml_m']
quantile_list1=[0.5]
quantile_list2=[0.3, 0.7]
factor_result=pd.DataFrame()
quantile_list3=[0.3,0.7]
for factor in tqdm(factors):

        
    for i in range(1,3):
        for j in range(1,4):
            factor_test=backtest(factor_df=factor_df_size_ffill,quantile_1=quantile_list1,quantile_2=quantile_list2,factor_1='size',factor_2=factor)
            factor_test.sorting(lagging2=6)
            if factor=='mom':
                factor_test=backtest(factor_df=factor_df_size_ffill,quantile_1=quantile_list1,quantile_2=quantile_list3,factor_1='size',factor_2=factor)
                factor_test.sorting()
                tmp=factor_test.run(score1=i,score2=j, value_weighted=True,not_monthly_rebalancing=False)

            elif factor=='devil_hml_m':
                factor_test=backtest(factor_df=factor_df_size_ffill,quantile_1=quantile_list1,quantile_2=quantile_list2,factor_1='size',factor_2=factor)
                factor_test.sorting()
                tmp=factor_test.run(score1=i,score2=j, value_weighted=True,not_monthly_rebalancing=False)

            else:
                tmp=factor_test.run(score1=i,score2=j, value_weighted=True)
            if j!=2:
                factor_result[f'size_{i}_{factor}_{4-j}']=tmp


# %%
real_factor_result=pd.DataFrame(index=factor_result.index)

####bm
real_factor_result['HML']=(factor_result['size_1_bm_1']+factor_result['size_2_bm_1']-factor_result['size_1_bm_3']-factor_result['size_2_bm_3'])/2

####op
real_factor_result['RMW']=(factor_result['size_1_op_1']+factor_result['size_2_op_1']-factor_result['size_1_op_3']-factor_result['size_2_op_3'])/2

#####invit
real_factor_result['CMA']= -(factor_result['size_1_invit_1']+factor_result['size_2_invit_1']-factor_result['size_1_invit_3']-factor_result['size_2_invit_3'])/2

#####mom
real_factor_result['UMD']=(factor_result['size_1_mom_1']+factor_result['size_2_mom_1']-factor_result['size_1_mom_3']-factor_result['size_2_mom_3'])/2

#####devil_hml
real_factor_result['devil_HML']=(factor_result['size_1_devil_hml_1']+factor_result['size_2_devil_hml_1']-factor_result['size_1_devil_hml_3']-factor_result['size_2_devil_hml_3'])/2

real_factor_result['devil_HML_m']=(factor_result['size_1_devil_hml_m_1']+factor_result['size_2_devil_hml_m_1']-factor_result['size_1_devil_hml_m_3']-factor_result['size_2_devil_hml_m_3'])/2



# %%
factor_result2

# %%
real_factor_result

# %%
(1+real_factor_result).cumprod()

# %%
(1+real_factor_result).cumprod()


# %%
factors=['bm','op','invit']
quantile_list1=[0.5]
quantile_list2=[1/3, 1-(1/3)]
quantile_list3=[0.3,0.7]
factor_result=pd.DataFrame()

for factor in tqdm(factors):

        
    for i in range(1,3):
        for j in range(1,4):
            factor_test=backtest(factor_df=factor_df_size_ffill,quantile_1=quantile_list3,quantile_2=quantile_list1,factor_1=factor,factor_2='size')
            factor_test.sorting(lagging1=6)
            
            if factor=='mom':
                tmp=factor_test.run(score1=j,score2=i, value_weighted=True)

            else:
                tmp=factor_test.run(score1=j,score2=i, value_weighted=True)
                #print(f'{i}_{j}')
       
            factor_result[f'size_{i}_{factor}_{4-j}']=tmp


# %%
factor_result

# %%
real_factor_result['SMB']=np.sum((factor_result.iloc[:,:3].values-factor_result.iloc[:,3:6].values)/3 + (factor_result.iloc[:,6:9].values-factor_result.iloc[:,9:12].values)/3+(factor_result.iloc[:,12:15].values-factor_result.iloc[:,15:18].values)/3,axis=1)/3
#real_factor_result['HML'] = -(factor_result.iloc[:,0] + factor_result.iloc[:,3] - factor_result.iloc[:,2] - factor_result.iloc[:, 5])/2

# %%
factor_result

# %%
#real_factor_result.drop(columns='devil_HML',inplace=True)

# %%

from pathlib import Path
from fndata import FnStockData,FnMarketData
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.offsets import YearEnd
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'
fndata_path = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'
fn = FnMarketData(fndata_path)
df = fn.get_data(format='wide')

# %%
rf_path = DATA_DIR / '통안채1년물_월평균_201301-202408.csv'
rf = pd.read_csv(rf_path)


# %%
df=df.reset_index()
rf['date']=df['date']


# %%

# %%
(rf['원자료']/12)/100

# %%
real_factor_result=real_factor_result.reset_index()
real_factor_result.rename(columns={'index':'date'},inplace=True)
real_factor_result

# %%
df['RF']=(rf['원자료']/12)/100
df['Mkt-RF']=df['MKF2000']-df['RF']
real_factor_result=pd.merge(real_factor_result,df[['date','RF','Mkt-RF']],how='left',on=['date'])

# %%
real_factor_result

# %%
real_factor_result[['HML','RMW',"CMA","UMD",'SMB',"RF","Mkt-RF",'devil_HML','devil_HML_m']]=real_factor_result[['HML','RMW',"CMA","UMD",'SMB',"RF","Mkt-RF",'devil_HML','devil_HML_m']]*100

# %%
submit=real_factor_result[['date','HML','RMW',"CMA","UMD",'SMB',"RF","Mkt-RF"]]
submit.to_csv('factor_port.csv')
print(submit)

# %%
# real_factor_result.to_csv('factor_port.csv')

# %%
draw=real_factor_result.set_index(['date'])
px.line((1+draw.dropna()/100).cumprod())

# %%
fn=pd.read_csv('fn_factor2.csv')
fn=fn.iloc[7:].T
fn=fn.iloc[6:]
fn.columns=['date','MOM_fn',"SMB_fn","HML_fn"]
fn['date']=pd.to_datetime(fn['date'])

# %%

# %%

import plotly.express as px
temp=pd.merge(draw.reset_index(),fn,on=['date'],how='left')
#temp[['HML_fn','SMB_fn']]=temp[['HML_fn','SMB_fn']]
#temp=temp.astype(float)

temp=temp.set_index(['date'])
temp=temp.astype(float)
px.line((1+temp.loc[temp.index>'2016']/100).cumprod())

# %%
temp.corr()

# %%
