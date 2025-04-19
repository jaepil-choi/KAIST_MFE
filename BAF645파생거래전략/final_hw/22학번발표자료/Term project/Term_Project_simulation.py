# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 400} id="-8UUbMlZ_Wto" executionInfo={"status": "error", "timestamp": 1701586212083, "user_tz": -540, "elapsed": 923, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="8d117e8d-1bfb-4cae-ce1c-dd4bc656559c"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import time
from scipy.ndimage.interpolation import shift
import scipy.stats
import FinanceDataReader as fdr
import yfinance as yf
import seaborn as sns


# %% id="l7qMoK3-_Wtu" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 22, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
def BSprice(PutCall, x, T, K, r, q, sigma):
    d1 =(1/(sigma*np.sqrt(T)))*(np.log(x/K) + (r - q + 0.5*sigma**2)*T)
    d2 = (1/(sigma*np.sqrt(T)))*(np.log(x/K) + (r - q - 0.5*sigma**2)*T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nd1m = norm.cdf(-d1)
    Nd2m = norm.cdf(-d2)

    if PutCall == 'C':
        price1 = x*np.exp(-q*T)*Nd1 - K*np.exp(-r*T)*Nd2
        price2 = np.exp(-q*T)*Nd1
    elif PutCall =='P':
        price1 = K*np.exp(-r*T)*Nd2m - x*np.exp(-q*T)*Nd1m
        price2 = -np.exp(-q*T)*Nd1m

    return(price1,price2)           # returns array



# %% [markdown] id="Q6gaH8Vo_Wtv"
# # Hedging Cost 비교

# %% id="5tgpEEOb_Wty" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 22, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
def hedging_Call(s0,M,mu,K,r,sigma,T,q,PutCall,simulation_number):

    M_s=20/M
    M_s=np.int(M_s)
    dt=M/52
    tt=np.repeat(T,M_s)
    for i in range(M_s-1):
        tt[i+1]=tt[i]-dt
    tt=np.append(tt,0)



    # Generating random
    generating_number=int(round(T*(1/dt)))+1

    # generating stock path
    z=np.random.randn(simulation_number,generating_number)
    stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
    stock_path_bm[:,0]=1
    stock_price=stock_path_bm.cumprod(1)*s0

    # Calculating Delta
    Delta=BSprice(PutCall,stock_price, tt, K, r, q, sigma)[1]
    Delta_diff=Delta[:,1:]-Delta[:,:-1]
    Delta_diff.shape

    initial_array=np.repeat(Delta[0][0],Delta_diff.shape[0]).reshape(Delta_diff.shape[0],1)
    Delta_change=np.concatenate([initial_array,Delta_diff],axis=1)

    # Delta 변화에따른 stock trade
    shares_purchased=Delta_change*100000
    shares_purchased=shares_purchased.round()  #정수만 purchase



    cost_purchased=shares_purchased*stock_price
    #trans_cost=np.where(cost_purchased<0,cost_purchased*0.0002,0) #transaction Cost
    trans_cost=cost_purchased*0.00015
    trans_cost=abs(trans_cost)
    transaction=np.cumsum(trans_cost,axis=1)
    cum_cost=np.cumsum(cost_purchased,axis=1)

    # Interest Cost 계산
    for i in range(1,cum_cost.shape[1]):

        cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/52)+cost_purchased[:,i]


    # Final cost 계산
    Final_cost=cum_cost[:,-1]
    hedge_cost=np.where(stock_price[:,-1]>50,Final_cost-5000000,Final_cost)

    Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
    hedge_cost_tr=np.where(stock_price[:,-1]>50,Final_cost_tr-5000000,Final_cost_tr)

    Mean=np.mean(hedge_cost)
    Performance_MSR=np.std(hedge_cost)/240000


    Mean_tr=np.mean(hedge_cost_tr)
    Performance_MSR_tr=np.std(hedge_cost_tr)/240000

    return(Mean,Performance_MSR,Mean_tr,Performance_MSR_tr)



# %% id="6vUgMd2k_Wtz" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 22, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
M=[5,4,2,1,0.5,0.25]
simulation_number=1000
s0=49
K=50
mu=0.13
r=0.05
sigma=0.2
q=0
PutCall='C'
T=20/52

# %% id="9VKr3k16_Wtz" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 22, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
for i in M:

    Mean,MSR,Mean_tr,MSR_tr=hedging_Call(s0,i,mu,K,r,sigma,T,q,PutCall,simulation_number)

    if i==5:
        m5=Mean
        m5_tr=Mean_tr
        std5=MSR
        std5_tr=MSR_tr


    if i==4:
        m4=Mean
        m4_tr=Mean_tr
        std4=MSR
        std4_tr=MSR_tr

    if i==2:
        m2=Mean
        m2_tr=Mean_tr
        std2=MSR
        std2_tr=MSR_tr


    if i==1:
        m1=Mean
        m1_tr=Mean_tr
        std1=MSR
        std1_tr=MSR_tr

    if i==0.5:
        m05=Mean
        m05_tr=Mean_tr
        std05=MSR
        std05_tr=MSR_tr

    if i==0.25:

        m025=Mean
        m025_tr=Mean_tr
        std025=MSR
        std025_tr=MSR_tr



# %% [markdown] id="DW5Mj_Vp_Wt0"
# ### without transaction cost

# %% id="BncL1jYF_Wt1" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 22, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df=pd.DataFrame([[m5/100000,m4/100000,m2/100000,m1/100000,m05/100000,m025/100000],[std5,std4,std2,std1,std05,std025]],columns=[5,4,2,1,0.5,0.25])
df.index=['Mean',"Performance MSR"]
df.columns.names=['Rebalance']
df

# %% [markdown] id="iusNfh_e_Wt2"
# ### transaction cost included

# %% id="IxtYFlo1_Wt3" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df=pd.DataFrame([[m5_tr/100000,m4_tr/100000,m2_tr/100000,m1_tr/100000,m05_tr/100000,m025_tr/100000],[std5_tr,std4_tr,std2_tr,std1_tr,std05_tr,std025_tr]],columns=[5,4,2,1,0.5,0.25])
df.index=['Mean',"Performance MSR"]
df.columns.names=['Rebalance']
df

# %% id="pI_SM4fX_Wt4" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

M=1
M_s=20/M
M_s=np.int(M_s)
dt=M/52
tt=np.repeat(T,M_s)
for i in range(M_s-1):
    tt[i+1]=tt[i]-dt
tt=np.append(tt,0)



# Generating random
generating_number=int(round(T*(1/dt)))+1

# generating stock path
z=np.random.randn(simulation_number,generating_number)
stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
stock_path_bm[:,0]=1
stock_price=stock_path_bm.cumprod(1)*s0

# %% id="lwkFlsrK_Wt4" executionInfo={"status": "aborted", "timestamp": 1701586212084, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
for i in range(1000):
    plt.plot(stock_price[i])

# %% id="7K3jB1Tl_Wt4" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="a1k-9uD8_Wt5" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="YqFVk8BZ_Wt5" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="9ccZjgOj_Wt5" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
-----------절취선-----------------    #여기까지 기존 과제

# %% [markdown] id="_RiTdlnU_Wt5"
# ### 현재가 50,000원 strike = 55,000원  수량# 100,000 계약
# ### OPT 매도 포지션일때 내재 변동성 45%로 마진을 쌓아 놓고 Hedging PNL을 계산한다
# ### Int cost =4% daily calculation
# ### Transaction Cost= 0.1% at selling

# %% id="CKf8MZsI_Wt6" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
PutCall='C'    # put / call 중에 어떤 상품인가
x=50    # 기초자산 현재가격
T=20/52    # 만기
K=55    # strike price
r=0.05   # 자금비용 & risk-free rate
q=0    # 배당 (무시)
sigma=0.46    # impvol



c_price=BSprice(PutCall, x, T, K, r, q, sigma)[0]
c_price*100000   #Margin

# %% id="C9Q-QuOh_Wt6" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
simulation_number=1000
s0=50
K=55
mu=0.13
r=0.04
sigma=0.4
q=0
PutCall='C'
T=20/52


# %% id="hkiSXFXL_Wt6" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
def hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number):
    M=1
    M_s=20/M
    M_s=np.int(M_s)
    dt=M/52
    tt=np.repeat(T,M_s)
    for i in range(M_s-1):
        tt[i+1]=tt[i]-dt
    tt=np.append(tt,0)



    # Generating random
    generating_number=int(round(T*(1/dt)))+1

    # generating stock path
    z=np.random.randn(simulation_number,generating_number)
    stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
    stock_path_bm[:,0]=1
    stock_price=stock_path_bm.cumprod(1)*s0

    # Calculating Delta
    Delta=BSprice(PutCall,stock_price, tt, K, r, q, sigma)[1]
    Delta_diff=Delta[:,1:]-Delta[:,:-1]
    Delta_diff.shape

    initial_array=np.repeat(Delta[0][0],Delta_diff.shape[0]).reshape(Delta_diff.shape[0],1)
    Delta_change=np.concatenate([initial_array,Delta_diff],axis=1)

    # Delta 변화에따른 stock trade
    shares_purchased=Delta_change*100000
    shares_purchased=shares_purchased.round()  #정수만 purchase



    cost_purchased=shares_purchased*stock_price
    trans_cost=np.where(cost_purchased<0,cost_purchased*0.001,0) #transaction Cost # 매도시에만 10BP
    #trans_cost=cost_purchased*0.00015
    trans_cost=abs(trans_cost)
    transaction=np.cumsum(trans_cost,axis=1)
    cum_cost=np.cumsum(cost_purchased,axis=1)

    # Interest Cost 계산
    for i in range(1,cum_cost.shape[1]):

        cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/52)+cost_purchased[:,i]


    # Final cost 계산
    Final_cost=cum_cost[:,-1]
    hedge_cost=np.where(stock_price[:,-1]>K,Final_cost-K*100000,Final_cost)

    Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
    hedge_cost_tr=np.where(stock_price[:,-1]>K,Final_cost_tr-K*100000,Final_cost_tr)

    ATM_C=np.where((stock_price[:,-1]<1.01*K)&(stock_price[:,-1]>0.99*K),hedge_cost,0)    #ATM 으로 끝났을때
    ITM_C=np.where(stock_price[:,-1]>K,hedge_cost,0)    #ITM 으로 끝났을때
    OTM_C=np.where(stock_price[:,-1]<K,hedge_cost,0)    #OTM 으로 끝났을때

    ATM_Case=ATM_C[ATM_C!=0]
    ITM_Case=ITM_C[ITM_C!=0]
    OTM_Case=OTM_C[OTM_C!=0]


    hedging_PNL=(c_price*100000-hedge_cost)

    Mean=np.mean(hedge_cost)    # transcation cost 포함하지 않은 hedging cost
    Performance_MSR=np.std(hedge_cost)/240000


    Mean_tr=np.mean(hedge_cost_tr)   #transaction cost 포함함 hedging cost
    Performance_MSR_tr=np.std(hedge_cost_tr)/240000
    Mean_hedging_PNL=np.mean(hedging_PNL)


    return(Mean,Performance_MSR,Mean_tr,Performance_MSR_tr,np.mean(ITM_Case),np.mean(OTM_Case),np.mean(ATM_Case),Mean_hedging_PNL)




# %% id="4f3QjHF8_Wt6" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
s0=49
K=50
mu=0.13
r=0.05
sigma=0.2
q=0
PutCall='C'
T=20/52
simulation_number=1000


# %% [markdown] id="aKBKulXW_Wt7"
# ## transaction cost 비교

# %% id="dXPKK86J_Wt7" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Mean_without_transcost=np.zeros(1000)
Mean_with_transcost=np.zeros(1000)
for i in tqdm(range(1000)):

    Mean_without_transcost[i]=hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number)[0]
    Mean_with_transcost[i]=hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number)[2]


# %% id="qCPCnGCP_Wt7" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
plt.figure(figsize=(10,10))
sns.distplot(Mean_without_transcost,color='blue',label='Without Cost')
plt.axvline(x=Mean_without_transcost.mean(),color='black',ls='--',lw=4)

sns.distplot(Mean_with_transcost,color='red',label="Wiht Cost")
plt.axvline(x=Mean_with_transcost.mean(),color='black',ls='--',lw=4)

plt.xlabel("Cost",size=20)
plt.legend()
plt.show()

# %% [markdown] id="UdQyjyUz_Wt7"
# ## Hedging PNL

# %% id="FeNCL8My_Wt7" executionInfo={"status": "aborted", "timestamp": 1701586212085, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
a=np.zeros(1000)
for i in tqdm(range(1000)):
    a[i]=hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number)[7]

# %% id="n2wr7Vkh_Wt7" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
sns.distplot(a,color='blue')
plt.axvline(x=a.mean(),color='black',ls='--',lw=4)

# %% [markdown] id="vTJ28qZt_Wt8"
# ### ITM ATM OTM case

# %% id="yMsRAWaJ_Wt8" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
ITM_=np.zeros(1000)
ATM_=np.zeros(1000)
OTM_=np.zeros(1000)
for i in tqdm(range(1000)):

    ITM_[i]=hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number)[4]
    OTM_[i]=hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number)[5]
    ATM_[i]=hedging_C(s0,mu,K,r,sigma,T,q,PutCall,simulation_number)[6]

# %% id="-fLzY7ea_Wt8" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
plt.figure(figsize=(10,10))
sns.distplot(ITM_,color='blue',label='ITM')
plt.axvline(x=ITM_.mean(),color='black',ls='--',lw=4)

sns.distplot(OTM_,color='red',label="OTM")
plt.axvline(x=OTM_.mean(),color='black',ls='--',lw=4)

sns.distplot(ATM_,color='gray',label="ATM")
plt.axvline(x=ATM_.mean(),color='black',ls='--',lw=4)

plt.xlabel("Cost",size=20)
plt.legend()
plt.show()

# %% id="6wH62w4Z_Wt8" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="umnUjshA_Wt8" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
-------------1차 절취선---------------

# %% id="SWBNb6Ss_Wt8" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="98C_Z0Xi_Wt9" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="vOp_XcN6_Wt9" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="MO_0sFVZ_Wt9" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="ScvVg1Pm_Wt9" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_c_all=pd.read_csv("kospi_opt_M11_K355.csv")
kospi_c_all.index=kospi_c_all['일자']
kospi_c_all=kospi_c_all.iloc[:,:11]
kospi_opt_df=kospi_c_all[kospi_c_all['거래량']>50]  # 거래량 50 이상
kospi_opt_df.iloc[-1]

# %% id="VOBZFjWd_Wt9" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df_etf=fdr.DataReader('069500')
df_etf=df_etf['Close']


# %% id="E3_7RDA0_Wt9" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df_etf_ret=df_etf.pct_change(1).dropna()

# %% id="IqfTIPkE_WuF" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
rolling_90_std=df_etf_ret.rolling(90).std()*np.sqrt(252)
rolling_90_std=rolling_90_std[(rolling_90_std.index<='2022-11-10')&(rolling_90_std.index>='2022-07-07')]
rolling_90_std

# %% id="4JnFaq5o_WuF" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df_etf_kospi=df_etf[(df_etf.index>="2022-07-07")&(df_etf.index<="2022-11-10")]
df_etf_kospi

# %% id="K1N5-qa5_WuF" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_200_s=pd.read_csv('Kospi_200_index.csv')
kospi_200_s.index=kospi_200_s['일자']
kospi_200_s=kospi_200_s[(kospi_200_s.index>="2022-07-07")&(kospi_200_s.index<="2022-11-10")]
kospi_200_s=kospi_200_s['종가'].loc[::-1]
kospi_200_s


# %% id="C8KMXCgZ_WuG" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
def BSprice(PutCall, x, T, K, r, q, sigma):
    d1 =(1/(sigma*np.sqrt(T)))*(np.log(x/K) + (r - q + 0.5*sigma**2)*T)
    d2 = (1/(sigma*np.sqrt(T)))*(np.log(x/K) + (r - q - 0.5*sigma**2)*T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nd1m = norm.cdf(-d1)
    Nd2m = norm.cdf(-d2)

    if PutCall == 'C':
        price1 = x*np.exp(-q*T)*Nd1 - K*np.exp(-r*T)*Nd2
        price2 = np.exp(-q*T)*Nd1
    elif PutCall =='P':
        price1 = K*np.exp(-r*T)*Nd2m - x*np.exp(-q*T)*Nd1m
        price2 = -np.exp(-q*T)*Nd1m

    return(price1,price2)


# %% id="vaZNTEf4_WuG" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
PutCall="C"
T=len(kospi_200_s.index)
tt=[]
for i in range(T-1):
    tt.append(T-1-i)
tt.append(0)
tt=np.array(tt)
T=tt
K=355
r=0.0321
sigma=rolling_90_std

# %% id="gWhxX6cf_WuG" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df_etf_kospi*0.01

# %% id="rc0TGQPH_WuG" executionInfo={"status": "aborted", "timestamp": 1701586212086, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
BSprice('C',df_etf_kospi*0.01,tt,K,r,0.0179,rolling_90_std)[0]  # Call price

# %% id="zG5JVFm-_WuH" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta=BSprice('C',df_etf_kospi*0.01,tt,K,r,0.0179,rolling_90_std)[1]   # Delta
Delta

# %% id="hOypbGxV_WuH" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_diff=Delta[1:]-Delta[:-1]

# %% id="IUNQLXUH_WuH" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_diff
Delta_change=np.repeat(Delta[0],len(Delta_diff)+1)
Delta_change[1:]=Delta_diff
Delta_change

# %% id="wBhqbG6y_WuH" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_change*100000

# %% id="osu0N9FW_WuL" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
1.39/0.05*250000

# %% id="IVVP3It__WuL" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
1.4/0.5*250000

# %% id="8ZDQMg3N_WuM" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="Djm5Y9bR_WuM" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
------ 절취선------- 아래는 실패작


# %% id="T_Xfazj5_WuM" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
def BSprice_Futures(PutCall, F, T, K, r, sigma):   ### John hull 책 415 페이지 관련내용
    d1 =(1/(sigma*np.sqrt(T)))*(np.log(F/K) + (0.5*sigma**2)*T)
    d2 = (1/(sigma*np.sqrt(T)))*(np.log(F/K) - (0.5*sigma**2)*T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nd1m = norm.cdf(-d1)
    Nd2m = norm.cdf(-d2)

    if PutCall == 'C':
        price1 = F*np.exp(-r*T)*Nd1 - K*np.exp(-r*T)*Nd2
        price2 = np.exp(-r*T)*Nd1
    elif PutCall =='P':
        price1 = K*np.exp(-r*T)*Nd2m - F*np.exp(-r*T)*Nd1m
        price2 = -np.exp(-r*T)*Nd1m

    return(price1,price2)           # returns array



# %% id="IO4iP1iD_WuM" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_c_all=pd.read_csv("kospi_opt_M11_K355.csv")
kospi_c_all.index=kospi_c_all['일자']
kospi_c_all=kospi_c_all.iloc[:,:11]
kospi_opt_df=kospi_c_all[kospi_c_all['거래량']>50]  # 거래량 50 이상
kospi_opt_df.iloc[-1]

# %% id="Au8_ezuR_WuM" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_200=pd.read_csv('Kospi_200_index.csv')
kospi_200.index=kospi_200['일자']
#kospi_200#=#kospi_200[(kospi_200.index>="2022-08-04")&(kospi_200.index<="2022-11-10")]
kospi_200=kospi_200['종가'].loc[::-1]
kospi_200=kospi_200[kospi_200.index=='2022-07-07']
kospi_200


# %% id="s32ngUHg_WuN" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_200_s=pd.read_csv('Kospi_200_index.csv')
kospi_200_s.index=kospi_200_s['일자']
kospi_200_s=kospi_200_s['종가']
kospi_200_s=kospi_200_s.pct_change(1).dropna()
kospi_200_s=kospi_200_s.loc[::-1]
rolling_90_std=kospi_200_s.rolling(90).std()*np.sqrt(252)
rolling_90_std=rolling_90_std.dropna()
rolling_90_std=rolling_90_std[(rolling_90_std.index<="2022-11-10")&(rolling_90_std.index>="2022-07-07")]
rolling_90_std

# %% id="4WsMnu7U_WuN" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_fut=pd.read_csv("kospi_200_Fut.csv") ## futures
kospi_fut.index=kospi_fut['일자']
kospi_fut=kospi_fut.iloc[:,1:]
kospi_fut=kospi_fut.iloc[:,:10].dropna()
kospi_fut=kospi_fut.loc[::-1]
kospi_fut=kospi_fut[(kospi_fut.index>='2022-07-07')&(kospi_fut.index<='2022-11-10')]
kospi_fut

# %% id="IroM0Sn0_WuN" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
# plt.figure(figsize=(20,10))
# kospi_fut=kospi_fut[kospi_fut.index>='2022-07-07']
# plt.plot(kospi_fut['종가'])
# plt.plot(kospi_200[kospi_200.index>='2022-07-07'])


# %% id="U2VPScHt_WuN" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
T=len(kospi_fut.index)
tt=[]
for i in range(T-1):
    tt.append(T-1-i)
tt.append(0)

# %% id="7OsrK-5G_WuO" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
PutCall="C"
F=kospi_fut['종가']
T=len(kospi_fut.index)
tt=[]
for i in range(T-1):
    tt.append(T-1-i)
tt.append(0)
tt=np.array(tt)
T=tt
K=355
r=0.0321
sigma=rolling_90_std

# %% id="J0Zl3G5K_WuO" executionInfo={"status": "aborted", "timestamp": 1701586212087, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta=BSprice_Futures(PutCall, F, T, K, r, sigma)[1]

# %% id="gx0WPve-_WuO" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
import numpy as np

# %% id="fwZUVWAM_WuO" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
(np.exp(0.03/12)*(6.23+4.05-2*5.05))/1.25**2

# %% id="2yzLxsZv_WuO" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta

# %% id="p_yB48Yp_WuO" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
plt.plot(Delta)

# %% id="eYrB4FrW_WuP" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
BSprice_Futures(PutCall, F, T, K, r, sigma)[0]

# %% id="Y21ll8s3_WuP" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_diff=Delta[:-1]-Delta[1:]
Delta_change=np.repeat(Delta[0],len(Delta_diff)+1)
Delta_change[1:]=Delta_diff

# %% id="FRsrCyUg_WuP" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_change

# %% id="NoukfUFE_WuP" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
shares_purchased=Delta_change*1000000
shares_purchased=shares_purchased.round()

# %% id="0Cv6wXNL_WuP" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
np.cumsum(shares_purchased*kospi_fut['종가'])[-1]

# %% id="9781yBBb_WuP" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_diff=Delta[:-1]-Delta[1:]


# %% id="ne9cTx1C_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_diff

# %% id="-dCLEkyb_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_change=np.repeat(Delta[0],len(Delta_diff)+1)
Delta_change[1:]=Delta_diff

# %% id="wKKOoS4T_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
shares_purchased=Delta_change*1000000
shares_purchased=shares_purchased.round()

# %% id="iPKXq7Un_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212088, "user_tz": -540, "elapsed": 16, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
shares_purchased

# %% id="FcoNwWZQ_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212089, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
np.cumsum(shares_purchased*kospi_fut['종가'])[-1]

# %% id="e1EHo_dj_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212089, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
1.39*1000000

# %% id="D4zulf1-_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212089, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_fut

# %% id="SPLwfSM7_WuQ" executionInfo={"status": "aborted", "timestamp": 1701586212089, "user_tz": -540, "elapsed": 17, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
T

# %% id="blq1AUeb_WuR" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
kospi_fut['종가']

# %% id="rzPn3RLy_WuR" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
Delta_diff.round(3)

# %% id="R4loxsqd_WuR" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
len(Delta_diff)


# %% id="UM5cJvpG_WuR" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
fdr.DataReader('069500','2022-01-01')

# %% id="TDtGbXYf_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="V16WUNSD_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
8+5+8+8

# %% id="sT5D1drD_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212090, "user_tz": -540, "elapsed": 18, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
9+7+10+10


# %% id="w51B9nty_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212091, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
--------절취선----------   # 위에까지가 Futures

# %% id="YQ8TpZll_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212091, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="YNw6_flx_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
s0=50
K=55
mu=0.13
r=0.04
sigma=0.4
q=0
PutCall='C'
T=20/52

# %% id="hPsq6WXq_WuS" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

M=1
M_s=20/M
M_s=np.int(M_s)
dt=M/52
tt=np.repeat(T,M_s)
for i in range(M_s-1):
    tt[i+1]=tt[i]-dt
tt=np.append(tt,0)



# Generating random
generating_number=int(round(T*(1/dt)))+1

# generating stock path
z=np.random.randn(simulation_number,generating_number)
stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
stock_path_bm[:,0]=1
stock_price=stock_path_bm.cumprod(1)*s0

# Calculating Delta
Delta=BSprice(PutCall,stock_price, tt, K, r, q, sigma)[1]
Delta_diff=Delta[:,1:]-Delta[:,:-1]
Delta_diff.shape

initial_array=np.repeat(Delta[0][0],Delta_diff.shape[0]).reshape(Delta_diff.shape[0],1)
Delta_change=np.concatenate([initial_array,Delta_diff],axis=1)

# Delta 변화에따른 stock trade
shares_purchased=Delta_change*100000
shares_purchased=shares_purchased.round()  #정수만 purchase



cost_purchased=shares_purchased*stock_price
trans_cost=np.where(cost_purchased<0,cost_purchased*0.001,0) #transaction Cost # 매도시에만 10BP
#trans_cost=cost_purchased*0.00015
trans_cost=abs(trans_cost)
transaction=np.cumsum(trans_cost,axis=1)
cum_cost=np.cumsum(cost_purchased,axis=1)

# Interest Cost 계산
for i in range(1,cum_cost.shape[1]):

    cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/52)+cost_purchased[:,i]


# Final cost 계산
Final_cost=cum_cost[:,-1]
hedge_cost=np.where(stock_price[:,-1]>K,Final_cost-K*100000,Final_cost)

Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
hedge_cost_tr=np.where(stock_price[:,-1]>K,Final_cost_tr-K*100000,Final_cost_tr)

ATM_C=np.where(stock_price[:,-1]==K,hedge_cost,0)    #ATM 으로 끝났을때
ITM_C=np.where(stock_price[:,-1]>K,hedge_cost,0)    #ITM 으로 끝났을때
OTM_C=np.where(stock_price[:,-1]<K,hedge_cost,0)    #OTM 으로 끝났을때

ATM_Case=ATM_C[ATM_C!=0]
ITM_Case=ITM_C[ITM_C!=0]
OTM_Case=OTM_C[OTM_C!=0]


hedging_PNL=(405567.6897503858-hedge_cost)

Mean=np.mean(hedge_cost)    # transcation cost 포함하지 않은 hedging cost
Performance_MSR=np.std(hedge_cost)/240000


Mean_tr=np.mean(hedge_cost_tr)   #transaction cost 포함함 hedging cost
Performance_MSR_tr=np.std(hedge_cost_tr)/240000
Mean_hedging_PNL=np.mean(hedging_PNL)


Mean,Performance_MSR,Mean_tr,Performance_MSR_tr,np.mean(ITM_Case),np.mean(OTM_Case),np.mean(ATM_Case)





# %% id="NCv0iM9z_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="IQuqZNuA_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="zRjNRHnh_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="MWxb4nIu_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="Mk3ecPVM_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
345*1.05

# %% id="3trm5w1R_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
yf.download('005930.KS',start='2022-05-13')

# %% id="5dop3kNk_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212092, "user_tz": -540, "elapsed": 19, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="oukxZ5DU_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="_NkXn8Ux_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="i1XwkLcc_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="3OhLHrXf_WuT" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="iKBLKVYf_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="a12NfrKx_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="ssYEG-TJ_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="K39j-7Ds_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="MIqKYus9_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="N_8aJZCv_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="YVyUe_q-_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="93oVI2a7_WuU" executionInfo={"status": "aborted", "timestamp": 1701586212093, "user_tz": -540, "elapsed": 20, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
-------- 2차 절취선-----------



# %% id="OFbRt6AV_WuV" executionInfo={"status": "aborted", "timestamp": 1701586212094, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
def hedging_Call(s0,M,mu,K,r,sigma,T,q,PutCall,simulation_number):
    M_s=20/M
    M_s=np.int(M_s)
    dt=M/52
    tt=np.repeat(T,M_s)
    for i in range(M_s-1):
        tt[i+1]=tt[i]-dt
    tt=np.append(tt,0)



    # Generating random
    generating_number=int(round(T*(1/dt)))+1

    # generating stock path
    z=np.random.randn(simulation_number,generating_number)
    stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
    stock_path_bm[:,0]=1
    stock_price=stock_path_bm.cumprod(1)*s0

    # Calculating Delta
    Delta=BSprice(PutCall,stock_price, tt, K, r, q, sigma)[1]
    Delta_diff=Delta[:,1:]-Delta[:,:-1]
    Delta_diff.shape

    initial_array=np.repeat(Delta[0][0],Delta_diff.shape[0]).reshape(Delta_diff.shape[0],1)
    Delta_change=np.concatenate([initial_array,Delta_diff],axis=1)

    # Delta 변화에따른 stock trade
    shares_purchased=Delta_change*100000
    shares_purchased=shares_purchased.round()  #정수만 purchase



    cost_purchased=shares_purchased*stock_price
    #trans_cost=np.where(cost_purchased<0,cost_purchased*0.0002,0) #transaction Cost
    trans_cost=cost_purchased*0.00015
    trans_cost=abs(trans_cost)
    transaction=np.cumsum(trans_cost,axis=1)
    cum_cost=np.cumsum(cost_purchased,axis=1)

    # Interest Cost 계산
    for i in range(1,cum_cost.shape[1]):

        cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/52)+cost_purchased[:,i]


    # Final cost 계산
    Final_cost=cum_cost[:,-1]
    hedge_cost=np.where(stock_price[:,-1]>50,Final_cost-5000000,Final_cost)

    Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
    hedge_cost_tr=np.where(stock_price[:,-1]>50,Final_cost_tr-5000000,Final_cost_tr)

    Mean=np.mean(hedge_cost)
    Performance_MSR=np.std(hedge_cost)/240000


    Mean_tr=np.mean(hedge_cost_tr)
    Performance_MSR_tr=np.std(hedge_cost_tr)/240000

    return(Mean,Performance_MSR,Mean_tr,Performance_MSR_tr)


M=[5,4,2,1,0.5,0.25]
simulation_number=1000
s0=49
K=50
mu=0.13
r=0.05
sigma=0.2
q=0
PutCall='C'
T=20/52
for i in M:

    Mean,MSR,Mean_tr,MSR_tr=hedging_Call(s0,i,mu,K,r,sigma,T,q,PutCall,simulation_number)

    if i==5:
        m5=Mean
        m5_tr=Mean_tr
        std5=MSR
        std5_tr=MSR_tr


    if i==4:
        m4=Mean
        m4_tr=Mean_tr
        std4=MSR
        std4_tr=MSR_tr

    if i==2:
        m2=Mean
        m2_tr=Mean_tr
        std2=MSR
        std2_tr=MSR_tr


    if i==1:
        m1=Mean
        m1_tr=Mean_tr
        std1=MSR
        std1_tr=MSR_tr

    if i==0.5:
        m05=Mean
        m05_tr=Mean_tr
        std05=MSR
        std05_tr=MSR_tr

    if i==0.25:

        m025=Mean
        m025_tr=Mean_tr
        std025=MSR
        std025_tr=MSR_tr


### without transaction cost

# %% id="Va_vffGY_WuV" executionInfo={"status": "aborted", "timestamp": 1701586212094, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df=pd.DataFrame([[m5_tr/100000,m4_tr/100000,m2_tr/100000,m1_tr/100000,m05_tr/100000,m025_tr/100000],[std5_tr,std4_tr,std2_tr,std1_tr,std05_tr,std025_tr]],columns=[5,4,2,1,0.5,0.25])
df.index=['Mean',"Performance MSR"]
df.columns.names=['Rebalance']
df

# %% id="_hvM70Hh_WuV" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 22, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
ITM_C=np.where(stock_price[:,-1]>K,hedge_cost,0)
OTM_C=np.where(stock_price[:,-1]<K,hedge_cost,0)
ITM_Case=ITM_C[ITM_C!=0]
OTM_Case=OTM_C[OTM_C!=0]
np.mean(ITM_Case),np.mean(OTM_Case)



# %% id="7X9aPdoT_WuV" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
ITM_Case[ITM_Case!=0]


# %% id="qxcegA5z_WuV" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

# %% id="3ynx2w_h_WuW" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
df_1=pd.DataFrame([stock_price[-1,:],Delta[-1,:],shares_purchased[-1,:],cost_purchased[-1,:],cum_cost[-1,:]],index=['stock','delta','#purch','$purch','cum']).T
df_1.round(3)

# %% id="4isUIwPE_WuW" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
# Generating random
generating_number=int(round(T*(1/dt)))+1

# generating stock path
z=np.random.randn(simulation_number,generating_number)
stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
stock_path_bm[:,0]=1
stock_price=stock_path_bm.cumprod(1)*s0

# %% id="XZloLA49_WuW" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
stock_price[0]

# %% id="YZLkheqA_WuX" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
PutCall="C"
x=50
T=20/52
K=55
r=0.04
q=0
sigma=0.46

# %% id="WSc8gwfN_WuX" executionInfo={"status": "aborted", "timestamp": 1701586212095, "user_tz": -540, "elapsed": 3241, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
BSprice(PutCall, x, T, K, r, q, sigma)[0]*100000

# %% id="-SnyvKXQ_WuX" executionInfo={"status": "aborted", "timestamp": 1701586212096, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
d1 =(1/(sigma*np.sqrt(T)))*(np.log(x/K) + (r - q + 0.5*sigma**2)*T)
d2 = (1/(sigma*np.sqrt(T)))*(np.log(x/K) + (r - q - 0.5*sigma**2)*T)
Nd1 = norm.cdf(d1)
Nd2 = norm.cdf(d2)
Nd1m = norm.cdf(-d1)
Nd2m = norm.cdf(-d2)



# %% [markdown] id="_dY1SNa9_WuY"
# # 유의성 검정

# %% id="CckIjL0B_WuY" executionInfo={"status": "aborted", "timestamp": 1701586212096, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
M=[5,4,2,1,0.5,0.25]
simulation_number=1000
s0=49
K=50
mu=0.13
r=0.05
sigma=0.2
q=0
PutCall='C'
T=20/52
Mean_5=[]
std_5=[]
Mean_4=[]
std_4=[]
Mean_2=[]
std_2=[]
Mean_1=[]
std_1=[]
Mean_05=[]
std_05=[]
Mean_025=[]
std_025=[]

Mean_tr_5=[]
std_tr_5=[]
Mean_tr_4=[]
std_tr_4=[]
Mean_tr_2=[]
std_tr_2=[]
Mean_tr_1=[]
std_tr_1=[]
Mean_tr_05=[]
std_tr_05=[]
Mean_tr_025=[]
std_tr_025=[]

for j in tqdm(range(1000)):
    for i in M:

        Mean,MSR,Mean_tr,MSR_tr=hedging_Call(s0,i,mu,K,r,sigma,T,q,PutCall,simulation_number)

        if i==5:
            Mean_5.append(Mean)
            std_5.append(MSR)
            Mean_tr_5.append(Mean_tr)
            std_tr_5.append(MSR_tr)

        if i==4:

            Mean_4.append(Mean)
            std_4.append(MSR)
            Mean_tr_4.append(Mean_tr)
            std_tr_4.append(MSR_tr)

        if i==2:

            Mean_2.append(Mean)
            std_2.append(MSR)
            Mean_tr_2.append(Mean_tr)
            std_tr_2.append(MSR_tr)

        if i==1:

            Mean_1.append(Mean)
            std_1.append(MSR)
            Mean_tr_1.append(Mean_tr)
            std_tr_1.append(MSR_tr)

        if i==0.5:

            Mean_05.append(Mean)
            std_05.append(MSR)
            Mean_tr_05.append(Mean_tr)
            std_tr_05.append(MSR_tr)

        if i==0.25:

            Mean_025.append(Mean)
            std_025.append(MSR)
            Mean_tr_025.append(Mean_tr)
            std_tr_025.append(MSR_tr)



# %% id="eKrhQ1hm_WuY" executionInfo={"status": "aborted", "timestamp": 1701586212096, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
s0=49
K=50
mu=0.13
r=0.05
sigma=0.2
q=0
PutCall='C'
T=20/52
simulation_number=10000

# %% id="ax8R00Mp_WuY" executionInfo={"status": "aborted", "timestamp": 1701586212096, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}

M=1
M_s=20/M
M_s=np.int(M_s)
dt=M/52
tt=np.repeat(T,M_s)
for i in range(M_s-1):
    tt[i+1]=tt[i]-dt
tt=np.append(tt,0)



# Generating random
generating_number=int(round(T*(1/dt)))+1

# generating stock path
z=np.random.randn(simulation_number,generating_number)
stock_path_bm=np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
stock_path_bm[:,0]=1
stock_price=stock_path_bm.cumprod(1)*s0

# Calculating Delta
Delta=BSprice(PutCall,stock_price, tt, K, r, q, sigma)[1]
Delta_diff=Delta[:,1:]-Delta[:,:-1]
Delta_diff.shape

initial_array=np.repeat(Delta[0][0],Delta_diff.shape[0]).reshape(Delta_diff.shape[0],1)
Delta_change=np.concatenate([initial_array,Delta_diff],axis=1)

# Delta 변화에따른 stock trade
shares_purchased=Delta_change*100000
shares_purchased=shares_purchased.round()  #정수만 purchase



cost_purchased=shares_purchased*stock_price
trans_cost=np.where(cost_purchased<0,cost_purchased*0.001,0) #transaction Cost # 매도시에만 10BP
#trans_cost=cost_purchased*0.00015
trans_cost=abs(trans_cost)
transaction=np.cumsum(trans_cost,axis=1)
cum_cost=np.cumsum(cost_purchased,axis=1)

# Interest Cost 계산
for i in range(1,cum_cost.shape[1]):

    cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/52)+cost_purchased[:,i]


# Final cost 계산
Final_cost=cum_cost[:,-1]
hedge_cost=np.where(stock_price[:,-1]>K,Final_cost-K*100000,Final_cost)

Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
hedge_cost_tr=np.where(stock_price[:,-1]>K,Final_cost_tr-K*100000,Final_cost_tr)

ATM_C=np.where((stock_price[:,-1]<(K+1))&(stock_price[:,-1]>(K-1)),hedge_cost,0)   #ATM 으로 끝났을때
ITM_C=np.where(stock_price[:,-1]>K,hedge_cost,0)    #ITM 으로 끝났을때
OTM_C=np.where(stock_price[:,-1]<K,hedge_cost,0)    #OTM 으로 끝났을때

ATM_Case=ATM_C[ATM_C!=0]
ITM_Case=ITM_C[ITM_C!=0]
OTM_Case=OTM_C[OTM_C!=0]


hedging_PNL=(c_price*100000-hedge_cost)

Mean=np.mean(hedge_cost)    # transcation cost 포함하지 않은 hedging cost
Performance_MSR=np.std(hedge_cost)/240000


Mean_tr=np.mean(hedge_cost_tr)   #transaction cost 포함함 hedging cost
Performance_MSR_tr=np.std(hedge_cost_tr)/240000
Mean_hedging_PNL=np.mean(hedging_PNL)
Mean,Performance_MSR,Mean_tr,Performance_MSR_tr,np.mean(ITM_Case),np.mean(OTM_Case),np.mean(ATM_Case),Mean_hedging_PNL




# %% id="EU5skAX-_WuY" executionInfo={"status": "aborted", "timestamp": 1701586212096, "user_tz": -540, "elapsed": 3242, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}}
ATM_Case
