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

# %% id="c7a23864"
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
# import FinanceDataReader as fdr
import yfinance as yf
import seaborn as sns


# %% id="c1069c91"
def BSprice(PutCall, S0, T, K, r, q, imp_vol):
    d1 =(1/(imp_vol*np.sqrt(T)))*(np.log(S0/K) + (r - q + 0.5*imp_vol**2)*T)
    d2 = (1/(imp_vol*np.sqrt(T)))*(np.log(S0/K) + (r - q - 0.5*imp_vol**2)*T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nd1m = norm.cdf(-d1)
    Nd2m = norm.cdf(-d2)

    if PutCall == 'C':
        price1 = S0 * np.exp(-q*T) * Nd1 - K * np.exp(-r*T) * Nd2
        price2 = np.exp(-q*T) * Nd1
    elif PutCall =='P':
        price1 = K * np.exp(-r*T) * Nd2m - S0 * np.exp(-q*T) * Nd1m
        price2 = -np.exp(-q*T) * Nd1m

    return(price1,price2)           # returns array


# %% [markdown] id="67c09760"
# ### 현재가 50,000원 strike = 55,000원  수량# 100,000 계약
# ### OPT 매도 포지션일때 내재 변동성 46%로 마진을 쌓아 놓고 Hedging PNL을 계산한다
# ### Int cost =4% daily calculation
# ### Transaction Cost= 0.1% at selling

# %% colab={"base_uri": "https://localhost:8080/"} id="e468f15f" executionInfo={"status": "ok", "timestamp": 1701589109904, "user_tz": -540, "elapsed": 7, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="b85cf3cf-ff0a-47ea-f8d4-2d0a9f276778"
PutCall='C'    # put / call 중에 어떤 상품인가
S0=50    # 기초자산 현재가격
T=1/2    # 만기
K=55    # strike price
r=0.04   # 자금비용 & risk-free rate
q=0    # 배당 (무시)
imp_vol=0.46    # impvol

c_price=BSprice(PutCall, S0, T, K, r, q, imp_vol)[0]
c_price*100000   #Margin

# %% id="fe4b5040"
simulation_number=1000
mu=0.13
sigma=0.4


# %% id="3511859d"
def hedging_C(S0, mu, K, r, sigma, T, q, PutCall, c_price, simulation_number):
    M=1
    M_s=180/M
    M_s=np.int(M_s)
    dt=M/360
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
    stock_price=stock_path_bm.cumprod(1)*S0

    # Calculating Delta
    Delta=BSprice(PutCall,stock_price, tt, K, r, q, imp_vol)[1]
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

        cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/360)+cost_purchased[:,i]


    # Final cost 계산
    Final_cost=cum_cost[:,-1]
    hedge_cost=np.where(stock_price[:,-1]>K,Final_cost-K*100000,Final_cost)

    Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
    hedge_cost_tr=np.where(stock_price[:,-1]>K,Final_cost_tr-K*100000,Final_cost_tr)

    ATM_C=np.where((stock_price[:,-1]<1.01*K)&(stock_price[:,-1]>0.99*K),hedge_cost,0)
    ITM_C=np.where(stock_price[:,-1]>K,hedge_cost,0)
    OTM_C=np.where(stock_price[:,-1]<K,hedge_cost,0)

    ATM_Case=ATM_C[ATM_C!=0]
    ITM_Case=ITM_C[ITM_C!=0]
    OTM_Case=OTM_C[OTM_C!=0]


    hedging_PNL=(c_price*100000-hedge_cost)

    Mean=np.mean(hedge_cost)    # transcation cost 포함하지 않은 hedging cost
    Performance_MSR=np.std(hedge_cost)/c_price


    Mean_tr=np.mean(hedge_cost_tr)   #transaction cost 포함함 hedging cost
    Performance_MSR_tr=np.std(hedge_cost_tr)/c_price
    Mean_hedging_PNL=np.mean(hedging_PNL)


    return(Mean,Performance_MSR,     Mean_tr,Performance_MSR_tr,        np.mean(ITM_Case),np.mean(OTM_Case),np.mean(ATM_Case),       Mean_hedging_PNL, stock_price)

# %% colab={"base_uri": "https://localhost:8080/", "height": 448} id="402647b5" executionInfo={"status": "ok", "timestamp": 1701589206926, "user_tz": -540, "elapsed": 97026, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="5d58fa95-a9e1-42fb-ba60-9ab77a918ce0"
for i in tqdm(range(1000)):
    plt.plot(hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[8][i])

# %% colab={"base_uri": "https://localhost:8080/"} id="0ba06540" executionInfo={"status": "ok", "timestamp": 1701589206926, "user_tz": -540, "elapsed": 8, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="0684a118-6176-491a-fe15-a985ba8ef577"
hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[8]

# %% [markdown] id="1627bb16"
# ### Transaction cost 비교

# %% colab={"base_uri": "https://localhost:8080/"} id="ebec8066" executionInfo={"status": "ok", "timestamp": 1701589369110, "user_tz": -540, "elapsed": 162189, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="645b76a2-1dd6-4a59-f758-94f60becc4e7"
Mean_without_transcost=np.zeros(1000)
Mean_with_transcost=np.zeros(1000)
for i in tqdm(range(1000)):

    Mean_without_transcost[i]=hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[0]
    Mean_with_transcost[i]=hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[2]

# %% colab={"base_uri": "https://localhost:8080/", "height": 864} id="3fe094af" executionInfo={"status": "ok", "timestamp": 1701589369911, "user_tz": -540, "elapsed": 828, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="e3c9d9d1-b08c-4e94-efa8-c0d650d0ab28"
plt.figure(figsize=(10,10))
sns.distplot(Mean_without_transcost,color='blue',label='Without Cost')
plt.axvline(x=Mean_without_transcost.mean(),color='black',ls='--',lw=4)

sns.distplot(Mean_with_transcost,color='red',label="Wiht Cost")
plt.axvline(x=Mean_with_transcost.mean(),color='black',ls='--',lw=4)

plt.xlabel("Cost",size=20)
plt.legend()
plt.show()

# %% [markdown] id="43ff0e7b"
# ## Hedging PNL

# %% colab={"base_uri": "https://localhost:8080/"} id="d589862d" executionInfo={"status": "ok", "timestamp": 1701589451908, "user_tz": -540, "elapsed": 82003, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="d414a422-60f8-4ecf-c2c2-e24fd0e89bdd"
a=np.zeros(1000)
for i in tqdm(range(1000)):
    a[i]=hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[7]

# %% colab={"base_uri": "https://localhost:8080/", "height": 448} id="f3b19971" executionInfo={"status": "ok", "timestamp": 1701589453069, "user_tz": -540, "elapsed": 524, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="c81c9d0b-023e-4760-8616-5d27152d7a67"
sns.distplot(a,color='blue')
plt.axvline(x=a.mean(),color='black',ls='--',lw=4)

# %% [markdown] id="5f0c20dc"
# ### ITM ATM OTM case

# %% colab={"base_uri": "https://localhost:8080/"} id="ec55d1b6" executionInfo={"status": "ok", "timestamp": 1701589692408, "user_tz": -540, "elapsed": 239343, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="2a33e89f-6542-4d7d-8291-26478de1db1f"
ITM_=np.zeros(1000)
ATM_=np.zeros(1000)
OTM_=np.zeros(1000)
for i in tqdm(range(1000)):

    ITM_[i]=hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[4]
    OTM_[i]=hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[5]
    ATM_[i]=hedging_C(S0,mu,K,r,sigma,T,q,PutCall, c_price, simulation_number)[6]

# %% colab={"base_uri": "https://localhost:8080/"} id="ut5N_CqwplMr" executionInfo={"status": "ok", "timestamp": 1701589865785, "user_tz": -540, "elapsed": 6, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="035576f0-537b-4445-fef7-4bf29a75d8eb"
ITM_

# %% colab={"base_uri": "https://localhost:8080/", "height": 864} id="b535130b" executionInfo={"status": "ok", "timestamp": 1701589693180, "user_tz": -540, "elapsed": 796, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="3ea81d90-3d21-4e92-d38d-1e1ba2091527"
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

# %% [markdown] id="87100ecc"
# ### Delta trigger

# %% colab={"base_uri": "https://localhost:8080/"} id="d6f3b87e" executionInfo={"status": "ok", "timestamp": 1701589693180, "user_tz": -540, "elapsed": 12, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="50501fbd-e43a-4b1a-c2f0-9b24a89a5a6e"
c_price=BSprice(PutCall="C", S0=50, T=26/52, K=55, r=0.04, q=0, imp_vol=0.46)[0]
c_price*100000   #Margin
c_price_origin=BSprice(PutCall="C", S0=50, T=26/52, K=55, r=0.04, q=0, imp_vol=0.40)[0]
c_price_origin   #Margin


# %% id="4f127103"
def delta_trigger_result(s0,K,mu,r,sigma,q,T,simulation_number,trigger):
    M=1
    M_s=180/M
    M_s=np.int(M_s)
    dt=M/360
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
    Delta=BSprice(PutCall,stock_price, tt, K, r, q, imp_vol)[1]
    Delta_diff=Delta[:,1:]-Delta[:,:-1]
    Delta_diff.shape
    Delta_diff_trigger=np.where(Delta_diff>Delta[:,:-1]*trigger,Delta_diff,0)

    initial_array=np.repeat(Delta[0][0],Delta_diff_trigger.shape[0]).reshape(Delta_diff_trigger.shape[0],1)
    Delta_change=np.concatenate([initial_array,Delta_diff_trigger],axis=1)

    # initial_array=np.repeat(Delta[0][0],Delta_diff.shape[0]).reshape(Delta_diff.shape[0],1)
    # Delta_change=np.concatenate([initial_array,Delta_diff],axis=1)

    # Delta 변화에따른 stock trade

    shares_purchased=Delta_change*100000
    np.where(shares_purchased)
    shares_purchased=shares_purchased.round()  #정수만 purchase

    cost_purchased=shares_purchased*stock_price
    trans_cost=np.where(cost_purchased<0,cost_purchased*0.001,0) #transaction Cost # 매도시에만 10BP
    #trans_cost=cost_purchased*0.00015
    trans_cost=abs(trans_cost)
    transaction=np.cumsum(trans_cost,axis=1)
    cum_cost=np.cumsum(cost_purchased,axis=1)

    # Interest Cost 계산
    for i in range(1,cum_cost.shape[1]):

        cum_cost[:,i]=cum_cost[:,i-1]*np.exp(r*M/360)+cost_purchased[:,i]


    # Final cost 계산
    Final_cost=cum_cost[:,-1]
    hedge_cost=np.where(stock_price[:,-1]>K,Final_cost-K*100000,Final_cost)

    Final_cost_tr=cum_cost[:,-1]+transaction[:,-1]
    hedge_cost_tr=np.where(stock_price[:,-1]>K,Final_cost_tr-K*100000,Final_cost_tr)

    # ATM_C=np.where((stock_price[:,-1]<(K+1))&(stock_price[:,-1]>(K-1)),hedge_cost,0)   #ATM 으로 끝났을때
    # ITM_C=np.where(stock_price[:,-1]>K,hedge_cost,0)    #ITM 으로 끝났을때
    # OTM_C=np.where(stock_price[:,-1]<K,hedge_cost,0)    #OTM 으로 끝났을때

    # ATM_Case=ATM_C[ATM_C!=0]
    # ITM_Case=ITM_C[ITM_C!=0]
    # OTM_Case=OTM_C[OTM_C!=0]


    hedging_PNL=(c_price*100000-hedge_cost)

    Mean=np.mean(hedge_cost)    # transcation cost 포함하지 않은 hedging cost
    Performance_MSR=np.std(hedge_cost)/(c_price_origin*100000)


    Mean_tr=np.mean(hedge_cost_tr)   #transaction cost 포함함 hedging cost
    Performance_MSR_tr=np.std(hedge_cost_tr)/(c_price_origin*100000)
    Mean_hedging_PNL=np.mean(hedging_PNL)

    return(Mean,Performance_MSR)
    #,Mean_tr,Performance_MSR_tr#,np.mean(ITM_Case),np.mean(OTM_Case),np.mean(ATM_Case),Mean_hedging_PNL




# %% id="64c639f6"
def delta_trigger_sim(trigger):
    mean_trig=[]
    std_trig=[]
    s0=50
    K=55
    mu=0.13
    r=0.04
    sigma=0.4
    q=0
    T=26/52
    simulation_number=1000
    for g in trigger:
        mean_trig.append(delta_trigger_result(s0,K,mu,r,sigma,q,T,simulation_number,g)[0])
        std_trig.append(delta_trigger_result(s0,K,mu,r,sigma,q,T,simulation_number,g)[1])

    return(mean_trig,std_trig)


# %% id="63060177"
trigger=np.linspace(0.05,0.4,50)
trigger=list(trigger)

# %% colab={"base_uri": "https://localhost:8080/"} id="f041b941" executionInfo={"status": "ok", "timestamp": 1701589701205, "user_tz": -540, "elapsed": 8032, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="05422598-a10c-469f-90f6-bcc80f886e1c"
delta_trigger_sim(trigger)

# %% [markdown] id="fb5b4abb"
# ### Trigger를 크게 줄수록 hedging cost 줄어듦

# %% colab={"base_uri": "https://localhost:8080/", "height": 880} id="b5d80be2" executionInfo={"status": "ok", "timestamp": 1701589715522, "user_tz": -540, "elapsed": 14321, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="2e85aa37-9fd7-41a1-c551-bdea883d173d"
plt.figure(figsize=(10,10))

y=delta_trigger_sim(trigger)[0]
x=trigger
plt.scatter(x,y,s=100,c='#33FFCE')
plt.plot(x,y,linestyle='solid',color='blue',label="Mean")
plt.xlabel("Trigger",labelpad=15)
plt.ylabel("Mean Cost",labelpad=15)
plt.legend(fontsize=14)
plt.show()

# %% [markdown] id="c9060808"
# ### Trigger 크게줄수록 STD증가

# %% colab={"base_uri": "https://localhost:8080/", "height": 865} id="b6867328" executionInfo={"status": "ok", "timestamp": 1701589723332, "user_tz": -540, "elapsed": 7815, "user": {"displayName": "\uac15\ubcd1\ud45c", "userId": "10016608623122375854"}} outputId="4f704d57-e41c-4d83-d272-24ac758f8de4"
plt.figure(figsize=(10,10))

y=delta_trigger_sim(trigger)[1]
x=trigger
plt.scatter(x,y,s=100,c='#33FFCE')
plt.plot(x,y,linestyle='solid',color='blue',label="STD")
plt.xlabel("Trigger",labelpad=15)
plt.ylabel("STD",labelpad=15)
plt.legend(fontsize=14)
plt.show()

# %% id="7wQP-q-XnVp-"
