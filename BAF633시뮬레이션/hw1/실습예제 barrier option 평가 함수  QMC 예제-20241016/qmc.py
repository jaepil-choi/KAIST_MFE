#%%
import numpy as np
import pandas as pd
from scipy.stats import qmc
from blackscholes import bsprice
from mcs_0 import mcprice
from ql_barrier_option import ql_barrier_price
from mc_barrier_option import mc_barrier_price

def qmcprice(s,k,r,q,t,sigma,nsim,flag):
    dist = qmc.MultivariateNormalQMC(mean=[0], cov=[[1]])
    z = dist.random(nsim)
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    callOrPut = 1 if flag.lower()=='call' else -1    
    payoff = np.maximum(callOrPut*(st-k), 0)    
    disc_payoff = np.exp(-r*t)*payoff
    price = disc_payoff.mean()
    return price

def qmc_barrier_price(s,k,r,q,t,sigma,option_flag,nsim,b,barrier_flag,m):
    dt = t/m
    dist = qmc.MultivariateNormalQMC(mean=np.zeros(m), cov=np.identity(m))
    z = dist.random(nsim)
    z = z.cumsum(1)
    dts = np.arange(dt,t+dt,dt)
    st = s*np.exp((r-q-0.5*sigma**2)*dts + sigma*np.sqrt(dt)*z)
    barrier_knock = st.max(1)>=b if barrier_flag.split("-")[0].lower()=='up' else st.min(1)<=b
    if barrier_flag.split('-')[1].lower()=="out": 
        barrier_knock = ~barrier_knock
    callOrPut = 1 if option_flag.lower()=='call' else -1
    payoff = np.maximum(callOrPut*(st[:,-1]-k), 0) * barrier_knock
    disc_payoff = np.exp(-r*t)*payoff
    price = disc_payoff.mean()    
    return price


#%%
#plain-vanilla
s, k, r, q, t, sigma = 100, 90, 0.03, 0.01, 0.25, 0.2
flag = 'put'

#Analytic Formula
price = bsprice(s,k,r,q,t,sigma,flag)
print(f"Anlytic Price = {price:0.6f}")
prcs = pd.DataFrame(columns=["Analytic","MC","upper","lower","QMC"])
for n in range(5,20):
    nsim = 2**n
    mc_price, se = mcprice(s,k,r,q,t,sigma,nsim,flag)
    qmc_price = qmcprice(s,k,r,q,t,sigma,nsim,flag)
    prcs.loc[n] = [price, mc_price, mc_price+2*se, mc_price-2*se, qmc_price]

prcs.plot()


#%%
#barrier option
s,k,r,q,t,sigma = 100, 100, 0.03, 0, 1, 0.2
b, rebate = 130, 0
m = 5
option_flag = 'call'
barrier_flag = 'up-out'
#Analytic Formula
an_price = ql_barrier_price(s,k,r,t,sigma,option_flag,b,rebate,barrier_flag)

prcs = pd.DataFrame(columns=["Analytic","MC","QMC"])
for n in range(5,20):
    nsim = 2**n
    mc_price, se = mc_barrier_price(s,k,r,q,t,sigma,option_flag,nsim,b,barrier_flag,m)
    qmc_price = qmc_barrier_price(s,k,r,q,t,sigma,option_flag,nsim,b,barrier_flag,m)
    prcs.loc[n] = [an_price, mc_price, qmc_price]

prcs.plot()

