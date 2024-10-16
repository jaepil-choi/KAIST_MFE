#%%
import numpy as np
import time 

def mc_barrier_price(s,k,r,q,t,sigma,option_flag,nsim,b,barrier_flag,m):
    dt = t/m
    z = np.random.randn(nsim,m)
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
    se = disc_payoff.std(ddof=1) / np.sqrt(nsim)
    return price, se
