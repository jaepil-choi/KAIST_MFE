#%%
import numpy as np
from blackscholes import bsprice

def mcprice_controlvariates(s,k,r,q,t,sigma,nsim,flag):
    z = np.random.randn(nsim)
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    callOrPut = 1 if flag.lower()=='call' else -1    
    payoff = np.maximum(callOrPut*(st-k), 0)    
    disc_payoff = np.exp(-r*t)*payoff
    price = disc_payoff.mean()    
    se = disc_payoff.std(ddof=1) / np.sqrt(nsim)

    c = np.cov((disc_payoff, st), ddof=1) # covariance를 계산하고
    cv_disc_payoff = disc_payoff - c[1,0]/c[1,1]*(st-s*np.exp((r-q)*t)) # 공분산 나누기 분산
    cv_price = cv_disc_payoff.mean()
    cv_se = cv_disc_payoff.std(ddof=1) / np.sqrt(nsim)

    return price, se, cv_price, cv_se 


def mcprice_antithetic(s,k,r,q,t,sigma,nsim,flag):
    z = np.random.randn(nsim)
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    callOrPut = 1 if flag.lower()=='call' else -1    
    payoff = np.maximum(callOrPut*(st-k), 0)    
    disc_payoff = np.exp(-r*t)*payoff
    price = disc_payoff.mean()    
    se = disc_payoff.std(ddof=1) / np.sqrt(nsim)

    z[nsim/2:] = -z[:nsim]
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    payoff = np.maximum(callOrPut*(st-k), 0)    
    disc_payoff = np.exp(-r*t)*payoff
    price2 = disc_payoff.mean()    
    se2 = disc_payoff.std(ddof=1) / np.sqrt(nsim)
    return price, se, price2, se2 

s, k, r, q, t, sigma = 100, 100, 0.03, 0.01, 0.25, 0.2
flag = 'put'

#Analytic Formula
price = bsprice(s,k,r,q,t,sigma,flag)
print(f"   Price = {price:0.6f}")
print("-"*50)
#Control-Variates Simulation
nsim = 10000
mc_price, se, cv_price, cv_se= mcprice_controlvariates(s,k,r,q,t,sigma,nsim,flag)
print(f"MC Price = {mc_price:0.6f} / se = {se:0.6f}")
print(f"CV Price = {cv_price:0.6f} / se = {cv_se:0.6f}")
print("-"*50)
#Antithetic
mc_price, se, price2, se2= mcprice_controlvariates(s,k,r,q,t,sigma,nsim,flag)
print(f"MC Price = {mc_price:0.6f} / se = {se:0.6f}")
print(f"Antithetic Price = {price2:0.6f} / se = {se2:0.6f}")
print("-"*50)
# %%
