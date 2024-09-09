import numpy as np

def mcprice(s,k,r,q,t,sigma,nsim,flag):
    z = np.random.randn(nsim)
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    callOrPut = 1 if flag.lower()=='call' else -1    
    payoff = np.maximum(callOrPut*(st-k), 0)    
    disc_payoff = np.exp(-r*t)*payoff
    price = disc_payoff.mean()    
    se = disc_payoff.std(ddof=1) / np.sqrt(nsim)
    return price, se