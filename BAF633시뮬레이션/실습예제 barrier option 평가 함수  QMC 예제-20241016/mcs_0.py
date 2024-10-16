import numpy as np

def mcprice(s,k,r,q,t,sigma,nsim,flag):
    z = np.random.randn(nsim)
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    callOrPut = 1 if flag.lower()=='call' else -1    
    payoff = np.maximum(callOrPut*(st-k), 0)    
    price = np.exp(-r*t)*payoff.mean()    
    return price