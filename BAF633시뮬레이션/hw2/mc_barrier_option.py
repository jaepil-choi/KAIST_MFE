# BAF633 시뮬레이션 과제2: Barrier Option Pricing using Variance Reduction
# 20249433 최재필

import numpy as np
import scipy.stats as sst

def bsprice(s, k, r, q, t, sigma, flag):
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    callOrPut = 1 if flag.lower()=='call' else -1
    nd1 = sst.norm.cdf(callOrPut*d1)
    nd2 = sst.norm.cdf(callOrPut*d2)
    price = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)
    return price


def mc_barrier_price(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    callOrPut = 1 if option_flag.lower() == 'call' else -1
    upOrDown, inOrOut = barrier_flag.lower().split('-')

    dt = t / m  # Time step
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    # Antithetic variates
    n_half = n // 2
    z = np.random.randn(n_half, m)
    z = np.vstack((z, -z))

    # Simulate paths
    increments = drift + vol * z
    log_s = np.log(s) + np.cumsum(increments, axis=1)
    s_paths = np.exp(log_s)
    s_paths = np.hstack((s * np.ones((n, 1)), s_paths))

    # Payoffs
    if upOrDown == 'up':
        if inOrOut == 'out':
            barrier_crossed = np.any(s_paths >= b, axis=1)
            payoff_paths = np.where(barrier_crossed, 0.0, np.maximum(callOrPut * (s_paths[:, -1] - k), 0))
        else:  # 'in' 
            barrier_crossed = np.any(s_paths >= b, axis=1)
            payoff_paths = np.where(barrier_crossed, np.maximum(callOrPut * (s_paths[:, -1] - k), 0), 0.0)
    else:  # 'down'
        if inOrOut == 'out':
            barrier_crossed = np.any(s_paths <= b, axis=1)
            payoff_paths = np.where(barrier_crossed, 0.0, np.maximum(callOrPut * (s_paths[:, -1] - k), 0))
        else:  # 'in' 
            barrier_crossed = np.any(s_paths <= b, axis=1)
            payoff_paths = np.where(barrier_crossed, np.maximum(callOrPut * (s_paths[:, -1] - k), 0), 0.0)

    # Discount payoffs
    disc_payoffs = np.exp(-r * t) * payoff_paths

    # Control Variates: Use the vanilla option as control variate
    vanilla_price = bsprice(s, k, r, q, t, sigma, option_flag)
    vanilla_payoffs = np.maximum(callOrPut * (s_paths[:, -1] - k), 0)
    disc_vanilla_payoffs = np.exp(-r * t) * vanilla_payoffs

    # Compute covariance and adjust payoffs
    cov = np.cov(disc_payoffs, disc_vanilla_payoffs)
    beta = cov[0, 1] / cov[1, 1]
    adjusted_payoffs = disc_payoffs - beta * (disc_vanilla_payoffs - vanilla_price)

    # Estimate price
    price = np.mean(adjusted_payoffs)

    return price


if __name__ == '__main__':
    import numpy as np

    s = 100         
    k = 100         
    r = 0.05        
    q = 0.02        
    t = 1.0         
    sigma = 0.2     
    option_flag = 'call' 
    n = 100000      
    b = 120         
    barrier_flag = 'up-out'  
    m = 252         

    price = mc_barrier_price(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m)
    print(price)
