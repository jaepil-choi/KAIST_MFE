# 시뮬레이션방법론 최종과제 소스코드
# 20249132 김형환

import numpy as np
import scipy.stats as sst

def mc_barrier_price(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    
    # Set parameters
    dt = t/m
    dts = np.arange(dt, t+dt, dt)
    barrier_up, barrier_out = barrier_flag.startswith('up'), barrier_flag.endswith('out')
    option_call = option_flag.lower() == 'call'
    option_type = 1 if option_call else -1
    moneyness_otm = 1 if option_type * (k - (1 + option_type * 0.2 * np.sqrt(t) * sigma) * s) >= 0 else 0
    
    # (1) Stratified sampling, z_t makes price at T & z makes brownian bridge
    z_t = sst.norm.ppf((np.arange(n) + np.random.uniform(0,1,n)) / n)
    z = np.random.randn(n,m)
    
    # (2) Moment matching in z_t
    z_t = np.where(n>=100, (z_t - z_t.mean()) / z_t.std(ddof=1), z_t - z_t.mean())
    
    # (3) Antithetic variate
    z_t, z = np.concatenate([z_t, -z_t], axis=0), np.concatenate([z, -z], axis=0)
    
    # (4) Importance sampling at z_t
    if barrier_out:
        if moneyness_otm: mu = (np.log(k/s) - (r-q-0.5*sigma**2)*t) / (sigma*np.sqrt(t))
        else: mu = 0 # Knock-out & ATM, OTM then importance sampling is not applied.
    else:
        if barrier_up + option_call == 1: mu = 0 # Down-In call & Up-In put are not applied.
        else: mu = (np.log(b/s) - (r-q-0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    z_t = z_t + mu
    likelihood_ratio = np.exp(-mu*z_t + 0.5*mu**2)
    
    # Generate underlying paths using brownian bridge
    w_t, w = z_t * np.sqrt(t), z.cumsum(axis=1) * np.sqrt(dt) # winner process
    bridge = dts * ((w_t- w[:,-1]).reshape(len(w),1) + w / dts) # brownian bridge
    paths = s*np.exp((r-q-0.5*sigma**2)*dts + sigma*bridge) # underlying price path

    # Determine whether barrier touch or not (exists payoff or not)
    if barrier_up: knock = paths.max(axis=1) >= b
    else: knock = paths.min(axis=1) <= b
    if barrier_out: knock = ~knock
    
    # Caculate options payoff
    plain_npv = np.maximum(option_type*(paths[:,-1]-k), 0) * np.exp(-r*t) * likelihood_ratio
    barrier_npv = knock * plain_npv
    
    # (5) Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(option_type*d1), sst.norm.cdf(option_type*d2)
    plain_bsprice = option_type*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)
    
    cov_npv = np.cov(barrier_npv,plain_npv,ddof=1)
    beta = np.where(cov_npv[1,1]==0,0,cov_npv[0,1] / cov_npv[1,1])
    barrier_CVnpv = barrier_npv - beta * (plain_npv - plain_bsprice)

    barrier_price = barrier_CVnpv.mean()

    return barrier_price