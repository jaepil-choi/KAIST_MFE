# %%
# 20249272 윤수한
import numpy as np
import scipy.stats as stats

def mc_barrier_price(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    n = int(n/2)
    dt = t/m
    dts = np.arange(dt,t+dt,dt)
    callorput = 1 if option_flag.lower()=='call' else -1
    upordown = 1 if barrier_flag.split('-')[0] == 'up' else -1
    inorout = 1 if barrier_flag.split('-')[1] == 'in' else -1
    
    # 난수 추출 - Terminal Stratified method을 위한 z , z_tm 생성
    U = np.random.uniform(0,1,n)
    V = (np.arange(n) + U) / n
    z_tm = stats.norm.ppf(V)
    z = np.random.randn(n,m)
    
    # moment matching method로 normal 분포 맞춰주기
    z_tm = (z_tm - z_tm.mean())/z_tm.std(ddof=1)
    z = (z - z.mean(axis = 1)[:,np.newaxis])/z.std(axis=1,ddof=1)[:,np.newaxis]
    
    # matching 후 antithetic 방법으로 난수 복제
    z_tm = np.concatenate([z_tm, -z_tm], axis = 0)
    z = np.concatenate([z, -z], axis = 0)

    # 각 알고리즘에 따라 Important sampling 진행
    if inorout == 1 and upordown == 1:
        if callorput == 1:
            mu = (np.log(b/s) - (r-q)*t) / (sigma*np.sqrt(t)) # up-in call 일땐 barrier와 strike price에 상관없이 모두 적용, 
        else:
            mu = (np.log(b/s) - (r-q)*t) / (sigma*np.sqrt(t)) if k > s else 0 # up-in put 일땐 ITM, ATM 에서만 적용
    elif inorout == 1 and upordown == -1:
        if callorput == 1:
            mu = (np.log(b/s) - (r-q)*t) / (sigma*np.sqrt(t)) if s > k else 0 # down-in call 일땐 ITM, ATM 에서만 적용
        else:
            mu = (np.log(b/s) - (r-q)*t) / (sigma*np.sqrt(t)) # down-in put 일땐 barrier가 strike 에 상관없이 모두 적용
    elif inorout == -1 and upordown == 1:
        if callorput == 1:
            # up-out call 일땐 OTM일때만 적용 또한 barrier의 95% 신뢰구간 안에 strike price가 없다면 적용 안함
            mu = (np.log(k/s) - (r-q)*t) / (sigma*np.sqrt(t)) if s < k and np.abs(k-s) < 2 else 0 
        else:
            # up-out put 일땐 OTM일때만 적용 또한 barrier의 95% 신뢰구간 안에 strike price가 없다면 적용 안함
            mu = (np.log(k/s) - (r-q)*t) / (sigma*np.sqrt(t)) if s > k and np.abs(k-s) < 2 else 0 
    elif inorout == -1 and upordown == -1:
        if callorput == 1:
            # down-out call 일땐 OTM일때만 적용 또한 barrier의 95% 신뢰구간 안에 strike price가 없다면 적용 안함
            mu = (np.log(k/s) - (r-q)*t) / (sigma*np.sqrt(t)) if s < k and np.abs(k-s) < 2 else 0 
        else:
            # down-out put 일땐 OTM일때만 적용 또한 barrier의 95% 신뢰구간 안에 strike price가 없다면 적용 안함
            mu = (np.log(k/s) - (r-q)*t) / (sigma*np.sqrt(t)) if s > k and np.abs(k-s) < 2 else 0 
    else:
        mu = 0 # 포함되지 않는 경우가 있을 수 있으므로 mu = 0 처리
    
    # 분포 옮기기
    z, z_tm = z + mu, z_tm + mu
    likelihood_ratio = np.exp(-mu*z_tm + 0.5*mu**2) # Likelyhood 구하기

    # Terminal Stratification method로 Brownian motion 생성
    w_tm = np.sqrt(t) * z_tm
    z = z.cumsum(axis=1)
    w = np.sqrt(dt) * z
    w_br = w + (w_tm - w[:, -1])[:,np.newaxis] * dts
    
    # 생성된 Brownian motion으로 주가 path 생성
    st = s * np.exp((r-q-0.5*sigma**2)*dts + sigma*w_br)
    
    # barrier knock 여부 판단
    barrier_knock = st.max(axis = 1)>=b if barrier_flag.split("-")[0].lower()=='up' else st.min(axis = 1)<=b
    if barrier_flag.split('-')[1].lower()=="out": barrier_knock = ~barrier_knock 

    # plain vanilla option 가격 계산
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1 = stats.norm.cdf(callorput*d1)
    nd2 = stats.norm.cdf(callorput*d2)
    bsprice = callorput*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)
        
    # payoff 계산
    pv_disc_payoff = np.exp(-r * t) * np.maximum(callorput * (st[:,-1]-k), 0) * likelihood_ratio
    barrier_disc_payoff = pv_disc_payoff * barrier_knock
    
    # plain vanilla option을 이용한 control covariate method 적용
    cov_matrix = np.cov((pv_disc_payoff, barrier_disc_payoff), ddof=1)
    y_ib = barrier_disc_payoff - cov_matrix[1,0]/cov_matrix[1,1] * (pv_disc_payoff - bsprice)
    barrier_price = y_ib.mean()
    
    return barrier_price
