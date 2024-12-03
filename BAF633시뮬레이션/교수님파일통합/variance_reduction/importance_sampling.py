
#%%
import numpy as np
from blackscholes import bsprice

def mcprice_importance_sampling(s,k,r,q,t,sigma,nsim,flag):
    z = np.random.randn(nsim)

    #original
    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    callOrPut = 1 if flag.lower()=='call' else -1    
    payoff = np.maximum(callOrPut*(st-k), 0)    
    disc_payoff = np.exp(-r*t)*payoff
    price = disc_payoff.mean()    
    se = disc_payoff.std(ddof=1) / np.sqrt(nsim)

    #importance sampling
    mu = (np.log(k/s) - (r-q-0.5*sigma**2)*t) / (sigma*np.sqrt(t)) # g는 다 다를 수 있는데, 교수님은 여기서 g를 표준정규분포에서 mu만큼만 옮긴걸로. 
    # 행사가 근처가 우리가 가장 필요한 구간이다. S_T와 K가 비슷하게 되도록 S_0 을 K까지 shift시켜줄 수 있는 mu를 찾아. 
    # deep OTM인 경우도, 분포를 strike price 근처로 옮겨주는 것이 중요하다.
    # 위의 mu 값이 바로, shift했을 때 K로 만들어주는 mu 값이다.

    z += mu # 그리고 평균에 mu 더해줬으니 이제 평균이 mu 
    likelihood_ratio = np.exp(-mu*z + 0.5*mu**2)

    st = s*np.exp((r-q-0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
    payoff = np.maximum(callOrPut*(st-k), 0)
    disc_payoff = np.exp(-r*t)*payoff*likelihood_ratio
    price_is = disc_payoff.mean()
    se_is = disc_payoff.std(ddof=1) / np.sqrt(nsim)

    return price, se, price_is, se_is, st.mean()


s, k, r, q, t, sigma = 100, 100, 0.03, 0.01, 0.25, 0.2

# MC Price = 3.766154 / se = 0.053175
# IS Price = 3.766154 / se = 0.053175
# se 차이를 봐라. 차이가 없네. 

s, k, r, q, t, sigma = 100, 90, 0.03, 0.01, 0.25, 0.2

# MC Price = 0.619989 / se = 0.020502
# IS Price = 0.641730 / se = 0.007187
# 확 줄었다. 

s, k, r, q, t, sigma = 100, 80, 0.03, 0.01, 0.25, 0.2

# MC Price = 0.033577 / se = 0.003905
# IS Price = 0.034263 / se = 0.000408
# ㄹㅇ 더 효과적. 


s, k, r, q, t, sigma = 100, 110, 0.03, 0.01, 0.25, 0.2 
# 망하는 케이스. se가 4배 커진다. 
# MC Price = 10.453360 / se = 0.082378
# IS Price = 10.163093 / se = 0.271772
# ITM에서 했으니까. 


flag = 'put'

#Analytic Formula
price = bsprice(s,k,r,q,t,sigma,flag)
print(f"Anlytic Price = {price:0.6f}")
print("-"*50)

#Control-Variates Simulation
nsim = 10000
mc_price, se, cv_price, cv_se, mu = mcprice_importance_sampling(s,k,r,q,t,sigma,nsim,flag)
print(f"MC Price = {mc_price:0.6f} / se = {se:0.6f}")
print(f"IS Price = {cv_price:0.6f} / se = {cv_se:0.6f}")
print("-"*50)
print(mu)
# %%

# 그냥 n 늘려서 1/3으로 줄이려면 9배를 늘려야 하는데 
# 겨우 행사가로 딱 shift했을 뿐인데도 훨씬 효과가 좋았다. 
# 극히 드문 케이스로 나타나는 보험 같은 것의 simulation도. 
# 겨우 1% 일어나는 일 simulatoin? 1개 path만 쓰고 99개는 버린다. 
# 이 경우 importance sampling이 매우 효과적. 
# 그 1%인 구간에서 집중적으로 sampling하도록 분포를 바꾸고 
# likelihood ratio로 조정
# 근데 잘못쓰면 안쓰니만 못하다. 

