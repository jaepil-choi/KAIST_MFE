#%%
import numpy as np 
import pandas as pd
from scipy.linalg import solve_banded, solve, solveh_banded
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

# explicit은 따로 만들었다. 
# 왜? tri-diagonal matrix가 곱해질 때 따로 계산하는 것이 더 빠르기 때문이다.
# 안그러면 0인 부분들이 계속 곱해지기 때문에 계산량이 늘어난다.
def exfdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds
    a = dt*(vol*s[1:-1])**2 / (2*ds**2) # 맨 앞과 맨 뒤는 쓰지 않는다. 
    b = dt*(r-q)*s[1:-1] / (2*ds) # 마찬가지. 
    d, m, u = a-b, -2*a-dt*r, a+b

    v = np.maximum(callOrPut*(s-k), 0) # terminal condition. (만기에서의 payoff)

    for j in range(M-1,-1,-1):
        temp = d * v[:-2] + (1 + m) * v[1:-1] + u * v[2:] # v값 업데이트
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        v[1:-1] = temp
    f = interp1d(s,v)
    return pd.DataFrame({"S":s,"V":v}), f(s0)


#%%
s0, k, r, q, t, vol = 100, 100, 0.03, 0.01, 0.25, 0.4
optionType, maxS, N, M = "call", s0*2, 200, 2000
theta = 0

def fdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M, theta=1):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds

    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    A = np.diag(d[1:],-1) + np.diag(m) + np.diag(u[:-1],1)
    B = np.zeros((N-1,2))
    B[0,0], B[-1,1] = d[0], u[-1]

    Am = np.identity(N-1) - theta*A
    Ap = np.identity(N-1) + (1-theta)*A
    ab = np.zeros((3, N-1))
    ab[0,1:] = np.diag(Am,1)
    ab[1] = np.diag(Am)
    ab[2,:-1] = np.diag(Am,-1)

    v = np.maximum(callOrPut*(s-k), 0)
    for j in range(M-1,-1,-1):    
        # temp = Ap @ v[1:-1] + theta*B @ v[[0,-1]] # 이렇게 하면 계산량이 늘어난다. 특히 Ap @ v[1:-1] 부분.
        # 속도가 약 2배 차이남
        temp = (1-theta)*d * v[:-2] + (1 + (1-theta)*m) * v[1:-1] + (1-theta)*u * v[2:]
        temp[0] += theta*d[0]*v[0]
        temp[-1] += theta*u[-1]*v[-1]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        temp += (1-theta)*B @ v[[0,-1]]
        v[1:-1] = solve_banded((1,1), ab, temp)

    f = interp1d(s,v)
    return pd.DataFrame({"S":s,"V":v}), f(s0)


#%%
if __name__=="__main__":
    s = 100
    k = 100
    r = 0.03
    q = 0.01
    t = 0.25
    sigma = 0.2
    optionType = 'put'
    maxS, n, m = s*2, 1000, 10000
    v, ex_price = exfdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m)
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, ex_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m, 0)
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, im_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m)
    print(f"IM-FDM Price = {im_price:0.6f}")

    v, cn_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m, 0.5)
    print(f"CN-FDM Price = {cn_price:0.6f}")
