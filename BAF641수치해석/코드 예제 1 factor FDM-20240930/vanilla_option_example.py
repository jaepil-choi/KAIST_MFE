#%%
from blackscholes import bsprice
from fdm import fdm_vanilla_option, exfdm_vanilla_option
import numpy as np 
import time


### 기본 파라미터
s = 100
k = 100
r = 0.03
q = 0.01
t = 0.25
sigma = 0.2
optionType = 'put'

#Analytic Formula
t0 = time.time()
price = bsprice(s,k,r,q,t,sigma,optionType)
print(f"Analytic Price = {price:0.6f}")
print("computation time = ", time.time()-t0, "\n")


## 이러면 주가의 max가 100*2가 적절하냐? 는 정하기 나름임. 
maxS, n, m = s*2, 1000, 10000 # n는 주가의 갯수, m은 시간의 갯수
t0 = time.time()
v, ex_price = exfdm_vanilla_option(s, k, r, q, t, sigma, optionType, # explicit으로 계산
                                   maxS, n, m)
print(f"EX-FDM Price = {ex_price:0.6f}")
print("computation time = ", time.time()-t0, "\n")

t0 = time.time()
v, ex_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, # explicit
                                   maxS, n, m, 0)
print(f"EX-FDM Price = {ex_price:0.6f}")
print("computation time = ", time.time()-t0, "\n")

t0 = time.time()
v, im_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, # implicit
                                   maxS, n, m)
print(f"IM-FDM Price = {im_price:0.6f}")
print("computation time = ", time.time()-t0, "\n")

t0 = time.time()
v, cn_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, # 0.5를 주면 Crank-Nicolson
                                   maxS, n, m, 0.5)
print(f"CN-FDM Price = {cn_price:0.6f}")
print("computation time = ", time.time()-t0, "\n")