#%%
from blackscholes import bsprice
import numpy as np 

s = 100
k = 100
r = 0.03
q = 0.01
t = 0.25
sigma = 0.2
flag = 'put'

#Analytic Formula
price = bsprice(s,k,r,q,t,sigma,flag)
print(f"   Price = {price:0.6f}")

#Monte-Carlo Simulation
from mcs_0 import mcprice
nsim = 1000000
mc_price = mcprice(s,k,r,q,t,sigma,nsim,flag)
print(f"MC Price = {mc_price:0.6f}")

# TODO: 가격의 std error와 신뢰구간을 구해보자. 100번 iter 해서 그 안에 95번 들어오는지도 check. 
# TODO: BS 공식에서 0~T가 아니라 j에서 k 막 이러면 식은 어떻게 바뀌어야 하나? 
