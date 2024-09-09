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
nsim = 10000
mc_price, se = mcprice(s,k,r,q,t,sigma,nsim,flag)
print(f"MC Price = {mc_price:0.6f} / se = {se:0.6f}")


#%%
import scipy.stats as sst
z = sst.norm.ppf(0.975)

nval = 10000
count = 1
for i in range(nval):
    print(i+1)
    mc_price, se = mcprice(s,k,r,q,t,sigma,nsim,flag)
    if price>mc_price+se*z or price<mc_price-se*z:
        count += 1

print("{0:0.4%}".format(count/nval))
