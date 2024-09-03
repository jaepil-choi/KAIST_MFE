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
