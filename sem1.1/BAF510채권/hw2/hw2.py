# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HW3
#
# 20249433 최재필

# %%
import sympy as sp
import numpy as np
import math

# %% [markdown]
# ## 2. par yield 빈 칸 채우기

# %% [markdown]
# year 0.5부터
#
# $$
#
# 100 = \frac{C_{\frac{1}{2}} + 100 }{(1+\frac{z_{0.5}}{2})^{2 \times 0.5}} 
#
# $$
#
# 에서 $ C_{\frac{1}{2}} $ 는 주어짐. $ z_{\frac{1}{2}} $ 를 구해야 함. 

# %%
# c = sp.symbols('c')

c_bi = (0.0536 * 100)/ 2
z05 = sp.symbols('z05')


# %%
eq = sp.Eq(
    100, 
    (c_bi+100)/(1+z05/2)**(2 * 0.5)
    )
eq

# %%
z05_n = sp.nsolve(eq, z05, 0.05)
z05_n

# %% [markdown]
# year 1.0
#
# $$
#
# 100 = \frac{C_{\frac{1}{2}}}{(1+\frac{z_{0.5}}{2})^{2 \times 0.5}}  
# + \frac{C_{\frac{1}{2}} + 100 }{(1+\frac{z_{1.0}}{2})^{2 \times 1.0}} 
#
# $$
#
# 에서 $ C_{\frac{1}{2}} $ 는 주어짐. $ z_{\frac{1}{2}} $ 를 구해야 함. 

# %%
c_bi = (0.0501 * 100)/ 2

z10 = sp.symbols('z10')


# %%
eq = sp.Eq(
    100, 
    (c_bi)/(1+z05_n/2)**(2 * 0.5) + 
    (c_bi+100)/(1+z10/2)**(2 * 1.0)
    )
eq

# %%
z10_n = sp.nsolve(eq, z10, 0.05)
z10_n

# %% [markdown]
# year 1.5
#
# $$
#
# 100 = \frac{C_{\frac{1}{2}}}{(1+\frac{z_{0.5}}{2})^{2 \times 0.5}}  
# + \frac{C_{\frac{1}{2}}}{(1+\frac{z_{1.0}}{2})^{2 \times 1.0}} 
# + \frac{C_{\frac{1}{2}} + 100 }{(1+\frac{z_{1.5}}{2})^{2 \times 1.5}} 
#
# $$
#
# 에서 이제 C 가 주어지지 않음. 앞에서 구한 것을 바탕으로 1.5의 par yield를 구해야 함.

# %%
c15 = sp.symbols('c15')

z15 = sp.symbols('z15')


# %%
eq = sp.Eq(
    100, 
    (c15)/(1+z05_n/2)**(2 * 0.5) + 
    (c15)/(1+z10_n/2)**(2 * 1.0) +
    (c15+100)/(1+z15/2)**(2 * 1.5)
    )
eq

# %%
z10_n = sp.nsolve(eq, z10, 0.05)
z10_n

# %%

# %%

# %%

# %%
p0 = 110/1.1005
p0

# %%
p_u = 110/1.1015 # price up
p_u

# %%
p_d = 110/1.0995 # price down
p_d

# %%
p_d = 100

# %%
bp20 = 0.002

duration = -(1/p0) * (p_u - p_d)/bp20
duration

# %%
(100-99.8638)/(99.9546*0.002)

# %% [markdown]
# accrued interest

# %%
# 103-22+ US Treasury
T_value = 103 + 22/32 + 1/64
T_value


# %%
c_rate = 0.06125

C_semi = 100 * c_rate/2
C_semi

# %%
accrued_interest = C_semi * 152/183
accrued_interest

# %%
full_price = accrued_interest + T_value
full_price

# %%
