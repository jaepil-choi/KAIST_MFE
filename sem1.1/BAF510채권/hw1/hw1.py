# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: sandbox311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 채권분석 hw1
#
#

# %%
import sympy as sp
import numpy as np
import math

# %% [markdown]
# ## Q3

# %%
FV = 100 # 액면가
P = 92.5067 # 딜러 요구 가격

C_r = 0.08 # 쿠폰 이자율

frequency = 2 # 6개월마다 쿠폰 지급
maturity = 3 # 잔존만기 3년

# %%
r = sp.symbols('r', real=True) # 채권의 수익률

# %% [markdown]
# $$
# P = \sum_{i=1}^{2\times3}\frac{C_{\frac{1}{2}}}{(1+\frac{r}{2})^i} + \frac{FV}{(1+\frac{r}{2})^6}
# $$

# %%
C_biannual = FV * C_r / frequency
C_biannual

# %%
i = sp.symbols('i')
coupon_sum = sp.Sum(C_biannual / (1 + r / frequency) ** i, (i, 1, frequency * maturity)).doit()
coupon_sum

# %%
equation = sp.Eq(P, coupon_sum + FV / (1 + r / frequency) ** (frequency * maturity))
equation

# %%
solution = sp.solve(equation, r)
solution

# %%
ans = [s for s in solution if s.is_real and s >= 0][0]
ans

# %% [markdown]
# ## Q4

# %%
e = sp.E

# %%
coupon_sum = sp.Sum(C_biannual / e**((i/frequency)*r), (i, 1, frequency * maturity)).doit()
coupon_sum

# %%
equation = sp.Eq(P, coupon_sum + FV / e**(r*(maturity*frequency/frequency)))
equation

# %%
# solution = sp.solve(equation, r)
# solution

# %%
solution = sp.nsolve(equation, r, 0.10) # numeric solution으로 빠르게 찾는 법. 대신 어느 정도 근사값이 필요
solution

# %% [markdown]
# 또는 이렇게 풀면 된다. 
#
# 3번에서 구한 연 2회 이산 복리를 연속 복리로 "변환"

# %%
equation = sp.Eq(e**r, (1+ans/frequency)**frequency)
equation

# %%
sp.solve(equation, r)

# %%
math.log((1+ans/frequency)**frequency)


# %% [markdown]
# 나중에 쓰기 위한 함수화

# %%
def get_coupon_sum(C_r, maturity, frequency, FV=100):
    i = sp.symbols('i')
    r = sp.symbols('r', real=True) # 채권의 수익률
    e = sp.E

    C_1period = FV * C_r / frequency
    coupon_sum = sp.Sum(C_1period / e**((i/frequency)*r), (i, 1, frequency * maturity)).doit()

    return coupon_sum



# %%
def get_bond_equation(PV, C_r, maturity, frequency, FV=100):
    i = sp.symbols('i')
    r = sp.symbols('r', real=True) # 채권의 수익률
    e = sp.E

    coupon_sum = get_coupon_sum(C_r, maturity, frequency, FV)
    bond_equation = sp.Eq(PV, coupon_sum + FV / e**(r*(maturity*frequency/frequency)))

    return bond_equation



# %%
FV = 100 # 액면가
PV = 92.5067 # 딜러 요구 가격

C_r = 0.08 # 쿠폰 이자율

frequency = 2 # 6개월마다 쿠폰 지급
maturity = 3 # 잔존만기 3년

# %%
equation = get_bond_equation(PV, C_r, maturity, frequency, FV)
init_r = 0.10

r = sp.symbols('r', real=True) # 채권의 수익률

solution = sp.nsolve(equation, r, init_r)
