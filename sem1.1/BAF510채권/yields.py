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
# # Yields

# %%
import sympy as sp
import numpy as np
import math


# %% [markdown]
# zero rates
#
# $$
#
# Pm = \frac{FV}{(1+z)^T}
#
# $$

# %% [markdown]
# $$
#
# z = (\frac{FV}{Pm})^{1/T} - 1
#
# $$

# %%
def get_zero_rate(Pm, T, FV=100):
    z = (FV/Pm)**(1/T) - 1

    return z


# %% [markdown]
# forward rate
# $$
#
# (1+z_{n})^n = (1+z_{n-1})^{n-1} \times (1+f_{n,1})
#
# $$

# %%
def get_forward_from_zero(z_n, z_n_prev, n):
    f = (1 + z_n)**n / (1 + z_n_prev)**(n-1) - 1

    return f


# %%

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
##TODO: Discrete sum 할 수 있도록 해야. e로 하면 continuous compounding. discrete, continuous 나눠줘야.



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

# %%
get_bond_equation(100, 0.0857, 4, 1, 100)

# %%
