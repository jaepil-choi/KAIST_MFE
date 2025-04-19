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
# # 채권 hw3

# %%
from datetime import datetime, timedelta
import sympy as sp
import math

# %% [markdown] vscode={"languageId": "plaintext"}
# ## 1

# %%
treasury = 101 + 25/32 + 1/64 # 101-25+
treasury

# %%
FV = 1000
Cr = 0.07
C_semi = FV * Cr/2


# %%
def date_difference(start_date_str, end_date_str, is_inclusive=False):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    day_difference = (end_date - start_date).days
    if is_inclusive:
        day_difference += 1
    
    return day_difference



# %%
paydate = '2024-02-05'

last_coupon_date = '2023-10-15'
upcoming_coupon_date = '2024-04-15'

# days_to_next_coupon = date_difference(paydate, upcoming_coupon_date)
days_from_the_last_coupon = date_difference(last_coupon_date, paydate)
coupon_period = date_difference(last_coupon_date, upcoming_coupon_date)

days_from_the_last_coupon, coupon_period

# %%
accrued_interest = C_semi * days_from_the_last_coupon / coupon_period
accrued_interest

# %%
full_price = FV*(treasury/100) + accrued_interest 
full_price

# %% [markdown]
# ## 2

# %%
zero1 = 0.04
zero2 = 0.05
p = 1/2
FV = 100

log_std = 0.005  

# %%
PV = 100 / (1+zero2)**2
PV

# %%
ru = sp.symbols('ru')
rd = sp.symbols('rd')
e = math.e

# %%
eq1 = sp.Eq(ru, rd * e**(2*log_std))
eq1

# %%
eq2 = sp.Eq(PV, 1/(1+zero1) * 1/2 * ( 100/(1+ru) + 100/(1+rd) ))
eq2

# %%
solution = sp.solve([eq1, eq2], (ru, rd))
solution

# %%
ru_solve = solution[1][0]
rd_solve = solution[1][1]

ru_solve, rd_solve

# %% [markdown]
# ## 3

# %%
maturity = 2
coupon = 8
freq = 1

# %%
call_strike = 100
market_price = 101.9

# %%

# %% [markdown]
# ## 4

# %%

# %% [markdown]
# ## 5

# %%
A_par = 300
B_par = 200
C_par = 200
D_par = 250

# %%
A_coupon = 0.07
B_coupon = 0.0675
C_coupon = 0.0725
D_coupon = 0.0775

# %%
r = 0.09

# %%
pool = A_par * (r - A_coupon) + B_par * (r - B_coupon) + C_par * (r - C_coupon) + D_par * (r - D_coupon)
pool

# %%
pool * 1000000 / r

# %%
