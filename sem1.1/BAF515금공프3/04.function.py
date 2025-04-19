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
# # Chapter 4. Function

# %%
import numpy as np


# %% [markdown]
# ![image.png](attachment:image.png)

# %%
def count_odd(l):
    odd_count = 0
    for i in l:
        if i % 2 != 0:
            odd_count += 1
    
    return odd_count


# %%
a = count_odd(l=[1,2,5,7,8])
a

# %%
np.where(np.array([1,2,5,7,8]) % 2 != 0, 1, 0).sum()


# %% [markdown]
# Self-Exercise: Numerical Square Root

# %%
def num_sqrt(n):
    # Initial Values
    epsilon = 0.00001
    
    r = n/2
    lin1 = n/r
    lin2 = r 

    while abs(lin1 - lin2) >= epsilon:
        r = (lin1 + lin2) / 2 # 이걸 잘못 써서 느렸던 것. 처음에 (lin1 - lin2) / 2로 했었다. 수렴하지 않는 방향. 
        lin1 = n/r
        lin2 = r
    
    return r


# %%
num_sqrt(9)

# %%
num_sqrt(10000)

# %%
10000**(1/2)


# %%
# 교수님 풀이 

def mySqrt(n):
    old = 0
    new = n/2

    while abs(old-new) >= 1e-10:
        old = new
        new = 1/2 * (old + n/old)
    
    return new


# %%
mySqrt(9)

# %% [markdown]
# Function Scope 
#
# - Local vs Global 
#     - global은 원래 함수 내에서 읽을 수만 있고 수정은 안됨. 
#     - `global`을 통해 함수의 local scope 내에서도 write 할 수 있다. 
# - 함수가 variable 을 탐색하는 순서:
#     - LEGB (Local Enclosed Global Built-in)
#     - nested inner의 경우 enclosed를 local 다음으로 읽고, 할당이 안된다. 

# %% [markdown]
# 질문: 
#
# - function 종료될 때 local 에 할당한 변수들 다 메모리 회수되나? 
# - pandas 같은것에서 변수에 dataframe을 하나씩 transform 해가며 할당할 때, 가급적 같은 변수에 계속 재할당 하는 것이 좋은지? (garbage collection 관점에서)

# %%
a = 200
b = 300
def outer():
    a = 10
    def inner():
        c = 30
        print(a, b, c)
    inner()
    a = 22
    inner()


# %%
a = 200
b = 300
def outer():
    a = 10
    def inner():
        c = 30
        a = 99 # enclosed 변수는 nested inner에서 수정할 수 없다.  
        print('inner', a, b, c)
    inner()
    print('outer', a, b)
    a = 22
    inner()


# %%
outer()

# %%
