# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: sandbox39
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BAF515 금공프3 Quiz 2
#
# **20249433 MFE 최재필**
#

# %% [markdown]
# ## 1. 
#
# ![image.png](attachment:image.png)

# %%
icecream = {
    'cheery': [12, 100], # price, inventory
    'cookies': [18, 200],
    'greentea': [15, 140],
    'mango': [10, 200],
}

# %% [markdown]
# ### (1)

# %%
icecream['cookies'][0] = 13
icecream

# %% [markdown]
# ### (2)
#

# %%
icecream['Strawberry'] = [12, 170]
icecream['pistachio'] = [15, 100]
icecream

# %% [markdown]
# ## 2. 
#
# ![image.png](attachment:image.png)

# %%
d1 = {
    'key1': 1,
    'key2': 3,
    'key3': 2,
    'key10': 7,
}

d2 = {
    'key3': 1,
    'key2': 2,
    'key7': 2,
    'key5': 4,
    'key1': 7,
    'key9': 8,
    'key0': 7,
}

# %% [markdown]
# ### (1)

# %%
sorted(list(d2.values()))[-2:]

# %% [markdown]
# ### (2)

# %%
set(d2.values())

# %% [markdown]
# ### (3)

# %%
if set(d1.keys()) <= set(d2.keys()):
    print('True')
else:
    print('False')

# %% [markdown]
# ### (4)

# %%
set(d1.values()) & set(d2.values())

# %% [markdown]
# ## 3.
#
# ![image.png](attachment:image.png)

# %%
A = input('Arbitrary String:')
A

# %%
if A.startswith('#'):
    A = '*' + A[1:]
else:
    A = '*' + A

A


# %% [markdown]
# ## 4. 
#
# ![image.png](attachment:image.png)

# %%
n = 10

summation = 0
for i in range(1, n+1):
    summation += i**4

summation
