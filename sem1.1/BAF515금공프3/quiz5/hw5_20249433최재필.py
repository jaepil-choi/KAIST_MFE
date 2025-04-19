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
# # 금공프 HW5
#
# 20249433 최재필

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. 

# %%
np.random.seed(123)

arr = np.random.randn(8, 10)

# %% [markdown]
# ### (1)    

# %%
arr.sum(axis=0)

# %% [markdown]
# ### (2)

# %%
row_idx, col_idx = np.where(arr > 2)

print(f'row index: {row_idx}')
print(f'column index: {col_idx}')

# %% [markdown]
# ## 2. 

# %%
values = [0, 1, 2, 3]
index = ['a', 'b', 'c', 'd']

# %%
# 1. Create series from list/numpy array
s = pd.Series(data=values, index=index)
s

# %%
# 2. Create series from dictionary
s = pd.Series(data={k: v for k, v in zip(index, values)})
s

# %%
# 3. Create series from scalar value
s = pd.Series(data=0, index=index)

s['b'] = 1
s['c'] = 2
s['d'] = 3

s

# %% [markdown]
# ## 3. 

# %%
Snew = pd.Series({
    'a': 1,
    'b': 4,
    'c': 2,
    'd': 3,
})
Snew

# %%
# Use index number to access values
Snew[1:3]

# %%
# Use index value to access values
Snew[['b', 'c']]

# %%
# Use the .loc[] accessor to select elements by index
Snew.loc['b':'c']

# %%
# Use the .iloc[] accessor to select elements by position
Snew.iloc[1:3]

# %% [markdown]
# ## 4

# %%
DF = pd.DataFrame(
    data=np.random.randn(6, 7), 
    columns=list('abcdefg'), 
    index=[3, 2, 4, 5, 1, 0],
    )
DF

# %% [markdown]
# ### (1)

# %%
DF.iloc[:, 2:5]

# %% [markdown]
# ### (2)

# %%
DF[['c', 'd', 'e']]

# %% [markdown]
# ## 5. 

# %% [markdown]
# ### (1)

# %%
DF.loc[DF['c'] < 0, 'c']

# %% [markdown]
# ### (2)

# %%
DF[(DF['c'] < 0)]['c']
