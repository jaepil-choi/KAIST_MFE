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
# # 금공프3 퀴즈4
#
# MFE 20249433 최재필

# %% [markdown]
# ## 1. 

# %% [markdown]
# Answer: 3
#
# Why?
# - Python's import system is hierarchical, which means that importing a top-level package does not automatically import its subpackages or modules. 
# - Given `__init__.py ` is empty, importing just a top-level package `game` does not import its subpackages/modules. 

# %% [markdown]
# ## 2. 

# %%
import numpy as np

abc = np.arange(5)
abc[0] = 10.2345
abc

# %%
abc.dtype

# %% [markdown]
# Why?
#
# - Because the array was originally created as an array of integers, its `dtype` is `int32`. 
# - To maintain the same type across all elements, numpy transformed the float value to int. 

# %% [markdown]
# ## 3. 

# %%
np.random.seed(1)
rmat = np.random.randint(10, size=(3, 4))
rmat

# %%
rmat2 = rmat[0]
rmat2

# %% [markdown]
# Why?
#
# - `rmat2` is of course one-dimensional because it's selecting the first (0th) element of `rmat`, which is basically an array of arrays. 
# - The first row is the first (0th) element of `rmat`, so naturally `rmat2` is one-dimensional instead of two. 

# %% [markdown]
# ## 4. 

# %%
a = np.arange(15).reshape(3, 5)
a.shape

# %%
b = np.arange(3)
b.shape

# %%
np.hstack([a, b])

# %% [markdown]
# Why? 
# - `a`'s shape is (3, 5) while `b` is (3, )
# - To perform `hstack`, all dimensions should match except for the concatenation axis. 
# - Since `a`'s concatenation axis is 5 and `b` is None, they cannot be stacked. 

# %% [markdown]
# ## 5. 

# %%
a = np.arange(15).reshape(3, 5)
a.shape

# %%
b = np.arange(3)
b.shape

# %%
b[None, :].shape # The same as b[np.newaxis, :]

# %% [markdown]
# Why?
#
# - `+` is element-wise operation in Numpy. To perform element-wise operation, the arrays should be in the same shape or are broadcastable. 
# - To use broadcasting, `b` should be changed like the below. 

# %%
a + b[:, None]
