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

# %%
import numpy as np

# %%
a = 0.1

# %%
a + a

# %%
a + a + a

# %%
3 * a == 0.3

# %%
4 * a == 0.4

# %% [markdown]
# 컴퓨터는 이진수기 때문에 표현 안되는 숫자들이 있다. 
#
# 그래서 truncation error을 피하기 위해 무조건 작은 epsilon을 쓰는 것이 장땡이 아니게 되는 것임. 
