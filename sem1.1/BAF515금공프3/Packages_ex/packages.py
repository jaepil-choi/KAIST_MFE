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
# # Packages 실습

# %% [markdown]
# - `SmartPhones`가 package, `TMP`가 subpackage

# %%
from SmartPhones.TMP import abc

# %%
abc.abc_ftn()

# %%

from SmartPhones import IPhone, Galaxy, TMP

# %%
IPhone.IPhone_ftn()

# %%
Galaxy.Galaxy_ftn()

# %%
