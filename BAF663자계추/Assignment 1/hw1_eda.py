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
# # 자계추 HW1 EDA

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %% [markdown]
# ## Load data

# %%
CRSP_M_df = pd.read_csv(DATA_DIR / 'CRSP_M.csv')
permno_df = pd.read_csv(DATA_DIR / 'compustat_permno.csv') 
sample_df = pd.read_csv(DATA_DIR / 'assignment1_sample_data.csv')

# %%
sample_df # 이게 우리가 만들어야 하는 대상. 

# %%
CRSP_M_df # 교수님이 한 번 처리해놓은 데이터. 이거 가지고 작업하면 됨. 

# %%
permno_df # CRSP COMPUSTAT MERGE (CCM) / market data 들어있음. 당연히 써야 함. 

# %%
