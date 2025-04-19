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
# # 우섭형 delta band 케이스별로 
#
#

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from tqdm import tqdm


plt.rc('font', family='Malgun Gothic')


# %%
pd.set_option('display.float_format', '{:.0f}'.format)

# %%
CWD_PATH = Path.cwd()
DATA_PATH = CWD_PATH / 'data'
OUTPUT_PATH = CWD_PATH / 'output'

# %%
SELECTION = [
    '삼성전자',
    '현대모비스',
    'NAVER',
    '카카오'
]

# %%
dfs = []

for selec in SELECTION:
    temp_df = pd.read_pickle(OUTPUT_PATH / f'all_final_results_{selec}.pkl')
    temp_df['udly_name'] = selec
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True, sort=False, axis=0)

# %%
sorted_df = df.sort_values(by=['udly_name', 'hedge_cost'], ascending=[True, True])
top_k = 5

top_k_df = sorted_df.groupby('udly_name').head(top_k).copy()

# %%
top_k_df['in_group_rank'] = top_k_df.groupby('udly_name')['hedge_cost'].rank(ascending=True, method='first')

# %%
top_k_df.sort_values(by=['udly_name', 'in_group_rank'], ascending=[True, True], inplace=True)

# %%
top_k_df

# %%
