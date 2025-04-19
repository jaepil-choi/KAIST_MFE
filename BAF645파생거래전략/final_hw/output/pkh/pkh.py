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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from scipy.ndimage.interpolation import shift
import scipy.stats as sst
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model

# %%
pd.set_option('display.float_format', '{:.2f}'.format)

# %%
from pathlib import Path
# OUTPUT_PATH = Path(r"C:\Users\USER\OneDrive\바탕 화면\임우섭\01. 카이스트\02. 입학후\03. 24-2학기\08. 파생상품 거래전략\04. 과제\02. Term project")

# CWD_PATH = Path.cwd()
# OUTPUT_PATH = CWD_PATH / "output"
# DATA_PATH = CWD_PATH / "data"

# %%
import os
import pandas as pd

# Get the current directory
current_dir = os.getcwd()

# List all files in the current directory
all_files = os.listdir(current_dir)

# Filter files starting with 'hc_pnl' and ending with '.pkl'
pickle_files = [file for file in all_files if file.startswith('hc_pnl') and file.endswith('.pkl')]

# Initialize an empty list to store DataFrames
dataframes = []

# Load each pickle file into a DataFrame and append it to the list
for file in pickle_files:
    df = pd.read_pickle(file)
    dataframes.append(df)

# Merge all DataFrames into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Display the merged DataFrame


# %%
merged_df

# %%
merged_df.describe()

# %%
