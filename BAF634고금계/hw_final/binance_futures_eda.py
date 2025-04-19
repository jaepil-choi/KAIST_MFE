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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'binance_futures.csv'  # Update this path if needed
binance_futures_df = pd.read_csv(file_path, header=[0, 1])

# Drop any fully empty columns or rows
binance_futures_df.dropna(how='all', axis=1, inplace=True)  # Drop fully empty columns
binance_futures_df.dropna(how='all', axis=0, inplace=True)  # Drop fully empty rows

# Descriptive statistics for key metrics
descriptive_stats = binance_futures_df.describe(include=[float])
print("Descriptive Statistics of Binance Futures Data:")
descriptive_stats


# %%

# Flatten multi-level column names for easier processing
binance_futures_df.columns = ['_'.join(col).strip() for col in binance_futures_df.columns]

# Filter for columns containing 'quote_volume' to calculate average quote volumes per symbol
quote_volume_cols = [col for col in binance_futures_df.columns if 'quote_volume' in col]

# Calculate the mean quote volume for each symbol
quote_volume_means = binance_futures_df[quote_volume_cols].mean()
quote_volume_means = quote_volume_means.reset_index()
quote_volume_means.columns = ['Symbol', 'Average Quote Volume']


# %%
print("\nAverage Quote Volume per Symbol:")
quote_volume_means['Symbol'] = quote_volume_means['Symbol'].str.replace('/USDT:USDT_quote_volume', '')
quote_volume_means.set_index('Symbol', inplace=True)
quote_volume_means.sort_values('Average Quote Volume', ascending=False, inplace=True)

# %%
quote_volume_means.plot(kind='bar', title='Average Quote Volume per Symbol', ylabel='Average Quote Volume', xlabel='Symbol', figsize=(15, 7))

# %%
quote_volume_means_percent = quote_volume_means / quote_volume_means.sum() * 100

quote_volume_means_percent.round(2)

# %%
