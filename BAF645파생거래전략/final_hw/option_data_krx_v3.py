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
# # 거래소에서 option data 가져와보기
#
# - daily snapshot으로 전 행사가, 전 만기를 가져올 수 있어 좋다. 데이터 양이 너무 많다는 것이 문제라면 문제 
# - 너무 많이 요청했을 때 차단 당한다. 
#     - 차단 안당하게 충분히 sleep 넣어주고, retry도 multiplier 높여주기. 
#     - 카이스트 ip 차단 방지하기 위해 피씨방에 돈 충전하고 크롤링은 parsec으로 원격으로 돌리기. 
# - 차단/중단시 그동안 한거라도 건지기 위해 계속 parquet으로 daily chunk를 저장. 
#
# - o1 정식 모델을 사용하니 코드 퀄리티가 훨씬 올라갔다. 

# %%
# main.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List
from tqdm import tqdm
import os
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

from pathlib import Path

# Import configurations
from krx_config import API_URL, HEADERS, PAYLOAD_TEMPLATE, H5_SCHEMA

# Adjust these paths and constants as needed
DATA_PATH = Path('data')
BACKUP_PATH = Path('backup')
OUTPUT_PATH = Path('output')
BACKUP_INTERVAL = 10  # Backup after every 10 successful scrapes
PARQUET_DIR = DATA_PATH / "krx_option_parquet"  # Directory to store parquet files
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """
    Returns a descending list of business days between start_date and end_date in 'YYYYMMDD' format.
    """
    tz = ZoneInfo('Asia/Seoul')
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz)
    delta = end - start
    date_list = []
    for i in range(delta.days + 1):
        current_date = end - timedelta(days=i)
        if current_date.weekday() < 5:
            date_list.append(current_date.strftime("%Y%m%d"))
    return date_list

def setup_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retry_strategy = Retry(
        total=3,
        status_forcelist=[400, 429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        backoff_factor=1000
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_option_data(session: requests.Session, trade_date: str) -> pd.DataFrame:
    payload = PAYLOAD_TEMPLATE.copy()
    payload["trdDd"] = trade_date
    response = session.post(API_URL, data=payload)
    response.raise_for_status()
    data = response.json()
    if "output" not in data:
        return pd.DataFrame()
    return pd.DataFrame(data["output"])

def save_data_parquet(df: pd.DataFrame, trade_date: str):
    # Convert data types if needed; for parquet, this is often optional, but let's ensure strings:
    for col, (dtype, _) in H5_SCHEMA.items():
        if col in df.columns and dtype == 'object':
            df[col] = df[col].astype(str)
    # Save each day's data as a separate parquet file
    file_path = PARQUET_DIR / f"option_data_{trade_date}.parquet"
    df.to_parquet(file_path, index=False)

def backup_parquet_dir(source_dir: Path, backup_path: Path):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = backup_path / f"krx_option_parquet_backup_{timestamp}"
    shutil.copytree(source_dir, backup_dir)
    print(f"Backup created at {backup_dir}")

def scrape_krx_option_data(start_date: str, end_date: str) -> None:
    dates = generate_date_range(start_date, end_date)
    success_count = 0
    session = setup_session()

    for date in tqdm(dates, desc="Fetching data"):
        try:
            daily_data = fetch_option_data(session, date)
            if not daily_data.empty:
                daily_data['Trade_Date'] = pd.to_datetime(date, format='%Y%m%d')
                save_data_parquet(daily_data, date)
                success_count += 1
                if success_count % BACKUP_INTERVAL == 0:
                    backup_parquet_dir(PARQUET_DIR, BACKUP_PATH)
            
            time.sleep(10)
        
        except requests.HTTPError as http_err:
            print(f"HTTP error for date {date}: {http_err}")
            time.sleep(100)
        except Exception as err:
            print(f"Error for date {date}: {err}")
            time.sleep(100)

    # Final backup after completion
    backup_parquet_dir(PARQUET_DIR, BACKUP_PATH)


# %%
# # Example usage:
# start = '2024-12-02'
# end = '2024-12-04'
# scrape_krx_option_data(start, end)

# # pc방에서 처리함


# %%
import pandas as pd
from pathlib import Path

# Schema and type info from your snippet
column_to_name = {
    'ISU_CD': 'option_sid(full)',
    'ISU_SRT_CD': 'option_sid(short)',
    'ISU_NM': 'option_name',
    'TDD_CLSPRC': 'close_price',
    'FLUC_TP_CD': 'up_or_down',
    'CMPPREVDD_PRC': 'price_change',
    'TDD_OPNPRC': 'open_price',
    'TDD_HGPRC': 'high_price',
    'TDD_LWPRC': 'low_price',
    'IMP_VOLT': 'im_vol',
    'NXTDD_BAS_PRC': 'next_day_base_price',
    'ACC_TRDVOL': 'trade_volume',
    'ACC_TRDVAL': 'trade_value',
    'ACC_OPNINT_QTY': 'open_interest_quantity',
    'SECUGRP_ID': 'security_type',
    'Trade_Date': 'trade_date',
    'Uly_Code': 'underlying_code',
}

float_cols = [
    'close_price', 'price_change', 'open_price', 'high_price', 'low_price',
    'im_vol', 'next_day_base_price', 'trade_volume', 'trade_value',
    'open_interest_quantity', 'expiration', 'up_or_down'
]

DATA_PATH = Path('data')
PARQUET_DIR = DATA_PATH / "krx_option_parquet"

df_all = pd.DataFrame()

for fidx, file in enumerate(PARQUET_DIR.glob("*.parquet")):
    daily_df = pd.read_parquet(file)
    # Rename columns
    daily_df = daily_df.rename(columns=column_to_name)

    # Convert columns to numeric
    for float_col in float_cols:
        if float_col in daily_df.columns:
            daily_df[float_col] = daily_df[float_col].str.replace('-', '', regex=False)
            daily_df[float_col] = daily_df[float_col].str.replace(',', '', regex=False)
            daily_df[float_col] = pd.to_numeric(daily_df[float_col], errors='raise')

    # Convert trade_date to datetime
    if 'trade_date' in daily_df.columns:
        daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])

    # Concatenate to the main DataFrame
    df_all = pd.concat([df_all, daily_df], ignore_index=True)

    if fidx % 100 == 0:
        print(f"Processed {fidx} files")
        print(f'Current DataFrame memory usage: {df_all.memory_usage(deep=True).sum() / (1024**2):.2f} MB')

# print(df_all.head())
print("Final DataFrame memory usage: {:.2f} MB".format(
    df_all.memory_usage(deep=True).sum() / (1024**2)
))


# %%
df_all.info()

# %%
# strike price 등의 정보를 이름에서 파싱 

import re

def parse_option_name(option_name):
    pattern = r"(?P<underlying>[\w가-힣]+)\s+(?P<call_or_put>[CP])\s+(?P<expiration>\d{6})\s+(?P<strike>[\d,]+)\(\s*(?P<multiplier>\d+)\)"

    # Match the pattern
    match = re.match(pattern, option_name)

    # Extract data if a match is found
    if match:
        data = {
            "underlying": match.group("underlying"),
            "call_or_put": match.group("call_or_put"),
            "expiration": match.group("expiration"),
            "strike": float(match.group("strike").replace(",", "")),
            "multiplier": int(match.group("multiplier")),
        }

        return data
        
    else:
        data = {
            "underlying": None,
            "call_or_put": None,
            "expiration": None,
            "strike": None,
            "multiplier": None,
        }

        return data


# %%
# strike price 등의 정보를 이름에서 파싱 
# 특이 케이스 때문에 re 수정

import re

def parse_option_name(option_name):
    # Updated pattern to handle underlying names with spaces
    pattern = r"(?P<underlying>[A-Za-z가-힣\s]+)\s+(?P<call_or_put>[CP])\s+(?P<expiration>\d{6})\s+(?P<strike>[\d,]+)\(\s*(?P<multiplier>\d+)\)"

    # Match the pattern
    match = re.match(pattern, option_name)

    # Extract data if a match is found
    if match:
        data = {
            "underlying": match.group("underlying").strip(),  # Strip leading/trailing spaces
            "call_or_put": match.group("call_or_put"),
            "expiration": match.group("expiration"),
            "strike": float(match.group("strike").replace(",", "")),
            "multiplier": int(match.group("multiplier")),
        }
        return data
        
    else:
        # Return None values if the pattern doesn't match
        data = {
            "underlying": None,
            "call_or_put": None,
            "expiration": None,
            "strike": None,
            "multiplier": None,
        }
        return data

# Example usage
option_name = "LS ELECTRIC C 202401 120,000(100)"
parsed_data = parse_option_name(option_name)
print(parsed_data)


# %%
parsed_data = df_all['option_name'].apply(parse_option_name)
parsed_df = pd.DataFrame(parsed_data.tolist())

# Combine the parsed data with the original DataFrame

df_final = pd.concat([df_all, parsed_df], axis=1)

# %%
print("Real final DataFrame memory usage: {:.2f} MB".format(
    df_final.memory_usage(deep=True).sum() / (1024**2)
))

# %%
df_final.columns

# %% [markdown]
# Final processing

# %%
index_cols = [
    'underlying',
    'call_or_put',
    'expiration',
    'trade_date',
    'strike',
]

# %%
value_cols = [
    'close_price',
    'open_price',
    'high_price',
    'low_price',
    'im_vol',
    'next_day_base_price',
    'trade_volume',
    'trade_value',
    'open_interest_quantity',
]

# %%
# df_final = df_final[index_cols + value_cols].set_index(index_cols, inplace=False)
df_final = df_final[index_cols + value_cols]

# %%
print("Real final DataFrame memory usage: {:.2f} MB".format(
    df_final.memory_usage(deep=True).sum() / (1024**2)
))

# %%
df_final

# %%
df_final.to_pickle(OUTPUT_PATH / "krx_option_data_20220101-20241204.pkl")
df_final.to_csv(OUTPUT_PATH / "krx_option_data_20220101-20241204.csv", index=True)

# %%
df_final.to_parquet(OUTPUT_PATH / "krx_option_data_20220101-20241204.parquet", index=True, engine='pyarrow')

# %%
