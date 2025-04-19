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
# - 차단/중단시 그동안 한거라도 건지기 위해 계속 h5로 append, copy하여 저장. 

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9 이상

from typing import List, Tuple 
from tqdm import tqdm

import requests

# %%
from pathlib import Path

CWD_PATH = Path.cwd()
DATA_PATH = CWD_PATH / 'data'
BACKUP_PATH = CWD_PATH / 'backup'
OUTPUT_PATH = CWD_PATH / 'output'

# %%
CWD_PATH

# %%
## custom libs

from krx_config import (
API_URL, 
HEADERS, 
PAYLOAD_TEMPLATE, 
PAYLOAD_OPTION_HOME, 
HEADERS_OPTION_HOME, 
PAYLOAD_TEMPLATE2,
H5_SCHEMA,
)

# %% [markdown]
# ## Scraping code 

# %%
# main.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List
from tqdm import tqdm
import h5py
import os
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from zoneinfo import ZoneInfo  # Python 3.9 이상

# Import configurations
from krx_config import API_URL, HEADERS, PAYLOAD_TEMPLATE



# Constants for backup
BACKUP_INTERVAL = 10  # Backup after every 100 successful scrapes
H5_FILE = DATA_PATH / "krx_option_data.h5"

def generate_date_range(start_date: str, end_date: str) -> List[str]: 
    """
    종료일(end_date)부터 시작일(start_date)까지의 비주말(평일) 날짜를 YYYYMMDD 형식으로 반환합니다.
    날짜는 한국 서울 시간대 기준입니다.

    Args:
        start_date (str): 시작 날짜 ('YYYY-MM-DD' 형식).
        end_date (str): 종료 날짜 ('YYYY-MM-DD' 형식).

    Returns:
        List[str]: 내림차순으로 정렬된 비주말 날짜 리스트 ('YYYYMMDD' 형식).
    """
    tz = ZoneInfo('Asia/Seoul')
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz)
    delta = end - start
    date_list = []
    for i in range(delta.days + 1):
        current_date = end - timedelta(days=i)
        if current_date.weekday() < 5:  # 0-4는 월요일~금요일
            date_list.append(current_date.strftime("%Y%m%d"))
    return date_list

def setup_session() -> requests.Session:
    """
    Sets up a requests session with headers and retry strategy.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[400, 429, 500, 502, 503, 504],
        # method_whitelist=["POST"], # Deprecated
        allowed_methods=["POST"],
        backoff_factor=1000 # 1000s * 2^0 = 1000s = 16m 40s 기다리고, 
        # 1000s * 2^1 = 2000s = 33m 20s 기다리고, 1000s * 2^2 = 4000s = 1시간 6분 40초 기다림
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def fetch_option_data(session: requests.Session, trade_date: str) -> pd.DataFrame:
    """
    Fetches option data for a specific trade date.
    """
    payload = PAYLOAD_TEMPLATE.copy()
    payload["trdDd"] = trade_date

    response = session.post(API_URL, data=payload)
    response.raise_for_status()

    data = response.json()

    if "output" not in data:
        return pd.DataFrame()

    return pd.DataFrame(data["output"])

from pathlib import Path
import pandas as pd

def save_data_h5(df: pd.DataFrame, filename: str, schema: dict):
    # Enforce schema: cast object columns to string
    for col, (dtype, _) in schema.items():
        if dtype == 'object':
            df[col] = df[col].astype(str)

    # Prepare min_itemsize mapping
    min_itemsize = {col: max_len for col, (dtype, max_len) in schema.items() if dtype == 'object'}

    # Update min_itemsize if file exists
    if Path(filename).exists():
        with pd.HDFStore(filename, 'r') as store:
            if 'option_data' in store:
                storer = store.get_storer('option_data')
                for col in min_itemsize:
                    if col in storer.min_itemsize:
                        min_itemsize[col] = max(min_itemsize[col], storer.min_itemsize[col])

    # Append DataFrame
    with pd.HDFStore(filename, 'a') as store:
        store.append('option_data', df, format='table', data_columns=True, min_itemsize=min_itemsize)


def backup_h5_file(source: str, backup_path: Path):
    """
    Copies the HDF5 file to the backup directory with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = BACKUP_PATH / f"krx_option_data_backup_{timestamp}.h5"
    shutil.copy(source, backup_path)
    print(f"Backup created at {backup_path}")

def scrape_krx_option_data(start_date: str, end_date: str) -> None:
    """
    Scrapes KRX option data between start_date and end_date and saves to HDF5.
    Implements retry and backup strategies.
    """
    dates = generate_date_range(start_date, end_date)
    success_count = 0

    session = setup_session()
    
    for date in tqdm(dates, desc="Fetching data"):
        try:
            daily_data = fetch_option_data(session, date)
            if not daily_data.empty:
                daily_data['Trade_Date'] = pd.to_datetime(date, format='%Y%m%d')
                save_data_h5(daily_data, H5_FILE, H5_SCHEMA)
                success_count += 1
                
                if success_count % BACKUP_INTERVAL == 0:
                    backup_h5_file(H5_FILE, BACKUP_PATH)
            
            time.sleep(10)  # Delay to mimic human behavior
        
        
        except requests.HTTPError as http_err: # 이미 Retry에서 처리되므로, 별로 필요 없음
            print(f"HTTP error for date {date}: {http_err}")
            time.sleep(100)  # Wait before retrying
        except Exception as err:
            print(f"Error for date {date}: {err}")
            time.sleep(100)  # Wait before continuing
    
    # Final backup after completion
    backup_h5_file(H5_FILE, BACKUP_PATH)


# %%

# %%
start = '2024-12-02'
end = '2024-12-04'

# %%
generate_date_range(start, end)

# %%
# with requests.Session() as session:
#     session.headers.update(HEADERS)
    
#     payload = PAYLOAD_TEMPLATE.copy()
#     payload["trdDd"] = '20241202'

#     response = session.post(API_URL, data=payload)
#     response.raise_for_status()
#     data = response.json()

# %%
# dd = pd.DataFrame(data['output'])
# dd.head(1)

# %%
# dd.columns

# %%
# dd.info()

# %%
H5_SCHEMA

# %%
scrape_krx_option_data(start, end)

# %%
