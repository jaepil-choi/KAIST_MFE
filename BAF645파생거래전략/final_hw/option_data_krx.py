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

from krx_config import API_URL, HEADERS, PAYLOAD_TEMPLATE, PAYLOAD_OPTION_HOME, HEADERS_OPTION_HOME, PAYLOAD_TEMPLATE2


# %% [markdown]
# ## API scraping code

# %% [markdown]
# ### Prompt 1
#
# I need to write code to scrape data from KRX(한국거래소) website. The website uses POST to get daily snapshot of all the options traded on that day. 
#
# Below is the information for its endpint (getJsonData.cmd) and what are in the headers and what the payload is and what the response looks like. 
#
# Write Python code that takes start date and end date as an input and iteratively get the option price data from the api. 
#
# You should make the requests to look like human by setting web browser user-agent and other stuff that can be detected by site admin. 
#
# Although, you don't have to use dynamic scraping like selenium because as you can see, we're communicating with an API in restful way, with POST method. 
#
# When you write the code, make the code modular and keep the function document as simple as possible. Never make verbose docstring that can bloat the code with unnecessary details. 
#
# Below are the information that you need to write the correct API call scraper. 

# %%
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

def fetch_option_data(session: requests.Session, trade_date: str) -> pd.DataFrame:
    """
    Fetches option data for a specific trade date.

    Args:
        session (requests.Session): The requests session with headers set.
        trade_date (str): Trade date in 'YYYYMMDD' format.

    Returns:
        pd.DataFrame: DataFrame containing option data for the trade date.
    """
    payload = PAYLOAD_TEMPLATE.copy()
    payload["trdDd"] = trade_date

    response = session.post(API_URL, data=payload)
    response.raise_for_status()

    data = response.json()

    if "output" not in data:
        return pd.DataFrame()

    return pd.DataFrame(data["output"])

def scrape_krx_option_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scrapes KRX option data between start_date and end_date.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Combined DataFrame containing option data for all dates.
    """
    dates = generate_date_range(start_date, end_date)
    all_data = []

    with requests.Session() as session:
        session.headers.update(HEADERS)
        for date in tqdm(dates, desc="Fetching data"):
            try:
                daily_data = fetch_option_data(session, date)
                if not daily_data.empty:
                    daily_data['Trade_Date'] = datetime.strptime(date, "%Y%m%d").date()
                    all_data.append(daily_data)
                time.sleep(1)  # Delay to mimic human behavior
            except requests.HTTPError as http_err:
                print(f"HTTP error for date {date}: {http_err}")
            except Exception as err:
                print(f"Error for date {date}: {err}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def save_to_excel(df: pd.DataFrame, filename: str):
    """
    Saves the DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Filename for the Excel file.
    """
    df.to_excel(filename, index=False)
    print(f"Data saved to {filename}")


# %%
sample_date = '20241204'

# %%
with requests.Session() as session:
    session.headers.update(HEADERS)
    payload = PAYLOAD_TEMPLATE.copy()
    payload["trdDd"] = sample_date

    response = session.post(API_URL, data=payload)

# %%
response.content

# %%
dd = response.json()

# %% [markdown]
# 거래소 주식 옵션 전체로 조회하는거 막혔음. 나만 막힌게 아니고 전부. 
#
# 하지만 개별 underlying 종목을 설정해서 조회할 경우 불러올 수 있음. 딱 1년치만 불러오자. 
#

# %% [markdown]
# ### 개별 주식 옵션 데이터 불러오기 테스트

# %% [markdown]
# 또 막힐 수 있으니까 일단 삼성만 1년치 불러오고 나머지는 다시 하는걸로? 

# %%
# 거래소 주식 옵션 종목들 조회 

with requests.Session() as session:
    session.headers.update(HEADERS_OPTION_HOME)
    payload = PAYLOAD_OPTION_HOME.copy()
    response = session.post(API_URL, data=payload)

# %%
available_underlyings = pd.DataFrame(response.json()['output'])

# %%
available_underlyings.head()

# %%
len(available_underlyings)

# %%
avail_under_code_list = available_underlyings['value'].tolist()[1:]

# %%
samsung_elec = 'KRDRVOPS11'

# %%
# 개별 종목 call 되나 확인 

with requests.Session() as session:
    session.headers.update(HEADERS)
    payload = PAYLOAD_TEMPLATE2.copy()
    
    payload['trdDd'] = sample_date
    payload['trdDdBox1'] = sample_date
    payload['trdDdBox2'] = sample_date
    
    payload["ulyId"] = samsung_elec

    response = session.post(API_URL, data=payload)

# %%
dd1 = pd.DataFrame(response.json()['output'])
len(dd1)

# %%
sample_date

# %%
# 개별 종목 call, 기간 줘도 되나 확인. 

with requests.Session() as session:
    session.headers.update(HEADERS)
    payload = PAYLOAD_TEMPLATE2.copy()
    
    payload['trdDd'] = sample_date
    payload['trdDdBox1'] = '20241202'
    payload['trdDdBox2'] = sample_date
    
    payload["ulyId"] = samsung_elec

    response = session.post(API_URL, data=payload)

# %%
dd2 = pd.DataFrame(response.json()['output'])

# %%
len(dd2)

# %%
# trdDdBox1, trdDdBox2 는 그냥 deprecated된 legacy parameter로 보인다. 

# %%
# 개별 종목 call, 기간 줘도 되나 확인. 

with requests.Session() as session:
    session.headers.update(HEADERS)
    payload = PAYLOAD_TEMPLATE2.copy()
    
    payload['trdDd'] = "" # 빠지면 안됨.
    payload['trdDdBox1'] = '20241202'
    payload['trdDdBox2'] = sample_date
    
    payload["ulyId"] = samsung_elec

    response = session.post(API_URL, data=payload)

# %%
dd3 = pd.DataFrame(response.json()['output'])

# %%
len(dd3)


# %% [markdown]
# ### 개별 주식 옵션 데이터 불러와서 합치기

# %% [markdown]
# 일단 테스트로 삼성만

# %%
def fetch_option_data1(session: requests.Session, uly_code: str, trade_date: str,) -> pd.DataFrame:
    payload = PAYLOAD_TEMPLATE.copy()
    
    payload["trdDd"] = trade_date
    payload['trdDdBox1'] = trade_date
    payload['trdDdBox2'] = trade_date

    payload["ulyId"] = uly_code

    response = session.post(API_URL, data=payload)
    response.raise_for_status()

    data = response.json()

    if "output" not in data:
        return pd.DataFrame()

    return pd.DataFrame(data["output"])

def scrape_krx_option_data1(start_date: str, end_date: str, ulycode_list: list) -> pd.DataFrame:
    dates = generate_date_range(start_date, end_date)
    
    all_data = []
    for uly_code in ulycode_list:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            for date in tqdm(dates, desc="Fetching data"):
                try:
                    daily_data = fetch_option_data(session, uly_code, date)
                    if not daily_data.empty:
                        daily_data['Trade_Date'] = datetime.strptime(date, "%Y%m%d").date()
                        daily_data['Uly_Code'] = uly_code
                        all_data.append(daily_data)
                    time.sleep(3)  # Delay to mimic human behavior
                except requests.HTTPError as http_err:
                    print(f"HTTP error for date {date}: {http_err}")
                except Exception as err:
                    print(f"Error for date {date}: {err}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()
        


# %%
start = '2024-01-01'
end = '2024-12-04'

samsung_list = [samsung_elec]

# %%
samsung_df = scrape_krx_option_data1(start, end, samsung_list)

# %%
samsung_df.to_pickle(DATA_PATH / 'samsung_option.pkl')

# %%
len(samsung_df)

# %%
samsung_df = pd.read_pickle(DATA_PATH / 'samsung_option.pkl')

# %%
samsung_df.columns

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
parsed_data = samsung_df['ISU_NM'].apply(parse_option_name)

# %%
parsed_df = pd.DataFrame(parsed_data.tolist())

# %%
samsung_df = pd.concat([samsung_df, parsed_df], axis=1)

# %%
column_to_name = {
    'ISU_CD': 'option_sid(full)',  # 종목코드(full)
    'ISU_SRT_CD': 'option_sid(short)',  # 종목코드(short)
    'ISU_NM': 'option_name',  # 종목명
    'TDD_CLSPRC': 'close_price',  # 종가
    'FLUC_TP_CD': 'up_or_down',  # 등락구분
    'CMPPREVDD_PRC': 'price_change',  # 대비
    'TDD_OPNPRC': 'open_price',  # 시가
    'TDD_HGPRC': 'high_price',  # 고가
    'TDD_LWPRC': 'low_price',  # 저가
    'IMP_VOLT': 'im_vol',  # 내재변동성
    'NXTDD_BAS_PRC': 'next_day_base_price',  # 익일기준가
    'ACC_TRDVOL': 'trade_volume',  # 거래량
    'ACC_TRDVAL': 'trade_value',  # 거래대금
    'ACC_OPNINT_QTY': 'open_interest_quantity',  # 미결제약정수량
    'SECUGRP_ID': 'security_type',  # 증권유형
    'Trade_Date': 'trade_date',  # 거래일자
    'Uly_Code': 'underlying_code',  # 기초자산코드
}

# %%
samsung_df.rename(columns=column_to_name, inplace=True)

# %%
float_cols = [
    'close_price', 'price_change', 'open_price', 'high_price', 'low_price',
    'im_vol', 'next_day_base_price', 'trade_volume', 'trade_value',
    'open_interest_quantity', 
    'expiration', 
    'up_or_down',
    # 'strike', 
    # 'multiplier',
]

# %%
for float_col in float_cols:
    print(f'converting {float_col} to numeric')
    samsung_df[float_col] = samsung_df[float_col].str.replace('-', '')
    samsung_df[float_col] = samsung_df[float_col].str.replace(',', '')
    samsung_df[float_col] = pd.to_numeric(samsung_df[float_col], errors='raise')

# %%
samsung_df['trade_date'] = pd.to_datetime(samsung_df['trade_date'])

# %%
samsung_df.info()

# %% [markdown]
# 데이터를 처리하기 쉬운 형태로 변환

# %%
samsung_df.columns

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
samsung_final_df = samsung_df[index_cols + value_cols].copy()
samsung_final_df.set_index(index_cols, inplace=True)


# %%
samsung_final_df

# %%
samsung_final_df.to_pickle(DATA_PATH / 'samsung_option_final.pkl')

# %%
START_DATE = '2021-01-01'
END_DATE = '2024-12-05'

# %% [markdown]
#
# ### New Prompt
#
# Implement retry strategy in this code. 
# Retry strategy should:
# - when there's bad requests like 400, step back and wait and retry. 
#
# You should suggest me additional measure to avoid getting blocked. Is there a free VPN service that I can use to hide my identity? 
#
# Also, to avoid losing all the data when I get blocked, write the code to:
#
# - start from end date to start date
# - for each date scraped, save it in the data by appending the data (not overwriting). You should use h5py to save it in h5 format to append the data. 
# - periodically (per 100 scrape) copy that h5 file to somewhere else for me to safely use it without racing condition. ( I will copy the copy of the h5 and use it at somewhere else)

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
BACKUP_INTERVAL = 100  # Backup after every 100 successful scrapes
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
        total=5,
        status_forcelist=[400, 429, 500, 502, 503, 504],
        method_whitelist=["POST"],
        backoff_factor=1
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

def save_data_h5(df: pd.DataFrame, filename: str):
    """
    Appends DataFrame to an HDF5 file.
    """
    with pd.HDFStore(filename, mode='a') as store:
        store.append('option_data', df, format='table', data_columns=True)

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
                daily_data['Trade_Date'] = datetime.strptime(date, "%Y%m%d").date()
                save_data_h5(daily_data, H5_FILE)
                success_count += 1
                
                if success_count % BACKUP_INTERVAL == 0:
                    backup_h5_file(H5_FILE, BACKUP_PATH)
            
            time.sleep(1)  # Delay to mimic human behavior
        except requests.HTTPError as http_err:
            print(f"HTTP error for date {date}: {http_err}")
            time.sleep(5)  # Wait before retrying
        except Exception as err:
            print(f"Error for date {date}: {err}")
            time.sleep(5)  # Wait before continuing
    
    # Final backup after completion
    backup_h5_file(H5_FILE, BACKUP_PATH)



# %%

def main():
    # Example usage
    START_DATE = "2024-12-01"
    END_DATE = "2024-12-06"
    
    print(f"Scraping KRX option data from {START_DATE} to {END_DATE}...")
    scrape_krx_option_data(START_DATE, END_DATE)
    print("Scraping completed.")

if __name__ == "__main__":
    main()


# %%
