# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: mfe311
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### SAS 코드 구현(ADJRET, B/M)

# %%
import numpy as np
import pandas as pd
import datetime

# %%
Compustat = pd.read_csv("./compustat_permno.csv")
np.sum(Compustat['pstkrv']<0)

# %%
#permno 열 Nan이 아닌 행만 선택
Compustat = Compustat[Compustat['permno'].notna()]

# datadate 열을 datetime 형식으로 변환 후 연도 추출 
Compustat['datadate'] = pd.to_datetime(Compustat['datadate'].astype(str), format='%Y%m%d')
Compustat['year'] = Compustat['datadate'].dt.year
Compustat = Compustat.loc[Compustat['permno'] != '0']

#pstkrv열 음수인 값 0으로 변환
Compustat['pstkrv'] = np.where(Compustat['pstkrv'] < 0, 0, Compustat['pstkrv'])

#BVPS: Book value of preferred stock
Compustat['BVPS'] = Compustat['pstkrv'].fillna(Compustat['pstkl']).fillna(Compustat['pstkl']).fillna(Compustat['pstk']).fillna(0)
# BE = SEQ + TXDB + ITCB + BVPS
Compustat['BE'] = Compustat['seq'] + Compustat['txdb'] + Compustat['itcb'].fillna(0) - Compustat['BVPS']
# BE<0이면 NaN처리
Compustat['BE'] = Compustat['BE'].where(Compustat['BE']>0)

compustat_BE = Compustat[['gvkey', 'datadate', 'year', 'BE', 'permno', 'permco' ]]
compustat_BE


# %%
compustat_BE = compustat_BE.sort_values(by=['gvkey', 'permno', 'year', 'datadate'])

compustat_BE = compustat_BE.drop_duplicates(subset=['gvkey', 'permno', 'year'], keep='last')

compustat_BE

# %%
crsp = pd.read_csv("./CRSP_M.csv")
crsp.columns = crsp.columns.str.lower()
crsp

# %%
crsp['Meq'] = np.where((crsp['altprc'].abs() > 0) & (crsp['shrout'] > 0), crsp['altprc'].abs() * crsp['shrout'] / 1000, np.nan)
crsp['date'] = pd.to_datetime(crsp['date'].astype(str), format='%Y%m%d')
crsp['year'] = crsp['date'].dt.year
print(crsp.shape)
crsp.head()

# %%
#if dlret missing, Use Shumway (1997)
crsp['dlret'] = np.where(crsp['dlstcd'].notna() & crsp['dlret'].isna() & ((crsp['dlstcd'].isin([500, 520, 574, 580, 584])) | 
                ((crsp['dlstcd'] >= 551) & (crsp['dlstcd'] <= 573))), -0.3, 
                np.where(crsp['dlstcd'].notna() & crsp['dlret'].isna(), -1, crsp['dlret']))

count = crsp[(crsp['dlret'].isna()) & (crsp['dlstcd'].notna())].shape[0]
print(count)
crsp

# %%
# Delisting이 있을 경우 조정 수익률 계산
crsp['retadj'] = np.where(crsp['dlstcd'].notna() & crsp['ret'].notna(),
                (1 + crsp['ret'])*(1 + crsp['dlret']) - 1,
                np.where(crsp['dlstcd'].notna() & crsp['ret'].isna(), crsp['dlret'], crsp['ret'])) 
# 초과 수익 계산
crsp['eretadj'] = crsp['retadj'] - crsp['rf']
crsp_M2 = crsp.sort_values(by=['Meq', 'permco', 'date'])
crsp_M2


# %%
crsp_M3 = crsp_M2.copy()

# 'date'와 'permco' 기준으로 그룹화하여 각 그룹 내의 개수 계산
crsp_M3['group_size'] = crsp_M3.groupby(['date', 'permco'])['Meq'].transform('size')

# 그룹 내 permco가 하나인 경우: ME 값을 Meq로 할당
crsp_M3['ME'] = np.where(crsp_M3['group_size'] == 1, crsp_M3['Meq'], np.nan)

# 그룹 내 permco가 두 개 이상인 경우: Meq의 누적합을 계산하여 ME에 할당
crsp_M3.loc[crsp_M3['group_size'] > 1, 'ME'] = crsp_M3.groupby(['date', 'permco'])['Meq'].cumsum()

# 'date', 'permco' 기준으로 그룹화 후 ME 값이 가장 큰 값이 마지막에 오도록 정렬
crsp_M3 = crsp_M3.sort_values(by=['date', 'permco', 'ME'], ascending=[True, True, False]) 

# 각 그룹의 첫 번째 행만 남기기 (가장 큰 ME 값을 가진 행이 첫 번째로 옴)
crsp_M3 = crsp_M3.drop_duplicates(subset=['date', 'permco'], keep='first').reset_index(drop=True)
crsp_M3 = crsp_M3.drop(columns=['group_size'])

crsp_M3


# %% [markdown]
# ##### SAS 6

# %%
crsp_M3 = crsp_M3.copy()

crsp_M3['date'] = pd.to_datetime(crsp_M3['date'], format='%Y-%m-%d')

# 6월 데이터 필터링
me_jun = crsp_M3[(crsp_M3['date'].dt.month == 6) & (crsp_M3['ME'].notna())].copy()
me_jun['t'] = crsp_M3['date'].dt.year  # 연도 추출
me_jun = me_jun[['permno', 't', 'ME']]  
me_jun.rename(columns={'ME': 'me_jun'}, inplace=True)

# 12월 데이터 필터링
me_last_Dec = crsp_M3[(crsp_M3['date'].dt.month == 12) & (crsp_M3['ME'].notna())].copy()
me_last_Dec['t'] = crsp_M3['date'].dt.year + 1  # 연도 추출 후 +1
me_last_Dec = me_last_Dec[['permno', 't', 'ME']]  
me_last_Dec.rename(columns={'ME': 'me_last_Dec'}, inplace=True)

# BE 데이터 필터링
be_last_year = compustat_BE[compustat_BE['BE'].notna()].copy()  # BE가 있는 데이터만 선택
be_last_year['t'] = be_last_year['year'] + 1  # 연도에 +1
be_last_year = be_last_year[['permno', 't', 'BE']]  # 필요한 열만 선택
be_last_year.rename(columns={'BE': 'be_last_year'}, inplace=True)

# permno, t 기준으로 정렬
me_jun = me_jun.sort_values(by=['permno', 't'])
me_last_Dec = me_last_Dec.sort_values(by=['permno', 't'])
be_last_year = be_last_year.sort_values(by=['permno', 't'])
me_jun, me_last_Dec, be_last_year

# %%
# %%time

"""
data ME_BM; 
    merge ME_Jun (in = a) BE_last_year (in = b) ME_last_Dec (in = c); 
    by permno t; 
    if a & b & c;
    BM = BE_last_year/ME_last_Dec; 
    keep permno t ME_Jun BM; 
run;
"""

# me_bm = pd.merge(me_jun, me_last_Dec, on=['permno', 't'], how='inner')
# me_bm = pd.merge(me_bm, be_last_year, on=['permno', 't'], how='inner')
# me_bm['bm'] = me_bm['be_last_year'] / me_bm['me_last_Dec']

# me_bm = me_bm[['permno', 't', 'me_jun', 'bm']]

# 모든 데이터프레임에서 permno를 int로 변환
me_jun['permno'] = me_jun['permno'].astype(int)
me_last_Dec['permno'] = me_last_Dec['permno'].astype(int)
be_last_year['permno'] = be_last_year['permno'].astype(int)

# BM 계산
me_bm = pd.merge(me_jun, me_last_Dec, on=['permno', 't'], how='inner')
me_bm = pd.merge(me_bm, be_last_year, on=['permno', 't'], how='inner')

# BM 값 계산 (Book-to-Market)
me_bm['bm'] = me_bm['be_last_year'] / me_bm['me_last_Dec']

# 필요한 열만 선택
me_bm = me_bm[['permno', 't', 'me_jun', 'bm']]

# 결과 출력
me_bm

# %%
# NOTE: There were 174169 observations read from the data set WORK.ME_BM.
me_bm.shape

# %%
# %%time

"""
* Match each permno's monthly return to the corresponding BM and ME;

data ret; 
    set CRSP_M3; 
    if month(date)>6 then t = year(date); 
    else t = year(date)-1; 
run; 
"""

crsp_M3['t'] = crsp_M3['date'].apply(lambda date: date.year if date.month > 6 else date.year-1)

# proc sort Mata = ret; by permno t date; run; 
crsp_M3 = crsp_M3.sort_values(by=['permno', 't', 'date'])

# %%
# The data set WORK.RET has 2892465 observations and 20 variables.
crsp_M3.shape

# %%
# %%time

"""
data ret_ME_BM; 
    merge ret (in = a) ME_BM (in = b); 
    by permno t; 
    if a; 
run;
"""

ret_me_bm = pd.merge(crsp_M3, me_bm, on=['permno', 't'], how='left')
ret_me_bm = ret_me_bm.drop_duplicates(subset=['permno', 'date', 'year'], keep='last')

# %%
# proc sort data = ret_ME_BM; by permno date; run; 
ret_me_bm = ret_me_bm.sort_values(by=['permno', 'date'])

# %%
# The data set WORK.RET_ME_BM has 2892465 observations and 22 variables.
ret_me_bm.shape

# %%
# %%time

"""
* Also add the mktcap and stock price from the previous month; 
data ret_ME_BM; 
    set ret_ME_BM;
    
    altprc_lag1 = lag1(altprc); 
    ME_lag1 = lag1(ME);

    permno_lag1 = lag1(permno); 
    date_lag1 = lag1(date);

    if (permno NE permno_lag1) or (intck('month',date_lag1,date)>1) then do; 
        altprc_lag1 = .; 
        ME_lag1 = .; 
            end; 
run;
"""

altprc_lag_df = pd.pivot_table(ret_me_bm, index='date', columns='permno', values='altprc').sort_index().shift(1)
altprc_lag = altprc_lag_df.reset_index().melt(id_vars='date', var_name='permno', value_name='altprc_lag1').dropna()

me_lag_df = pd.pivot_table(ret_me_bm, index='date', columns='permno', values='ME').sort_index().shift(1)
me_lag = me_lag_df.reset_index().melt(id_vars='date', var_name='permno', value_name='me_lag1').dropna()

ret_me_bm = pd.merge(ret_me_bm, altprc_lag, on=['date', 'permno'], how='left')
ret_me_bm = pd.merge(ret_me_bm, me_lag, on=['date', 'permno'], how='left')

# %%
# NOTE: The data set WORK.TT has 38518 observations and 26 variables.
ret_me_bm['altprc_lag1'].isna().sum()

# %%
# %%time

"""
* Exclude observations with missing values; 

data assignment1_data; 
    retain permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 ME_Jun BM; 
        set ret_ME_BM; 
        if nmiss(retadj, ME_lag1, ME_Jun, BM) = 0; 
        keep permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 ME_Jun BM;         
run;
"""

ret_me_bm = ret_me_bm[['permno', 'date', 'year', 'exchcd', 'siccd', 'retadj', 'eretadj', 'altprc_lag1', 'me_lag1', 'me_jun', 'bm']]
ret_me_bm = ret_me_bm.dropna(subset=['retadj', 'me_lag1', 'me_jun', 'bm'])

# %%
# The data set WORK.ASSIGNMENT1_DATA has 1983365 observations and 11
ret_me_bm

# %% [markdown]
# ##### Save sample data and summary stats

# %%
ret_me_bm.iloc[:25]

# %%
from pandas.tseries.offsets import MonthEnd

ret_me_bm['date'] = ret_me_bm['date'] + MonthEnd(0)

summ_dates = [str(x) + '-12-31' for x in range(1970, 2013, 1)]
summ_stats1 = pd.DataFrame(index=summ_dates, columns=['mean', 'std', 'min', 'max', "N of permno's"])

for date in summ_dates:
    tmp_eretadj = ret_me_bm[ret_me_bm['date']==date]['eretadj']
    tmp_permno = ret_me_bm[ret_me_bm['date']==date]['permno']
    summ_stats1.loc[date] = [tmp_eretadj.mean(), tmp_eretadj.std(), tmp_eretadj.min(), tmp_eretadj.max(), len(tmp_permno.unique())]

summ_stats1

# %%
specific_years = ['1970-12-31', '1980-12-31', '1990-12-31', '2000-12-31', '2010-12-31']

summ_stats_answer = summ_stats1.loc[specific_years]

summ_stats_answer

# %%
import logging
from fpdf import FPDF

# Step 1: 로그 파일 작성
log_filename = 'program.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)

logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.debug("This is a debug message")

# Step 2: 로그 파일을 PDF로 변환
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(200, 10, 'Log Report', ln=True, align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def log_to_pdf(log_file, output_pdf):
    pdf = PDF()
    pdf.add_page()

    pdf.set_font('Arial', '', 12)

    # 로그 파일 내용을 읽어서 PDF에 추가
    with open(log_file, 'r') as file:
        for line in file:
            pdf.cell(200, 10, line, ln=True)

    # PDF 파일 저장
    pdf.output(output_pdf)

# PDF 파일 생성
log_to_pdf(log_filename, 'log_report.pdf')

print("PDF 파일이 생성되었습니다.")

