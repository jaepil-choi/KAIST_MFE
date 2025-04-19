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
# # 자계추 hw1: Create Dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %% [markdown]
# ## Load Datasets

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
CRSP_M_df = pd.read_csv(DATA_DIR / 'CRSP_M.csv')
compustat_df = pd.read_csv(DATA_DIR / 'compustat_permno.csv') 
sample_df = pd.read_csv(DATA_DIR / 'assignment1_sample_data.csv')

# %% [markdown]
# ## SAS3
#
# Construct BE Data
#
# Compustat 데이터 사용
#
# fiscal year 별로 되어있음. 

# %%
# * Calculate BE; 
# data BE; 
#  set compustat_permno (where = (missing(permno) = 0)); 

compustat_df.dropna(subset=['permno'], inplace=True)

# %%
# NOTE: The data set WORK.BE has 264450 observations and 6 variables.

compustat_df.shape 

# %%
#  year = year(datadate); 

compustat_df['year'] = compustat_df['datadate'] // 10000

# %%
# if missing(ITCB) then ITCB = 0; * investment tax credit; 

compustat_df['itcb'] = compustat_df['itcb'].fillna(0)

# %%
# SAS 코드에는 없는 내용. 하지만 확인해보면 preferred stock redemption value가 음수인 경우가 있음.
# 일단 원본 코드에 없으므로 무시하고 넘어감.

# compustat_df.loc[
#     compustat_df['pstkrv'] < 0,
#     'pstkrv'
#     ] = 0

# %%
# BVPS = PSTKRV; * bool value of preferred stock (BVPS) = preferred stock 의 redemption value로 일단 놓고; 
#  if missing(BVPS) then BVPS = PSTKL; * 없으면 preferred stock 의 liquidating value; 
#  if missing(BVPS) then BVPS = PSTK; * 또 없으면 preferred stock의 par value; 
#  if missing(BVPS) then BVPS = 0; * 다 없으면 0;

compustat_df['bvps'] = compustat_df['pstkrv'].fillna(compustat_df['pstkl']) \
                                               .fillna(compustat_df['pstk']) \
                                                .fillna(0)

# %%
# BE = SEQ + TXDB + ITCB - BVPS; * If SEQ or TXDB is missing, BE, too, will be missing; 
#  if BE<=0 then BE = .; * If BE<0, the value of BE is taken to be missing;  

compustat_df['be'] = compustat_df['seq'] \
                    + compustat_df['txdb'] \
                    + compustat_df['itcb'] \
                    - compustat_df['bvps']

compustat_df.loc[compustat_df['be'] <= 0, 'be'] = np.nan

# %%
# * In some cases, firms change the month in which their fiscal year ends,  
# * resulting in two entries in the Compustat database for the same calendar year y.  
# * In such cases, data from the latest in the given calendar year y are used.;  
# proc sort data = BE; by gvkey permno year datadate; run; 
# data BE; 
#  set BE; 
#  by gvkey permno year datadate; 
#  if last.year; 
# run; 
# proc sort data = BE nodupkey; by gvkey permno year datadate; run;

compustat_df.sort_values(['gvkey', 'permno', 'year', 'datadate'], inplace=True)

compustat_df['idx'] = compustat_df.index

groupbylast = compustat_df.groupby(['gvkey', 'permno', 'year']).last().reset_index()
dropduplicates = compustat_df.drop_duplicates(subset=['gvkey', 'permno', 'year'], keep='last').reset_index()

# compustat_df = compustat_df.groupby(['gvkey', 'permno', 'year', ]).last().reset_index()
compustat_df.drop_duplicates(subset=['gvkey', 'permno', 'year'], inplace=True, keep='last')

# 이 두가지의 결과가 다르다. 당장 row 수는 같게 나오는데 나중에 결과가 다르다. 

# %%
groupbylast['idx']

# %%
dropduplicates['idx']

# %%
(groupbylast['idx'] == dropduplicates['idx']).all()

# %% [markdown]
# ??? 심지어 원래의 index를 비교해도 같다? 그럼 정말 둘이 같다는 소리인데... 

# %%
compustat_df

# %%
# NOTE: The data set WORK.BE has 263854 observations and 6 variables.

compustat_df.shape

# %% [markdown]
# ## SAS5
#
# Construct ME and return data
#
# CRSP 데이터 사용
#
# Monthly data로 되어있음 

# %%
# * SAS 4: Merge CRSP stock and event file and add risk-free rate *******************; 

# # %let filter=%str(shrcd in (10,11) and exchcd in (1,31,2,32,3,33)); 

# # %crspmerge(s = m, outset = CRSP, 
# start = &start_date, end = &end_date, 
# sfvars = permco ret vol shrout prc altprc, 
# sevars = siccd shrcd exchcd dlstcd dlret, 
# filters=&filter);   


# filters # 사실 필터는 이미 적용되어 있음
filter_common_stocks = [10, 11] # SHRCD
filter_exchange = [ # EXCHCD
    1, 31, # NYSE
    2, 32, # AMEX
    3, 33, # NASDAQ
]

CRSP_M_df = CRSP_M_df[ CRSP_M_df['SHRCD'].isin(filter_common_stocks) ]
CRSP_M_df = CRSP_M_df[ CRSP_M_df['EXCHCD'].isin(filter_exchange) ]

# %%
CRSP_M_df

# %%
CRSP_M_df.shape # NOTE: The data set WORK.CRSP has 2921193 observations and 13 variables.

# %%
# * SAS 5: Construct ME and return data *************************************; 

# * Calculate excess return adjusted for delising; 
# data CRSP_M2; 
#  set CRSP_M; 
#  year = year(date); *** date, 매달 마지막 거래일 값이 들어가있다. 거기서 연도를 뽑아냄. ;

CRSP_M_df['YEAR'] = CRSP_M_df['DATE'] // 10000

# %%
# * calculate market capitalization; 
#  if abs(altprc)>0 and shrout>0 then Meq = abs(altprc)*shrout/1000;  
# *** 절대값이 0보다 크면, market equity 값을 계산을 해라. (Meq) 만족 안하면 missing으로 처리.;
# ** ALTPRC: last non-missing price over all days in the month인데,  ;
# ** CRSP는 거래가 없을 경우 last bid와 last ask의 평균을 - 로 report함. ;
# ** 즉, 가격이 -인 것이 오류가 아니라는 소리임. 날려버리면 안됨. ; 
# ** 진짜 데이터가 available하지 않은 경우는 0이나 missing으로 표시해줌. ;

CRSP_M_df['MEQ'] = np.nan
CRSP_M_df.loc[ 
    (CRSP_M_df['ALTPRC'].abs() > 0) & (CRSP_M_df['SHROUT'] > 0) , 
    'MEQ'
    ] = CRSP_M_df['ALTPRC'].abs() * CRSP_M_df['SHROUT'] / 1000

# %%
# * if dlret is missing, follow Shumway (1997) to determine dlret; 
#  if missing(dlstcd) = 0 and missing(dlret) =1 then do; *** delisting code(사유)는 있고 delisting return이 missing이면, 아래와 같이 처리.;
#   if dlstcd in (500, 520, 574, 580, 584) or (dlstcd>=551 and dlstcd<=573)  
#    then dlret = -0.3; *** 위 사유들에 대해선 적당히 -0.3으로 처리;
#   else dlret = -1; *** 그 외에는 -1 (-100%)로 처리;
#  end; 

dlstcd_filter = [500, 520, 574, 580, 584] + list(range(551, 573+1))
CRSP_M_df.loc[
    (CRSP_M_df['DLSTCD'].isin(dlstcd_filter)) & 
    (CRSP_M_df['DLRET'].isna()), 
    'DLRET'
    ] = -0.3

CRSP_M_df.loc[
    (~CRSP_M_df['DLSTCD'].isin(dlstcd_filter)) & \
    CRSP_M_df['DLSTCD'].notna() & \
    (CRSP_M_df['DLRET'].isna()), \
    'DLRET'
    ] = CRSP_M_df['DLRET'].fillna(-1)

# %%
#  * calculate return adjusted for delisting; 
#  if missing(dlstcd) = 0 then do; 
#   if missing(ret) = 0 then retadj = (1+ret)*(1+dlret)-1; 
#   else retadj = dlret; 
#  end; 
#  else retadj = ret; 
#  eretadj = retadj - rf; *** 이게 최종적으로 사용하는 return. risk-free rate를 빼준 것. ;
# run;
# proc sort data = CRSP_M2; by date permco Meq; run; 

# dlstcd가 있을 때 
CRSP_M_df.loc[ # delisting 날의 ret가 있으면 (1+ret)*(1+dlret)-1
    CRSP_M_df['DLSTCD'].notna() & CRSP_M_df['RET'].notna(),
    'RETADJ'
    ] = (1 + CRSP_M_df['RET']) * (1 + CRSP_M_df['DLRET']) - 1

CRSP_M_df.loc[ # delisting 날의 ret가 없으면 dlret
    CRSP_M_df['DLSTCD'].notna() & CRSP_M_df['RET'].isna(),
    'RETADJ'
    ] = CRSP_M_df['DLRET']

# dlstcd가 없을 때
CRSP_M_df.loc[
    CRSP_M_df['DLSTCD'].isna(),
    'RETADJ'
    ] = CRSP_M_df['RET']

CRSP_M_df['ERETADJ'] = CRSP_M_df['RETADJ'] - CRSP_M_df['rf']
CRSP_M_df.sort_values(['DATE', 'PERMCO', 'MEQ'], inplace=True)

# %%
CRSP_M_df

# %%
CRSP_M_df.shape # NOTE: The data set WORK.CRSP_M2 has 2921193 observations and 18 variables.

# %%
# * There are cases when the same firm (permco) has two or more securities (permno)  
# at the same date.  
# * We aggregate all ME for a given permco and date,       
# * and assign this aggregated ME to the permno with the largest ME; 
# data CRSP_M3; 
#  set CRSP_M2; 
#  by date permco Meq; 
#  retain ME;  
#  if first.permco and last.permco then do; 
#   ME = Meq; *** Meq는 각 share class의 Market equity, ME는 각 회사(permco)의 Market equity의 합. ;
#   output; 
#  end; 


# nan 포함하여 groupby 하여 개수 확인
CRSP_M_df['count_permno'] = CRSP_M_df.groupby(['DATE', 'PERMCO'])['PERMNO'].transform('size')

# ME를 일단 nan으로 초기화
CRSP_M_df['ME'] = np.nan

# first.permco and last.permco 즉 1개인 경우
CRSP_M_df.loc[CRSP_M_df['count_permno'] == 1, 'ME'] = CRSP_M_df['MEQ']


# %%
#  else do; 
#   if first.permco then ME = Meq; *** ME는 Meq의 누적합. ;
#   else ME = sum(Meq, ME); *** 누적합하는 컬럼 ME를 만들었으니, 누적합하는데 사용한 그 이전의 row들은 다 날림. ;
#   If last.permco then output; 
#  end; 

# 2개 이상인 경우, MEQ를 합
# CRSP_M_df.loc[CRSP_M_df['count_permno'] > 1, 'ME'] = CRSP_M_df.groupby(['DATE', 'PERMCO'])['MEQ'].transform('sum')
CRSP_M_df.loc[CRSP_M_df['count_permno'] > 1, 'ME'] = CRSP_M_df.groupby(['DATE', 'PERMCO'])['MEQ'].transform('cumsum')

# 가장 큰 ME를 가진 PERMNO를 선택
CRSP_M_df = CRSP_M_df.sort_values(['DATE', 'PERMCO', 'ME'], ascending=[True, True, False])
## MEQ가 아니라 ME로 변경 (그래도 차이는 없음)

# CRSP_M_df = CRSP_M_df.groupby(['DATE', 'PERMCO']).last().reset_index()
CRSP_M_df = CRSP_M_df.drop_duplicates(subset=['DATE', 'PERMCO'], keep='first').reset_index(drop=True)

# %% [markdown]
# 여기서도 groupby last 대신 drop duplicates keep last 를 사용하니 또 달라짐. 
#
# 확실히 두 operation은 다름

# %%


# 임시 컬럼 제거
CRSP_M_df.drop(columns=['count_permno'], inplace=True)

# %%
CRSP_M_df

# %%
CRSP_M_df.shape # NOTE: The data set WORK.CRSP_M3 has 2892465 observations and 19 variables.

# %% [markdown]
# ## SAS6
#
# Merge BE and ME with return data

# %%
# proc sort data = crsp_m3 nodupkey; by permno date; run; *** duplicates 있는지 확인하려고 매번 체크하는 부분; 

# * SAS 6: Merge BE and ME with Return Data *************************************; 

# * Calculate BM from the previous year and June ME from this year for each permno; 
# data ME_Jun; 
#  set CRSP_M3 (where = (month(date) = 6 & missing(ME) = 0)); 
#  t = year(date); ** 1999 Dec ME --> t=2000 다음 해에 trading signal로 쓰도록. ; 
#  ME_Jun = ME; 
#  keep permno t ME_Jun; ** 이것들만 남기고 나머지는 버려라. ;
# run; 
CRSP_ME_JUN_df = CRSP_M_df.copy()
CRSP_ME_JUN_df['T'] = CRSP_ME_JUN_df['DATE'] // 10000
CRSP_ME_JUN_df.loc[
    (CRSP_ME_JUN_df['DATE'] % 10000 // 100 == 6 ) & \
    CRSP_ME_JUN_df['ME'].notna(), 
    'ME_JUN'
] = CRSP_ME_JUN_df['ME']

CRSP_ME_JUN_df = CRSP_ME_JUN_df[['PERMNO', 'T', 'ME_JUN',]]
CRSP_ME_JUN_df.sort_values(['PERMNO', 'T'], inplace=True)

# %%
CRSP_ME_JUN_df.dropna(subset=['ME_JUN'], inplace=True)

# %%
CRSP_ME_JUN_df

# %%
CRSP_ME_JUN_df.shape # NOTE: There were 239521 observations read from the data set WORK.ME_JUN.

# %%
# data ME_last_Dec; 
#  set CRSP_M3 (where = (month(date) = 12 & missing(ME) = 0)); 
#  t = year(date)+1; ** 마찬가지로. +1 해준다. ;  
#  ME_last_Dec = ME; 
#  keep permno t ME_last_Dec; 
# run; 
# proc sort data = ME_last_Dec; by permno t; run; 

CRSP_ME_LAST_DEC_df = CRSP_M_df.copy()
CRSP_ME_LAST_DEC_df['T'] = CRSP_ME_LAST_DEC_df['DATE'] // 10000 + 1
CRSP_ME_LAST_DEC_df.loc[
    (CRSP_ME_LAST_DEC_df['DATE'] % 10000 // 100 == 12 ) & \
    CRSP_ME_LAST_DEC_df['ME'].notna(), 
    'ME_LAST_DEC'
] = CRSP_ME_LAST_DEC_df['ME']

CRSP_ME_LAST_DEC_df = CRSP_ME_LAST_DEC_df[['PERMNO', 'T', 'ME_LAST_DEC',]]
CRSP_ME_LAST_DEC_df.sort_values(['PERMNO', 'T'], inplace=True)

# %%
CRSP_ME_LAST_DEC_df.dropna(subset=['ME_LAST_DEC'], inplace=True)

# %%
CRSP_ME_LAST_DEC_df

# %%
CRSP_ME_LAST_DEC_df.shape # NOTE: There were 242805 observations read from the data set WORK.ME_LAST_DEC.

# %%
# data BE_last_year; 
#  set BE (where = (missing(BE) = 0)); 
#  t = year+1; 
#  BE_last_year = BE; 
#  keep permno t BE_last_year; 
# run; 
# proc sort data = BE_last_year; by permno t; run;

compustat_be_last_year_df = compustat_df.copy()
compustat_be_last_year_df['t'] = compustat_be_last_year_df['year'] + 1
# compustat_be_last_year_df.dropna(subset=['be'], inplace=True)
compustat_be_last_year_df.loc[
    compustat_be_last_year_df['be'].notna(),
    'be_last_year'
    ] = compustat_be_last_year_df['be']

compustat_be_last_year_df = compustat_be_last_year_df[['permno', 't', 'be_last_year',]]
compustat_be_last_year_df.sort_values(['permno', 't'], inplace=True)
compustat_be_last_year_df.dropna(subset=['be_last_year'], inplace=True)

# %%
compustat_be_last_year_df

# %%
compustat_be_last_year_df.shape # NOTE: There were 213229 observations read from the data set WORK.BE_LAST_YEAR.

# %%
# data ME_BM; 
#  merge ME_Jun (in = a) BE_last_year (in = b) ME_last_Dec (in = c); ** permno t ME_Jun ME_last_Dec BE_last_year ;
#  ** ME_Jun은 올해 6월, ME_last_Dec, BE_last_year은 작년 ;
#  by permno t; 
#  if a & b & c; 

ME_BM_df = pd.merge(
    left=CRSP_ME_JUN_df, 
    right=CRSP_ME_LAST_DEC_df,
    how='inner',
    on=['PERMNO', 'T'],
)

ME_BM_df = pd.merge(
    left=ME_BM_df,
    right=compustat_be_last_year_df,
    how='inner',
    left_on=['PERMNO', 'T'],
    right_on=['permno', 't'],
)

# %%
#  BM = BE_last_year/ME_last_Dec; 
#  keep permno t ME_Jun BM; 
# run;

ME_BM_df['BM'] = ME_BM_df['be_last_year'] / ME_BM_df['ME_LAST_DEC']
ME_BM_df = ME_BM_df[['PERMNO', 'T', 'ME_JUN', 'BM']]

# %%
ME_BM_df

# %%
ME_BM_df.shape # NOTE: The data set WORK.ME_BM has 174169 observations and 4 variables.

# %% [markdown]
# 아래부턴 교수님이 코드를 주셨으니 일단 그대로 씀 
#

# %% [markdown]
# ```python
# """
# * Match each permno's monthly return to the corresponding BM and ME;
#
# data ret; 
#     set CRSP_M3; 
#     if month(date)>6 then t = year(date); 
#     else t = year(date)-1; 
# run; 
# """
#
# crsp_m3['t'] = crsp_m3['date'].apply(lambda date: date.year if date.month > 6 else date.year-1)
#
# # proc sort data = ret; by permno t date; run; 
# crsp_m3 = crsp_m3.sort_values(by=['permno', 't', 'date'])
# ```

# %%
RET_df = CRSP_M_df.copy()

RET_df['pddate'] = pd.to_datetime(RET_df['DATE'], format='%Y%m%d')
RET_df['T'] = RET_df['pddate'].apply(lambda date: date.year if date.month > 6 else date.year - 1)
# RET_df.loc[
#     RET_df['pddate'].dt.month > 6,
#     'T'
#     ] = RET_df['pddate'].dt.year
# RET_df.loc[
#     RET_df['pddate'].dt.month <= 6,
#     'T'
#     ] = RET_df['pddate'].dt.year - 1


# %%
RET_df.sort_values(['PERMNO', 'T', 'pddate'], inplace=True)

# %%
RET_df.shape # NOTE: The data set WORK.RET has 2892465 observations and 20 variables.

# %% [markdown]
# ```python
# %%time
#
# """
# data ret_ME_BM; 
#     merge ret (in = a) ME_BM (in = b); 
#     by permno t; 
#     if a; 
# run;
# """
#
# ret_me_bm = pd.merge(crsp_m3, me_bm, on=['permno', 't'], how='left')
# ret_me_bm = ret_me_bm.drop_duplicates(subset=['permno', 'date', 'year'], keep='last')
# ```

# %%
RET_ME_BM_df = pd.merge(
    left=RET_df,
    right=ME_BM_df,
    how='left',
    on=['PERMNO', 'T'],
)

# %%
RET_ME_BM_df.drop_duplicates(
    subset=['PERMNO', 'pddate', 'YEAR'], 
    inplace=True, 
    keep='last',
    )

# %%
RET_ME_BM_df.sort_values(['PERMNO', 'pddate'], inplace=True)

# %%
RET_ME_BM_df.shape

# %% [markdown]
# ```python
# %%time
#
# """
# * Also add the mktcap and stock price from the previous month; 
# data ret_ME_BM; 
#     set ret_ME_BM;
#     
#     altprc_lag1 = lag1(altprc); 
#     ME_lag1 = lag1(ME);
#
#     permno_lag1 = lag1(permno); 
#     date_lag1 = lag1(date);
#
#     if (permno NE permno_lag1) or (intck('month',date_lag1,date)>1) then do; 
#         altprc_lag1 = .; 
#         ME_lag1 = .; 
#             end; 
# run;
# """
#
# altprc_lag_df = pd.pivot_table(ret_me_bm, index='date', columns='permno', values='altprc').sort_index().shift(1)
# altprc_lag = altprc_lag_df.reset_index().melt(id_vars='date', var_name='permno', value_name='altprc_lag1').dropna()
#
# me_lag_df = pd.pivot_table(ret_me_bm, index='date', columns='permno', values='me').sort_index().shift(1)
# me_lag = me_lag_df.reset_index().melt(id_vars='date', var_name='permno', value_name='me_lag1').dropna()
#
# ret_me_bm = pd.merge(ret_me_bm, altprc_lag, on=['date', 'permno'], how='left')
# ret_me_bm = pd.merge(ret_me_bm, me_lag, on=['date', 'permno'], how='left')
# ```

# %%
altprc_lag_df = pd.pivot_table(
    RET_ME_BM_df, 
    index='pddate', 
    columns='PERMNO', 
    values='ALTPRC'
    ).sort_index().shift(1)

altprc_lag = altprc_lag_df.reset_index().melt(
    id_vars='pddate', 
    var_name='PERMNO', 
    value_name='ALTPRC_LAG1'
    ).dropna()

me_lag_df = pd.pivot_table(
    RET_ME_BM_df, 
    index='pddate', 
    columns='PERMNO', 
    values='ME'
    ).sort_index().shift(1)

me_lag = me_lag_df.reset_index().melt(
    id_vars='pddate', 
    var_name='PERMNO', 
    value_name='ME_LAG1'
    ).dropna()

RET_ME_BM_df = pd.merge(
    RET_ME_BM_df, 
    altprc_lag, 
    on=['pddate', 'PERMNO'], 
    how='left'
    )
RET_ME_BM_df = pd.merge(
    RET_ME_BM_df, 
    me_lag, 
    on=['pddate', 'PERMNO'], 
    how='left'
    )

# %% [markdown]
# 여기부터 안맞음. 차이는 아주 작은데... 

# %%
RET_ME_BM_df['ALTPRC_LAG1'].isna().sum() # NOTE: The data set WORK.TT has 38518 observations and 26 variables.

# %% [markdown]
# ```python
# %%time
#
# """
# * Exclude observations with missing values; 
#
# data assignment1_data; 
#     retain permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 ME_Jun BM; 
#         set ret_ME_BM; 
#         if nmiss(retadj, ME_lag1, ME_Jun, BM) = 0; 
#         keep permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 ME_Jun BM;         
# run;
# """
#
# ret_me_bm = ret_me_bm[['permno', 'date', 'year', 'exchcd', 'siccd', 'retadj', 'eretadj', 'altprc_lag1', 'me_lag1', 'me_jun', 'bm']]
# ret_me_bm = ret_me_bm.dropna(subset=['retadj', 'me_lag1', 'me_jun', 'bm'])
# ```

# %%
# # %%time

# """
# * Exclude observations with missing values; 

# data assignment1_data; 
#     retain permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 ME_Jun BM; 
#         set ret_ME_BM; 
#         if nmiss(retadj, ME_lag1, ME_Jun, BM) = 0; 
#         keep permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 ME_Jun BM;         
# run;
# """

# ret_me_bm = ret_me_bm[['permno', 'date', 'year', 'exchcd', 'siccd', 'retadj', 'eretadj', 'altprc_lag1', 'me_lag1', 'me_jun', 'bm']]
# ret_me_bm = ret_me_bm.dropna(subset=['retadj', 'me_lag1', 'me_jun', 'bm'])

RET_ME_BM_df = RET_ME_BM_df[
    [
        'PERMNO',
        'DATE',
        'YEAR',
        'EXCHCD',
        'SICCD',
        'RETADJ',
        'ERETADJ',
        'ALTPRC_LAG1',
        'ME_LAG1',
        'ME_JUN',
        'BM',
    ]
]
RET_ME_BM_df.dropna(subset=['RETADJ', 'ME_LAG1', 'ME_JUN', 'BM'], inplace=True)


# %% [markdown]
# 여기도 안맞음 딱 1차이남

# %%
RET_ME_BM_df.shape # NOTE: The data set WORK.ASSIGNMENT1_DATA has 1983365 observations and 11

# %%
RET_ME_BM_df[ RET_ME_BM_df['DATE'] == 20121231]

# %%
RET_ME_BM_df.iloc[:25]

# %% [markdown]
# ```python
# from pandas.tseries.offsets import MonthEnd
#
# ret_me_bm['date'] = ret_me_bm['date'] + MonthEnd(0)
#
# summ_dates = [str(x) + '-12-31' for x in range(1970, 2013, 1)]
# summ_stats1 = pd.DataFrame(index=summ_dates, columns=['mean', 'std', 'min', 'max', "N of permno's"])
#
# for date in summ_dates:
#     tmp_eretadj = ret_me_bm[ret_me_bm['date']==date]['eretadj']
#     tmp_permno = ret_me_bm[ret_me_bm['date']==date]['permno']
#     summ_stats1.loc[date] = [tmp_eretadj.mean(), tmp_eretadj.std(), tmp_eretadj.min(), tmp_eretadj.max(), len(tmp_permno.unique())]
#
# summ_stats1
# ```

# %%
from pandas.tseries.offsets import MonthEnd

RET_ME_BM_df['pddate'] = pd.to_datetime(RET_ME_BM_df['DATE'], format='%Y%m%d')
RET_ME_BM_df['pddate'] = RET_ME_BM_df['pddate'] + MonthEnd(0)

summ_dates = [str(x) + '-12-31' for x in range(1970, 2013, 1)]
summ_stats1 = pd.DataFrame(index=summ_dates, columns=['mean', 'std', 'min', 'max', "N of permno's"])

for date in summ_dates:
    tmp_eretadj = RET_ME_BM_df[RET_ME_BM_df['pddate']==date]['ERETADJ']
    tmp_permno = RET_ME_BM_df[RET_ME_BM_df['pddate']==date]['PERMNO']
    summ_stats1.loc[date] = [tmp_eretadj.mean(), tmp_eretadj.std(), tmp_eretadj.min(), tmp_eretadj.max(), len(tmp_permno.unique())]

summ_stats1

# %%
summ_stats1.index

# %%
summ_stats1.loc[summ_stats1.index.str[3] == '0', :]

# %% [markdown]
# ```python
# from pandas.tseries.offsets import MonthEnd
# aa = pd.read_csv('./assignment1_data.csv', encoding='cp949')
# aa['date'] = pd.to_datetime(aa['date'], format='%Y%m%d') + MonthEnd(0)
#
# summ_dates = [str(x) + '-12-31' for x in range(1970, 2013, 1)]
# summ_stats2 = pd.DataFrame(index=summ_dates, columns=['mean', 'std', 'min', 'max', "N of permno's"])
#
# for date in summ_dates:
#     tmp_eretadj = aa[aa['date']==date]['eretadj']
#     tmp_permno = aa[aa['date']==date]['permno']
#     summ_stats2.loc[date] = [tmp_eretadj.mean(), tmp_eretadj.std(), tmp_eretadj.min(), tmp_eretadj.max(), len(tmp_permno.unique())]
#
# summ_stats2
# ```

# %%
from pandas.tseries.offsets import MonthEnd

summ_dates = [str(x) + '-12-31' for x in range(1970, 2013, 1)]
summ_stats2 = pd.DataFrame(index=summ_dates, columns=['mean', 'std', 'min', 'max', "N of permno's"])

for date in summ_dates:
    tmp_eretadj = RET_ME_BM_df[RET_ME_BM_df['pddate']==date]['ERETADJ']
    tmp_permno = RET_ME_BM_df[RET_ME_BM_df['pddate']==date]['PERMNO']
    summ_stats2.loc[date] = [tmp_eretadj.mean(), tmp_eretadj.std(), tmp_eretadj.min(), tmp_eretadj.max(), len(tmp_permno.unique())]
    

# %%
summ_stats2

# %%
(summ_stats1 - summ_stats2).astype(float).round(5)

# %%

# %%

# %% [markdown]
# SAS 코드만 보고 직접 구현 시도

# %%
# * Match each permno's monthly return to the corresponding BM and ME; 
# data ret; 
#  set CRSP_M3;  
#  if month(date)>6 then t = year(date);  ** 6월이후의 리턴이면 (2000.07~2000.12) --> t=2000, 6월 이전 2000.01~2000.06 --> t=1999 ;
#  else t = year(date)-1; 
# run; 
# proc sort data = ret; by permno t date; run; 

RET_df = CRSP_M_df.copy()
RET_df['T'] = np.where(
    RET_df['DATE'] % 10000 // 100 > 6, 
    RET_df['DATE'] // 10000, 
    RET_df['DATE'] // 10000 - 1
    )

RET_df = RET_df[['PERMNO', 'T', 'DATE', 'ALTPRC']]
RET_df.sort_values(['PERMNO', 'T', 'DATE'], inplace=True)

# %%
RET_df.shape # NOTE: There were 2892465 observations read from the data set WORK.RET.

# %%
# data ret_ME_BM; 
#  merge ret (in = a) ME_BM (in = b); 
#  by permno t; 
#  if a; 
# run;
# proc sort data = ret_ME_BM; by permno date; run; 

RET_ME_BM_df = pd.merge(
    left=RET_df,
    right=ME_BM_df,
    how='inner',
    on=['PERMNO', 'T'],
)
RET_ME_BM_df.sort_values(['PERMNO', 'DATE'], inplace=True)


# %%
RET_ME_BM_df.shape

# %%
# * Also add the mktcap and stock price from the previous month; 
# ** 전 달 mktcap, stock price 추가 ;

# data ret_ME_BM; 
#  set ret_ME_BM; 
#  altprc_lag1 = lag1(altprc); ** illiquid한 stock의 경우 lag가 2칸인 경우도 있을 것이다. (한 달동안 거래 안됨) ;
#  ** 이 경우 무조건 lag1으로 shift 쓰면 안됨. ;
#  ** 비어있는 month를 넣어준 다음에야 shift했을 때 한 칸씩 오롯이 잘 밀리게 됨. ;
#  ** multiindex로 보든지 해야 shift했는데 다른 permno의 lag1이랑 섞이지 않게 할 수 있음. ;
#  ME_lag1 = lag1(ME); 
#  permno_lag1 = lag1(permno); 
#  date_lag1 = lag1(date); 
#  if (permno NE permno_lag1) or (intck('month',date_lag1,date)>1) then do; ** SAS니까, 변수 순서 때문에 넣은 줄이라고 하심. Python은 무관 ; 
#  ** 의미하는 것은, date와 date.lag1 차이가 1달보다 크면 ;
#   altprc_lag1 = .; 
#   ME_lag1 = .; 
#   end; 
# run;

RET_ME_BM_df['ALT_PRC_LAG1'] = RET_ME_BM_df.groupby('PERMNO')['ALTPRC'].shift(1)
RET_ME_BM_df['ME_LAG1'] = RET_ME_BM_df.groupby('PERMNO')['ME_JUN'].shift(1)
RET_ME_BM_df['PERMNO_LAG1'] = RET_ME_BM_df.groupby('PERMNO')['PERMNO'].shift(1)
RET_ME_BM_df['DATE_LAG1'] = RET_ME_BM_df.groupby('PERMNO')['DATE'].shift(1)

# %%
RET_ME_BM_df

# %%
