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
# # 자계추 hw2
#
# 20249433 최재필

# %%
import pandas as pd
import numpy as np

from scipy.stats.mstats import winsorize

# %% [markdown]
# ## Load Data

# %% [markdown]
# ### hw1 data
#
# firm-level ME, BM, adjret

# %%
hw1_df = pd.read_csv('assignment1_data.csv')
hw1_df.head()

# %%
hw1_df.info()

# %% [markdown]
# ### beta

# %%
beta_df = pd.read_csv('monthly_beta_data.csv')
beta_df.head()

# %%
beta_df.info()

# %% [markdown]
# ### CPI

# %%
cpi_df = pd.read_excel('CPIAUCSL.xls', skiprows=10)
cpi_df.head()

# %%
cpi_df.info()

# %% [markdown]
# ## SAS 2

# %%
# * SAS 2: Add the beta measure to the monthly stock data set *******************************;
# * Keep only the last observation of each month to get monthly betas;

# %%
# * Generate the month-end date of each monthly stock observation;
# ** last trading date이지, 무조건 월말이 아니다. 예를들어 그 달은 내내 거래 안해서 5일이 마지막 거래일이다? 그럼 5일의 beta로 ;
# data monthly_stock_data1; 
#     set my_lib.assignment1_data; 
#     t = intnx('month', date, 0, 'end'); ** 날짜변수를 숫자변수인 t로 바꿔주는 과정 ;
#     format t yymmddn8.; 
# run; 

hw1_df['pddate'] = pd.to_datetime(hw1_df['date'].astype(str), format='%Y%m%d')
hw1_df['t'] = (hw1_df['pddate'] + pd.offsets.MonthEnd(0)).dt.strftime('%Y%m%d').astype(int) # date를 매달 마지막 날짜로 바꿔줌 (date가 2013-05-20과 같이 끊길 수도 있어서)

# %%
# * Add beta information to the monthly stock data set;
# data monthly_stock_data2; 
#     merge monthly_stock_data1 (in = a) monthly_beta_data (in = b); 
#     by permno t; 
#     if a and b and missing(b_mkt) = 0; 
# run; 

monthly_stock_data2 = pd.merge(
    left=hw1_df, 
    right=beta_df, 
    left_on=['permno', 't'],
    right_on=['PERMNO', 't'],
    how='inner'
    )

monthly_stock_data2.dropna(subset=['B_MKT'], inplace=True)
monthly_stock_data2.drop(columns=['PERMNO'], inplace=True)

# %%
monthly_stock_data2.head()

# %%
# NOTE: The data set WORK.MONTHLY_STOCK_DATA2 has 1972533 observations and 13 variables.
monthly_stock_data2.shape 

# %% [markdown]
# ## SAS 3

# %%
# * SAS 3: Generate Mktcap_CPI and Size variables ******************************************;

# %%
# * CPI in June of each year;
# data CPI_Jun; 
#     set CPI (where = (month(cpi_date) = 6)); 
#     t = year(cpi_date); 
#     keep t cpi; 
# run; 

CPI_Jun = cpi_df.loc[cpi_df['observation_date'].dt.month == 6, :].copy()
CPI_Jun['t'] = CPI_Jun['observation_date'].dt.year

# %%
# NOTE: There were 73 observations read from the data set WORK.CPI.
#       WHERE MONTH(cpi_date)=6;

CPI_Jun.shape

# %%
# * CPI in Dec, 2012;
# # %let CPI_2012 = 231.221; 

CPI_2012 = cpi_df[ cpi_df['observation_date'] == '2012-12-01']['CPIAUCSL'].values[0]
CPI_2012

# %%
# * Calculate Mktcap_CPI, Size, and log_BM variables;
# data monthly_stock_data2; 
#     set monthly_stock_data2; ** beta까지 넣어둔 assignment 1 data를 가져온다. ;
#     if month(date) > 6 then t = year(date); 
#     else t = year(date) - 1; 
# run; 

monthly_stock_data2['t'] = monthly_stock_data2['pddate'].apply(
    lambda x: x.year if x.month > 6 else x.year - 1
    ) # 이번엔 t가 year인데, 7월 이후면 그냥 year, 6월 이전이면 year-1로 바꿔줌

# %%
# proc sort data = monthly_stock_data2; 
#     by t date permno; 
# run; 

monthly_stock_data2.sort_values(by=['t', 'pddate', 'permno'], inplace=True)

# %%
monthly_stock_data2.head()

# %%
monthly_stock_data2.shape

# %%
# data monthly_stock_data3; 
#     merge monthly_stock_data2 (in = a) CPI_Jun (in = b); 
#     by t; 
#     if a; 

monthly_stock_data3 = pd.merge(
    left=monthly_stock_data2, 
    right=CPI_Jun, 
    on='t',
    how='left'
    )


# %%
#     ME_Jun_CPI = (ME_Jun / cpi) * &CPI_2012; ** cpi-adjusted ME ;
#     size = log(ME_Jun); ** size도 log 씌워줌 ;
#     size_CPI = log(ME_Jun_CPI); 
#     log_BM = log(BM); 
#     keep permno date year exchcd siccd retadj eretadj altprc_lag1 ME_lag1 
#     b_mkt size size_CPI BM log_BM; ** 이 5개가 중요한 데이터 ;
# run; 

monthly_stock_data3['ME_Jun_CPI'] = (monthly_stock_data3['ME_Jun'] / monthly_stock_data3['CPIAUCSL']) * CPI_2012
monthly_stock_data3['size'] = np.log(monthly_stock_data3['ME_Jun'])
monthly_stock_data3['size_CPI'] = np.log(monthly_stock_data3['ME_Jun_CPI'])
monthly_stock_data3['log_BM'] = np.log(monthly_stock_data3['BM'])

monthly_stock_data3 = monthly_stock_data3[['permno', 'date', 't', 'exchcd', 'siccd', 'retadj', 'eretadj', 'altprc_lag1', 'ME_lag1', 
                                           'B_MKT', 'size', 'size_CPI', 'BM', 'log_BM']]

# %%
monthly_stock_data3.head()

# %%
monthly_stock_data3.shape

# %% [markdown]
# ## SAS 4

# %%
# * SAS 4: Winsorize stock characteristic variables ****************************************;

# %%
# * Rename characteristic variables;
# data monthly_stock_data3; 
#     set monthly_stock_data3; 
#     ** _o : original data ;
#     rename b_mkt = b_mkt_o size = size_o size_CPI = size_CPI_o BM = BM_o log_BM = log_BM_o; 
# run; 




# 그냥 winsorize하면 되므로 SAS의 이런 과정 필요 없음. 

# monthly_stock_data3 = monthly_stock_data3.rename(
#     columns={
#         'B_MKT': 'b_mkt_o', 
#         'size': 'size_o', 
#         'size_CPI': 'size_CPI_o', 
#         'BM': 'BM_o', 
#         'log_BM': 'log_BM_o'
#         }
#     )

# %%
# * Calculate 0.5% and 99.5% level of each characteristic variable on a monthly basis;
# proc sort data = monthly_stock_data3; 
#     by date; 
# run; 

# proc univariate data = monthly_stock_data3 noprint; 
#     by date; 
#     var b_mkt_o size_o size_CPI_o BM_o log_BM_o; ** _o 붙은게 winsorize 된 것들 ;
#     output out = bounds pctlpts = 0.5 99.5 pctlpre = b_mkt_ size_ size_CPI_ BM_ log_BM_; 
# run; 

# * Merge the bounds with the monthly stock data and winsorize characteristic variables;
# data monthly_stock_data4; 
#     merge monthly_stock_data3 bounds; 
#     by date; 

#     array original(5) b_mkt_o size_o size_CPI_o BM_o log_BM_o; 
#     array winsorized(5) b_mkt size size_CPI BM log_BM; 
#     array l_bound(5) b_mkt_0_5 size_0_5 size_CPI_0_5 BM_0_5 log_BM_0_5; 
#     array u_bound(5) b_mkt_99_5 size_99_5 size_CPI_99_5 BM_99_5 log_BM_99_5; 

#     do ii = 1 to 5; 
#         if original(ii) < l_bound(ii) then winsorized(ii) = l_bound(ii); 
#         else if original(ii) > u_bound(ii) then winsorized(ii) = u_bound(ii); 
#         else winsorized(ii) = original(ii); 
#     end; 

#     drop b_mkt_0_5--log_BM_99_5 ii b_mkt_o size_o size_CPI_o BM_o log_BM_o; 
# run; 


WINSORIZE_LEVEL = 0.005

winsorize_cols = ['B_MKT', 'size', 'size_CPI', 'BM', 'log_BM']
monthly_stock_data4 = monthly_stock_data3.copy()
for col in winsorize_cols:
    monthly_stock_data4[f'{col}_o'] = monthly_stock_data4[col]

monthly_stock_data4[winsorize_cols] = monthly_stock_data4.groupby('date').transform(
    lambda x: winsorize(x, limits=(WINSORIZE_LEVEL, WINSORIZE_LEVEL))
    )[winsorize_cols]

# %%
# winsorize했기 때문에 original data와 다른 것들을 확인할 수 있다.
monthly_stock_data4[monthly_stock_data4['BM'] != monthly_stock_data4['BM_o']].sort_values(by='date').head()


# %% [markdown]
# ## SAS 5

# %%
# * SAS 5: Calculate summary statistics;

# %%
# # %let varlist = b_mkt size size_CPI BM log_BM; 

# ods exclude all; 

# proc sort data = monthly_stock_data4; 
#     by date permno; 
# run; 

# proc means data = monthly_stock_data4 mean std skew kurt min p5 p25 median p75 max n stackodsoutput nolabels; 
#     by date; 
#     var &varlist; 
#     ods output summary = stats_by_month; 
# run; 

# ods exclude none; 

def agg_pct(p):
    def percentiles(x):
        return np.percentile(x, p)

    percentiles.__name__ = f'p{p}'

    return percentiles

stats = [
    'mean',
    'std',
    'skew',
    # 'kurtosis', # Not a method of DataFrameGroupBy
    pd.Series.kurt,
    'count',

    'min',
    agg_pct(5),
    agg_pct(25),
    'median',
    agg_pct(75),
    'max',
]

summary_stats_df = monthly_stock_data4.groupby('date').agg(
    {'B_MKT': stats, 
     'size': stats, 
     'size_CPI': stats, 
     'BM': stats, 
     'log_BM': stats}
    )

summary_stats_df.head()

# %%
# * Calculate the time-series-means of the summary statistics for the variables in the "varlist";

winsorize_cols

# %%
# proc sort data = stats_by_month; ** cross sectional로 구해놓은 bm mean들을 한 번 더 time-series로 mean ;
#     by variable date; 
# run; 

# proc means data = stats_by_month mean nolabels noprint; 
#     by variable; 
#     var mean stddev skew kurt min p25 median p75 max n; 
#     output out = stats (drop = _TYPE_ _FREQ_) mean(mean stddev skew kurt min p25 median p75 max n) = mean stddev skew kurt min p25 median p75 max n; 
# run; 

ts_summary_stats_df = pd.DataFrame(summary_stats_df.mean(), columns=['ts_mean'])
ts_summary_stats_df.index.set_names(['var', 'stats'], inplace=True)

ts_summary_stats_df = ts_summary_stats_df.unstack(level='stats')
ts_summary_stats_df.columns = ts_summary_stats_df.columns.droplevel(0) # ts_mean이라는 level(0)을 제거하고 남은 columns들을 반환
ts_summary_stats_df

# %% [markdown]
# ## SAS 6

# %%
# * SAS 6: Calculate correlations;

# %%
# proc corr data = monthly_stock_data4 outp = pcorr_by_month (where = (_TYPE_ = "CORR")) noprint; 
#     by date; 
#     var &varlist; 
# run; 

monthly_corr = monthly_stock_data4.groupby('date')[winsorize_cols].corr()


# %%
# * Calculate the time-series-means of the correlations for variables in the "varlist";
# proc sort data = pcorr_by_month; 
#     by _name_ date; 
# run; 

# proc means data = pcorr_by_month mean nolabels noprint; 
#     by _name_; 
#     var &varlist; 
#     output out = pcorr (keep = _NAME_ &varlist) mean(&varlist) = &varlist; 
# run; 

ts_monthly_corr = pd.DataFrame(monthly_corr.unstack().mean(), columns=['ts_mean'])
ts_monthly_corr.index.set_names(['var1', 'var2'], inplace=True)
ts_monthly_corr = ts_monthly_corr.unstack(level='var2')
ts_monthly_corr.columns = ts_monthly_corr.columns.droplevel(0)

ts_monthly_corr

# %% [markdown]
# ## SAS 7

# %%
# * SAS 7: Dependent-sort stocks into 25 portfolios based on size and BM *******************************;

# %%
winsorize_cols

# %%
monthly_stock_data4

# %%
# * Calculate size breakpoints as 20th, 40th, 60th, and 80th size percentiles among NYSE stocks in each month;
# *** NYSE stock들만 가지고 breakpoint를 찾아라. ;
# proc univariate data = monthly_stock_data4 (where = (exchcd in (1, 31))) noprint; 
#     by date; 
#     var size; 
#     output out = size_breakpoints pctlpts = 20 40 60 80 pctlpre = size_; 
# run; 

# * Merge the size breakpoints with the monthly stock data and define size sorted portfolios;
# data monthly_stock_data5; 
#     merge monthly_stock_data4 size_breakpoints; 
#     by date; 

#     if size < size_20 then p1 = 1; 
#     else if size < size_40 then p1 = 2; 
#     else if size < size_60 then p1 = 3; 
#     else if size < size_80 then p1 = 4; 
#     else p1 = 5; 
# run; 

nyse_size_breakpoints = monthly_stock_data4[ monthly_stock_data4['exchcd'].isin([1, 31]) ].groupby('date')['size'].quantile([0.2, 0.4, 0.6, 0.8])
nyse_size_breakpoints = nyse_size_breakpoints.unstack(level=1)
nyse_size_breakpoints.columns = [f'size_{int(p*100)}' for p in nyse_size_breakpoints.columns]
nyse_size_breakpoints.reset_index(inplace=True, drop=False)

monthly_stock_data4 = pd.merge(
    left=monthly_stock_data4, 
    right=nyse_size_breakpoints, 
    on='date',
    how='left'
    )


# monthly_stock_data4['p1'] = monthly_stock_data4.apply(
#     lambda row: row[ ['size_20', 'size_40', 'size_60', 'size_80'] ].searchsorted(row['size']) + 1,
#     axis=1
#     )

## 너무 느리다. vectorized operation으로 바꿔보자.

size_bounds = monthly_stock_data4[['size_20', 'size_40', 'size_60', 'size_80']].values
size = monthly_stock_data4['size'].values.reshape(-1, 1)
size_group = ( size_bounds <= size ).sum(axis=1)

monthly_stock_data4['p1'] = size_group + 1

# %%
# Not used

# def get_qcut_breakpoints(x, q=[20, 40, 60, 80]):
#     q = np.array(q)
#     lower_q = q[q<=50]
#     upper_q = q[q>50]

#     lower_bounds = np.percentile(x, lower_q, method='lower')
#     upper_bounds = np.percentile(x, upper_q, method='higher')

#     return np.concatenate([lower_bounds, upper_bounds])

# %%
# proc univariate data = monthly_stock_data5 noprint; 
#     by date p1; 
#     var BM; 
#     output out = BM_breakpoints pctlpts = 20 40 60 80 pctlpre = BM_; 
# run; 

# * Merge the BM breakpoints with the monthly stock data and define BM sorted portfolios in each size sorted portfolio;
# data monthly_stock_data6; 
#     merge monthly_stock_data5 BM_breakpoints; 
#     by date p1; 

#     if BM < BM_20 then p2 = 1; 
#     else if BM < BM_40 then p2 = 2; 
#     else if BM < BM_60 then p2 = 3; 
#     else if BM < BM_80 then p2 = 4; 
#     else p2 = 5; 
# run; 


## 중요한 차이: independent sort가 아니라 dependent sort임.

monthly_stock_data4['p2'] = monthly_stock_data4.groupby(['date', 'p1'])['BM'].transform(
    lambda x: pd.qcut(x, q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False, duplicates='drop') + 1
    )

# %%
monthly_stock_data4.head()

# %% [markdown]
# ### SAS 7 (alternative)
#
# - size_20, ... , size_80 컬럼과 bm_20, ..., bm_80 컬럼을 만들기
# - `pd.qcut`을 쓸 경우 필요하지 않지만, 과제2 요건에 맞추기 위해 만듦. (만들고 쓰진 않음)

# %%
# def get_breakpoints_sas(x, n):
#     i = int(100/n)
#     return [x.quantile(p/100, interpolation='higher') if p >= 50 
#             else x.quantile(p/100, interpolation='lower')
#             for p in range(0, 100 + i, i)]


# quantiles = [0.5, 20, 40, 60, 80, 99.5] # in ascending order
quantiles = [20, 40, 60, 80,] # in ascending order

## size columns 
for q in quantiles:
    interpolation = 'higher' if q >= 50 else 'lower'
    monthly_stock_data4[f'size_{q}'] = monthly_stock_data4.groupby('date')['size'].transform(lambda x: np.percentile(x, q, method=interpolation))
    monthly_stock_data4[f'bm_{q}'] = monthly_stock_data4.groupby(['date', 'p1'])['BM'].transform(lambda x: np.percentile(x, q, method=interpolation))

# %%
monthly_stock_data4.columns

# %%
report_cols = [
    'permno',
    'date',
    'B_MKT',
    'size',
    'size_CPI',
    'BM',
    'log_BM',
    # 'size_0.5',
    'size_20',
    'size_40',
    'size_60',
    'size_80',
    # 'size_99.5',
    # 'bm_0.5',
    'bm_20',
    'bm_40',
    'bm_60',
    'bm_80',
    # 'bm_99.5',
    'p1',
    'p2'
]

monthly_stock_data4[report_cols].sort_values(by=['date', 'permno']).head(25)

# %%
# * Calculate the time-series average number of stocks in each portfolio;
# proc sort data = monthly_stock_data6; 
#     by date p1 p2; 
# run; 

# proc means data = monthly_stock_data6 n nolabels noprint; 
#     by date p1 p2; ** 서로 다른 permno들이 각 p1, p2에 몇개씩 있는지 count ;
#     var permno; 
#     output out = nstocks_per_p n = nstocks; 
# run; 

# proc sort data = nstocks_per_p; 
#     by p1 p2; 
# run; 

nstocks_per_p = pd.DataFrame(monthly_stock_data4.groupby(['date', 'p1', 'p2']).size(), columns=['nstocks'])
nstocks_per_p.index.set_names(['date', 'p1', 'p2'], inplace=True)
nstocks_per_p = nstocks_per_p.unstack(level=['p1', 'p2'])
nstocks_per_p.columns = nstocks_per_p.columns.droplevel(0)

ts_nstocks_per_p = pd.DataFrame(nstocks_per_p.mean(), columns=['ts_mean'])
ts_nstocks_per_p = ts_nstocks_per_p.unstack(level='p2')
ts_nstocks_per_p.columns = ts_nstocks_per_p.columns.droplevel(0)
ts_nstocks_per_p

# %%
