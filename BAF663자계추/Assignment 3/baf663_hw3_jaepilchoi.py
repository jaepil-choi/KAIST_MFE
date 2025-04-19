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
# # 자계추 hw3
#
# 20249433 최재필

# %%
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance
from statsmodels.regression.linear_model import OLS

# %% [markdown]
# ## Load data

# %%
# hw2 data
portfolio = pd.read_csv('assignment2_data.csv')

# ff3 factors (monthly)
ff3 = pd.read_csv('factors_monthly.csv')


# %%
portfolio.columns

# %% [markdown]
# ## SAS 3
#
# - Calculate the monthly value-weighted portfolio returns

# %%
# data portfolio1;
#  set My_lib.assignment2_data;
#  eretadj_ME = (eretadj*100)*ME_lag1; ** value weighted 조정초과수익률 구함. ME lag 1을 쓰는 것을 주목. ; 
# run;

portfolio['eretadj_ME'] = (portfolio['eretadj']*100)*portfolio['ME_lag1']
# 여기선 우선 1 month lagged ME를 곱해 수익률에 ME만큼 가중치를 곱해줌. (그러나 아직 총 ME로 나눠주지 않음)

# %%
# proc sort data = portfolio1;
#  by date p1 p2;
# run;

portfolio = portfolio.sort_values(by=['date', 'p1', 'p2'])

# %%
# proc means data = portfolio1 sum noprint;
#  by date p1 p2;
#  var eretadj_ME ME_lag1;
#  output out = portfolio2 (drop = _FREQ_ _TYPE_) sum = / autoname;
# run;

# data portfolio2;
#  set portfolio2;
#  vw_pret = eretadj_ME_Sum/ME_lag1_Sum;
#  keep date p1 p2 vw_pret;
# run;

portfolio2 = portfolio.groupby(['date', 'p1', 'p2'], as_index=False).agg(
    {
        'eretadj_ME': 'sum', 
        'ME_lag1': 'sum',
    }
)
portfolio2['vw_pret'] = portfolio2['eretadj_ME'] / portfolio2['ME_lag1']
# Transform 쓰면 duplicate rows 생김. agg 쓰는 것이 안전. 

portfolio2 = portfolio2.loc[:, ['date', 'p1', 'p2', 'vw_pret']]
portfolio2.head()

# %%
# * Calculate the return difference between the fifth (p2 = 5) and first (p2 = 1) BM sorted portfolios within each Size sorted portfolio;
# data portfolio3;
#  set portfolio2(where = (p2 in (1,5))); ** 사이즈 내에서 BM1과 BM5 차이 ;
# run;

# proc sort data = portfolio3;
#  by date p1 p2;
# run;

# proc transpose data = portfolio3 out = portfolio4; ** pivot table해서 index date, column vw_pret ;
#  by date p1;
#  id p2;
#  var vw_pret;
# run;

# data portfolio4;
#  set portfolio4;
#  p2 = 51;
#  vw_pret = _5 - _1;
#  keep date p1 p2 vw_pret; ** 똑같이 요것만 남긴다. 테이블 port 2 아래로 append 하기 위해 ;
# run;

portfolio4 = portfolio2[portfolio2['p2'].isin([1, 5])].pivot_table(
    index=['date', 'p1'], 
    columns='p2', # 1, 5
    values='vw_pret'
    ).reset_index()

portfolio4 = portfolio4.rename_axis(None, axis=1)
portfolio4['p2'] = 51

portfolio4['vw_pret'] = portfolio4[5] - portfolio4[1]
portfolio4 = portfolio4.loc[:, ['date', 'p1', 'p2', 'vw_pret']]
portfolio4.head()

# %%
# * Append the two datasets;
# data portfolio5;
#  set portfolio2 portfolio4;
#  year = year(date);
#  month = month(date);
# run;

# proc sort data = portfolio5;
#  by year month date p1 p2;
# run;

portfolio5 = pd.concat([portfolio2, portfolio4], axis=0)
portfolio5['year'] = portfolio5['date'] // 10000
portfolio5['month'] = portfolio5['date'] % 10000 // 100
portfolio5 = portfolio5.sort_values(by=['year', 'month', 'date', 'p1', 'p2'])
portfolio5.head()

# %% [markdown]
# ## SAS4
#
# - Add FF-3 factors to the portfolio return data set

# %%
#  * convert factors from decimal to percent;
#  mktrf = mktrf*100;
#  smb = smb*100;
#  hml = hml*100;
#  keep year month mktrf smb hml;
# run;

ff3[['mktrf', 'smb', 'hml']] = ff3[['mktrf', 'smb', 'hml']] * 100
ff3 = ff3.loc[:, ['year', 'month', 'mktrf', 'smb', 'hml']]

# %%
# * Merge;
# data portfolio6; ** 팩터가 더 길지만 left join이라 길이가 변하지 않는다. ; 
#  merge portfolio5 (in = a) factors (in = b);
#  by year month;
#  if a;
# run;

portfolio6 = pd.merge(portfolio5, ff3, on=['year', 'month'], how='left')
portfolio6.head()

# %% [markdown]
# ## SAS5
#
# - Test if the BM5 portfolio has a higher expected return than BM1 portfolio within each size group using time-series regressions

# %%
# proc sort data = portfolio6;
#  by p1 p2 date;
# run;

portfolio6 = portfolio6.sort_values(by=['p1', 'p2', 'date'])

# %%
portfolio6.columns

# %%
# * To perform Newey-West standard error correction, PROC MODEL is run specifying the GMM estimation method in the FIT statement. KERNEL=(BART, L+1, 0) is also specified which requests the Bartlett kernel with a lag length of L. The VARDEF(specify the denominator for computing variances and covariances)=n option is specified to be consistent with the original Newey-West formula;

# * Calculate the FF3 alpha of the long-short portfolio (p2=51);
# proc model data = portfolio6 (where = (p2 = 51)); ** BM5 - BM1 이 51;
#  by p1;
#  exog mktrf hml smb; ** FF3 팩터들을 regression ;
#  instruments _exog_;
#  vw_pret = a + b1*mktrf + b2*hml + b3*smb;
#  fit vw_pret / gmm kernel = (bart, 7, 0) vardef = n; ** gmm kernel이 newey west를 쓰는 것이다. 7이 max lag ;
#  ** newey west 쓰는 이유: time series에선 error term의 autocorrelation이 있을 수 있기 때문에 이를 보정하기 위해 ;
#  ** 안쓰면, autocorrelation에 의해 standard error가 작게 나올 수 있다. ;
#  ** 즉, 원래는 유의미하지 않은데, autocorrelation으로 인해 유의미하게 나올 수 있다. ;
#  ** SAS에선 7 = max_lag + 1이다. python에선 그냥 max_lag를 넣어야 한다.;
#  ods output parameterestimates = table3;
#  quit;
# ods exclude none;

p51_reg = portfolio6[portfolio6['p2'] == 51].loc[:, ['date', 'p1', 'vw_pret', 'mktrf', 'hml', 'smb']]
p51_reg.head()


# %%
def apply_ff3reg_by_group(group, max_lag=6):
    y = group['vw_pret']
    X = group[['mktrf', 'hml', 'smb']]
    X = sm.add_constant(X)
    
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag}, kernel='bartlett')
    # Specifies the use of Heteroskedasticity and Autocorrelation Consistent (HAC) covariance with a Bartlett kernel and a maximum lag of 6. 
    # This setup mirrors the Newey-West standard error correction in SAS.

    params = model.params
    tstats = model.tvalues

    result = {
        'const': params.get('const', np.nan),
        'const_tstats': tstats.get('const', np.nan),
        'b1': params.get('mktrf', np.nan),
        'b1_tstats': tstats.get('mktrf', np.nan),
        'b2': params.get('hml', np.nan),
        'b2_tstats': tstats.get('hml', np.nan),
        'b3': params.get('smb', np.nan),
        'b3_tstats': tstats.get('smb', np.nan),
    }

    return result

def apply_capmreg_by_group(group, max_lag=6):
    y = group['vw_pret']
    X = group[['mktrf']]
    X = sm.add_constant(X)
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag}, kernel='bartlett')

    params = model.params
    tstats = model.tvalues

    result = {
        'const': params.get('const', np.nan),
        'const_tstats': tstats.get('const', np.nan),
        'b1': params.get('mktrf', np.nan),
        'b1_tstats': tstats.get('mktrf', np.nan),
    }

    return result



# %%
capm_reg_result = p51_reg.groupby('p1').apply(apply_capmreg_by_group, include_groups=False).apply(pd.Series)
capm_reg_result

# %%
ff3_reg_results = p51_reg.groupby('p1').apply(apply_ff3reg_by_group, include_groups=False).apply(pd.Series)
ff3_reg_results

# %%
final_table = pd.DataFrame(portfolio6.groupby(['p1', 'p2'])['vw_pret'].mean())
final_table = final_table.pivot_table(index='p2', columns='p1', values='vw_pret')

final_table.index = ['BM1', 'BM2', 'BM3', 'BM4', 'BM5', 'BM5-BM1']
final_table.columns = ['Size1', 'Size2', 'Size3', 'Size4', 'Size5']

# %%
capm_alpha = capm_reg_result[['const', 'const_tstats']].rename(columns={'const': 'CAPM alpha', 'const_tstats': 'CAPM alpha tstats'})
capm_alpha.index = ['Size1', 'Size2', 'Size3', 'Size4', 'Size5']
capm_alpha = capm_alpha.T

# %%
ff3_alpha = ff3_reg_results[['const', 'const_tstats']].rename(columns={'const': 'FF3 alpha', 'const_tstats': 'FF3 alpha tstats'})
ff3_alpha.index = ['Size1', 'Size2', 'Size3', 'Size4', 'Size5']
ff3_alpha = ff3_alpha.T

# %%
final_table = pd.concat([final_table, capm_alpha, ff3_alpha], axis=0)
final_table
