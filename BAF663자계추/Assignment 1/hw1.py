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
# # 자계추 hw1: Create dataset
#
# - `compustat_permno`와 `CRSP_M` 사용
#     - SAS 코드 따라가며 python으로 포팅. 
# - 최종 결과인 `Assignment1_data` 를 만들기 
#     - 최종 결과는 permno/date 순으로 정렬하여 first 25 obs 를 보일 것. 
#     - month of December for year 1970, 1980, 1990, 2000, 2010에 대하여 아래를 report:
#         - number of distinct permnos
#         - mean/std/min/max of the monthly delisting-adjusted excess returns 
#

# %% [markdown]
# ## Lecture Note에서 기억할 내용들
#
# - Compustat vs CRSP
#     - Compustat
#         - id: `GVKEY`, `DATADATE`
#         - owner: S&P Global 
#     - CRSP
#         - id: `PERMNO` (and `PERMCO`)
#         - owner: University of Chicago Booth School of Business
#     - `CCMXPF_LNKUSED` CCM 즉, merged table을 사용

# %% [markdown]
# ## 가이드
# - SAS log 확인하며 중간중간 단계에서 같은 결과가 나오는지 확인해라. 
#     - shape check
# - sample data는 정답지. 최종적으로 output이 일치하는지 확인. 
# - SAS 를 파이썬으로 옮겨준 코드도 참고하기. 
#     - summary statistics 등 뽑는거는 본인 코드 있으면 그거 쓰기. 

# %% [markdown]
# ## 질문했던 것들
#
# - long table vs wide table 
#     - 왜 굳이 wide 안쓰고 long 써서 각종 문제가 생기게 하는지... permno를 1개만 만들어놓을 수 있다면 그냥 그걸 가지고 pivot table 하고나면 그 다음엔 ffill 등이 훨씬 용이해 짐. 
#     - 이 wide를 하고 shift를 쓰는 것을 교수님도 말하심. missing date 찐빠가 날 일이 없음. 그냥 그 자리에 NaN이 차고 말지. 
#     - 교수님이 말씀하시는 단점:
#         - RDBMS 관점에서 비효율적임 
#         - 테이블이 너무 많이 생김. 그 부분 비효율도 생각해라. 

# %% [markdown]
# ## SAS --> Python 포팅

# %% [markdown]
# - SAS1: Connect WRDS from your PC
#     - Get stock data (CRSP)
#     - Get event data (CRSP)
#     - Merge stock & event data
#     - Remove duplicates (by permno, date)
#     - House Cleaning
# - SAS2: Define libs & macro variables
#     - SAS 코드의 주석 참고. compustat 데이터에서 WHERE 로 조건 넣어 필터링함. 
#         - 금융주 제외
#         - standardized report만 쓰고 (?)
#         - domestic report만 쓰고 
#         - consolidated report (연결재무제표)만 쓴다. 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %% [markdown]
# ## Load datasets

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
CRSP_M_df = pd.read_csv(DATA_DIR / 'CRSP_M.csv')
compustat_df = pd.read_csv(DATA_DIR / 'compustat_permno.csv') 
sample_df = pd.read_csv(DATA_DIR / 'assignment1_sample_data.csv')

# %% [markdown]
# ### CRSP

# %%
CRSP_M_df.columns

# %%
CRSP_M_df.shape

# %%
CRSP_M_df.info()

# %%
CRSP_M_df['PERMNO'].nunique()

# %%
CRSP_M_df['PERMCO'].nunique()

# %%
# Is date-permno unique?
CRSP_M_df[['DATE', 'PERMNO']].duplicated().sum() # Yes

# %%
CRSP_M_df['EXCHCD'].unique() # 이미 필터는 처리 되어있다. 

# %% [markdown]
# 그래도 아래 따로 filter 구현. 

# %%
# filters

filter_common_stocks = [10, 11] # SHRCD
filter_exchange = [ # EXCHCD
    1, 31, # NYSE
    2, 32, # AMEX
    3, 33, # NASDAQ
]

# %% [markdown]
# plots

# %%
# TODO: Stock Exchange Composition을 groupby 사용하여 만들기. 별도 column에 NYSE, AMEX, NASDAQ, Other 표시
# TODO: Number of stocks 로 한 번, Market Cap으로 한 번 plot

# %%
# apply filters

CRSP_M_df = CRSP_M_df[ CRSP_M_df['SHRCD'].isin(filter_common_stocks) ]
CRSP_M_df = CRSP_M_df[ CRSP_M_df['EXCHCD'].isin(filter_exchange) ]

# %%
CRSP_M_df.shape

# %%
CRSP_M_df.head()

# %% [markdown]
# ### compustat

# %%
compustat_df.columns

# %%
compustat_df.shape

# %%
compustat_df.info()

# %%
compustat_df['gvkey'].nunique()

# %%
compustat_df['permno'].nunique()

# %%
compustat_df['permco'].nunique()

# %% [markdown]
# datadate는 fiscal year end date이다. 

# %%
# Is date-permno unique?
compustat_df[['datadate', 'permno']].duplicated().sum() # No

# %%
# Is date-gvkey unique?
compustat_df[['datadate', 'gvkey']].duplicated().sum() # No

## 수업시간에 다룬 내용. non-unique한 이유는 기업이 fiscal year을 바꾸거나 할 경우 두 데이터가 동시에 존재할 수 있기 때문이다. 


# %%
# compustat_df.dropna(subset=['permno'], inplace=True) 
# permno 없는 row 여기서 삭제하는게 맞으나, 이걸 해주면 아래에서 row 수가 달라져서 검증 불가하기에 일단 놔둠. 

# %%
compustat_df.head()

# %% [markdown]
# Null인 데이터가 꽤 보인다. 

# %% [markdown]
# CRSP, Compustat date 를 살펴보자

# %%
CRSP_M_df['DATE'].sample(10)

# %%
compustat_df['datadate'].sample(10) # fiscal end date

# %% [markdown]
# ## SAS 3
#
# Construct BE data

# %% [markdown]
# ### Merge CRSP-Compustat using CCM
#
# - pk
#     - crsp: [DATE, PERMNO]
#     - compustat: [datadate, gvkey]
#         - compustat 테이블에 ccm을 통해 생성한 permno 있음. 이를 기준으로 join

# %% [markdown]
# ** 질문: 왜 (inner join 안쓰고) LEFT JOIN 쓰는지?  right table인 CRSP에 데이터 없으면 분석 불가한거 아닌가? **

# %% [markdown]
# ```
# * Add permno and permco to BE data using the link-used table;
# * The nobs might increase because a firm can be matched to multiple permno's; 
# proc sql; 
#  create table compustat_permno  
#  as select distinct a.*, b.upermno as permno, b.upermco as permco  
#  from compustat as a 
#  left join my_lib.ccmxpf_lnkused  
#   ( keep = ugvkey upermno upermco ulinkdt ulinkenddt usedflag ulinktype  
#   where = (usedflag = 1 and ulinktype in ("LU","LC")) ) as b 
#  on a.gvkey = b.ugvkey 
#  and (b.ulinkdt <= a.datadate or missing(b.ulinkdt) = 1) 
#  and (a.datadate <= b.ulinkenddt or missing(b.ulinkenddt) = 1) 
#  order by a.datadate, a.gvkey; 
# quit;
# proc sort data = compustat_permno; by gvkey datadate; run;
# ```

# %% [markdown]
# 위 merge는 지금 주어진 CRSP, Compustat 테이블로 한게 아님. 

# %%
df = pd.merge(
    left=compustat_df, 
    right=CRSP_M_df, 
    left_on=['datadate', 'permno'], 
    right_on=['DATE', 'PERMNO'],
    how='left',
    )

# %%
df.sort_values(by=['gvkey', 'datadate'], inplace=True)

# %%
df[ ['permno', 'datadate'] ].duplicated().sum() # compustat쪽 

# %%
df[ ['DATE', 'PERMNO'] ].duplicated().sum() # crsp쪽. merge 전엔 중복이 없었는데, merge 후 중복이 생겼다.

# %%
(df['DATE'] == df['datadate']).sum()

# %%
df['DATE'].isnull().sum()

# %%
df['datadate'].isnull().sum()

# %%
df.shape # NOTE: Table WORK.COMPUSTAT_PERMNO created, with 434269 rows and 10 columns.

# %% [markdown]
# ```SAS
# * Calculate BE; 
# data BE; 
# set compustat_permno (where = (missing(permno) = 0)); 
# year = year(datadate); 
# ```

# %%
len( df.dropna(subset=['permno'], inplace=False) ) # left에 대해서만 수행해주면 됨. right는 애초에 없으면 붙지 않았음. 

    # NOTE: There were 264450 observations read from the data set
    #   WORK.COMPUSTAT_PERMNO.
    #   WHERE MISSING(permno)=0;

# %%
# 위를 보면 그냥 permno만 dropna 해주는게 숫자가 맞는 것 같은데... 
# datadate, DATE, PERMNO가 null인 경우도 빼줘야 하는거 아닌가? 

df.dropna(subset=['permno'], inplace=True)

# %%
# 이 부분, join 문제라 생각해 내가 임의로 넣은 부분임. 이게 문제인가? 

# df.dropna(subet=['datadate', 'DATE', 'permno', 'PERMNO'], how='any', inplace=True) # key가 없는 row들 삭제

# %%
len(df)

# %%
# 날짜 만들기 
df['YEAR'] = df['DATE'] // 10000 # int로 된 연도
df['pd_DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d') # 원래 SAS코드에는 없는, pd용 datetime


# %%
df.columns

# %% [markdown]
# ```
# if missing(ITCB) then ITCB = 0; * investment tax credit; 
# ```

# %%
# ITCB(Investment Tax Credit Balance): 없는 경우 0으로
# 이건 없는 경우가 많다고 함. 없는 회사를 다 뺄 수는 없으니 0으로. 

df['itcb'] = df['itcb'].fillna(0)

# %% [markdown]
# ```
# BVPS = PSTKRV; * preferred stock - redemption value; 
# if missing(BVPS) then BVPS = PSTKL; * preferred stock - liquidating value; 
# if missing(BVPS) then BVPS = PSTK; * preferred stock- par value; 
# if missing(BVPS) then BVPS = 0; 
# ```

# %% [markdown]
# BE = SEQ + TXDB + ITCB - BVPS 를 위해 BVPS를 구하는데, 
#
# 여기서 BVPS에 많은 처리가 들어간다. 우선주의 가치를 어떻게 산정해야 하지? 
#
# 1. PSTKRV, preferred stock의 redemption value가 있다면 이걸로. 
#     - redemption value: 회사가 자진상장폐지 등의 이유로 주식을 재매입할 때의 금액
# 2. 그게 없으면 PSTKL, liquidating value로 
# 3. 또 없으면 PSTK, par value로 
# 4. 다 없으면 0으로 우선주 가치를 판정

# %% [markdown]
# 이런 식의 operation이 앞으로도 계속 나옴. 
#
# 뭐가 available하면 뭐를 쓰고... 그게 안되면 이러저러한 조건일 때 저걸 쓰고 등등.. 
#
# 이걸 매번 일일이 만들면 너무 힘드므로 처리 가능한 함수를 만들겠음. 
#
# 하지만 조건이 까다로움
#
# - 우선순위를 정해 list로 넣을 수 있어야. 
# - 가장 간단하게, x가 없으면 y를 쓴다 는 같은 row 내에서 가능 (추후 row apply하면 됨)
# - 조건이 달릴 경우. 같은 row 내에서 x가 없으면 A일 때 y를 쓴다 는 식의 로직 처리 가능해야
# - ts 방향으로도 fill이 가능해야 함. ffill 처럼. 이 경우 wide 형식의 panel data인 경우 편하게 할 수 있지만 long data의 경우일 때 처리 가능해야 함. 
#     - groupby ffill하면 가능함. 
#     - groupby 전 permno-date로 sort되어있어야 함. 
#
# 구체적으로 
# - output
#     - 원래의 df 형태를 유지한 채, 빈 곳의 값들이 채워져 나와야 한다. 
# - input
#     - 원래의 df
#     - 그 df에서 채울 대상
#     - row 로직으로 채울껀지 
#     - ts 로직으로 채울껀지
# - row-wise logic
#     - If row['target'] is empty, 
#     - Additional condition
#     - Fill something
# - ts-wise logic
#     - ts series만들어놓고 
#     - if row['target'] is empty, 
#     - Additional condition
#     - Fill pre-made ts series

# %%
from abc import ABC, abstractmethod

class FillLogic(ABC):
    def __init__(self, target_col):
        self.target_col = target_col
    
    def check_empty(self, row):
        # return row[self.target_col] is np.nan
        return pd.isna(row[self.target_col])
    
    def run(self, row):
        if self.check_empty(row):
            return self.fill(row)
        else:
            return row[self.target_col]
    
    @abstractmethod
    def fill(self, row):
        raise NotImplementedError



# %%
class FillZero(FillLogic):
    def __init__(self, target_col):
        super().__init__(target_col)

    def fill(self, row):
        return 0

class FillReplace(FillLogic):
    def __init__(self, target_col, replace_col):
        super().__init__(target_col)
        self.replace_col = replace_col 

    def fill(self, row):
        return row[self.replace_col]


# %%
df.columns

# %%
df['bvps'] = df['pstkrv']

# %% [markdown]
# 돌리기 전

# %% [markdown]
# ** 질문: 애초에 pstkrv에 -가 있는데, 빼주고 시작해야하지 않나?  **
#
# 마지막에 - BVPS 해주니까, 이 경우 - 값들이 다 +로 바뀌면서 더해지는 경우가 생길텐데???

# %%
df['bvps'].describe()

# %%
df['bvps'].isnull().sum()

# %% [markdown]
# 돌린 후

# %%
fill_pstkl = FillReplace('bvps', 'pstkl').run
fill_pstk = FillReplace('bvps', 'pstk').run
fill_zero = FillZero('bvps').run


# %%
df['bvps'] = df.apply(lambda row: fill_pstkl(row), axis=1)
df['bvps'] = df.apply(lambda row: fill_pstk(row), axis=1)
df['bvps'] = df.apply(lambda row: fill_zero(row), axis=1)

# %%
df['bvps'].isnull().sum()

# %%
df['bvps'].describe()

# %% [markdown]
# ```
# BE = SEQ + TXDB + ITCB - BVPS; * If SEQ or TXDB is missing, BE, too, will be missing; 
#
# if BE<=0 then BE = .; * If BE<0, the value of BE is taken to be missing;  
#
# label datadate = "Fiscal Year End Date"; 
# keep gvkey datadate year BE permno permco; 
# run;
# ```

# %%
df['be'] = df['seq'] + df['txdb'] + df['itcb'] - df['bvps']

# %%
df['be'].isnull().sum()

# %%
df.loc[ df['be'] <= 0, 'be' ] = 0

# %% [markdown]
# fiscal year != calendar year이기 때문에, 
#
# 기업이 fiscal year을 바꿀 경우 한 calendar year에 두 결과값이 나오는 경우들이 있다. 
#
# 이 경우 given calendar year에서 가장 뒤에 있는 데이터를 사용

# %% [markdown]
# ** 질문: gvkey랑 permco랑 안맞는 상황. 그래도 sort by gvkey, permno로 해도 되는지... ** 
#
# 현재 한 date에 company가 유일하지 않음. 

# %%
gvkey_permco = df.groupby(['datadate', 'gvkey'])['permco'].nunique()

# %%
gvkey_permco[ gvkey_permco > 1 ] 
# 위에서 싹다 dropna 처리해주면 없어지는데...안하면 이렇게 남아있음. 

# %%
permco_gvkey = df.groupby(['datadate', 'permco'])['gvkey'].nunique()
permco_gvkey[ permco_gvkey > 1 ] # 이건 있다. 이건 말이 되나? 

# %% [markdown]
# ```
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
# ```

# %%
df.sort_values(by=['gvkey', 'permno', 'YEAR', 'datadate',], inplace=True)

# %%
len(df)

# %%
df = df.groupby(['gvkey', 'permno', 'YEAR', 'datadate']).last().reset_index()

# %%
len(df) # NOTE: The data set WORK.BE has 263854 observations and 6 variables.

# TODO: 잘못나온다. 너무 많이 짤렸다. 156741

# %% [markdown]
# ## SAS 5
#
# Construct ME and return data (delisting adjusted)

# %% [markdown]
# ### delisting returns

# %%
CRSP_M_df


# %%
def process_delisting_returns(row):
    DLRET = row['DLRET']
    DLSTCD = row['DLSTCD']

    loss30_codes = [500, 520] + list(range(551, 574)) + [574, 580, 584] # -30%, other values는 -100%
    # TODO: 하다 말고 잔다. 이어서 하기. 

# %%

# %% [markdown]
# ## SAS 6
#
# Merge BE and ME with return data

# %%
