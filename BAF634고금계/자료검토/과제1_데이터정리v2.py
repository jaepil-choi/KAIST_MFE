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
# # 데이터 정리 v2
#
# - long data를 기본으로, panel로도 불러올 수 있도록 처리 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %% [markdown]
# ## 데이터 로드

# %%
fn1 = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'


# %%
## 추출한 fnguide 데이터를 dataframe으로 전처리

def preprocess_dataguide_csv(
        fn_file_path, 
        cols=['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency',], # 날짜가 아닌 컬럼들
        skiprows=8, 
        encoding="cp949",
        ):
    fn_df = pd.read_csv(fn_file_path, encoding=encoding, skiprows=skiprows, thousands=",")
    fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")

    return fn_df


# %%
fn1_df = preprocess_dataguide_csv(fn1, encoding='utf-8')

# %%
fn1_df['Item Name '].unique()

# %%
symbol_to_name = fn1_df[['Symbol', 'Symbol Name']].drop_duplicates().set_index('Symbol').to_dict()['Symbol Name']
name_to_symbol = {v:k for k, v in symbol_to_name.items()}

# %%
# string value를 가진 FnGuide Sector의 경우 pivot_table이 안됨. 
# 이래서 차라리 FnGuide Sector Code 로 가져오는 것이 훨씬 유용한듯. 

sectors = fn1_df[ fn1_df['Item Name '] == 'FnGuide Sector' ].pivot(
    index=['date', 'Symbol', 'Symbol Name', 'Kind', 'Frequency',],
    columns='Item Name ',
    values='value',
).reset_index()


# %%
sectors

# %%
sectors[ sectors['FnGuide Sector'] == '금융']

# %%
sectors.groupby('date').count()['FnGuide Sector']


# %%
sectors.groupby('date').size()

# %%
new_df = fn1_df.pivot_table(
    index=['date', 'Symbol', 'Symbol Name', 'Kind', 'Frequency',],
    columns='Item Name ',
    values='value',
    aggfunc='first',
    dropna=True, # False 로 하면 memory error 남. 
)

# %%
new_df.reset_index(inplace=True)
new_df.index.name = None

# %%
new_df.columns

# %%
for col in new_df.columns:
    try:
        new_df[col] = new_df[col].replace(',', '', regex=True).infer_objects()
        new_df[col] = pd.to_numeric(new_df[col]) # Catch exception explicitly
    except:
        pass

# %%
new_df.info()

# %%
new_df

# %%
new_df.groupby('date')['수익률 (1개월)(%)'].count()

# %%
existing = new_df.groupby('Symbol').filter(
    lambda x: x['종가(원)'].notnull().any()
)

univ_list = existing['Symbol'].unique()

# %%
len(univ_list)

# %%
new_df.pivot_table(
    index='date',
    columns='Symbol',
    values='종가(원)',
)

# %% [markdown]
# 모듈화한 것 테스트

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
fn1 = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'

# %%
# from fndata import FnData
from fndata import FnStockData

# %%
# fnd = FnData(fn1)
fnd = FnStockData(fn1)

# %%
item = '종가(원)'

# %%
fnd.get_data()

# %%
fnd.get_data(item)

# %%
fnd.get_data().info()

# %%
multi_items = ['종가(원)', '수익률 (1개월)(%)']

# %%
fnd.get_data(multi_items)

# %%
fnd.long_format_df

# %% [markdown]
# ## 디버깅

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
fn1 = DATA_DIR / '고금계과제1_v3.3_201301-202408.csv'

# %%

NUMERIC_DATA = [
    '종가(원)',
    '수정주가(원)',
    '수정계수',
    '수익률 (1개월)(%)',
    # '상장주식수 (보통)(주)',
    # '시가총액 (상장예정주식수 포함)(백만원)',
    # '시가총액 (보통-상장예정주식수 포함)(백만원)',
    '기말발행주식수 (보통)(주)',
    '보통주자본금(천원)',
    '자본잉여금(천원)',
    '이익잉여금(천원)',
    '자기주식(천원)',
    '이연법인세부채(천원)',
    '매출액(천원)',
    '매출원가(천원)',
    '이자비용(천원)',
    '영업이익(천원)',
    '총자산(천원)'
    ]

UNIV_REFERENCE_ITEMS = [
    '수정주가(원)',
    '종가(원)',
    '수익률 (1개월)(%)',
    '수익률 (%)'
    ]

DIV_BY_100 = [
    '수익률 (%)',
    '수익률 (1개월)(%)',
    ]

MULTIPLY_BY_1000 = [
    '보통주자본금(천원)',
    '자본잉여금(천원)',
    '이익잉여금(천원)',
    '자기주식(천원)',
    '이연법인세부채(천원)',
    '매출액(천원)',
    '매출원가(천원)',
    '이자비용(천원)',
    '영업이익(천원)',
    '총자산(천원)',
    ]

FN_INDEX_COLS = ['date', 'Symbol', 'Symbol Name', 'Kind', 'Frequency',]

# %%
import pandas as pd

# Constants (formerly class variables)
NUMERIC_DATA = [
    '종가(원)', '수정주가(원)', '수정계수', '수익률 (1개월)(%)',
    '기말발행주식수 (보통)(주)', '보통주자본금(천원)', '자본잉여금(천원)', '이익잉여금(천원)',
    '자기주식(천원)', '이연법인세부채(천원)', '매출액(천원)', '매출원가(천원)',
    '이자비용(천원)', '영업이익(천원)', '총자산(천원)'
]

UNIV_REFERENCE_ITEMS = [
    '수익률 (1개월)(%)',
]

DIV_BY_100 = [
    '수익률 (%)', '수익률 (1개월)(%)'
]

MULTIPLY_BY_1000 = [
    '보통주자본금(천원)', '자본잉여금(천원)', '이익잉여금(천원)', '자기주식(천원)', 
    '이연법인세부채(천원)', '매출액(천원)', '매출원가(천원)', '이자비용(천원)', 
    '영업이익(천원)', '총자산(천원)'
]

# FN_INDEX_COLS = ['date', 'Symbol', 'Symbol Name', 'Kind', 'Frequency']
FN_INDEX_COLS = ['date', 'Symbol', 'Symbol Name', ]

def melt_dataguide_csv(fn_file_path, cols=['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency'], skiprows=8, encoding="cp949"):
    fn_df = pd.read_csv(fn_file_path, encoding=encoding, skiprows=skiprows, thousands=",")
    fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")
    fn_df.drop(columns=['Kind', 'Item', 'Frequency'], inplace=True)
    return fn_df

def pivot_nonnumeric(fn1_df, item_name):
    nonnumeric_data = fn1_df[fn1_df['Item Name '] == item_name].pivot(
        index=FN_INDEX_COLS,
        columns='Item Name ',
        values='value'
    ).reset_index()
    return nonnumeric_data

def pivot_numerics(fn1_df):
    numeric_data = fn1_df.pivot_table(
        index=FN_INDEX_COLS,
        columns='Item Name ',
        values='value',
        aggfunc='first',
        dropna=True
    ).reset_index()
    return numeric_data

def preprocess_numerics(long_format_df):
    obj_cols = long_format_df.select_dtypes(include='object').columns
    obj_cols = [obj_col for obj_col in obj_cols if obj_col in NUMERIC_DATA]
    long_format_df[obj_cols] = long_format_df[obj_cols].replace(',', '', regex=True).infer_objects(copy=False)
    long_format_df[obj_cols] = long_format_df[obj_cols].apply(pd.to_numeric, errors='raise')
    return long_format_df

def make_filters(fn1_df):
    finance_sector = pivot_nonnumeric(fn1_df, 'FnGuide Sector')
    finance_sector = finance_sector[finance_sector['FnGuide Sector'] == '금융']

    is_under_supervision = pivot_nonnumeric(fn1_df, '관리종목여부')
    is_under_supervision = is_under_supervision[is_under_supervision['관리종목여부'] == '관리']

    is_trading_halted = pivot_nonnumeric(fn1_df, '거래정지여부') 
    is_trading_halted = is_trading_halted[is_trading_halted['거래정지여부'] == '정지']

    return [
        finance_sector,
        is_under_supervision,
        is_trading_halted,
    ]

def apply_filters(long_format_df, filter_dfs):
    for filter_df in filter_dfs:
        filter_df['_flag_right'] = 1
        long_format_df = long_format_df.merge(
            filter_df,
            on=['date', 'Symbol'],
            how='left',
            suffixes=('', '_right')
        )
        long_format_df = long_format_df[long_format_df['_flag_right'].isnull()] 
        long_format_df.drop(columns=[c for c in long_format_df.columns if c.endswith('_right')], inplace=True)
        long_format_df.reset_index(drop=True, inplace=True)
    return long_format_df

def get_univ_list(long_format_df, reference_item='수익률 (1개월)(%)'):
    assert reference_item in UNIV_REFERENCE_ITEMS, f"유니버스 구축을 위해 {UNIV_REFERENCE_ITEMS} 중 하나가 필요합니다."
    only_existing = long_format_df.groupby('Symbol').filter(
        lambda x: x[reference_item].notnull().any()
    )
    return only_existing['Symbol'].unique()

def get_wide_format_df(long_format_df, item_name):
    return long_format_df.pivot_table(
        index='date',
        columns='Symbol',
        values=item_name,
    )

def get_data(long_format_df, items, univ_list, item: list | str | None = None, multiindex: bool = True):
    if isinstance(item, str):
        assert item in items, f"{item} is not in the item list"
        assert item in NUMERIC_DATA, f"{item} is not a numeric data"
        data = get_wide_format_df(long_format_df, item)
        data = data.reindex(columns=univ_list)
        if item in DIV_BY_100:
            data = data / 100
        elif item in MULTIPLY_BY_1000:
            data = data * 1000
    elif isinstance(item, list):
        for i in item:
            assert i in items, f"{i} is not in the item list"
            assert i in NUMERIC_DATA, f"{i} is not a numeric data"
        data = long_format_df.loc[:, FN_INDEX_COLS + item]
        for col in data.columns:
            if col in DIV_BY_100:
                data[col] = data[col] / 100
            elif col in MULTIPLY_BY_1000:
                data[col] = data[col] * 1000
        if multiindex:
            data.drop(columns=['Symbol Name',], inplace=True)
            data.index.name = None
            data.set_index(['date', 'Symbol'], inplace=True)
        data = data.reindex(univ_list, level=1)
    elif item is None:
        data = long_format_df.copy()
        if multiindex:
            data.drop(columns=['Symbol Name',], inplace=True)
            data.index.name = None
            data.set_index(['date', 'Symbol'], inplace=True)
        data = data.reindex(univ_list, level=1)
    else:
        raise ValueError("""
                         item은 
                         - str (1개 item만 wide-format 반환) 
                         - list (선택한 item들 long-format 반환)
                         - None (전체 long-format 반환)
                         중 하나여야 합니다.
                         (numeric data만 선택 가능)
                         """)
    return data

def symbol_to_name(symbol_code, symbol_to_name_mapping):
    return symbol_to_name_mapping[symbol_code]

def name_to_symbol(symbol_name, name_to_symbol_mapping):
    return name_to_symbol_mapping[symbol_name]



# %%
fn1_df = melt_dataguide_csv(fn1, encoding='utf-8')
items = fn1_df['Item Name '].unique()

# %%
fn1_df.drop(columns=['Kind', 'Item', 'Frequency'], inplace=True)

# %%
items

# %%
long_format_df = pivot_numerics(fn1_df)


# %%
long_format_df[ long_format_df['Symbol'] == 'A000020']

# %%

long_format_df = preprocess_numerics(long_format_df)



# %%

# Apply filters: e.g., for 금융 제거, 관리종목여부, 거래정지여부
filter_dfs = make_filters(fn1_df)
long_format_df = apply_filters(long_format_df, filter_dfs)


# %%
univ_list = get_univ_list(long_format_df, '수익률 (1개월)(%)')
print(univ_list)


# %%
len(univ_list)

# %%
data = get_data(long_format_df, items, univ_list, item='수익률 (1개월)(%)') # wide는 정상작동

# %%
data = get_data(long_format_df, items, univ_list, item='이자비용(천원)') # wide는 정상작동

# %%
data.shape

# %%
data = get_data(long_format_df, items, univ_list, item=['수익률 (1개월)(%)', '이자비용(천원)'], multiindex=True) # long은 정상작동

# %%
data

# %%
item=['수익률 (1개월)(%)', '이자비용(천원)']
data = long_format_df.loc[:, FnData.FN_INDEX_COLS + item]

# %%
data.drop(columns=['Symbol Name', 'Kind', 'Frequency'], inplace=True)

# %%
data.index.name = None

# %%
univ = set(univ_list)
data_univ = set(data['Symbol'].unique())

# %%
univ - data_univ

# %%
data[ data[['date', 'Symbol']].duplicated() ]

# %%
check = data[ data['Symbol'] == 'A000020']
check

# %%
check['date'].value_counts()

# %%
check2 = fn1_df[ fn1_df['Symbol'] == 'A000020']
check2 = check2[ check2['Item Name '] == '수익률 (1개월)(%)' ]
check2

# %%
data.set_index(['date', 'Symbol'], inplace=True)

# %%
data

# %%
data.reindex(univ_list, level=1)

# %%

# %%

# %%
dup = long_format_df[ long_format_df[['date', 'Symbol']].duplicated() ][['date', 'Symbol']]
dup

# %%
long_format_df[  ]

# %%

# %%
right_df = filter_dfs[0].copy()
right_df['_flag'] = 1


# %%
right_df

# %%

dd = long_format_df.merge(
    right_df,
    on=['date', 'Symbol'],
    how='left',
    suffixes=('', '_right')
)

# %%
len(dd)

# %%
dd['_flag'].value_counts()

# %%
dd['_flag'].isnull().sum()

# %%
dd.columns

# %%
dd['_merge'].unique()

# %% [markdown]
# ## 시장수익률

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# %%
CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

# %%
fnmkt = DATA_DIR / '고금계과제_시장수익률_201301-202408.csv'

# %%
fn_df = pd.read_csv(
    fnmkt,
    encoding='utf-8', 
    skiprows=8, 
    thousands=","
    )

# %%
fn_df

# %%
cols=['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency',] # 날짜가 아닌 컬럼들

# %%
fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")

# %%
fn_df

# %%
FN_INDEX_COLS = ['date', 'Symbol', 'Symbol Name',]

numeric_data = fn_df.pivot_table(
    index=FN_INDEX_COLS,
    columns='Item Name ',
    values='value',
    aggfunc='first',
    dropna=True
).reset_index()

# %%
numeric_data

# %%
numeric_data.info()

# %%
NUMERIC_DATA = [
    '종가(원)',
    '수정주가(원)',
    '수정계수',
    '수익률 (1개월)(%)',
    # '상장주식수 (보통)(주)',
    # '시가총액 (상장예정주식수 포함)(백만원)',
    # '시가총액 (보통-상장예정주식수 포함)(백만원)',
    '기말발행주식수 (보통)(주)',
    '보통주자본금(천원)',
    '자본잉여금(천원)',
    '이익잉여금(천원)',
    '자기주식(천원)', 
    '이연법인세부채(천원)',
    '매출액(천원)',
    '매출원가(천원)',
    '이자비용(천원)',
    '영업이익(천원)',
    '총자산(천원)'
    ]


# %%
def _preprocess_numerics(numeric_data):

    obj_cols = numeric_data.select_dtypes(include='object').columns
    obj_cols = [obj_col for obj_col in obj_cols if obj_col in NUMERIC_DATA]
    numeric_data[obj_cols] = numeric_data[obj_cols].replace(',', '', regex=True).infer_objects(copy=False)
    numeric_data[obj_cols] = numeric_data[obj_cols].apply(pd.to_numeric, errors='raise') 
    
    return


# %%
_preprocess_numerics(numeric_data)

# %%
numeric_data.info()

# %%
numeric_data

# %%
long = numeric_data.copy()

long.drop(columns=['Symbol Name',], inplace=True)
long.index.name = None
long.set_index(['date', 'Symbol'], inplace=True)

# %%
long / 100

# %%
wide = numeric_data.pivot_table(
    index='date',
    columns='Symbol',
    values='수익률 (1개월)(%)',
)

# %%
wide

# %% [markdown]
#
# ## 무위험이자율

# %%
