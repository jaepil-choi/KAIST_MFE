# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
# ---

# %%
from functools import reduce

# %%
# EDA asset dataframe

# Hypothesis: Five cases of nan values in the dataframe
# 1. The first rows of the series has nan values
c0 = [1, 1, 0, 0, 0]

# 2. The last rows of the series has nan values
c1 = [0, 0, 0, 1, 1]


# 3. The middle rows of the series has nan values
c2 = [0, 1, 1, 0, 0]

# 4. The first few and last few rows of the series has nan values
c3 = [1, 1, 0, 0, 1]

# 5. No nan values in the series
c4 = [0, 0, 0, 0, 0]


# %%
def reduce_same(result, x):
    if result[-1] == x:
        return result
    else:
        result.append(x)
        return result


# %%
reduce(reduce_same, c0, [-1])


# %%
def make_unique_case_df(df):
    unique_case_df = pd.DataFrame()
    cases = {
        (-1, 1, 0): 0, # 존재하지 않았다가 나타난 경우
        (-1, 0, 1): 1, # 존재했다가 사라진 경우
        (-1, 0, 1, 0): 2, # 존재했다가 사라지다가 다시 나타난 경우
        (-1, 1, 0, 1): 3, # 존재하지 않았다가 나타나다가 사라진 경우
        (-1, 1): 4, # 계속 존재하지 않은 경우
        (-1, 0): 5, # 계속 존재하는 경우
    }
    case_satisfied = [0] * len(cases.keys())
    result_cases = []

    for col in df.columns:
        series = df[col]
        isna = (series.isna() * 1).tolist()

        case_tuple = tuple(reduce(reduce_same, isna, [-1]))
        
        assert case_tuple in cases.keys() , f"Case tuple {case_tuple} not in cases"

        if case_satisfied[cases[case_tuple]] != 1:
            unique_case_df[col] = series
            case_satisfied[cases[case_tuple]] = 1
            result_cases.append(cases[case_tuple])
        else:
            if sum(case_satisfied) == len(cases.keys()):
                break
            else:
                continue
        
    return unique_case_df, result_cases


# %%
uc_df, _ = make_unique_case_df(asset)

# %%
uc_df


# %%
def shorten_nan_cases(unique_case_df, minimum_rows_per_case=1):
    isna_df = unique_case_df.isna()

    unique_rows_df = isna_df.drop_duplicates()
    rowwise_cases = [tuple(c.values()) for c in list(unique_rows_df.T.to_dict().values())]
    cases_mapping = {case: i for i, case in enumerate(rowwise_cases)}

    case_satisfied_count = np.zeros(len(cases_mapping))

    if isinstance(minimum_rows_per_case, int):
        case_minimum_counts = np.ones(len(cases_mapping)) * minimum_rows_per_case
    elif isinstance(minimum_rows_per_case, list):
        assert len(minimum_rows_per_case) == len(cases_mapping), f"Required cases: {len(cases_mapping)} != Input cases: {len(minimum_rows_per_case)}"
        case_minimum_counts = minimum_rows_per_case
    
    boolmask_df = pd.DataFrame()
    
    already_sampled = [] # Assuming unique time-series index

    while not (case_satisfied_count >= case_minimum_counts).all():
        sample_row = isna_df.sample(1)

        if sample_row.index[0] in already_sampled:
            continue
        else:
            already_sampled.append(sample_row.index[0])
        
        row_case = tuple(sample_row.values[0])
        case_idx = cases_mapping[row_case]

        if case_satisfied_count[case_idx] >= case_minimum_counts[case_idx]:
            continue
        else:
            case_satisfied_count[case_idx] += 1
            boolmask_df = pd.concat([boolmask_df, sample_row], axis=0)
        
    result_df = unique_case_df.reindex(index=boolmask_df.index, columns=boolmask_df.columns).sort_index()

    return result_df


# 주의: 경계값을 생각하지 않음. 

# %%
small_asset = shorten_nan_cases(uc_df, minimum_rows_per_case=[3, 3, 2, 1])
small_asset

# %%
# small_asset = asset.iloc[:10, :5].copy()
# small_asset.iloc[:7, 0] = np.nan
# small_asset.iloc[3:5, 1] = np.nan
# small_asset.iloc[-7:, 2] = np.nan

# small_asset # 극단적인 케이스 가정하여 만듦

# %%
small_asset_2d = small_asset.to_numpy()
small_asset_2d


# %%
small_asset_2d.shape

# %%
small_X = np.log(small_asset_2d[1:]/small_asset_2d[:-1])
small_X

# %%
small_X.shape

# %%
# to address nan values, use pandas for covariance matrix

small_X_df = pd.DataFrame(small_X, columns=small_asset.columns)

# %%
small_X_df.cov(min_periods=1)

# %%
small_Q = small_X_df.cov(min_periods=2).to_numpy()
small_Q

# %%
np.isnan(small_Q).sum()

# %%
small_Q.shape

# %%
small_r = np.nanmean(small_X, axis=0)
small_r = small_r.reshape(-1, 1)
small_r.shape


# %%
small_r

# %%
small_l = np.ones(small_r.shape)
small_l

# %%
small_l.shape

# %%
small_Q_l_r = np.hstack([small_Q, small_l, small_r])
small_Q_l_r

# %%
small_Q_l_r.shape

# %%
small_l_0_0 = np.hstack([small_l.T, [[0]], [[0]]])
small_l_0_0

# %%
small_l_0_0.shape

# %%
small_r_0_0 = np.hstack([small_r.T, [[0]], [[0]]])
small_r_0_0

# %%
small_r_0_0.shape

# %%
small_L = np.vstack([small_Q_l_r, small_l_0_0, small_r_0_0])
small_L

# %%
small_L.shape

# %%
small_zero = np.zeros(small_l.shape)
small_zero

# %%
small_zero_l_mu = np.vstack([small_zero, [[0]], [[MU]]])
small_zero_l_mu

# %%
small_L_inv = np.linalg.inv(small_L)
small_L_inv.shape

# %%
small_L_inv

# %%
small_w_lmda1_lmda2 = small_L_inv @ small_zero_l_mu
small_w_lmda1_lmda2

# %%
small_w = small_w_lmda1_lmda2[:-2]
small_lmda1 = small_w_lmda1_lmda2[-2]
small_lmda2 = small_w_lmda1_lmda2[-1]

# %%
small_var = small_w.T @ small_Q @ small_w
small_var
