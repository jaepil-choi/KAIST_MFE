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
# # 금공프3 중간대체과제 
#
# MFE 20249433 최재필

# %%
import numpy as np


# %% [markdown]
# ## 1. 채권 가격과 듀레이션
#
# ### (1)
#
# ** 문제에 제시된 공식을 아래와 같이 구현하였습니다. 
#
# - 채권 가격: 
#     - 마지막 기에 $ \frac{FV}{(1 + \frac{(y/100)}{f})^t} $ 더해줘야 함. (만기 원금)
#     - 이는 마지막 기의 $ C_n $ 에 포함됨. 
# - 듀레이션:
#     - $ t $가 아닌 $ \frac{t}{f} $ 를 곱해줘야 함. 
#     - 또한 채권 가격과 마찬가지로 만기 원금을 더해줘야 함. - 이 또한 마지막 기의 $ C_n $ 에 포함됨. 
#     
#  즉, $$ D = \frac{1}{P} ( \sum_{t=1}^n \frac{t}{f} \cdot \frac{C_t}{(1 + \frac{(y/100)}{f})^t} + \frac{n}{f} \cdot \frac{FV}{(1 + \frac{(y/100)}{f})^n} )$$
#
#  로 수정. 

# %%
def bondftn(facevalue, couprate, y, maturity, frequency):
    """계산된 채권가격과 듀레이션을 튜플로 반환하는 함수

    Args:
        facevalue (float): 액면가격
        couprate (float): 쿠폰이자율
        y (float): 만기수익률
        maturity (float): 만기
        frequency (float): 연간쿠폰지급횟수

    Returns:
        tuple: (채권가격, 듀레이션)
    """    
    frequencies = {
        'annual': 1,
        'semi-annual': 2,
        'quarterly': 4,
    }

    if frequency in frequencies:
        f = frequencies[frequency]
    else:
        print(f'Invalid frequency: {frequency}')
        return
    
    c = couprate / 100
    ytm = y / 100
    c_dollar = facevalue * c / f
    nper = maturity * f

    ## 채권 가격
    P = 0
    for t in range(1, nper+1):
        P += c_dollar / (1 + ytm/f)**t
    
    P += facevalue / (1 + ytm/f)**t

    ## 듀레이션
    D = 0
    for t in range(1, nper+1):
        D += t/f * ( c_dollar / (1 + ytm/f)**t )
    
    D += t/f * ( facevalue / (1 + ytm/f)**t )
    D = D/P
    
    return P, D
    


# %%
test_case = {
    'facevalue': 100,
    'couprate': 5,
    'y': 4.5,
    'maturity': 2,
    'frequency': 'quarterly',
}

# %%
bondftn(**test_case)


# %% [markdown]
# ### (2)

# %%
def price_change(facevalue, couprate, y_old, y_new, maturity, frequency):
    """만기수익률 변화에 따른 가격변화율을 계산하는 함수

    Args:
        y_old (float): 변화 전 만기수익률
        y_new (float): 변화 후 만기수익률

    Returns:
        float: 가격변화율
    """    
    old_price = bondftn(facevalue, couprate, y_old, maturity, frequency)[0]
    new_price = bondftn(facevalue, couprate, y_new, maturity, frequency)[0]

    return (new_price - old_price) / old_price


# %%
y_old = 10
y_new = 11
frequency = 'annual'
facevalue = 100

result_dict = {}

test_maturities = [5, 4, 3, 2, 1]
test_couprates = [5, 4, 3, 2, 1]

for m in test_maturities:
    result_dict[f'M={m}'] = {}
    for c in test_couprates:
        result_dict[f'M={m}'][f'{c}%'] = price_change(
            facevalue=facevalue, 
            couprate=c, 
            y_old=y_old, 
            y_new=y_new, 
            maturity=m, 
            frequency=frequency,
            )


# %%
result_dict 

# %%
result_dict['M=5']['5%']

# %% [markdown]
# ### (3)

# %%
result_dict_dur = {}

for m in test_maturities:
    result_dict_dur[f'M={m}'] = {}
    for c in test_couprates:
        result_dict_dur[f'M={m}'][f'{c}%'] = bondftn(
            facevalue=facevalue, 
            couprate=c, 
            y=y_old, 
            maturity=m, 
            frequency=frequency
            )[1]

# %%
result_dict_dur['M=5']['4%']

# %% [markdown]
# ## 2. 자동차 보험회사에 관한 몬테카를로 시뮬레이션

# %%
# poisson (연간청구건수)
poi_mean = 100

# gamma (청구건수 별 청구금액)
alpha = 2 # 모양
beta = 1/2 # 척도

# uniform (청구건수 별 청구발생시점)
start = 0
end = 1

# 보험료 수입
slope = 150

# %% [markdown]
# ### (1)

# %%
# 연간 청구 건수를 포아송 분포에서 샘플링
poisson_samples = np.random.poisson(lam=poi_mean, size=10000)

case_count = np.random.choice(poisson_samples, 1)[0]
case_count 

# %%
# 청구 건수별로 청구금액을 감마 분포에서 샘플링
claims = np.random.gamma(alpha, scale=beta, size=case_count)

# %%
# 청구 건수별 청구 발생시점을 균등 분포에서 샘플링
times = np.random.uniform(start, end, size=case_count)

# %%
sort_idx = np.argsort(times) # 시간순으로 정렬하기 위한 인덱스

claims_timeseries = claims[sort_idx]
times_timeseries = times[sort_idx]
revenue_timeseries = slope * times_timeseries # 보험료 수입

cumulative_claims_timeseries = np.cumsum(claims_timeseries) # 누적 청구금액
balance_timeseries = revenue_timeseries - cumulative_claims_timeseries # 누적 수입 - 누적 청구금액

# %%
balance = np.insert(balance_timeseries, 0, 0) # 첫 번째 값은 0으로 삽입
balance


# %% [markdown]
# ### (2)

# %%
def generate_balance_path(
        poisson_size=10000,
        poi_mean=100,
        alpha=2,
        beta=1/2,
        start=0,
        end=1,
        slope=150
        ):
    """Monte Carlo 실험을 위해 balance의 path를 generate하는 함수

    Returns:
        np.ndarray: 잔고의 path
    """    
    
    # 연간 청구 건수를 포아송 분포에서 샘플링
    poisson_samples = np.random.poisson(lam=poi_mean, size=poisson_size)
    case_count = np.random.choice(poisson_samples, 1)[0]

    # 청구 건수별로 청구금액을 감마 분포에서 샘플링
    claims = np.random.gamma(alpha, scale=beta, size=case_count)

    # 청구 건수별 청구 발생시점을 균등 분포에서 샘플링
    times = np.random.uniform(start, end, size=case_count)

    sort_idx = np.argsort(times) # 시간순으로 정렬하기 위한 인덱스

    claims_timeseries = claims[sort_idx]
    times_timeseries = times[sort_idx]
    revenue_timeseries = slope * times_timeseries # 보험료 수입

    cumulative_claims_timeseries = np.cumsum(claims_timeseries) # 누적 청구금액
    balance_timeseries = revenue_timeseries - cumulative_claims_timeseries # 누적 수입 - 누적 청구금액
    
    balance = np.insert(balance_timeseries, 0, 0) # 첫 번째 값은 0으로 삽입

    return balance


# %% [markdown]
# #### (a)

# %%
num_experiments = 10000

# 최종 balance만 generate
simulate_final_balance = [generate_balance_path()[-1] for _ in range(num_experiments)]

# %%
# balance의 기대값
np.mean(simulate_final_balance)

# %% [markdown]
# #### (b)

# %%
# balance path들을 generate
balance_paths = [generate_balance_path() for _ in range(num_experiments)]

# %%
# 1년 중 한 번 이상 -5 이하로 떨어질 확률
p = np.mean([np.any(balance <= -5) for balance in balance_paths])
p
