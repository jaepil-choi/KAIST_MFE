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
# # BAF633 시뮬레이션 HW2 보고서
#
# 20249433 최재필
#
# - 사용한 분산감소기법 설명

# %% [markdown]
# ## Antithetic Variates

# %% [markdown]
# ```python
#     # Antithetic Variates
#     n_half = n // 2
#     z = np.random.randn(n_half, m)
#     z = np.vstack((z, -z))
# ```
#
# - 가장 쉬운 방법으로, Z와 -Z를 동시에 뽑아 난수 생성
# - payoff 함수를 거친 Z와 -Z로 만든 값들이 negative correlation이 있으면 분산감소효과 있음. 
#     - straddle에선 오히려 positive correlation이 있어 분산이 증가할 수 있고 
#     - K 아래의 평평한 payoff에선 그냥 cov = 0

# %% [markdown]
# ## Control Variates

# %% [markdown]
# ```python
#     # Control Variate: bs analytic price (ground truth)
#     vanilla_price = bsprice(s, k, r, q, t, sigma, option_flag)
#
#     # Vanilla payoff
#     vanilla_payoffs = np.maximum(callOrPut * (s_paths[:, -1] - k), 0)
#     disc_vanilla_payoffs = np.exp(-r * t) * vanilla_payoffs * total_lr
#
#     # Beta
#     cov_matrix = np.cov(disc_payoffs, disc_vanilla_payoffs)
#     beta = cov_matrix[0, 1] / cov_matrix[1, 1]
#     adjusted_payoffs = disc_payoffs - beta * (disc_vanilla_payoffs - vanilla_price)
# ```
#
# - 알려진 추정치에 대한 오류 정보를 활용하여 목표 시뮬레이션에서의 분산을 감소시킴
# - 알려진 추정치와 시뮬레이션하려는 상품과 correlation이 커야 효과적임
# - control variate를 s 대신 vanilla option payoff를 사용
#     - vanilla payoffs가 추정치의 시뮬레이션
#     - vanilla price가 bs 방정식으로 구한 해석적 ground truth  

# %% [markdown]
# ## 해봤는데 실패한 것들 
#
# - Brownian Bridge & In-Out Parity 세트
#     - 시도:
#         - Terminal price 분포에서 n개를 샘플링하고 
#         - 그 중 barrier out인 것들을 제거하고 
#         - 남은 샘플들과 initial price (S0)를 연결하는 line을 각각 만든 뒤 
#         - T/2에 위치한 mid point를 찾아서,
#         - 이 mid point의 가격을 중심으로 하는 정규분포 (표준편차는 모두 T/2 시점의 수준) 각각에서 1개씩 샘플링을 한 뒤 
#         - 이 mid point sample들 중 barrier out인 것들을 또 제거한 뒤, 
#         - 줄어든 initial price -- mid point price -- terminal price 점들의 set를 가지고 그 중간을 brownian bridge로 채워 path를 생성하려고 했음. 
#             - 여기서부터가 문제
#         - 생성한 path들만 대상으로 다시 barrier touch 여부를 simulation하여 최종 MCS discounted payoff를 만드려고 했음. 
#         - out-option을 만들었으므로 in-out-parity를 사용하면 in-option을 path 생성없이 구할 수 있음. (out-option 가격을 캐싱)
#     - 실패 이유:
#         - brownian bridge 특성상 두 점을 잇는 brownian path를 만드려면 m step을 for loop으로 iteratively 돌아야 하는 것 같음. 매우 비효율적.
# - Importance Sampling
#     - 시도:
#         - 주가가 배리어 b에 닿기 위한 shift(mu) 계산하여 Z를 shift
#         - 배리어 touch가 option payoff를 결정하기에 더 나은 결과가 나올 줄 알았음. 
#     - 실패 이유: 
#         - QuantLib Barrier 가격을 기준으로 벤치마크 결과 오히려 안 넣었을 떄보다 느려지고 bias, variance 모두 커짐. 

# %% [markdown]
#
