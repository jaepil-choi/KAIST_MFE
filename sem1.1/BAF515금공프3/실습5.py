# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
# ---

# %% [markdown]
# # 블랙 리터만 모델
#
#

# %% [markdown]
# - market portfolio 를 optimal tangent portfolio로 친다. 
# - risk aversion factor를 sharpe ratio로 쓴다. 
#     - 근데 excess return이 아니라 그냥 return / std
#     - "market price of risk": 한음이가 파생에서 발표했던 것과 같음. 
# - `asset_returns`: 
#     - level 2 DM 형태로 되어있음. 
#     - 자산은 마치 GAPS 처럼 bond, large cap growth, large cap value 등등으로 나눠짐. 
# - `excess_asset_returns`: 
#     - 2d에서 1d 수익률 빼줘야 하니까 `np.newaxis` 로 맞춰줘야 broadcasting 가능. 
# - var-covar matrix 구할 때 
#     - `rowvar=False` 꼭 해줘야 함. numpy에서 default는 row를 변수라고 취급해놓음. 
# - global return, std를 행렬 연산으로 구해줌. 
#     - 코드 잘못됐다. `global_sd`에서 sqrt 안씌워줬네. 
# - Black Litterman의 전망을 반영하기 위해 
#     - `Q`: 전망치
#     - `P`: 
#         - absolute return이면 그냥 1 적어주고 
#         - over/underperform의 relative 한 전망이면 1, -1 와 같이 합이 1이 되게 만들어 줌. 
#     - tau: tuning constant
#         - 예측값과 모델값 사이의 weight를 나타냄. ( w vs (1-w) )
#         - 이게 높으면 예측값에 더 무게를 싣는 것. 
#     - Omega: 
#         - 전망 하나하나에 대한 불확실성. 
# - 최종적으로 BL return vector를 계산할 때는
#     - $ E[R] = ( (\tau \Sigma)^{-1} + P^T \Omega^{-1} ?? ... ) $ 
# - 그 다음 최초의 asset weight와 BL 모델을 반영한 asset weight를 비교해보자. 
