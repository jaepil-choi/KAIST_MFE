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
# # Factor 기반 Grouping 데이터 평가 방법론 제안
#
# Methodology to evaluate stock groupings based on common factor exposures
#
# KAIST 금융공학프로그램(MFE) 석사과정 최재필
#
# 논문 내용 구상 중

# %% [markdown]
# ## 초록
#
# S&P Global의 GICS, FnGuide의 WICS 등의 산업분류는 주식회사 사업보고서의 주요 매출원 등을 기준으로 판단하여 유사한 주식을 섹터/산업 등으로 묶는 grouping 데이터이다. 
#
# 기존 연구들에선 이런 Grouping 데이터 (섹터 등)을 더미변수로 사용해 주가/포트폴리오 수익률에 어느 정도의 factor loading을 가지는지를 주로 살펴보았다. 
#
# 이 연구에선 Grouping 데이터 자체에 집중하여 어떤 것이 "유사한" 주식인지, 어떻게 주식을 grouping해야 factor loading을 높일 수 있는지를 고찰한다. 
#
# 연구의 가설은 그룹이:
# - 그룹 내 주식들의 common factor exposure의 유사도가 높고
# - 그룹 외 주식들과 유사도가 낮을 때 
#
# 가장 효과적인 그룹이라는 것이다. 
#
# 이를 바탕으로 임의의 그룹 (섹터, 산업, 통계기반 클러스터링 등)이 주어졌을 때 밝혀진 150개+ 의 팩터를 이용하여 그룹이 얼마나 주식들을 잘 묶고 있는지 평가하는 방법론을 제안한다. 
#
# 또한 이 방법론의 evaluation metric을 직접 타겟해 의도적으로 유사한 factor exposure끼리 주식을 묶는 grouping을 만들어보고, 이 grouping의 residual을 이용했을 때  임의의 전략 (밸류, 모멘텀 등)에서 불필요한 팩터의 헷징을 통해 전략의 volatility를 얼마나 효과적으로 줄여주는지 검증한다. 

# %% [markdown]
# ## 목차
#
# 1. **서론**  
#    - 금융에서의 Grouping 데이터 배경  
#    - Factor Exposure 기반 Grouping 평가의 필요성  
#    - 연구 목적 및 기여  
#
# 2. **문헌 연구**  
#    - 기존 Grouping 데이터 개요 (예: GICS, WICS)  
#    - 기존 연구: Factor Loading과 Grouping 평가  
#    - 기존 연구의 한계  
#
# 3. **이론적 배경**  
#    - "효과적인 Grouping"의 정의  
#    - Grouping 효과성에 대한 가설:  
#      - 그룹 내 주식들의 Common Factor Exposure 유사도 높음  
#      - 그룹 외 주식들과의 유사도 낮음  
#
# 4. **제안된 방법론**  
#    - Grouping 효과성 평가를 위한 메트릭  
#      - Factor Exposure 유사성 측정 지표  
#      - 통계적 모델 및 기법  
#    - 임의의 Grouping 데이터 평가 절차  
#      - 예시: 섹터, 산업, 통계 기반 클러스터링  
#
# 5. **실험 설계**  
#    - 데이터셋 및 Factor 선정 (예: 150개+ Factor)  
#    - 기존 Grouping 데이터 평가 절차  
#    - Factor Exposure를 기반으로 의도적으로 설계된 Grouping 생성  
#
# 6. **실증 분석**  
#    - 기존 Grouping 데이터(GICS, WICS 등)의 성과 분석  
#    - Factor 기반 Grouping과의 비교  
#    - 전략 변동성 감소 효과 분석  
#      - 밸류, 모멘텀 등 다양한 전략  
#
# 7. **결과 및 논의**  
#    - Grouping 효과성에 대한 주요 발견  
#    - Factor 기반 Grouping과 기존 Grouping의 비교  
#    - 포트폴리오 관리에 대한 시사점  
#
# 8. **결론**  
#    - 연구 결과 요약  
#    - 문헌에 대한 기여  
#    - 향후 연구 방향  
#
# 9. **참고문헌**  
#
# 10. **부록**  
#     - 추가 데이터 테이블  
#     - 상세 통계 테스트  
#     - 보충 분석  
#

# %% [markdown]
# ## 참고: Group Neutralization
#
# ### Group Neutralization의 APT 기반 설명
#
# APT에 근거하여 같은 반도체 그룹 내의 삼성전자와 하이닉스를 factor decomposition하면 다음과 같이 나타낼 수 있다. 
#
# $$
#
# R_{\text{삼전}} = \beta_{\text{mkt}, \text{삼전}} \cdot f_{\text{mkt}} + \beta_{\text{size}, \text{삼전}} \cdot f_{\text{size}} + \beta_{\text{value}, \text{삼전}} \cdot f_{\text{value}} + \beta_{\text{unknown}, \text{삼전}} \cdot f_{\text{unknown}} + \alpha_{\text{삼전}} + \epsilon_{\text{삼전}}
#
# \\
#
# R_{\text{하닉}} = \beta_{\text{mkt}, \text{하닉}} \cdot f_{\text{mkt}} + \beta_{\text{size}, \text{하닉}} \cdot f_{\text{size}} + \beta_{\text{value}, \text{하닉}} \cdot f_{\text{value}} + \beta_{\text{unknown}, \text{하닉}} \cdot f_{\text{unknown}} + \alpha_{\text{하닉}} + \epsilon_{\text{하닉}}
#
# $$
#
# 이런 방식으로 생각한다면 반도체 그룹 내의 평균 수익률은 아래와 같이 나타낼 수 있다. 
#
# 이 때, $ E(\epsilon) = 0 $ 이므로 제거된다. 
#
#
# $$
#
# \bar{R}_{\text{반도체}} = \bar{\beta}_{\text{mkt}} \cdot f_{\text{mkt}} + \bar{\beta}_{\text{size}} \cdot f_{\text{size}} + \bar{\beta}_{\text{value}} \cdot f_{\text{value}} + \bar{\beta}_{\text{unknown}} \cdot f_{\text{unknown}} + \bar{\alpha}
#
# $$
#
# 그렇다면 Group Neutralization을 적용하는 것은 아래 수식과 같이 이해할 수 있다. 
#
# $$
# R_{\text{삼전}} - \bar{R}_{\text{반도체}} = (\beta_{\text{mkt}, \text{삼전}} - \bar{\beta}_{\text{mkt}}) \cdot f_{\text{mkt}} + (\beta_{\text{size}, \text{삼전}} - \bar{\beta}_{\text{size}}) \cdot f_{\text{size}} + (\beta_{\text{value}, \text{삼전}} - \bar{\beta}_{\text{value}}) \cdot f_{\text{value}} \\
# + (\beta_{\text{unknown}, \text{삼전}} - \bar{\beta}_{\text{unknown}}) \cdot f_{\text{unknown}} + (\alpha_{\text{삼전}} - \bar{\alpha})
#
# \\
#
# R_{\text{하닉}} - \bar{R}_{\text{반도체}} = (\beta_{\text{mkt}, \text{하닉}} - \bar{\beta}_{\text{mkt}}) \cdot f_{\text{mkt}} + (\beta_{\text{size}, \text{하닉}} - \bar{\beta}_{\text{size}}) \cdot f_{\text{size}} + (\beta_{\text{value}, \text{하닉}} - \bar{\beta}_{\text{value}}) \cdot f_{\text{value}} \\
# + (\beta_{\text{unknown}, \text{하닉}} - \bar{\beta}_{\text{unknown}}) \cdot f_{\text{unknown}} + (\alpha_{\text{하닉}} - \bar{\alpha})
# $$
#
# 즉, 각 종목의 수익률에서 그룹의 평균수익률을 빼는 행위는 단순해 보여도 각 factor의 beta 역시 그룹의 평균만큼 빼주는 효과가 있기 때문에 팩터를 어느 정도 헷징하는 효과를 기대할 수 있는 것이다. 
#
# 또 하나의 효과는, 이미 알려진, known factor가 아닌 unknown (common) factor에 대해서도 이 방법을 통해 헷징이 가능하다는 점이다. 
#
# ### Group Neutralization을 이용한 그룹의 팩터 헷징 효과 분석 방법
#
# 위의 맥락에서 $ R_{\text{삼전}} - \bar{R}_{\text{반도체}} $ 를 factor regression 하여 얻어지는 beta는 demaned-beta라고 생각할 수 있다. 
#
# 만약 어떤 임의의 그룹 데이터 (예: GICS 섹터) 가 factor exposure가 유사한 종목들을 잘 묶어준다면 demeaned-beta 분포의 분산이 작을 것이고, 그룹이 factor exposure가 유사한 종목끼리 묶어주지 못한다면 (예: 무작위 그룹) demeaned-beta 분포의 분산이 클 것이다. 
#
# (논리 생각 중) 따라서 이 demeaned-beta (또는 Z-normalized beta)의 분포를 통해 어떤 그룹 데이터가 얼마나 효과적으로 각 factor exposure를 헷징할 수 있는지 판단할 수 있다. 
#
# 임의의 그룹의 각 팩터별 헷징 효과를 측정할 수 있게되면 여러 그룹 데이터들(GICS, TICS, 테마주, correlation clustering 등) 중 어떤 데이터가 더 리스크 팩터 헷징에 있어 더 효과적인지 평가하고 비교하는 것도 가능할 것이다. 

# %% [markdown]
#
