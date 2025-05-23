# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="GnuloPwZugnF"
# # 과제 3

# %% [markdown] id="lnAXr_5HuDO_"
# 이 문제 세트에서는 초과 수익률의 또 다른 잠재적 위험 원인/설명인 우량성(quality)을 살펴봅니다.
#
# 본과제를 위해 다음 스프레드 시트가 포함된 Problem_Set_3Data.xls를 사용합니다.
#
# 1) 49개 시가가중 산업 포트폴리오의 월별 수익률;
# 2) 파마-프렌치 포트폴리오의 월별 수익률 RMRF, SMB, HML, UMD, CMA, RMW 및 rf;
# 3) 베타에 대한 베팅(BAB) 팩터, 우량주 마이너스 정크주( QMJ) 팩터의 월간 수익률
#

# %% [markdown] id="HeOYZIIWuHOJ"
# # 1부: 개념:
#
# 1.	우량성(quality)이란 무엇인가요?  어떻게 정의할 수 있을까요? 자산 기대 수익률에 영향을 미친다고 생각하시나요?  이렇게 당연한 것일까요, 아니면 놀라운 것일까요?
#
# 2.	우량성은 어떻게 측정하나요?  우량성을 측정할 수 있는 확실한 방법이 있나요?
#
#
# 3.	다음에서 사용하는 우량성 척도를 비교하고 대조하십시오.
# a.	Robert Novy-Marx (The Other side of Value)
# b.	Frazzini and Pedersen (Betting Against Beta)
# c.	Asness, Frazzini, Pedersen (Quality Minus Junk)
# d.	Fama, French (A five-factor asset pricing model)
# 각각 '우량성'이라는 개념을 어떻게 측정하나요?  어떤 측정법이 더 합리적이라고 생각하시나요?  이러한 측정법이 수익률을 예측해야 하는 이유는 무엇인가요?  여러분의 설명이 효율적인 시장과 일치합니까, 아니면 일치하지 않습니까?
#

# %% [markdown] id="50hbfPnj7qJR"
# 1. 우량성(quality)은 투자자에 따라 여러 가지로 정의될 수 있지만 일반적으로 "다른 모든 것이 동일할 때 기꺼이 높은 가격을 지불할 의사가 있는 것"을 의미합니다. 우량성에 대한 몇 가지 정의는 다음과 같습니다:
#
#   * 수익성
#   * 안전함
#   * 좋은 지비구조
#   * 양호한 성장성(자산, 수익 등의 측면이 될 수 있음)
#   * 높은 배당금
#   * 좋은 신용도
#   * 좋은 경영
#   * 안정성
#
#  우량성은 자산 기대 수익률에 영향을 미쳐야 합니다. 예를 들어, 투자자는 자산의 안전성에 대한 대가(예: 낮은 기대 수익률)를 지불해야 합니다. 반면에 수익성이 좋은 자산은 기대수익률에 긍정적인 영향을 미칠 수 있습니다. 그런 의미에서 그러한 "가격"이 무엇인지, 즉 어떤 방향과 규모여야 하는지는 명확하지 않습니다.
#
# 2. 우량성은 수익성, 투자자의 지불 의향, 주식의 베타 또는 위험성으로 측정할 수 있습니다. 전반적으로 투자자는 수익성, 성장성, 배당금 전망이 높은 주식에 더 많은 가격을 지불할 의향이 있으며, 인지된 위험 수준, 즉 베타가 낮은 주식에 더 많은 가격을 지불할 의향이 있습니다. 베타가 낮은 주식은 덜 위험하고 우량 주식일 가능성이 높습니다.
#
# 3.
#  **a. Robert Novy-Marx (The Other side of Value):**
#
#  **우량성 측정법:**
#
#  총수익성= (매출-비용 또는 COGS)/자산
#
#  노비-마르크스는 총이익율을 사용하여 우량성을 측정합니다.
#
#  총이익/자산 = 매출/자산 * 총이익/매출
#
#  노비-마르크스는 높은 이익을 내는 기업과 낮은 이익을 내는 기업을 비교한 결과, 수익성이 높은 기업이 수익성이 낮은 기업보다 투자자에게 더 높은 수익률을 안겨준다는 사실을 발견했습니다. 또한 총이익률이 수익률의 강력한 예측 변수라는 사실도 발견했습니다.
#
#  **수익률 예측:**
#    
#  특히 장부가 대비 시장가치를 통제할 때 자산 대비 총이익은 수익률의 강력한 예측 변수입니다. 수익성이 높은 기업은 수익성이 낮은 기업보다 평균 수익률이 높은 경향이 있습니다.
#
# 수익성은 모멘텀과 가치에 비해 수익률의 중요한 동인임이 입증되었습니다. 또한 수익성으로 측정된 우량성은 평균 수익률의 횡단면을 결정하는 중요한 요인입니다.   
#
#  **b. Frazzini and Pedersen (Betting Against Beta):**
#
#  **우량성 측정:**
#
# 우량주와 저베타 주식 모두 위험이 낮은 경향이 있으므로 저베타 주식은 우량성에 대한 좋은 예측 변수입니다. 따라서 우량성과 베타는 반비례하므로 베타가 낮은 주식은 우량성이 높은 자산의 척도가 될 수 있습니다.
#
#    **수익률 예측:**
#
# 실제로 베타가 높은 주식은 CAPM이 예측하는 수익률보다 낮은 수익을 창출합니다. 즉, 이 논문은 CAPM과 반직관적인 결론을 내리고 있습니다. 따라서 우리는 CAPM이 말하는 것과는 반대로 고베타 주식보다 위험 단위당 더 높은 수익을 창출하는 저베타 주식에 투자해야 합니다.
#
#  **c. Asness, Frazzini, Pedersen (Quality Minus Junk)**
#
#  **우량성 측정:**
#
# 우량성은 수익성, 성장성, 안전성, 배당성향 등 높은 가격을 의미하는 모든 변수로 측정됩니다.
#
# 고든의 성장 모델은 이러한 변수를 사용하여 주식의 장부 가치 대비 주가(P/B)를 구합니다.
#
# 예: P/B = (이익/장부가치 * 배당금/이익 또는 배당 성향)/(필요 수익률 - 성장률)
#
#  수익성: 장부가액 단위당 이익
#
#  성장성은 이전 5년 성장률
#
#  투자자는 요구수익률이 낮은 주식에 더 많은 금액을 지불할 것이므로 안전성은 요구수익률을 통해 파악됩니다.
#
#  배당성향은 주주에게 지급되는 이익의 비율로 캡처됩니다.
#
#  각 변수는 z 점수로 표준화되어 품질 점수로 결합됩니다: 품질 = z(수익성 + 성장성 + 안전성 + 배당금)
#
#   **수익률 예측:**
#
#  우량 주식은 정크 주식보다 수익률이 훨씬 높습니다.
#
#  **d. Fama, French (A five-factor asset pricing model)**
#
#  **우량성 측정:**
#
# 수익성은 파마 & 프렌치 5요인 자산 가격 모델에서 우량성을 측정하는 척도입니다. 수익성 있는 기업을 매수하고 수익성 없는 기업을 매도하는 포트폴리오가 최적입니다. 이 논문에서는 RMW라는 개념을 사용합니다: "견고한 수익성에서 약한 수익성을 뺀 값".  
#
#    **수익률 예측:**
#
#  평균 주식 수익률은 수익성과 사후적으로 상관관계가 있습니다. 즉, 수익성을 우량성으로 측정할 때 수익성은 수익률을 설명할 때 강력한 예측력을 가질 수 있습니다.
#
# *******
#  노비-마르크스 측정법과 방정식이 가장 직관적으로 보입니다. 수익성은 우량성 및 위험 조정 수익률과 관련이 있습니다. 또한 베타가 높은 주식은 우량성이 낮다는 것을 나타내는 지표라는 것도 직관적입니다. 파마 & 프렌치 모델은 수익성을 포함하지만, 앞서 설명한 다른 변수보다 우량성에 더 중점을 두지는 않습니다.
#
# *******
# **설명이 효율적 시장과 일치합니까, 아니면 일치하지 않습니까?
# 효율적 시장에서는 높은 수익률이 높은 위험과 상관관계가 있습니다. 따라서 노비-마르크스 측정법은 기존의 효율적 시장 가정과 일치하지 않습니다.
#

# %% [markdown] id="v7pdeDtq_DDo"
# # 데이터 전처리

# %% id="YKW8krwE8C_d" executionInfo={"status": "ok", "timestamp": 1727253841924, "user_tz": -540, "elapsed": 653, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import warnings
warnings.filterwarnings('ignore')

# %% colab={"base_uri": "https://localhost:8080/"} id="djgY7d20F5Mu" executionInfo={"status": "ok", "timestamp": 1727253883020, "user_tz": -540, "elapsed": 41103, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}} outputId="bb6a8553-ad4a-43f6-b903-2a230e30ca9c"
from google.colab import drive
drive.mount('/content/drive')

# %% id="XBQFJWnoF5Cg" executionInfo={"status": "ok", "timestamp": 1727253883348, "user_tz": -540, "elapsed": 334, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
import os
os.chdir("/content/drive/MyDrive/2024년 카이스트 고급 금융 계량/과제3")

# %% id="cOCZrgB4_rwk"
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# %% id="4dM1Ts5i_FvF"
#--------------------Read subsheet 1--------------------
sheet1=pd.read_excel("Problem_Set_3Data.xlsx", sheet_name='49 Industry Portfolios',
                     skiprows=11, index_col=None, na_values=-99.99)
sheet1["Date"] = pd.to_datetime(sheet1["Unnamed: 0"], format="%Y%m")
industries = sheet1.set_index("Date")
industries = industries.loc[industries.index.dropna()]
industries = industries.drop("Unnamed: 0", axis = 1)

#--------------------Read subsheet 2--------------------
sheet1=pd.read_excel("Problem_Set_3Data.xlsx", sheet_name='BAB,QMJ',
                     index_col=None)
sheet1["Date"] = pd.to_datetime(sheet1["Unnamed: 0"], format="%Y%m")
factor = sheet1.set_index("Date")
factor = factor.loc[factor.index.dropna()]
factor = factor.drop("Unnamed: 0", axis = 1)


#--------------------Read subsheet 3--------------------
sheet1=pd.read_excel("Problem_Set_3Data.xlsx", sheet_name='FamaFrenchPortfolios',
                     skiprows=3, index_col=None)
sheet1["Date"] = pd.to_datetime(sheet1["Unnamed: 0"], format="%Y%m")
ff_port = sheet1.set_index("Date")
ff_port = ff_port.loc[ff_port.index.dropna()]
ff_port = ff_port.drop("Unnamed: 0", axis = 1)

# %% [markdown] id="qirB3tXvuJfh"
# # 2부: 49 산업 포트폴리오 데이터

# %% [markdown] id="hOMyPxk-uQHA"
# ## a. 시장 프록시와 무위험 이자율을 사용하여 각 포트폴리오의 β_m을 추정하시요.

# %% id="5v2HPffVDQ4i" outputId="97977597-aec1-466e-9e9d-2d5501d7cbd9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727246933958, "user_tz": -540, "elapsed": 15, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
market = ff_port[['Mkt-RF', 'RF']]
industries_1 = industries[:-2]
beta_1 = []

x = sm.add_constant(market['Mkt-RF'])

for name in industries_1.columns:
    y_temp = industries_1[name] - market['RF']
    mod = sm.OLS(y_temp, x, missing = 'drop').fit()
    beta_1.append(mod.params[1])

beta_1

# %% [markdown] id="739ygCpZuT2h"
# ## b. 지정된 기간 동안 각 포트폴리오의 기대 수익률을 계산하시오(즉, 기대 수익률을 추정하시오).

# %% id="mXT67zW0dJaB" outputId="9a061929-695e-4c3e-b6b3-b24754bd4109" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727246933958, "user_tz": -540, "elapsed": 12, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
return_1 = []

for name in industries_1.columns:
    return_temp = industries_1[name].mean()
    return_1.append(return_temp)

return_1

# %% [markdown] id="4GXzLnL_IkBk"
# ## c. 기대 수익률 대 β_m의 축에 데이터를 플롯하시오.

# %% id="xM0nuc26d3g6" outputId="8bfc5b75-a4ac-410c-9b71-e0d5122ba4ba" colab={"base_uri": "https://localhost:8080/", "height": 583} executionInfo={"status": "ok", "timestamp": 1727246935719, "user_tz": -540, "elapsed": 1770, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fig=plt.figure(figsize=(12,6))
plt.plot(beta_1, return_1, '.', markersize=10)
plt.xlabel('Beta_m')
plt.ylabel('Expected REturn');
plt.title('Part II c): Expected Return vs Industries Beta', fontsize=16)
plt.xlim(0, 2)
plt.ylim(0, 2)

# %% [markdown] id="fsCvZRs0wOFh"
#
# ## d. (CAPM과 같은) 이론이 사실이라고 가정했을 때 이 데이터에 대해 예상할 수 있는 모양인가요?
#

# %% [markdown] id="lyYbJ1H7d5pl"
# 이는 우리가 기대하는 형태가 아닙니다. 일반적으로 데이터는 낮은 $\beta$ 구간에서 기대 수익률과 $\beta$ 사이에 약한 양의 관계가 있는 것처럼 보이지만, 음의 관계는 아니더라도 상당히 평탄하다는 것을 알 수 있습니다. 실제로 이러한 관찰은 $\beta$가 증가함에 따라 특히 강해집니다. 즉, $\beta$가 증가함에 따라 기대 수익률은 평평해지거나 심지어 감소하는 것처럼 보입니다. 이는 $\beta$가 높을수록 기대 수익률이 높아진다는 CAPM과 같은 이론(즉, 높은 $\beta$와 높은 수익률 사이의 강한 선형 관계)과는 상반되는 결과입니다.

# %% [markdown] id="MoaUY-KKwPca"
# ## 이론적으로 예상되는 증권 시장선(SML)을 그리시오(참고: 선은 두 점으로 지정할 수 있으며, 알고 있는 SML의 두 점을 생각하라).
#
#

# %% id="-07x19IugON9" outputId="9d94621a-c653-4074-c96f-cf4cbc933c01" colab={"base_uri": "https://localhost:8080/", "height": 583} executionInfo={"status": "ok", "timestamp": 1727246935719, "user_tz": -540, "elapsed": 21, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fig=plt.figure(figsize=(12,6))
plt.plot([0, 2], [0, 2], '--')
plt.plot(beta_1, return_1, '.', markersize=10)
plt.xlabel('Beta_m')
plt.ylabel('Expected REturn');
plt.title('Part II e): Security Market Line', fontsize=16)
plt.xlim(0, 2)
plt.ylim(0, 2)

# %% [markdown] id="Gd46DZNUwQxt"
# ## f. c)에서 플롯한 데이터를 통해 가장 잘 맞는 선을 플롯하시오.  이 선은 e)의 이론적 선과 어떻게 비교되나요?  그 원인은 무엇이며 자산 가격 결정 모델에 어떤 영향을 미칠 수 있나요?
#

# %% id="jgFE0H_XhlJh" outputId="b18f21c5-2197-4b3f-91ff-dfee78583dd2" colab={"base_uri": "https://localhost:8080/", "height": 583} executionInfo={"status": "ok", "timestamp": 1727246937174, "user_tz": -540, "elapsed": 1472, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
fig=plt.figure(figsize=(12,6))
plt.plot([0, 2], [0, 2], '--')
plt.plot(beta_1, return_1, '.', markersize=10)
plt.plot(np.unique(beta_1), np.poly1d(np.polyfit(beta_1, return_1, 1))(np.unique(beta_1)), '--')
plt.xlabel('Beta_m')
plt.ylabel('Expected REturn');
plt.title('Part II f): Security Market Line and Best Fit Line', fontsize=16)
plt.xlim(0, 2)
plt.ylim(0, 2)

# %% [markdown] id="TFQUzbb7ikjj"
# 파트 d에서 설명한 것처럼 최적 적합선은 기대 수익률과 $\beta$ 사이에 음의 관계가 있으며, 이는 이론적 증권 시장선과 완전히 반대되는 것(우량성 이상현상)을 알 수 있습니다.
#
# 투자자들이 CAPM과 같은 자산 가격 모델을 압도적으로 채택한다면, 투자자들은 더 높은 수익률을 얻기 위해 더 높은 $\beta$ 자산에 투자하기를 원할 것입니다. 그러나 그렇게 되면 자산 가격이 상승하여 수익률이 낮아질 수 있으며, 이는 높은 $\beta$의 낮은 수익률 데이터를 설명합니다. 그 반대의 경우도 마찬가지인데, 투자자들이 낮은 $\beta$ 자산을 피하기 때문에 해당 자산의 가격은 지속적으로 저평가되지만 수익률은 높습니다. 이 플롯은 CAPM과 같은 자산 가격 결정 모델이 낮은 $\beta$ 자산은 저평가하고 높은 $\beta$ 자산은 과대평가할 수 있음을 보여줍니다.

# %% [markdown] id="_Ogy1vFxwSHD"
# ## 지. 실제 생활에서 어떤 종류의 제약이 이러한 효과를 초래할 수 있나요?
#
#

# %% [markdown] id="aFdecScPkzv5"
# 기대 수익률과 $\beta$ 사이의 평탄하거나 심지어 마이너스 관계를 초래할 수 있는 실제 제약 조건에는 차입 제약과 투자자의 레버리지 수준이 포함됩니다.

# %% [markdown] id="H1b9GjrQwUPE"
# ## h. CAPM이 올바른 모델이라고 생각한다면 그래프의 결과를 어떻게 활용하겠는가?
#

# %% [markdown] id="yor6pNcCmHn4"
# 이 결과를 활용하려면 낮은 $\beta$ 자산을 매수하고 높은 $\beta$ 자산을 매도하여 포트폴리오의 전체 수익률을 높일 수 있습니다.
#
# 즉, SML 위의 산업은 저평가되어 있고 SML 아래의 산업은 고평가되어 있으므로 고평가된 주식은 숏하고 저평가된 주식은 롱해야 합니다.

# %% [markdown] id="CTKolf_5ub5i"
# # 3부: BAB 팩터의 활용

# %% [markdown] id="CY2an7r8I5gC"
# ### 가. 다음과 같은 형태의 팩터 모델 테스트(횡단면 회귀 포함)를 고려합니다:
#
# $$R_{i}=\gamma_{0}+\gamma_{M} \beta_{i m}+\hat{\gamma}_{h m l} \beta_{i, h m l}+\gamma_{U M D} \beta_{i, U M D}+\gamma_{B A B} \beta_{i, B A B}+\eta_{i}$$
#
# 여기서 BAB는 파마/프렌치 스타일로 구성된 "베타에 베팅하는" 포트폴리오이며, 다른 팩터는 파마/프렌치에서 만든 것과 같습니다.

# %% [markdown] id="YGWkyUi7JCC8"
# ## b. 위의 모델을 사용하여 49개 포트폴리오에 대해 횡단면 회귀 테스트를 실행합니다.  파마-맥베스 스타일로 수행합니다:
# - 각 포트폴리오에 대해 β_i를 추정합니다. 여기서 β_i는 모든 설명 요인에 대한 계수의 벡터입니다.
# - 두 번째 단계: 각 월 t_i에 대해 R_i와 β_i의 회귀를 실행하여 주어진 각 월의 모든 γ를 계산합니다.
# - 이제 γ_i에 대한 시계열을 가져와 각 요인에 대한 t-stat, 표준 편차, 표준 오차, p 값 등을 계산합니다.

# %% id="d7Amfn4lunT9"
# Step 1
factors = ff_port[['Mkt-RF', 'HML', 'UMD']]
factors = pd.concat([factors, factor['BAB']], axis=1)
industries_2 = industries[:-2]

beta_m = []
beta_hml = []
beta_umd = []
beta_bab = []

x = sm.add_constant(factors)

for name in industries_2.columns:
  y_temp = industries_1[name] - market['RF']
  mod2 = sm.OLS(y_temp, x, missing = 'drop').fit()
  beta_m.append(mod2.params[1])
  beta_hml.append(mod2.params[2])
  beta_umd.append(mod2.params[3])
  beta_bab.append(mod2.params[4])

# %% id="akxvZuAP-qZJ"
# Step 2 and Step 3
d = {'beta_m': beta_m, 'beta_hml': beta_hml, "beta_umd": beta_umd, "beta_bab": beta_bab}
betas = pd.DataFrame(d)
betas.index = industries_2.columns
x = sm.add_constant(betas)
avg_return = pd.DataFrame(industries.mean())

mod3 = sm.OLS(avg_return, x).fit()

# %% [markdown] id="B6dYr4jkJF0G"
# ## c. 위의 각 위험 요인에 대한 위험 프리미엄 γ에 대한 p값, t-값 및 추정치를 보고하시오.

# %% id="jWncyx5C4RUW" outputId="bc7e0538-97a9-4d69-b934-6f172ab456e1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727246937175, "user_tz": -540, "elapsed": 14, "user": {"displayName": "\uc774\uae30\ud64d", "userId": "00707751831574141695"}}
print("coefficient->","gamma_0: ",round(mod3.params[0],2), "gamma_market: ",round(mod3.params[1],2),\
      "gamma_hml: ", round(mod3.params[2],2), "gamma_umd: ", round(mod3.params[3],2), "gamma_bab", round(mod3.params[4],2))
print("Standard Error->","gamma_0: ",round(mod3.bse[0],2), "gamma_market: ",round(mod3.bse[1],2),\
      "gamma_hml: ", round(mod3.bse[2],2), "gamma_umd: ", round(mod3.bse[3],2), "gamma_bab", round(mod3.bse[4],2))
print("t-stat->","gamma_0: ",round(mod3.tvalues[0],2), "gamma_market: ",round(mod3.tvalues[1],2),\
      "gamma_hml: ", round(mod3.tvalues[2],2), "gamma_umd: ", round(mod3.tvalues[3],2), "gamma_bab", round(mod3.tvalues[4],2))
print("p-value->","gamma_0: ",round(mod3.pvalues[0],2), "gamma_market: ",round(mod3.pvalues[1],2),\
      "gamma_hml: ", round(mod3.pvalues[2],2), "gamma_umd: ", round(mod3.pvalues[3],2), "gamma_bab", round(mod3.pvalues[4],2))
print("r-squared->", round(mod3.rsquared,2))

# %% [markdown] id="cxxZrzSGJJK7"
# ## d. 여기서 BAB 요인에 대한 노출이 기대 수익률의 변화를 설명하는 데 도움이 되나요?  이게 타당한가요?
#
#

# %% [markdown] id="xZy9t0b85WPo"
# 위의 요약 통계에서 상수 항을 제외하고 다른 $\gamma$ 팩터에 대한 p-값이 압도적으로 너무 높기 때문에 BAB 팩터에 대한 노출이 기대 수익률의 변화를 설명하는 데 도움이 되지 않는 것 같습니다. 이는 해당 팩터가 통계적으로 유의하지 않다는 것을 의미합니다. 이는 BAB가 SMB와 유사하고, 이전 숙제에서 보았듯이 규모 팩터가 기대 수익률을 설명하는 데 도움이 되지 않기 때문에 어느 정도 이해가 됩니다.
#
#

# %% [markdown] id="0XsourLUwTXL"
# ## e. 테스트 결과 중 마음에 들지 않는 부분이 있나요? $r^2$는 어떤가요? 마켓 베타의 계수는?

# %% [markdown] id="u0XJN-6NEe4l"
# 회귀에 대한 r-제곱도 매우 낮아 실제로 5%에 불과하며, 이는 이러한 요인을 합쳐도 기대 수익률의 5%만 설명할 수 있음을 나타내며, 이는 BAB가 없을 때보다 훨씬 낮아 보입니다. 또한 시장 베타 계수는 음수인데, 이는 수익률이 시장 수익률과 양의 관계를 가져야 한다는 직관에도 어긋나는 결과입니다.

# %% [markdown] id="EmDd9_g4JQ-l"
# # 4부: 우량성(quality) 비교

# %% [markdown] id="jhsIGcxEJTUA"
# ## a) 3부를 반복하되, BAB 대신 QMJ 계수에 대해 반복하시오.

# %% [markdown] id="9nhGh6K2JVwu"
# ## b) 3부를 반복하되, BAB 대신 RMW 계수에 대해 반복하시오.

# %% [markdown] id="5bdSBZ8SJZfX"
# ## c) QMJ 요인에 대한 결과가 BAB 요인에 대한 여러분의 믿음에 어떤 영향을 미칩니까?
#
#

# %% [markdown] id="OcY7HDG7Jdfw"
#
# ## d) 어떤 것이 더 나은 우량성(quality) 척도라고 생각하시나요? 그 이유는 무엇인가요?  동일한 효과를 얻을 수 있나요?
#
#

# %% [markdown] id="5RGQ0-HeJezn"
# ## e) 기대 수익률의 횡단면을 설명하는 데 어느 것이 더 도움이 된다고 생각하십니까?

# %% id="oRtSq7FoFuoo"
