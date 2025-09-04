# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

#PEPSI 13856 / COCA-COLA 11308
prices = (1+sp500[['13856', '11308']]).cumprod()["2000":"2020"]
prices.columns = ['PEPSI', 'COCA']

#Cointegratiion
import statsmodels.api as sm

def engle_granger_test(y, x):
    # 1단계: 공적분 회귀 수행
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    residuals = results.resid

    adf_test = adfuller(residuals)

    return {'alpha': results.params.iloc[0], 
            'beta': results.params.iloc[1], 
            'p_value': adf_test[1],
            'residuals': residuals}

# 예시: PEPSI와 COCA-COLA 주가의 공적분 테스트
y = np.log(prices['PEPSI'])  # 로그 변환
x = np.log(prices['COCA'])   # 로그 변환

# 공적분 테스트 수행
test_results = engle_granger_test(y, x)

# 결과 출력
print("=== Engle-Granger Cointegration Test ===")
print(f"p-value: {test_results['p_value']:.4f}")
print("Coint Equation:", end=" ")
print(f"log(PEPSI) = {test_results['alpha']:.4f} + {test_results['beta']:.4f} * log(COCA)")


# %%
#coint 함수 사용
print("=== coint 함수 사용 ===")
coint_results = coint(y, x) # statsmodels 에 있는 공적분 검정 함수 (Engel Granger)
print(f"p-value: {coint_results[1]:.4f}")


# %%
# 시각화
residuals = test_results['residuals']
plt.figure(figsize=(15, 10))

# subplot 1: 원본 시계열
plt.subplot(211)
plt.plot(y, label='log(PEPSI)')
plt.plot(x, label='log(COCA)')
plt.title('Log Prices')
plt.legend()
plt.grid(True)

# subplot 2: 잔차 (공적분 관계)
plt.subplot(212)
plt.plot(residuals, label='Residuals')
plt.title('Cointegration Residuals')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# %%
#ACF, PACF 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot ACF  
plot_acf(residuals, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')

# Plot PACF
plot_pacf(residuals, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


# OLS 추정을 위한 데이터 준비
Y = residuals.iloc[1:]  # t 시점 데이터
X = residuals.iloc[:-1]  # t-1 시점 데이터
X = sm.add_constant(X)

# OLS 모델 적합
model = sm.OLS(Y.values, X.values)
results = model.fit()

# 파라미터 추정값
phi0_est = results.params[0]  # 상수항
phi1_est = results.params[1]  # AR(1) 계수
sigma_est = np.sqrt(results.mse_resid)  # 잔차의 표준편차

dt = 1/252
print(f"phi0_est: {phi0_est:.4f}, phi1_est: {phi1_est:.4f}, sigma_est: {sigma_est:.4f}")
print(f"mean-reversion: {(1-phi1_est)/dt:.4f}")
print(f"half-life: {np.log(2)/((1-phi1_est)/dt):.4f}")

# 결과 해석
# phi0_est: 0.0001, phi1_est: 0.9958, sigma_est: 0.0124
# mean-reversion: 1.0656
# half-life: 0.6505
# mean reversion parameter가 1 정도 나온다. half-life로 환산해본 것. 

# %%
