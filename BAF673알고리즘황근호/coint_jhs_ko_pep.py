# %%
import pandas as pd 
import numpy as np 
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
# vector_ar: VAR 모델, 그 안에 VECM이 포함되어 있음
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sp500 = pd.read_csv("./data/sp500_return.csv", parse_dates=['date'], index_col=0)
sp500_list = pd.read_csv("./data/sp500_list.csv", index_col=0, parse_dates=['start','ending'])
stock_id = pd.read_csv("./data/stock_id.csv", index_col=0, parse_dates=['namedt','nameendt'])

#PEPSI 13856 / COCA-COLA 11308
prices = (1+sp500[['13856', '11308']]).cumprod()["2000":"2020"]
prices.columns = ['PEPSI', 'COCA']
log_prices = np.log(prices)

#정상성 테스트
adf_result = adfuller(log_prices['PEPSI'])
print(f"PEPSI의 검정 결과: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

adf_result = adfuller(log_prices['COCA'])
print(f"COCA의 검정 결과: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

# %%
#VAR 모델 추정
var = VAR(log_prices, )
var_result = var.fit(maxlags=1)
display(var_result.summary())

A = var_result.params.iloc[1:,:].T
ev, evec = np.linalg.eig(A)
print("="*50)
print("Eigenvalues:")
print("lambda_1 = {:.4f}".format(ev[0]))
print("lambda_2 = {:.4f}".format(ev[1]))
print("\nEigenvectors:")
print("eigenvector_1 = ", evec[:, 0])
print("eigenvector_2 = ", evec[:, 1])
print("="*50)

# 결과:
# ==================================================
# Eigenvalues:
# lambda_1 = 0.9999
# lambda_2 = 0.9958

# Eigenvectors:
# eigenvector_1 =  [0.72946301 0.68402026]
# eigenvector_2 =  [-0.25293677  0.96748281]
# ==================================================



# %%
x = np.random.random((2,1000)) * 10 # 2차원 평면에 random 하게 점을 찍고, 
for i in range(50):
    x = A.values @ x
plt.plot(x[0,:], x[1,:], '.')
plt.xlabel("PEPSI")
plt.ylabel("COCA")
plt.grid() # 만약 장기 균형 있다면 모여야. 


x = np.random.random((2,1000)) * 10  
for i in range(500): # 500으로 올려보자. 
    x = A.values @ x
plt.plot(x[0,:], x[1,:], '.')
plt.xlabel("PEPSI")
plt.ylabel("COCA")
plt.grid() # 잘 모이는 것을 볼 수 있음. 


# %%
#Cointegration Test
res = coint_johansen(log_prices, 1, 0)

# 결과 출력
print("=== Johansen Cointegration Test ===")
print("\nTrace Statistics:")
print("-" * 70)
print(f"{'H0: r ≤':<10}{'Test Stat':>15}{'90%':>15}{'95%':>15}{'99%':>15}")
print("-" * 70)
for i in range(len(res.lr1)):
    print(f"{i:<10}{res.lr1[i]:>15.4f}{res.cvt[i, 0]:>15.4f}{res.cvt[i, 1]:>15.4f}{res.cvt[i, 2]:>15.4f}")

print("\nMax Eigenvalue Statistics:")
print("-" * 70)
print(f"{'H0: r ≤':<10}{'Test Stat':>15}{'90%':>15}{'95%':>15}{'99%':>15}")
print("-" * 70)
for i in range(len(res.lr2)):
    print(f"{i:<10}{res.lr2[i]:>15.4f}{res.cvm[i, 0]:>15.4f}{res.cvm[i, 1]:>15.4f}{res.cvm[i, 2]:>15.4f}")


# === Johansen Cointegration Test ===

# Trace Statistics:
# ----------------------------------------------------------------------
# H0: r ≤         Test Stat            90%            95%            99%
# ----------------------------------------------------------------------
# 0                 24.2707        16.1619        18.3985        23.1485
# 1                  5.1619         2.7055         3.8415         6.6349

# 0은 공적분 관계가 없다는 뜻. 
# 즉, 1개의 공적분 관계는 있을 것이다. 

# Max Eigenvalue Statistics:
# ----------------------------------------------------------------------
# H0: r ≤         Test Stat            90%            95%            99%
# ----------------------------------------------------------------------
# 0                 19.1088        15.0006        17.1481        21.7465
# 1                  5.1619         2.7055         3.8415         6.6349

# %%
# VECM 추정
print("\n=== VECM Estimation ===")
vecm = VECM(log_prices, k_ar_diff=1, coint_rank=1)
vecm_result = vecm.fit()
vecm_result.summary()


# %%
# VECM 결과 출력
print("\nVECM Results:")
print("-" * 50)
print("Cointegration Matrix (beta):")
print(vecm_result.beta)
print("\nAdjustment Matrix (alpha):")
print(vecm_result.alpha)

# VECM Results:
# --------------------------------------------------
# Cointegration Matrix (beta):
# [[ 1.        ]
#  [-0.95951432]]

# Adjustment Matrix (alpha):
# [[-0.00155177] 
#  [ 0.0020748 ]]

# %%
residuals = log_prices @ vecm_result.beta


plt.figure(figsize=(15, 10))

# subplot 1: 원본 시계열
plt.subplot(211)
plt.plot(log_prices['PEPSI'], label='log(PEPSI)')
plt.plot(log_prices['COCA'], label='log(COCA)')
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



# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot ACF  
plot_acf(residuals, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')

# Plot PACF
plot_pacf(residuals, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

adf = adfuller(residuals)
print(f"ADF Statistic: {adf[0]:.4f}, p-value: {adf[1]:.4f}")


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

# ADF Statistic: -3.9760, p-value: 0.0015
# phi0_est: -0.0000, phi1_est: 0.9964, sigma_est: 0.0117
# mean-reversion: 0.8977
# half-life: 0.7721

# mean reversion speed가 줄어서, 반감기가 좀 늘어남. 
# %%
