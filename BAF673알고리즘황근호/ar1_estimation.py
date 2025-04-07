#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ar1_process import ar1_process, random_walk
import statsmodels.api as sm
import seaborn as sns

# AR(1) 프로세스 생성
phi0 = 0.5
phi1 = 0.5
sigma = 1
T = 30 # 30개만 가지고 OLS estimatoin 하면 bias가 상당히 크다.
N = 3000

# AR(1) 프로세스 생성
phi0 = 0.5
phi1 = 0.9 # mean reversoin speed. 1에 가까울수록 느린 것. 
sigma = 1
T = 30 # 여전히 30개는 너무 적어서 bias 된 결과가 나온다. 
N = 3000

# AR(1) 프로세스 생성
phi0 = 0.5
phi1 = 0.9
sigma = 1
T = 300 # 300개로 늘려봤다. 아직 skew는 있지만...
N = 3000

## 결론: OLS 할 때 sample size 매우 중요. 

phi0_ests = np.zeros(N)
phi1_ests = np.zeros(N)
sigma_ests = np.zeros(N)

for i in range(N):
    y = ar1_process(phi0=phi0, phi1=phi1, sigma=sigma, T=T)

    # OLS 추정을 위한 데이터 준비
    Y = y[1:]  # t 시점 데이터
    X = y[:-1]  # t-1 시점 데이터
    X = sm.add_constant(X)

    # OLS 모델 적합
    model = sm.OLS(Y, X)
    results = model.fit()

    # 파라미터 추정값
    phi0_est = results.params[0]  # 상수항
    phi1_est = results.params[1]  # AR(1) 계수
    sigma_est = np.sqrt(results.mse_resid)  # 잔차의 표준편차

    phi0_ests[i] = phi0_est
    phi1_ests[i] = phi1_est
    sigma_ests[i] = sigma_est   

fig, axes = plt.subplots(3, 1, figsize=(5, 10))
sns.histplot(phi0_ests, bins=50, ax=axes[0], kde=True, alpha=0.5)
axes[0].set_title(r'$\phi_0$')
axes[0].axvline(x=phi0_ests.mean(), color='r', linestyle='--', alpha=0.5)

sns.histplot(phi1_ests, bins=50, ax=axes[1], kde=True, alpha=0.5)
axes[1].set_title(r'$\phi_1$')
axes[1].axvline(x=phi1_ests.mean(), color='r', linestyle='--', alpha=0.5)

sns.histplot(sigma_ests, bins=50, ax=axes[2], kde=True, alpha=0.5)
axes[2].set_title(r'$\sigma$')
axes[2].axvline(x=sigma_ests.mean(), color='r', linestyle='--', alpha=0.5)
fig.tight_layout()
plt.show()


# %%
