#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ar1_process import ar1_process, random_walk
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 예시 시계열 (AR(1))
phi0 = 0
phi1 = 0.9
sigma = 1
n = 1000
y = ar1_process(phi0, phi1, sigma, n)
z = random_walk(sigma, n)

# ACF & PACF 그리기
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(y, lags=30, ax=axes[0])
axes[0].set_title("ACF (Autocorrelation Function)")

plot_pacf(y, lags=30, ax=axes[1], method='ywmle')
axes[1].set_title("PACF (Partial Autocorrelation Function)")
fig.suptitle(f"AR(1) Process: $\\phi_1 = {phi1}$")
plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(z, lags=30, ax=axes[0])
axes[0].set_title("ACF (Autocorrelation Function)")

plot_pacf(z, lags=30, ax=axes[1], method='ywmle')
axes[1].set_title("PACF (Partial Autocorrelation Function)")
fig.suptitle("Random Walk")
plt.tight_layout()
plt.show()


#%%
a = sigma * np.random.randn(1000)
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(a, lags=30, ax=axes[0])
axes[0].set_title("ACF (Autocorrelation Function)")

plot_pacf(a, lags=30, ax=axes[1], method='ywmle')
axes[1].set_title("PACF (Partial Autocorrelation Function)")
fig.suptitle("White Noise")
plt.tight_layout()
plt.show()




