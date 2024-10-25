#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

u1 = 2*np.random.rand(10000) - 1
u2 = 2*np.random.rand(10000) - 1
idx = u1**2+u2**2<1
u1 = u1[idx]
u2 = u2[idx]
r = np.sqrt(u1**2 + u2**2) # 피타고라스 단위 원 안에 있는지만 확인
z1 = u1*np.sqrt(-2*np.log(r)/(r**2)) # 덕분에 sin, cos 계산 없다.
z2 = u2*np.sqrt(-2*np.log(r)/(r**2))


fig, ax = plt.subplots(2,2,figsize=(10,10))
ax[0,0].plot(u1,u2,'.')
ax[0,0].set_xlabel("u1")
ax[0,0].set_ylabel("u2")
ax[0,1].plot(z1,z2,'.')
ax[0,1].set_xlabel("z1")
ax[0,1].set_ylabel("z2")

z = np.concatenate([z1,z2])
ax[1,0].hist(z, bins=50)
stats.probplot(z, dist="norm", plot=ax[1,1])

z = pd.Series(z)
print("Mean = ", z.mean())
print("Std = ", z.std()) # ?? 교수님께서 std가 이상하다고 하심. 
print("Skewness = ", z.skew())
print("Kurtosis = ", z.kurt())
# %%
