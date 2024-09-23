#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# uniform에서 뽑는
u1 = np.random.rand(10000) # 길이
u2 = np.random.rand(10000) # 각도

# 원 그리며 극좌표 변환
z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)

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
print("Std = ", z.std())
print("Skewness = ", z.skew())
print("Kurtosis = ", z.kurt())

# 아... 너무 간단해서 좋은데....
# 효율성이 좋지가 않다. 
# 왜? 삼각함수, 로그함수 계산하는 것이 효율적이지 않다.
# 그래서 더 개선된 방법이 필요하다.
# %%
