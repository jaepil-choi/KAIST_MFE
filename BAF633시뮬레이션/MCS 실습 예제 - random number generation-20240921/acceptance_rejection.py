#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

f = lambda x: 2/np.sqrt(2*np.pi)*np.exp(-x**2/2) # 원래 샘플링 하려는 복잡한 함수
g = lambda x: np.exp(-x) # lambda parameter 가 1 인 지수함수 (단순화 함수)
Ginv = lambda x: -np.log(1-x) # g 적분한 것의 역함수
x = np.linspace(0,5,501)
c = np.sqrt(2/np.pi)*np.exp(0.5)

#c = 1 #이걸 그냥 넣고 하면 1을 넘어가는데 
plt.plot(x,f(x)/(c*g(x))) # c를 잘 구해서 넣으면 f(x) / (c*g(x)) < 1임을 확인할 수 있다. 
plt.show()


plt.plot(x,f(x))
plt.plot(x,c*g(x)) # 딱 붙는 것이 효율적으로 샘플링하는 것. 이것도 c=1 놓고 해보면 접점 없이 나온다. 
plt.show()


#%%
#random sampling from Exponential dist.
n = 100000
e = np.random.rand(n)
x = Ginv(e) # 아주 쉽게 exponential dist. 샘플링 가능. 그러라고 쓰는거. 
plt.hist(x, bins=50)
plt.show()

#%%
#acceptance-rejection
u = np.random.rand(n) # 쓸지 말지 결정하는 것.
idx = u < (f(x) / (c*g(x)))
y = x[idx]

#signx
s = np.random.rand(len(y))
sign = (+1)*(s>0.5) + (-1)*(s<=0.5) # 부호 결정하는 random number
z  = y * sign # 최종적인 randomly generated number들 

fig, ax = plt.subplots(2,1,figsize=(5,10))
ax[0].hist(z, bins=50)
stats.probplot(z, dist="norm", plot=ax[1]) # Q-Q plot. 정규분포인지 확인
plt.show()

z = pd.Series(z)
print("Size = ", len(z)) # 10만개 중 7.6만개 정도가 accept됨. 
print("Mean = ", z.mean())
print("Std = ", z.std())
print("Skewness = ", z.skew())
print("Kurtosis = ", z.kurt())
# %%

# 이 짓을 왜하나? 정규분포의 cdf 즉, F(x)를 모르고 
# 그것의 역함수 F(x)^-1를 모르는 경우에도 샘플링을 할 수 있게 해준다.

# numpy 등에 이렇게 되어있다는 것을 알고 있자. 직접 구현해서 쓸 일은 없을 것이다. 
# 실제 numpy에서는 대량의 random number 뽑는 acceptance-rejection을 약간 개선한 지구라트 방법이란걸 쓴다고 함.
# 메소포타미아의 지구라트 
# 미리 계산된 파라미터를 써서 더 빠르게 샘플링 가능.
# 대략적으로, 정규분포를 막대기처럼 쪼개서 쓰는 그런 방법이라고 함. 컨셉적으로만...
