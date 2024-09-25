# 설명 어렵다. 기왕이면 notability 녹음된거 들으면서 봐라. 

#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

corr = np.array([[1,0.3,0.5],[0.3,1,0.6],[0.5,0.6,1]])
pos_def = np.all(np.linalg.eigvals(corr) > 0)
print(corr)
print(pos_def)

#%%
#Cholesky Decomposition
c = np.linalg.cholesky(corr)
x = np.random.randn(10000,3)
y = x @ c.T # X_c = B@X 랑 X_c = X@B.T 랑 다르다고 한거 기억하기. 

y = pd.DataFrame(y, columns=['z1','z2','z3'])
print("Mean")
print(y.apply(['mean','std']))
print()

print("Correlation")
print(y.corr())


#%%
#Positive Definite 하지 않은 상관계수 행렬 생성
pos_def = True
while pos_def:
    x = np.random.randn(1000, 2)
    x = np.concatenate([x[:,0:1], x[:,0:1]+x[:,1:2], x[:,0:1]-2*x[:,1:2]], axis=1)
    # 평균은 0이지만, 분산은 아니다. 
    corr = pd.DataFrame(x).corr()
    pos_def = np.all(np.linalg.eigvals(corr) > 0)

print(corr)
print(pos_def)

#%%
#cholesky: error
#c = np.linalg.cholesky(corr)
# LinAlgError: Matrix is not positive definite. 

#Eigenvalue Decomposition
values, vectors = np.linalg.eig(corr) # positive definite하지 않아도 계산이 됨. 대신 value 중 일부가 - 일 수 있음. 
values = np.maximum(0, values) # 그래서 -를 다 0으로 바꿔줌. 이래도 되냐? 된다. 거의 비슷하게 나옴. 
B = vectors @ np.diag(np.sqrt(values))
print(B)
print()
print(B @ B.T)
print()

z = np.random.randn(10000,3)
y = z @ B.T # random number generation

y = pd.DataFrame(y, columns=['z1','z2','z3'])
print("Mean")
print(y.apply(['mean','std']))
print()
print("Correlation")
print(y.corr())

# Correlation
#           z1        z2        z3
# z1  1.000000  0.709331  0.464715
# z2  0.709331  1.000000 -0.294503
# z3  0.464715 -0.294503  1.000000

#%%
#Singular value decomposition
print("=== original data ===")
print(pd.DataFrame(x).apply(['mean','std']))
print(pd.DataFrame(x).corr())
print()

# original data correlation

#           0         1         2
# 0  1.000000  0.700772  0.476241
# 1  0.700772  1.000000 -0.293554
# 2  0.476241 -0.293554  1.000000

U, S, Vh = np.linalg.svd(x)
np.allclose(U[:,:3] @ np.diag(S) @ Vh, x)

B = Vh.T @ np.diag(S) / np.sqrt(len(x))
z = np.random.randn(10000,3)
y = z @ B.T

print("=== simulation data ===")
y = pd.DataFrame(y, columns=['z1','z2','z3'])
print("Mean")
print(y.apply(['mean','std']))
print()
print("Correlation")
print(y.corr())
# %%

# Correlation
#           z1        z2        z3
# z1  1.000000  0.697139  0.482821
# z2  0.697139  1.000000 -0.291242
# z3  0.482821 -0.291242  1.000000