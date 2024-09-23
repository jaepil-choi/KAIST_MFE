#%%
import numpy as np
import matplotlib.pyplot as plt 

# 예제 행렬 A와 벡터 b 정의
A = np.array([[1,2,3],[3,2,1],[5,1,1]], dtype=float)
A = A.T@A
b = np.array([12,10,8], dtype=float)

print("\nSolution x from A inverse")
sol = np.linalg.inv(A).dot(b)
print(sol)


#%%
#Gauss-Seidel
n = len(b)
x = np.ones(n) #initial value
idx = np.arange(n)
iter = 50 # 원래는 iteration을 정하는게 아니라 수렴할때 까지 하고 수렴 조건을 정하는 것임. 
xs = np.empty((iter,n))
for j in range(iter):
    for i in range(n):
        mask = idx!=i # 마스킹. 업데이트하려는 i번째를 제외하고 가져온다. 
        x[i] = (b[i] - (A[i,mask] * x[mask]).sum()) / A[i,i]
    xs[j,:] = x

plt.plot(xs,'.-')
for i in range(n):
    plt.plot(np.arange(iter), np.ones(iter)*sol[i], ":y")

#%%
#SOR
omega = 0.8 # 이거 바꾸면 변화 볼 수 있음. 
n = len(b)
x = np.ones(n) #initial value
iter = 50
xs = np.empty((iter,n))
for j in range(iter):
    for i in range(n):
        x[i] = x[i] + omega * (b[i] - (A[i,:] * x).sum()) / A[i,i]
    xs[j,:] = x

plt.plot(xs,'.-')
for i in range(n):
    plt.plot(np.arange(iter), np.ones(iter)*sol[i], ":y")
# %%
