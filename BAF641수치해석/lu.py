#%%
import numpy as np
from scipy.linalg import lu

# 예제 행렬 A와 벡터 b 정의
A = np.array([[1,2,3],[3,2,1],[5,1,1]], dtype=float)
b = np.array([12,10,8], dtype=float)

# LU 분해 수행
P, L, U = lu(A)

# 전방 대입 (Ly = Pb)
Pb = np.dot(P.T, b)
y = np.zeros_like(b)
y[0] = Pb[0]/L[0,0]
for i in range(1,len(y)):
    y[i] = (Pb[i] - np.dot(L[i, :i], y[:i])) / L[i,i]

# 후방 대입 (Ux = y)
x = np.zeros_like(y)
x[-1] = y[-1] / U[-1,-1]
for i in range(len(x)-2, -1, -1):
    x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

# 결과 출력
print("Solution x from PLU 분해:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(A).dot(b))