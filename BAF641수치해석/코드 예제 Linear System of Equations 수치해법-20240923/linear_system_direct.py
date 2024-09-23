#%%
import numpy as np
from scipy.linalg import lu
from numpy.linalg import eig, cholesky, qr, svd
import time

# 예제 행렬 A와 벡터 b 정의
A = np.array([[1,2,3],[3,2,1],[5,1,1]], dtype=float)
b = np.array([12,10,8], dtype=float)

#
A = np.random.randn(100,100)
b = np.random.randn(100)

###################################
# Forward / Backward Substitution
###################################
def forward(L, b):
    # 전방 대입 (Ly = Pb)
    y = np.zeros_like(b)
    y[0] = b[0]/L[0,0]
    for i in range(1,len(y)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i,i]
    return y

def backward(U, y):
    # 후방 대입 (Ux = y)
    x = np.zeros_like(y)
    x[-1] = y[-1] / U[-1,-1]
    for i in range(len(x)-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


#%%
###################################
# LU Decomposition
###################################
# LU 분해 수행: A = P @ L @ U
P, L, U = lu(A)
Pb = np.dot(P.T, b)
y = forward(L, Pb)
x = backward(U, y)

# 결과 출력
print("Solution x from PLU decomposition:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(A).dot(b))

#%%
###################################
#cholesky decomposition
###################################
#check whether A is positive definite
eigen_values, eigen_vectors = eig(A)
print(eigen_values)
#c = cholesky(A)  #error

#%%
B = A.T @ A
print("B = \n", B, '\n')

eigen_values, eigen_vectors = eig(B)
print("Eigen values = ", eigen_values)

c = cholesky(B)  #A = c@c'
print(np.allclose(B, c@c.T), "\n")

y = forward(c, b)
x = backward(c.T, y)
print("Solution x from Cholesky decomposition:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(B).dot(b))


#%%
###################################
#QR decomposition
###################################
Q, R = qr(A)
print(np.allclose(A, Q@R))

x = backward(R, Q.T @ b)
print("Solution x from QR decomposition:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(A).dot(b))

#%%
###################################
#SVD decomposition
###################################
U, S, Vh = svd(A)
print(np.allclose(A, U @ np.diag(S) @ Vh))

x = Vh.T @ np.diag(1/S) @ U.T @ b
print("Solution x from SVD decomposition:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(A).dot(b))