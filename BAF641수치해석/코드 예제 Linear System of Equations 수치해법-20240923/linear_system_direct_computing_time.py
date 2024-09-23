#%%
import numpy as np
from scipy.linalg import lu
from numpy.linalg import eig, cholesky, qr, svd
import time

#
n = 3000
A = np.random.randn(n,n)
b = np.random.randn(n)

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

###################################
# LU Decomposition
###################################
# LU 분해 수행: A = P @ L @ U
t0 = time.time()
P, L, U = lu(A)
Pb = np.dot(P.T, b)
y = forward(L, Pb)
x = backward(U, y)
t1 = time.time()

# 결과 출력
print("Solution x from PLU decomposition:")
print("Time = ", t1-t0)

###################################
#cholesky decomposition
###################################
B = A.T @ A

t0 = time.time()
c = cholesky(B)  #A = c@c'
y = forward(c, b)
x = backward(c.T, y)
t1 = time.time()

print("Solution x from Cholesky decomposition:")
print("Time = ", t1-t0)

###################################
#QR decomposition
###################################
t0 = time.time()
Q, R = qr(A)
x = backward(R, Q.T @ b)
t1 = time.time()

print("Solution x from QR decomposition:")
print("Time = ", t1-t0)

###################################
#SVD decomposition
###################################
t0 = time.time()
U, S, Vh = svd(A)
x = Vh.T @ np.diag(1/S) @ U.T @ b
t1 = time.time()
print("Solution x from SVD decomposition:")
print("Time = ", t1-t0)

###################################
#A inverse
###################################
print("\nSolution x from A inverse")
t0 = time.time()
Ainv = np.linalg.inv(A)
x = Ainv.dot(b)
t1 = time.time()
print("Time = ", t1-t0)