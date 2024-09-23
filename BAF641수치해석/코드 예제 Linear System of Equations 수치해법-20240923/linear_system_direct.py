#%%
import numpy as np
from scipy.linalg import lu
from numpy.linalg import eig, cholesky, qr, svd
import time

# 예제 행렬 A와 벡터 b 정의
A = np.array([[1,2,3],[3,2,1],[5,1,1]], dtype=float)
b = np.array([12,10,8], dtype=float)

#
A = np.random.randn(100,100) # 큰거 구할 때 예시
b = np.random.randn(100)

###################################
# Forward / Backward Substitution
###################################
def forward(L, b): # 아래 삼각형 L, 백터 y, 곱하면 Pb
    # 전방 대입 (Ly = Pb)
    y = np.zeros_like(b)
    y[0] = b[0]/L[0,0]
    for i in range(1,len(y)): # 직전까지 row의 값들을 가져와 y의 i 전까지의 값 내적 구하면 scalar 나오고,... 
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i,i] # L로 나눠주고
    return y

def backward(U, y): # 아래에서 위로 올라옴. 맨 마지막부터. 
    # 후방 대입 (Ux = y)
    x = np.zeros_like(y)
    x[-1] = y[-1] / U[-1,-1]
    for i in range(len(x)-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i] # i+1 부터 뒷부분부터 계산
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
#c = cholesky(A)  #error --> A is not positive definite

#%%
B = A.T @ A # positive definite이 된다. symmetric하다. 
print("B = \n", B, '\n')

eigen_values, eigen_vectors = eig(B) # eigen values 다 + 임. 
print("Eigen values = ", eigen_values)

c = cholesky(B)  #A = c@c' # lower triangular matrix를 c로 뒀다. 
# 교안에서 R은 upper triangular matrix로 뒀다.
print(np.allclose(B, c@c.T), "\n")

y = forward(c, b)
x = backward(c.T, y)
print("Solution x from Cholesky decomposition:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(B).dot(b)) # 촐레스키나 이거나 같다.


#%%
###################################
#QR decomposition
###################################
Q, R = qr(A) # 당연히 True 
print(np.allclose(A, Q@R))

x = backward(R, Q.T @ b)
print("Solution x from QR decomposition:")
print(x)

print("\nSolution x from A inverse")
print(np.linalg.inv(A).dot(b))

## 둘이 같게 나오는 것 확인. 

#%%
###################################
#SVD decomposition
###################################
U, S, Vh = svd(A)
print(np.allclose(A, U @ np.diag(S) @ Vh)) # diagonal 로 matrix 만들어줌. 

x = Vh.T @ np.diag(1/S) @ U.T @ b # forward back substitution이 없어도 된다. 
print("Solution x from SVD decomposition:")
print(x) # A랑 똑같아진다. 

print("\nSolution x from A inverse")
print(np.linalg.inv(A).dot(b))
# %%