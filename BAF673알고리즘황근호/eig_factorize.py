#%%
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2/3, 1/6], [1/3, 5/6]])

eig_val, eig_vec = np.linalg.eig(A)
A_  = eig_vec @ np.diag(eig_val) @ np.linalg.inv(eig_vec)

print(A)
print()
print(A_)

#%%
n = 500
x = np.random.random((n,2)) * 10
_x = x.copy()
fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0,0].plot(x[:,0], x[:,1], '.')
ax[0,0].set_title('original')
ax[0,0].set_xlim(0,13)
ax[0,0].set_ylim(0,13)
for i in range(1,4):
    x = (A @ x.T).T
    ax[i//2,i%2].plot(x[:,0], x[:,1], '.')
    ax[i//2,i%2].set_title(f'transformed {i}')
    ax[i//2,i%2].set_xlim(0,13)
    ax[i//2,i%2].set_ylim(0,13)

plt.show()


# %%
A  = eig_vec @ np.diag([0.5, 1]) @ np.linalg.inv(eig_vec)
plt.figure(figsize=(6,6))
x = np.array([[10,10,3,12,1], [10,2,12,12,5]])
plt.plot(x[0], x[1], 's')
for i in range(10):
    x = A @ x
    plt.plot(x[0], x[1], '.')
plt.xlim(0,20)
plt.ylim(0,20)
plt.plot([0,10],[0,20],':')
plt.plot([20,0],[0,20],':')
#plt.grid(True)
plt.show()
