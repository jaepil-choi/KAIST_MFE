#%%
import numpy as np
from bertram0_functions import phi1
from scipy.special import gamma, factorial
from scipy.optimize import fsolve
INF = 100

def fun(a, c):
    s1, s2 = 0, 0
    for n in range(INF):
        s1 += (np.sqrt(2)*a)**(2*n+1) / factorial(2*n+1) * gamma((2*n+1)/2) 
        s2 += (np.sqrt(2)*a)**(2*n) / factorial(2*n) * gamma((2*n+1)/2) 

    return s1 - (a-c/2)*np.sqrt(2)*s2


theta = 100      #mean reversion speed
sigma = 0.15     #volatility
c_tilde = 0.001  #trading cost

c = c_tilde*np.sqrt(2*theta)/sigma  #transformed cost
root = fsolve(fun, args=(c,), x0 = 1)
sol = root[0] * sigma/np.sqrt(2*theta)
print(f"optimal a = {root[0]: .4f}")
print(f"optimal a_tilde = {sol: .4f}")

#%%
import matplotlib.pyplot as plt

# mean reversion speed
for sigma in [0.05, 0.15, 0.3]:
    thetas = np.arange(10, 201, 10)
    sols = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        c = c_tilde*np.sqrt(2*theta)/sigma  #transformed cost
        root = fsolve(fun, args=(c,), x0 = 1)
        sols[i] = root[0] * sigma/np.sqrt(2*theta)

    plt.plot(thetas, sols, 's-', label=rf"$\sigma = {sigma}$")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\tilde {a}^*$")
plt.grid()
plt.legend()
plt.show()

#%%
# sigma
for theta in [25, 100, 400]:
    sigmas = np.arange(0.05, 0.31, 0.01)
    sols = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        c = c_tilde*np.sqrt(2*theta)/sigma  #transformed cost
        root = fsolve(fun, args=(c,), x0 = 1)
        sols[i] = root[0] * sigma/np.sqrt(2*theta)

    plt.plot(sigmas, sols, 's-', label=rf"$\theta = {theta}$")

plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\tilde {a}^*$")
plt.grid()
plt.legend()
plt.show()

#%%
# transaction cost
for theta in [25, 100, 400]:
    cs = np.arange(0.001, 0.011, 0.001)
    sols = np.zeros(len(cs))
    for i, c_tilde in enumerate(cs):
        c = c_tilde*np.sqrt(2*theta)/sigma  #transformed cost
        root = fsolve(fun, args=(c,), x0 = 1)
        sols[i] = root[0] * sigma/np.sqrt(2*theta)

    plt.plot(cs, sols, 's-', label=rf"$\theta = {theta}$")

plt.xlabel(r"$\tilde {c}$")
plt.ylabel(r"$\tilde {a}^*$")
plt.grid()
plt.legend()
plt.show()


#%%
theta = 100      #mean reversion speed
sigma = 0.15     #volatility
c_tilde = 0.001  #trading cost

c = c_tilde * np.sqrt(2*theta) / sigma  #transformed cost
root = fsolve(fun, args=(c,), x0 = 1)
a = root[0]
b = -a
a_tilde = sigma / np.sqrt(2*theta) * a
profit_per_trade = sigma / np.sqrt(2*theta) * (a-b-c)
print(f"optimal a tilde = {a_tilde: .4f}")
print(f"profit per trade = {profit_per_trade: .4f}")

ET = 0
for n in range(INF):
    ET += 0.5 * ((np.sqrt(2)*a)**(2*n+1) - (np.sqrt(2)*b)**(2*n+1)) / factorial(2*n+1) * gamma((2*n+1)/2) 
ET /= theta
print(f"expected trade length = {ET: .4f}")

mu = profit_per_trade / ET
print(f"optimal mu = {mu: .4f}")



# %%
