# %%
# %%
import numpy as np
from scipy.special import erfi, gamma, digamma, factorial

# %%
#First passage time for OU process
#dY = -Ydt + sqrt(2)dW
def eof(x, dt, kappa=1, theta=0):
    return x*np.exp(-kappa*dt) + theta*(1-np.exp(-kappa*dt))

# %%
def vof(dt, kappa=1, theta=0, sigma=np.sqrt(2)):
    return sigma**2/(2*kappa)*(1-np.exp(-2*kappa*dt))

# %%
def simulate_ou_first_passage(y0, b, years, n0):
    n_days = 10000   #time steps per year
    dt = 1 / n_days  #time step
    T = years * n_days  #total time steps
    variance = vof(dt)  #variance per time step
    y, n = y0, n0
    hit_time = np.zeros(T)
    for i in range(T):
        expected = eof(y, dt)  #expected value of Y at next time step
        y1 = expected + np.sqrt(variance)*np.random.randn(n)  #sample from normal distribution
        if b>y0:
            number_passed = (y1>=b).sum()  #number of paths that have passed the target level
            y = y1[y1<b]  #remaining paths
        else:
            number_passed = (y1<=b).sum()  #number of paths that have passed the target level
            y = y1[y1>b]  #remaining paths
        hit_time[i] = number_passed  #number of paths that have passed the target level at time i
        n -= number_passed  #update number of remaining paths
        
    hit_time[i] += n  #add remaining paths to the last time step
    times = np.arange(1,T+1) / n_days 
    avg = np.sum(hit_time*times)/n0  #average first passage time
    var = np.sum(hit_time*times**2)/n0 - avg**2  #variance of first passage time
    return avg, var, n

# %%
INF = 100
def psi(x):
    return digamma(x) - digamma(1) # 프사이

# %% [markdown]
# expected FPT (first passage time) 계산하는 부분. 

# %%
def phi1(x):
    k = np.arange(1, INF)
    return 0.5 * (gamma(k/2) * (np.sqrt(2) * x)**k / factorial(k)).sum()

# %%
def phi2(x):
    k = np.arange(1, INF)
    return 0.5 * (gamma(k/2) * psi(k/2) * (np.sqrt(2) * x)**k / factorial(k)).sum()

# %%
def w1(z):
    k = np.arange(1, INF)
    return (0.5 * (gamma(k/2) * (np.sqrt(2) * z)**k / factorial(k)).sum())**2 \
            - (0.5 * ((-1)**k * gamma(k/2) * (np.sqrt(2) * z)**k / factorial(k)).sum())**2 
def w2(z):
    k = np.arange(1, INF)
    return (gamma((2*k-1)/2) * psi((2*k-1)/2) * (np.sqrt(2)*z)**(2*k-1) / factorial(2*k-1)).sum()

# %% [markdown]
# 거래 사이클 기간의 기대값과 분산
#
# -Erfi(x) : Imaginary Error Function

# %%
def expected_trade_length(a, b):
    return (erfi(b/np.sqrt(2))-erfi(a/np.sqrt(2)) ) *np.pi 

# %%
def variance_trade_length(a, b):
    return w1(b) - w1(a) - w2(b) + w2(a)
