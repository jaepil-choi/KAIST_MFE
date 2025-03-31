#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dx = rho*(mu-x)*dt + sigma*dW
#T=t_hat/rho
#t_hat = 0.5*np.log(1+0.5*(np.sqrt((c**2-3)**2+4*c**2)+c**2-3))
#c = (x0-mu)/sigma*np.sqrt(rho*2)
def first_passage(x0, mu, sigma, rho):
    c = (x0-mu)/sigma*np.sqrt(rho*2)
    t_hat = 0.5*np.log(1+0.5*(np.sqrt((c**2-3)**2+4*c**2)+c**2-3))
    max_first_passage_time = t_hat/rho

    n = 10000
    dts = 10001
    dt = 1/250
    x = np.zeros((n, dts))
    x[:,0] = x0

    for i in range(1, dts):
        x[:,i] = x[:,i-1] + rho*(mu-x[:,i-1])*dt + sigma*np.random.randn(n)*np.sqrt(dt)

    if x0>mu:
        first_passage_times = (x<mu).argmax(axis=1) /250
    else:
        first_passage_times = (x>mu).argmax(axis=1) /250
    return max_first_passage_time, first_passage_times



rhos = [1, 3, 5]
mu = 0
sigma = 0.05
x0 = 0.1


for i, rho in enumerate(rhos):
    fig, ax = plt.subplots(1,1,figsize=(5, 2))
    max_first_passage_time, first_passage_times = first_passage(x0, mu, sigma, rho)
    sns.histplot(first_passage_times, bins=100, kde=True, ax=ax)
    ax.set_xlabel(f"First passage time: {max_first_passage_time:.3f}")
    ax.set_title(f"$x_0$={x0},  $\\rho$={rho},  $\\mu$={mu},  $\\sigma$={sigma}")
    ax.set_xlim(0, 5)
    ax.axvline(max_first_passage_time, color='red')


    
    
# %%
