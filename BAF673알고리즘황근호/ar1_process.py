#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ar1_process(phi0, phi1, sigma, T):
    y = np.zeros(T+1)
    y[0] = phi0/(1-phi1)   #장기균형에서의 시작값
    for i in range(T):
        e = np.random.randn()
        y[i+1] = phi0 + phi1 * y[i] + sigma * e
    return y

def random_walk(sigma, T):
    y = np.zeros(T+1)
    for i in range(T):
        e = np.random.randn()
        y[i+1] = y[i] + sigma * e
    return y

def plot_process(y, title):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(y)
    ax.set_title(title)
    ax.grid()
    plt.show()

if __name__ == '__main__':
    phi0, phi1, sigma = 4, 0.2, 1
    y = ar1_process(phi0, phi1, sigma, 250)
    plot_process(y, f'$\\phi_0 = {phi0}, \\phi_1 = {phi1}, \\sigma = 1$')

    phi0, phi1, sigma = 1, 0.8, 1
    y = ar1_process(phi0, phi1, sigma, 250)
    plot_process(y, f'$\\phi_0 = {phi0}, \\phi_1 = {phi1}, \\sigma = 1$')


    #%%
    phi0, phi1, sigma = 0, 0.2, 1
    y = ar1_process(phi0, phi1, sigma, 100)
    plot_process(y, f'$\\phi_0 = {phi0}, \\phi_1 = {phi1}, \\sigma = 1$')

    phi0, phi1, sigma = 0, 0.99, 1
    y = ar1_process(phi0, phi1, sigma, 1000)
    plot_process(y, f'$\\phi_0 = {phi0}, \\phi_1 = {phi1}, \\sigma = 1$')


    #%%
    # Random Walk과 AR(1) 비교
    y = random_walk(1, 1000)
    plot_process(y, 'Random Walk')



