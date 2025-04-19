# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Liquidation with Permanent Price Impact

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Introduction
#
# The agent controls their speed of trading $\nu$, and assumes that she receives an execution price of
# $$
# S_t^{\nu} - k\, Q_t^{\nu}
# $$
# which is linear in her trading rate to account for the cost of walking the limit order book. Furthermore, her trading induces a drift in the asset price, which is otherwise assmed to be a Brownian motion. Hence, $S^{\nu}_t$ satisfies the SDE
# $$
# dS_t^\nu = a \,\nu_t\,dt + \sigma \, dW_t
# $$
# She aims to maximize the  performance criterion 
# $$
# \mathbb{E}\Big[ X_T^{\nu} + Q_T^{\nu}(S_T^{\nu} - \alpha Q_T^{\nu}) - \phi \int_0^T (Q_u^{\nu})^2\, du\Big]
# $$
# where $X_T^\nu$ is the total cash from trading, and the term $\alpha\,(Q_T^\nu)^2$ represents a penalty on holding terminal inventory (which we will send to infinity to guarantee liquidation).

# %% [markdown]
# ### Solving DPE
#
# The dynamic programming principle implies that the value function satisfies the $\textbf{HJB}$ equation
# \begin{equation}
# 0 = (\partial_t + \tfrac{1}{2}\sigma^2\partial_{SS})H - \phi q^2 + \sup_{\nu}\{(\nu(S - f(\nu))\partial_x - g(\nu) \partial_S - \nu \partial_q)H\}
# \end{equation}
#
# subject to the terminal condition $H(T,x,S,q) = x + Sq - \alpha q^2$.
#
# For the case of linear impact functions $f(\nu)=k\,\nu$ and $g(\nu)=b\,\nu$, the  $\textbf{DPE}$ can be solved explicitly and the optimal speed to trade is
# \begin{equation}
# \nu_t^* = \gamma \frac{\zeta e^{\gamma(T-t)} + e^{-\gamma(T-t)}}{\zeta e^{\gamma(T-t)} - e^{-\gamma(T-t)}}\, Q_t^{{\nu}^*}
# \end{equation}
#
# From this equation, it is also possible to obtain the agent's inventory $Q_t^{\nu^*}$ that results from following this strategy.
#
# \begin{equation}
# Q_t^{\nu^*} = \frac{\zeta e^{\gamma(T-t)} + e^{-\gamma(T-t)}}{\zeta e^{\gamma(T-t)} - e^{-\gamma(T-t)}} \,Q_0.
# \end{equation}
#
# Now we show the implementation of the optimal speed to trade at and inventory along the optimal strategy in the following code.

# %%
# Import necessary libraries
import numpy as np
import import_ipynb
import matplotlib.pyplot as plt

# %% [markdown]
# The following code computes the optimal speed to trade at $\nu_t^*$ and inventory along the optimal strategy $Q_t^{\nu^*}$.

# %%
# Parameter settings
T = 1
Ndt = 6.5 * 3600 # 1 day = 6.5 hours = 6.5 * 3600 seconds
dt = T / Ndt
t = np.arange(0, T + dt, dt)

k = 0.001
b = 0.001

tau = T - t
tt = np.arange(0, 1 + 1 / 10, 1 / 10) # time points to be marked

myphi = np.array([10e-10, 0.001, 0.01, 0.1])
symb = np.array(["o", "*", "+", "^"])

alpha = 100


# %%
# Functions
def inter_extrapolation(x, y, e):
    """ Extrapolation and interpolation.
    
    :param x: a numpy array
    :param y: a numpy array
    :param e: a numpy array, equivalent of x
    :return: a numpy array
    """
    
    # Write codes here
    
    return

def calculate_inventory_trading_speed(alpha, phi, t, tt, T, b, k):
    """ For given points t, this function solve for the optimal speed of trading as nu, and investor's inventory along the
        optimal path as Q. 
        This function also returns optimal speed of trading as nut, and investor's inventory along the optimal path Qt as a
        function of time, tt, which is a vector of time points to be marked.
    """

    # Write codes here
    
    return


def plot_inventory_trading_speed(alpha0, phi, symb, t, tt, T, b, k, labels, main):
    """ This function plots the optimal inventory & trading speed paths using above function
        to calculate inventory and speed of tading vs time.
    """
    
    # Write codes here


# %% [markdown]
# Now we use show the change of the investor's inventory along the optimal path for various levels of the running penalty $\phi$ as $\alpha$ changes.

# %%
# Plot figures
plot_inventory_trading_speed(0.01, myphi, symb, t, tt, T, b, k, [r"$\phi = $" + str(i) for i in myphi], main=r"$\alpha = 0.01$")
plot_inventory_trading_speed(100, myphi, symb, t, tt, T, b, k, [r"$\phi = $" + str(i) for i in myphi], main=r"$\alpha = +\infty$")

# %%
