# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dynamic Delta Hedging
#
# Hull 11ed Table 19.2, 19.4 의 Dynamic Delta Hedging adjustment 과정 을 구현

# %%
import yaml
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

# %% [markdown]
# ## GPT Prompt

# %%
prompt_yaml = 'gpt_prompt.yaml'
with open(prompt_yaml, 'r') as f:
    prompts = yaml.safe_load(f)

# %%
prompts = pd.DataFrame(prompts['prompts'])

# %% [markdown]
# ## GPT Output

# %% [markdown]
# 문제 있음. 만기에 ATM인 경우 delta --> (+/-)infinite 가는데 이 때 어떻게 되는지는 코드에 없음. 구현을 따로 해야하나?? 어떻게 해야하나?? GPT도 헤맨다. 

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

class StockPriceSimulator:
    def __init__(self, initial_price, mu, sigma, n_steps, random_seed=None):
        """
        Initialize the stock price simulator.

        :param initial_price: Initial stock price.
        :param mu: Drift coefficient (expected return).
        :param sigma: Volatility coefficient.
        :param n_steps: Total number of simulation steps (days).
        :param random_seed: Seed for reproducibility.
        """
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_price_path(self):
        """
        Simulate the stock price path using Geometric Brownian Motion in a vectorized manner.

        :return: numpy array of simulated stock prices.
        """
        dt = 1 / 252  # Assume 252 trading days in a year
        # Generate random normal variables
        Z = np.random.standard_normal(self.n_steps)
        # Calculate the log returns
        log_returns = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
        # Cumulative sum to get the log price
        log_price = np.log(self.initial_price) + np.cumsum(log_returns)
        # Exponentiate to get the price path
        price_path = np.exp(log_price)
        # Insert the initial price at the beginning
        price_path = np.insert(price_path, 0, self.initial_price)
        return price_path

class OptionPricer:
    def __init__(self, option_type='call', strike=50, risk_free_rate=0.05, sigma=0.2):
        """
        Initialize the option pricer.

        :param option_type: 'call' or 'put'.
        :param strike: Strike price.
        :param risk_free_rate: Risk-free interest rate.
        :param sigma: Volatility of the underlying asset.
        """
        self.option_type = option_type.lower()
        self.K = strike
        self.r = risk_free_rate
        self.sigma = sigma

    def calculate_delta(self, S, T):
        """
        Calculate the Black-Scholes delta.

        :param S: Current stock price.
        :param T: Time to maturity in years.
        :return: Delta of the option.
        """
        if T <= 0:
            # For T=0, delta approaches 1 for calls and -1 for puts if S > K,
            # otherwise approaches 0 for calls and 0 for puts
            if self.option_type == 'call':
                return 1.0 if S > self.K else 0.0
            else:
                return -1.0 if S > self.K else 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / \
             (self.sigma * np.sqrt(T))
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        elif self.option_type == 'put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        return delta

class DeltaHedgingSimulator:
    def __init__(self, initial_stock_price, option_pricer, time_horizon_weeks,
                 random_seed=None):
        """
        Initialize the delta hedging simulator.

        :param initial_stock_price: Initial stock price.
        :param option_pricer: Instance of OptionPricer.
        :param time_horizon_weeks: Total number of weeks to simulate.
        :param random_seed: Seed for reproducibility.
        """
        self.S0 = initial_stock_price
        self.option_pricer = option_pricer
        self.time_horizon_weeks = time_horizon_weeks
        self.random_seed = random_seed

    def run_simulation(self, mu, sigma, interest_rate):
        """
        Run the delta hedging simulation.

        :param mu: Drift coefficient for stock price simulation.
        :param sigma: Volatility for stock price simulation.
        :param interest_rate: Interest rate for cost accumulation.
        :return: pandas DataFrame with simulation results.
        """
        weeks_in_year = 52
        days_per_week = 5
        total_weeks = self.time_horizon_weeks
        total_days = total_weeks * days_per_week

        # Initialize the stock price simulator
        simulator = StockPriceSimulator(
            initial_price=self.S0,
            mu=mu,
            sigma=sigma,
            n_steps=total_days,
            random_seed=self.random_seed
        )
        price_path = simulator.simulate_price_path()

        # Initialize variables
        df_records = []
        cumulative_cost = 0.0  # In $000
        option_holding = 100000  # Sold 100,000 call options
        previous_shares = 0.0

        for week in range(0, total_weeks + 1):
            day = week * days_per_week
            if day > total_days:
                day = total_days  # Handle any rounding issues

            S = price_path[day]
            T = max((total_weeks - week) / weeks_in_year, 1e-6)  # Time to maturity in years

            delta = self.option_pricer.calculate_delta(S, T)

            desired_shares = delta * option_holding
            shares_purchased = desired_shares - previous_shares
            shares_purchased = int(round(shares_purchased))  # Ensure integer shares
            cost_shares = shares_purchased * S / 1000  # Convert to $000

            cumulative_cost += cost_shares

            # Record 'Cumulative Cost Including Interest' before adding interest
            cumulative_cost_including_interest = cumulative_cost

            # Calculate interest based on current cumulative cost
            interest_cost = cumulative_cost * (interest_rate / weeks_in_year)

            # Add interest to cumulative cost
            cumulative_cost += interest_cost

            # Calculate residual delta after hedge
            shares_held = previous_shares + shares_purchased
            residual_delta = (delta * option_holding) - shares_held
            delta_after_hedge = residual_delta / option_holding

            # Record the data
            df_records.append({
                'Week': week,
                'Stock Price': round(S, 2),
                'Delta': round(delta, 3),
                'Shares Purchased': shares_purchased,
                'Cost of Shares Purchased ($000)': round(cost_shares, 1),
                'Cumulative Cost Including Interest ($000)': round(cumulative_cost_including_interest, 1),
                'Interest Cost ($000)': round(interest_cost, 1),
                'Delta After Hedge': round(delta_after_hedge, 3)
            })

            # Update previous_shares
            previous_shares = shares_held

        df = pd.DataFrame(df_records)
        return df

def main():
    """
    Main function to set up and run the delta hedging simulation.

    :return: pandas DataFrame with simulation results.
    """
    # Define simulation parameters
    initial_stock_price = 49.00  # As per sample table
    option_type = 'call'
    strike_price = 50
    risk_free_rate = 0.05  # 5% annual
    sigma = 0.2  # 20% annual volatility
    time_horizon_weeks = 20
    mu = risk_free_rate  # Assuming drift equals risk-free rate
    interest_rate = 0.05  # 5% annual interest
    random_seed = 42  # For reproducibility

    # Initialize OptionPricer
    option_pricer = OptionPricer(
        option_type=option_type,
        strike=strike_price,
        risk_free_rate=risk_free_rate,
        sigma=sigma
    )

    # Initialize DeltaHedgingSimulator
    simulator = DeltaHedgingSimulator(
        initial_stock_price=initial_stock_price,
        option_pricer=option_pricer,
        time_horizon_weeks=time_horizon_weeks,
        random_seed=random_seed
    )

    # Run simulation
    df = simulator.run_simulation(
        mu=mu,
        sigma=sigma,
        interest_rate=interest_rate
    )

    return df



# %%
main()

# %%
norm.cdf(np.inf)

# %%
norm.cdf(-np.inf)

# %%
