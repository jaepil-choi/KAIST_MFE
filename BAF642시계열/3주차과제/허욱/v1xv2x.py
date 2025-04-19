# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3.11.10 ('kaist311')
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter


# %% [markdown]
# # V1X는 S&P500 의 V1X, V2X는 EURO의 V1X 지표. 2020.11.20~ 2024.11.19 최근 5년 일일데이터 사용

# %%
# 2020.11.20~ 2024.11.19 최근 5년 일일데이터 사용
snp=pd.read_csv('/Users/danielkhur/Documents/카이스트 MFE/MFE 2024-2학기/후반기/금융시계열분석(9-16)/3주차과제/CBOE Volatility Index Historical Data.csv')
euro=pd.read_csv('/Users/danielkhur/Documents/카이스트 MFE/MFE 2024-2학기/후반기/금융시계열분석(9-16)/3주차과제/STOXX 50 Volatility VSTOXX EUR Historical Data.csv')

# %%
snp=snp[['Date','Price']]
snp.head()

# %%
euro=euro[['Date','Price']]
euro.head()

# %%
snp.columns=['date','snpprice']
euro.columns=['date','europrice']

# %%
snp_euro=pd.merge(snp, euro, on='date', how='inner')
snp_euro

# %%
snp_euro = snp_euro[::-1].reset_index(drop=True)
snp_euro

# %%
v1x=snp_euro['snpprice'].values[::-1]
v2x=snp_euro['europrice'].values[::-1]
dates=snp_euro['date'].values[::-1]

# %% [markdown]
# # 1. V1X와 V2X의 시계열을 추세, 계절성 및 잡음으로 분해하라.

# %%
decomposition_snp = sm.tsa.seasonal_decompose(v1x, model='additive', period = 12) 
y = pd.Series(v1x, index=dates)
observed = pd.Series(decomposition_snp.observed, index=y.index)
trend = pd.Series(decomposition_snp.trend, index=y.index)
seasonal = pd.Series(decomposition_snp.seasonal, index=y.index)
resid = pd.Series(decomposition_snp.resid, index=y.index)

# Plot the decomposition with a title
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
fig.suptitle('Seasonal Decomposition of Time Series (S&P V1X)', fontsize=16)

observed.plot(ax=ax1, title='Observed', legend=False)
trend.plot(ax=ax2, title='Trend', legend=False)
seasonal.plot(ax=ax3, title='Seasonal', legend=False)
resid.plot(ax=ax4, title='Residual', legend=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()

# %%
decomposition_euro = sm.tsa.seasonal_decompose(v2x, model='additive', period = 12) 
y = pd.Series(v2x, index=dates)
observed = pd.Series(decomposition_euro.observed, index=y.index)
trend = pd.Series(decomposition_euro.trend, index=y.index)
seasonal = pd.Series(decomposition_euro.seasonal, index=y.index)
resid = pd.Series(decomposition_euro.resid, index=y.index)

# Plot the decomposition with a title
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
fig.suptitle('Seasonal Decomposition of Time Series (EURO V2X)', fontsize=16)

observed.plot(ax=ax1, title='Observed', legend=False)
trend.plot(ax=ax2, title='Trend', legend=False)
seasonal.plot(ax=ax3, title='Seasonal', legend=False)
resid.plot(ax=ax4, title='Residual', legend=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()

# %% [markdown]
# # 2. V1X와 V2X 각각에 대해 단위근 테스트를 실행하고 정상성을 살펴보라.

# %%
y = pd.Series(v1x, index=dates)

# Augmented Dickey-Fuller test
result = adfuller(y)

# Extracting the results
print('ADF Statistic for V1X:', result[0])
print('V1X p-value:', result[1])
print('V1X Critical Values:', result[4])

# Interpretation
if result[1] <= 0.05:
    print("The V1X time series is stationary (reject the null hypothesis of unit root).")
else:
    print("The V1X time series is non-stationary (fail to reject the null hypothesis of unit root).")

# %%
y = pd.Series(v2x, index=dates)

# Augmented Dickey-Fuller test
result = adfuller(y)

# Extracting the results
print('ADF Statistic for V2X:', result[0])
print('V2X p-value:', result[1])
print('V2X Critical Values:', result[4])

# Interpretation
if result[1] <= 0.05:
    print("The V2X time series is stationary (reject the null hypothesis of unit root).")
else:
    print("The V2X time series is non-stationary (fail to reject the null hypothesis of unit root).")

# %% [markdown]
# # V1X, V2X 다 정상성이라는 결과가 나왔다. 

# %% [markdown]
# # 3. V1X와 V2X 각각에 대해 ARIMA GARCH 모델을 적용하라.

# %%
# V1X에 대해 ARIMA, GARCH(1,1)을 각각 적용
v1x_series = pd.Series(v1x, index=dates)
# Fit ARIMA model (Example: ARIMA(1, 1, 1))
arima_model = ARIMA(v1x_series, order=(1, 1, 1))
arima_result = arima_model.fit()

# Summary of the ARIMA model
print(arima_result.summary())

# Plot residuals
residuals_v1x = arima_result.resid
plt.figure(figsize=(10, 5))
plt.plot(residuals_v1x, label='ARIMA Residuals')
plt.title('ARIMA Residuals for V1X')
plt.legend()
plt.show()


# %%
# Fit GARCH(1, 1) model to ARIMA residuals
garch_model = arch_model(residuals_v1x, vol='Garch', p=1, q=1)
garch_result = garch_model.fit()

# Summary of the GARCH model
print(garch_result.summary())

# Plot the volatility
plt.figure(figsize=(10, 5))
plt.plot(garch_result.conditional_volatility, label='Conditional Volatility (GARCH)')
plt.title('GARCH Conditional Volatility of V1X')
plt.legend()
plt.show()


# %%
# V2X에 대해 ARIMA, GARCH(1,1)을 각각 적용
v2x_series = pd.Series(v2x, index=dates)
# Fit ARIMA model (Example: ARIMA(1, 1, 1))
arima_model = ARIMA(v2x_series, order=(1, 1, 1))
arima_result = arima_model.fit()

# Summary of the ARIMA model
print(arima_result.summary())

# Plot residuals
residuals_v2x = arima_result.resid
plt.figure(figsize=(10, 5))
plt.plot(residuals_v2x, label='ARIMA Residuals')
plt.title('ARIMA Residuals for V2X')
plt.legend()
plt.show()


# %%
# Fit GARCH(1, 1) model to ARIMA residuals
garch_model = arch_model(residuals_v2x, vol='Garch', p=1, q=1)
garch_result = garch_model.fit()

# Summary of the GARCH model
print(garch_result.summary())

# Plot the volatility
plt.figure(figsize=(10, 5))
plt.plot(garch_result.conditional_volatility, label='Conditional Volatility (GARCH)')
plt.title('GARCH Conditional Volatility of V2X')
plt.legend()
plt.show()


# %% [markdown]
# # 4. V1X와 V2X에 대해 Multivariate GARCH 모델를 적용해보라.

# %%
returns = pd.DataFrame({'V1X': snp_euro['snpprice'], 'V2X': snp_euro['europrice']}).dropna()

# Step 2: Fit Univariate GARCH(1,1) for each series
v1x_garch = arch_model(returns['V1X'], vol='Garch', p=1, q=1).fit(disp="off")
v2x_garch = arch_model(returns['V2X'], vol='Garch', p=1, q=1).fit(disp="off")

# Extract standardized residuals
v1x_std_resid = v1x_garch.resid / v1x_garch.conditional_volatility
v2x_std_resid = v2x_garch.resid / v2x_garch.conditional_volatility

# Combine standardized residuals
std_residuals = np.column_stack((v1x_std_resid, v2x_std_resid))

# Step 3: Initialize DCC-GARCH parameters
T, k = std_residuals.shape
Q_bar = np.cov(std_residuals.T)  # Unconditional covariance matrix
Q = Q_bar.copy()
alpha, beta = 0.05, 0.9  # DCC parameters (to be tuned)
R_matrices = np.zeros((T, k, k))  # Store dynamic correlation matrices

# Step 4: Estimate Dynamic Conditional Correlation (DCC)
for t in range(T):
    # Update Q_t (dynamic covariance matrix)
    outer_product = np.outer(std_residuals[t], std_residuals[t])
    Q = (1 - alpha - beta) * Q_bar + alpha * outer_product + beta * Q

    # Convert Q to correlation matrix R
    diag_Q = np.sqrt(np.diag(Q))
    R = Q / np.outer(diag_Q, diag_Q)  # Normalize to get correlation matrix
    R_matrices[t] = R

# Step 5: Extract and visualize dynamic correlations
dynamic_corr = [R[0, 1] for R in R_matrices]

# Plot dynamic correlations
plt.plot(dynamic_corr)
plt.title("Dynamic Conditional Correlation (DCC) Between V1X and V2X using DCC GARCH(1,1)")
plt.xlabel("Time")
plt.ylabel("Correlation")
plt.show()



# %% [markdown]
# # 5. V1X와 V2X를 이용해 VAR 모델을 만들고, Granger causality 테스트를 실행하고, Impulse response 그래프와 분산 분해(Varinace Decomposion)을 계산하고 그래프로 보여라.
#

# %%
# VAR 모델 생성
model = VAR(returns)

# Select the optimal lag order using AIC
lag_order = model.select_order(maxlags=10)
print("Optimal lag order based on AIC:", lag_order.aic)

# Fit the model
var_model = model.fit(lag_order.aic)
print(var_model.summary())



# %%
# granger-casuality 테스트 진행
print("Granger Causality Test:")
grangercausalitytests(returns, maxlag=lag_order.aic)



# %%
# impulse response 계산
irf = var_model.irf(10)  # Impulse response for 10 steps ahead

# Plot impulse response functions
irf.plot(orth=False)
plt.show()


# %%
# variance decomposition 계산 후 그래프로 표현
fevd = var_model.fevd(10)  # 10-step ahead variance decomposition

# Plot variance decomposition
fevd.plot()
plt.show()

# %% [markdown]
# # 6. V1X와 V2X를 이용해 공적분관계를 확인하고, VECM 모델을 구축하라.

# %%
# 2. Load and Prepare Data
# Assuming V1X and V2X are your time series arrays
# Combine them into a DataFrame
data = returns

# Check stationarity of individual series using differencing (if needed)
data_diff = data.diff().dropna()

# Visualize
data.plot(title="Original Series")
plt.show()

data_diff.plot(title="Differenced Series")
plt.show()


# %%
# 3. Check for Cointegration Using Johansen Test
# Johansen Test for Cointegration
result = coint_johansen(data, det_order=0, k_ar_diff=1)  # det_order=0 assumes no deterministic trend
trace_stat = result.lr1  # Trace statistics
crit_values = result.cvt  # Critical values

# Print results
print("Johansen Cointegration Test")
print(f"Trace Statistics: {trace_stat}")
print(f"Critical Values:\n{crit_values}")

# Check if Trace Statistics > Critical Values for cointegration
if trace_stat[0] > crit_values[0, 1]:  # Compare with 5% critical value
    print("Cointegration exists at the 5% significance level.")
else:
    print("No cointegration at the 5% significance level.")


# %%
# 4. Construct VECM Model
# Once cointegration is confirmed, construct a VECM model.
# Fit VECM model
vecm = VECM(data, k_ar_diff=1, coint_rank=1)  # coint_rank=1 for one cointegrating relationship
vecm_fit = vecm.fit()

# Display VECM summary
print(vecm_fit.summary())


# %%
# 5. Impulse Response and Diagnostics
# After fitting the VECM model, you can analyze impulse responses or conduct model diagnostics.
# Impulse Response Analysis
irf = vecm_fit.irf(10)  # Impulse response for 10 periods
irf.plot()
plt.show()

# Diagnostics (Residuals)
vecm_resid = vecm_fit.resid
plt.plot(vecm_resid)
plt.title("VECM Residuals")
plt.show()


# %% [markdown]
# # 7. 페어트레이딩 (트레이딩 경계: 평균 +/- some 표준편차)
# # (1) 단순 스프레드를 이용한 페어 트레이딩, (2) 공적분관계를 이용한 페어트레이딩, 
# # (3) 칼만 필터를 이용한 페어트레이딩

# %%
data=returns
entry_threshold = 1  # Entry threshold in terms of standard deviations
exit_threshold = 0   # Exit when spread returns to mean


# Function to calculate trading signals
def generate_signals(spread, mean, std_dev, entry_threshold, exit_threshold):
    signals = pd.DataFrame(index=spread.index, columns=['Spread', 'Position'])
    signals['Spread'] = spread

    # Entry and exit signals
    signals['Position'] = np.where(spread > mean + entry_threshold * std_dev, -1,  # Short
                                   np.where(spread < mean - entry_threshold * std_dev, 1,  # Long
                                            0))  # Exit
    return signals


# Pair Trading Using Simple Spread
def pair_trading_simple(data):
    # Calculate spread as the difference between the two series
    spread = data['V1X'] - data['V2X']
    mean = spread.mean()
    std_dev = spread.std()

    # Generate trading signals
    signals = generate_signals(spread, mean, std_dev, entry_threshold, exit_threshold)
    return signals


# Pair Trading Using Cointegration
def pair_trading_coint(data):
    # Cointegration test
    _, p_value, _ = coint(data['V1X'], data['V2X'])
    if p_value > 0.05:
        raise ValueError("No cointegration relationship found")

    # Calculate spread using regression coefficients
    hedge_ratio = np.polyfit(data['V2X'], data['V1X'], 1)[0]
    spread = data['V1X'] - hedge_ratio * data['V2X']
    mean = spread.mean()
    std_dev = spread.std()

    # Generate trading signals
    signals = generate_signals(spread, mean, std_dev, entry_threshold, exit_threshold)
    return signals


# Pair Trading Using Kalman Filter
def pair_trading_kalman(data):
    # Kalman Filter for dynamic hedge ratio
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)

    # Apply Kalman Filter
    spread = []
    hedge_ratios = []
    state_means, _ = kf.filter(data['V1X'].values - data['V2X'].values)
    for i in range(len(data)):
        hedge_ratio = state_means[i][0]
        hedge_ratios.append(hedge_ratio)
        spread.append(data['V1X'].iloc[i] - hedge_ratio * data['V2X'].iloc[i])

    spread = pd.Series(spread, index=data.index)
    mean = spread.mean()
    std_dev = spread.std()

    # Generate trading signals
    signals = generate_signals(spread, mean, std_dev, entry_threshold, exit_threshold)
    return signals, hedge_ratios


# Execute strategies
simple_signals = pair_trading_simple(data)
coint_signals = pair_trading_coint(data)
kalman_signals, kalman_hedge_ratios = pair_trading_kalman(data)
'''
# Plot the spread and trading boundaries for the first strategy (simple spread)
plt.figure(figsize=(12, 6))
plt.plot(simple_signals['Spread'], label='Spread', color='blue')
plt.axhline(simple_signals['Spread'].mean(), color='green', linestyle='--', label='Mean')
plt.axhline(simple_signals['Spread'].mean() + entry_threshold * simple_signals['Spread'].std(),
            color='red', linestyle='--', label='Upper Bound')
plt.axhline(simple_signals['Spread'].mean() - entry_threshold * simple_signals['Spread'].std(),
            color='red', linestyle='--', label='Lower Bound')
plt.title("Simple Spread Pair Trading")
plt.legend()
plt.show()
'''

# %%
# Plot all three spreads as subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows, 1 column

# Simple Spread Plot
axes[0].plot(simple_signals['Spread'], label='Simple Spread', color='blue')
axes[0].axhline(simple_signals['Spread'].mean(), color='green', linestyle='--', label='Mean')
axes[0].axhline(simple_signals['Spread'].mean() + entry_threshold * simple_signals['Spread'].std(),
                color='red', linestyle='--', label='Upper Bound')
axes[0].axhline(simple_signals['Spread'].mean() - entry_threshold * simple_signals['Spread'].std(),
                color='red', linestyle='--', label='Lower Bound')
axes[0].set_title("Simple Spread Pair Trading")
axes[0].legend()
axes[0].grid()

# Cointegration Spread Plot
coint_spread = data['V1X'] - np.polyfit(data['V2X'], data['V1X'], 1)[0] * data['V2X']
axes[1].plot(coint_spread, label='Cointegration Spread', color='orange')
axes[1].axhline(coint_spread.mean(), color='green', linestyle='--', label='Mean')
axes[1].axhline(coint_spread.mean() + entry_threshold * coint_spread.std(),
                color='red', linestyle='--', label='Upper Bound')
axes[1].axhline(coint_spread.mean() - entry_threshold * coint_spread.std(),
                color='red', linestyle='--', label='Lower Bound')
axes[1].set_title("Cointegration Spread Pair Trading")
axes[1].legend()
axes[1].grid()

# Kalman Filter Spread Plot
kalman_spread = pd.Series(kalman_signals['Spread'], index=data.index)
axes[2].plot(kalman_spread, label='Kalman Filter Spread', color='green')
axes[2].axhline(kalman_spread.mean(), color='green', linestyle='--', label='Mean')
axes[2].axhline(kalman_spread.mean() + entry_threshold * kalman_spread.std(),
                color='red', linestyle='--', label='Upper Bound')
axes[2].axhline(kalman_spread.mean() - entry_threshold * kalman_spread.std(),
                color='red', linestyle='--', label='Lower Bound')
axes[2].set_title("Kalman Filter Spread Pair Trading")
axes[2].legend()
axes[2].grid()

# Adjust layout
plt.tight_layout()
plt.show()

