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
# ### 1번 문제

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from scipy.ndimage.interpolation import shift
import scipy.stats as sst
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# %%
pd.set_option('display.float_format', '{:.2f}'.format)


# %%
#BSM 옵션 이론가 계산
def bsprice(S, X, r, q, t_a, sigma, flag):
    d1 = (np.log(S/X) + (r - q + 0.5*sigma**2)*t_a) / (sigma*np.sqrt(t_a))
    d2 = d1 - sigma*np.sqrt(t_a)
    callOrPut = 1 if flag.lower()=='call' else -1
    nd1 = sst.norm.cdf(callOrPut*d1)
    nd2 = sst.norm.cdf(callOrPut*d2)
    price = callOrPut*(S*np.exp(-q*t_a)*nd1 - X*np.exp(-r*t_a)*nd2)
    return price


# %%
# Delta 계산 함수
def delta(S, X, r, q, t_a, sigma, flag):
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma ** 2) * t_a) / (sigma * np.sqrt(t_a))
    return sst.norm.cdf(d1) if flag.lower() == 'call' else sst.norm.cdf(d1) - 1


# %%
def gamma(S, X, r, q, t_a, sigma, flag):
    # t_a는 연 단위 만기시간
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma ** 2) * t_a) / (sigma * np.sqrt(t_a))
    return np.exp(-q * t_a) * sst.norm.pdf(d1) / (S * sigma * np.sqrt(t_a))


# %%
def hedge_cost(S, X, r, mu, q, sigma, t, nsim, M, flag, shares_per_opt, im_vol, optbs):
    
    t_a = t / 360
    # BS 이론가 계산
    theoretical_price = bsprice(S, X, r, q, t_a, im_vol, flag) * shares_per_opt

    # Monte Carlo 시뮬레이션을 이용해 만기 일수 만큼 일별 주가 path 생성
    z = np.random.randn(nsim, t) 
    schange = np.exp((mu - q - 0.5 * sigma ** 2) * (1/360) + sigma * np.sqrt(1/360) * z)
    stock_paths = np.zeros((nsim, t+1))
    stock_paths[:, 0] = S
    stock_paths[:, 1:] = S * np.cumprod(schange, axis=1)  # 누적 주가 경로
    
    # 2. 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))  # 중복 제거 및 정렬
    
    final_costs = np.zeros(nsim)
    
    for i in range(nsim):
        path = stock_paths[i,:]
        
        # 각 시뮬레이션별 누적비용 및 이전 델타 기록
        cumulative_cost = 0.0
        prev_delta = 0.0
        
        for idx, hp in enumerate(hedge_points):
            # 남은 만기 t_remaining = (t - hp)
            t_remaining = t - hp
            tau = t_remaining / 360
            
            # 헷지 시점의 주가
            s_current = path[hp]
            
            # 마지막 헷지 시점(만기)에서는 delta 결정 방식이 다름
            if hp == t:
                # 만기 시점 delta 결정
                # call: S_t >= X (ITM or ATM) -> delta=1 else 0
                # put: S_t >= X (ITM or ATM 이면) -> delta=0 else 1
                if flag.lower() == 'call':
                    if s_current >= X:
                        if optbs.lower() == 'sell':
                            current_delta = 1.0
                        else:
                            current_delta = -1.0
                    else:
                        current_delta = 0.0
                else: # put
                    if s_current >= X:
                        current_delta = 0.0
                    else:
                        if optbs.lower() == 'sell':
                            current_delta = 1.0
                        else:
                            current_delta = -1.0
            else:
                # 만기 전 일반 delta 계산
                current_delta = delta(s_current, X, r, q, tau, sigma, flag)
            
            # 계약 당 주식수 반영
            current_shares = current_delta * shares_per_opt
            prev_shares = prev_delta * shares_per_opt
            
            # 델타 차이만큼 주식 거래 필요
            delta_change = current_shares - prev_shares
            
            # 거래 비용 계산 (매수 시 양(+), 매도 시 음(-))
            # delta_change > 0 -> 매수
            # delta_change < 0 -> 매도
            if delta_change > 0:
                # 매수
                trade_cost = delta_change * s_current
            else:
                # 매도
                # 매도대금: (-delta_change)*s_current
                # 수수료: 매도대금의 0.1% = 0.001 * (-delta_change)*s_current
                # 실질 매도 순유입: (매도대금 - 수수료)
                # 비용 관점에서: 유입금(음수비용) + 수수료(양수비용)
                # trade_cost = - (매도순유입)
                proceeds = (-delta_change)*s_current
                fee = 0.001 * proceeds
                net_inflow = proceeds - fee
                trade_cost = -net_inflow  # 유입이므로 음수
                
            # 이자비용 계산
            # 이자비용 = 이전 누적비용 * r * M (hp=0 일 때는 이전비용=0이므로 0)
            if idx > 0:
                # 이전 헷지 시점과 현재 헷지 시점 사이의 일 수
                delta_t = hp - hedge_points[idx-1]
                interest_cost = cumulative_cost * r * (delta_t / 360)
            else:
                interest_cost = 0.0
          
            # 현재 누적비용 갱신
            cumulative_cost = cumulative_cost + trade_cost + interest_cost
            
            # 다음 헷지를 위해 현재 델타 갱신
            prev_delta = current_delta
        
        # 만기 시점 옵션 상태에 따른 최종 헷지코스트 계산
        s_final = path[-1]
        if flag.lower() == 'call':
            # ITM 또는 ATM: S_T >= X
            if s_final >= X:
                final_cost = cumulative_cost - X*shares_per_opt
            else:
                # OTM
                final_cost = cumulative_cost
        else:
            # put
            # ITM 또는 ATM: S_T <= X
            if s_final <= X:
                final_cost = X*shares_per_opt - cumulative_cost
            else:
                # OTM
                final_cost = cumulative_cost
        
        final_costs[i] = final_cost


    # Moneyness case 분류
    final_prices = stock_paths[:, -1]
    if flag.lower() == 'call':
        atm_case = final_costs[(final_prices < 1.01 * X) & (final_prices > 0.99 * X)]
        itm_case = final_costs[final_prices > X]
        otm_case = final_costs[final_prices < X]
    else:  # put
        atm_case = final_costs[(final_prices < 1.01 * X) & (final_prices > 0.99 * X)]
        itm_case = final_costs[final_prices < X]
        otm_case = final_costs[final_prices > X]


    # 옵션 매수매도 구분
    buyorsell = 1 if optbs.lower()=='sell' else -1

    # mean hedge cost
    mean_hedge_cost = np.mean(final_costs)
    mean_hedge_cost_atm = np.mean(atm_case)
    mean_hedge_cost_itm = np.mean(itm_case)
    mean_hedge_cost_otm = np.mean(otm_case)

    # hedging_pnl
    hedging_pnl = buyorsell * (theoretical_price-final_costs)
    hedging_pnl_atm = buyorsell * (theoretical_price-atm_case)
    hedging_pnl_itm = buyorsell * (theoretical_price-itm_case)
    hedging_pnl_otm = buyorsell * (theoretical_price-otm_case)

    # mean hedge pnl
    mean_hedging_pnl = np.mean(hedging_pnl)
    mean_hedging_pnl_atm = np.mean(hedging_pnl_atm)
    mean_hedging_pnl_itm = np.mean(hedging_pnl_itm)
    mean_hedging_pnl_otm = np.mean(hedging_pnl_otm)
    
    # hedge performance
    performance = np.std(theoretical_price - final_costs) / theoretical_price
    performance_atm = np.std(theoretical_price - atm_case) / theoretical_price
    performance_itm = np.std(theoretical_price - itm_case) / theoretical_price
    performance_otm = np.std(theoretical_price - otm_case) / theoretical_price

    # bias
    bias = np.abs((np.mean(final_costs) - theoretical_price))/theoretical_price
    bias_atm = np.abs((np.mean(atm_case) - theoretical_price))/theoretical_price
    bias_itm = np.abs((np.mean(itm_case) - theoretical_price))/theoretical_price
    bias_otm = np.abs((np.mean(otm_case) - theoretical_price))/theoretical_price

    results = []
    results.append([M, mean_hedge_cost, mean_hedge_cost_atm, mean_hedge_cost_itm, mean_hedge_cost_otm, mean_hedging_pnl, mean_hedging_pnl_atm, mean_hedging_pnl_itm, mean_hedging_pnl_otm, performance, performance_atm, performance_itm, performance_otm, bias, bias_atm, bias_itm, bias_otm])
    df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "mean_hedge_cost", "mean_hedge_cost_atm", "mean_hedge_cost_itm", "mean_hedge_cost_otm", "mean_hedging_pnl", "mean_hedging_pnl_atm", "mean_hedging_pnl_itm", "mean_hedging_pnl_otm", "performance", "performance_atm", "performance_itm", "performance_otm", "bias", "bias_atm", "bias_itm", "bias_otm"])

    
    return (stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    )


# %%
S, X, r, mu, q, sigma, t, nsim, M = 50000, 55000, 0.04, -0.13, 0, 0.4, 180, 1000, 2

im_vol = 0.46
flag = 'call'
optbs = 'sell'
shares_per_opt = 10000


# %%
(stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    )= hedge_cost(S, X, r, mu, q, sigma, t, nsim, M, flag, shares_per_opt, im_vol, optbs)

# %%
theoretical_price

# %%
df

# %%
plt.figure(figsize=(10,10))
sns.distplot(itm_case,color='blue',label='ITM')
plt.axvline(x=itm_case.mean(),color='black',ls='--',lw=4)

sns.distplot(otm_case,color='red',label="OTM")
plt.axvline(x=otm_case.mean(),color='black',ls='--',lw=4)

sns.distplot(atm_case,color='gray',label="ATM")
plt.axvline(x=atm_case.mean(),color='black',ls='--',lw=4)

plt.xlabel("Cost",size=20)
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10,10))
sns.distplot(hedging_pnl_itm,color='blue',label='ITM')
plt.axvline(x=hedging_pnl_itm.mean(),color='black',ls='--',lw=4)

sns.distplot(hedging_pnl_otm,color='red',label="OTM")
plt.axvline(x=hedging_pnl_otm.mean(),color='black',ls='--',lw=4)

sns.distplot(hedging_pnl_atm,color='gray',label="ATM")
plt.axvline(x=hedging_pnl_atm.mean(),color='black',ls='--',lw=4)

plt.xlabel("pnl",size=20)
plt.legend()
plt.show()

# %%
np.random.seed(42)

M_grid = [1, 2, 4, 5, 10, 20, 30, 60]  # 헷지 주기 리스트

# 결과를 저장할 리스트 초기화
results_all = []

# M_grid에 대한 루프 실행
for M_val in M_grid:
    print(f"Processing M = {M_val} days...")
    
    # hedge_cost 함수 호출
    (
       stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    ) = hedge_cost(
        S, X, r, mu, q, sigma, t, nsim, M_val, flag, shares_per_opt, im_vol, optbs
    )
    
    # DataFrame을 리스트에 추가
    results_all.append(df)

# %%
final_df = pd.concat(results_all, ignore_index=True)

# %%
final_df


# %%
def hedge_cost_delta_band(S, X, r, mu, q, sigma, t, nsim, M, flag, shares_per_opt, im_vol, optbs, delta_band=0.05):
    t_a = t / 360
    # BS 이론가 계산
    theoretical_price = bsprice(S, X, r, q, t_a, im_vol, flag) * shares_per_opt

    # Monte Carlo 시뮬레이션을 이용해 만기 일 수 만큼 일별 주가 path 생성
    z = np.random.randn(nsim, t) 
    schange = np.exp((mu - q - 0.5 * sigma ** 2) * (1/360) + sigma * np.sqrt(1/360) * z)
    stock_paths = np.zeros((nsim, t+1))
    stock_paths[:, 0] = S
    stock_paths[:, 1:] = S * np.cumprod(schange, axis=1)  # 누적 주가 경로
    
    # 2. 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))  # 중복 제거 및 정렬
    
    final_costs = np.zeros(nsim)
    
    for i in range(nsim):
        path = stock_paths[i,:]
        
        # 각 시뮬레이션별 누적비용 및 이전 델타 기록
        cumulative_cost = 0.0
        prev_delta = 0.0
        
        for idx, hp in enumerate(hedge_points):
            t_remaining = t - hp
            tau = t_remaining / 360
            
            s_current = path[hp]

            # 만기 시점 델타 결정
            if hp == t:
                # 만기 시점 델타 결정
                if flag.lower() == 'call':
                    if s_current >= X:
                        # 콜 ITM
                        # 롱 포지션이면 +1, 숏 포지션이면 -1 (하지만 원본 로직은 다소 반대였으므로 재정비)
                        current_delta = 1.0 if optbs.lower() == 'sell' else -1.0
                    else:
                        current_delta = 0.0
                else: # put
                    if s_current >= X:
                        current_delta = 0.0
                    else:
                        # 풋 ITM
                        # 롱 포지션: -1, 숏 포지션: +1
                        current_delta = -1.0 if optbs.lower() == 'sell' else 1.0
            else:
                # 만기 전 일반 delta 계산
                current_delta = delta(s_current, X, r, q, tau, sigma, flag)
            
            # 밴드 체크: |current_delta - prev_delta|가 delta_band를 초과할 때만 리밸런싱
            if np.abs(current_delta - prev_delta) > delta_band or hp == t:
                # 계약 당 주식수 반영
                current_shares = current_delta * shares_per_opt
                prev_shares = prev_delta * shares_per_opt

                # 델타 차이만큼 주식 거래 필요
                delta_change = current_shares - prev_shares

                # 거래 비용 계산
                if delta_change > 0:
                    # 매수
                    trade_cost = delta_change * s_current
                else:
                    # 매도
                    proceeds = (-delta_change)*s_current
                    fee = 0.001 * proceeds
                    net_inflow = proceeds - fee
                    trade_cost = -net_inflow  # 유입이므로 음수
                
                # 이자비용 계산
                if idx > 0:
                    # 이전 헷지 시점과 현재 헷지 시점 사이의 일 수
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360)
                else:
                    interest_cost = 0.0

                # 누적비용 갱신
                cumulative_cost = cumulative_cost + trade_cost + interest_cost
                
                # 델타 업데이트(거래 수행 시만 prev_delta 갱신)
                prev_delta = current_delta
            else:
                # 밴드 내에 있으므로 거래 없음, 이자비용만 반영
                if idx > 0:
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360)
                else:
                    interest_cost = 0.0
                
                cumulative_cost = cumulative_cost + interest_cost
                # prev_delta 변화 없음(거래 안함)
        
        # 만기 시점 옵션 상태에 따른 최종 헷지코스트 계산
        s_final = path[-1]
        if flag.lower() == 'call':
            # ITM 또는 ATM: S_T >= X
            if s_final >= X:
                final_cost = cumulative_cost - X*shares_per_opt
            else:
                # OTM
                final_cost = cumulative_cost
        else:
            # put
            # ITM 또는 ATM: S_T <= X
            if s_final <= X:
                final_cost = X*shares_per_opt - cumulative_cost
            else:
                # OTM
                final_cost = cumulative_cost
        
        final_costs[i] = final_cost


    # Moneyness case 분류
    final_prices = stock_paths[:, -1]
    if flag.lower() == 'call':
        atm_case = final_costs[(final_prices < 1.01 * X) & (final_prices > 0.99 * X)]
        itm_case = final_costs[final_prices > X]
        otm_case = final_costs[final_prices < X]
    else:  # put
        atm_case = final_costs[(final_prices < 1.01 * X) & (final_prices > 0.99 * X)]
        itm_case = final_costs[final_prices < X]
        otm_case = final_costs[final_prices > X]


    # 옵션 매수매도 구분
    buyorsell = 1 if optbs.lower()=='sell' else -1

    # mean hedge cost
    mean_hedge_cost = np.mean(final_costs)
    mean_hedge_cost_atm = np.mean(atm_case) if len(atm_case)>0 else np.nan
    mean_hedge_cost_itm = np.mean(itm_case) if len(itm_case)>0 else np.nan
    mean_hedge_cost_otm = np.mean(otm_case) if len(otm_case)>0 else np.nan

    # hedging_pnl
    hedging_pnl = buyorsell * (theoretical_price-final_costs)
    hedging_pnl_atm = buyorsell * (theoretical_price-atm_case) if len(atm_case)>0 else np.array([np.nan])
    hedging_pnl_itm = buyorsell * (theoretical_price-itm_case) if len(itm_case)>0 else np.array([np.nan])
    hedging_pnl_otm = buyorsell * (theoretical_price-otm_case) if len(otm_case)>0 else np.array([np.nan])

    # mean hedge pnl
    mean_hedging_pnl = np.mean(hedging_pnl)
    mean_hedging_pnl_atm = np.mean(hedging_pnl_atm) if len(hedging_pnl_atm)>0 else np.nan
    mean_hedging_pnl_itm = np.mean(hedging_pnl_itm) if len(hedging_pnl_itm)>0 else np.nan
    mean_hedging_pnl_otm = np.mean(hedging_pnl_otm) if len(hedging_pnl_otm)>0 else np.nan
    
    # hedge performance
    performance = np.std(theoretical_price - final_costs) / theoretical_price
    performance_atm = np.std(theoretical_price - atm_case) / theoretical_price if len(atm_case)>0 else np.nan
    performance_itm = np.std(theoretical_price - itm_case) / theoretical_price if len(itm_case)>0 else np.nan
    performance_otm = np.std(theoretical_price - otm_case) / theoretical_price if len(otm_case)>0 else np.nan

    # bias
    bias = np.abs((np.mean(final_costs) - theoretical_price))/theoretical_price
    bias_atm = np.abs((np.mean(atm_case) - theoretical_price))/theoretical_price if len(atm_case)>0 else np.nan
    bias_itm = np.abs((np.mean(itm_case) - theoretical_price))/theoretical_price if len(itm_case)>0 else np.nan
    bias_otm = np.abs((np.mean(otm_case) - theoretical_price))/theoretical_price if len(otm_case)>0 else np.nan

    results = []
    results.append([M, mean_hedge_cost, mean_hedge_cost_atm, mean_hedge_cost_itm, mean_hedge_cost_otm, 
                    mean_hedging_pnl, mean_hedging_pnl_atm, mean_hedging_pnl_itm, mean_hedging_pnl_otm, 
                    performance, performance_atm, performance_itm, performance_otm, 
                    bias, bias_atm, bias_itm, bias_otm])
    df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "mean_hedge_cost", 
                                        "mean_hedge_cost_atm", "mean_hedge_cost_itm", "mean_hedge_cost_otm", 
                                        "mean_hedging_pnl", "mean_hedging_pnl_atm", "mean_hedging_pnl_itm", "mean_hedging_pnl_otm", 
                                        "performance", "performance_atm", "performance_itm", "performance_otm", 
                                        "bias", "bias_atm", "bias_itm", "bias_otm"])

    return (stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    )


# %%
(stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    )= hedge_cost_delta_band(S, X, r, mu, q, sigma, t, nsim, M, flag, shares_per_opt, im_vol, optbs)

# %%
df

# %%
# 결과를 저장할 리스트 초기화
results_all_deltaband = []

# M_grid에 대한 루프 실행
for M_val in M_grid:
    print(f"Processing M = {M_val} days...")
    
    # hedge_cost 함수 호출
    (
       stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    ) = hedge_cost_delta_band(
        S, X, r, mu, q, sigma, t, nsim, M_val, flag, shares_per_opt, im_vol, optbs
    )
    
    # DataFrame을 리스트에 추가
    results_all_deltaband.append(df)

# %%
final_df_deltaband = pd.concat(results_all_deltaband, ignore_index=True)

# %%
final_df_deltaband


# %%
def hedge_cost_delta_gamma_band(S, X, r, mu, q, sigma, t, nsim, M, flag, shares_per_opt, im_vol, optbs, delta_band=0.05, gamma_scale=10.0):
    t_a = t / 360
    # BS 이론가 계산
    theoretical_price = bsprice(S, X, r, q, t_a, im_vol, flag) * shares_per_opt

    # Monte Carlo 시뮬레이션을 이용해 만기 일 수 만큼 일별 주가 path 생성
    z = np.random.randn(nsim, t) 
    schange = np.exp((mu - q - 0.5 * sigma ** 2) * (1/360) + sigma * np.sqrt(1/360) * z)
    stock_paths = np.zeros((nsim, t+1))
    stock_paths[:, 0] = S
    stock_paths[:, 1:] = S * np.cumprod(schange, axis=1)  # 누적 주가 경로
    
    # 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))  # 중복 제거 및 정렬
    
    final_costs = np.zeros(nsim)
    
    for i in range(nsim):
        path = stock_paths[i,:]
        
        # 각 시뮬레이션별 누적비용 및 이전 델타 기록
        cumulative_cost = 0.0
        prev_delta = 0.0
        
        for idx, hp in enumerate(hedge_points):
            t_remaining = t - hp
            tau = t_remaining / 360
            s_current = path[hp]

            # 만기 시점 델타 결정
            if hp == t:
                if flag.lower() == 'call':
                    if s_current >= X:
                        # 콜 ITM
                        current_delta = 1.0 if optbs.lower() == 'sell' else -1.0
                    else:
                        current_delta = 0.0
                else: # put
                    if s_current >= X:
                        current_delta = 0.0
                    else:
                        # 풋 ITM
                        current_delta = -1.0 if optbs.lower() == 'sell' else 1.0
                # 만기는 항상 최종 조정이 필요하므로 밴드 체크 없이 리밸런싱
                effective_delta_band = 0.0
            else:
                # 만기 전 일반 delta/gamma 계산
                current_delta = delta(s_current, X, r, q, tau, sigma, flag)
                current_gamma = gamma(s_current, X, r, q, tau, sigma, flag)
                
                # 감마가 클수록 밴드를 줄여서 더 자주 리밸런싱
                # 예: effective_band = delta_band / (1 + gamma_scale * gamma)
                effective_delta_band = delta_band / (1.0 + gamma_scale * current_gamma)
            
            # 델타 조정 필요성 판단
            if (np.abs(current_delta - prev_delta) > effective_delta_band) or (hp == t):
                # 계약 당 주식수 반영
                current_shares = current_delta * shares_per_opt
                prev_shares = prev_delta * shares_per_opt

                # 델타 차이만큼 주식 거래 필요
                delta_change = current_shares - prev_shares

                # 거래 비용 계산
                if delta_change > 0:
                    # 매수
                    trade_cost = delta_change * s_current
                else:
                    # 매도
                    proceeds = (-delta_change)*s_current
                    fee = 0.001 * proceeds
                    net_inflow = proceeds - fee
                    trade_cost = -net_inflow  # 유입이므로 음수
                
                # 이자비용 계산
                if idx > 0:
                    # 이전 헷지 시점과 현재 헷지 시점 사이의 일 수
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360)
                else:
                    interest_cost = 0.0

                # 누적비용 갱신
                cumulative_cost = cumulative_cost + trade_cost + interest_cost
                
                # 델타 업데이트(거래 수행 시만 prev_delta 갱신)
                prev_delta = current_delta
            else:
                # 밴드 내에 있으므로 거래 없음, 이자비용만 반영
                if idx > 0:
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360)
                else:
                    interest_cost = 0.0
                cumulative_cost = cumulative_cost + interest_cost
                # prev_delta 변화 없음(거래 안함)
        
        # 만기 시점 옵션 상태에 따른 최종 헷지코스트 계산
        s_final = path[-1]
        if flag.lower() == 'call':
            # ITM 또는 ATM: S_T >= X
            if s_final >= X:
                final_cost = cumulative_cost - X*shares_per_opt
            else:
                # OTM
                final_cost = cumulative_cost
        else:
            # put
            # ITM 또는 ATM: S_T <= X
            if s_final <= X:
                final_cost = X*shares_per_opt - cumulative_cost
            else:
                # OTM
                final_cost = cumulative_cost
        
        final_costs[i] = final_cost


    # Moneyness case 분류
    final_prices = stock_paths[:, -1]
    if flag.lower() == 'call':
        atm_case = final_costs[(final_prices < 1.01 * X) & (final_prices > 0.99 * X)]
        itm_case = final_costs[final_prices > X]
        otm_case = final_costs[final_prices < X]
    else:  # put
        atm_case = final_costs[(final_prices < 1.01 * X) & (final_prices > 0.99 * X)]
        itm_case = final_costs[final_prices < X]
        otm_case = final_costs[final_prices > X]

    # 옵션 매수매도 구분
    buyorsell = 1 if optbs.lower()=='sell' else -1

    # mean hedge cost
    mean_hedge_cost = np.mean(final_costs)
    mean_hedge_cost_atm = np.mean(atm_case) if len(atm_case)>0 else np.nan
    mean_hedge_cost_itm = np.mean(itm_case) if len(itm_case)>0 else np.nan
    mean_hedge_cost_otm = np.mean(otm_case) if len(otm_case)>0 else np.nan

    # hedging_pnl
    hedging_pnl = buyorsell * (theoretical_price-final_costs)
    hedging_pnl_atm = buyorsell * (theoretical_price - atm_case) if len(atm_case)>0 else np.array([np.nan])
    hedging_pnl_itm = buyorsell * (theoretical_price - itm_case) if len(itm_case)>0 else np.array([np.nan])
    hedging_pnl_otm = buyorsell * (theoretical_price - otm_case) if len(otm_case)>0 else np.array([np.nan])

    # mean hedge pnl
    mean_hedging_pnl = np.mean(hedging_pnl)
    mean_hedging_pnl_atm = np.mean(hedging_pnl_atm) if len(hedging_pnl_atm)>0 else np.nan
    mean_hedging_pnl_itm = np.mean(hedging_pnl_itm) if len(hedging_pnl_itm)>0 else np.nan
    mean_hedging_pnl_otm = np.mean(hedging_pnl_otm) if len(hedging_pnl_otm)>0 else np.nan
    
    # hedge performance
    performance = np.std(theoretical_price - final_costs) / theoretical_price
    performance_atm = np.std(theoretical_price - atm_case) / theoretical_price if len(atm_case)>0 else np.nan
    performance_itm = np.std(theoretical_price - itm_case) / theoretical_price if len(itm_case)>0 else np.nan
    performance_otm = np.std(theoretical_price - otm_case) / theoretical_price if len(otm_case)>0 else np.nan

    # bias
    bias = np.abs((np.mean(final_costs) - theoretical_price))/theoretical_price
    bias_atm = np.abs((np.mean(atm_case) - theoretical_price))/theoretical_price if len(atm_case)>0 else np.nan
    bias_itm = np.abs((np.mean(itm_case) - theoretical_price))/theoretical_price if len(itm_case)>0 else np.nan
    bias_otm = np.abs((np.mean(otm_case) - theoretical_price))/theoretical_price if len(otm_case)>0 else np.nan

    results = []
    results.append([M, mean_hedge_cost, mean_hedge_cost_atm, mean_hedge_cost_itm, mean_hedge_cost_otm, 
                    mean_hedging_pnl, mean_hedging_pnl_atm, mean_hedging_pnl_itm, mean_hedging_pnl_otm, 
                    performance, performance_atm, performance_itm, performance_otm, 
                    bias, bias_atm, bias_itm, bias_otm])
    df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "mean_hedge_cost", 
                                        "mean_hedge_cost_atm", "mean_hedge_cost_itm", "mean_hedge_cost_otm", 
                                        "mean_hedging_pnl", "mean_hedging_pnl_atm", "mean_hedging_pnl_itm", "mean_hedging_pnl_otm", 
                                        "performance", "performance_atm", "performance_itm", "performance_otm", 
                                        "bias", "bias_atm", "bias_itm", "bias_otm"])

    return (stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    )


# %%
# 결과를 저장할 리스트 초기화
results_all_deltagammaband = []

# M_grid에 대한 루프 실행
for M_val in M_grid:
    print(f"Processing M = {M_val} days...")
    
    # hedge_cost 함수 호출
    (
       stock_paths, 
            theoretical_price, 
            final_costs, 
            atm_case, 
            itm_case, 
            otm_case, 
            mean_hedge_cost, 
            mean_hedge_cost_atm, 
            mean_hedge_cost_itm, 
            mean_hedge_cost_otm, 
            hedging_pnl, 
            hedging_pnl_atm, 
            hedging_pnl_itm, 
            hedging_pnl_otm, 
            mean_hedging_pnl, 
            mean_hedging_pnl_atm, 
            mean_hedging_pnl_itm, 
            mean_hedging_pnl_otm, 
            performance, 
            performance_atm, 
            performance_itm, 
            performance_otm, 
            bias, 
            bias_atm, 
            bias_itm, 
            bias_otm, 
            df
    ) = hedge_cost_delta_band(
        S, X, r, mu, q, sigma, t, nsim, M_val, flag, shares_per_opt, im_vol, optbs
    )
    
    # DataFrame을 리스트에 추가
    results_all_deltagammaband.append(df)

# %%
final_df_deltagammaband = pd.concat(results_all_deltagammaband, ignore_index=True)

# %%
final_df_deltagammaband

# %% [markdown]
# ## 2번문제

# %%
from pathlib import Path
# OUTPUT_PATH = Path(r"C:\Users\USER\OneDrive\바탕 화면\임우섭\01. 카이스트\02. 입학후\03. 24-2학기\08. 파생상품 거래전략\04. 과제\02. Term project")

CWD_PATH = Path.cwd()
WS_PATH = CWD_PATH.parent
OUTPUT_PATH = WS_PATH / "output"
DATA_PATH = WS_PATH / "data"

# %%
df = pd.read_parquet(OUTPUT_PATH / 'option_data_targets_20220101-20241204.parquet')

# %%
print(df.columns)

# %%
# df.to_excel(r"C:\Users\USER\OneDrive\바탕 화면\임우섭\01. 카이스트\02. 입학후\03. 24-2학기\08. 파생상품 거래전략\04. 과제\02. Term project\output.xlsx", index=False)

# %%



samsung_data = df[df['underlying_full'] == '삼성전자']
# 각 trade_date 기준으로 첫 번째 행에서 udly_close 값 추출
samsumg_df = samsung_data.groupby('trade_date').first().reset_index()[['trade_date', 'udly_close', 'udly_return']]
samsumg_df

# %% [markdown]
# 어차피 옵션데이터 안 쓴다... 걍 yfinance에서 불러오자

# %%
# import yfinance as yf
from arch import arch_model
# samsung_yf = yf.download('005930.KS', start='2000-01-04', end='2024-12-10')
# samsung_yf

# %% [markdown]
# ### 여기만 바꿔주면 됨. 여기서 종목 선정도 함. 

# %%
all_udly_df = pd.read_pickle(OUTPUT_PATH / 'all_udly_df.pkl')

# %%
SELECTION = [
    '삼성전자',
    '현대모비스',
    'NAVER',
    '카카오'
]

# %%
all_udly_df = all_udly_df[all_udly_df['underlying_full'].isin(SELECTION)]

CURRRENT_SELECTION = '카카오'

all_udly_df = all_udly_df[all_udly_df['underlying_full'] == CURRRENT_SELECTION]

# %%
# samsung_df = pd.read_pickle(OUTPUT_PATH / 'samsung_df.pkl')

samsung_yf = pd.DataFrame()
samsung_yf['Adj Close'] = all_udly_df['udly_close']

# %%
# 로그수익률 계산
samsung_yf['ret'] = np.log(samsung_yf['Adj Close'] / samsung_yf['Adj Close'].shift(1))
samsung_yf = samsung_yf.dropna(subset=['ret'])

# 결과를 저장할 컬럼 추가 (초기값 NaN)
samsung_yf['garch_vol_pred'] = np.nan
samsung_yf['ewma_vol_pred'] = np.nan

lambda_ = 0.94

# EWMA 초기값 설정 (처음 몇 기간에 대한 초기화 필요)
# 초기 구간: 첫 N일간의 분산 사용(여기서는 모든 데이터가 아니라 초기 구간)
initial_period = 30
initial_var = samsung_yf['ret'].iloc[:initial_period].var()
sigma2_ewma = initial_var

# GARCH, EWMA 예측치는 t일 종가 이후 t+1일 변동성 예측값을 t일자 행에 기록한다고 가정.
for i in range(initial_period, len(samsung_yf)-1):
    # [EWMA 업데이트]
    # i번째 수익률까지 고려한 EWMA 추정
    r = samsung_yf['ret'].iloc[i]
    sigma2_ewma = lambda_ * sigma2_ewma + (1 - lambda_) * (r**2)
    ewma_pred = np.sqrt(sigma2_ewma)
    
    # [GARCH 업데이트]
    # GARCH는 0~i일까지의 데이터를 사용해 모델 피팅 후, i+1일 변동성 예측
    # (rolling fitting)
    data_for_garch = samsung_yf['ret'].iloc[:i+1]  # 0~i까지 데이터
    model = arch_model(data_for_garch, mean='Zero', vol='GARCH', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=1)
    garch_var = forecast.variance.values[-1,0]
    garch_pred = np.sqrt(garch_var)
    
    # 예측값을 DataFrame에 저장
    # t일 종가 이후 t+1일 변동성 예측치이므로 i번째 인덱스에 기록
    samsung_yf['ewma_vol_pred'].iloc[i] = ewma_pred
    samsung_yf['garch_vol_pred'].iloc[i] = garch_pred

# 마지막 행은 다음날 예측 불가하므로 NaN 유지
# 필요한 경우 dropna 처리


# %%
samsung_yf['roll_90_vol']=samsung_yf['ret'].rolling(90).std()*np.sqrt(252)
samsung_yf

# %%
samsung_yf['garch_vol_pred'] = samsung_yf['garch_vol_pred' ]*np.sqrt(252)
samsung_yf['ewma_vol_pred'] = samsung_yf['ewma_vol_pred' ]*np.sqrt(252)
samsung_yf

# %%
samsung_yf['10%_OTM_K'] = samsung_yf['Adj Close'] * 0.9
samsung_yf['roll_63_vol']=samsung_yf['ret'].rolling(63).std()*np.sqrt(252)
samsung_yf['roll_126_vol']=samsung_yf['ret'].rolling(126).std()*np.sqrt(252)
samsung_yf['sell_imvol']=samsung_yf['roll_126_vol'] * 1.15
samsung_yf['sell_premium']=bsprice(S=samsung_yf['Adj Close'], X=samsung_yf['10%_OTM_K'], r=0.04, q=0, t_a=126/252, sigma=samsung_yf['sell_imvol'], flag='call')

# %%
import matplotlib.pyplot as plt

# 2024-06-03 이후의 데이터 필터링
filtered_data = samsung_yf.loc['2024-06-03':'2024-12-03']

# 'Adj Close' 컬럼 선택
adj_close_values = filtered_data['Adj Close']

# 경로(Path) 그리기
plt.figure(figsize=(10, 6))
plt.plot(adj_close_values, marker='', linestyle='-', label='Adj Close Price')
plt.title('Samsung Adj Close Price (Filtered from 2024-06-03)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Adj Close Price', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# %%
def hedge_cost_real_path(yf_data, start_date, t, M, flag, shares_per_opt, optbs, r, q, sigma_col):
    """
    실제 주가 경로(yf_data['Adj Close'])와 날짜별 변동성(sigma_col)을 사용하여 
    다이나믹 헷지를 수행하는 함수.
    
    잔여변동성 측정:
    각 헷지 시점마다 BSM 이론가와 누적 헷지 비용(cumulative_cost)의 차이를 측정하고,
    이 차이들의 표준편차를 잔여 변동성(Residual Volatility)로 정의한다.
    """

    # start_date 기준으로 행사가격 10% OTM K 가져오기
    X = yf_data.loc[start_date, '10%_OTM_K']

    # 사용 날짜 인덱스 생성
    all_dates = yf_data.loc[start_date:].index[:t+1]  # t+1개 행: start_date 포함
    # 주가 경로 및 변동성 추출
    adj_close_path = yf_data.loc[all_dates, 'Adj Close'].values
    sigmas = yf_data.loc[all_dates, sigma_col].values

    # 시작 시점 이론가: start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    s_start = adj_close_path[0]
    end_date = all_dates[-1]
    final_vol = yf_data.loc[end_date, 'roll_126_vol']
    t_a = t / 252.0  # 초기 만기시간(연 단위, 252거래일 가정)
    theoretical_price_realvol = bsprice(s_start, X, r, q, t_a, final_vol, flag) * shares_per_opt

    # 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))

    cumulative_cost = 0.0
    prev_delta = 0.0

    records = []

    # 잔여 오차 기록용 리스트
    residual_errors = []

    for idx, hp in enumerate(hedge_points):
        t_remaining = t - hp
        # 주석상 360일을 사용 중이었으나 앞서 252일 사용 -> 일관성 있게 252일 사용 권장
        # 여기서는 기존 코드 유지하되, 주석 수정:
        # tau = t_remaining / 360.0  # 원 코드 유지
        
        # tau = t_remaining / 360.0
        tau = t_remaining / 252 # 수정 by Jaepil

        s_current = adj_close_path[hp]
        current_sigma = sigmas[hp]

        # 만기 시점 델타 결정
        if hp == t:
            if flag.lower() == 'call':
                if s_current >= X:
                    current_delta = 1.0 if optbs.lower() == 'sell' else -1.0
                else:
                    current_delta = 0.0
            else: # put
                if s_current >= X:
                    current_delta = 0.0
                else:
                    current_delta = 1.0 if optbs.lower() == 'sell' else -1.0
        else:
            # 만기 전 델타 계산
            current_delta = delta(s_current, X, r, q, tau, current_sigma, flag)

        current_shares = current_delta * shares_per_opt
        prev_shares = prev_delta * shares_per_opt
        delta_change = current_shares - prev_shares

        # 거래 비용 계산
        if delta_change > 0:
            trade_cost = delta_change * s_current
        else:
            proceeds = (-delta_change) * s_current
            fee = 0.001 * proceeds
            net_inflow = proceeds - fee
            trade_cost = -net_inflow

        if idx > 0:
            delta_t = hp - hedge_points[idx-1]
            interest_cost = cumulative_cost * r * (delta_t / 360.0)
        else:
            interest_cost = 0.0

        cumulative_cost += (trade_cost + interest_cost)

        # 현재 시점 BSM 이론가 계산 (현재 주가, current_sigma, 남은 만기 tau)
        if tau > 0:
            current_theoretical = bsprice(s_current, X, r, q, tau, current_sigma, flag) * shares_per_opt
        else:
            # 만기일 경우 tau=0, 이론가는 내재가치
            intrinsic = max(s_current - X, 0) if flag.lower()=='call' else max(X - s_current, 0)
            current_theoretical = intrinsic * shares_per_opt

        # 잔여 오차: 이론가 - 누적헷지비용
        residual_error = current_theoretical - cumulative_cost
        residual_errors.append(residual_error)

        records.append({
            'date': all_dates[hp],
            's_current': s_current,
            'delta': current_delta,
            'delta_change': delta_change,
            'trade_cost': trade_cost,
            'interest_cost': interest_cost,
            'cumulative_cost': cumulative_cost,
            'current_theoretical': current_theoretical,
            'residual_error': residual_error
        })

        prev_delta = current_delta

    # 만기 시점 payoff 반영
    s_final = adj_close_path[-1]
    if flag.lower() == 'call':
        if s_final >= X:
            final_cost = cumulative_cost - X * shares_per_opt
        else:
            final_cost = cumulative_cost
    else:
        if s_final <= X:
            final_cost = X * shares_per_opt - cumulative_cost
        else:
            final_cost = cumulative_cost

    df_result = pd.DataFrame(records)

    # 잔여 변동성: residual_errors 표준편차
    residual_volatility = np.std(residual_errors)

    buyorsell = 1 if optbs.lower()=='sell' else -1
    hedging_pnl = buyorsell * (theoretical_price - final_cost)
    realvol_pnl = buyorsell * (theoretical_price - theoretical_price_realvol)

    results = []
    results.append([M, final_cost, hedging_pnl, theoretical_price, realvol_pnl, theoretical_price_realvol, residual_volatility])
    df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "hedge_cost", "hedging_pnl", 
                                        "theoretical_price", "realvol_pnl", "theoretical_price_realvol", "residual_volatility"])

    return cumulative_cost, final_cost, df_result, theoretical_price, df



# %%
## 날짜 부족해서 시작일 수정
START_DATE_NEW = '2024-05-02'

cumulative_cost, final_cost, df_result, theoretical_price, df = hedge_cost_real_path(yf_data=samsung_yf, start_date=START_DATE_NEW, t=126, M=10, flag="call", shares_per_opt=100000, optbs="sell", r=0.04, q=0.00, sigma_col="garch_vol_pred")

# %%
# 가정: hedge_cost_real_path 함수, samsung_yf DataFrame, 그리고 필요한 함수/모듈 이미 로드 완료

M_list = [1, 5, 10, 20, 30]
sigma_cols = ['garch_vol_pred', 'ewma_vol_pred', 'roll_90_vol', 'roll_63_vol', 'roll_126_vol']

# 결과를 저장할 리스트
results_all = []

for M_val in M_list:
    for sigma_val in sigma_cols:
        cumulative_cost, final_cost, df_result, theoretical_price, df = hedge_cost_real_path(
            yf_data=samsung_yf, 
            start_date=START_DATE_NEW, 
            t=126, 
            M=M_val, 
            flag="call", 
            shares_per_opt=100000, 
            optbs="sell", 
            r=0.04, 
            q=0.00, 
            sigma_col=sigma_val
        )
        
        # df에 어떤 M, sigma_col을 썼는지 표시하는 컬럼 추가
        df['M'] = M_val
        df['sigma_col'] = sigma_val
        
        # 리스트에 df 추가
        results_all.append(df)

# 모든 df를 하나의 DataFrame으로 합치기
final_results = pd.concat(results_all, ignore_index=True)

# final_results를 확인
final_results


# %%

# %%
def hedge_cost_delta_band_real_path(yf_data, start_date, t, M, flag, shares_per_opt, optbs, r, q, sigma_col, delta_band=0.05):
    # start_date 기준으로 행사가격 10% OTM K 가져오기
    X = yf_data.loc[start_date, '10%_OTM_K']

    # 사용 날짜 인덱스 생성
    all_dates = yf_data.loc[start_date:].index[:t+1]  # t+1개 행: start_date 포함, start_date+ t일 뒤까지
    # 주가 경로 및 변동성 추출
    adj_close_path = yf_data.loc[all_dates, 'Adj Close'].values
    sigmas = yf_data.loc[all_dates, sigma_col].values

    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    s_start = adj_close_path[0]
    end_date = all_dates[-1]
    final_vol = yf_data.loc[end_date, 'roll_126_vol']
    t_a = t / 252.0  # 초기 만기시간 기준
    theoretical_price_realvol = bsprice(s_start, X, r, q, t_a, final_vol, flag) * shares_per_opt

    # 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))

    cumulative_cost = 0.0
    prev_delta = 0.0

    records = []

    for idx, hp in enumerate(hedge_points):
        t_remaining = t - hp
        tau = t_remaining / 360.0
        s_current = adj_close_path[hp]
        current_sigma = sigmas[hp]

        # 만기 시점 델타 결정
        if hp == t:
            if flag.lower() == 'call':
                if s_current >= X:
                    current_delta = 1.0 if optbs.lower()=='sell' else -1.0
                else:
                    current_delta = 0.0
            else: # put
                if s_current >= X:
                    current_delta = 0.0
                else:
                    current_delta = 1.0 if optbs.lower()=='sell' else -1.0
        else:
            # 만기 전 델타 계산 (BSM 델타)
            current_delta = delta(s_current, X, r, q, tau, current_sigma, flag)

        # 델타 밴드 체크
        if (abs(current_delta - prev_delta) > delta_band) or (hp == t):
            # 거래 수행
            current_shares = current_delta * shares_per_opt
            prev_shares = prev_delta * shares_per_opt
            delta_change = current_shares - prev_shares

            if delta_change > 0:
                trade_cost = delta_change * s_current
            else:
                proceeds = (-delta_change)*s_current
                fee = 0.001 * proceeds
                net_inflow = proceeds - fee
                trade_cost = -net_inflow

            if idx > 0:
                delta_t = hp - hedge_points[idx-1]
                interest_cost = cumulative_cost * r * (delta_t / 360.0)
            else:
                interest_cost = 0.0

            cumulative_cost += (trade_cost + interest_cost)
            prev_delta = current_delta
        else:
            # 거래 없음, 이자비용만
            if idx > 0:
                delta_t = hp - hedge_points[idx-1]
                interest_cost = cumulative_cost * r * (delta_t / 360.0)
            else:
                interest_cost = 0.0
            cumulative_cost += interest_cost

        records.append({
            'date': all_dates[hp],
            's_current': s_current,
            'delta': current_delta,
            'delta_change': (current_delta - prev_delta)*shares_per_opt if (abs(current_delta - prev_delta) > delta_band or hp == t) else 0,
            'trade_cost': trade_cost if (abs(current_delta - prev_delta) > delta_band or hp == t) else 0.0,
            'interest_cost': interest_cost,
            'cumulative_cost': cumulative_cost
        })

    # 만기 payoff 반영
    s_final = adj_close_path[-1]
    if flag.lower() == 'call':
        if s_final >= X:
            final_cost = cumulative_cost - X * shares_per_opt
        else:
            final_cost = cumulative_cost
    else:
        if s_final <= X:
            final_cost = X * shares_per_opt - cumulative_cost
        else:
            final_cost = cumulative_cost

    df_result = pd.DataFrame(records)

    buyorsell = 1 if optbs.lower()=='sell' else -1
    hedging_pnl = buyorsell * (theoretical_price - final_cost)
    realvol_pnl = buyorsell * (theoretical_price - theoretical_price_realvol)

    results = []
    results.append([M, final_cost, hedging_pnl, theoretical_price, realvol_pnl, theoretical_price_realvol])
    df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "hedge_cost", "hedging_pnl", "theoretical_price", "realvol_pnl", "theoretical_price_realvol"])

    return cumulative_cost, final_cost, df_result, theoretical_price, df




# %%
def hedge_cost_delta_gamma_band_real_path(yf_data, start_date, t, M, flag, shares_per_opt, optbs, r, q, sigma_col, delta_band=0.05, gamma_scale=1.0):
    # start_date 기준으로 행사가격 10% OTM K 가져오기
    X = yf_data.loc[start_date, '10%_OTM_K']

    # 사용 날짜 인덱스 생성
    all_dates = yf_data.loc[start_date:].index[:t+1]  # t+1개 행
    # 주가 경로 및 변동성 추출
    adj_close_path = yf_data.loc[all_dates, 'Adj Close'].values
    sigmas = yf_data.loc[all_dates, sigma_col].values

    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    s_start = adj_close_path[0]
    end_date = all_dates[-1]
    final_vol = yf_data.loc[end_date, 'roll_126_vol']
    t_a = t / 252.0  # 초기 만기시간 기준
    theoretical_price_realvol = bsprice(s_start, X, r, q, t_a, final_vol, flag) * shares_per_opt

    # 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))

    cumulative_cost = 0.0
    prev_delta = 0.0

    records = []

    for idx, hp in enumerate(hedge_points):
        t_remaining = t - hp
        tau = t_remaining / 252.0
        s_current = adj_close_path[hp]
        current_sigma = sigmas[hp]

        if hp == t:
            if flag.lower() == 'call':
                if s_current >= X:
                    current_delta = 1.0 if optbs.lower()=='sell' else -1.0
                else:
                    current_delta = 0.0
            else:
                if s_current >= X:
                    current_delta = 0.0
                else:
                    current_delta = 1.0 if optbs.lower()=='sell' else -1.0
            effective_delta_band = 0.0
        else:
            current_delta = delta(s_current, X, r, q, tau, current_sigma, flag)
            current_gamma = gamma(s_current, X, r, q, tau, current_sigma, flag)
            effective_delta_band = delta_band / (1.0 + gamma_scale * current_gamma)

        if (abs(current_delta - prev_delta) > effective_delta_band) or (hp == t):
            current_shares = current_delta * shares_per_opt
            prev_shares = prev_delta * shares_per_opt
            delta_change = current_shares - prev_shares

            if delta_change > 0:
                trade_cost = delta_change * s_current
            else:
                proceeds = (-delta_change)*s_current
                fee = 0.001 * proceeds
                net_inflow = proceeds - fee
                trade_cost = -net_inflow

            if idx > 0:
                delta_t = hp - hedge_points[idx-1]
                interest_cost = cumulative_cost * r * (delta_t / 360.0)
            else:
                interest_cost = 0.0

            cumulative_cost += (trade_cost + interest_cost)
            prev_delta = current_delta
        else:
            # 거래 없음
            if idx > 0:
                delta_t = hp - hedge_points[idx-1]
                interest_cost = cumulative_cost * r * (delta_t / 360.0)
            else:
                interest_cost = 0.0
            cumulative_cost += interest_cost
            trade_cost = 0.0
            delta_change = 0.0

        records.append({
            'date': all_dates[hp],
            's_current': s_current,
            'delta': current_delta,
            'delta_change': delta_change,
            'trade_cost': trade_cost,
            'interest_cost': interest_cost,
            'cumulative_cost': cumulative_cost
        })

    s_final = adj_close_path[-1]
    if flag.lower() == 'call':
        if s_final >= X:
            final_cost = cumulative_cost - X * shares_per_opt
        else:
            final_cost = cumulative_cost
    else:
        if s_final <= X:
            final_cost = X * shares_per_opt - cumulative_cost
        else:
            final_cost = cumulative_cost

    df_result = pd.DataFrame(records)

    buyorsell = 1 if optbs.lower()=='sell' else -1
    hedging_pnl = buyorsell * (theoretical_price - final_cost)
    realvol_pnl = buyorsell * (theoretical_price - theoretical_price_realvol)

    results = []
    results.append([M, final_cost, hedging_pnl, theoretical_price, realvol_pnl, theoretical_price_realvol])
    df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "hedge_cost", "hedging_pnl", "theoretical_price", "realvol_pnl", "theoretical_price_realvol"])

    return cumulative_cost, final_cost, df_result, theoretical_price, df



# %%
cumulative_cost, final_cost, df_result, theoretical_price, df = hedge_cost_delta_band_real_path(yf_data=samsung_yf, start_date=START_DATE_NEW, t=126, M=10, flag="call", shares_per_opt=100000, optbs="sell", r=0.04, q=0.00, sigma_col="roll_126_vol")

# %%
df

# %%
results_all = []

for M_val in M_list:
    for sigma_val in sigma_cols:
        cumulative_cost, final_cost, df_result, theoretical_price, df = hedge_cost_delta_band_real_path(
            yf_data=samsung_yf, 
            start_date=START_DATE_NEW, 
            t=126, 
            M=M_val, 
            flag="call", 
            shares_per_opt=100000, 
            optbs="sell", 
            r=0.04, 
            q=0.00, 
            sigma_col=sigma_val
        )
        
        # df에 어떤 M, sigma_col을 썼는지 표시하는 컬럼 추가
        df['M'] = M_val
        df['sigma_col'] = sigma_val
        
        # 리스트에 df 추가
        results_all.append(df)

# 모든 df를 하나의 DataFrame으로 합치기
final_results = pd.concat(results_all, ignore_index=True)

# final_results를 확인
final_results.sort_values('hedge_cost')

# %%
cumulative_cost, final_cost, df_result, theoretical_price, df = hedge_cost_delta_gamma_band_real_path(yf_data=samsung_yf, start_date=START_DATE_NEW, t=126, M=10, flag="call", shares_per_opt=100000, optbs="sell", r=0.04, q=0.00, sigma_col="roll_126_vol")

# %%
df

# %%
results_all = []

for M_val in M_list:
    for sigma_val in sigma_cols:
        cumulative_cost, final_cost, df_result, theoretical_price, df = hedge_cost_delta_gamma_band_real_path(
            yf_data=samsung_yf, 
            start_date=START_DATE_NEW, 
            t=126, 
            M=M_val, 
            flag="call", 
            shares_per_opt=100000, 
            optbs="sell", 
            r=0.04, 
            q=0.00, 
            sigma_col=sigma_val
        )
        
        # df에 어떤 M, sigma_col을 썼는지 표시하는 컬럼 추가
        df['M'] = M_val
        df['sigma_col'] = sigma_val
        
        # 리스트에 df 추가
        results_all.append(df)

# 모든 df를 하나의 DataFrame으로 합치기
final_results = pd.concat(results_all, ignore_index=True)

# final_results를 확인
final_results

# %%
BENCHMARK_HEDGE_COST = final_results.sort_values('hedge_cost').iloc[0]['hedge_cost']
BENCHMARK_HEDGE_COST

# %% [markdown]
# ## 재필 모멘텀 시그널 추가

# %%
samsung_signals = pd.read_pickle(OUTPUT_PATH / 'samsung_df.pkl')
samsung_signals = samsung_signals[
    [
        'mom10',
        'mom9',
        'mom7',
    ]
].copy()

new_samsung_yf = samsung_yf.copy()
new_samsung_yf = new_samsung_yf.join(samsung_signals)

# %%
samsung_signals.loc[START_DATE_NEW:, :].hist()


# %%
def hedge_cost_delta_gamma_band_real_path_temp(yf_data, start_date, t, M, flag, shares_per_opt, optbs, r, q, sigma_col, delta_band=0.05):
    # start_date 기준으로 행사가격 10% OTM K 가져오기
    X = yf_data.loc[start_date, '10%_OTM_K']

    # 사용 날짜 인덱스 생성
    all_dates = yf_data.loc[start_date:].index[:t+1]  # t+1개 행
    # 주가 경로 및 변동성 추출
    adj_close_path = yf_data.loc[all_dates, 'Adj Close'].values
    sigmas = yf_data.loc[all_dates, sigma_col].values

    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    s_start = adj_close_path[0]
    end_date = all_dates[-1]
    final_vol = yf_data.loc[end_date, 'roll_126_vol']
    t_a = t / 252.0  # 초기 만기시간 기준
    theoretical_price_realvol = bsprice(s_start, X, r, q, t_a, final_vol, flag) * shares_per_opt


    ### 모멘텀 시그널들
    signal1 = yf_data.loc[all_dates, 'mom10'].values
    signal2 = yf_data.loc[all_dates, 'mom9'].values
    signal3 = yf_data.loc[all_dates, 'mom7'].values


    # 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))


    def hedge(edb_func=None, name=None, signal=None):

        cumulative_cost = 0.0
        prev_delta = 0.0

        records = []

        for idx, hp in enumerate(hedge_points):
            t_remaining = t - hp
            tau = t_remaining / 252.0
            s_current = adj_close_path[hp]
            current_sigma = sigmas[hp]

            ## 재필 시그널들
            if signal is not None:
                signal1_current = signal[hp]
            else:
                signal1_current = None



            if hp == t:
                if flag.lower() == 'call':
                    if s_current >= X:
                        current_delta = 1.0 if optbs.lower()=='sell' else -1.0
                    else:
                        current_delta = 0.0
                else:
                    if s_current >= X:
                        current_delta = 0.0
                    else:
                        current_delta = 1.0 if optbs.lower()=='sell' else -1.0
                effective_delta_band = 0.0

            
            else:
                current_delta = delta(s_current, X, r, q, tau, current_sigma, flag)
                current_gamma = gamma(s_current, X, r, q, tau, current_sigma, flag)

                ### Signal 추가하는 부분
                # effective_delta_band = delta_band / (1.0 + current_sigma * current_gamma)
                effective_delta_band = edb_func(delta_band, s_current, X, current_delta, current_sigma, current_gamma, signal1_current)

                # if idx == 10:
                #     print(effective_delta_band)

            if (abs(current_delta - prev_delta) > effective_delta_band) or (hp == t):
                current_shares = current_delta * shares_per_opt
                prev_shares = prev_delta * shares_per_opt
                delta_change = current_shares - prev_shares

                if delta_change > 0:
                    trade_cost = delta_change * s_current
                else:
                    proceeds = (-delta_change)*s_current
                    fee = 0.001 * proceeds
                    net_inflow = proceeds - fee
                    trade_cost = -net_inflow

                if idx > 0:
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360.0)
                else:
                    interest_cost = 0.0

                cumulative_cost += (trade_cost + interest_cost)
                prev_delta = current_delta
            else:
                # 거래 없음
                if idx > 0:
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360.0)
                else:
                    interest_cost = 0.0
                cumulative_cost += interest_cost
                trade_cost = 0.0
                delta_change = 0.0

            records.append({
                'date': all_dates[hp],
                's_current': s_current,
                'delta': current_delta,
                'delta_change': delta_change,
                'trade_cost': trade_cost,
                'interest_cost': interest_cost,
                'cumulative_cost': cumulative_cost
            })


        s_final = adj_close_path[-1]
        if flag.lower() == 'call':
            if s_final >= X:
                final_cost = cumulative_cost - X * shares_per_opt
            else:
                final_cost = cumulative_cost
        else:
            if s_final <= X:
                final_cost = X * shares_per_opt - cumulative_cost
            else:
                final_cost = cumulative_cost

        df_result = pd.DataFrame(records)

        buyorsell = 1 if optbs.lower()=='sell' else -1
        hedging_pnl = buyorsell * (theoretical_price - final_cost)
        realvol_pnl = buyorsell * (theoretical_price - theoretical_price_realvol)

        results = []
        results.append([M, final_cost, hedging_pnl, theoretical_price, realvol_pnl, theoretical_price_realvol])
        df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "hedge_cost", "hedging_pnl", "theoretical_price", "realvol_pnl", "theoretical_price_realvol"])
        df['name'] = name

        return cumulative_cost, final_cost, df_result, theoretical_price, df
    
    # return hedge()


    #### 시그널 만들기
    def edb_func_basic(delta_band, s_current, X, current_delta, current_sigma, current_gamma, signal):
        # sigma: 대략 0.2~0.4 / ㅎgamma는 양수
        return delta_band / (1.0 + current_sigma * current_gamma)
    
    def edb_func_1(delta_band, s_current, X, current_delta, current_sigma, current_gamma, signal):
        # signal: (-1, 1) 범위의 momentum 시그널
        # 아이디어: 우리는 숏감마 포지션. 주식 가격이 오르면 사야하고, 내리면 팔아야 한다. 문제 상황에서 팔 때만 거래비용이 발생하므로, 
        # 주식 가격이 내릴 때 덜 팔기 위해 delta_band를 더 크게 늘린다. 반대로 말하면 주식 가격이 오를 때 더 사기 위해 delta_band를 작게 한다.

        original_adj = current_sigma * current_gamma
        adj_signal = signal + 1 # signal: (-1, 1) -> (0, 2)

        k = 10

        current_moneyness = s_current / X

        return delta_band / (1.0 + original_adj + adj_signal * current_moneyness * k )
    
    def edb_func_2(delta_band, s_current, X, current_delta, current_sigma, current_gamma, signal):
        # signal: (-1, 1) 범위의 momentum 시그널
        # 아이디어: 이번엔 방향성 상관없이 momentum 시그널을 vol 시그널이라 치고 크면 band를 늘려보자. 

        adj_signal = signal ** 2 # signal: (-1, 1) -> (0, 1)

        current_moneyness = s_current / X

        k = 10

        return delta_band / (1.0 + current_gamma * adj_signal * current_moneyness * k)

    
    result0 = hedge(edb_func=edb_func_basic, name='basic')
    
    result11 = hedge(edb_func=edb_func_1, name='mom10', signal=signal1)
    # result12 = hedge(edb_func=edb_func_1, name='mom9', signal=signal2)
    # result13 = hedge(edb_func=edb_func_1, name='mom7', signal=signal3)

    result21 = hedge(edb_func=edb_func_2, name='mom10', signal=signal1)

    all_results = pd.concat([
        result0[-1], 

        result11[-1],
        # result12[-1],
        # result13[-1]

        result21[-1]

        ], ignore_index=True, axis=0)

    return all_results


# %%
### GPT로 fix 한 것. 이제 delta band를 range로 받음. 


def hedge_cost_delta_gamma_band_real_path_temp(yf_data, start_date, t, M, flag, shares_per_opt, optbs, r, q, sigma_col, 
                                               delta_lower=-0.2, delta_upper=0.1):
    # start_date 기준으로 행사가격 10% OTM K 가져오기
    X = yf_data.loc[start_date, '10%_OTM_K']

    # 사용 날짜 인덱스 생성
    all_dates = yf_data.loc[start_date:].index[:t+1]  # t+1개 행
    # 주가 경로 및 변동성 추출
    adj_close_path = yf_data.loc[all_dates, 'Adj Close'].values
    sigmas = yf_data.loc[all_dates, sigma_col].values

    # 이론가 대신 start_date의 sell_premium 사용
    theoretical_price = yf_data.loc[start_date, 'sell_premium'] * shares_per_opt
    s_start = adj_close_path[0]
    end_date = all_dates[-1]
    final_vol = yf_data.loc[end_date, 'roll_126_vol']
    t_a = t / 252.0
    theoretical_price_realvol = bsprice(s_start, X, r, q, t_a, final_vol, flag) * shares_per_opt

    # 모멘텀 시그널들
    signal1 = yf_data.loc[all_dates, 'mom10'].values
    signal2 = yf_data.loc[all_dates, 'mom9'].values
    signal3 = yf_data.loc[all_dates, 'mom7'].values

    # 헷지 시점 정의
    hedge_points = list(range(0, t+1, M))
    if hedge_points[-1] != t:
        hedge_points.append(t)
    hedge_points = sorted(set(hedge_points))

    # edb_func 들이 이제 lower, upper band를 리턴한다고 가정
    # 각 함수는 (lower_bound, upper_bound) 형식으로 반환
    def edb_func_basic(delta_lower, delta_upper, s_current, X, current_delta, current_sigma, current_gamma, signal):
        # 기본적으로 delta_lower와 delta_upper를 그대로 사용, gamma 조정
        # 예: gamma 비례로 band 축소 또는 확대
        adj_factor = (1.0 + current_sigma * current_gamma)
        return (delta_lower / adj_factor, delta_upper / adj_factor)

    def edb_func_1(delta_lower, delta_upper, s_current, X, current_delta, current_sigma, current_gamma, signal):
        # signal (-1,1)->(0,2), 주가 상승에는 band 작게, 하락에는 band 크게
        adj_signal = signal + 1  
        current_moneyness = s_current / X
        k = 10
        original_adj = 1.0 + current_sigma * current_gamma + adj_signal * current_moneyness * k
        return (delta_lower / original_adj, delta_upper / original_adj)

    def edb_func_2(delta_lower, delta_upper, s_current, X, current_delta, current_sigma, current_gamma, signal):
        # signal^2 (0,1)로 변환, 모멘텀이 커질수록 gamma band 조정
        adj_signal = signal ** 2  
        current_moneyness = s_current / X
        k = 10
        original_adj = 1.0 + current_gamma * adj_signal * current_moneyness * k
        return (delta_lower / original_adj, delta_upper / original_adj)


    def hedge(edb_func=None, name=None, signal=None):
        cumulative_cost = 0.0
        prev_delta = 0.0

        records = []

        for idx, hp in enumerate(hedge_points):
            t_remaining = t - hp
            tau = t_remaining / 252.0
            s_current = adj_close_path[hp]
            current_sigma = sigmas[hp]

            if signal is not None:
                signal_current = signal[hp]
            else:
                signal_current = None

            # 만기 시점
            if hp == t:
                if flag.lower() == 'call':
                    if s_current >= X:
                        current_delta = 1.0 if optbs.lower()=='sell' else -1.0
                    else:
                        current_delta = 0.0
                else:
                    if s_current >= X:
                        current_delta = 0.0
                    else:
                        current_delta = 1.0 if optbs.lower()=='sell' else -1.0
                lower_bound, upper_bound = (0.0, 0.0)
            else:
                current_delta = delta(s_current, X, r, q, tau, current_sigma, flag)
                current_gamma = gamma(s_current, X, r, q, tau, current_sigma, flag)

                # edb_func로부터 (lower_bound, upper_bound) 획득
                lower_bound, upper_bound = edb_func(delta_lower, delta_upper, s_current, X, current_delta, current_sigma, current_gamma, signal_current)

            # 현재 델타가 이전 델타 범위(prev_delta+lower_bound, prev_delta+upper_bound)를 벗어나면 트레이드
            if (current_delta < prev_delta + lower_bound) or (current_delta > prev_delta + upper_bound) or (hp == t):
                current_shares = current_delta * shares_per_opt
                prev_shares = prev_delta * shares_per_opt
                delta_change = current_shares - prev_shares

                if delta_change > 0:
                    trade_cost = delta_change * s_current
                else:
                    proceeds = (-delta_change)*s_current
                    fee = 0.001 * proceeds
                    net_inflow = proceeds - fee
                    trade_cost = -net_inflow

                if idx > 0:
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360.0)
                else:
                    interest_cost = 0.0

                cumulative_cost += (trade_cost + interest_cost)
                prev_delta = current_delta
            else:
                # 거래 없음
                if idx > 0:
                    delta_t = hp - hedge_points[idx-1]
                    interest_cost = cumulative_cost * r * (delta_t / 360.0)
                else:
                    interest_cost = 0.0
                cumulative_cost += interest_cost
                trade_cost = 0.0
                delta_change = 0.0

            records.append({
                'date': all_dates[hp],
                's_current': s_current,
                'delta': current_delta,
                'delta_change': delta_change,
                'trade_cost': trade_cost,
                'interest_cost': interest_cost,
                'cumulative_cost': cumulative_cost
            })

        s_final = adj_close_path[-1]
        if flag.lower() == 'call':
            if s_final >= X:
                final_cost = cumulative_cost - X * shares_per_opt
            else:
                final_cost = cumulative_cost
        else:
            if s_final <= X:
                final_cost = X * shares_per_opt - cumulative_cost
            else:
                final_cost = cumulative_cost

        df_result = pd.DataFrame(records)
        buyorsell = 1 if optbs.lower()=='sell' else -1
        hedging_pnl = buyorsell * (theoretical_price - final_cost)
        realvol_pnl = buyorsell * (theoretical_price - theoretical_price_realvol)

        results = []
        results.append([M, final_cost, hedging_pnl, theoretical_price, realvol_pnl, theoretical_price_realvol])
        df = pd.DataFrame(results, columns=["Rebalancing Interval(days)", "hedge_cost", "hedging_pnl", "theoretical_price", "realvol_pnl", "theoretical_price_realvol"])
        df['name'] = name

        return cumulative_cost, final_cost, df_result, theoretical_price, df

    # 기본, 모멘텀 시나리오
    result0 = hedge(edb_func=edb_func_basic, name='basic')
    result11 = hedge(edb_func=edb_func_1, name='mom10', signal=signal1)
    result12 = hedge(edb_func=edb_func_2, name='mom10', signal=signal1)

    result21 = hedge(edb_func=edb_func_1, name='mom9', signal=signal2)
    result22 = hedge(edb_func=edb_func_2, name='mom9', signal=signal2)
    
    result31 = hedge(edb_func=edb_func_1, name='mom7', signal=signal3)
    result32 = hedge(edb_func=edb_func_2, name='mom7', signal=signal3)

    all_results = pd.concat([
        result0[-1],
        result11[-1],
        result12[-1],
        result21[-1],
        result22[-1],
        result31[-1],
        result32[-1]
    ], ignore_index=True, axis=0)

    return all_results

# %%
results_all = []

for M_val in M_list:
    for sigma_val in sigma_cols:
        df = hedge_cost_delta_gamma_band_real_path_temp(
            yf_data=new_samsung_yf, 
            start_date=START_DATE_NEW, 
            t=126, 
            M=M_val, 
            flag="call", 
            shares_per_opt=100000, 
            optbs="sell", 
            r=0.04, 
            q=0.00, 
            sigma_col=sigma_val
        )
        
        # df에 어떤 M, sigma_col을 썼는지 표시하는 컬럼 추가
        df['M'] = M_val
        df['sigma_col'] = sigma_val
        
        # 리스트에 df 추가
        results_all.append(df)

# 모든 df를 하나의 DataFrame으로 합치기
final_results = pd.concat(results_all, ignore_index=True)

# %%
comparison = final_results.sort_values('hedge_cost', ascending=True).copy()
comparison['benchmark_hedge_cost'] = BENCHMARK_HEDGE_COST
comparison['cost_diff'] = comparison['hedge_cost'] - comparison['benchmark_hedge_cost']
comparison

# %%

# %%

# %% [markdown]
# ## Selection 4 종목 먼저 데이터 뽑기 

# %%
func_list = [
    hedge_cost_real_path,
    hedge_cost_delta_band_real_path,
    hedge_cost_delta_gamma_band_real_path
]

# %%
common_args = (
    new_samsung_yf, # yf_data
    START_DATE_NEW, # start_date
    126, # t
    None, # M ---> Should be replaced!
    "call",  # flag
    100000, # shares_per_opt
    "sell", # optbs
    0.04, # r
    0.00, # q
)

# %%
hedge_cost_real_path_kwargs = {
}

hedge_cost_delta_band_real_path_kwargs = {
    'delta_band': 0.2
}

hedge_cost_delta_gamma_band_real_path_kwargs = {
    'delta_band': 0.2,
    'gamma_scale': 10.0
}

kwargs_list = [
    hedge_cost_real_path_kwargs,
    hedge_cost_delta_band_real_path_kwargs,
    hedge_cost_delta_gamma_band_real_path_kwargs
]


# %%
def one_method_all_combinations(func, *common, **kwargs):
    results_all = []

    for M_val in M_list:
        for sigma_val in sigma_cols:
            common = list(common)
            common[3] = M_val

            _, _, _, _, df = func(
                *common, 
                # M=M_val, 
                sigma_col=sigma_val, 
                **kwargs
            )
            
            # df에 어떤 M, sigma_col을 썼는지 표시하는 컬럼 추가
            df['M'] = M_val
            df['sigma_col'] = sigma_val
            
            # 리스트에 df 추가
            results_all.append(df)

    # 모든 df를 하나의 DataFrame으로 합치기
    final_results = pd.concat(results_all, ignore_index=True)
    final_results['func_name'] = func.__name__

    return final_results


# %%
all_dfs = []

for func, kwarg in zip(func_list, kwargs_list):
    temp_df = one_method_all_combinations(func, *common_args, **kwarg)
    all_dfs.append(temp_df)

all_final_results = pd.concat(all_dfs, ignore_index=True, axis=0)

# %%
all_final_results.to_pickle(OUTPUT_PATH / f'all_final_results_{CURRRENT_SELECTION}.pkl')

# %%
all_final_results

# %%
