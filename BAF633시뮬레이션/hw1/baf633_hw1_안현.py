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
# 현이 과제 참고용

# %%
import QuantLib as ql

def monte_carlo_barrier_option_pricing(S, K, B, T, r, vol, isUP, isIN, option_type, rebate=0, n_sims=10000, time_steps=252):
    """
    Monte Carlo 시뮬레이션을 사용한 배리어 옵션 가격 계산 함수 (QuantLib 사용)
    
    매개변수:
    S: 기초 자산 가격
    K: 행사가격
    B: 배리어 가격
    T: 만기 기간 (연 단위)
    r: 무위험 이자율
    vol: 변동성
    isUP: True면 Up 배리어, False면 Down 배리어
    isIN: True면 In 배리어, False면 Out 배리어
    option_type: 옵션 타입 (콜/풋)
    rebate: 배리어를 넘어섰을 때의 리베이트
    n_sims: 시뮬레이션 횟수 (기본값 10000)
    time_steps: 연간 거래일 (기본값 252)
    
    반환값:
    Monte Carlo를 사용한 배리어 옵션의 가격
    """

    # 배리어 타입 결정
    if isUP and isIN:
        barrier_type = ql.Barrier.UpIn
    elif isUP and not isIN:
        barrier_type = ql.Barrier.UpOut
    elif not isUP and isIN:
        barrier_type = ql.Barrier.DownIn
    else:
        barrier_type = ql.Barrier.DownOut

    # 날짜 설정
    today = ql.Date().todaysDate()
    maturity = today + ql.Period(int(T * 365), ql.Days)  # T년 단위에서 일 단위로 변환

    # 배리어 옵션 설정
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(maturity)
    barrier_option = ql.BarrierOption(barrier_type, B, rebate, payoff, exercise)

    # Black-Scholes-Merton 프로세스 설정
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    bsm_process = ql.BlackScholesProcess(spot_handle, rate_handle, vol_handle)

    # Monte Carlo Pricing Engine 설정
    mc_engine = ql.MCBarrierEngine(bsm_process, 'pseudorandom', timeStepsPerYear=time_steps, requiredSamples=n_sims, antitheticVariate=True)

    # Pricing Engine을 옵션에 설정
    barrier_option.setPricingEngine(mc_engine)
    
    # 옵션 가격 계산
    price = barrier_option.NPV()

    return price

# 함수 호출 예시
S = 100   # 기초 자산 가격
K = 100   # 행사가격
B = 120   # 배리어 가격
T = 1     # 만기 (1년)
r = 0.03  # 무위험 이자율
vol = 0.2  # 변동성
isUP = True  # Up 배리어
isIN = False  # Out 배리어
option_type = ql.Option.Call  # 콜 옵션

# Monte Carlo 시뮬레이션을 사용한 배리어 옵션 가격 계산
price = monte_carlo_barrier_option_pricing(S, K, B, T, r, vol, isUP, isIN, option_type)
print(f"Monte Carlo Barrier Option Price: {price}")
