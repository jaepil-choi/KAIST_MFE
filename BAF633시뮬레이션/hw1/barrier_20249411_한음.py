#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# np.random.seed(124)

# %%
def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps]) #0부터 1사이의 난수를 n by m 개 matrix로 생성
    X = np.zeros([NoOfPaths, NoOfSteps+1]) #exponantial term을 담을 array
    S = np.zeros([NoOfPaths, NoOfSteps+1]) # 주가를 담을  array
    time = np.zeros([NoOfSteps+1]) #0 부터 T 까지 n개의 step이 있으면 n+1개의 array가 필요
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
     
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma * sigma) * dt + sigma *\
        np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
        
    
    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths
# %%
def BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,Btype,Otype):
    path = GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    S_path = path["S"]
    payoff = np.zeros([NoOfPaths,1])
    if Otype == 'C' :
        flag = 1
    else :
        flag = -1  
    if Btype == 'UpOut':
        event = np.where(S_path > B, True,False)
        event1 = np.sum(event, axis = 1)
        for i in range(NoOfPaths):
            if event1[i] == 0:
                payoff[i] = np.maximum(flag*(S_path[i,-1]-K),0)
    elif Btype == "UpIn":
        event = event = np.where(S_path > B, True,False)
        event1 = np.sum(event, axis = 1)
        for i in range(NoOfPaths):
            if event1[i] != 0:
                payoff[i] = np.maximum(flag*(S_path[i,-1]-K),0)
    elif Btype == "DownOut":
        event = np.where(S_path < B, True,False)
        event1 = np.sum(event, axis = 1)
        for i in range(NoOfPaths):
            if event1[i] == 0:
                payoff[i] = np.maximum(flag*(S_path[i,-1]-K),0)

    elif Btype == "DownIn" : 
        event = np.where(S_path < B, True,False)
        event1 = np.sum(event, axis = 1)
        for i in range(NoOfPaths):
            if event1[i] != 0:
                payoff[i] = np.maximum(flag*(S_path[i,-1]-K),0)
    else:
        event1 = np.zeros([NoOfPaths,1])
        for i in range(NoOfPaths):
            if event1[i] == 0:
                payoff[i] = np.maximum(flag*(S_path[i,-1]-K),0)

    #0이면 옵션이 있는 상태, 0이 아니면 옵션이 없는 상태
    d_payoff = payoff * np.exp(-r*T)
    pv = np.sum(payoff * np.exp(-r*T))/NoOfPaths
    std = (np.sum((d_payoff-pv)**2))/NoOfPaths
    return pv ,std, d_payoff
# %%
NoOfPaths = 1000
NoOfSteps = 1000
T = 1
r = 0.03
sigma= 0.2
S_0 = 100
B = 90
K = 100



# %%
[BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'UpOut','C')[0],
BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'UpIn','C')[0],
BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'DownOut','C')[0],
BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'DownIn','C')[0],
BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'None','C')[0]]
# %%
UpOut = BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'UpOut','P')[0]
UpIn = BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'UpIn','P')[0]
DownOut = BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'DownOut','P')[0]
DownIn = BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'DownIn','P')[0]
Put = BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'Call','P')[0]

# %%
UpOut + UpIn - Put
# %%
DownOut+DownIn - Put

# %%
import QuantLib as ql

#Market Info.
S = S_0
r = r
vol = sigma

#Product Spec.
T = T
K = 100
B = 90
rebate = 0
barrierType = ql.Barrier.DownIn
optionType = ql.Option.Call

#Barrier Option
today = ql.Date().todaysDate()
maturity = today + ql.Period(T, ql.Years)

payoff = ql.PlainVanillaPayoff(optionType, K)
euExercise = ql.EuropeanExercise(maturity)
barrierOption = ql.BarrierOption(barrierType, B, rebate, payoff, euExercise)

#Market
spotHandle = ql.QuoteHandle(ql.SimpleQuote(S))
flatRateTs = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
flatVolTs = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
bsm = ql.BlackScholesProcess(spotHandle, flatRateTs, flatVolTs)
analyticBarrierEngine = ql.AnalyticBarrierEngine(bsm)

#Pricing
barrierOption.setPricingEngine(analyticBarrierEngine)
price = barrierOption.NPV()

print("Price = ", price)

#%% 

budget = 1000000
n = np.array([100,200,400,500,1000,2000,5000,10000,20000])
m = (budget / n).astype(int)
m
var = np.zeros([len(n)])
bias = np.zeros([len(n)])
mse = np.zeros([len(n)])

# %%
for j in range(len(n)):
    NoOfPaths = n[j]
    NoOfSteps = m[j]
    T = 1
    r = 0.03
    sigma= 0.2
    S_0 = 100
    B = 90
    K = 100


    alpha_hats = np.zeros(30)
    for k in range(30):
        alpha_hats[k] = BarrierOption(NoOfPaths,NoOfSteps,T,r,sigma,S_0,B,K,'DownIn','C')[0]
        E_alpha_hats = np.average(alpha_hats)

    #var[j] = np.std(alpha_hats,ddof=1)
    var[j] = np.mean((alpha_hats - E_alpha_hats)**2)
    bias[j] =(E_alpha_hats - price)**2
    mse[j] = np.mean((alpha_hats - price)**2)


# %%

plt.figure(figsize=(10, 6))

# 각각의 값을 선으로 표현
plt.plot(n, var, label='Variance', marker='o')
plt.plot(n, bias, label='Bias', marker='s')
plt.plot(n, mse, label='MSE', marker='^')

# x축과 y축에 라벨 추가
plt.xlabel('n')
plt.ylabel('Values')

# 로그 스케일로 변경 (n 값이 매우 크므로)
plt.xscale('log')

# 제목 및 범례 추가
plt.title('Variance, Bias, and MSE')
plt.legend()

# 그래프 출력
plt.grid(True)
plt.show()

#%%
