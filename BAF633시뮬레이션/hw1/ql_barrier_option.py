#%%
import QuantLib as ql

#Market Info.
S = 100
r = 0.03
vol = 0.2

#Product Spec.
T = 1
K = 100
B = 120
# B = 110 # barrier가 낮아지면 option price가 낮아짐. 
# B = 130 # barrier가 높아지면 option price가 높아짐.
# B = 100000 # barrier가 아주 높다면, plain vanilla option과 같아짐.

rebate = 0
barrierType = ql.Barrier.UpOut
optionType = ql.Option.Call
# optionType = ql.Option.Put

## 이 아래부턴 좀 복잡하니까 일단 그냥 써라. 
## Quantlib은 정교한 잘 만들어진 라이브러리. industry에서도 많이 씀. 
## 그러나 엄청 exotic한 것은 없음. 예를 들어 step down ELS 같은거. 
## 회사에서 만들 땐 저런 quantlib을 베이스로 기능을 만듦. 
## python의 quantlib은 c++의 일부 기능만 가져온 것.
## 많은 python package가 베이스는 C++ 등으로 만들어져 있고, python은 glue language의 역할을 함. 
## 뭔가 기능 수정이나 추가가 필요하면 C++로 가서 수정해야 함. 그래서 C++이 필요. 

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

# %%
