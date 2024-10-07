#%%
import QuantLib as ql

def ql_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, option_type, oh):

    callOrPut = ql.Option.Call if option_type.lower()=="call" else ql.Option.Put
    today = ql.Date().todaysDate()
    exp_date = today + ql.Period(t, ql.Years)

    exercise = ql.EuropeanExercise(exp_date)
    vanillaPayoff = ql.PlainVanillaPayoff(ql.Option.Call, k)
    payoffMin = ql.MinBasketPayoff(vanillaPayoff)
    basketOptionMin = ql.BasketOption(payoffMin, exercise)

    vanillaPayoff0 = ql.PlainVanillaPayoff(ql.Option.Call, k-1/oh)
    payoffMin0 = ql.MinBasketPayoff(vanillaPayoff0)
    basketOptionMin0 = ql.BasketOption(payoffMin0, exercise)

    # Create a StochasticProcessArray for the various underlyings
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    spot1 = ql.QuoteHandle(ql.SimpleQuote(s1))
    spot2 = ql.QuoteHandle(ql.SimpleQuote(s2))
    volTS1 = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma1, day_count))
    volTS2 = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma2, day_count))
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    dividendTS1 = ql.YieldTermStructureHandle(ql.FlatForward(today, q1, day_count))
    dividendTS2 = ql.YieldTermStructureHandle(ql.FlatForward(today, q2, day_count))

    process1 = ql.GeneralizedBlackScholesProcess(spot1, dividendTS1, riskFreeTS, volTS1)
    process2 = ql.GeneralizedBlackScholesProcess(spot2, dividendTS2, riskFreeTS, volTS2)

    engine = ql.StulzEngine(process1, process2, corr)
    basketOptionMin.setPricingEngine(engine)
    price = basketOptionMin.NPV()

    basketOptionMin0.setPricingEngine(engine)
    price0 = basketOptionMin0.NPV()
    
    return oh*(price0 - price)



if __name__=="__main__":
    s1, s2 = 100,100
    r = 0.02
    q1, q2 = 0.01, 0.015
    k = 100
    t = 1
    sigma1, sigma2 = 0.15, 0.2
    corr = 0.5
    option_type = "call"
    oh = 1

    price = ql_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, option_type, oh)
    print("Price = ", price)
