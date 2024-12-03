import numpy as np
from scipy.interpolate import RectBivariateSpline
import QuantLib as ql

def adi_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    S1_max, S2_max = s1 * 3, s2 * 3
    dS1, dS2 = S1_max / nx, S2_max / ny
    dt = t / nt

    S1 = np.linspace(0, S1_max, nx + 1)
    S2 = np.linspace(0, S2_max, ny + 1)
    grid = np.zeros((nt + 1, ny + 1, nx + 1))

    payoff = np.where(np.minimum(S1, S2) >= k, 10000, 0)
    grid[-1, :, :] = oh * payoff

    for t_step in reversed(range(nt)):
        V = grid[t_step + 1, :, :].copy()

        for j in range(1, ny):
            for i in range(1, nx):
                cross = corr * sigma1 * sigma2 * S1[i] * S2[j] * dt / (4 * dS1 * dS2)
                V[j, i] = (V[j, i] + cross * (V[j+1, i+1] - V[j+1, i-1] - V[j-1, i+1] + V[j-1, i-1])) / (1 + r * dt)

        grid[t_step, :, :] = V

    spline = RectBivariateSpline(S1, S2, grid[0, :, :])
    price = spline(s1, s2)[0, 0]

    delta1 = (spline(s1 + dS1, s2) - spline(s1 - dS1, s2)) / (2 * dS1)
    delta2 = (spline(s1, s2 + dS2) - spline(s1, s2 - dS2)) / (2 * dS2)
    gamma1 = (spline(s1 + dS1, s2) - 2 * spline(s1, s2) + spline(s1 - dS1, s2)) / (dS1 ** 2)
    gamma2 = (spline(s1, s2 + dS2) - 2 * spline(s1, s2) + spline(s1, s2 - dS2)) / (dS2 ** 2)
    cross_gamma = (spline(s1 + dS1, s2 + dS2) - spline(s1 + dS1, s2 - dS2) -
                   spline(s1 - dS1, s2 + dS2) + spline(s1 - dS1, s2 - dS2)) / (4 * dS1 * dS2)
    theta = (grid[1, int(s2 / dS2), int(s1 / dS1)] - price) / dt

    return (price, delta1, delta2, gamma1, gamma2, cross_gamma, theta)

def osm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    S1_max, S2_max = s1 * 3, s2 * 3
    dS1, dS2 = S1_max / nx, S2_max / ny
    dt = t / nt

    S1 = np.linspace(0, S1_max, nx + 1)
    S2 = np.linspace(0, S2_max, ny + 1)
    grid = np.zeros((nt + 1, ny + 1, nx + 1))

    payoff = np.where(np.minimum(S1, S2) >= k, 10000, 0)
    grid[-1, :, :] = oh * payoff

    for t_step in reversed(range(nt)):
        V = grid[t_step + 1, :, :].copy()

        for j in range(1, ny):
            for i in range(1, nx):
                cross = corr * sigma1 * sigma2 * S1[i] * S2[j] * dt / (4 * dS1 * dS2)
                V[j, i] = (V[j, i] + cross * (V[j+1, i+1] - V[j+1, i-1] - V[j-1, i+1] + V[j-1, i-1])) / (1 + r * dt)

        grid[t_step, :, :] = V

    spline = RectBivariateSpline(S1, S2, grid[0, :, :])
    price = spline(s1, s2)[0, 0]

    delta1 = (spline(s1 + dS1, s2) - spline(s1 - dS1, s2)) / (2 * dS1)
    delta2 = (spline(s1, s2 + dS2) - spline(s1, s2 - dS2)) / (2 * dS2)
    gamma1 = (spline(s1 + dS1, s2) - 2 * spline(s1, s2) + spline(s1 - dS1, s2)) / (dS1 ** 2)
    gamma2 = (spline(s1, s2 + dS2) - 2 * spline(s1, s2) + spline(s1, s2 - dS2)) / (dS2 ** 2)
    cross_gamma = (spline(s1 + dS1, s2 + dS2) - spline(s1 + dS1, s2 - dS2) -
                   spline(s1 - dS1, s2 + dS2) + spline(s1 - dS1, s2 - dS2)) / (4 * dS1 * dS2)
    theta = (grid[1, int(s2 / dS2), int(s1 / dS1)] - price) / dt

    return (price, delta1, delta2, gamma1, gamma2, cross_gamma, theta)

def ql_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, option_type, oh):
    callOrPut = ql.Option.Call if option_type.lower()=="call" else ql.Option.Put
    today = ql.Date().todaysDate()
    exp_date = today + ql.Period(int(t), ql.Years)

    exercise = ql.EuropeanExercise(exp_date)
    vanillaPayoff = ql.PlainVanillaPayoff(callOrPut, k)
    payoffMin = ql.MinBasketPayoff(vanillaPayoff)
    basketOptionMin = ql.BasketOption(payoffMin, exercise)

    vanillaPayoff0 = ql.PlainVanillaPayoff(callOrPut, k - 1/oh)
    payoffMin0 = ql.MinBasketPayoff(vanillaPayoff0)
    basketOptionMin0 = ql.BasketOption(payoffMin0, exercise)

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

if __name__ == "__main__":
    s1, s2 = 100, 100
    r = 0.02
    q1, q2 = 0.015, 0.01
    k = 95
    t = 1
    sigma1, sigma2 = 0.15, 0.2
    corr = 0.5
    option_type = "call"
    oh = 0.5
    nx, ny, nt = 200, 200, 4000

    price_adi, delta1_adi, delta2_adi, gamma1_adi, gamma2_adi, cross_gamma_adi, theta_adi = adi_worst_of(
        s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt
    )
    print("ADI Price = ", price_adi)

    price_osm, delta1_osm, delta2_osm, gamma1_osm, gamma2_osm, cross_gamma_osm, theta_osm = osm_worst_of(
        s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt
    )
    print("OSM Price = ", price_osm)

    analytic_price = ql_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, option_type, oh)
    print("QuantLib Price = ", analytic_price)

    error_adi = abs(price_adi - analytic_price)
    error_osm = abs(price_osm - analytic_price)
    print("ADI Error = ", error_adi)
    print("OSM Error = ", error_osm)
