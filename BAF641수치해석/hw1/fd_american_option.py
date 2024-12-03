import numpy as np
from scipy.interpolate import interp1d

def fd_american_option(s, k, r, q, t, sigma, option_type, n, m):
    omega = 1.2 # relaxation parameter. 
    tol = 1e-6 # convergence tolerance
    max_iter = 10000 # 무한 루프 방지

    # 그리드
    S_max = s * 3 # 최대 3배
    dS = S_max / n
    dt = t / m

    asset_prices = np.linspace(0, S_max, n + 1)
    grid = np.zeros((m + 1, n + 1)) # 행 시간 / 열 주가
    
    # 만기
    if option_type.lower() == 'call':
        grid[-1, :] = np.maximum(asset_prices - k, 0)
    elif option_type.lower() == 'put':
        grid[-1, :] = np.maximum(k - asset_prices, 0)

    # put call의 boundary. 모두 + 유지. 
    for i in range(m + 1):
        tau = t - i * dt
        if option_type.lower() == 'call':
            grid[i, -1] = S_max - k * np.exp(-r * tau)
            grid[i, 0] = 0
        elif option_type.lower() == 'put':
            grid[i, 0] = k * np.exp(-r * tau)
            grid[i, -1] = 0

    # fdm coeffs
    j = np.arange(1, n)
    a = 0.5 * dt * (sigma**2 * j**2 - (r - q) * j) # lower diagonal of tridiagonal matrix
    b = 1 + dt * (sigma**2 * j**2 + r) # main diagonal of tridiagonal matrix
    c = 0.5 * dt * (-(sigma**2 * j**2 + (r - q) * j)) # upper diagonal of tridiagonal matrix

    # 역순
    for i in reversed(range(m)):
        rhs = a * grid[i + 1, j - 1] + b * grid[i + 1, j] + c * grid[i + 1, j + 1]
        V_old = grid[i, j].copy()
        V_new = V_old.copy()

        # 조기행사 조건
        if option_type.lower() == 'call':
            intrinsic = asset_prices[j] - k
        else:
            intrinsic = k - asset_prices[j]
        intrinsic = np.maximum(intrinsic, 0)

        # PSOR
        error = 1.0
        iteration = 0
        while error > tol and iteration < max_iter:
            error = 0.0
            for index in range(len(j)):
                res = a[index] * V_new[index - 1] if index != 0 else 0
                res += b[index] * V_new[index]
                res += c[index] * V_new[index + 1] if index != len(j) -1 else 0
                res = rhs[index] - res
                V_temp = V_new[index] + omega * res / b[index]
                V_temp = max(V_temp, intrinsic[index])
                error = max(error, abs(V_temp - V_new[index]))
                V_new[index] = V_temp
            iteration += 1

        grid[i, j] = V_new

    # Interpolation
    option_price_func = interp1d(asset_prices, grid[0, :], kind='cubic')
    price = float(option_price_func(s))

    # Central difference
    dS = asset_prices[1] - asset_prices[0]
    s_up = s + dS
    s_down = s - dS

    if s_up > S_max:
        s_up = S_max
    if s_down < 0:
        s_down = 0

    price_up = float(option_price_func(s_up))
    price_down = float(option_price_func(s_down))

    # Greeks 
    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2 * price + price_down) / (dS ** 2)

    index = np.argmin(np.abs(asset_prices - s))
    if index == 0:
        theta = (grid[1, index] - grid[0, index]) / dt
    elif index == n:
        theta = (grid[1, index] - grid[0, index]) / dt
    else:
        theta = (grid[1, index] - price) / dt

    return (price, delta, gamma, theta)

if __name__ == "__main__":
    s = 100
    k = 100
    r = 0.05
    q = 0.02
    t = 1
    sigma = 0.25
    option_type = "call"
    n = 100
    m = 1000

    price, delta, gamma, theta = fd_american_option(s, k, r, q, t, sigma, option_type, n, m)
    print(price, delta, gamma, theta)
