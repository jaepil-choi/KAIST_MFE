#%%
import numpy as np 
import pandas as pd
from scipy.linalg import solve_banded, solve, solveh_banded
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

# explicit은 따로 만들었다. 
# 왜? tri-diagonal matrix가 곱해질 때 따로 계산하는 것이 더 빠르기 때문이다.
# 안그러면 0인 부분들이 계속 곱해지기 때문에 계산량이 늘어난다.
def exfdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds
    a = dt*(vol*s[1:-1])**2 / (2*ds**2) # 맨 앞과 맨 뒤는 쓰지 않는다. 
    b = dt*(r-q)*s[1:-1] / (2*ds) # 마찬가지. 
    d, m, u = a-b, -2*a-dt*r, a+b

    v = np.maximum(callOrPut*(s-k), 0) # terminal condition. (만기에서의 payoff)

    for j in range(M-1,-1,-1):
        temp = d * v[:-2] + (1 + m) * v[1:-1] + u * v[2:] # v값 업데이트
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        v[1:-1] = temp
    f = interp1d(s,v)
    return pd.DataFrame({"S":s,"V":v}), f(s0)


#%%
s0, k, r, q, t, vol = 100, 100, 0.03, 0.01, 0.25, 0.4
optionType, maxS, N, M = "call", s0*2, 200, 2000
theta = 0

def fdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M, theta=1):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds

    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    A = np.diag(d[1:],-1) + np.diag(m) + np.diag(u[:-1],1)
    B = np.zeros((N-1,2))
    B[0,0], B[-1,1] = d[0], u[-1]

    Am = np.identity(N-1) - theta*A
    Ap = np.identity(N-1) + (1-theta)*A
    ab = np.zeros((3, N-1)) # solve_banded 를 쓰기 위해 만드는 것. 
    ab[0,1:] = np.diag(Am,1) # u, N-2
    ab[1] = np.diag(Am) # m, N-1
    ab[2,:-1] = np.diag(Am,-1) # d, N-2

    v = np.maximum(callOrPut*(s-k), 0)
    for j in range(M-1,-1,-1):    
        # temp = Ap @ v[1:-1] + theta*B @ v[[0,-1]] # 이렇게 하면 계산량이 늘어난다. 특히 Ap @ v[1:-1] 부분.
        # 속도가 약 2배 차이남
        temp = (1-theta)*d * v[:-2] + (1 + (1-theta)*m) * v[1:-1] + (1-theta)*u * v[2:] # d, m, u 가지고 우변 계산. 필요한 연산만 할 수 있게. 
        temp[0] += theta*d[0]*v[0]
        temp[-1] += theta*u[-1]*v[-1]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        temp += (1-theta)*B @ v[[0,-1]]
        v[1:-1] = solve_banded((1,1), ab, temp) # np가 아니라 scipy.linalg.solve_banded
        # ? 첫 번째 인자 (1, 1)는 .... ?? 아래로 1개, 위로 1개

        # 어떻게 하면 더 빠르게 할까? 하는 고민을 해야한다는 예시. 

    f = interp1d(s,v)
    return pd.DataFrame({"S":s,"V":v}), f(s0)


#%%
if __name__=="__main__":
    s = 100
    k = 100
    r = 0.03
    q = 0.01
    t = 0.25
    sigma = 0.2
    optionType = 'put'
    maxS, n, m = s*2, 1000, 10000 # 지금은 timestep이 촘촘하다. 
    v, ex_price = exfdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m) # 따로 계산하는 것이 훨씬 빠름. 
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, ex_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m, 0) # Theta method 느리다. 
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, im_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m) # 빠르지도 않은데.. 그럼 implicit, CN 은 왜 하냐? 
    print(f"IM-FDM Price = {im_price:0.6f}")

# explicit의 엄청난 단점이 있다. 
# 계산 결과가 수치적으로 불안정할 가능성이 높다. 
# timestep이 촘촘하지 않을 때, 그러니까 위에서 10000을 5000정도로 바꾸면, inf 뜨면서 오류난다. 
# explicit 가격이 갑자기 -inf 에 가까운 숫자가... 
# 200, 100 정도로 바꾸니까, 갑자기 EX-FDM method는 -inf로 가버린다. (diverge해버림)
# implicit method, CN method는 그런 문제가 없다.
# 상대적으로 delta S < delta t 면 explicit FDM이 발산할 가능성이 높다.
# delta s가 늘어나는 것에 비해 delta t가 훨씬 더 작아져야 한다. 
# 200, 200은 수렴한다. 그러면 400, 400도 수렴할까? --> 수렴하지 않는다. 훨씬 더 많이 줄어들어야 한다. 
# 400, 2000 쯤 되어야 수렴하지, 400, 1000 이런거로는 어림도 없다. 
# 수렴 조건이 있는데 그걸 굳이 설명해주지 않는 이유는 그 수렴조건을 아는게 별 의미가 없기 때문이다.
# 정확도가 높아지는 것, time step을 늘리는 것으로는 큰 도움 안되고, 그냥 수렴할 정도로만 늘리면 된다. 
# 정확도를 높이려면, delta s를 더 촘촘하게 해야한다. (해상도를 높인다)
# explicit FDM에선 delta s를 늘릴 때 time step을 너무 크게 느려야 해서 scalable하지 않다. 
# 만기가 길어지면 - t가 0.25에서 0.4만 되더라도 바로 발산한다. (t가 늘어나면 delta t가 같이 늘어나기 때문에)
# 또 다른 수렴 조건. 더 중요한 수렴 조건. 변동성 파라미터. 
# 변동성 파라미터가 커지면 수렴 가능성이 낮아진다. 
# sigma가 0.2 --> 0.25만 되어도 수렴하지 않는다. 
# 기초 자산이 2개면 corr까지 영향을 미친다. 
# explicit FDM을 쓰면 현재 변동성에선 평가가 되다가 변동성이 커지면 평가가 안되는 말도 안되는 상황이 발생할 수 있다. 
# 그냥 보면 explicit이 implicit보다 빨라보이지? 한 4배? 하지만 explicit을 수렴시키기 위해선 time step이 거의 10배 이상 필요할 수도 있다. 
# 그래서 종합적으로 보면 implicit이 더 빠르다.

    v, cn_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m, 0.5)
    print(f"CN-FDM Price = {cn_price:0.6f}")
