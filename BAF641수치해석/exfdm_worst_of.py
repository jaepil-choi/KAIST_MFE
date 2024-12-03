#%%
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt 

def exfdm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, nx, ny, nt, oh):
    smax1, smax2 = s1*2, s2*2
    smin1, smin2 = s1*0, s2*0
    ds1, ds2 = (smax1-smin1) / nx, (smax2-smin2) / ny # ds: 주가의 간격. 주가의 step 갯수로 나눠줌. 
    dt = t / nt # time step으로 나눠줌. 

    i1, i2 = np.arange(nx+1), np.arange(ny+1) 
    #vector & mesh of underlying assets
    s1_v, s2_v = smin1+i1*ds1, smin2+i2*ds2 # 그리드 상에서의 주가 레벨
    s_1, s_2 = np.meshgrid(s1_v, s2_v) #을 가지고 meshgrid를 만듦.
    # x, y = meshrid(v1, v2) 면 우선 x는 v1가 column축, v2가 row축이 됨.
    # y는 그냥 v2를 row에 반복한 것이 된다고 함. 이건 해보면서 이해해야할듯...  

    #terminal condition = payoff 
    v = np.zeros((ny+1, nx+1)) # T에서의 payoff.
    v = np.where(np.minimum(s_1, s_2)>=k, 1, v)
    v = np.where((np.minimum(s_1, s_2)>=k-1/oh) & (np.minimum(s_1, s_2)<k), oh*(np.minimum(s_1, s_2)-(k-1/oh)), v)

    #coefficients
    ## 여기있는게 Explicit FDM 슬라이드의 두 번째 수식덩어리 부분. 
    a1 = dt*(sigma1*s_1)**2 / (2*ds1**2)
    b1 = dt*(r-q1)*s_1 / (2*ds1)
    a2 = dt*(sigma2*s_2)**2 / (2*ds2**2)
    b2 = dt*(r-q2)*s_2 / (2*ds2)
    c = dt*corr*sigma1*sigma2*s_1*s_2 / (4*ds1*ds2) # cross 항들. corr 들어가는. 
    d1, u1, d2, u2 = a1-b1, a1+b1, a2-b2, a2+b2
    m = -2*a1 - 2*a2 - dt*r # 슬라이드 수식 덩어리 마지막 부분. (3번째)
    
    d1, u1, d2, u2, c, m = \
        d1[1:-1,1:-1], u1[1:-1,1:-1], d2[1:-1,1:-1], u2[1:-1,1:-1], c[1:-1,1:-1], m[1:-1,1:-1]
    # coefficient를 안쪽만 쓴다. d1, u2 등등은 다 matrix. 
    
    
    # 아래의 backwardation에 대한 설명들
    # backwardation 시의 변화를 3d plot을 볼 수도 있다. 이것도 올려주셨을듯? exfdm_worst_of 에 코드 추가하신듯. 
    
    # 1. terminal 시점에서의 payoff를 보면 그냥 디지털마냥 절벽처럼 나온다. 두 지수 모두 행사가 이상이여야 payoff가 발생하는 그림. 
    # 그림으로 찍어서 이런 식으로 나와야 제대로 하고 있는 것. 절벽처럼 보이지만 기울기가 있다. 
    # oh 파라미터를 0.1 주면 잘 안보이지만 0.5만 줘도 수직이 아니고 기울기라는 것이 보인다. 
    # 현재 시점으로 backwardation 해 오는 과정에서 기울기가 점점 줄어든다. 마지막 시점 = 현재로 오면 상당히 곡선이 됨. 

    # 2. 

    #time backwardation 
    for j in range(nt-1,-1,-1):
        temp = d1 * v[1:-1,:-2] + \
            d2 * v[:-2,1:-1] + \
            u1 * v[1:-1,2:] + \
            u2 * v[2:,1:-1] + \
            c * v[:-2,:-2] + \
            c * v[2:, 2:] - \
            c * v[:-2, 2:] - \
            c * v[2:,:-2] + \
            (1 + m) * v[1:-1, 1:-1] # 위치 9개를 한 번에 잡은 것. explicit이니 전부 우변에 값들이 있다. 
        
        # implicit이었으면 tridiagonal이 좌변에 생겼을 것. 
        
        v[1:-1, 1:-1] = temp # update v

        #boundary condition: gamma = 0
        v[0,:] = 2*v[1,:]-v[2,:]
        v[-1,:] = 2*v[-2,:]-v[-3,:]
        v[:,0] = 2*v[:,1]-v[:,2]
        v[:,-1] = 2*v[:,-2]-v[:,-3]

        # cross gamma가 0가 되게 하는 condition들. 
        v[0,0] = v[0,1] + v[1,0] - v[1,1]
        v[-1,0] = v[-2,0] + v[-1,1] - v[-2,1]
        v[0,-1] = v[0,-2] + v[1,-1] - v[1,-2]
        v[-1,-1] = v[-1,-2] + v[-2,-1] - v[-2,-2]

    f = RectBivariateSpline(s2_v, s1_v, v) # 2d interpolation을 위한 spline 함수.
    return f(s2, s1)[0,0] # interpolation을 해서 주가를 찾는다. 



if __name__=="__main__":
    from ql_worst_of import ql_worst_of
    option_type = "call"
    
    s1, s2 = 100,100
    r = 0.02
    # 금리를 하나만 썼는데, 오히려 국내 지수를 잘 안쓴다. 그러면 다 그 국가의 r을 써줘야. 통화별로 r이 다 다르니까.
    # 할인은 원화로 하지만, drift term은 각 나라의 r이 들어가줘야. 
    # 기초자산은 달러인데, 지급해야하는건 원화면 (대부분이 이렇다) --> 콴토옵션. 
    # 가격도 달라지고, quanto adjustment도 해줘야 함.
    # 이거 헤지는 또 어떻게? 풀기 어려운... 리스크이다... 

    q1, q2 = 0.015, 0.01
    k = 95
    t = 1

    # (콜옵션일 때)
    # 변동성. ITM에 있을 때는 제발 내려가지 말아라. gamma가 낮아라 제발. = short gamma
    # OTM에 있을 때는 제발 한 번 터져서 올라가라. gamma가 높아라 제발. = long gamma
    # 변동성이 낮을 때 수익이 나야한다 = 옵션을 팔아야 한다. ** vega hedge 어쩌구 설명해주셨는데 이해가 안갔다. 
    # 2시간 38분 지점 notability 들어보기. lecture #4
    # 헤지 운용, 안정적일꺼 같지? ㄴㄴ. 2조 book 이면 하루 손익이 100억씩 난다. -100억 찍히면 난리난다. 

    sigma1, sigma2 = 0.15, 0.2 # explicit이라, 이걸 크게 만들면 수렴 안할꺼다. 
    # sigma1, sigma2 = 0.25, 0.25 # 이러면 수렴 안함. 
    nx, ny, nt = 200, 200, 4000 
    # nx, ny, nt = 100, 100, 4000 # 이 경우 x, y 스탭 줄이고 t 스탭 늘리면 되긴하는데... 알지? scalable하지 않아. 

    corr = 0.5 # corr 올라가면 worst of에서 더 payoff가 좋아지고, corr이 낮아지면 반대임. 
    # corr이 극단적으로 1이면 1개 자산 가지고 하는거랑 같고, 
    # corr이 극단적으로 -1이면 하나 오르면 하나 무조건 반대로 떨어지는데 worst 인 것을 가지고 payoff 정하니까 개불리. 
    # trader의 입장에서 corr이 올라가면 반대로 엄청 깨진다. 
    # corr은 hedge가 직접 안된다. 하려면 간접적으로... 주가 빠질 때 corr 올라가는것을 이용해 주식 delta로 어느 정도는 hedge가 되지만 
    # 정확히 delta, gamma 헤지하듯 하는 것은 못한다. 
    # 그래서 이런거 운용할 때 corr이 가장 큰 리스크고, corr을 어떻게 측정하느냐가 중요.
    # corr은 어떻게든 추정해서 의사결정에 반영해야 하는 것. 
    # 이것도 안전빵으로, corr=0.5쯤이라 생각되면 확 그냥 0.8로 quote해버림. 

    oh = 0.5 # overhedge parameter. 디지털 옵션을 call spread 구조로 만들 때의 기울기. 
    # 기울기 작을수록 overhedge를 하는 것. 
    # 걍 기울기 작게하면 장땡? ㄴㄴ. ELS 파는것도 다 경쟁인데... 이러면 당연히 quote가 안좋아짐. 
    # 은행이 quote 비교해보고 overhedge 심하게 하면 coupon 조금밖에 못주니까 안사줌. 

    fdm_price = exfdm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, nx, ny, nt, oh)
    print("FDM Price = ", fdm_price)

    analytic_price = ql_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, option_type, oh)
    print("Stulz Price = ", analytic_price) # analytic 가격 


# 디지털 옵션 관련 설명 ppt에. 

