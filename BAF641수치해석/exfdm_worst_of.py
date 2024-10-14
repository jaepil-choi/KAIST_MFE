#%%
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt 

def exfdm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, nx, ny, nt, oh):
    smax1, smax2 = s1*2, s2*2
    smin1, smin2 = s1*0, s2*0
    ds1, ds2 = (smax1-smin1) / nx, (smax2-smin2) / ny
    dt = t / nt

    i1, i2 = np.arange(nx+1), np.arange(ny+1)
    #vector & mesh of underlying assets
    s1_v, s2_v = smin1+i1*ds1, smin2+i2*ds2
    s_1, s_2 = np.meshgrid(s1_v, s2_v)

    #terminal condition = payoff
    v = np.zeros((ny+1, nx+1))
    v = np.where(np.minimum(s_1, s_2)>=k, 1, v)
    v = np.where((np.minimum(s_1, s_2)>=k-1/oh) & (np.minimum(s_1, s_2)<k), oh*(np.minimum(s_1, s_2)-(k-1/oh)), v)

    #coefficients
    a1 = dt*(sigma1*s_1)**2 / (2*ds1**2)
    b1 = dt*(r-q1)*s_1 / (2*ds1)
    a2 = dt*(sigma2*s_2)**2 / (2*ds2**2)
    b2 = dt*(r-q2)*s_2 / (2*ds2)
    c = dt*corr*sigma1*sigma2*s_1*s_2 / (4*ds1*ds2)
    d1, u1, d2, u2 = a1-b1, a1+b1, a2-b2, a2+b2
    m = -2*a1 - 2*a2 - dt*r
    
    d1, u1, d2, u2, c, m = \
        d1[1:-1,1:-1], u1[1:-1,1:-1], d2[1:-1,1:-1], u2[1:-1,1:-1], c[1:-1,1:-1], m[1:-1,1:-1]
    
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
            (1 + m) * v[1:-1, 1:-1]
        
        v[1:-1, 1:-1] = temp

        #boundary condition: gamma = 0
        v[0,:] = 2*v[1,:]-v[2,:]
        v[-1,:] = 2*v[-2,:]-v[-3,:]
        v[:,0] = 2*v[:,1]-v[:,2]
        v[:,-1] = 2*v[:,-2]-v[:,-3]
        v[0,0] = v[0,1] + v[1,0] - v[1,1]
        v[-1,0] = v[-2,0] + v[-1,1] - v[-2,1]
        v[0,-1] = v[0,-2] + v[1,-1] - v[1,-2]
        v[-1,-1] = v[-1,-2] + v[-2,-1] - v[-2,-2]

    f = RectBivariateSpline(s2_v, s1_v, v)
    return f(s2, s1)[0,0]



if __name__=="__main__":
    from ql_worst_of import ql_worst_of
    option_type = "call"
    
    s1, s2 = 100,100
    r = 0.02
    q1, q2 = 0.015, 0.01
    k = 95
    t = 1
    sigma1, sigma2 = 0.15, 0.2
    corr = 0.5
    nx, ny, nt = 200, 200, 4000
    oh = 0.5

    fdm_price = exfdm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, nx, ny, nt, oh)
    print("FDM Price = ", fdm_price)

    analytic_price = ql_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, option_type, oh)
    print("Stulz Price = ", analytic_price)
