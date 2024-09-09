
#%%
import numpy as np 
import matplotlib.pyplot as plt 
def fun(x):
    return np.cos(x**x) - np.sin(np.exp(x))

def fprime(x):
    return -np.sin(x**x)*(x**x)*(np.log(x)+1)  - np.cos(np.exp(x))*np.exp(x)


x = np.linspace(0.5,2.5,101)
y = fun(x)
plt.plot(x,y,'-')

#%%
x = 1.5 # 이 점에서의 미분계수를 구해보자. 
d = fprime(x) # 이건 analytic solution
print("derivative = ", d)


#%%
p = np.linspace(1,16,151)
h = 10**(-p)

def forward_difference(x,h): # 수치적 방법들
    return (fun(x+h)-fun(x)) / h

def central_difference(x,h):
    return (fun(x+h)-fun(x-h)) / (2*h)

fd = forward_difference(x, h)
cd = central_difference(x, h)
print("forward = ", fd)
print("central = ", cd)

fd_error = np.log(np.abs(fd-d)/np.abs(d)) # 로그로 바꿔서 지수끼리만 계산할 수 있도록. 
cd_error = np.log(np.abs(cd-d)/np.abs(d))
plt.plot(p,fd_error, p, cd_error)
plt.legend(['forward difference', 'central difference'])

# 해석
# - 가로축이 h. 2, 4, 등의 단위는 10^-2, 10^-4 를 나타냄. 
# - 세로축이 오차를 log scale로 쓴 것. 
# - forward 가 central보다 훨씬 빨리 줄어든다. (order of h^2라서)
# - 직선처럼 줄어드는 것: truncation error 줄어드는 것. 
# - 올라가는 것, 불규칙하게 올라가는데, rounding error 때문에 올라가는 것. 
# - 적절한 h를 찾는 것이 목표니 제일 작은 h 값의 위치를 고르면 된다. 
# - 그래서 다른 코드들에서도 대략적으로 h를 10^-6 정도로 잡는다. 이것이 best practice. 
