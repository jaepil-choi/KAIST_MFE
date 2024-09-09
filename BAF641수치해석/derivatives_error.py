
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
x = 1.5
d = fprime(x)
print("derivative = ", d)


#%%
p = np.linspace(1,16,151)
h = 10**(-p)

def forward_difference(x,h):
    return (fun(x+h)-fun(x)) / h

def central_difference(x,h):
    return (fun(x+h)-fun(x-h)) / (2*h)

fd = forward_difference(x, h)
cd = central_difference(x, h)
print("forward = ", fd)
print("central = ", cd)

fd_error = np.log(np.abs(fd-d)/np.abs(d))
cd_error = np.log(np.abs(cd-d)/np.abs(d))
plt.plot(p,fd_error, p, cd_error)
plt.legend(['forward difference', 'central difference'])