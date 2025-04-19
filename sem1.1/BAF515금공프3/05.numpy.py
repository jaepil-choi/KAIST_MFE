# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chp 5. Numpy

# %%
import numpy as np

# %%
np.__version__ # 최신 버전 쓰고 있음. 

# %% [markdown]
# - same dtype으로 array에 들어감. 

# %%
arr = np.array([
    [1.5, 1, 3],
    [2, 2, 2],
    ]) # 하나가 float이면 전체가 float

# %%
arr.ndim

# %%
arr.size

# %%
arr.shape

# %%
arr.dtype # float의 default는 float64

# %% [markdown]
# - dtypes:
#     - int32
#     - int64
#     - float32
#     - float64
#     - complex128
#     - bool

# %% [markdown]
# - Empty array creation
#
# 틀 만들어놓고 시작하는 경우 유용

# %%
arr = np.zeros((3, 4))
arr

# %%
arr = np.ones((3, 4))
arr

# %%
arr = np.empty((3, 4))
arr # random한 값들이 채워져 있어야 하는데 1.0이 계속 들어가네. 이상하다...

# %%
arr[0, 0]

# %%
arr = np.eye(3)
arr # 3x3의 identity matrix

# %%
arr = np.full((3, 4), 99.)
arr

# %% [markdown]
# - sequence of number를 가지고 create하는 방법

# %%
arr = np.arange(1, 10, 0.1)
arr

# %%
arr = np.linspace(start=1, stop=10, num=50) # 1이상 10이하 50개의 숫자를 만들어라. 같은 간격. high도 포함됨.(이하)
arr

# %% [markdown]
# - random number creation

# %%
np.random.seed(42)

# %%
# uniform distribution (0~1)
arr = np.random.rand(3, 4) # shape가 들어감. tuple 아닌 채로. 
arr

# %%
# uniform distribution (low~high)
arr = np.random.uniform(low=10, high=100, size=(3, 4))
arr

# %%
# standard normal distribution
arr = np.random.randn(3, 4) # shape가 들어감. tuple 아닌 채로.
arr

# %%
# normal distribution
arr = np.random.normal(loc=6, scale=4, size=(3, 4)) # loc: mean, scale: std
arr

# %%
# random integer. discrete uniform distribution
arr = np.random.randint(low=1, high=100, size=(3, 4)) # 1이상 100미만의 정수를 3x4 matrix로 만들어라.
arr

# %% [markdown]
# 이 외에도 `np.random` 내에 
#
# - binomial, 
# - chisquare, 
# - exponential, 
# - f, 
# - normal, 
# - poisson, 
# - standard_t, 
# - standard_normal, 
# - uniform 
#
# 분포가 준비되어 있음. 

# %% [markdown]
# Exercise 1

# %% [markdown]
# Exercise 1 
#
# - Create an 1 dimensional array of all the even integers from 30 to 70.
# - Create an 3x5 array filled with 15 random numbers from a standard normal distribution.
# - Create a 2 dimensional array with shape (2, 5) by randomly extracting 10 integers between 10 and 30 inclusive.
# - Create an array of shape (5,6), filled with integer zeros.

# %%
arr = np.arange(30, 70+1)
arr

# %%
arr = np.random.randn(3, 5)
arr

# %%
arr = np.random.randint(10, 30+1, (2, 5)) # inclusive하게 뽑으려면 +1 해줘야 함.
arr

# %%
arr = np.zeros((5, 6), dtype=int) # default는 floating point zero. integer로 뽑으려면 dtype을 지정해줘야 함.
arr

# %%
arr.dtype # int의 default는 int32

# %% [markdown]
# - accessing values in numpy arrays
#     - indexing
#     - slicing
#     - numericalindexing
#     - logical indexing (boolean masking)
#         - masking은 진짜 중요함. Haafor에서의 기억...

# %% [markdown]
# - indexing
#     - one-dimensional & multi-dimensional

# %%
arr = np.random.randn(3, 4) 
arr

# %%
arr[0][2] # 아무도 이렇게 안 씀. 

# %%
arr[0, 2]

# %%
arr[2, :]

# %%
arr[:, 2]

# %% [markdown]
# exercise 2

# %%
grid = np.arange(6) + np.arange(0, 60, 10).reshape(-1, 1)
grid

# %%
#1
grid[0, 3:5]

# %%
#2
grid[4:6, 4:6]

# %%
#3
grid[:, 2]

# %%
#4
grid[:, 2:3]

# %%
#5
grid[2:5:2, 0:5:2]

# %% [markdown]
# exercise 3

# %%
grid = np.zeros((8,8), dtype=np.int64)
grid

# %%
grid[1::2, 0::2] = 1
grid

# %%
grid

# %% [markdown]
# `np.newaxis` == None

# %%
a = np.array([1,2,3])
a.reshape(1,3)

# %%
a[np.newaxis, :]

# %%
a[None, :]

# %%
a[:, None]

# %% [markdown]
# 그 외 shape manipulations

# %%
A = np.random.rand(2, 3)
A

# %%
A.flatten()

# %%
A.flatten('F') # flatten with Fortran order. 위 아래 위래 위 아래 이런 식으로 column 먼저 flatten

# %% [markdown]
# concatenation

# %%
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# %%
y = np.array([
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
])

# %%
np.concatenate([x, y], axis=0)

# %%
np.concatenate([x, y], axis=1) # 모양 안맞으면 에러 남. 

# %% [markdown]
# concat으로 쓰는 대신 명시적으로 
#
# `vstack`, `hstack` 써주자. 

# %%
np.vstack([x, y])

# %%
np.hstack([x, x])

# %%
two = np.array([
    [1, 2],
    [3, 4]
])

one = np.array([5, 6])

np.column_stack([two, one]) # 이런거도 있긴한데 굳이.. 그냥 one을 reshape해서 column으로 만들어서 hstack해도 됨.

# %% [markdown]
# 반대로 split으로 찢을 수도 있음

# %%
grid

# %%
np.vsplit(grid, 2) # 2개로 나눠라.

# %%
np.vsplit(grid, [2]) # 2번 인덱스에서 나눠라.

# %%
np.split(grid, [2], axis=0)

# %% [markdown]
# array끼리의 연산
#
# 원래는 element-wise. shape가 같아야 함. 
#
# 하지만 broadcasting이 가능한 경우는 ㄱㅊ
#
# - `@` 와 `np.dot` 같음
#

# %% [markdown]
#
# 그 외 universal function들이 있다. 
#
# - `np.exp`
# - `np.log`
# - `np.log10`
# - `np.sqrt`
#     - `np.cbrt`도 있고 
#     - `np.power` 쓸 수 있음. 
#     - 아니면 그냥 `**(1/3) ` 이게 제일 깔끔하지
# - `np.sin` 외 삼각함수 2종
# - `np.rint`
#     - nearest "even" integer로 감. 반올림이 아님. 2.5면 2로 가고 3.5면 4로 감. 
#     - 반올림은 `np.round(9.3456, 2)` >> 9.35 써야. 
# - `np.ceil` 올림
# - `np.floor` 버림
# - `np.abs`
# - `np.isnan`
# - `np.inf`
#

# %% [markdown]
# numpy broadcasting
#
# - 차원의 shape가 같거나
# - 적어도 둘 중 하나의 작은 dimension이 1이라 그걸 늘렸을 때 다른 array와 shape를 맞출 수 있어야 함. 
#

# %%
a = np.array([1, 2, 3])
a

# %%
b = np.array([
    [4, 5, 6],
    [7, 8, 9]
])
b

# %%
a + b

# %% [markdown]
# aggregation operations 
#
# 이거 좋다. 속도가 빠름. 앵간한 것은 이걸로 해주는 것이 좋다. 
#
# 그리고 nan-safe version도 있음. nan 무시하고 연산. 
#
# - `np.sum`, `np.nansum`
# - `np.prod`, `np.nanprod`
# - `np.mean`, `np.nanmean` 매우 소중하다
# - `np.std`, `np.nanstd`
# - `np.var`, `np.nanvar`
# - `np.min`, `np.nanmin`
# - `np.max`, `np.nanmax`
# - `np.median`, `np.nanmedian` 
# - `np.percentile`, `np.nanpercentile` 매우 소중하다
# - `np.any` boolean으로 판단
# - `np.all` boolean으로 판단

# %%
a = np.array([1, 2, 3, np.nan])
np.prod(a)

# %%
np.nanprod(a)

# %% [markdown]
# sorting & searching function
#
# 이것들도 매우 중요하다. 
#
# - `np.sort`
# - `np.argsort` 특히 중요
# - `np.argmax`, `np.argmin`
# - `np.nonzero`
# - `np.where(condition, [, x, y])` 매우 중요

# %%
p = np.random.randn(3, 4)
p

# %%
np.where(p < 0)

# %% [markdown]
# condition만 들어가면 이렇게 참인 곳의 array만 shape 순으로 나옴. 
#
# (3, 4)의 shape에서 row 먼저 좌표가 쭉 나오고, column 좌표가 그 다음에 나옴. 
#

# %%
# 일반적인 사용 

condition = np.array([
    [True, False],
    [False, True]
])

x = np.array([[1, 2],
              [3, 4]])

y = np.array([[5, 6],
              [7, 8]])

np.where(condition, x, y)

# %% [markdown]
#
# condition 뒤에 x, y가 들어가면 true일 때 x, false일때 y의 그 자리에 있는 것을 반환. 
#
# 이 때 일반적으로 shape가 맞아야 하지만, broadcastable 하면 아래처럼 동작은 할 수 있음. 
#
# 근데 너무 헷갈리니까 그냥 얌전히 condtiion, x, y 모두 사이즈를 맞춰서 넣어주는 것이 좋음. 

# %%
# Condition: 2x2 boolean array
condition = np.array([[True, False],
                      [True, True]])

# x: 2x2 array
x = np.array([[1, 2],
              [3, 4]])

# y: 1x2 array
y = np.array([5, 6]) # broadcasting 사용

# Attempt to use numpy.where with incompatible shapes
result = np.where(condition, x, y)

print(result)


# %%
p

# %%
argmin = np.argmin(p)
argmin

# %%
argmin = np.argmin(p, axis=0)
argmin

# %% [markdown]
# 위처럼, argmin 했을 때 그냥 flatten된 index가 나와서 불편함. row, column 위치가 나오지 않음. 
#
# 이 경우 아래와 같이 shape를 주고 위치를 원복하는 것이 좋음. 

# %%
np.unravel_index(argmin, p.shape)

# %%
# exercise 9
np.random.seed(123)
mat = np.random.randn(6, 4)
mat

# %%
vs = np.exp(mat[:, 0])[None, :]
vs.T

# %%
np.hstack([mat, vs.T])

# %% [markdown]
# Logic functions
#
# - `np.unique(a)`: unique, sorted
# - `np.intersect1d(a)`: intersect, sorted
# - `np.union1d(a)`: union, sorted
# - `np.in1d(a)`: sorted, 순서 중요

# %%
sid1 = np.array(['aapl', 'msft', 'nvda', 'cpng', 'amzn', 'amzn'])
sid2 = np.array(['msft', 'nvda', 'amzn', 'meta', 'intc', 'intc', 'intc'])

# %%
np.unique(sid1) 

# %%
np.intersect1d(sid1, sid2)

# %%
np.union1d(sid1, sid2)

# %%
np.in1d(sid1, sid2) # sid1 in sid2 ?

# %%
np.in1d(sid2, sid1) # sid2 in sid1?

# %% [markdown]
# 선형대수 함수들
#
# - `np.transpose(a)`: `.T` 랑 같음. 
# - `np.dot(a, b)`: = `@`
# - `np.diag(a)`: 대각 원소 반환

# %%
a = np.random.randn(3, 5)
a


# %%
np.diag(a) # 꼭 정방행렬이 아니어도 diag를 반환함. 1d로. 

# %%
np.trace(a) # 대각선의 합 = trace

# %%
b = np.random.randn(3, 3)

np.linalg.det(b) # 행렬식 determinant 반환. 

# %%
np.linalg.eig(b) # eigenvalue와 eigenvector 반환

# %%
