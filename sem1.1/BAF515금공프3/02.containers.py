# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: sandbox311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter 2. Containers

# %% [markdown]
# - string은 immutable해서 replace 안됨. 명시적으로 `.replace()` method 써주지 않는 이상. 
#     - `.replace()` 하면 해당 object를 지우고 다시 지정해주는 것임. 
#     - `.upper()`로 바뀌는 것도 새로운 값을 return하는 것이지 str object 자체는 안변함. 
# - list같은 mutable한 object들, str같은 immutable과는 달리 method로 바꿀 때 inplace로 객체 자체가 바뀌는 경우들이 있음. 
# - `sorted()`와 `.sort()`의 차이: inplace 여부
#     - inplace로 update하는 식으로 하면 좀 더 메모리 효율적
#     - `.sort()` 와 같이 inplace하는 메소드들은 None return 하고 끝난다는 것을 유의. 
# - list copy 방법:
#     - `[:]` 쓰던가 `.copy()` 매소드 사용
#     - 그냥 다른 변수에 할당하는 것은 reference를 바꾸는 것에 불과함. 
#     - 나중에 pd, np 쓰게되면, 그땐 view vs copy의 문제가 나옴. 
# - set operations
#     - `<=` or `.issubset()`
#     - `>=` or `.issuperset()`
#     - `^`: 대칭차집합 - 합집합에서 교집합 빼준 것. 둘 다 포함되는 것들 아예 제외. 
#     - `.add()` vs `.update()`
#         - inplace로 바꾸는 것은 동일
#         - add는 1개 원소
#         - update는 set(또는 collection)을 합해줌
#     - `.discard()` vs `.remove()`
#         - set에서 제거하는 것은 동일하나
#         - remove는 error을 raise. 
#

# %% [markdown]
# set 연산자

# %%
A = {1, 2, 3, 4}
B = {1, 2}
C = {3, 4, 5}

# %%
B <= A

# %%
A >= B

# %%
A^C

# %% [markdown]
# - shallow copy vs deep copy 
#     - shallow copy는 이름 그대로 list 등의 원소가 mutable 할 경우 그대로 reference만 가져온다. 
#         - element of list가 mutable일 때 그게 변하면 shallow copy 된 object에도 반영이 되어버림. 
#     - deep copy의 경우 안에 있는 것도 다 reference 대신 새로 memory 따기 때문에 진짜 안바뀜. 

# %%
import copy

a = [1,2,3,[5,6,7]]

b = copy.copy(a)
c = copy.deepcopy(a)

# %%
b[-1][0] = 99

# %%
a

# %%
c[-1][-1] = 11

# %%
a

# %%
c

# %% [markdown]
# |       | seq | mutable | elem_type                 |
# |-------|-----|---------|---------------------------|
# | str   | o   | x       | char                      |
# | list  | o   | o       | any                       |
# | tuple | o   | x       | any                       |
# | dict  | x   | o       | key: immutable value: any |
# | set   | x   | o       | immutable                 |

# %% [markdown]
# - Immutability is about whether an object can change its state or not.
# - Hashability is about whether an object can return a consistent hash value, making it usable as a dictionary key or set element.
#
# The key relationship between the two is that all immutable objects in Python are hashable by default because their immutable nature means their hash value will not change over time. However, not all hashable objects are necessarily immutable; a custom object can be made hashable by defining both a __hash__() method that returns a consistent hash value and an __eq__() method, but this does not make the object immutable.

# %% [markdown]
#
