# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: sandbox39
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BAF515 금공프3 Quiz 1
#
# **20249433 MFE 최재필**
#
#
# 수업 중에 배운 내용만을 이용해서 코드를 작성해주시면 됩니다.
#
# 사용자 정의함수, 조건문, 반복문 등 아직 배우지 않은 내용을 활용하면 안됩니다.
#

# %%
import math

# %% [markdown]
# ## 1. 
#
# Calculate the following

# %% [markdown]
# ### (1)
#
# ![image.png](attachment:image.png)

# %%
a = 2
b = -1
c = -15

roots = [
    (-b + math.sqrt(b**2 - 4*a*c)) / (2*a), 
    (-b - math.sqrt(b**2 - 4*a*c)) / (2*a),
    ]

roots

# %% [markdown]
# ### (2)
#
# ![image.png](attachment:image.png)

# %%
mu = 2
variance = 3

x = 1

f = (1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-((x - mu)**2) / (2 * variance))
f

# %% [markdown]
# ## 2. 
#
# Briefly explain why the error occur in the following expression

# %% [markdown]
# ### (1)
#
# ![image.png](attachment:image.png)

# %%
a = input('enter a number')

# %%
a

# %%
a + 3

# %% [markdown]
# As shown above, it's because `input()` function takes the input as `str`, and `str` cannot add with 3, which is an `int`. 
#
# To correct this: 

# %%
int(a) + 3

# %% [markdown]
# ### (2)
#
# ![image.png](attachment:image.png)

# %%
tmp = 'My String'

# %%
tmp[10]

# %% [markdown]
# As shown above, it's because string index is out of range. The given string, `tmp` only has 0~8 indices (length: 9)

# %%
len(tmp)

# %%
tmp[8]

# %% [markdown]
# ### (3)
#
# ![image.png](attachment:image.png)

# %%
ex1 = 'sample string'
ex2 = ex1.upper

# %%
ex2[:4]

# %% [markdown]
# This error occurs because `.upper()` method was not called properly. It should have `()` in the end. If `()` is missing, `.upper` is the method function itself. 

# %%
type(ex2)

# %% [markdown]
# To correct this:

# %%
ex2 = ex1.upper()

ex2[:4]

# %% [markdown]
# ## 3. 
#
# Create the following string object 'grade'
#
# ![image.png](attachment:image.png)

# %%
grade = 'ABCDF'

# %% [markdown]
# ### (1)
#
# Using the `+` operator on `grade`, create `grade_str` as follows.
#
# ![image.png](attachment:image.png)

# %%
grade_str = grade + 'F' + grade[::-1]

grade_str

# %% [markdown]
# ### (2)
#
# Count the number of 'A' in `grade_str`

# %%
list(grade_str).count('A')

# %% [markdown]
# ### (3)
#
# Present 4 different slicing expressions to extract 'FFF' in `grade_str`

# %%
# 1 Normal Slicing
grade_str[4:7]

# %%
# 2 Using negative step
grade_str[6:3:-1]

# %%
# 3 Using negative indices
grade_str[-7:-4]

# %%
# 4 Using negative indices and negative step
grade_str[-5:-8:-1]

# %% [markdown]
# ### (4)
#
# Modify `grade_str` in (3) as the following
#
# ![image.png](attachment:image.png)

# %%
grade_str = grade_str.replace('FFF', 'AAA')
grade_str

# %% [markdown]
# ### (5)
#
# Change all letters of `grade_str` to lower case

# %%
grade_str.lower()

# %% [markdown]
# ## 4.
#
# Briefly explain why the error occurs in the following expression. 

# %% [markdown]
# ### (1)
#
# ![image.png](attachment:image.png)

# %%
L = [
    [1, 3, 5, 7, 9],
    [2, 4, 6, 8, 10],
    ]

# %%
L[0][1:2] = 30

# %% [markdown]
# This error occurs because an `int` value instead of iterable(`list`) was assigned. 

# %%
L[0][1:2]

# %% [markdown]
# To correct this:

# %%
L[0][1:2] = [30]

# %%
L

# %% [markdown]
# ### (2)
#
# ![image.png](attachment:image.png)

# %%
T = (10, 20, 30)

# %%
T[:2] + (40)

# %% [markdown]
# This error occurs because `(40)` was interpretted as `int` instead of `tuple`
#
# To correct this:

# %%
T[:2] + (40,)

# %% [markdown]
# ### (3)
#
# ![image.png](attachment:image.png)

# %%
D = {
    'A': 10,
    'B': 20,
    'C': 30,
    }

# %%
D2 = D

# %%
del D2['A']

# %%
D['A']

# %% [markdown]
# This error occurs because `D`'s key `'A'` has already been deleted. 
#
# Although variable `D2 = D` was declared, it is merely referencing D object in the memory. 
#
# Thus, the change in the `D2` object itself is shown in `D`. 
#
# To correct this:

# %%
D = {
    'A': 10,
    'B': 20,
    'C': 30,
    }

# %%
D2 = D.copy()

# %%
del D2['A']

# %%
D['A']

# %%
D

# %%
D2

# %% [markdown]
# ### (4)
#
# ![image.png](attachment:image.png)

# %%
D3 = {
    ['Park', 'male']: 30,
    ['Lee', 'female']: 28,
    ['Kim', 'male']: 34,
}

# %% [markdown]
# This error occurs because the dictionary's key is unhashable type, `list`. 
#
# To correct this: 

# %%
D3 = {
    ('Park', 'male'): 30,
    ('Lee', 'female'): 28,
    ('Kim', 'male'): 34,
}

# %% [markdown]
# ### (5)
#
# ![image.png](attachment:image.png)

# %%
dict_y = {
    (1,): 10, 
    (2,): 20,
    (3,): 30,
    (4,): 40,
}

# %%
dict_y[-2:]

# %% [markdown]
# This error occurs because dictionary, although it maintains its order from Python 3.7+, is not indexable nor subscriptible. 
#
# To correct this: (Although it is not common/recommended to slice dictionary key-value pairs,)

# %%
list(dict_y)[-2:] # Only maintains the keys

# %%
list(dict_y.items())[-2:] # Maintains the key-value pairs

# %% [markdown]
# ## 5. 
#
# Create the following list object 'days'.
#
# ![image.png](attachment:image.png)

# %%
days = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', ['Sat', 'Sun']]

# %% [markdown]
# ### (1)
#
# Extract elements from `days` as shown below
#
# ![image.png](attachment:image.png)

# %%
# 1

days[-1:]

# %%
# 2

days[-1:-6:-2]

# %%
# 3

days[-1][0]

# %% [markdown]
# ### (2)
#
# Modify `days` as shown below by applying the slicing(`:`) and concatenation operator(`+`) and name it `days2`
#
# ![image.png](attachment:image.png)

# %%
days2 = [days[:5]] + days[-1]
days2

# %% [markdown]
# ### (3)
#
# Modify `days2` in (2) as follows, by removing the 2 items 'Wed' and 'Fri'.
#
# ![image.png](attachment:image.png)

# %%
days2[0].remove('Wed')
days2[0].remove('Fri')

# %%
days2

# %% [markdown]
# ### (4)
#
# Modify `days2` in (3) as following, by inserting 'W' at the given position.
#
# ![image.png](attachment:image.png)

# %%
days2[0].insert(2, 'W')

# %%
days2

# %% [markdown]
# ## 6. 
#
# Create the following list object `Nums`.
#
# ![image.png](attachment:image.png)

# %%
Nums = [1, 5, 2, 7, 3, 6, 4]

# %% [markdown]
# ### (1)
#
# Append the largest element of `Nums` to the end of `Nums`.
#
# ![image.png](attachment:image.png)

# %%
Nums = Nums + [max(Nums)]
Nums

# %% [markdown]
# ### (2)
#
# Sort the elements in `Nums` in decreasing order. 
#
# ![image.png](attachment:image.png)

# %%
Nums.sort(reverse=True)

# %%
Nums

# %% [markdown]
# ### (3)
#
# Modify the `Nums` in (2) as the following. (Replace the 1st, 3rd, 5th and 7th elements in `Nums` with 'a'.)
#
# ![image.png](attachment:image.png)

# %%
Nums[0::2] = ['a']*4

# %%
Nums

# %% [markdown]
# ## 7. 
#
# Create the following tuple object 'price'.
#
# ![image.png](attachment:image.png)

# %%
price = (180, 130, 110, 160, 140, 170)

# %% [markdown]
# ### (1)
#
# Sort the items of `price` in ascending order so that `price` is displayed as below. 
#
# ![image.png](attachment:image.png)

# %%
price = tuple(sorted(price))
price

# %% [markdown]
# ### (2)
#
# Write a code that returns `True` if `price` has the value 170 and `False` otherwise. 

# %%
170 in price

# %% [markdown]
# ### (3)
#
# Insert 3 zeros instead of 5th value 160 in `price`, so that `price` is displayed as below. 
#
# ![image.png](attachment:image.png)

# %%
price[:4] + (0, 0, 0) + price[-1:]
