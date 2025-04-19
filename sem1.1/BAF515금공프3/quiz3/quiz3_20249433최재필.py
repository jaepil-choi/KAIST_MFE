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
# # 금공프3 Quiz 3 
#
# MFE 20249433 최재필

# %% [markdown]
# ## Q1 
#
# ![image.png](attachment:image.png)

# %%
i = 1
summation = 0

while i<=10 and i**4 < 1000:
    summation += i**4
    print(f'{i}: i^4={i**4} / Summation={summation}')
    i += 1

# %% [markdown]
# ## Q2
#
# ![image.png](attachment:image.png)

# %%
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]

# %% [markdown]
# ### (1)

# %%
z = [[a, b] for a in x for b in y]
z

# %% [markdown]
# ### (2)

# %%
z = [[a, b] for a in x for b in y if (a+b) >= 8]
z


# %% [markdown]
# ## Q3
#
# Grading Score

# %%
def grading(score):
    if not (0 <= score <= 100):
        return 'Score must be a number between 0 and 100!!'

    if 90 < score:
        grade = 'A'
    elif 80 < score:
        grade = 'B'
    elif 70 < score:
        grade = 'C'
    elif 60 < score:
        grade = 'D'
    else:
        grade = 'F'
    
    return f'Grade is {grade}!'


# %%
grading(score=75)

# %%
grading(-5)


# %% [markdown]
# ## Q4
#
# Explain why the error occrus

# %% [markdown]
# ### (1)

# %%
def infoprint(name, age, gender):
    print(name, 'is', age, 'years old', gender, '.')


# %%
infoprint(name='Kim', 'male')

# %% [markdown]
# Error occured because positional arguments should always come before keyword arguments

# %%
# To fix this, correct the order and add a missing argument
infoprint('Kim', 13, gender='male')

# %% [markdown]
# ### (2)

# %%
infoprint('Kim', gender='male')

# %% [markdown]
# The function requires all three arguments: `name`, `age`, and `gender` inthe correct order. 

# %%
# To fix this, add a missing argument
infoprint('Kim', 13, gender='male')

# %% [markdown]
# ### (3)

# %%
fac = 1

def myfactorial(n):
    for i in range(n):
        fac *= i + 1
    
    return fac


# %%
myfactorial(n=5)

# %% [markdown]
# A variable `fac` was not declared inside a function, `myfactorial()`. 
#
# Either `global` should be used or `fac` should be declared inside the function. 

# %%
# To fix this (1)

fac = 1

def myfactorial(n):
    global fac
    for i in range(n):
        fac *= i + 1
    
    return fac


# %%
myfactorial(n=5)


# %%
# To fix this (2)

def myfactorial(n):
    fac = 1
    for i in range(n):
        fac *= i + 1
    
    return fac


# %%
myfactorial(n=5)
