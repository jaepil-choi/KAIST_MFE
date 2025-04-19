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
# # 경통분 HW6

# %%
import numpy as np
import pandas as pd

from scipy.stats import norm
import statsmodels.api as sm

# %% [markdown]
# ## 1. 

# %%
data = pd.DataFrame(
    [
        (0, 4), 
        (1, 3), 
        (6, 0), 
        (3, 2), 
        (5, 1),
    ], 
    columns=['x', 'y']
)

data

# %% [markdown]
# ### (a)

# %%
X = sm.add_constant(data['x'])
model = sm.OLS(data['y'], X)
results = model.fit()

# %%
print(results.summary())

# %% [markdown]
# ### (b)

# %% [markdown]
# Check the above result. 

# %% [markdown]
# ### (c)

# %%
new_input = pd.DataFrame({'x': [1]})
new_X = sm.add_constant(new_input, has_constant='add')
new_X

# %%
pred_y = results.get_prediction(new_X)

# %%
pred_y.summary_frame(alpha=0.05)

# %% [markdown]
# ### (d)

# %% [markdown]
# check the regression result

# %% [markdown]
# ## 2. 

# %% [markdown]
# ### (a)

# %% [markdown]
# ### (b)

# %% [markdown]
# ### (c)

# %% [markdown]
# ### (d)

# %%
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Given model coefficients and errors
intercept = 23.6409
slope = 0.6527
rse = 1.779
df = 12

# New data point
new_x = 70
new_data = pd.DataFrame({'x': [new_x]})
new_data = sm.add_constant(new_data, has_constant='add')

# Manually create the model's coefficient array
coeffs = np.array([intercept, slope])

# Predict the new value of y
predicted_y = np.dot(new_data, coeffs)[0]

# Calculate t-critical value for 95% confidence interval
t_critical = stats.t.ppf(1 - 0.025, df)

# Calculate the standard error of the prediction
X = sm.add_constant(data['x'])
XtX_inv = np.linalg.inv(np.dot(X.T, X))
se_pred = np.sqrt(rse**2 + np.dot(np.dot(new_data, XtX_inv), new_data.T)[0, 0])

# Construct the prediction interval
margin_of_error = t_critical * se_pred
prediction_interval = (predicted_y - margin_of_error, predicted_y + margin_of_error)

# Output the results
print(f"Predicted value of y for x = {new_x}: {predicted_y}")
print(f"95% prediction interval: {prediction_interval}")


# %%
