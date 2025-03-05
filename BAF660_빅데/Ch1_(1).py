#%% 
# [1]
import pandas as pd
import numpy as np

missdict = {
    'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'f2': [10., None, 20., 30., None, 50., 60., 70., 80., 90.],
    'f3': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C']
}
missdata = pd.DataFrame(missdict)
missdata.info()

#%% 
# [2]
missdata.isna().mean()

#%% 
# [3]
tmpdata1 = missdata.dropna()
tmpdata1

#%% 
# [4]
tmpdata2 = missdata.dropna(subset=['f3'])
tmpdata2

#%% 
# [5]
numdata = missdata.select_dtypes(include=['int64', 'float64'])
tmpdata3 = numdata.fillna(-999, inplace=False)
tmpdata3.describe()

#%% 
# [6]
numdata.mean()

#%% 
# [7]
tmpdata4 = numdata.fillna(numdata.mean(), inplace=False)
tmpdata4

#%% 
# [8]
missdata.groupby('f3')['f2'].mean()

#%% 
# [9]
missdata.groupby('f3')['f2'].transform('mean')

#%% 
# [10]
tmpdata5 = numdata.copy()
tmpdata5['f2'].fillna(missdata.groupby('f3')['f2'].transform('mean'), inplace=True)
tmpdata5

#%% 
# [11]
missdata_tr = missdata.dropna()
x_tr = missdata_tr[['f1']]
y_tr = missdata_tr['f2']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_tr, y_tr)

missdata_ts = missdata[missdata.isnull().any(axis=1)]
x_ts = missdata_ts[['f1']]

predicted_values = model.predict(x_ts)
tmpdata6 = missdata.copy()
tmpdata6.loc[tmpdata6['f2'].isnull(), 'f2'] = predicted_values
tmpdata6

#%% 
# [12]
missdata_num = missdata.copy()
missdata_num['f3'] = missdata_num['f3'].map({'A': 1, 'B': 2, 'C': 3})

#%% 
# [13]
missdata_num

#%% 
# [14]
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)

#%% 
# [15]
tmpdata7 = imputer.fit_transform(missdata_num)
pd.DataFrame(tmpdata7)

#%% 
# [16]
outdict = {
    'A': [10, 0.02, 0.3, 40, 50, 60, 712, 80, 90, 1003],
    'B': [0.05, 0.00015, 25, 35, 45, 205, 65, 75, 85, 3905]
}
outdata = pd.DataFrame(outdict)
Q1 = outdata.quantile(0.25)
Q3 = outdata.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

((outdata < lower_bound) | (outdata > upper_bound))

#%% 
# [17]
outliers = ((outdata < lower_bound) | (outdata > upper_bound)).any(axis=1)
outliersdata = outdata[outliers]
outliersdata

#%% 
# [18]
standardizeddata = (outdata - outdata.mean()) / outdata.std()
standardizeddata

#%% 
# [19]
outliers2 = ((standardizeddata < -3) | (standardizeddata > 3)).any(axis=1)
outliersdata2 = outdata[outliers2]
outliersdata2

#%% 
# [20]
import matplotlib.pyplot as plt
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers + 2, X_inliers - 2, X_outliers]

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], color='k', s=20)

#%% 
# [21]
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)  # 1: inlier, -1: outlier
outlier_mask = y_pred == -1

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], color='b', s=20, label='Inliers')
plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], color='r', s=50, label='Outliers')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

#%% 
# [22]
from sklearn.ensemble import IsolationForest
clf2 = IsolationForest(contamination=0.1)
clf2.fit(X)
y_pred2 = clf2.predict(X)  # 1: inlier, -1: outlier
outlier_mask2 = y_pred2 == -1

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], color='b', s=20, label='Inliers')
plt.scatter(X[outlier_mask2, 0], X[outlier_mask2, 1], color='r', s=50, label='Outliers')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
