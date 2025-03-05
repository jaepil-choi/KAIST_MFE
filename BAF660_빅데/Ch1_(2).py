# # Ch1_(2)
# **March 4, 2025**
#
# This code is extracted from Ch1_(2).pdf.
#
# [Comparison Note]: The fresh extraction closely mirrors the previous attempt.
# Minor differences include explicitly naming parameters (e.g., in clip) for clarity.

# [1]
import pandas as pd
import numpy as np

datadict = {
    'F1': np.random.rand(100),
    'F2': np.random.randint(1, 100, size=100),
    'F3': np.random.randn(100),
    'F4': np.random.uniform(0, 10, size=100),
    'F5': np.random.normal(50, 10, size=100),
    'F6': np.random.exponential(5, size=100),
}
data = pd.DataFrame(datadict)
X_train = data[:75]
X_test = data[75:]

# [2]
X_train.info()

# [3]
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(X_train)

X_train1 = scaler1.transform(X_train)
X_test1 = scaler1.transform(X_test)

print(X_train1.mean(axis=0))
print(X_test1.mean(axis=0))
print(X_train1.std(axis=0))
print(X_test1.std(axis=0))

# [4]
from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
scaler2.fit(X_train)

X_train2 = scaler2.transform(X_train)
X_test2 = scaler2.transform(X_test)

print(X_train2.max(axis=0))
print(X_test2.max(axis=0))
print(X_train2.min(axis=0))
print(X_test2.min(axis=0))

# [5]
datadict2 = {
    'F1': np.random.gamma(2, 2, 1000),
    'F2': np.random.normal(0, 1, 1000),
    'F3': np.random.uniform(0, 1, 1000)
}
data2 = pd.DataFrame(datadict2)

# [6]
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
pt.fit(data2)
data2tr = pt.transform(data2)

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 4))
plt.hist(data2['F1'], bins=30, color='yellow', alpha=0.5)
plt.hist(data2tr[:, 0], bins=30, color='blue', alpha=0.5)

# [7]
data2_01 = data2.quantile(0.01)
data2_99 = data2.quantile(0.99)
# Explicitly naming lower and upper for clarity.
data2tr2 = data2.clip(lower=data2_01, upper=data2_99, axis=1)

plt.figure(figsize=(5, 4))
plt.hist(data2['F1'], bins=30, color='yellow', alpha=0.5)
plt.hist(data2tr2['F1'], bins=30, color='blue', alpha=0.5)

# [8]
bin_bdr = [0, 2.5, 5.0, 7.5, float('inf')]
F1_bin = pd.cut(data2['F1'], bins=bin_bdr, labels=False)
F1_bin

# [9]
F2_rank = data2['F2'].rank()
F2_normalized_rank = F2_rank / data2.shape[0]
F2_normalized_rank

# [10]
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
qt.fit(data2)
data2tr3 = qt.transform(data2)
plt.figure(figsize=(5, 4))
plt.hist(data2['F1'], bins=30, color='yellow', alpha=0.5)
plt.hist(data2tr3[:, 0], bins=30, color='blue', alpha=0.5)

# [11]
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
city = {
    'city': ['Seoul', 'Tokyo', 'Paris', 'Paris', 'Tokyo', 'Seoul',
             'London', 'Madrid', 'Seoul', 'Beijing', 'London', 'Paris']
}
citydf = pd.DataFrame(city)
label_encoder = LabelEncoder()
citydf['city_encoded'] = label_encoder.fit_transform(citydf['city'])
citydf

# [12]
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(citydf[['city']])
pd.DataFrame(one_hot_encoded, columns=label_encoder.classes_)

# [13]
freq = citydf['city'].value_counts()
freq

# [14]
citydf['city'].map(freq)

# [15]
tgdict = {
    'F1': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'F2': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male'],
    'Y': [20, 50, 60, 80, 30, 50]
}
tgdf = pd.DataFrame(tgdict)

# [16]
tg_mean = tgdf.groupby('F2')['Y'].mean()
tg_mean

# [17]
tgdf['F3'] = tgdf['F2'].map(tg_mean)

# [18]
# Alternative: tgdf['F3'] = tgdf.groupby('F2')['Y'].transform('mean')

# [19]
tgdf

# [20]
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=8, n_redundant=12, random_state=1)
print(X.shape, y.shape)

# [21]
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
skb = SelectKBest(f_classif)
skbfit = skb.fit(X, y)
dfscores = pd.DataFrame(skbfit.scores_, columns=['score'])
dfscores.sort_values('score', ascending=False)

# [22]
skb = SelectKBest(f_classif, k=5)
skbfit = skb.fit(X, y)
skb.get_support()

# [23]
skb.transform(X).shape

# [24]
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=8, verbose=1)
rfefit = rfe.fit(X, y)

# [25]
rfefit.get_support()

# [26]
rfefit.transform(X).shape

# [27]
from sklearn.feature_selection import RFECV
rfecv = RFECV(model, cv=5)
rfecvfit = rfecv.fit(X, y)

# [28]
rfecvfit.get_support()

# [29]
rfecvfit.transform(X).shape

# [30]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
sfsfit = sfs.fit(X, y)

# [31]
sfsfit.get_support()

# [32]
sfsfit.transform(X).shape

# [33]
# Note: In a script, shell commands like the following should be run externally.
# ! pip install Boruta

# [34]
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=123, max_depth=5)
brtfs = BorutaPy(rf, n_estimators=7, max_iter=15, verbose=1, random_state=123, alpha=0.01)
np.int = np.int64
np.float = np.float64
np.bool = np.bool_
brtfs.fit(X, y)

# [35]
brtfs.support_

# [36]
brtfs.transform(X).shape

# [37]
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
selector = SelectFromModel(estimator=RandomForestClassifier())
selector.fit(X, y)

# [38]
selector.get_support()

# [39]
selector.transform(X).shape

# [40]
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], random_state=1)

# [41]
pd.Series(y).value_counts()

# [42]
plt.figure(figsize=(5, 4))
plt.plot(X[y==0, 0], X[y==0, 1], 'b+', label="class 0")
plt.plot(X[y==1, 0], X[y==1, 1], 'r*', label="class 1")
plt.legend()

# [43]
from imblearn.under_sampling import CondensedNearestNeighbour
undersample1 = CondensedNearestNeighbour(n_neighbors=1)
X1, y1 = undersample1.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(X1[y1==0, 0], X1[y1==0, 1], 'b+', label="class 0")
plt.plot(X1[y1==1, 0], X1[y1==1, 1], 'r*', label="class 1")
plt.legend()

# [44]
from imblearn.under_sampling import TomekLinks
undersample2 = TomekLinks()
X2, y2 = undersample2.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(X2[y2==0, 0], X2[y2==0, 1], 'b+', label="class 0")
plt.plot(X2[y2==1, 0], X2[y2==1, 1], 'r*', label="class 1")
plt.legend()

# [45]
from imblearn.over_sampling import SMOTE
oversample1 = SMOTE()
OX1, Oy1 = oversample1.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(OX1[Oy1==0, 0], OX1[Oy1==0, 1], 'b+', label="class 0")
plt.plot(OX1[Oy1==1, 0], OX1[Oy1==1, 1], 'r*', label="class 1")
plt.legend()

# [Comparison Note - Ch1_(2)]: This fresh extraction is essentially the same as the previous one,
# with only minor clarifications (e.g., explicit parameter names in clip) for readability.

# [Final Linter Check]:
# - All variables are defined.
# - All necessary libraries are imported.
# - There are no syntax errors; shell commands are commented for script use.
