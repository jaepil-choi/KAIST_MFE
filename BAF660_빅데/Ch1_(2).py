#%% 
# [1]

########## # 1. Feature Transformation ##########

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

#%% 
# [2]
X_train.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 75 entries, 0 to 74
# Data columns (total 6 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   F1      75 non-null     float64
#  1   F2      75 non-null     int32  
#  2   F3      75 non-null     float64
#  3   F4      75 non-null     float64
#  4   F5      75 non-null     float64
#  5   F6      75 non-null     float64
# dtypes: float64(5), int32(1)
# memory usage: 3.4 KB

#%% 
# [3]
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler() # 표준화 변환
scaler1.fit(X_train) # scaler fitting은 train 데이터만 수행

X_train1 = scaler1.transform(X_train)
X_test1 = scaler1.transform(X_test) # test set transform도 train에서 학습한 scaler 사용
# information leackage 방지

print(X_train1.mean(axis=0))
print(X_test1.mean(axis=0))
print(X_train1.std(axis=0))
print(X_test1.std(axis=0))

# [ 1.03620816e-17  1.06951485e-16 -5.92118946e-18 -5.47710025e-17
#   2.87177689e-16  1.22864681e-16]
# [-0.0394831  -0.21916239 -0.05676997  0.08733307 -0.23170978  0.6437487 ]
# => test set, train set 에서 학습된 scaler로 전처리했지만 마찬가지로 거의 0

# [1. 1. 1. 1. 1. 1.]
# [0.9513569  0.88798781 1.19877918 0.87206489 1.13146366 1.44152395]
# => 같은 이유로 거의 1. train과 test의 population이 아주 다르지는 않다는 것. 

#%% 
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

# [1. 1. 1. 1. 1. 1.]
# [0.94294425 0.98958333 0.96394344 0.96091117 0.98759588 1.06503   ]
# [0. 0. 0. 0. 0. 0.]
# [ 0.02885686  0.         -0.00666648  0.03262773 -0.04797744 -0.00045541]

#%% 

# Power transformation을 위한 데이터 생성

# [5]
datadict2 = {
    'F1': np.random.gamma(2, 2, 1000),
    'F2': np.random.normal(0, 1, 1000),
    'F3': np.random.uniform(0, 1, 1000)
}
data2 = pd.DataFrame(datadict2)

#%% 
# [6]
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson') # box-cox는 negative에서 불가해서 이걸 씀. 
pt.fit(data2)
data2tr = pt.transform(data2)

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 4))
plt.hist(data2['F1'], bins=30, color='yellow', alpha=0.5) # 이게 오리지날날
plt.hist(data2tr[:, 0], bins=30, color='blue', alpha=0.5) # 이게 변환한거
# alpha=0.5 줘서 겹치는 부분 보이게하는 센스

#%% 
# [7]
data2_01 = data2.quantile(0.01)
data2_99 = data2.quantile(0.99)
data2tr2 = data2.clip(lower=data2_01, upper=data2_99, axis=1) # clip으로 outlier 날리고

plt.figure(figsize=(5, 4))
plt.hist(data2['F1'], bins=30, color='yellow', alpha=0.5) 
plt.hist(data2tr2['F1'], bins=30, color='blue', alpha=0.5)
# clip하면 그 부분 벽 막힌 것 처럼 팍 튀는거 보임 

#%% 
# [8]
bin_bdr = [0, 2.5, 5.0, 7.5, float('inf')] # 직접 range specify 한 것. 
F1_bin = pd.cut(data2['F1'], bins=bin_bdr, labels=False) # number --> category
# bins에 정수 넣으면 등간격으로 만들어 줌
F1_bin

#%% 
# [9]
F2_rank = data2['F2'].rank()
F2_normalized_rank = F2_rank / data2.shape[0]
F2_normalized_rank

#%% 
# [10]
from sklearn.preprocessing import QuantileTransformer
# 데이터를 cdf에서 각 누적 %에 매핑되는 값으로 변환해줌. 
# 이론상 이런 분포를 따라야 할 때 이런식으로 접근 가능. 

qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
qt.fit(data2)
data2tr3 = qt.transform(data2)
plt.figure(figsize=(5, 4))
plt.hist(data2['F1'], bins=30, color='yellow', alpha=0.5)
plt.hist(data2tr3[:, 0], bins=30, color='blue', alpha=0.5)

#%% 
# [11]
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# label encoder는 정수값으로 변환해줌. 
city = {
    'city': ['Seoul', 'Tokyo', 'Paris', 'Paris', 'Tokyo', 'Seoul',
             'London', 'Madrid', 'Seoul', 'Beijing', 'London', 'Paris']
}
citydf = pd.DataFrame(city)
label_encoder = LabelEncoder()
citydf['city_encoded'] = label_encoder.fit_transform(citydf['city']) # 이게 표준 .
citydf



#%% 
# [12]

# onehot encoder는 각각의 값에 대해 0, 1로 변환해줌.
one_hot_encoder = OneHotEncoder(sparse_output=False) # False 일 때 dense matrix로 반환.
one_hot_encoded = one_hot_encoder.fit_transform(citydf[['city']])
pd.DataFrame(one_hot_encoded, columns=label_encoder.classes_)

# dense로 하면 column에 label을 넣어 0 0 0 0 1 0 이런 식으로 row를 가지게 해 더 efficient 함
# 목적이 기준 테이블에 column으로 붙여야하는게 아니면 이렇게 만들어 쓰는 것도 가능. 

#%% 
# [13]
freq = citydf['city'].value_counts()
freq

# city
# Seoul      3
# Paris      3
# Tokyo      2
# London     2
# Madrid     1
# Beijing    1
# Name: count, dtype: int64

#%% 
# [14]
citydf['city'].map(freq) # series object를 mapper로 써서 dict처럼 mapping 해줌

# 0     3
# 1     2
# 2     3
# 3     3
# 4     2
# 5     3
# 6     2
# 7     1
# 8     3
# 9     1
# 10    2
# 11    3
# Name: city, dtype: int64

#%% 
# [15]
tgdict = {
    'F1': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'F2': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male'],
    'Y': [20, 50, 60, 80, 30, 50]
}
tgdf = pd.DataFrame(tgdict)

#%% 
# [16]
tg_mean = tgdf.groupby('F2')['Y'].mean()
tg_mean

#%% 
# [17]
tgdf['F3'] = tgdf['F2'].map(tg_mean) # 마찬가지로 series를 mapper로 쓰는 것. 

#%% 
# [18]
# Alternative: tgdf['F3'] = tgdf.groupby('F2')['Y'].transform('mean')

#%% 
# [19]
tgdf

#%% 
# [20]

########## # 2. Feature Selection ##########

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=8, n_redundant=12, random_state=1)
print(X.shape, y.shape)

#%% 
# [21]

########## ## 2.1 Filter 방식 ##########

from sklearn.feature_selection import SelectKBest
# ============ SelectKBest parameter =================
# score_func
#   : f_regression, mutual_info_regression # 여기의 f는 F-stats
#   : chi2, f_classif, mutual_info_classif 
# # chi2 는 독립성 검정통계량
# # 여기의 f는 anova F-stats
# # mutual_info는 correlation이랑 비스무리한 것. 
# # corr은 linear 한 관계를 봄.
# # mutual_info는 독립인지 연관되어있는지 joint distribution과 marginal distribution을 비교해봄. 
# # marginal joint dist의 곱이 joint dist와 같으면 두 변수는 독립이라고 볼 수 있음.
# # 이게 얼마나 다른지를 본다. 클수록 연관성 크고 작을수록 연관성 작고(즉 negative corr)
# k, percentile :

from sklearn.feature_selection import f_classif
skb = SelectKBest(f_classif)
skbfit = skb.fit(X, y)
dfscores = pd.DataFrame(skbfit.scores_, columns=['score'])
dfscores.sort_values('score', ascending=False)
# score을 볼 수 있다. 

#%% 
# [22]
skb = SelectKBest(f_classif, k=5)
skbfit = skb.fit(X, y)
skb.get_support() # feature selection 하는 애들이 다 쓸 수 있는 method 
# 변수가 선택되었는지 여부를 True False로 list로 반환. 

# array([False, False, False, False,  True, False, False, False, False,
#        False,  True, False, False,  True, False,  True, False,  True,
#        False, False])

#%% 
# [23]
skb.transform(X).shape

#%% 
# [24]

########## ## 2.2 Wrapper 방식 ##########

## Wrapper 방식. 
# regression에서 walk forward 같이 하나씩 넣어보며 하는 방식

# 참고: wrapper vs embedded
# wrapper: feature selection을 search problem으로 본다. 하나씩 늘려가고 줄여가며 eval metric은 따로 두고 고른다
# embedded: feature selection을 model에 포함시킨다. feature selection을 모델의 결과로써 학습시킨다. 

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE # Recursive Feature Elimination

# ============ RFE  parameter =================
# estimator : coef_ 와  feature_importances_ 가 산출가능한 sklearn 지도학습 알고리즘 쓰면 됨. (트리같은거)
# 절차: coef 절대값 작은 것 == 도움 안되는 것 제거 --> 또 모델 돌리고 --> 반복
# n_features_to_select : 선택할 특성수. default는 전체 특성변수의 절반. (10개면 5개)
# step : 매 단계 제거되는 특성 수를 정할 수 있다. default  1. 엄청 많으면 더 늘릴수도. 

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=8, verbose=1)
rfefit = rfe.fit(X, y)

#%% 
# [25]
rfefit.get_support()

#%% 
# [26]
rfefit.transform(X).shape

#%% 
# [27]

# 하지만, 대체 어디까지 select 해야하지? 몇 개의 특성을 남겨야 하지? 
# 그 방법이 이것. 
# k-fold로 cross validation을 하면서 evaluation metric으로 비교. 
# evaluation은 k-fold에서의 validation set으로 하는 것. 

# feature 를 20개에서 출발하면 여기서 k-fold, 19개면 또 여기서 k-fold... 

from sklearn.feature_selection import RFECV
# 당연히 coef_, feature_importances_ 가 있는 estimator(지도학습 모델)을 써야 함.
rfecv = RFECV(model, cv=5)
rfecvfit = rfecv.fit(X, y)

#%% 
# [28]
rfecvfit.get_support()

#%% 
# [29]
rfecvfit.transform(X).shape

## 하지만 자동으로 하게 두면 그렇게 잘 generalizae 하진 또 않음... 

#%% 
# [30]

# RFECV의 제일 큰 단점은 coef_, feature_importances_ 가 없는 알고리즘에는 적용이 불가능하다는 것.
# 최근에 이런게 만들어 짐. SequentialFeatureSelector
# coef_, feature_imporatnce_ 없는 sklearn의 임의의 지도학습 모델을 모두 사용 가능함. 

# 어떻게 선택하는가? 

# 내부 원리를 간단히 설명하면, 

# 1. forward면 상수항에서 출발해서
# 2. 기존 방식처럼 cross validation (CV) 쓰며 iterate 하는 것은 같은데
# 3. k-fold 에서 하나 추가하는 것이 제일 성능을 많이 올렸다면 그것을 선택
# 4. 이러면 진짜 걍 cv만 하면서 성능만 보는거니까 모델 dependent 하지 않음. 

# 단점으로, n_features_to_select를 무조건 지정해야 함. 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector

# ============ SequentialFeatureSelector  parameter =================
# n_features_to_select :
# direction : 'forward'    'backward'
# scoring :. None  estimator score     

knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
sfsfit = sfs.fit(X, y)

#%% 
# [31]
sfsfit.get_support()

#%% 
# [32]
sfsfit.transform(X).shape

#%% 
# [33]
# Note: Shell commands like ! pip install Boruta should be run externally; thus, it is commented out.
# ! pip install Boruta

#%% 
# [34]
from boruta import BorutaPy
# random forest를 여러 차례 반복해서 선택하는 것. --> wrapper 방식

# n_estimators : 각 iteration 마다 포함되는 estimator의 수 (RF의 tree 갯수)
# max_iter : 최대 iteration 회수 (강의안에서의 =m) 
# alpha : 최종 가설검정에서 H0: "random하게 hit 했다." 를 기각할 수 있어야 함. 이에 대한 threshold


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=123, max_depth=5)
brtfs = BorutaPy(rf, n_estimators=7, max_iter=15, verbose=1, random_state=123, alpha=0.01)

# As of 2025/03/12, boruta는 예전 numpy에 맞게 코딩이 되어있어서 이것을 이렇게 새로 할당을 해줘야 boruta가 돌아감.
np.int = np.int64
np.float = np.float64
np.bool = np.bool_

brtfs.fit(X, y)

#%% 
# [35]
brtfs.support_

#%% 
# [36]
brtfs.transform(X).shape

#%% 
# [37]

########## ## 2.3 Embedded 방식 ##########
# parametric 방식에선 lasso 같은게 fitting 과정에서 feature를 drop 하니까 이것이 embedded
# tree 기반 모델은 feature importace를 알아서 model fitting 과정에서 사용하니까 이것도 embedded

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
selector = SelectFromModel(estimator=RandomForestClassifier())
selector.fit(X, y)

#%% 
# [38]
selector.get_support()

#%% 
# [39]
selector.transform(X).shape

#%% 
# [40]

# 3. Under/Over Sampling

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], random_state=1)

#%% 
# [41]
pd.Series(y).value_counts()

#%% 
# [42]
plt.figure(figsize=(5, 4))
plt.plot(X[y==0, 0], X[y==0, 1], 'b+', label="class 0")
plt.plot(X[y==1, 0], X[y==1, 1], 'r*', label="class 1")
plt.legend()

#%% 
# [43]

## CNN (Condensed Nearest Neighbour)
# 
# 1NN을 사용해서 majority class를 undersample하는 방식.
# 시간 많이 걸림. 

from imblearn.under_sampling import CondensedNearestNeighbour
undersample1 = CondensedNearestNeighbour(n_neighbors=1)
X1, y1 = undersample1.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(X1[y1==0, 0], X1[y1==0, 1], 'b+', label="class 0")
plt.plot(X1[y1==1, 0], X1[y1==1, 1], 'r*', label="class 1")
plt.legend()

#%% 
# [44]

## Tomek Links

# major이 거의 줄지 않음. 경계선만 깔끔하게 해주는거라. 

from imblearn.under_sampling import TomekLinks
undersample2 = TomekLinks()
X2, y2 = undersample2.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(X2[y2==0, 0], X2[y2==0, 1], 'b+', label="class 0")
plt.plot(X2[y2==1, 0], X2[y2==1, 1], 'r*', label="class 1")
plt.legend()

#%% 
# [45]

# SMOTE (Synthetic Minority Over-sampling Technique)
# option으로 k_neighbors 있음.
# 돌리면 major, minior 갯수 같게 맞춰줌. 

from imblearn.over_sampling import SMOTE
oversample1 = SMOTE()
OX1, Oy1 = oversample1.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(OX1[Oy1==0, 0], OX1[Oy1==0, 1], 'b+', label="class 0")
plt.plot(OX1[Oy1==1, 0], OX1[Oy1==1, 1], 'r*', label="class 1")
plt.legend()

#%%
# [46]

# ADASYN (Adaptive Synthetic Sampling)
# SMOTE의 변형.
# major과 가까운 곳에 minor를 많이 생성. 

from imblearn.over_sampling import ADASYN
oversample2 = ADASYN()

OX2, Oy2 = oversample2.fit_resample(X, y)
plt.figure(figsize=(5, 4))
plt.plot(OX2[Oy2==0, 0], OX2[Oy2==0, 1], 'b+', label="class 0")
plt.plot(OX2[Oy2==1, 0], OX2[Oy2==1, 1], 'r*', label="class 1")
plt.legend()
