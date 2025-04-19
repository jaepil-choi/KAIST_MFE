# %% [markdown]
# Ch3
# 
# April 9, 2025
# 
# 1 Lending Club Data
# 
# • Lending Club 데이터
# – 2007~2017 3분기의 기간 동안의 대출 데이터 (Kaggle 제공, https://www.kaggle.com/wendykan/lending-club-loan-data)
# – 원 자료는 88만개 이상의 사례의 150개 특성변수에 대한 정보를 담고 있음.
# – 10만 건을 random sampling한 자료에 대해 데이터 정제, 특성 선택 및 변환 등의 전처리를 적용한 데이터를 이용하여 분석.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
datasets = pd.read_csv('/content/drive/MyDrive/2025 Spring/빅데이터와 금융자료분석/LoanData.csv')
datasets.info()

# %%
datasetX, datasetY = datasets.drop('charged_off', axis=1), datasets['charged_off']

# %%
X_train, X_test, Y_train, Y_test = train_test_split(datasetX, datasetY, test_size=0.2, random_state=123)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=1/8, random_state=456)

# %%
print(X_train.shape, X_val.shape, X_test.shape)

# %%
X_train.shape

# %%
Y_train.value_counts()

# %% [markdown]
# 2 XGBoost
#
# • xgboost.XGBClassifier(분류)와 xgboost.XGBRegressor(회귀) 클래스
# – Implementation of the scikit-learn API for XGBoost regressor/classifier
#
# XGBoost 파라미터
#
# • General 파라미터
# – booster
#   * ‘gbtree’(default), ‘gblinear’, ‘dart’ 중 하나를 선택.
# – verbosity
#   * 메세지 출력 범위 설정, 0(silent), 1(warning), 2(info), 3(debug)
# – n_jobs
#   * xgboost를 실행하는 데 사용되는 병렬 thread의 수.
#   * default는 모두 사용하는 것。
#
# • Booster 파라미터 (‘gbtree’ 기준)
# – n_estimators
#   * 몇 개의 estimator를 포함하는지를 입력.
#   * default는 100。
#
# – learning_rate
#   * 학습률。 default는 0.3이고 0~1의 사이로 입력。
#   * 과적합을 방지하기 위해 각 estimator의 가중치를 줄여주는 역할을 함。
#   * 작을수록 모델이 견고해지고 과적합 방지에 좋지만、 더 많은 estimators가 요구되며 학습시간은 길어짐。
#
# – min_child_weight
#   * 트리 노드 분할 시 자식노드에 속한 자료들의 weight의 합에 대한 최소값。
#
# – gamma
#   * pruning 관련 하이퍼파라미터
#   * 양의 실수 값으로 설정。 default는 0임。
#   * 클 수록 과적합을 방지하나、 너무 큰 경우 언더피팅이 될 수 있음。
#
# – reg_lambda
#   * L2 규제 하이퍼파라미터
#   * 커질수록 보수적인 모델이 되어 과적합을 방지하나、 너무 큰 경우 언더피팅이 될 수 있음。
#
# – reg_alpha
#   * L1 규제 하이퍼파라미터
#   * 커질수록 보수적인 모델이 되어 과적합을 방지하나、 너무 큰 경우 언더피팅이 될 수 있음。
#   * 특성변수가 sparse하거나 매우 많을 때 적용 권장。
#
# – max_depth
#   * estimator로 사용되는 각 트리의 최대깊이。 default는 6임。
#   * 최대 깊이의 트리는 -1로 설정。
#   * 데이터의 복잡도에 따라 적정한 깊이가 설정되어야 함।
#   * 너무 작으면 언더피팅이 될 수 있고、 너무 크면 오버피팅의 가능성이 높아지며 학습시간이 길어짐。
#
# – subsample
#   * 각 트리 단위로 적용되는 Row sampling 비율로 과적합을 방지하고 학습시간을 줄여줌。
#   * 0~1 사이의 값을 입력。 default는 1임。
#
# – colsample_bytree
#   * 각 트리 단위로 적용되는 Column sampling 비율로 과적합을 방지하고 학습시간을 줄여줌。
#   * 0~1 사이의 값을 입력。 default는 1임。
#
# – tree_method
#   * ‘auto’(default)、‘exact’、‘approx’、‘hist’、‘gpu_hist’ 중 하나를 선택。
#   * ‘auto’는 데이터의 크기가 작은 경우에는 ’exact’、 큰 경우에는 ’approx’ 를 적용하는 것임。
#
# – sketch_eps
#   * tree_method가 ’approx’인 경우에만 적용됨。
#   * 버킷의 수는 1/sketch_eps로 결정되며、 default는 0.03임。
#
# – scale_pos_weight
#   * 범주의 비중이 불균형인 이진 분류 문제에서 cost-sensitive training을 하도록 함。
#   * sum(negative instances) / sum(positive instances) 로 입력。
#
# • Learning Task 파라미터
#
# – objective
#   * 손실함수 지정。 이를 최소로 하도록 학습함。
#   * 자주 활용되는 손실함수
#     · reg:squarederror : 회귀용。 오차제곱 손실。
#     · reg:squaredlogerror : 회귀용。 오차로그제곱 손실。
#     · binary:logistic : 이진 분류용。 예측 확률 반환。
#     · multi:softmax : 다중 분류용。 예측 클래스 반환。
#     · multi:softprob : softmax와 동일한데、 예측 확률 반환。
#
# – base_score
#   * 초기편향치로、 defaul t는 0.5임。
#
# – eval_metric
#   * 평가지표 지정。 각 스텝마다 완성된 모델을 이 지표를 통해 평가함。
#   * 자주 활용되는 평가지표
#     · rmse : root mean square error (회귀의 default)
#     · mae : mean absolute error
#     · logloss : negative log-likelihood
#     · mlogloss : multiclass logloss
#     · error : binary classification error rate 로 임계치는 0.5 기준 (분류의 default) (error@t : 임계치 t 를 적용한 error rate)
#     · merror : multiclass classification error rate
#     · auc : area under the curve
#
# – early_stopping_rounds
#   * eval_metric의 지표가 early_stopping_rounds 횟수 동안 개선되지 않으면、 n_estimators 에 도달하기 전에 멈추도록 함。
#   * 양의 정수값으로 입력。 default는 0。
#   * fit() 메서드에서 검증용 데이터를 eval_set 에 지정해 주어야 함。
#   * 훈련이 끝난 인스턴스의 best_ntree_limit 속성을 이용하여 최적의 tree 갯수를 확인할 수 있으며、 predict() 메서드로 예측 시에는 ntree_limit=best_ntree_limit 가 적용됨。
#
# XGBoost 모델 학습과 예측
#
# • 주요 메서드
# – fit(X_train, y_train) 메서드로 모델을 학습
#   * eval_set : [(X_eval, y_eval)] 형식으로 검증데이터 지정할 수 있음。
#
# – predict(X_test) 메서드로 학습된 모델을 이용한 예측
#   * ntree_limit : default는 0(모든 tree를 사용) 인데、 early_stopping 이 적용된 경우 best_ntree_limit 이 적용됨。
#
# – evals_result() 메서드로 검증데이터에 대한 평가 결과를 확인。 단、 fit() 메서드에서 eval_set 에 검증용 데이터가 지정되어 있어야 함。
#
# • 주요 속성
# – feature_importances_ : 각 특성변수 별 특성 중요도

# %%
!pip install xgboost

# %%
from xgboost import XGBClassifier
model = XGBClassifier(booster='gbtree', objective='binary:logistic',
                      learning_rate=0.1,
                      scale_pos_weight=float(Y_train.value_counts()[0]) / Y_train.value_counts()[1],
                      verbosity=1)

# %%
params = {
    'n_estimators': [100, 500],
    'max_depth': [3, 6, 9],
    'min_child_weight': [0, 0.1, 0.3, 0.5],
    'gamma': [0, 0.1, 1, 5],
    'colsample_bytree': [0.6, 0.8],
}

# %%
from sklearn.model_selection import RandomizedSearchCV
grid_xgb = RandomizedSearchCV(model,
                              param_distributions=params,
                              n_iter=25,
                              cv=3,
                              scoring='accuracy',
                              refit=True)
grid_xgb.fit(X_train, Y_train)

# %%
print(grid_xgb.best_params_)
# %%
print(grid_xgb.best_score_)

# %%
gridresult = pd.DataFrame(grid_xgb.cv_results_).iloc[:, [4,5,6,7,8,13,14,15]]
gridresult.sort_values(['rank_test_score'])[:10]

# %%
finalmodel = XGBClassifier(booster='gbtree', objective='binary:logistic',
                           verbosity=0,
                           colsample_bytree=0.6, gamma=0.1, max_depth=9,
                           min_child_weight=0.1,
                           n_estimators=10000, learning_rate=0.01,
                           scale_pos_weight=float(Y_train.value_counts()[0]) / Y_train.value_counts()[1],
                           eval_metric='logloss',
                           early_stopping_rounds=1000)
finalmodel.fit(X_train, Y_train,
               eval_set=[(X_train, Y_train), (X_val, Y_val)])

# %%
result = finalmodel.evals_result()
print(result.keys())

# %%
print(result['validation_0'].keys())

# %%
print(result['validation_0']['logloss'][:10])

# %%
plt.plot(result['validation_0']['logloss'], label='training_logloss')
plt.plot(result['validation_1']['logloss'], label='validation_logloss')
plt.xlabel('Number of Trees')
plt.ylabel('log loss')
plt.legend()

# %%
imp_values = pd.Series(finalmodel.feature_importances_, index=X_train.columns)
imp_values = imp_values.sort_values(ascending=False)
plt.figure(figsize=(20, 10))
sns.barplot(x=imp_values, y=imp_values.index)

# %%
print(finalmodel.predict(X_train.iloc[0:1, :]))

# %%
print(finalmodel.predict_proba(X_train.iloc[0:1, :]))

# %%
Y_pred = finalmodel.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(Y_test, Y_pred))

# %%
print(accuracy_score(Y_test, Y_pred))

# %%
!pip install shap

# %%
import shap
idx = 12
print(X_train.iloc[idx, :])

# %%
print(Y_train.iloc[idx])

# %%
print(finalmodel.predict(X_train.iloc[idx:(idx+1), :]))

# %%
print(finalmodel.predict_proba(X_train.iloc[idx:(idx+1), :]))

# %%
explainer = shap.TreeExplainer(finalmodel)
shap_values = explainer.shap_values(X_train)

# %%
print(shap_values)

# %%
print(explainer.expected_value)

# %%
shap.force_plot(explainer.expected_value, shap_values[idx, :], X_train.iloc[idx, :], matplotlib=True)

# %%
shap.summary_plot(shap_values, X_train)
