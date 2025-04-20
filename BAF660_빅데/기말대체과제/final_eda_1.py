# %% [markdown]
# # BAF660 빅데이터와 금융자료분석 - 기말대체과제 EDA
#
# 이 노트북에서는 포르투갈 은행의 정기예금 프로모션 전화 데이터(`prob1_bank.csv`)에 대한 
# 탐색적 데이터 분석(EDA)을 수행합니다.

# %% [markdown]
# ## 1. 데이터 로드 및 기본 정보 확인

# %%
# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import os # 파일 경로 처리를 위해 os 모듈 추가

# 시각화 설정
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

# 플롯 저장 디렉토리 생성
plot_dir_eda1 = 'plots/eda1'
os.makedirs(plot_dir_eda1, exist_ok=True)

# 데이터 로드
bank_data = pd.read_csv('prob1_bank.csv')

# 데이터 기본 정보 확인
print("데이터셋 형태:", bank_data.shape)
print("\n데이터셋 정보:")
print(bank_data.info())
print("\n데이터셋 통계 요약:")
print(bank_data.describe())

# %%
# 상위 5개 행 확인
print("\n데이터셋 상위 5개 행:")
print(bank_data.head())

# %% [markdown]
# ## 2. 결측치 분석
#
# Ch1_(1).py에서 배운 결측치 처리 방법을 활용하여 분석합니다.

# %%
# 결측치 확인
missing_values = bank_data.isnull().sum()
missing_percent = bank_data.isnull().mean() * 100

missing_df = pd.DataFrame({
    '결측치 수': missing_values,
    '결측치 비율(%)': missing_percent
})

print("결측치 분석:")
print(missing_df[missing_df['결측치 수'] > 0])

if missing_df['결측치 수'].sum() == 0:
    print("결측치가 없습니다.")

# %% [markdown]
# ## 3. 데이터 분포 탐색

# %% [markdown]
# ### 3.1 범주형 변수 분석

# %%
# 범주형 변수와 수치형 변수 분리
categorical_cols = bank_data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('y')  # 'y'는 타겟 변수로 제외

numerical_cols = bank_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("범주형 변수:", categorical_cols)
print("수치형 변수:", numerical_cols)

# %%
# 범주형 변수 분포 시각화
fig, axes = plt.subplots(len(categorical_cols), 2, figsize=(15, len(categorical_cols) * 5))

for i, col in enumerate(categorical_cols):
    # 변수 분포
    sns.countplot(y=col, data=bank_data, order=bank_data[col].value_counts().index, ax=axes[i, 0])
    axes[i, 0].set_title(f'{col} Distribution')
    axes[i, 0].set_xlabel('Frequency')
    axes[i, 0].set_ylabel(col)
    
    # 각 범주별 타겟 변수 분포
    cross_tab = pd.crosstab(bank_data[col], bank_data['y'], normalize='index') * 100
    cross_tab.plot(kind='bar', stacked=False, ax=axes[i, 1])
    axes[i, 1].set_title(f'{col} vs Target Variable Distribution')
    axes[i, 1].set_ylabel('Ratio (%)')
    axes[i, 1].set_xlabel(col)
    axes[i, 1].tick_params(axis='x', rotation=45)
    axes[i, 1].legend(title='Target Variable')
    
plt.tight_layout()
plot_filename = os.path.join(plot_dir_eda1, 'categorical_distributions_vs_target.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

# %% [markdown]
# ### 3.1.1 범주형 변수 변환
#
# 이전 논의에 따라 다음과 같은 변환 전략을 적용합니다:
# - **job, marital, contact**: 순서 없는 범주형 변수 -> 원-핫 인코딩
# - **education**: 순서 있는 범주형 변수 -> 순서형 인코딩 (primary=0, secondary=1, tertiary=2), 'unknown'은 NaN으로 처리
# - **default, housing, loan, y**: 이진 변수 -> 0/1 인코딩
# - **month**: 순환형 변수 -> Sin/Cos 변환

# %%
# 변환을 위한 데이터프레임 복사
bank_transformed = bank_data.copy()

# education 변환 (순서형 + unknown 처리)
education_map = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': np.nan}
bank_transformed['education_encoded'] = bank_transformed['education'].map(education_map)
print("\nEducation 변환 후 (unknown은 NaN):")
print(bank_transformed[['education', 'education_encoded']].head(10))
print(bank_transformed['education_encoded'].isnull().sum(), "개의 unknown 값 (NaN으로 변환)")

# 이진 변수 변환 (default, housing, loan, y)
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
    map_dict = {'no': 0, 'yes': 1}
    new_col_name = f'{col}_encoded'
    bank_transformed[new_col_name] = bank_transformed[col].map(map_dict)
    print(f"\n{col} 변환 후:")
    print(bank_transformed[[col, new_col_name]].head())

# month 변환 (Sin/Cos)
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
bank_transformed['month_num'] = bank_transformed['month'].map(month_map)
bank_transformed['month_sin'] = np.sin(2 * np.pi * bank_transformed['month_num'] / 12)
bank_transformed['month_cos'] = np.cos(2 * np.pi * bank_transformed['month_num'] / 12)
print("\nMonth 변환 후 (Sin/Cos):")
print(bank_transformed[['month', 'month_num', 'month_sin', 'month_cos']].head())

# 원-핫 인코딩 변수 (job, marital, contact)
# EDA 단계에서는 get_dummies를 사용하여 간단히 확인
one_hot_cols = ['job', 'marital', 'contact']
bank_transformed = pd.get_dummies(bank_transformed, columns=one_hot_cols, drop_first=True, prefix=one_hot_cols)
print("\n원-핫 인코딩 적용 후 컬럼 일부:")
print(bank_transformed.filter(regex='job_|marital_|contact_').head())

# 최종 변환된 데이터 확인
print("\n최종 변환된 데이터프레임 정보:")
print(bank_transformed.info())
print("\n최종 변환된 데이터프레임 상위 5개 행:")
print(bank_transformed.head())

# %% [markdown]
# ### 3.2 수치형 변수 분석

# %%
# 수치형 변수 기술 통계량 계산
numerical_stats = bank_data[numerical_cols].describe().T
numerical_stats['분산'] = bank_data[numerical_cols].var()
numerical_stats['왜도'] = bank_data[numerical_cols].skew()
numerical_stats['첨도'] = bank_data[numerical_cols].kurtosis()

print("수치형 변수 기술 통계량:")
print(numerical_stats)

# %%
# 수치형 변수 분포 시각화 (개별 저장)
for i, col in enumerate(numerical_cols):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 히스토그램 및 KDE 플롯
    sns.histplot(bank_data[col], kde=True, ax=axes[0])
    axes[0].axvline(bank_data[col].mean(), color='r', linestyle='--', label='Mean')
    axes[0].axvline(bank_data[col].median(), color='g', linestyle='-.', label='Median')
    axes[0].set_title(f'{col} Distribution')
    axes[0].legend()
    
    # 박스플롯
    sns.boxplot(x=bank_data[col], ax=axes[1])
    axes[1].set_title(f'{col} Boxplot')
    
    plt.tight_layout()
    plot_filename = os.path.join(plot_dir_eda1, f'numerical_distribution_{col}.png')
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 4. 이상치 탐지
#
# Ch1_(1).py에서 배운 이상치 탐지 방법을 활용하여 수치형 변수의 이상치를 분석합니다.

# %% [markdown]
# ### 4.1 IQR 방법을 이용한 이상치 탐지

# %%
# IQR(사분위 범위) 방법으로 이상치 탐지 함수 정의 (재사용)
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound

# 각 수치형 변수에 대해 이상치 탐지 및 시각화 (개별 저장)
outlier_stats = {}
for i, col in enumerate(numerical_cols):
    fig = plt.figure(figsize=(15, 5))
    outliers, lower_bound, upper_bound = detect_outliers_iqr(bank_data, col)
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(bank_data)) * 100
    
    outlier_stats[col] = {
        'outlier_count': outlier_count,
        'outlier_percent': outlier_percent,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    plt.scatter(bank_data.index, bank_data[col], alpha=0.5)
    plt.axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
    plt.axhline(y=upper_bound, color='r', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
    plt.title(f'{col} Outliers (IQR Method) - {outlier_count} outliers ({outlier_percent:.2f}%)')
    plt.ylabel(col)
    plt.legend()
    
    plot_filename = os.path.join(plot_dir_eda1, f'outlier_iqr_{col}.png')
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

# 이상치 통계 출력
outlier_summary = pd.DataFrame.from_dict(outlier_stats, orient='index')
print("IQR 방법 이상치 요약:")
print(outlier_summary)

# %% [markdown]
# ### 4.2 Z-점수 방법을 이용한 이상치 탐지

# %%
# Z-점수 방법으로 이상치 탐지 함수 정의 (재사용)
def detect_outliers_zscore(df, col, threshold=3):
    z_scores = stats.zscore(df[col])
    outliers = df[abs(z_scores) > threshold]
    return outliers, z_scores

# 각 수치형 변수에 대해 Z-점수 이상치 탐지 및 시각화 (개별 저장)
zscore_outlier_stats = {}
for i, col in enumerate(numerical_cols):
    fig = plt.figure(figsize=(15, 5))
    outliers, z_scores = detect_outliers_zscore(bank_data, col)
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(bank_data)) * 100
    
    zscore_outlier_stats[col] = {
        'outlier_count': outlier_count,
        'outlier_percent': outlier_percent
    }
    
    plt.scatter(bank_data.index, abs(z_scores), alpha=0.5)
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold (3)')
    plt.title(f'{col} Outliers (Z-score Method) - {outlier_count} outliers ({outlier_percent:.2f}%)')
    plt.ylabel('|Z-score|')
    plt.legend()
    
    plot_filename = os.path.join(plot_dir_eda1, f'outlier_zscore_{col}.png')
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

# Z-점수 이상치 통계 출력
zscore_summary = pd.DataFrame.from_dict(zscore_outlier_stats, orient='index')
print("Z-점수 방법 이상치 요약:")
print(zscore_summary)

# %% [markdown]
# ### 4.3 머신러닝 기반 이상치 탐지 (IsolationForest)

# %%
from sklearn.ensemble import IsolationForest

# IsolationForest 이상치 탐지 함수 정의 (재사용)
def detect_outliers_iforest(df, cols, contamination=0.05):
    X = df[cols].copy() # 원본 데이터프레임 변경 방지
    clf = IsolationForest(contamination=contamination, random_state=42)
    y_pred = clf.fit_predict(X)
    return df, y_pred == -1 # 원본 df와 아웃라이어 여부 마스크 반환

# 모든 수치형 변수에 대해 IsolationForest 적용
_, is_outlier = detect_outliers_iforest(bank_data, numerical_cols)

# 이상치 결과 출력
outlier_count = np.sum(is_outlier)
outlier_percent = (outlier_count / len(bank_data)) * 100

print(f"IsolationForest로 탐지한 이상치: {outlier_count}개 ({outlier_percent:.2f}%)")

# 시각화를 위해 PCA로 차원 축소
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(bank_data[numerical_cols]) # 원본 데이터 사용

# 이상치 시각화
fig = plt.figure(figsize=(10, 8))
plt.scatter(X_pca[~is_outlier, 0], X_pca[~is_outlier, 1], c='blue', s=5, label='Normal Data')
plt.scatter(X_pca[is_outlier, 0], X_pca[is_outlier, 1], c='red', s=20, label='Outliers')
plt.title('Outliers Detection by IsolationForest (PCA Dimension Reduction Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plot_filename = os.path.join(plot_dir_eda1, 'outlier_iforest_pca.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

# %% [markdown]
# ### 4.4 LOF(Local Outlier Factor)를 이용한 이상치 탐지

# %%
from sklearn.neighbors import LocalOutlierFactor

# LOF 이상치 탐지 함수 정의 (재사용)
def detect_outliers_lof(df, cols, contamination=0.05):
    X = df[cols].copy() # 원본 데이터프레임 변경 방지
    clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    y_pred = clf.fit_predict(X)
    return df, y_pred == -1 # 원본 df와 아웃라이어 여부 마스크 반환

# 모든 수치형 변수에 대해 LOF 적용
_, lof_is_outlier = detect_outliers_lof(bank_data, numerical_cols)

# 이상치 결과 출력
lof_outlier_count = np.sum(lof_is_outlier)
lof_outlier_percent = (lof_outlier_count / len(bank_data)) * 100

print(f"LOF로 탐지한 이상치: {lof_outlier_count}개 ({lof_outlier_percent:.2f}%)")

# 시각화
fig = plt.figure(figsize=(10, 8))
plt.scatter(X_pca[~lof_is_outlier, 0], X_pca[~lof_is_outlier, 1], c='blue', s=5, label='Normal Data')
plt.scatter(X_pca[lof_is_outlier, 0], X_pca[lof_is_outlier, 1], c='red', s=20, label='Outliers')
plt.title('Outliers Detection by LOF (PCA Dimension Reduction Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plot_filename = os.path.join(plot_dir_eda1, 'outlier_lof_pca.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

# %% [markdown]
# ### 4.5 이상치 처리 방법 비교

# %%
# 이상치 탐지 방법 비교
outlier_comparison = pd.DataFrame({
    'IQR 방법': [outlier_stats[col]['outlier_count'] for col in numerical_cols],
    'Z-점수 방법': [zscore_outlier_stats[col]['outlier_count'] for col in numerical_cols],
    'IsolationForest': [outlier_count] * len(numerical_cols), # 전체 데이터셋 기준
    'LOF': [lof_outlier_count] * len(numerical_cols) # 전체 데이터셋 기준
}, index=numerical_cols)

print("이상치 탐지 방법별 이상치 개수 비교:")
print(outlier_comparison)

# 이상치 처리 방법 시연: balance 변수에 대해 다양한 방법 적용
outlier_col = 'balance'
bank_outlier = bank_data.copy()

# IQR 기반 클리핑 함수 (재사용)
def clip_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

# 1. 이상치 제거
outliers_iqr_bal, _, _ = detect_outliers_iqr(bank_outlier, outlier_col)
bank_outlier_removed = bank_outlier[~bank_outlier.index.isin(outliers_iqr_bal.index)]

# 2. 이상치 대체: 상하한선으로 클리핑
bank_outlier_clipped = bank_outlier.copy()
bank_outlier_clipped[outlier_col] = clip_outliers_iqr(bank_outlier_clipped[outlier_col])

# 3. 이상치 대체: 평균으로 대체
bank_outlier_mean = bank_outlier.copy()
bank_outlier_mean.loc[bank_outlier_mean.index.isin(outliers_iqr_bal.index), outlier_col] = bank_outlier[outlier_col].mean()

# 시각화: 이상치 처리 전후 비교
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(bank_outlier[outlier_col], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Original Data')

sns.histplot(bank_outlier_removed[outlier_col], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Outliers Removed')

sns.histplot(bank_outlier_clipped[outlier_col], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Outliers Clipped')

sns.histplot(bank_outlier_mean[outlier_col], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Outliers Replaced with Mean')

plt.tight_layout()
plot_filename = os.path.join(plot_dir_eda1, f'outlier_handling_{outlier_col}.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

print(f"원본 데이터 크기: {len(bank_outlier)}")
print(f"이상치 제거 후 데이터 크기: {len(bank_outlier_removed)}")
print(f"제거된 이상치 비율: {(1 - len(bank_outlier_removed)/len(bank_outlier))*100:.2f}%")

# %% [markdown]
# ## 5. 특성 변환 (Feature Transformation)
#
# Ch1_(2).py에서 배운 특성 변환 방법을 활용하여 데이터를 변환합니다.

# %% [markdown]
# ### 5.1 수치형 변수 변환

# %%
# 스케일링 방법 비교 (StandardScaler, MinMaxScaler, PowerTransformer)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

# 변환을 적용할 데이터 준비 (이상치 처리된 데이터 사용)
X_numeric = bank_outlier_clipped[numerical_cols].copy() # Use clipped data

# StandardScaler 적용
scaler_standard = StandardScaler()
X_scaled_standard = scaler_standard.fit_transform(X_numeric)
X_scaled_standard_df = pd.DataFrame(X_scaled_standard, columns=numerical_cols)

# MinMaxScaler 적용
scaler_minmax = MinMaxScaler()
X_scaled_minmax = scaler_minmax.fit_transform(X_numeric)
X_scaled_minmax_df = pd.DataFrame(X_scaled_minmax, columns=numerical_cols)

# PowerTransformer 적용 (Yeo-Johnson 변환)
scaler_power = PowerTransformer(method='yeo-johnson')
X_scaled_power = scaler_power.fit_transform(X_numeric)
X_scaled_power_df = pd.DataFrame(X_scaled_power, columns=numerical_cols)

# 변환 결과 시각화 (개별 저장)
for col in numerical_cols:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.histplot(X_numeric[col], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(f'Original Clipped Data: {col}')
    
    sns.histplot(X_scaled_standard_df[col], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title(f'StandardScaler: {col}')
    
    sns.histplot(X_scaled_minmax_df[col], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title(f'MinMaxScaler: {col}')
    
    sns.histplot(X_scaled_power_df[col], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title(f'PowerTransformer: {col}')
    
    plt.tight_layout()
    plot_filename = os.path.join(plot_dir_eda1, f'scaling_comparison_{col}.png')
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

# %% [markdown]
# ### 5.2 범주형 변수 인코딩

# %%
# 범주형 변수 인코딩 방법 비교 (Label Encoding, One-Hot Encoding)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 예시로 사용할 범주형 변수 선택
cat_col_example = 'job'

# LabelEncoder 적용
le = LabelEncoder()
label_encoded = le.fit_transform(bank_data[cat_col_example])
print(f"LabelEncoder 적용 결과 (처음 10개): {label_encoded[:10]}")
print(f"원본 범주: {bank_data[cat_col_example].iloc[:10].values}")
print(f"인코딩 매핑: {dict(zip(le.classes_, range(len(le.classes_))))}")

# OneHotEncoder 적용
ohe = OneHotEncoder(sparse_output=False, drop='first')
onehot_encoded = ohe.fit_transform(bank_data[[cat_col_example]])
onehot_df = pd.DataFrame(
    onehot_encoded, 
    columns=[f"{cat_col_example}_{cat}" for cat in le.classes_[1:]]
)
print("\nOneHotEncoder 적용 결과 (처음 5개 행, 처음 5개 열):")
print(onehot_df.iloc[:5, :5])

# %% [markdown]
# ### 5.3 빈화(Binning) 및 구간화

# %%
# 수치형 변수 binning 예시 - age 변수
bin_col = 'age'
bank_binned = bank_data.copy()

# 동일 너비 구간화
equal_width_bins = pd.cut(bank_data[bin_col], bins=5)
bank_binned['age_equal_width'] = equal_width_bins

# 동일 빈도 구간화
equal_freq_bins = pd.qcut(bank_data[bin_col], q=5)
bank_binned['age_equal_freq'] = equal_freq_bins

# 사용자 정의 구간화
custom_bins = [0, 30, 40, 50, 60, 100]
custom_labels = ['0-30', '31-40', '41-50', '51-60', '60+']
custom_binned = pd.cut(bank_data[bin_col], bins=custom_bins, labels=custom_labels)
bank_binned['age_custom'] = custom_binned

# 결과 비교
binning_comparison = pd.DataFrame({
    '원본': bank_data[bin_col],
    '동일 너비 구간화': bank_binned['age_equal_width'],
    '동일 빈도 구간화': bank_binned['age_equal_freq'],
    '사용자 정의 구간화': bank_binned['age_custom']
})

print("Binning 비교 (처음 10개 행):")
print(binning_comparison.head(10))

# 구간화 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(bank_data[bin_col], kde=True, bins=20, ax=axes[0, 0])
axes[0, 0].set_title('Original Data')

sns.countplot(y='age_equal_width', data=bank_binned, ax=axes[0, 1])
axes[0, 1].set_title('Equal-width Binning')

sns.countplot(y='age_equal_freq', data=bank_binned, ax=axes[1, 0])
axes[1, 0].set_title('Equal-frequency Binning')

sns.countplot(y='age_custom', data=bank_binned, ax=axes[1, 1])
axes[1, 1].set_title('Custom Binning')

plt.tight_layout()
plot_filename = os.path.join(plot_dir_eda1, f'binning_{bin_col}.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

# %% [markdown]
# ## 6. 특성 선택 (Feature Selection)

# %% [markdown]
# ### 6.1 필터 방식 특성 선택

# %%
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# 목표 변수 인코딩
target = bank_data['y'].map({'yes': 1, 'no': 0})

# 수치형 특성에 대한 F-통계량 기반 특성 선택
X_num = StandardScaler().fit_transform(bank_outlier_clipped[numerical_cols]) # 스케일링된 데이터 사용
f_selector = SelectKBest(f_classif, k='all')
f_selector.fit(X_num, target)

# 결과 시각화
feature_scores = pd.DataFrame({
    'Feature': numerical_cols,
    'F-statistic': f_selector.scores_,
    'p-value': f_selector.pvalues_
})
feature_scores = feature_scores.sort_values('F-statistic', ascending=False)

fig = plt.figure(figsize=(12, 6))
sns.barplot(x='F-statistic', y='Feature', data=feature_scores)
plt.title('F-statistic Based Importance of Numerical Features')
plot_filename = os.path.join(plot_dir_eda1, 'feature_importance_f_statistic.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

print("F-통계량 기반 수치형 특성 중요도:")
print(feature_scores)

# 범주형 특성에 대한 카이제곱 기반 특성 선택
# OneHotEncoder를 사용하여 범주형 변수 인코딩
X_cat_data = bank_data[categorical_cols[:-1]]  # 'y' 제외
ohe = OneHotEncoder(sparse_output=False)
X_cat = ohe.fit_transform(X_cat_data)

# 특성명 생성
feature_names = []
for i, col in enumerate(categorical_cols[:-1]):
    categories = ohe.categories_[i]
    feature_names.extend([f"{col}_{cat}" for cat in categories])

# 카이제곱 통계량 계산
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X_cat, target)

# 결과 시각화
chi2_scores = pd.DataFrame({
    'Feature': feature_names,
    'Chi-square Statistic': chi2_selector.scores_,
    'p-value': chi2_selector.pvalues_
})
chi2_scores = chi2_scores.sort_values('Chi-square Statistic', ascending=False)

fig = plt.figure(figsize=(12, 10))
sns.barplot(x='Chi-square Statistic', y='Feature', data=chi2_scores.head(20))
plt.title('Chi-square Statistic Based Importance of Categorical Features (Top 20)')
plot_filename = os.path.join(plot_dir_eda1, 'feature_importance_chi2.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

print("카이제곱 통계량 기반 범주형 특성 중요도 (상위 10개):")
print(chi2_scores.head(10))

# 상호정보량 기반 특성 선택
# 모든 특성에 대한 상호정보량 계산
X_combined = np.hstack([X_num, X_cat])
all_feature_names = numerical_cols + feature_names

mi_selector = SelectKBest(mutual_info_classif, k='all')
mi_selector.fit(X_combined, target)

mi_scores = pd.DataFrame({
    'Feature': all_feature_names,
    'Mutual Information': mi_selector.scores_
})
mi_scores = mi_scores.sort_values('Mutual Information', ascending=False)

fig = plt.figure(figsize=(12, 8))
sns.barplot(x='Mutual Information', y='Feature', data=mi_scores.head(20))
plt.title('Mutual Information Based Importance of Features (Top 20)')
plot_filename = os.path.join(plot_dir_eda1, 'feature_importance_mutual_info.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

print("상호정보량 기반 특성 중요도 (상위 10개):")
print(mi_scores.head(10))

# %% [markdown]
# ### 6.2 임베디드 방식 특성 선택

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 모든 특성에 대한 랜덤 포레스트 기반 특성 중요도
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold='median')
selector.fit(X_combined, target)

# 특성 중요도 계산
importances = selector.estimator_.feature_importances_
rf_scores = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
})
rf_scores = rf_scores.sort_values('Importance', ascending=False)

fig = plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_scores.head(20))
plt.title('Random Forest-based Feature Importance (Top 20)')
plot_filename = os.path.join(plot_dir_eda1, 'feature_importance_rf.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

print("랜덤 포레스트 기반 특성 중요도 (상위 10개):")
print(rf_scores.head(10))

# 선택된 특성
selected_features = np.array(all_feature_names)[selector.get_support()]
print(f"\n선택된 특성 ({len(selected_features)}개):")
print(selected_features)

# %% [markdown]
# ## 7. 클래스 불균형 분석

# %%
# 목표 변수의 클래스 분포 확인
target_counts = bank_data['y'].value_counts()
target_percent = bank_data['y'].value_counts(normalize=True) * 100

print("목표 변수 분포:")
print(target_counts)
print(f"\n비율: {target_percent[0]:.2f}% vs {target_percent[1]:.2f}%")

# 시각화
fig = plt.figure(figsize=(10, 6))
sns.countplot(x='y', data=bank_data)
plt.title('Target Variable Distribution')
plt.xlabel('Term Deposit Subscription')
plt.ylabel('Frequency')
for i, count in enumerate(target_counts):
    plt.text(i, count + 100, f"{count} ({target_percent[i]:.1f}%)", ha='center')
plot_filename = os.path.join(plot_dir_eda1, 'target_distribution.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

# 클래스 불균형 처리 방법 시연
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN

# 특성과 목표 변수 준비
X = X_combined # 이미 인코딩 및 스케일링된 데이터 사용
y = target.values

# 1. 랜덤 언더샘플링
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

# 2. Tomek Links
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)

# 3. SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# 4. ADASYN
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

# 결과 비교
sampling_results = pd.DataFrame({
    '방법': ['원본 데이터', '랜덤 언더샘플링', 'Tomek Links', 'SMOTE', 'ADASYN'],
    '표본 크기': [len(y), len(y_rus), len(y_tl), len(y_smote), len(y_adasyn)],
    '클래스 0 (미가입) 비율': [
        (y == 0).sum() / len(y) * 100,
        (y_rus == 0).sum() / len(y_rus) * 100,
        (y_tl == 0).sum() / len(y_tl) * 100,
        (y_smote == 0).sum() / len(y_smote) * 100,
        (y_adasyn == 0).sum() / len(y_adasyn) * 100
    ],
    '클래스 1 (가입) 비율': [
        (y == 1).sum() / len(y) * 100,
        (y_rus == 1).sum() / len(y_rus) * 100,
        (y_tl == 1).sum() / len(y_tl) * 100,
        (y_smote == 1).sum() / len(y_smote) * 100,
        (y_adasyn == 1).sum() / len(y_adasyn) * 100
    ]
})

print("클래스 불균형 처리 방법 비교:")
print(sampling_results)

# 시각화
fig = plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(sampling_results['방법']))

plt.bar(index, sampling_results['클래스 0 (미가입) 비율'], bar_width, label='No Subscription (Class 0)')
plt.bar(index + bar_width, sampling_results['클래스 1 (가입) 비율'], bar_width, 
        label='Subscription (Class 1)')

plt.xlabel('Sampling Method')
plt.ylabel('Ratio (%)')
plt.title('Class Imbalance Handling Method Comparison')
plt.xticks(index, sampling_results['방법'], rotation=45)
plt.legend()
plt.tight_layout()
plot_filename = os.path.join(plot_dir_eda1, 'sampling_comparison.png')
plt.savefig(plot_filename)
plt.show()
plt.close(fig)

# %% [markdown]
# ## 8. 변수 간 상관관계 분석

# %%
# 수치형 변수 간 상관관계 (피어슨 상관계수)
corr_matrix = bank_data[numerical_cols].corr()

fig_pearson = plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Coefficient Between Numerical Variables')
plot_filename_pearson = os.path.join(plot_dir_eda1, 'correlation_pearson.png')
plt.savefig(plot_filename_pearson)
plt.show()
plt.close(fig_pearson)

# 스피어만 상관계수 (비선형 관계 탐지에 유용)
spearman_corr = bank_data[numerical_cols].corr(method='spearman')

fig_spearman = plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Spearman Correlation Coefficient Between Numerical Variables')
plot_filename_spearman = os.path.join(plot_dir_eda1, 'correlation_spearman.png')
plt.savefig(plot_filename_spearman)
plt.show()
plt.close(fig_spearman)

# 범주형 변수와 목표 변수 간 관계 분석 (크래머 V 계수)
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return 0.0 # Cannot compute for single row/column
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) <= 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# 각 범주형 변수와 목표 변수 간의 크래머 V 계수 계산
cramers_results = {}
for col in categorical_cols[:-1]:
    cramers_results[col] = cramers_v(bank_data[col], bank_data['y'])

cramers_df = pd.DataFrame({
    'Variable': list(cramers_results.keys()),
    'Cramer V Coefficient': list(cramers_results.values())
}).sort_values('Cramer V Coefficient', ascending=False)

fig_cramer = plt.figure(figsize=(12, 6))
sns.barplot(x='Cramer V Coefficient', y='Variable', data=cramers_df)
plt.title('Cramer V Coefficient Between Categorical Variables and Target Variable')
plot_filename_cramer = os.path.join(plot_dir_eda1, 'correlation_cramer_v.png')
plt.savefig(plot_filename_cramer)
plt.show()
plt.close(fig_cramer)

print("범주형 변수와 목표 변수 간 크래머 V 계수:")
print(cramers_df)

# %% [markdown]
# ## 9. 차원 축소 및 군집화 분석
#
# Ch2.py에서 배운 차원 축소 및 군집화 방법을 활용하여 데이터를 분석합니다.

# %% [markdown]
# ### 9.1 PCA (주성분 분석)

# %%
from sklearn.decomposition import PCA

# 수치형 데이터에 대해 PCA 수행
pca = PCA()
pca.fit(X_num)

# 설명된 분산 비율
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 결과 시각화 (설명된 분산)
fig_pca_var = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Each Principal Component')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio')
plt.legend()

plt.tight_layout()
plot_filename_pca_var = os.path.join(plot_dir_eda1, 'pca_explained_variance.png')
plt.savefig(plot_filename_pca_var)
plt.show()
plt.close(fig_pca_var)

# 결과 시각화 (주성분 기여도 및 2D 산점도)
fig_pca_scatter = plt.figure(figsize=(14, 8))
components = pd.DataFrame(pca.components_, columns=numerical_cols)

plt.subplot(1, 2, 1)
sns.heatmap(components.iloc[:2], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Contribution for First 2 Principal Components')

X_pca_all = pca.transform(X_num)
pca_df = pd.DataFrame(X_pca_all[:, :2], columns=['PC1', 'PC2'], index=bank_data.index)
pca_df['y'] = bank_data['y'].values

plt.subplot(1, 2, 2)
sns.scatterplot(x='PC1', y='PC2', hue='y', data=pca_df, alpha=0.6)
plt.title('PCA Result Visualization (First 2 Principal Components)')

plt.tight_layout()
plot_filename_pca_scatter = os.path.join(plot_dir_eda1, 'pca_2d_scatter.png')
plt.savefig(plot_filename_pca_scatter)
plt.show()
plt.close(fig_pca_scatter)

# %% [markdown]
# ### 9.2 t-SNE (T-distributed Stochastic Neighbor Embedding)

# %%
from sklearn.manifold import TSNE

# t-SNE 수행
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_combined) 

tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
tsne_df['y'] = bank_data['y'].values

# 결과 시각화
fig_tsne = plt.figure(figsize=(12, 10))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='y', data=tsne_df, alpha=0.6)
plt.title('t-SNE Visualization')
plt.legend(title='Term Deposit Subscription')
plot_filename_tsne = os.path.join(plot_dir_eda1, 'tsne_2d_scatter.png')
plt.savefig(plot_filename_tsne)
plt.show()
plt.close(fig_tsne)

# %% [markdown]
# ### 9.3 K-Means 군집화

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 최적의 군집 수 결정 (엘보우 방법 및 실루엣 점수)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_combined)
    inertia.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_combined, labels))

# 엘보우 곡선 및 실루엣 점수 시각화
fig_kmeans_k, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_range, inertia, 'bo-')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

axes[1].plot(k_range, silhouette_scores, 'ro-')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score Method')

plt.tight_layout()
plot_filename_kmeans_k = os.path.join(plot_dir_eda1, 'kmeans_optimal_k.png')
plt.savefig(plot_filename_kmeans_k)
plt.show()
plt.close(fig_kmeans_k)

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"최적의 군집 수 (k): {optimal_k} (실루엣 점수: {max(silhouette_scores):.3f})")

kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_optimal.fit_predict(X_combined)

pca_df['cluster'] = kmeans_labels

# 결과 시각화 (PCA)
fig_kmeans_pca = plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='viridis', 
                data=pca_df, alpha=0.6, s=50)
plt.title(f'K-Means Clustering Result (k={optimal_k}) - PCA Visualization')
plt.legend(title='Cluster')
plot_filename_kmeans_pca = os.path.join(plot_dir_eda1, f'kmeans_cluster_pca_k{optimal_k}.png')
plt.savefig(plot_filename_kmeans_pca)
plt.show()
plt.close(fig_kmeans_pca)

# 군집별 목표 변수 분포 확인
cluster_target_dist = pd.crosstab(kmeans_labels, bank_data['y'], normalize='index') * 100

print("군집별 목표 변수 분포 (%):")
print(cluster_target_dist)

# 시각화 (Bar Chart)
fig_kmeans_target, ax = plt.subplots(figsize=(12, 6))
cluster_target_dist.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Cluster')
ax.set_ylabel('Ratio (%)')
ax.set_title('Cluster-wise Target Variable Distribution')
ax.legend(title='Term Deposit Subscription')
ax.tick_params(axis='x', rotation=0)
plot_filename_kmeans_target = os.path.join(plot_dir_eda1, 'kmeans_cluster_target_dist.png')
plt.savefig(plot_filename_kmeans_target)
plt.show()
plt.close(fig_kmeans_target)

# %% [markdown]
# ### 9.4 DBSCAN 군집화

# %%
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# k-거리 그래프를 통한 eps 파라미터 결정
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_num) 
distances, indices = neighbors_fit.kneighbors(X_num)
distances = np.sort(distances[:, 4])

fig_dbscan_kdist = plt.figure(figsize=(12, 6))
plt.plot(distances)
plt.axhline(y=1.0, color='r', linestyle='--')
plt.title('k-distance Graph (k=5)')
plt.xlabel('Data Point (Sorted)')
plt.ylabel('Distance to 5th Neighbor')
plot_filename_dbscan_kdist = os.path.join(plot_dir_eda1, 'dbscan_k_distance.png')
plt.savefig(plot_filename_dbscan_kdist)
plt.show()
plt.close(fig_dbscan_kdist)

eps_dbscan = 1.0
min_samples_dbscan = 5
dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
dbscan_labels = dbscan.fit_predict(X_num)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN 결과 (eps={eps_dbscan}, min_samples={min_samples_dbscan}): {n_clusters}개 군집, {n_noise}개 노이즈 포인트 ({n_noise/len(dbscan_labels)*100:.2f}%)")

pca_df['dbscan_cluster'] = dbscan_labels

# 결과 시각화 (PCA)
fig_dbscan_pca = plt.figure(figsize=(12, 10))
colors = np.array(['gray' if label == -1 else plt.cm.viridis((label % max(1, n_clusters)) / max(1, n_clusters)) for label in dbscan_labels])
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=colors, alpha=0.6, s=50)
plt.title('DBSCAN Clustering Result - PCA Visualization')
plt.xlabel('PC1')
plt.ylabel('PC2')

noise_handle = plt.scatter([], [], c='gray', label='Noise')
cluster_handles = [plt.scatter([], [], c=plt.cm.viridis(i / max(1, n_clusters)), label=f'Cluster {i}') for i in range(n_clusters)]
plt.legend(handles=[noise_handle] + cluster_handles)
plot_filename_dbscan_pca = os.path.join(plot_dir_eda1, 'dbscan_cluster_pca.png')
plt.savefig(plot_filename_dbscan_pca)
plt.show()
plt.close(fig_dbscan_pca)

# DBSCAN 군집별 목표 변수 분포 (노이즈 포인트 제외)
dbscan_df = pd.DataFrame({'cluster': dbscan_labels, 'y': bank_data['y']})
dbscan_df = dbscan_df[dbscan_df['cluster'] != -1]

if len(dbscan_df) > 0 and n_clusters > 0:
    dbscan_target_dist = pd.crosstab(dbscan_df['cluster'], dbscan_df['y'], normalize='index') * 100
    print("\nDBSCAN 군집별 목표 변수 분포 (노이즈 포인트 제외) (%):")
    print(dbscan_target_dist)

    fig_dbscan_target, ax = plt.subplots(figsize=(12, 6))
    dbscan_target_dist.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('DBSCAN Cluster')
    ax.set_ylabel('Ratio (%)')
    ax.set_title('DBSCAN Cluster-wise Target Variable Distribution (Excluding Noise Points)')
    ax.legend(title='Term Deposit Subscription')
    ax.tick_params(axis='x', rotation=0)
    plot_filename_dbscan_target = os.path.join(plot_dir_eda1, 'dbscan_cluster_target_dist.png')
    plt.savefig(plot_filename_dbscan_target)
    plt.show()
    plt.close(fig_dbscan_target)
else:
    print("\nDBSCAN에서 노이즈가 아닌 포인트가 없거나 군집이 생성되지 않았습니다.")

# %% [markdown]
# ## 10. 결론 및 인사이트
