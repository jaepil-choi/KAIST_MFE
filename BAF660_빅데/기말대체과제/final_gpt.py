# %% [markdown]
# # 빅데이터 기말대체과제

# %% [markdown]
# ## 1. `prob1_bank.csv`

# %% [markdown]
# ### 0. 라이브러리 임포트 및 데이터 로드

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# Create plots directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the data
bank_data = pd.read_csv('prob1_bank.csv')

# Display basic information
print("Bank Dataset Shape:", bank_data.shape)
print("\nBank Dataset Info:")
print(bank_data.info())
# print("\nBank Dataset Summary Statistics:") # EDA에서 상세히 다룸
# print(bank_data.describe())

# Check for missing values (EDA에서 확인 완료)
# print("\nMissing values in each column:")
# print(bank_data.isnull().sum())

# Check target variable distribution (class imbalance) - EDA에서 확인 완료 및 시각화
print("\nTarget variable distribution (Initial Check):")
print(bank_data['y'].value_counts())
# print(bank_data['y'].value_counts(normalize=True) * 100)

# %% [markdown]
# ### 1. 범주형 변수 전처리
#
# - EDA 결과, 범주형 변수 간 특별한 순서나 계층 구조가 보이지 않으므로 One-Hot Encoding을 적용합니다.
# - 첫 번째 범주를 제거(`drop='first'`)하여 다중공선성을 방지합니다.

# %%
# Identify categorical and numerical columns
categorical_cols = bank_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = bank_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical variables:", categorical_cols)
print("Numerical variables:", numerical_cols)

# Preprocessing categorical variables using OneHotEncoder
# 목표 변수 'y'는 나중에 LabelEncoding으로 처리하므로 제외
categorical_features_to_encode = [col for col in categorical_cols if col != 'y']
categorical_preprocessor = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Apply one-hot encoding
X_categorical = bank_data[categorical_features_to_encode]
categorical_preprocessor.fit(X_categorical)
X_categorical_encoded = categorical_preprocessor.transform(X_categorical)

# Create DataFrame with encoded features
encoded_feature_names = categorical_preprocessor.get_feature_names_out(categorical_features_to_encode)
X_categorical_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names, index=bank_data.index)

print("\nShape of encoded categorical features:", X_categorical_df.shape)
# print(X_categorical_df.head())

# %% [markdown]
# ### 2. 수치형 변수 전처리
#
# - EDA 결과, `balance` 변수 등에서 상당한 왜도와 이상치가 관찰되었습니다.
# - 모델 성능에 이상치의 영향을 줄이기 위해 IQR 방법을 사용하여 이상치를 탐지하고 클리핑(Clipping)합니다.
# - 클리핑 후, `StandardScaler`를 사용하여 모든 수치형 변수를 표준화합니다.

# %%
# 이상치 처리 함수 (IQR 기반 Clipping)
def clip_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

# 수치형 변수에 이상치 처리 적용
X_numerical = bank_data[numerical_cols].copy()
for col in numerical_cols:
    original_skew = X_numerical[col].skew()
    X_numerical[col] = clip_outliers_iqr(X_numerical[col])
    clipped_skew = X_numerical[col].skew()
    print(f"Variable '{col}': Skewness before clipping: {original_skew:.2f}, Skewness after clipping: {clipped_skew:.2f}")

# Apply standardization to numerical features (after clipping)
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Create DataFrame with scaled features
X_numerical_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=bank_data.index)

print("\nShape of scaled numerical features:", X_numerical_df.shape)
# print(X_numerical_df.head())

# %% [markdown]
# ### 3. 클래스 불균형 처리
#
# - EDA 결과, 목표 변수 'y'가 약 88%의 'no'와 12%의 'yes'로 심각한 불균형을 보입니다.
# - 소수 클래스(yes)의 예측 성능 저하를 방지하기 위해 SMOTE(Synthetic Minority Over-sampling Technique)를 적용하여 오버샘플링합니다.

# %%
# Combine preprocessed features
X_preprocessed = pd.concat([X_numerical_df, X_categorical_df], axis=1)

# Convert target variable to binary using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(bank_data['y'])
print(f"\nTarget variable mapping: {label_encoder.classes_[0]} -> 0, {label_encoder.classes_[1]} -> 1")

# Check class imbalance before SMOTE
print("\nOriginal class distribution:")
print(Counter(y))

# Apply SMOTE to handle class imbalance
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

print("\nResampled class distribution:")
print(Counter(y_resampled))

# Visualize class distribution before and after SMOTE (optional, already done in EDA)
# plt.figure(figsize=(12, 5))
# ... (plotting code from final_eda.py or final_gpt.py) ...
# plt.close()

print("\nFinal preprocessed dataset shape (after SMOTE):", X_resampled.shape)
print("Preprocessing for prob1_bank.csv completed.")

# %% [markdown]
# ## 2. `prob2_card.csv`

# %% [markdown]
# ### 0. 라이브러리 임포트 및 데이터 로드

# %%
# Import necessary libraries (some may be re-imported for clarity)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load the data
card_data = pd.read_csv('prob2_card.csv')

# Display basic information
print("\n--- Credit Card Dataset Analysis ---")
print("Credit Card Dataset Shape:", card_data.shape)
print("\nCredit Card Dataset Info:")
print(card_data.info())
# print("\nCredit Card Dataset Summary Statistics:") # EDA에서 상세히 다룸
# print(card_data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(card_data.isnull().sum())
# EDA 결과 또는 간단한 확인 결과, 결측치가 없음을 확인.

# Set CUST_ID as index and exclude it from analysis
if 'CUST_ID' in card_data.columns:
    card_data.set_index('CUST_ID', inplace=True)

# %% [markdown]
# ### 1. 데이터 전처리 (스케일링)
#
# - 클러스터링 알고리즘은 거리에 기반하므로, 변수들의 스케일을 통일하는 것이 중요합니다.
# - StandardScaler를 사용하여 모든 특성을 표준화합니다.

# %%
# Preprocess data: scaling features
print("\nScaling credit card features using StandardScaler...")
scaler_card = StandardScaler()
card_data_scaled = scaler_card.fit_transform(card_data)
card_data_scaled_df = pd.DataFrame(card_data_scaled, columns=card_data.columns, index=card_data.index)
# print(card_data_scaled_df.head())

# %% [markdown]
# ### 2. K-Means Clustering
#
# - Elbow Method와 Silhouette Score를 사용하여 최적의 클러스터 수(k)를 결정합니다.
# - 결정된 k로 K-Means 클러스터링을 수행합니다.

# %%
# Find optimal number of clusters using Elbow Method and Silhouette Score
print("\nFinding optimal k for K-Means...")
inertia = []
silhouette_scores_kmeans = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(card_data_scaled)
    inertia.append(kmeans.inertia_)
    
    # Calculate silhouette score
    labels_kmeans = kmeans.labels_
    silhouette_avg_kmeans = silhouette_score(card_data_scaled, labels_kmeans)
    silhouette_scores_kmeans.append(silhouette_avg_kmeans)
    # print(f"For n_clusters = {k}, the K-Means silhouette score is {silhouette_avg_kmeans:.3f}")

# Plot Elbow Method and Silhouette Scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores_kmeans, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

plt.tight_layout()
# plt.savefig(os.path.join(plots_dir, 'kmeans_optimal_k.png')) # EDA에서 저장됨
plt.show()
plt.close()

# Select optimal k based on elbow method and silhouette score
optimal_k = k_range[np.argmax(silhouette_scores_kmeans)] 
print(f"Selected optimal k for K-Means: {optimal_k} (Silhouette Score: {max(silhouette_scores_kmeans):.3f})")

# Apply K-Means clustering with optimal k
print(f"\nApplying K-Means with k={optimal_k}...")
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_optimal.fit_predict(card_data_scaled)

# Add cluster labels to the original DataFrame for analysis
card_data['KMeans_Cluster'] = kmeans_labels

# Analyze K-Means clusters
kmeans_cluster_analysis = card_data.groupby('KMeans_Cluster').mean()
print("\nK-Means Cluster Analysis (Mean Values):")
print(kmeans_cluster_analysis)

# %% [markdown]
# ### 3. DBSCAN 클러스터링
#
# - k-distance plot을 사용하여 적절한 epsilon(eps) 값을 탐색합니다.
# - 선택된 eps와 min_samples로 DBSCAN 클러스터링을 수행합니다.

# %%
# Find optimal epsilon using k-distance graph for DBSCAN
print("\nFinding optimal epsilon for DBSCAN using k-distance graph...")
# Calculate distances to nearest neighbors (using k=5 based on common practice)
k_neighbors = 5 * 2 # min_samples * 2 is a common heuristic, or based on feature count
neighbors = NearestNeighbors(n_neighbors=k_neighbors)
neighbors_fit = neighbors.fit(card_data_scaled)
distances, indices = neighbors_fit.kneighbors(card_data_scaled)

# Sort distances to the k-th neighbor
distances_k = np.sort(distances[:, k_neighbors-1], axis=0)

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances_k)
plt.title(f'k-Distance Graph (k={k_neighbors})')
plt.xlabel('Data Points (sorted)')
plt.ylabel(f'Distance to {k_neighbors}-th Nearest Neighbor')
# plt.axhline(y=3.5, color='r', linestyle='--') # Example threshold line, adjust based on plot
plt.grid(True)
# plt.savefig(os.path.join(plots_dir, 'dbscan_epsilon_selection.png')) # EDA에서 저장됨
plt.show()
plt.close()

# Select eps and min_samples based on the k-distance plot and data characteristics
eps_dbscan = 3.5 # Adjust this value based on the 'elbow' in the k-distance plot
min_samples_dbscan = k_neighbors # Often set to k used for k-distance or slightly higher

print(f"Selected DBSCAN parameters: eps={eps_dbscan}, min_samples={min_samples_dbscan}")

# Apply DBSCAN with selected parameters
print("\nApplying DBSCAN...")
dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
dbscan_labels = dbscan.fit_predict(card_data_scaled)

# Add cluster labels to the original DataFrame
card_data['DBSCAN_Cluster'] = dbscan_labels

# Analyze DBSCAN results
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = list(dbscan_labels).count(-1)

print(f'\nDBSCAN found {n_clusters_dbscan} clusters and {n_noise_dbscan} noise points ({n_noise_dbscan/len(card_data)*100:.2f}%).')

# Analyze DBSCAN clusters (excluding noise)
dbscan_cluster_analysis = card_data[card_data['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster').mean()
print("\nDBSCAN Cluster Analysis (Mean Values, excluding noise):")
print(dbscan_cluster_analysis)

# %% [markdown]
# ### 4. 모델 비교 및 선택
#
# - Silhouette Score를 기준으로 K-Means와 DBSCAN 결과를 비교합니다.
# - DBSCAN의 경우, 노이즈 포인트를 제외하고 Silhouette Score를 계산합니다.
# - 더 높은 점수를 기록한 모델을 최종 모델로 선택합니다.

# %%
# Compare K-Means and DBSCAN using Silhouette Score
kmeans_silhouette_score = silhouette_score(card_data_scaled, kmeans_labels)

# Calculate silhouette score for DBSCAN (excluding noise points)
dbscan_valid_indices = card_data['DBSCAN_Cluster'] != -1
if np.sum(dbscan_valid_indices) > 1 and len(set(dbscan_labels[dbscan_valid_indices])) > 1:
    dbscan_silhouette_score = silhouette_score(card_data_scaled[dbscan_valid_indices], dbscan_labels[dbscan_valid_indices])
else:
    dbscan_silhouette_score = -1 # Cannot compute silhouette score

print(f"\nK-Means Silhouette Score: {kmeans_silhouette_score:.3f}")
print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette_score:.3f}")

# Decision on which model to choose
if kmeans_silhouette_score >= dbscan_silhouette_score:
    selected_model_name = "K-Means"
    selected_labels = kmeans_labels
    selected_cluster_col = 'KMeans_Cluster'
    print("\nSelected model: K-Means (Higher or Equal Silhouette Score)")
else:
    selected_model_name = "DBSCAN"
    selected_labels = dbscan_labels
    selected_cluster_col = 'DBSCAN_Cluster'
    print("\nSelected model: DBSCAN (Higher Silhouette Score)")

# %% [markdown]
# ### 5. 선택된 모델의 군집 특성 분석 (Radar Chart)
#
# - 선택된 클러스터링 모델(K-Means 또는 DBSCAN)의 결과를 사용하여 군집별 특성을 분석합니다.
# - Radar Chart를 사용하여 각 군집의 프로파일을 시각화합니다.

# %%
# Analyze selected model's clusters
print(f"\nAnalyzing clusters from {selected_model_name}...")
cluster_analysis_selected = card_data.groupby(selected_cluster_col).mean()

# Prepare data for radar chart (use scaled data for profile comparison)
# We need the mean of the *scaled* data per cluster
card_data_scaled_df['Cluster'] = selected_labels
radar_df_data = card_data_scaled_df.groupby('Cluster').mean()

# Remove noise cluster (-1) if DBSCAN was selected
if selected_model_name == "DBSCAN" and -1 in radar_df_data.index:
    radar_df_data = radar_df_data.drop(-1)

# Function to create radar chart
def radar_chart(df, title):
    categories = list(df.columns)
    N = len(categories)
    
    # Create angle list
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=11)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    # Determine sensible y-axis limits based on scaled data range
    min_val = df.min().min() - 0.5
    max_val = df.max().max() + 0.5
    yticks = np.linspace(np.floor(min_val), np.ceil(max_val), 5)
    plt.yticks(yticks, [f"{tick:.1f}" for tick in yticks], color="grey", size=10)
    plt.ylim(np.floor(min_val), np.ceil(max_val))
    
    # Plot each cluster
    for i in range(len(df)):
        values = df.iloc[i].values.tolist()
        values += values[:1]
        cluster_label = df.index[i]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {cluster_label}")
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)
    
    return fig

# Create radar chart for selected model
print("\nGenerating Radar Chart for cluster profiles...")
radar_fig = radar_chart(radar_df_data, f'{selected_model_name} Cluster Profiles (Standardized Means)')
# radar_fig.savefig(os.path.join(plots_dir, 'cluster_radar_chart.png')) # EDA에서 저장됨
plt.show()
plt.close(radar_fig)

# %% [markdown]
# ### 6. t-SNE 시각화
#
# - t-SNE 알고리즘을 적용하여 고차원 데이터를 2차원으로 축소합니다.
# - 선택된 모델의 군집 레이블에 따라 색상을 달리하여 2차원 산점도로 시각화합니다.

# %%
# Apply t-SNE for dimensionality reduction and visualization
print("\nApplying t-SNE for visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(card_data_scaled) # Use scaled data

# Create DataFrame with t-SNE results and selected cluster labels
tsne_df = pd.DataFrame(data={'t-SNE Dim 1': tsne_results[:, 0], 't-SNE Dim 2': tsne_results[:, 1]})
tsne_df['Cluster'] = selected_labels

# Visualize t-SNE results with cluster colors
plt.figure(figsize=(12, 10))
unique_labels = sorted(tsne_df['Cluster'].unique())
palette = sns.color_palette("hls", len(unique_labels))

scatter = sns.scatterplot(
    x='t-SNE Dim 1', y='t-SNE Dim 2',
    hue='Cluster',
    palette=palette,
    hue_order=unique_labels, # Ensure consistent color mapping
    data=tsne_df,
    legend="full",
    alpha=0.7
)
plt.title(f't-SNE Visualization with {selected_model_name} Clusters', fontsize=14)
plt.legend(title='Cluster')
# plt.savefig(os.path.join(plots_dir, 'tsne_visualization.png')) # EDA에서 저장됨
plt.show()
plt.close()

print("\nAnalysis for prob2_card.csv completed.")

# %% [markdown]
# ## 3. 주택 가격 데이터 (XGBoost 과제)

# %% [markdown]
# ### (설명 생략 - EDA 대상 아님)
# - 이 섹션은 `final_exam.md`의 XGBoost 관련 계산 과제를 위한 코드 또는 설명을 포함할 수 있으나, 현재 EDA의 범위는 아닙니다.
# - `final_gpt.py`의 원래 코드에는 이 부분에 대한 구현이 포함되어 있지 않았습니다.
# - 필요시, 별도의 셀에서 XGBoost 계산 과정을 수동으로 구현하거나 설명할 수 있습니다.

# %%
# 예시: 주택 가격 데이터셋 출력 (참고용)
housing_data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'X': [3, 1, 2, 4, None, None],
    'Y': [1.25, 1.20, 1.30, 1.50, 1.40, 1.30]
})

print("\n--- Housing Dataset (XGBoost Problem) ---")
print(housing_data)
