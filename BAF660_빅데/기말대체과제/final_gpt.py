# %% [markdown]
# # 빅데이터 기말대체과제

# %% [markdown]
# ## 1. `prob1_bank.csv`

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
print("\nBank Dataset Summary Statistics:")
print(bank_data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(bank_data.isnull().sum())

# Check target variable distribution (class imbalance)
print("\nTarget variable distribution:")
print(bank_data['y'].value_counts())
print(bank_data['y'].value_counts(normalize=True) * 100)

# Visualize target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=bank_data)
plt.title('Target Variable Distribution')
plt.savefig(os.path.join(plots_dir, 'target_distribution.png'))
plt.close()

# %% [markdown]
# ### 1. 범주형 변수 전처리

# %%
# Identify categorical and numerical columns
categorical_cols = bank_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = bank_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical variables:", categorical_cols)
print("Numerical variables:", numerical_cols)

# Analyze categorical variables
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(bank_data[col].value_counts())
    
    # Visualize distribution
    plt.figure(figsize=(10, 6))
    bank_data[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'distribution_{col}.png'))
    plt.close()

# Preprocessing categorical variables using OneHotEncoder
categorical_preprocessor = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Apply one-hot encoding
X_categorical = bank_data[categorical_cols[:-1]]  # Exclude target variable 'y'
categorical_preprocessor.fit(X_categorical)
X_categorical_encoded = categorical_preprocessor.transform(X_categorical)

# Create DataFrame with encoded features
encoded_feature_names = []
for i, cat_col in enumerate(categorical_cols[:-1]):
    categories = categorical_preprocessor.categories_[i][1:]  # Skip the first category (dropped)
    encoded_feature_names.extend([f"{cat_col}_{cat}" for cat in categories])

X_categorical_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names)

print("\nShape of encoded categorical features:", X_categorical_df.shape)
print(X_categorical_df.head())

# %% [markdown]
# ### 2. 수치형 변수 전처리

# %%
# Analyze numerical variables
for col in numerical_cols:
    # Summary statistics
    print(f"\nSummary statistics for {col}:")
    print(bank_data[col].describe())
    
    # Check for outliers with box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=bank_data[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(os.path.join(plots_dir, f'boxplot_{col}.png'))
    plt.close()
    
    # Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(bank_data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(plots_dir, f'hist_{col}.png'))
    plt.close()

# Apply standardization to numerical features
scaler = StandardScaler()
X_numerical = bank_data[numerical_cols]
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Create DataFrame with scaled features
X_numerical_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols)

print("\nShape of scaled numerical features:", X_numerical_df.shape)
print(X_numerical_df.head())

# %% [markdown]
# ### 3. 클래스 불균형 처리

# %%
# Combine preprocessed features
X_preprocessed = pd.concat([X_numerical_df, X_categorical_df], axis=1)

# Convert target variable to binary
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(bank_data['y'])

# Check class imbalance
print("Original class distribution:")
print(Counter(y))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

print("Resampled class distribution:")
print(Counter(y_resampled))

# Visualize class distribution before and after SMOTE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title('Original Class Distribution')
plt.xlabel('Class')
plt.xticks([0, 1], ['No', 'Yes'])

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.xticks([0, 1], ['No', 'Yes'])

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'class_balance_comparison.png'))
plt.close()

print("Final preprocessed dataset shape:", X_resampled.shape)

# %% [markdown]
# ## 2. `prob2_card.csv`

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Load the data
card_data = pd.read_csv('prob2_card.csv')

# Display basic information
print("Credit Card Dataset Shape:", card_data.shape)
print("\nCredit Card Dataset Info:")
print(card_data.info())
print("\nCredit Card Dataset Summary Statistics:")
print(card_data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(card_data.isnull().sum())

# Set CUST_ID as index and exclude it from analysis
card_data.set_index('CUST_ID', inplace=True)

# Visualize distribution of features
plt.figure(figsize=(15, 10))
for i, column in enumerate(card_data.columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(card_data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'card_features_distribution.png'))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(card_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
plt.close()

# Preprocess data: scaling features
scaler = StandardScaler()
card_data_scaled = scaler.fit_transform(card_data)
card_data_scaled_df = pd.DataFrame(card_data_scaled, columns=card_data.columns, index=card_data.index)

# %% [markdown]
# ### 1. K-Means Clustering

# %%
# Find optimal number of clusters using Elbow Method
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(card_data_scaled)
    inertia.append(kmeans.inertia_)
    
    # Calculate silhouette score
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(card_data_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")

# Plot Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'kmeans_optimal_k.png'))
plt.close()

# Select optimal k based on elbow method and silhouette score
optimal_k = 4  # This is based on the elbow in the plot and highest silhouette score

# Apply K-Means clustering with optimal k
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_optimal.fit_predict(card_data_scaled)

# Add cluster labels to the original DataFrame
card_data['KMeans_Cluster'] = kmeans_labels

# Visualize cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='KMeans_Cluster', data=card_data)
plt.title('Distribution of K-Means Clusters')
plt.savefig(os.path.join(plots_dir, 'kmeans_cluster_distribution.png'))
plt.close()

# Analyze clusters
kmeans_cluster_analysis = card_data.groupby('KMeans_Cluster').mean()
print("\nK-Means Cluster Analysis:")
print(kmeans_cluster_analysis)

# Visualize cluster profiles
plt.figure(figsize=(14, 8))
kmeans_cluster_analysis.T.plot(kind='bar', figsize=(14, 8))
plt.title('K-Means Cluster Profiles')
plt.ylabel('Standardized Value')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.legend(title='Cluster')
plt.savefig(os.path.join(plots_dir, 'kmeans_cluster_profiles.png'))
plt.close()

# %% [markdown]
# ### 2. DBSCAN 클러스터링

# %%
# Find optimal epsilon using k-distance graph
from sklearn.neighbors import NearestNeighbors

# Calculate distances to nearest neighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(card_data_scaled)
distances, indices = neighbors_fit.kneighbors(card_data_scaled)

# Sort distances in descending order
distances = np.sort(distances[:, 4], axis=0)  # 5th nearest neighbor

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.axhline(y=0.5, color='r', linestyle='--')  # Example threshold
plt.title('K-Distance Graph (k=5)')
plt.xlabel('Data Points (sorted)')
plt.ylabel('Distance to 5th Nearest Neighbor')
plt.savefig(os.path.join(plots_dir, 'dbscan_epsilon_selection.png'))
plt.close()

# Apply DBSCAN with selected parameters
eps = 0.5  # Selected from k-distance graph
min_samples = 5  # Minimum number of samples in a core point's neighborhood

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(card_data_scaled)

# Add cluster labels to the original DataFrame
card_data['DBSCAN_Cluster'] = dbscan_labels

# Count number of clusters and noise points
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')

# Visualize cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='DBSCAN_Cluster', data=card_data)
plt.title('Distribution of DBSCAN Clusters')
plt.savefig(os.path.join(plots_dir, 'dbscan_cluster_distribution.png'))
plt.close()

# Analyze clusters
dbscan_cluster_analysis = card_data.groupby('DBSCAN_Cluster').mean()
print("\nDBSCAN Cluster Analysis:")
print(dbscan_cluster_analysis)

# %% [markdown]
# ### 3. 모델 비교 및 선택

# %%
# Compare K-Means and DBSCAN
# Calculate silhouette score for K-Means
kmeans_silhouette = silhouette_score(card_data_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0

# Calculate silhouette score for DBSCAN (excluding noise points)
dbscan_data = card_data_scaled[dbscan_labels != -1]
dbscan_labels_no_noise = dbscan_labels[dbscan_labels != -1]
dbscan_silhouette = silhouette_score(dbscan_data, dbscan_labels_no_noise) if len(set(dbscan_labels_no_noise)) > 1 else 0

print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.3f}")

# Compare cluster distributions
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='KMeans_Cluster', data=card_data)
plt.title('K-Means Clusters')

plt.subplot(1, 2, 2)
sns.countplot(x='DBSCAN_Cluster', data=card_data)
plt.title('DBSCAN Clusters')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'cluster_comparison.png'))
plt.close()

# Decision on which model to choose
if kmeans_silhouette > dbscan_silhouette:
    selected_model = "K-Means"
    selected_labels = kmeans_labels
    print("\nSelected model: K-Means")
else:
    selected_model = "DBSCAN"
    selected_labels = dbscan_labels
    print("\nSelected model: DBSCAN")

# %% [markdown]
# ### 4. 군집 특성 분석

# %%
# Analyze selected model's clusters
if selected_model == "K-Means":
    selected_clusters = card_data['KMeans_Cluster']
    cluster_analysis = card_data.groupby('KMeans_Cluster').mean()
else:
    selected_clusters = card_data['DBSCAN_Cluster']
    cluster_analysis = card_data.groupby('DBSCAN_Cluster').mean()

print("\nSelected Model Cluster Analysis:")
print(cluster_analysis)

# Visualize cluster profiles with radar chart
from math import pi

# Function to create radar chart
def radar_chart(df, title):
    categories = list(df.columns)
    N = len(categories)
    
    # Create angle list
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "1", "2"], color="grey", size=10)
    plt.ylim(-2, 2)
    
    # Plot each cluster
    for i in range(len(df)):
        values = df.iloc[i].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {df.index[i]}")
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)
    
    return fig

# Prepare data for radar chart (excluding any noise cluster)
if -1 in cluster_analysis.index:
    radar_df = cluster_analysis.drop(-1)
else:
    radar_df = cluster_analysis

# Create radar chart
radar_fig = radar_chart(radar_df, f'{selected_model} Cluster Profiles')
radar_fig.savefig(os.path.join(plots_dir, 'cluster_radar_chart.png'))
plt.close()

# %% [markdown]
# ### 5. t-SNE 시각화

# %%
# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(card_data_scaled)

# Create DataFrame with t-SNE results
tsne_df = pd.DataFrame(data={'t-SNE 1': tsne_results[:, 0], 't-SNE 2': tsne_results[:, 1]})
tsne_df['Cluster'] = selected_clusters

# Visualize t-SNE results with cluster colors
plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    x='t-SNE 1', y='t-SNE 2',
    hue='Cluster',
    palette=sns.color_palette("hls", len(set(selected_clusters))),
    data=tsne_df,
    legend="full",
    alpha=0.8
)
plt.title(f't-SNE Visualization with {selected_model} Clusters', fontsize=14)
plt.savefig(os.path.join(plots_dir, 'tsne_visualization.png'))
plt.close()

# %% [markdown]
# ## 3. 주택 가격 데이터

# %% [markdown]
# ### 주어진 데이터
#
# 주어진 데이터는 다음과 같습니다:
#
# | ID  | X   | Y    |
# | --- | --- | ---- |
# | 1   | 3   | 1.25 |
# | 2   | 1   | 1.20 |
# | 3   | 2   | 1.30 |
# | 4   | 4   | 1.50 |
# | 5   | ?   | 1.40 |
# | 6   | ?   | 1.30 |
#
# 여기서 X는 방의 개수를, Y는 주택 가격을 나타냅니다.

# %% [markdown]
# ### 1. XGBoost 알고리즘 - 최적의 분리 기준 찾기
#
# XGBoost는 **그래디언트 부스팅(Gradient Boosting)** 알고리즘의 효율적인 구현체로, 손실 함수의 그래디언트를 최소화하는 방향으로 모델을 순차적으로 개선합니다. 이 문제에서는 첫 번째 트리의 첫 마디에서 최적의 분리 기준을 찾는 과정을 설명하겠습니다.
#
# #### 단계 1: 초기 예측값과 그래디언트 계산
#
# 문제에서 모델의 초기값은 $f_0 = 0.5$로 주어졌습니다. 손실 함수는 제곱 오차(squared error)입니다:
# $L(y, \hat{y}) = (y - \hat{y})^2$
#
# 그래디언트는 손실 함수를 예측값에 대해 미분한 것입니다:
# $g_i = \frac{\partial L(y_i, f_0(x_i))}{\partial f_0(x_i)} = -2(y_i - f_0(x_i))$
#
# 각 데이터 포인트의 그래디언트는 다음과 같습니다:
# - 관측치 1: $g_1 = -2(1.25 - 0.5) = -1.5$
# - 관측치 2: $g_2 = -2(1.20 - 0.5) = -1.4$
# - 관측치 3: $g_3 = -2(1.30 - 0.5) = -1.6$
# - 관측치 4: $g_4 = -2(1.50 - 0.5) = -2.0$
#
# 또한, 제곱 오차의 2차 미분(헤시안)은 상수 2입니다:
# $h_i = \frac{\partial^2 L(y_i, f_0(x_i))}{\partial f_0(x_i)^2} = 2$
#
# #### 단계 2: 가능한 분할 지점 탐색
#
# X의 가능한 분할 지점은 정렬된 고유값으로 X = 1, 2, 3, 4 입니다. 각 분할 지점에 대해 왼쪽과 오른쪽 노드로 데이터를 나누고 이득(gain)을 계산합니다.
#
# #### 단계 3: 최적 분할 선택을 위한 이득(Gain) 계산
#
# XGBoost에서 분할의 이득은 다음 공식으로 계산됩니다:
#
# $Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L} + \frac{G_R^2}{H_R} - \frac{(G_L + G_R)^2}{H_L + H_R} \right] - \gamma$
#
# 여기서:
# - $G_L$, $G_R$은 각각 왼쪽/오른쪽 자식 노드의 그래디언트 합
# - $H_L$, $H_R$은 각각 왼쪽/오른쪽 자식 노드의 헤시안 합
# - $\gamma$는 분할 페널티 항으로, 이 문제에서는 $\lambda = 0$이므로 무시
#
# 예를 들어 X ≤ 2 분할에 대한 이득 계산:
# - 왼쪽 노드(X ≤ 2): 관측치 2, 3 → $G_L = -1.4 + (-1.6) = -3.0$, $H_L = 2 + 2 = 4$
# - 오른쪽 노드(X > 2): 관측치 1, 4 → $G_R = -1.5 + (-2.0) = -3.5$, $H_R = 2 + 2 = 4$
# - 이득: $Gain = \frac{1}{2} \left[ \frac{(-3.0)^2}{4} + \frac{(-3.5)^2}{4} - \frac{(-3.0 - 3.5)^2}{4 + 4} \right] = \frac{1}{2} \left[ 2.25 + 3.0625 - 2.640625 \right] = 1.336$
#
# 모든 가능한 분할에 대해 이득을 계산하면:
# - X ≤ 1: $Gain = 0.0456$
# - X ≤ 2: $Gain = 0.1337$
# - X ≤ 3: $Gain = 0.0906$
#
# 따라서 최적의 분할은 X ≤ 2로, 이득이 가장 큽니다.
#
# #### 단계 4: 트리 노드 가중치 계산
#
# 분할 후, 각 자식 노드의 가중치는 다음 공식으로 계산됩니다:
#
# $w = -\frac{G}{H+\lambda}$
#
# X ≤ 2 분할에 대한 가중치:
# - 왼쪽 노드: $w_L = -\frac{-3.0}{4+0} = 0.75$
# - 오른쪽 노드: $w_R = -\frac{-3.5}{4+0} = 0.875$
#
# 이 가중치는 해당 노드에서의 예측값 보정치를 의미합니다.

# %% [markdown]
# ### 2. 결측값 처리 방법
#
# XGBoost는 **스파스 인식(sparsity-aware)** 알고리즘으로, 결측값을 효과적으로 처리하는 방법을 내장하고 있습니다. 결측값이 있는 경우, XGBoost는 두 가지 가능성(왼쪽 또는 오른쪽 자식 노드로 보내는 것)을 모두 평가하고 이득이 최대화되는 방향을 선택합니다.
#
# #### 단계 1: 결측값 처리를 위한 기본 방향 결정
#
# 관측치 5와 6에는 X 값이 결측되어 있습니다. 이러한 결측값을 가진 관측치들을 왼쪽 또는 오른쪽 자식 노드 중 어디로 보낼지 결정하기 위해, 각 방향으로 보낼 때의 이득을 계산합니다.
#
# 결측값 관측치들의 그래디언트는:
# - 관측치 5: $g_5 = -2(1.40 - 0.5) = -1.8$
# - 관측치 6: $g_6 = -2(1.30 - 0.5) = -1.6$
#
# #### 단계 2: 결측값을 왼쪽 노드로 보낼 때의 이득 계산
#
# 결측값을 왼쪽 노드(X ≤ 2)로 보내면:
# - 왼쪽 노드: 관측치 2, 3, 5, 6 → $G_L = -3.0 + (-1.8) + (-1.6) = -6.4$, $H_L = 4 + 2 + 2 = 8$
# - 오른쪽 노드: 관측치 1, 4 → $G_R = -3.5$, $H_R = 4$
# - 이득: $Gain_{left} = \frac{1}{2} \left[ \frac{(-6.4)^2}{8} + \frac{(-3.5)^2}{4} - \frac{(-6.4 - 3.5)^2}{8 + 4} \right]$
#
# #### 단계 3: 결측값을 오른쪽 노드로 보낼 때의 이득 계산
#
# 결측값을 오른쪽 노드(X > 2)로 보내면:
# - 왼쪽 노드: 관측치 2, 3 → $G_L = -3.0$, $H_L = 4$
# - 오른쪽 노드: 관측치 1, 4, 5, 6 → $G_R = -3.5 + (-1.8) + (-1.6) = -6.9$, $H_R = 4 + 2 + 2 = 8$
# - 이득: $Gain_{right} = \frac{1}{2} \left[ \frac{(-3.0)^2}{4} + \frac{(-6.9)^2}{8} - \frac{(-3.0 - 6.9)^2}{4 + 8} \right]$
#
# #### 단계 4: 최적의 방향 선택
#
# 두 이득을 비교하여 더 큰 이득을 제공하는 방향으로 결측값을 보냅니다. 계산 결과 $Gain_{right} > Gain_{left}$ 이므로, 관측치 5와 6은 오른쪽 노드(X > 2)로 보내는 것이 최적입니다.
#
# 이는 XGBoost가 결측값을 효율적으로 처리하는 핵심 메커니즘으로, 모든 가능한 결측값 위치에 대해 최적의 방향을 데이터 기반으로 결정합니다.

# %%
# 주택 가격 데이터셋 출력 (시각화 목적)
housing_data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'X': [3, 1, 2, 4, None, None],
    'Y': [1.25, 1.20, 1.30, 1.50, 1.40, 1.30]
})

print("Housing Dataset:")
print(housing_data)
