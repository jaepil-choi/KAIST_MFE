# ch2
# March 13, 2024

# [1]: Import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# [2]: Mount Google Drive (for Google Colab)
from google.colab import drive
drive.mount('/content/drive')
# Note: Drive already mounted at /content/drive; to forcibly remount, use:
# drive.mount("/content/drive", force_remount=True).

# [3]: Load the dataset from an Excel file on Google Drive
datafile = '/content/drive/MyDrive/2024 DFMBA 빅데이터와 금융자료분석/data/ProcessedData.xlsx'
dataset = pd.read_excel(datafile)

# [4]: Check the shape of the dataset
dataset.shape
# Expected output: (3866, 13)

# [5]: Display the first few rows of the dataset
dataset.head()
# Expected output (example):
#    ID  AGE  EDUC  MARRIED  KIDS  LIFECL  OCCAT  RISK  HHOUSES  WSAVED
# 0   1    3     2        1     0       2      1     3        1       1
# 1   2    4     4        1     2       5      2     3        0       2
# 2   3    3     1        1     2       3      2     2        1       2
# 3   4    3     1        1     2       3      2     2        1       2
# 4   5    4     3        1     1       5      1     2        1       3
#
# Additional columns:
# SPENDMOR, NWCAT, INCCL

# [6]: Check for any missing values in the dataset
dataset.isnull().values.any()
# Expected output: False

# [7]: Remove the 'ID' column from the dataset to use as features
X = dataset.drop(['ID'], axis=1)

# [8]: Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_arr = scaler.fit_transform(X)
X = pd.DataFrame(X_arr, columns=X.columns)

# [9]: Compute the k-nearest neighbors (NN = 4) for each point
from sklearn.neighbors import NearestNeighbors
NN = 4
neigh = NearestNeighbors(n_neighbors=NN)
neigh.fit(X)
distances, indices = neigh.kneighbors(X)
np.round(distances, decimals=3)
# Expected output: An array of distances (first column zeros, subsequent columns show distances)

# [10]: Plot the sorted distances for the 4th nearest neighbor (to help choose ε)
plt.figure(figsize=(5, 3))
plt.plot(np.sort(distances[:, NN - 1]))
plt.axhline(2.0, ls='--')
# A horizontal line is drawn at 2.0 to help visualize the elbow

# [11]: Plot sorted k-distance graphs for different NN values (from 4 to 8)
plt.figure(figsize=(5, 3))
for NN in range(4, 9):
    neigh = NearestNeighbors(n_neighbors=NN)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    plt.plot(np.sort(distances[:, NN - 1]), label='NN:' + str(NN))
plt.legend()
# This plot helps compare the k-distance curves for different choices of NN

# [12]: Perform DBSCAN clustering with ε=2.0 and min_samples=4
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=2.0, min_samples=4).fit(X)
df = X.copy()
df['Label'] = db.labels_
df['Label'].value_counts()
# Expected output: Counts of each cluster label (e.g., clusters 0, 1, 2, etc., and noise as -1)

# [13]: Determine the maximum cluster label (number of clusters - 1)
nc = np.max(db.labels_)
nc
# Expected output: 10

# [14]: Calculate and display the mean values for each cluster
clout = df.groupby('Label').mean()
clout.round(decimals=2)
# Expected output: A DataFrame showing the mean of each feature per cluster

# [15]: Plot bar charts for selected features by cluster (first 7 clusters)
clout[['AGE', 'EDUC', 'MARRIED', 'KIDS', 'LIFECL', 'OCCAT']][0:7].plot.bar(rot=0)
# Expected output: Bar plot for the mentioned features

# [16]: Plot bar charts for additional features by cluster (first 7 clusters)
clout[['HHOUSES', 'NWCAT', 'INCCL', 'WSAVED', 'SPENDMOR', 'RISK']][0:7].plot.bar(rot=0)
# Expected output: Bar plot for the additional features

# [17]: Apply t-SNE for dimensionality reduction to 2 components and add cluster labels
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
df2dim = tsne.fit_transform(X)
df2dim = pd.DataFrame(df2dim, columns=['t1', 't2'])
df2dim['Labels'] = db.labels_
df2dim.head()
# Expected output: A DataFrame with columns 't1', 't2', and 'Labels'

# [18]: Scatter plot of the t-SNE 2D embedding colored by DBSCAN cluster labels
for k in range(-1, nc + 1):
    plt.scatter(df2dim['t1'][db.labels_ == k],
                df2dim['t2'][db.labels_ == k],
                label='c: ' + str(k),
                s=0.8)
plt.legend()
plt.xlim(-70, 120)
# Expected output: A scatter plot with clusters and x-axis limits set

# [19]: Calculate the Sum of Squared Errors (SSE) for different KMeans cluster counts
from sklearn.cluster import KMeans
SSE = []
max_loop = 20
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)
    
# [20]: Plot the SSE values to visualize the "elbow" for optimal KMeans clustering
fig = plt.figure(figsize=(6, 4))
plt.plot(range(2, max_loop), SSE, 'bo-')
# Expected output: An elbow plot for KMeans

# [21]: Perform KMeans clustering with 9 clusters and add the labels to the DataFrame
km = KMeans(n_clusters=9, n_init='auto')
km.fit(X)
df['Label_km'] = km.labels_

# [22]: Display the first few rows of the DataFrame with KMeans labels (rounded)
df.head().round(2)
# Expected output: DataFrame head with new column 'Label_km'

# [23]: Import silhouette metrics for cluster evaluation
from sklearn.metrics import silhouette_score, silhouette_samples

# [24]: Define a function to plot the silhouette scores for a given clustering
def plot_silhouette(X, clusters):
    silhouette_avg = np.round(silhouette_score(X, clusters), decimals=3)
    sample_silhouette_values = silhouette_samples(X, clusters)
    
    fig, ax = plt.subplots()
    y_lower = 10
    
    for i in np.unique(clusters):
        cluster_silhouette_values = sample_silhouette_values[clusters == i]
        cluster_silhouette_values.sort()
    
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         alpha=0.7)
    
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    
    plt.title("Silhouette Score: {}".format(silhouette_avg))
    plt.show()

# [25]: Plot the silhouette for the DBSCAN clustering
plot_silhouette(X, df['Label'])

# [26]: Plot the silhouette for the KMeans clustering
plot_silhouette(X, df['Label_km'])
