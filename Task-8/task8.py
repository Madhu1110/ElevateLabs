import os
os.environ["OMP_NUM_THREADS"] = "1"


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load and preprocess dataset
df = pd.read_csv("C:/Users/madhu/OneDrive/Documents/ElevateLabs/task8/archive (7)/Mall_Customers.csv")  
print(df.head())
X = df.iloc[:, [3, 4]] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for 2D view
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

#  Fit KMeans with optimal K
optimal_k = 5  # Based on Elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set2', s=100)
plt.title('Customer Segments Visualized with PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Evaluate with Silhouette Score
score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score for k={optimal_k}: {score:.2f}")
