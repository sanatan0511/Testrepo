import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Data and Scaling (Repeat for consistency) ---
df_final = pd.read_csv("Processed_Data/Housing_Affordability_Clustered_Data_K3.csv")

feature_cols = df_final.select_dtypes(include=np.number).columns.tolist()
feature_cols.remove('Cluster') 
X = df_final[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Run K-Means (Fit the model again to get centroids) ---
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)

# --- 3. Run PCA and Project Data ---
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
pca_df['Cluster'] = df_final['Cluster']

# --- 4. Project Centroids onto PCA Space (The Key Step) ---

# Get the centroids from the fitted K-Means model (still in 12-dim space)
centroids_scaled = kmeans.cluster_centers_

# Transform the 12-dim centroids into the 2-dim PCA space
centroids_pca = pca.transform(centroids_scaled)
centroids_df = pd.DataFrame(data = centroids_pca, columns = ['PC1', 'PC2'])
centroids_df['Cluster'] = [0, 1, 2]


# --- 5. Plot Data Points and Centroids ---
plt.figure(figsize=(12, 8))
targets = [0, 1, 2]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

# Plotting Data Points (Cities)
for target, color in zip(targets, colors):
    # Plot cities belonging to this cluster
    indicesToKeep = pca_df['Cluster'] == target
    plt.scatter(pca_df.loc[indicesToKeep, 'PC1'],
                pca_df.loc[indicesToKeep, 'PC2'],
                c = color,
                s = 70,
                alpha = 0.7,
                edgecolors='w',
                label = f'Tier {target+1} Cities (Cluster {target})')
    
    # Plot Centroids (Cluster Centers)
    centroid_index = centroids_df['Cluster'] == target
    plt.scatter(centroids_df.loc[centroid_index, 'PC1'],
                centroids_df.loc[centroid_index, 'PC2'],
                marker='X', # Use 'X' marker for centroid
                c = color,
                s = 400, # Large size to stand out
                edgecolors='k',
                linewidths=2)


plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('Housing Affordability Policy Tiers (K=3) with Centroids')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('image/pca_visualization_with_centroids.png')
print("Final PCA visualization with centroids saved to image folder pca_visualization_with_centroids.png")