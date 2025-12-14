import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- NOTE: Replace the content below with your final data loading code if needed ---
# Assuming the file is now available after fixing the path locally.
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv")
# ----------------------------------------------------------------------------------

feature_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
X = df_imputed[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CLUSTERING
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
df_imputed['KM_Cluster'] = kmeans.fit_predict(X_scaled)
ahc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df_imputed['AHC_Cluster'] = ahc.fit_predict(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5) 
df_imputed['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_scaled)
pca_plotting_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
pca_plotting_df['KM_Cluster'] = df_imputed['KM_Cluster']
pca_plotting_df['AHC_Cluster'] = df_imputed['AHC_Cluster']
pca_plotting_df['DBSCAN_Cluster'] = df_imputed['DBSCAN_Cluster']

cluster_types = ['KM_Cluster', 'AHC_Cluster', 'DBSCAN_Cluster']
titles = ['K-Means Clustering (K=3)', 'Agglomerative Clustering (K=3)', 'DBSCAN Clustering']
filenames = ['pca_kmeans_clustering.png', 'pca_ahc_clustering.png', 'pca_dbscan_clustering.png']

def plot_pca_clusters(df, cluster_col, title, filename):
    plt.figure(figsize=(9, 6))
    unique_labels = sorted(df[cluster_col].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels) + 1))
    if -1 in unique_labels:
        unique_labels.remove(-1)
        unique_labels.append(-1)
        colors[-1] = np.array([0.5, 0.5, 0.5, 1.0]) 

    for i, label in enumerate(unique_labels):
        indices_to_keep = df[cluster_col] == label
        label_name = 'Noise (Outliers)' if label == -1 else f'Cluster {label}'
        color_index = i if label != -1 else len(unique_labels) - 1

        plt.scatter(df.loc[indices_to_keep, 'PC1'],
                    df.loc[indices_to_keep, 'PC2'],
                    c = colors[color_index],
                    s = 50,
                    alpha=0.7,
                    label = label_name)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.legend(title='Cluster Label', loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Generate plots
for col, title, filename in zip(cluster_types, titles, filenames):
    plot_pca_clusters(pca_plotting_df, col, title, filename)

print("SUCCESS: Three PCA plots generated and saved.")