import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Clustered Data ---
df_final = pd.read_csv("Processed_Data/Housing_Affordability_Clustered_Data_K3.csv")

# Identify numerical features and scale the data
feature_cols = df_final.select_dtypes(include=np.number).columns.tolist()
# Exclude the final 'Cluster' column from features
feature_cols.remove('Cluster') 
X = df_final[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Run PCA to Reduce to 2 Dimensions ---
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

# Add cluster labels back for plotting
pca_df['Cluster'] = df_final['Cluster']

# Calculate variance explained by the two components
variance_explained = pca.explained_variance_ratio_.sum()
print(f"Total Variance Explained by PC1 and PC2: {variance_explained*100:.2f}%")


# --- 3. Plot the Clusters in PCA Space ---
plt.figure(figsize=(10, 7))
targets = [0, 1, 2]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

for target, color in zip(targets, colors):
    indicesToKeep = pca_df['Cluster'] == target
    plt.scatter(pca_df.loc[indicesToKeep, 'PC1'],
                pca_df.loc[indicesToKeep, 'PC2'],
                c = color,
                s = 50,
                label = f'Tier {target+1} (Cluster {target})') # Labeling tiers for policy clarity

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('Housing Affordability Clustering (K=3) - Policy Tiers')
plt.legend()
plt.grid()
plt.savefig('image/pca_cluster_visualization.png')