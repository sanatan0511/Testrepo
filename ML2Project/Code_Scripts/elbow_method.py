import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Imputed Data ---
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv")

# Identify numerical features for clustering
feature_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
X = df_imputed[feature_cols]

# --- 2. Scaling the Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Determine Optimal K using the Elbow Method ---
wcss = []
# Test K from 1 up to 10
k_range = range(1, 11) 

for k in k_range:
    # Set n_init to 10 to suppress future warnings in newer sklearn versions
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia is the WCSS

# --- 4. Plot the Elbow Curve ---
plt.figure(figsize=(9, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K (WCSS)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('image/elbow_method_plot.png') 
plt.show()

print("\n--- Elbow Method Plot Generated: elbow_method_plot.png ---")