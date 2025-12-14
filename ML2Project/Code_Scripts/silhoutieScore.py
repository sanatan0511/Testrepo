import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# --- 1. Load Imputed Data ---
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv")

# Identify numerical features and scale the data (as done previously)
feature_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
X = df_imputed[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("1. Calculating Silhouette Scores...")

results = {}
for k in [3, 4]:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    # Calculate Silhouette Score
    score = silhouette_score(X_scaled, kmeans.labels_)
    results[k] = score

print(f"Silhouette Score for K=3: {results[3]:.4f}")
print(f"Silhouette Score for K=4: {results[4]:.4f}")

# Determine the best K based on the score
best_k = max(results, key=results.get)
print(f"\nBest K based on Silhouette Score: K={best_k}")