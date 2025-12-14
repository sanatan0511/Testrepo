import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

# --- 1. Load Data and Scaling ---
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv")

feature_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
X = df_imputed[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Model 1: DBSCAN Clustering ---")

# --- 2. Run DBSCAN ---
# Initial parameters:
# Epsilon (eps) is typically found via a K-distance plot (which is complex). 
# We'll start with a value (0.5) suitable for standardized data.
# min_samples = 2 * (number of features), which is 2 * 12 = 24. We'll try a common starting point of 5.
dbscan = DBSCAN(eps=0.5, min_samples=5) 
dbscan_labels = dbscan.fit_predict(X_scaled)

# --- 3. Analyze Results ---
df_imputed['DBSCAN_Cluster'] = dbscan_labels

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN Results (eps=0.5, min_samples=5):")
print(f"Total Clusters Found: {n_clusters_dbscan}")
print(f"Total Noise Points (Outliers): {n_noise}")
print("DBSCAN Cluster Assignments:")
print(df_imputed['DBSCAN_Cluster'].value_counts())

# Save results for comparison
df_imputed[['City', 'State_Standard', 'DBSCAN_Cluster']].to_csv("Processed_Data/DBSCAN_Cluster_Assignments.csv", index=False)