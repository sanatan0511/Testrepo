import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# --- 1. Load Data and Scaling ---
# This ensures X_scaled is defined within this script's memory
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv")

feature_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
X = df_imputed[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Model 2: Agglomerative Hierarchical Clustering (AHC) ---")

# --- 2. Run AHC with K=3 for Direct Comparison ---
ahc = AgglomerativeClustering(n_clusters=3, linkage='ward')
ahc_labels = ahc.fit_predict(X_scaled)

# --- 3. Analyze Results ---
df_imputed['AHC_Cluster'] = ahc_labels

print("AHC Cluster Assignments (K=3):")
print(df_imputed['AHC_Cluster'].value_counts())

# Save results for comparison
df_imputed[['City', 'State_Standard', 'AHC_Cluster']].to_csv("AHC_Cluster_Assignments.csv", index=False)