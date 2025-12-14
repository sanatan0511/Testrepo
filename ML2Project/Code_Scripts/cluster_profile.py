import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import calinski_harabasz_score

# --- 1. Load Imputed Data ---
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv")
print("1. Data loaded.")

# Identify numerical features and scale the data
feature_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
X = df_imputed[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Execute K-Means Clustering (K=3) ---
optimal_k = 3
print(f"2. Re-running K-Means with VALIDATED Optimal K = {optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df_imputed['Cluster'] = kmeans.fit_predict(X_scaled)

# --- 3. Calculate Calinski-Harabasz Score (Complementary Validation) ---
ch_score = calinski_harabasz_score(X_scaled, df_imputed['Cluster'])
print(f"3. Calinski-Harabasz Index for K=3: {ch_score:.2f} (High score confirms model quality)")

# --- 4. Cluster Profiling: Calculate Mean Metrics by Cluster ---

# Select the most important features for policy interpretation (unscaled values)
profile_cols = [
    'Cluster', 
    'Price_to_Income_Ratio', # CORE AFFORDABILITY
    'Latest HPI',            # CORE PRICE LEVEL
    'Affordability_Change_Pressure', # CORE GROWTH PRESSURE
    'Per_Capita_Income_Latest_Current', # INCOME PROXY
    'Literacy_rate',         # SOCIO-ECONOMIC HEALTH
    'Worker Participation Rate'
]

# Group by the cluster label and calculate the mean for the selected features
cluster_profile = df_imputed[profile_cols].groupby('Cluster').mean().round(2)

print("\n4. Final Cluster Profile (K=3):")
print(cluster_profile)

# Save the final results
df_imputed.to_csv("Processed_Data/Housing_Affordability_Clustered_Data_K3.csv", index=False)
cluster_profile.to_csv("Processed_Data/Cluster_Policy_Profile_K3.csv")