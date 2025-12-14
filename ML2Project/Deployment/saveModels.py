import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os

# Define the file name that contains your final K=3 cluster assignments
INPUT_FILE = "Processed_Data/Housing_Affordability_Clustered_Data_K3.csv"
MODEL_FILENAME = "Deployment/kmeans_model.joblib"
SCALER_FILENAME = "Deployment/scaler.joblib"

# --- 1. Load the Final Processed Data ---
try:
    df_imputed = pd.read_csv(INPUT_FILE) 
    print(f"Successfully loaded final data from {INPUT_FILE}.")
except FileNotFoundError:
    print(f"\nFATAL ERROR: The necessary input file ({INPUT_FILE}) is missing.")
    print("Please ensure your data pipeline script created this file in the current directory.")
    exit()

# Define feature columns (must be in the exact order as the Flask API)
feature_cols = [
    'Latest HPI', 'yoy_change', 'CPI_Housing_Index', 'P_LIT', 'TOT_WORK_P',
    'Literacy_rate', 'Worker Participation Rate', 'SC/ST Population Percentage',
    'Households per Capita', 'Per_Capita_Income_Latest_Current',
    'Price_to_Income_Ratio', 'Affordability_Change_Pressure'
]

X = df_imputed[feature_cols]

# --- 2. Fit and Save the StandardScaler (CRITICAL) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_FILENAME)
print(f"Successfully saved StandardScaler to {SCALER_FILENAME}.")

# --- 3. Fit and Save the K-Means Model ---
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled) 
joblib.dump(kmeans, MODEL_FILENAME)
print(f"Successfully saved K-Means Model to {MODEL_FILENAME}.")

print("\nDeployment assets are ready! You can now start the Flask application.")