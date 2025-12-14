# GMM.py - Gaussian Mixture Model Training, Validation, and Persistence

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from joblib import dump, load
import os

# --- 1. Configuration and File Paths ---
PROCESSED_DIR = os.path.join("Processed_Data")
DEPLOYMENT_DIR = os.path.join("Deployment")
INPUT_FILE = "Housing_Affordability_Clustered_Data_K3.csv" 
OUTPUT_FILE = "GMM_Cluster_Assignments.csv"
GMM_MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "gmm_model.joblib")
SCALER_MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "scaler.joblib") # Path for your fitted StandardScaler

# --- IMPORTANT: These are the 12 unscaled feature columns from your input CSV ---
# These columns are used to load the data and will be scaled using the saved StandardScaler.
ORIGINAL_FEATURES = [
    'Latest HPI', 'yoy_change', 'CPI_Housing_Index', 'P_LIT', 
    'TOT_WORK_P', 'Literacy_rate', 'Worker Participation Rate', 
    'SC/ST Population Percentage', 'Households per Capita', 
    'Per_Capita_Income_Latest_Current', 'Price_to_Income_Ratio', 
    'Affordability_Change_Pressure'
]

# The corresponding names for the scaled features (used for GMM training and naming)
SCALED_FEATURES_RENAMED = [f'Scaled_{col.replace(" ", "_")}' for col in ORIGINAL_FEATURES]


K_CLUSTERS = 3
KMEANS_LABEL_COL = 'Cluster' # The column containing your K=3 K-Means assignments
BORDERLINE_THRESHOLD = 0.20 
RANDOM_SEED = 42

def train_and_validate_gmm():
    """
    Loads unscaled data, applies scaling, trains GMM, validates against K-Means, 
    identifies borderline cities, and saves results/model.
    """
    print(f"--- Starting GMM Processing (K={K_CLUSTERS}) ---")
    
    # --- 2. Load Data and Prepare Matrices ---
    try:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, INPUT_FILE))
        print(f"Data loaded successfully from {INPUT_FILE}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {os.path.join(PROCESSED_DIR, INPUT_FILE)}")
        return

    # Check for required columns
    required_cols = [KMEANS_LABEL_COL] + ORIGINAL_FEATURES
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in the DataFrame: {missing}")
        return
        
    # --- 3. Apply Scaling using Saved StandardScaler ---
    try:
        # Load the fitted StandardScaler object
        scaler = load(SCALER_MODEL_PATH)
        print(f"\nStandardScaler loaded from {SCALER_MODEL_PATH}.")
    except FileNotFoundError:
        print(f"Error: StandardScaler file not found at {SCALER_MODEL_PATH}. Cannot proceed with scaling.")
        return

    # Extract the original feature matrix
    X_original = df[ORIGINAL_FEATURES].values
    
    # Apply the loaded scaler to get the scaled data (X_scaled)
    X_scaled = scaler.transform(X_original)
    
    # K-Means labels for validation
    kmeans_labels = df[KMEANS_LABEL_COL].values
    
    # --- 4. Train the GMM Model ---
    print("\nTraining Gaussian Mixture Model...")
    gmm_model = GaussianMixture(
        n_components=K_CLUSTERS, 
        random_state=RANDOM_SEED, 
        covariance_type='full' 
    )

    gmm_model.fit(X_scaled)
    
    # Get the hard cluster assignments (the most likely cluster)
    gmm_labels = gmm_model.predict(X_scaled)
    
    # Get the soft cluster probabilities (the policy insight)
    gmm_probabilities = gmm_model.predict_proba(X_scaled)
    print("GMM Model training complete.")
    
    # --- 5. Model Validation (ARI) ---
    ari_gmm_kmeans = adjusted_rand_score(kmeans_labels, gmm_labels)
    
    print("\n## Model Validation (GMM vs. K-Means) ##")
    print(f"Adjusted Rand Index (ARI): {ari_gmm_kmeans:.4f}")
    
    # --- 6. Integrate Results and Policy Analysis ---
    
    # Add scaled features and GMM results to the DataFrame
    df_scaled_features = pd.DataFrame(X_scaled, columns=SCALED_FEATURES_RENAMED)
    df = pd.concat([df, df_scaled_features], axis=1) # Add scaled features to the main DataFrame
    df['GMM_Tier'] = gmm_labels
    
    # Create probability columns (P_Tier_0, P_Tier_1, P_Tier_2)
    prob_cols = [f'P_Tier_{i}' for i in range(K_CLUSTERS)]
    df_probabilities = pd.DataFrame(gmm_probabilities, columns=prob_cols)
    # Concatenate the main DataFrame with the new probability columns
    df_results = pd.concat([df.reset_index(drop=True), df_probabilities], axis=1)

    # Identify Borderline Cities (Policy Insight)
    df_results['Is_Borderline'] = False
    
    for index, row in df_results.iterrows():
        gmm_tier = row['GMM_Tier']
        
        # Check if any *other* cluster has a probability >= threshold
        other_probs = [row[col] for i, col in enumerate(prob_cols) if i != gmm_tier]
        
        if other_probs and max(other_probs) >= BORDERLINE_THRESHOLD:
            df_results.loc[index, 'Is_Borderline'] = True
            
    num_borderline = df_results['Is_Borderline'].sum()

    print("\n## Policy Insights (Borderline Cities) ##")
    print(f"Total Borderline Cities Identified: {num_borderline}")
    print("\nExample GMM Output:")
    print(df_results[['City', KMEANS_LABEL_COL, 'GMM_Tier', 'Is_Borderline'] + prob_cols].head())
    
    # --- 7. Save Results and Model Persistence ---
    
    # Save the detailed results to CSV
    output_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)
    df_results.to_csv(output_path, index=False)
    print(f"\nDetailed GMM results saved to: {output_path}")
    
    # Save the fitted GMM model
    if not os.path.exists(DEPLOYMENT_DIR):
        os.makedirs(DEPLOYMENT_DIR)
        
    dump(gmm_model, GMM_MODEL_PATH)
    print(f"Fitted GMM model saved to: {GMM_MODEL_PATH}")

if __name__ == "__main__":
    train_and_validate_gmm()