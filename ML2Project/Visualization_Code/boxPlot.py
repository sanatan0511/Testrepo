import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: The file name used here should be the one containing the final K=3 labels.
# Assuming 'Housing_Affordability_Clustered_Data_K3.csv' is correct.
df_imputed = pd.read_csv("Processed_Data/Housing_Affordability_Clustered_Data_K3.csv")

# Create a mapping for clear policy labels
tier_map = {
    # Replace 'KM_Cluster' with the actual column name used in your CSV (likely 'Cluster')
    0: 'Tier 3: Stable Growth',
    1: 'Tier 1: Extreme Crisis',
    2: 'Tier 2: Affordability Risk'
}

# --- FIX APPLIED HERE: Using 'Cluster' instead of 'KM_Cluster' ---
df_imputed['Policy_Tier'] = df_imputed['Cluster'].map(tier_map)
# Also need to use the actual cluster column for sorting
df_imputed['KM_Cluster'] = df_imputed['Cluster'] # Create KM_Cluster alias for sorting

plt.figure(figsize=(10, 6))
# Plotting against the actual cluster column (KM_Cluster alias)
sns.boxplot(
    x='Policy_Tier',
    y='Price_to_Income_Ratio',
    data=df_imputed.sort_values(by='KM_Cluster'),
    palette=['#1f77b4', '#ff7f0e', '#2ca02c'] # Assign colors to tiers
)
plt.title('Distribution of Price-to-Income Ratio by Policy Tier (K=3)', fontsize=14)
plt.xlabel('Policy Tier Designation', fontsize=12)
plt.ylabel('Price-to-Income Ratio (P/I)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('image/box_plot_pi_ratio_by_tier.png')