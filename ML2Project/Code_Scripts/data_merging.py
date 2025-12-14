import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Load All Datasets (Using correct names from current directory) ---
print("1. Loading raw data...")
df_census = pd.read_csv("Raw_Data/CombinedData(census).csv")
df_cities_states = pd.read_csv("Raw_Data/Indian_Cities_States(Sheet1).csv")
df_residex = pd.read_csv("Raw_Data/CombinedData(Residex_Data).csv")
df_cpi = pd.read_csv("Raw_Data/CombinedData(Consumer Price Index).csv")
df_income = pd.read_excel("Raw_Data/Per_Capita_Income.xlsx")


# --- 2. Filter Income Data (Current Prices, Latest Year) ---
latest_year = df_income[df_income['Price Category'] == 'current']['Year'].max()
df_income_latest = df_income[
    (df_income['Price Category'] == 'current') &
    (df_income['Year'] == latest_year)
].copy()
df_income_latest.rename(
    columns={'Price (in ₹)': 'Per_Capita_Income_Latest_Current'},
    inplace=True
)
df_income_latest = df_income_latest[['State', 'Per_Capita_Income_Latest_Current']]


# --- 3. Clean and Standardize Names for Merging ---

# --- City Name Cleaning ---
df_census['UA_Name_Clean'] = df_census['UA Name'].str.replace(r' \(.*\)', '', regex=True).str.strip()
df_residex.rename(columns={'City': 'UA_Name_Clean'}, inplace=True)
df_residex['UA_Name_Clean'] = df_residex['UA_Name_Clean'].str.replace(r' \(.*\)', '', regex=True).str.strip()
df_residex = df_residex[['UA_Name_Clean', 'Latest HPI', 'yoy_change']] 

# --- State Name Standardization Function (Robust Fix for NaN/float issues) ---
def standardize_state(state):
    if pd.isna(state):
        return state
    # Robustly convert to string before operations
    state = str(state).upper().replace(' & ', ' AND ').replace('.', '').replace(' (TRICITY)', '').strip()
    
    # Specific standardizations
    if 'ANDAMAN' in state: return 'ANDAMAN AND NICOBAR ISLANDS'
    if 'CHANDIGARH' in state: return 'CHANDIGARH'
    if 'DELHI' in state: return 'DELHI'
    if 'UTTAR PRADESH' in state: return 'UTTAR PRADESH'
    if 'WEST BENGAL' in state: return 'WEST BENGAL'
    
    return state

# --- Apply State Standardization ---
df_cities_states.rename(columns={'State / Union Territory': 'State_Standard'}, inplace=True)
df_cities_states['State_Standard'] = df_cities_states['State_Standard'].apply(standardize_state)
df_income_latest.rename(columns={'State': 'State_Standard'}, inplace=True)
df_income_latest['State_Standard'] = df_income_latest['State_Standard'].apply(standardize_state)
df_cpi.rename(columns={'State / Union Teritory': 'State_Standard'}, inplace=True)
df_cpi['State_Standard'] = df_cpi['State_Standard'].apply(standardize_state)
df_cpi.columns = df_cpi.columns.str.strip() # Fix for hidden whitespace in CPI columns


# --- 4. Perform Left Merges (Keep All Cities) ---

print("2. Merging dataframes...")
# Use df_cities_states as the base to keep ALL city rows
df_merged = pd.merge(df_cities_states, df_census,
                     left_on='City', right_on='UA_Name_Clean', how='left', suffixes=('_Map', '_Census'))

# Merge 2: with HPI (City-level merge)
df_merged = pd.merge(df_merged, df_residex,
                     on='UA_Name_Clean', how='left')

# Merge 3 & 4: with Income and CPI (State-level merges)
df_merged = pd.merge(df_merged, df_income_latest,
                     on='State_Standard', how='left')
df_merged = pd.merge(df_merged, df_cpi,
                     on='State_Standard', how='left')

# --- 5. Feature Engineering (P/I Ratio and Affordability Features) ---

print("3. Engineering core features...")
df_merged['Latest HPI'] = pd.to_numeric(df_merged['Latest HPI'], errors='coerce')
df_merged['Per_Capita_Income_Latest_Current'] = pd.to_numeric(
    df_merged['Per_Capita_Income_Latest_Current'], errors='coerce')

# Calculate Price-to-Income (P/I) Ratio - NaN here if HPI or Income is NaN
df_merged['Price_to_Income_Ratio'] = (
    df_merged['Latest HPI'] / (df_merged['Per_Capita_Income_Latest_Current'] / 100000)
)

# Calculate Affordability Change Pressure - NaN here if HPI or yoy_change is NaN
df_merged['yoy_change'] = pd.to_numeric(df_merged['yoy_change'], errors='coerce')
df_merged['Affordability_Change_Pressure'] = df_merged['Latest HPI'] * (df_merged['yoy_change'] / 100)


# --- 6. Final Data Selection for Clustering (with imputation preparation) ---

clustering_features = [
    'City', # Use the base City name
    'State_Standard',
    'Latest HPI',
    'yoy_change',
    'CPI_Housing_Index',
    'P_LIT',
    'TOT_WORK_P',
    'Literacy_rate',
    'Worker Participation Rate',
    'SC/ST Population Percentage',
    'Households per Capita',
    'Per_Capita_Income_Latest_Current',
    'Price_to_Income_Ratio',
    'Affordability_Change_Pressure'
]

df_final = df_merged[clustering_features].copy()
numerical_cols = df_final.select_dtypes(include=np.number).columns.tolist()


# --- 7. Imputation using Median ---

print("4. Imputing missing numerical values with the column median...")
# We use the median as it's more robust to potential outliers than the mean.
imputer = SimpleImputer(strategy='median') 

# Fit and transform the numerical features
# This will fill all NaNs in the selected columns with the column's median
df_final[numerical_cols] = imputer.fit_transform(df_final[numerical_cols])


# --- 8. Final Verification and Export ---

print("\n--- Final Imputed Dataset Information ---")
print(f"Total cities prepared: {len(df_final)}")
print(df_final.info())
print("\n--- Final Imputed Dataset Head ---")
print(df_final.head(10))

# Export the final imputed data for clustering
df_final.to_csv("Processed_Data/Housing_Affordability_Imputed_Data_50_Cities.csv", index=False)
print("\nSUCCESS: Data saved to Housing_Affordability_Imputed_Data_50_Cities.csv")