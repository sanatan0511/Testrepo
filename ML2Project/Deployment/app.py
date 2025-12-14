from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os
import sys

# --- A. GLOBAL CONSTANTS ---

FEATURE_COLS = [
    'Latest HPI', 'yoy_change', 'CPI_Housing_Index', 'P_LIT', 'TOT_WORK_P',
    'Literacy_rate', 'Worker Participation Rate', 'SC/ST Population Percentage',
    'Households per Capita', 'Per_Capita_Income_Latest_Current',
    'Price_to_Income_Ratio', 'Affordability_Change_Pressure'
]

POLICY_TIERS = {
    0: 'Tier 3: Stable Growth Engines (Focus: Management & Preservation)',
    1: 'Tier 1: Extreme Crisis Intervention (Focus: Speculation Control)',
    2: 'Tier 2: Affordability Risk & Socio-Economic Focus (Focus: Social Upliftment)'
}

# --- B. FLASK INITIALIZATION ---

# Get absolute path for joblib files and templates
current_dir = os.path.dirname(os.path.abspath(__file__)) 
app = Flask(__name__, template_folder=os.path.join(current_dir, 'templates')) 

# --- C. ROUTES ---

def load_models_for_request():
    """Helper function to load models safely within the function scope."""
    MODEL_FILENAME = os.path.join(current_dir, 'kmeans_model.joblib')
    SCALER_FILENAME = os.path.join(current_dir, 'scaler.joblib')
    
    try:
        # Load models every time the function is called (less efficient, but resolves the error)
        kmeans_model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        return kmeans_model, scaler
    except FileNotFoundError:
        # If files are missing, raise an error that will be displayed on the webpage
        raise FileNotFoundError("ERROR: Model files not found. Check Deployment folder.")

@app.route('/', methods=['GET'])
def home():
    # Attempt to load models once to check if they exist before serving the page
    try:
        load_models_for_request() 
        return render_template('index.html', feature_names=FEATURE_COLS)
    except FileNotFoundError as e:
        # Show a critical error message directly on the page if files are missing
        return f"<h1>Deployment Error: {e}</h1><p>Check console for detailed path information.</p>", 500

@app.route('/images', methods=['GET'])
def show_images():
    return render_template('images.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load models and scaler inside the prediction function
        kmeans_model, scaler = load_models_for_request() 

        # 1. Prepare Data
        form_data = request.form
        input_data_list = [float(form_data[col]) for col in FEATURE_COLS]
        input_df = pd.DataFrame([input_data_list], columns=FEATURE_COLS)

        # 2. Predict
        input_scaled = scaler.transform(input_df)
        cluster_label = kmeans_model.predict(input_scaled)[0]
        
        # 3. Map Result
        policy_tier_name = POLICY_TIERS.get(cluster_label, 'Error: Unknown Cluster')

        return render_template(
            'index.html', 
            feature_names=FEATURE_COLS, 
            prediction_result=policy_tier_name,
            input_values=form_data
        )

    except Exception as e:
        return render_template('index.html', feature_names=FEATURE_COLS, prediction_result=f'Prediction Error: {e}')


if __name__ == '__main__':
    # Start the application
    print("Starting Flask application...")
    app.run(debug=True)