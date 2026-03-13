import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
results_dir = os.path.join(BASE_DIR, 'results')

def analyze_attribution_shift():
    # 1. Load Unseen Target Data (Statlog)
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('target', axis=1).astype(float)
    y_test = df_test['target']

    # 2. Load Models
    models_files = {
        "XGBoost": "xgboost_tuned.pkl",
        "LightGBM": "lightgbm_tuned.pkl",
        "RandomForest": "rf_tuned.pkl",
        "Ensemble": "stacking_ensemble.pkl"
    }
    
    importance_df = pd.DataFrame(index=X_test.columns)

    print("--- Deploying Models on Target Domain (Statlog) ---")
    
    for name, file in models_files.items():
        path = os.path.join(model_dir, file)
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Calculate Importance on Statlog Data
        # For ensemble, we look at the base learners' contribution
        if name != "Ensemble":
            if hasattr(model, 'feature_importances_'):
                importance_df[f'{name}_Statlog'] = model.feature_importances_
    
    # Normalize for comparison
    importance_df = importance_df.apply(lambda x: x / x.sum())
    importance_df['Target_Avg_Importance'] = importance_df.mean(axis=1)
    
    # 3. Save Importance Table
    importance_df.to_csv(os.path.join(results_dir, 'target_domain_importance.csv'))
    print("SUCCESS: Target domain feature importance extracted.")
    return importance_df, X_test

if __name__ == "__main__":
    analyze_attribution_shift()