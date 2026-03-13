import pandas as pd
import numpy as np
import os
import pickle

# ==========================================
# 1. DYNAMIC PATH SETUP (Based on your folder tree)
# ==========================================
# This gets the 'HeartDisease_Paper_Model_001' root folder automatically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
# Saving the table into your 'results' folder
results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

def export_feature_rankings():
    print(f"--- Accessing Root: {BASE_DIR} ---")
    
    # 2. LOAD DATA
    if not os.path.exists(train_path):
        print(f"Error: Could not find data at {train_path}")
        return
    
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1).astype(float)
    
    # 3. DEFINE MODELS TO RANK
    models_to_check = {
        "XGBoost": "xgboost_tuned.pkl",
        "LightGBM": "lightgbm_tuned.pkl",
        "RandomForest": "rf_tuned.pkl"
    }

    ranking_summary = pd.DataFrame(index=X.columns)

    # 4. EXTRACT IMPORTANCE
    for name, file in models_to_check.items():
        path = os.path.join(model_dir, file)
        if not os.path.exists(path):
            print(f"Skipping {name}: Model file not found.")
            continue
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Get feature importance (normalized)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            ranking_summary[name] = importances / np.sum(importances)

    # 5. CALCULATE ENSEMBLE AVERAGE
    ranking_summary['Ensemble_Average'] = ranking_summary.mean(axis=1)
    ranking_summary = ranking_summary.sort_values(by='Ensemble_Average', ascending=False)

    # 6. SAVE TO RESULTS
    save_path = os.path.join(results_dir, 'feature_importance_table.csv')
    ranking_summary.to_csv(save_path)
    
    print("\n" + "="*40)
    print("   TOP 5 DIAGNOSTIC FEATURES")
    print("="*40)
    print(ranking_summary['Ensemble_Average'].head(5))
    print("="*40)
    print(f"Table saved to: {save_path}")

if __name__ == "__main__":
    export_feature_rankings()