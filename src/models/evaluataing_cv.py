import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

def run_cross_validation():
    # ==========================================
    # 2. LOAD SOURCE DATASET
    # ==========================================
    print("--- Loading Cleveland Data ---")
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # ==========================================
    # 3. LOAD ALL SAVED MODELS
    # ==========================================
    models = {}
    try:
        with open(os.path.join(model_dir, 'xgboost_tuned.pkl'), 'rb') as f:
            models['Tuned XGBoost'] = pickle.load(f)
        with open(os.path.join(model_dir, 'lightgbm_tuned.pkl'), 'rb') as f:
            models['Tuned LightGBM'] = pickle.load(f)
        with open(os.path.join(model_dir, 'rf_tuned.pkl'), 'rb') as f:
            models['Tuned Random Forest'] = pickle.load(f)
        with open(os.path.join(model_dir, 'stacking_ensemble.pkl'), 'rb') as f:
            models['Stacking Ensemble'] = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading a model. Did you run all training scripts? Details: {e}")
        return

    # ==========================================
    # 4. PERFORM 5-FOLD CROSS VALIDATION
    # ==========================================
    # StratifiedKFold ensures the ratio of sick/healthy patients is the same in every fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*50)
    print("  5-FOLD CROSS-VALIDATION RESULTS (CLEVELAND)")
    print("="*50)
    print(f"{'Model Name':<25} | {'Mean Accuracy':<15} | {'Std Dev (+/-)'}")
    print("-" * 60)

    for name, model in models.items():
        # cross_val_score automatically splits the data, trains, and tests 5 times
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        
        # A low Standard Deviation means the model is very stable and not overfitting
        print(f"{name:<25} | {mean_score:.4f}          | +/- {std_dev:.4f}")
        
    print("="*60)
    print("Note: A smaller Std Dev indicates higher reliability across different data splits.")

if __name__ == "__main__":
    run_cross_validation()