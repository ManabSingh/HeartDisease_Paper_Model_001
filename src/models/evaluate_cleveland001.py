import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

def evaluate_cleveland_source():
    # 2. LOAD DATA
    print("--- Loading Cleveland Source Data ---")
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. LOAD THE STACKING ENSEMBLE
    print("--- Loading Trained Stacking Ensemble ---")
    model_path = os.path.join(model_dir, 'stacking_ensemble.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 4. GENERATE "HONEST" PREDICTIONS VIA CROSS-VALIDATION
    # We use CV here so we don't report inflated "training accuracy"
    print("Performing Stratified 5-Fold Cross-Validation on Cleveland...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict', n_jobs=-1)
    y_probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

    # 5. CALCULATE METRICS
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_probs)
    cm = confusion_matrix(y, y_pred)

    # 6. PRINT RESULTS TABLE
    print("\n" + "="*50)
    print("       CLEVELAND (SOURCE DOMAIN) PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    print("="*50)

if __name__ == "__main__":
    evaluate_cleveland_source()