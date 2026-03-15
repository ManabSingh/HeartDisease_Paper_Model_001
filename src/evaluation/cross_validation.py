import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import StratifiedKFold, cross_validate

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')

def run_cross_validation():
    print("--- Running 5-Fold Cross-Validation on Source Domain (Cleveland) ---")
    
    # Load Data
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1).astype(float)
    y = df['target']

    # Load Model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Setup CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Run CV
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    # Print Results for Paper
    print("\n" + "="*45)
    print(" CLEVELAND CROSS-VALIDATION RESULTS (5-FOLD)")
    print("="*45)
    print(f"Accuracy:  {np.mean(scores['test_accuracy']):.4f} (+/- {np.std(scores['test_accuracy']):.4f})")
    print(f"Precision: {np.mean(scores['test_precision']):.4f} (+/- {np.std(scores['test_precision']):.4f})")
    print(f"Recall:    {np.mean(scores['test_recall']):.4f} (+/- {np.std(scores['test_recall']):.4f})")
    print(f"F1-Score:  {np.mean(scores['test_f1']):.4f} (+/- {np.std(scores['test_f1']):.4f})")
    print(f"AUC-ROC:   {np.mean(scores['test_roc_auc']):.4f} (+/- {np.std(scores['test_roc_auc']):.4f})")
    print("="*45)

if __name__ == "__main__":
    run_cross_validation()