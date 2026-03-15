import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def get_metrics(y_true, y_pred, y_probs):
    """Helper function to calculate all standard clinical metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_probs)
    }

def generate_comparative_table():
    print("--- Generating Cross-Domain Comparative Table ---")
    
    # Load Model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ---------------------------------------------------------
    # PART A: SOURCE DOMAIN (CLEVELAND) via Cross-Validation
    # ---------------------------------------------------------
    print("Evaluating Source Domain (Cleveland)...")
    df_cleve = pd.read_csv(cleveland_path)
    X_cleve = df_cleve.drop('target', axis=1)
    y_cleve = df_cleve['target']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cleve = cross_val_predict(model, X_cleve, y_cleve, cv=cv, method='predict', n_jobs=-1)
    y_probs_cleve = cross_val_predict(model, X_cleve, y_cleve, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    cleveland_metrics = get_metrics(y_cleve, y_pred_cleve, y_probs_cleve)

    # ---------------------------------------------------------
    # PART B: TARGET DOMAIN (STATLOG) via Direct Inference
    # ---------------------------------------------------------
    print("Evaluating Target Domain (Statlog)...")
    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)
    y_stat = df_stat['target']
    
    y_pred_stat = model.predict(X_stat)
    y_probs_stat = model.predict_proba(X_stat)[:, 1]
    
    statlog_metrics = get_metrics(y_stat, y_pred_stat, y_probs_stat)

    # ---------------------------------------------------------
    # PART C: BUILD THE COMPARATIVE DATAFRAME
    # ---------------------------------------------------------
    # Combine dictionaries into a single DataFrame
    comparison_df = pd.DataFrame({
        'Metric': list(cleveland_metrics.keys()),
        'Source: Cleveland': list(cleveland_metrics.values()),
        'Target: Statlog': list(statlog_metrics.values())
    })

    # Calculate the exact drop/shift in performance
    comparison_df['Shift (Difference)'] = comparison_df['Target: Statlog'] - comparison_df['Source: Cleveland']
    
    # Format the columns for readability (rounding to 4 decimal places)
    comparison_df['Source: Cleveland'] = comparison_df['Source: Cleveland'].apply(lambda x: f"{x:.4f}")
    comparison_df['Target: Statlog'] = comparison_df['Target: Statlog'].apply(lambda x: f"{x:.4f}")
    
    # Format the shift as a percentage for impact (e.g., "-5.23%")
    comparison_df['Shift (Difference)'] = comparison_df['Shift (Difference)'].apply(lambda x: f"{x * 100:+.2f}%")

    # ---------------------------------------------------------
    # PART D: DISPLAY AND SAVE
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("   CROSS-DOMAIN GENERALIZATION METRICS (STACKING ENSEMBLE)")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)

    # Save to CSV
    save_path = os.path.join(results_dir, 'cross_domain_comparison_table.csv')
    comparison_df.to_csv(save_path, index=False)
    print(f"\nSaved publication-ready table to: {save_path}")

if __name__ == "__main__":
    generate_comparative_table()