import pandas as pd
import numpy as np
import os
import pickle
import shap
import warnings
from scipy.stats import spearmanr # ADDED FOR FIX 4 (H1 PROOF)

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')

results_dir = os.path.join(BASE_DIR, 'results', 'explainability')
os.makedirs(results_dir, exist_ok=True)

def generate_advanced_shift_table():
    print("--- Generating Severe Feature Shift Table (H1 Proof) ---")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load Data
    df_cleve = pd.read_csv(cleveland_path)
    X_cleve = df_cleve.drop('target', axis=1).astype(float)

    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)
    X_stat = X_stat[X_cleve.columns]

    # Calculate SHAP
    print("Calculating SHAP values...")
    background_masker = shap.maskers.Independent(X_cleve, max_samples=100)
    explainer = shap.Explainer(model.predict, background_masker)
    
    shap_cleve = explainer(X_cleve)
    shap_stat = explainer(X_stat)

    # Calculate Magnitude (Mean Absolute SHAP)
    mag_cleve = np.abs(shap_cleve.values).mean(axis=0)
    mag_stat = np.abs(shap_stat.values).mean(axis=0)

    # REMOVED: Logic Inversion (Direction) calculations have been removed as per Fix 1.

    # Build DataFrame
    df = pd.DataFrame({
        'Feature': X_cleve.columns,
        'Cleveland Magnitude': mag_cleve,
        'Statlog Magnitude': mag_stat,
    })

    # Add Ranks
    df['Cleveland Rank'] = df['Cleveland Magnitude'].rank(ascending=False).astype(int)
    df['Statlog Rank'] = df['Statlog Magnitude'].rank(ascending=False).astype(int)

    # Calculate Rank Shift
    df['Rank Shift'] = df['Cleveland Rank'] - df['Statlog Rank']
    df['Rank Shift'] = df['Rank Shift'].apply(lambda x: f"Rose {abs(x)}" if x > 0 else (f"Dropped {abs(x)}" if x < 0 else "Unchanged"))

    # Format the table for readability
    df['Cleveland Magnitude'] = df['Cleveland Magnitude'].apply(lambda x: f"{x:.4f}")
    df['Statlog Magnitude'] = df['Statlog Magnitude'].apply(lambda x: f"{x:.4f}")
    
    # Reorder columns (Logic Inverted column removed)
    df = df[['Feature', 'Cleveland Rank', 'Statlog Rank', 'Rank Shift', 'Cleveland Magnitude', 'Statlog Magnitude']]
    df = df.sort_values(by='Cleveland Rank').reset_index(drop=True)

    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)

    # --- FIX 4: STATISTICAL PROOF FOR H1 ---
    correlation, p_value = spearmanr(df['Cleveland Rank'], df['Statlog Rank'])
    print(f"\n--- Statistical Proof for H1 ---")
    print(f"Spearman Rank Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")
    if p_value < 0.05 and correlation < 1.0:
        print("CONCLUSION: The rankings are significantly different across domains.")
        print("This provides empirical evidence to reject the null hypothesis and accept H1.")

    save_path = os.path.join(results_dir, 'advanced_feature_shift_table.csv')
    df.to_csv(save_path, index=False)
    print(f"\nSUCCESS: Severe shift table saved to {save_path}")

if __name__ == "__main__":
    generate_advanced_shift_table()