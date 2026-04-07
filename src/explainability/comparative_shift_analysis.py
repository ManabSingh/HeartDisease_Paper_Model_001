import pandas as pd
import numpy as np
import os
import pickle
import shap
import warnings
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')

xgb_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'xgboost_tuned.pkl')
ensemble_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')

results_dir = os.path.join(BASE_DIR, 'results', 'explainability')
os.makedirs(results_dir, exist_ok=True)

# ==========================================
# 2. NON-LINEAR DIRECTION CHECK (QUARTILE METHOD)
# ==========================================
def get_nonlinear_direction(feature_values, shap_vals):
    """Determines if a feature drives predictions up (Positive) or down (Negative) at its extremes."""
    q75 = np.percentile(feature_values, 75)
    q25 = np.percentile(feature_values, 25)
    
    # Handle binary/categorical features where q75 might equal q25
    if q75 == q25:
        mean_high = shap_vals[feature_values >= q75].mean()
        mean_low = shap_vals[feature_values < q75].mean() if len(feature_values[feature_values < q75]) > 0 else 0
    else:
        mean_high = shap_vals[feature_values >= q75].mean()
        mean_low = shap_vals[feature_values <= q25].mean()
        
    return "Positive" if mean_high > mean_low else "Negative"

# ==========================================
# 3. CORE ANALYSIS FUNCTION
# ==========================================
def analyze_model_shift(model_path, model_name, X_cleve, X_stat):
    print(f"\n[{model_name}] Loading and computing SHAP values...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Calculate SHAP
    background_masker = shap.maskers.Independent(X_cleve, max_samples=100)
    explainer = shap.Explainer(model.predict, background_masker)
    
    shap_cleve = explainer(X_cleve)
    shap_stat = explainer(X_stat)

    # 1. Magnitude & Rank
    mag_cleve = np.abs(shap_cleve.values).mean(axis=0)
    mag_stat = np.abs(shap_stat.values).mean(axis=0)

    df = pd.DataFrame({'Feature': X_cleve.columns})
    df['Cleveland Rank'] = pd.Series(mag_cleve).rank(ascending=False).astype(int)
    df['Statlog Rank'] = pd.Series(mag_stat).rank(ascending=False).astype(int)
    
    # Calculate Rank Shift
    df['Rank Shift Formatted'] = df['Cleveland Rank'] - df['Statlog Rank']
    df['Rank Shift Formatted'] = df['Rank Shift Formatted'].apply(lambda x: f"Rose {abs(x)}" if x > 0 else (f"Dropped {abs(x)}" if x < 0 else "Unchanged"))

    # 2. Logic Inversion Check
    dir_cleve = [get_nonlinear_direction(X_cleve.iloc[:, i], shap_cleve.values[:, i]) for i in range(X_cleve.shape[1])]
    dir_stat = [get_nonlinear_direction(X_stat.iloc[:, i], shap_stat.values[:, i]) for i in range(X_stat.shape[1])]
    
    df['Cleveland Direction'] = dir_cleve
    df['Statlog Direction'] = dir_stat
    df['Logic Inverted?'] = np.where(df['Cleveland Direction'] != df['Statlog Direction'], 'Yes', 'No')

    # 3. Format and Sort
    df['Cleveland Mag'] = [f"{x:.4f}" for x in mag_cleve]
    df['Statlog Mag'] = [f"{x:.4f}" for x in mag_stat]
    
    df = df[['Feature', 'Cleveland Rank', 'Statlog Rank', 'Rank Shift Formatted', 'Cleveland Mag', 'Statlog Mag', 'Cleveland Direction', 'Statlog Direction', 'Logic Inverted?']]
    df = df.sort_values(by='Cleveland Rank').reset_index(drop=True)

    # 4. Statistical Proof
    correlation, p_value = spearmanr(df['Cleveland Rank'], df['Statlog Rank'])
    inversions = (df['Logic Inverted?'] == 'Yes').sum()
    
    print(f"--- {model_name} Results ---")
    print(f"Spearman Correlation (Stability): {correlation:.4f}")
    print(f"Logic Inversions Detected: {inversions}")
    
    # 5. Output to Console and CSV
    print(df.to_string(index=False))
    save_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_shift_analysis.csv')
    df.to_csv(save_path, index=False)
    print(f"\nSaved detailed table to: {save_path}")
    
    return correlation, inversions, df

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def run_comparative_analysis():
    print("="*80)
    print("STARTING COMPARATIVE ATTRIBUTION SHIFT ANALYSIS")
    print("="*80)

    # Load Data
    df_cleve = pd.read_csv(cleveland_path)
    X_cleve = df_cleve.drop('target', axis=1).astype(float)
    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)[X_cleve.columns] # Ensure column order matches

    # 1. Analyze The Problem (XGBoost)
    xgb_corr, xgb_inv, xgb_df = analyze_model_shift(xgb_path, "Standalone XGBoost", X_cleve, X_stat)

    # 2. Analyze The Solution (Ensemble)
    ens_corr, ens_inv, ens_df = analyze_model_shift(ensemble_path, "Stacking Ensemble", X_cleve, X_stat)

    print("\n" + "="*80)
    print("FINAL PAPER METRICS (COPY THESE FOR YOUR RESULTS SECTION)")
    print("="*80)
    print("THE PROBLEM (Standalone XGBoost):")
    print(f"- Feature Stability (Spearman ρ): {xgb_corr:.4f}")
    print(f"- Number of Logic Inversions: {xgb_inv}")
    if xgb_inv > 0:
        inverted_feats = xgb_df[xgb_df['Logic Inverted?'] == 'Yes']['Feature'].tolist()
        print(f"- Inverted Features to highlight in paper: {', '.join(inverted_feats)}")
    
    print("\nTHE SOLUTION (Stacking Ensemble):")
    print(f"- Feature Stability (Spearman ρ): {ens_corr:.4f}")
    print(f"- Number of Logic Inversions: {ens_inv}")
    print("="*80)

if __name__ == "__main__":
    run_comparative_analysis()