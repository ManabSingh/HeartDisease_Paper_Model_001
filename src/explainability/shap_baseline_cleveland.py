import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'explainability')
os.makedirs(results_dir, exist_ok=True)

def plot_baseline_shap():
    print("--- Generating SHAP Attribution for Baseline (Cleveland) ---")
    
    # 2. LOAD MODEL & DATA
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    df_cleve = pd.read_csv(cleveland_path)
    
    # FIX: Explicitly cast the dataframe to float to prevent numpy TypeError
    X_cleve = df_cleve.drop('target', axis=1).astype(float)

    # 3. INITIALIZE EXPLAINER
    print("Calculating SHAP values...")
    background_masker = shap.maskers.Independent(X_cleve, max_samples=100)
    explainer = shap.Explainer(model.predict, background_masker)
    
    # 4. GET VALUES & PLOT
    shap_values_cleve = explainer(X_cleve)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_cleve, X_cleve, show=False, plot_type="dot")
    plt.title("Baseline Domain (Cleveland): SHAP Feature Attributions", fontsize=14, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'shap_baseline_cleveland.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SUCCESS: Baseline SHAP saved to {save_path}")

if __name__ == "__main__":
    plot_baseline_shap()