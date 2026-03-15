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
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'explainability')
os.makedirs(results_dir, exist_ok=True)

def plot_target_shap():
    print("--- Generating SHAP Attribution for Target (Statlog) ---")
    
    # 2. LOAD MODEL & DATA
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load Source for Background Masker
    df_cleve = pd.read_csv(cleveland_path)
    # FIX: Added .astype(float) here to prevent the numpy TypeError
    X_cleve = df_cleve.drop('target', axis=1).astype(float)

    # Load Target for Evaluation
    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)
    
    # Ensure column order matches the model's training configuration exactly
    X_stat = X_stat[X_cleve.columns]

    # 3. INITIALIZE EXPLAINER
    print("Calculating SHAP values...")
    # Background MUST remain Cleveland so we can see how the logic changes relative to training
    background_masker = shap.maskers.Independent(X_cleve, max_samples=100)
    explainer = shap.Explainer(model.predict, background_masker)
    
    # 4. GET VALUES & PLOT (Applying to Statlog)
    shap_values_stat = explainer(X_stat)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_stat, X_stat, show=False, plot_type="dot")
    plt.title("Target Domain (Statlog): SHAP Feature Attributions", fontsize=14, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'shap_target_statlog.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SUCCESS: Target SHAP saved to {save_path}")

if __name__ == "__main__":
    plot_target_shap()