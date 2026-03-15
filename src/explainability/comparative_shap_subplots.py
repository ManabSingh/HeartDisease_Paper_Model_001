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

def generate_side_by_side_shap():
    print("--- Generating Side-by-Side SHAP Attribution Plots ---")
    
    # 2. LOAD MODEL
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 3. LOAD DATA & FIX THE NUMPY ERROR
    print("Loading datasets and converting to float...")
    
    # Source Domain (Background Masker & Baseline)
    df_cleve = pd.read_csv(cleveland_path)
    X_cleve = df_cleve.drop('target', axis=1).astype(float) # <-- ERROR FIX APPLIED HERE

    # Target Domain (Evaluation)
    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)   # <-- ERROR FIX APPLIED HERE
    
    # Ensure column order matches exactly
    X_stat = X_stat[X_cleve.columns]

    # 4. INITIALIZE EXPLAINER
    print("Calculating SHAP values (This may take a moment)...")
    background_masker = shap.maskers.Independent(X_cleve, max_samples=100)
    explainer = shap.Explainer(model.predict, background_masker)
    
    # Calculate values for both domains
    shap_values_cleve = explainer(X_cleve)
    shap_values_stat = explainer(X_stat)

    # 5. PLOT SIDE-BY-SIDE
    print("Generating comparative figure...")
    
    # Create a wide figure to hold two plots
    fig = plt.figure(figsize=(18, 8))
    
    # --- SUBPLOT 1: CLEVELAND ---
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    # plot_size=None is critical here so SHAP doesn't override the plt.figure size
    shap.summary_plot(shap_values_cleve, X_cleve, show=False, plot_size=None, plot_type="dot")
    plt.title("Source Domain (Cleveland): Feature Attribution", fontsize=15, pad=15)
    plt.xlabel("SHAP value (Impact on model output)")

    # --- SUBPLOT 2: STATLOG ---
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    shap.summary_plot(shap_values_stat, X_stat, show=False, plot_size=None, plot_type="dot")
    plt.title("Target Domain (Statlog): Feature Attribution", fontsize=15, pad=15)
    plt.xlabel("SHAP value (Impact on model output)")

    # 6. FORMAT AND SAVE
    # Add a main title for the whole image
    plt.suptitle("Cross-Domain Attribution Shift Analysis (Stacking Ensemble)", fontsize=18, fontweight='bold')
    
    # Adjust layout so the titles don't overlap with the graphs
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.3) 
    
    save_path = os.path.join(results_dir, 'comparative_shap_subplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSUCCESS: Comparative SHAP image saved to {save_path}")

if __name__ == "__main__":
    generate_side_by_side_shap()