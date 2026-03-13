import pandas as pd
import os
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
figures_dir = os.path.join(BASE_DIR, 'results', 'explainability')
os.makedirs(figures_dir, exist_ok=True)

def generate_shap_plots():
    # 2. LOAD & CLEAN DATA
    print("--- Loading Cleveland Data ---")
    df = pd.read_csv(train_path)
    X = df.drop("target", axis=1).astype(float)

    # 3. LOAD MODELS
    print("--- Loading Tuned Base Models ---")
    model_files = {
        "XGBoost": "xgboost_tuned.pkl",
        "LightGBM": "lightgbm_tuned.pkl",
        "RandomForest": "rf_tuned.pkl",
    }

    for model_name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path): continue

        print(f"\nProcessing {model_name}...")
        with open(path, "rb") as f:
            model = pickle.load(f)

        try:
            # ==========================================
            # 4. THE WRAPPER LOGIC (Fixes "can't set attribute")
            # ==========================================
            if model_name == "XGBoost":
                print("  Using Lambda Wrapper + KernelExplainer...")
                # We wrap the prediction call so SHAP cannot see the XGBoost object
                # and therefore cannot try to modify its attributes.
                model_predict = lambda x: model.predict_proba(x)
                
                # Summarize data for speed
                background = shap.kmeans(X, 10) 
                explainer = shap.KernelExplainer(model_predict, background)
                
                # Explaining 40 samples (Perfect for a research plot density)
                X_sample = X.iloc[:40]
                shap_values = explainer.shap_values(X_sample)
                X_plot = X_sample
            else:
                # TreeExplainer is safe for these
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                X_plot = X

            # ==========================================
            # 5. STANDARDIZATION
            # ==========================================
            if isinstance(shap_values, list):
                final_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif len(shap_values.shape) == 3:
                final_vals = shap_values[:, :, 1]
            else:
                final_vals = shap_values

            # ==========================================
            # 6. PLOTTING
            # ==========================================
            plt.figure(figsize=(10, 6))
            shap.summary_plot(final_vals, X_plot, show=False)
            plt.title(f"SHAP Importance: {model_name}", fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(figures_dir, f"shap_{model_name.lower()}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SUCCESS: Plot saved for {model_name}")

        except Exception as e:
            print(f"CRITICAL ERROR for {model_name}: {e}")

    print("\n" + "=" * 55)
    print(f"XAI COMPLETE. Plots saved in: {figures_dir}")
    print("=" * 55)

if __name__ == "__main__":
    generate_shap_plots()