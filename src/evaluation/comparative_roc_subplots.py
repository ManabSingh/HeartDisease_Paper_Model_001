import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def plot_side_by_side_roc():
    print("--- Generating Side-by-Side Comparative ROC Curves ---")
    
    # 2. LOAD DATA
    # Source Domain (Cleveland)
    df_cleve = pd.read_csv(cleveland_path)
    X_cleve = df_cleve.drop('target', axis=1)
    y_cleve = df_cleve['target']

    # Target Domain (Statlog)
    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)
    y_stat = df_stat['target']

    # 3. MODELS TO EVALUATE
    models_to_evaluate = {
        'XGBoost': 'xgboost_tuned.pkl',
        'LightGBM': 'lightgbm_tuned.pkl',
        'Random Forest': 'rf_tuned.pkl',
        'Stacking Ensemble': 'stacking_ensemble.pkl'
    }
    
    colors = ['blue', 'green', 'purple', 'red']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. INITIALIZE SUBPLOTS
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Cross-Domain Generalization: ROC Curves Comparison', fontsize=18, fontweight='bold', y=1.02)

    # 5. GENERATE CURVES FOR EACH MODEL
    for (name, file), color in zip(models_to_evaluate.items(), colors):
        path = os.path.join(model_dir, file)
        if not os.path.exists(path):
            print(f"  Skipping {name}: file not found at {path}.")
            continue
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        print(f"Processing {name}...")
        lw = 2.5 if name == 'Stacking Ensemble' else 1.5
        
        # ---------------------------------------------------------
        # PLOT 1: CLEVELAND (SOURCE DOMAIN) VIA CV
        # ---------------------------------------------------------
        y_probs_cleve = cross_val_predict(model, X_cleve, y_cleve, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        fpr_c, tpr_c, _ = roc_curve(y_cleve, y_probs_cleve)
        roc_auc_c = auc(fpr_c, tpr_c)
        axes[0].plot(fpr_c, tpr_c, color=color, lw=lw, label=f'{name} (AUC = {roc_auc_c:.3f})')

        # ---------------------------------------------------------
        # PLOT 2: STATLOG (TARGET DOMAIN) VIA DIRECT INFERENCE
        # ---------------------------------------------------------
        y_probs_stat = model.predict_proba(X_stat)[:, 1]
        fpr_s, tpr_s, _ = roc_curve(y_stat, y_probs_stat)
        roc_auc_s = auc(fpr_s, tpr_s)
        axes[1].plot(fpr_s, tpr_s, color=color, lw=lw, label=f'{name} (AUC = {roc_auc_s:.3f})')

    # 6. FORMAT BOTH PLOTS
    for i, ax_title in enumerate(['Source Domain: Cleveland (5-Fold CV)', 'Target Domain: Statlog (Direct Inference)']):
        axes[i].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate', fontsize=12)
        axes[i].set_ylabel('True Positive Rate', fontsize=12)
        axes[i].set_title(ax_title, fontsize=14)
        axes[i].legend(loc="lower right", fontsize=11)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()

    # 7. SAVE FIGURE
    save_path = os.path.join(results_dir, 'comparative_roc_subplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSUCCESS: Side-by-Side ROC Curve saved to {save_path}")

if __name__ == "__main__":
    plot_side_by_side_roc()