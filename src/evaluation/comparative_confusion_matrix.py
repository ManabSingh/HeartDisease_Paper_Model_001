import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def generate_comparative_confusion_matrix():
    print("--- Generating Side-by-Side Confusion Matrices ---")
    
    # Load Model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ---------------------------------------------------------
    # PART A: Get Cleveland (Source) Predictions via CV
    # ---------------------------------------------------------
    df_cleve = pd.read_csv(cleveland_path)
    X_cleve = df_cleve.drop('target', axis=1)
    y_cleve = df_cleve['target']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cleve = cross_val_predict(model, X_cleve, y_cleve, cv=cv, n_jobs=-1)
    cm_cleve = confusion_matrix(y_cleve, y_pred_cleve)

    # ---------------------------------------------------------
    # PART B: Get Statlog (Target) Predictions via Direct Inference
    # ---------------------------------------------------------
    df_stat = pd.read_csv(statlog_path)
    X_stat = df_stat.drop('target', axis=1).astype(float)
    y_stat = df_stat['target']
    
    y_pred_stat = model.predict(X_stat)
    cm_stat = confusion_matrix(y_stat, y_pred_stat)

    # ---------------------------------------------------------
    # PART C: Plot Side-by-Side
    # ---------------------------------------------------------
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Domain Generalization: Stacking Ensemble Confusion Matrices', fontsize=16, fontweight='bold', y=1.05)

    # Plot 1: Cleveland (Source)
    sns.heatmap(cm_cleve, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=['Healthy (0)', 'Heart Disease (1)'], 
                yticklabels=['Healthy (0)', 'Heart Disease (1)'],
                annot_kws={"size": 14})
    axes[0].set_title('Source Domain (Cleveland)\nStratified 5-Fold CV', fontsize=14)
    axes[0].set_ylabel('Actual Diagnosis', fontsize=12)
    axes[0].set_xlabel('Predicted Diagnosis', fontsize=12)

    # Plot 2: Statlog (Target)
    sns.heatmap(cm_stat, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Healthy (0)', 'Heart Disease (1)'], 
                yticklabels=['Healthy (0)', 'Heart Disease (1)'],
                annot_kws={"size": 14})
    axes[1].set_title('Target Domain (Statlog)\nDirect Inference', fontsize=14)
    axes[1].set_ylabel('Actual Diagnosis', fontsize=12)
    axes[1].set_xlabel('Predicted Diagnosis', fontsize=12)

    plt.tight_layout()

    # Save the combined figure
    save_path = os.path.join(results_dir, 'comparative_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SUCCESS: Comparative Confusion Matrix saved to {save_path}")

if __name__ == "__main__":
    generate_comparative_confusion_matrix()