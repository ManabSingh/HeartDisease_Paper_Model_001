import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def plot_comparative_roc_cleveland():
    print("--- Generating Comparative ROC Curves on Cleveland (Source Domain) ---")
    
    # Load Source Data
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']

    # Models to plot
    models_to_evaluate = {
        'XGBoost': 'xgboost_tuned.pkl',
        'LightGBM': 'lightgbm_tuned.pkl',
        'Random Forest': 'rf_tuned.pkl',
        'Stacking Ensemble': 'stacking_ensemble.pkl'
    }

    plt.figure(figsize=(9, 7))
    colors = ['blue', 'green', 'purple', 'red']
    
    # 5-Fold Cross Validation to prevent plotting overfitted training curves
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for (name, file), color in zip(models_to_evaluate.items(), colors):
        path = os.path.join(model_dir, file)
        if not os.path.exists(path):
            print(f"  Skipping {name}: file not found.")
            continue
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        # Get Probabilities via CV
        print(f"  Running CV for {name}...")
        y_probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        
        # Calculate Curve
        fpr, tpr, _ = roc_curve(y_train, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot Line
        lw = 2.5 if name == 'Stacking Ensemble' else 1.5
        plt.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC = {roc_auc:.3f})')

    # Formatting the Plot
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Source Domain Performance: ROC Curves on Cleveland (5-Fold CV)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(results_dir, 'comparative_roc_cleveland.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSUCCESS: Comparative ROC Curve saved to {save_path}")

if __name__ == "__main__":
    plot_comparative_roc_cleveland()