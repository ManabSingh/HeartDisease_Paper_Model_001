import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamic Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
results_dir = os.path.join(BASE_DIR, 'results', 'tables')
os.makedirs(results_dir, exist_ok=True)

def extract_target_importance():
    print("--- Extracting Feature Importance for Target Domain (Statlog) ---")
    
    # 1. Load Statlog
    df = pd.read_csv(test_path)
    X = df.drop('target', axis=1).astype(float)
    
    models = {
        "XGBoost": "xgboost_tuned.pkl",
        "LightGBM": "lightgbm_tuned.pkl",
        "RandomForest": "rf_tuned.pkl"
    }

    target_ranking = pd.DataFrame(index=X.columns)

    for name, file in models.items():
        path = os.path.join(model_dir, file)
        if not os.path.exists(path): continue
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Get importance values
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            target_ranking[name] = importances / np.sum(importances)

    # Calculate Target Average
    target_ranking['Target_Average'] = target_ranking.mean(axis=1)
    target_ranking = target_ranking.sort_values(by='Target_Average', ascending=False)

    # Save CSV for your paper's "Results" section
    save_path = os.path.join(results_dir, 'target_feature_importance.csv')
    target_ranking.to_csv(save_path)
    
    # Visual Plot for Feature Ranking
    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_ranking['Target_Average'], y=target_ranking.index, palette='magma')
    plt.title('Feature Importance: Target Domain (Statlog)')
    plt.xlabel('Importance Score (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'results', 'target_importance_plot.png'))
    
    print(f"SUCCESS: Target importance table and plot saved in 'results/'")
    print("\nTop 3 Features in Statlog Domain:")
    print(target_ranking['Target_Average'].head(3))

if __name__ == "__main__":
    extract_target_importance()