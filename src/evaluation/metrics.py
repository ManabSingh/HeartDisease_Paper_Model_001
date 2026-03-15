import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def evaluate_target_metrics():
    print("--- Evaluating Meta-Classifier on Target Domain (Statlog) ---")
    
    # Load Data & Model
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('target', axis=1).astype(float)
    y_test = df_test['target']

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'AUC_ROC': roc_auc_score(y_test, y_probs)
    }

    # Save to CSV for the paper
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(results_dir, 'statlog_final_metrics.csv'), index=False)

    print("\n" + "="*45)
    print(" STATLOG FINAL METRICS (TARGET DOMAIN)")
    print("="*45)
    for k, v in metrics.items():
        print(f"{k.ljust(15)}: {v:.4f}")
    print("="*45)
    print(f"Saved to: {results_dir}/statlog_final_metrics.csv")

if __name__ == "__main__":
    evaluate_target_metrics()