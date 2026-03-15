import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def plot_confusion_matrix():
    print("--- Generating Confusion Matrix for Statlog ---")
    
    # Load Data & Model
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('target', axis=1).astype(float)
    y_test = df_test['target']

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy (0)', 'Heart Disease (1)'], 
                yticklabels=['Healthy (0)', 'Heart Disease (1)'],
                annot_kws={"size": 14})
    
    plt.title('Statlog Domain: Stacking Ensemble Confusion Matrix', fontsize=14)
    plt.ylabel('Actual Diagnosis', fontsize=12)
    plt.xlabel('Predicted Diagnosis', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(results_dir, 'confusion_matrix_statlog.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"SUCCESS: Confusion Matrix saved to {save_path}")

if __name__ == "__main__":
    plot_confusion_matrix()