import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'stacking_ensemble.pkl')
results_dir = os.path.join(BASE_DIR, 'results', 'evaluation')
os.makedirs(results_dir, exist_ok=True)

def plot_cleveland_confusion_matrix():
    print("--- Generating Confusion Matrix for Cleveland (Source Domain) ---")
    
    # 1. Load Data & Model
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 2. Generate "Honest" Predictions via Cross-Validation
    print("Running Stratified 5-Fold CV to prevent training bias...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)

    # 3. Compute Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)

    # 4. Plot the Heatmap
    plt.figure(figsize=(7, 5))
    
    # Using a different color map ('Greens') to visually distinguish it from the Statlog ('Blues') plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Healthy (0)', 'Heart Disease (1)'], 
                yticklabels=['Healthy (0)', 'Heart Disease (1)'],
                annot_kws={"size": 14})
    
    plt.title('Cleveland Domain: Stacking Ensemble Confusion Matrix (CV)', fontsize=14)
    plt.ylabel('Actual Diagnosis', fontsize=12)
    plt.xlabel('Predicted Diagnosis', fontsize=12)
    plt.tight_layout()

    # 5. Save the Figure
    save_path = os.path.join(results_dir, 'confusion_matrix_cleveland.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"SUCCESS: Confusion Matrix saved to {save_path}")

if __name__ == "__main__":
    plot_cleveland_confusion_matrix()