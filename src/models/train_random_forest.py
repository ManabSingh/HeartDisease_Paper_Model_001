import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models','trained_models')

os.makedirs(model_dir, exist_ok=True)

def run_robust_rf_training():
    print("--- Loading Cleveland Data ---")
    df = pd.read_csv(train_path)
    
    X_train = df.drop('target', axis=1)
    y_train = df['target']

    # ==========================================
    # 2. INITIALIZE ROBUST RANDOM FOREST
    # ==========================================
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,               # 1. Shallower trees (reduced from 4)
        min_samples_split=15,      # 2. Needs at least 15 patients to create a new split
        min_samples_leaf=5,        # 3. A final decision leaf must contain at least 5 patients (stops 1.00 precision)
        max_samples=0.7,           # 4. Each tree only sees a random 70% of the dataset
        max_features='sqrt',       # 5. Forces trees to use different features
        random_state=42,
        n_jobs=-1
    )

    print("Training Robust Random Forest on Cleveland dataset...")
    rf_model.fit(X_train, y_train)

    # ==========================================
    # 3. EVALUATION (Cleveland Only)
    # ==========================================
    y_pred = rf_model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    
    print("\n" + "="*40)
    print("  SOURCE DOMAIN: CLEVELAND METRICS (RF)")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred))

    # ==========================================
    # 4. SAVE MODEL
    # ==========================================
    save_file = os.path.join(model_dir, 'random_forest.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(rf_model, f)
    
    print("\n" + "="*50)
    print(f"SUCCESS: Robust Random Forest saved to --> {save_file}")
    print("="*50)

if __name__ == "__main__":
    run_robust_rf_training()