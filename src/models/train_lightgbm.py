import pandas as pd
import os
import pickle
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

def run_lightgbm_training():
    print("--- Loading Cleveland Data ---")
    df = pd.read_csv(train_path)
    
    X_train = df.drop('target', axis=1)
    y_train = df['target']

    # ==========================================
    # 2. INITIALIZE & TRAIN ROBUST MODEL
    # ==========================================
    # LightGBM needs strict 'num_leaves' and 'min_child_samples' to stop overfitting
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=3,               # 1. Shallow trees
        num_leaves=7,              # 2. Strict limit on leaves (prevents complex leaf-wise growth)
        min_child_samples=15,      # 3. Needs 15 patients per leaf (stops 100% precision memorization)
        subsample=0.7,             # 4. Bagging fraction (uses 70% of rows)
        subsample_freq=1,          # Required for LightGBM bagging to work
        colsample_bytree=0.7,      # 5. Feature fraction (uses 70% of columns)
        reg_lambda=5.0,            # 6. L2 Regularization
        reg_alpha=2.0,             # 7. L1 Regularization
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    print("Training robust LightGBM on Cleveland dataset...")
    model.fit(X_train, y_train)

    # ==========================================
    # 3. EVALUATION (Cleveland Only)
    # ==========================================
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    
    print("\n" + "="*40)
    print("  SOURCE DOMAIN: CLEVELAND METRICS (LGBM)")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred))

    # ==========================================
    # 4. SAVE MODEL
    # ==========================================
    save_file = os.path.join(model_dir, 'lightgbm.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(model, f)
    
    print("\n" + "="*50)
    print(f"SUCCESS: Robust LightGBM saved to --> {save_file}")
    print("="*50)

if __name__ == "__main__":
    run_lightgbm_training()