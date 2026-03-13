import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

def run_cleveland_training():
    print("--- Loading Cleveland Data ---")
    df = pd.read_csv(train_path)
    
    X_train = df.drop('target', axis=1)
    y_train = df['target']

    # ==========================================
    # 2. INITIALIZE & TRAIN ROBUST MODEL
    # ==========================================
    # Added strict anti-overfitting constraints
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,              # 1. Shallower trees (down from 4)
        min_child_weight=5,       # 2. Requires more evidence to make a specific rule
        subsample=0.7,            # 3. Only uses 70% of rows per tree (adds randomness)
        colsample_bytree=0.7,     # 4. Only uses 70% of columns per tree
        reg_lambda=5,             # 5. L2 Regularization (penalizes large weights)
        reg_alpha=2,              # 6. L1 Regularization (ignores useless features)
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    print("Training robust model on Cleveland dataset...")
    model.fit(X_train, y_train)

    # ==========================================
    # 3. EVALUATION (Cleveland Only)
    # ==========================================
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    
    print("\n" + "="*40)
    print("  SOURCE DOMAIN: CLEVELAND METRICS")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred))

    # ==========================================
    # 4. SAVE MODEL
    # ==========================================
    save_file = os.path.join(model_dir, 'xgboost.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(model, f)
    
    print("\n" + "="*50)
    print(f"SUCCESS: Robust Model saved to --> {save_file}")
    print("="*50)

if __name__ == "__main__":
    run_cleveland_training()