import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

def build_manual_meta_classifier():
    # ==========================================
    # 2. LOAD DATA & BASE MODELS
    # ==========================================
    print("--- Loading Data and Models ---")
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1)
    y = df['target']

    with open(os.path.join(model_dir, 'xgboost_tuned.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(model_dir, 'lightgbm_tuned.pkl'), 'rb') as f:
        lgbm_model = pickle.load(f)
    with open(os.path.join(model_dir, 'rf_tuned.pkl'), 'rb') as f:
        rf_model = pickle.load(f)

    # ==========================================
    # 3. GENERATE OUT-OF-FOLD (OOF) PREDICTIONS
    # ==========================================
    # We use predict_proba to get the probability of Disease (Class 1)
    # This gives the Meta-Model much richer information than just 0 or 1
    print("\nGenerating blind Out-Of-Fold predictions (Level 0)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_oof = cross_val_predict(xgb_model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    lgbm_oof = cross_val_predict(lgbm_model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    rf_oof = cross_val_predict(rf_model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

    # ==========================================
    # 4. CREATE THE META-DATASET
    # ==========================================
    print("\nAssembling the Meta-Dataset (Level 1)...")
    meta_X = pd.DataFrame({
        'xgb_prob': xgb_oof,
        'lgbm_prob': lgbm_oof,
        'rf_prob': rf_oof
    })
    
    # Save this for your research paper!
    meta_X_saved = meta_X.copy()
    meta_X_saved['Actual_Diagnosis'] = y
    meta_data_path = os.path.join(BASE_DIR, 'data', 'processed', 'meta_dataset_cleveland.csv')
    meta_X_saved.to_csv(meta_data_path, index=False)
    print(f"-> Saved Meta-Dataset for your paper to: {meta_data_path}")

    # Show a sneak peek of what the Meta-Classifier actually sees
    print("\nPeek at the Meta-Dataset features:")
    print(meta_X.head())

    # ==========================================
    # 5. TRAIN THE META-CLASSIFIER (LOGISTIC REGRESSION)
    # ==========================================
    print("\nTraining the Logistic Regression Meta-Classifier...")
    meta_model = LogisticRegression(C=0.1, random_state=42)
    meta_model.fit(meta_X, y)

    # Evaluate how well the Meta-Classifier learned from the Base Models
    meta_preds = meta_model.predict(meta_X)
    acc = accuracy_score(y, meta_preds)

    print("\n" + "="*45)
    print("  MANUAL META-CLASSIFIER RESULTS (CLEVELAND)")
    print("="*45)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, meta_preds))

    # ==========================================
    # 6. SAVE THE META-CLASSIFIER
    # ==========================================
    save_file = os.path.join(model_dir, 'manual_meta_logistic.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(meta_model, f)
        
    print("\n" + "="*50)
    print(f"SUCCESS: Meta-Learner saved to --> {save_file}")
    print("="*50)

if __name__ == "__main__":
    build_manual_meta_classifier()