import pandas as pd
import os
import pickle
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

def build_stacking_model():
    # ==========================================
    # 2. LOAD SOURCE DATASET
    # ==========================================
    print("--- Loading Cleveland Data ---")
    train_df = pd.read_csv(train_path)
    X_train, y_train = train_df.drop('target', axis=1), train_df['target']

    # ==========================================
    # 3. LOAD TUNED BASE MODELS
    # ==========================================
    print("--- Loading Tuned Base Models ---")
    try:
        with open(os.path.join(model_dir, 'xgboost_tuned.pkl'), 'rb') as f:
            xgb_model = pickle.load(f)
        with open(os.path.join(model_dir, 'lightgbm_tuned.pkl'), 'rb') as f:
            lgbm_model = pickle.load(f)
        with open(os.path.join(model_dir, 'rf_tuned.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        print("Successfully loaded XGBoost, LightGBM, and Random Forest.")
    except FileNotFoundError as e:
        print(f"Error loading models. Please ensure you ran the tuning scripts. Details: {e}")
        return

    # ==========================================
    # 4. BUILD STACKING ENSEMBLE
    # ==========================================
    # The "Board of Doctors"
    estimators = [
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', rf_model)
    ]

    # The "Chief of Medicine" (Meta-Learner)
    # C=0.1 applies moderate regularization so the Meta-Learner doesn't overfit
    meta_model = LogisticRegression(C=0.1, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5, # 5-fold CV to train the meta-learner on out-of-fold predictions
        n_jobs=-1
    )

    # ==========================================
    # 5. TRAIN & EVALUATE ON SOURCE (CLEVELAND)
    # ==========================================
    print("\nTraining Stacking Ensemble on Cleveland Data... (This may take a moment)")
    stacking_clf.fit(X_train, y_train)

    train_preds = stacking_clf.predict(X_train)
    
    print("\n" + "="*45)
    print("  SOURCE DOMAIN: CLEVELAND METRICS (TRAIN)")
    print("="*45)
    print(f"Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, train_preds))

    # ==========================================
    # 6. SAVE FINAL ENSEMBLE
    # ==========================================
    save_file = os.path.join(model_dir, 'stacking_ensemble.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(stacking_clf, f)
        
    print("\n" + "="*50)
    print(f"SUCCESS: Final Ensemble saved to --> {save_file}")
    print("="*50)

if __name__ == "__main__":
    build_stacking_model()