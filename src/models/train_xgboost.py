import pandas as pd
import numpy as np
import os
import pickle
import time
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)
XGB_SEARCH_ITERS = int(os.getenv('XGB_SEARCH_ITERS', '20'))


def find_best_threshold(y_true, y_probs, steps=201):
    thresholds = np.linspace(0.1, 0.9, steps)
    best_thr = 0.5
    best_acc = -1.0

    for thr in thresholds:
        preds = (y_probs >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return float(best_thr), float(best_acc)


def tune_xgboost(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        'n_estimators': [150, 250, 350, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 4, 6],
        'subsample': [0.75, 0.85, 0.95, 1.0],
        'colsample_bytree': [0.65, 0.75, 0.85, 1.0],
        'gamma': [0.0, 0.1, 0.2, 0.4],
        'reg_alpha': [0.0, 0.01, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0, 10.0]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=XGB_SEARCH_ITERS,
        scoring='accuracy',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    return search.best_params_, float(search.best_score_)

def run_full_analysis():
    print("--- Loading Data ---")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # ==========================================
    # 2. HYPERPARAMETER SEARCH
    # ==========================================
    print("\n--- Hyperparameter Search (Randomized CV) ---")
    tune_start = time.time()
    best_params, best_cv_acc = tune_xgboost(X_train, y_train)
    print(f"Best CV Accuracy: {best_cv_acc:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Tuning Time: {time.time() - tune_start:.2f} seconds")

    # ==========================================
    # 3. TRAIN FINAL MODEL WITH EARLY STOPPING
    # ==========================================
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        **best_params
    )

    print("\n--- Training on Cleveland Dataset ---")
    start_time = time.time()
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"Total Training Time: {time.time() - start_time:.4f} seconds")

    # ==========================================
    # 4. THRESHOLD OPTIMIZATION FOR ACCURACY
    # ==========================================
    y_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    best_thr, best_val_acc = find_best_threshold(y_val, y_val_probs)
    print(f"Best Validation Threshold: {best_thr:.3f}")
    print(f"Validation Accuracy @ Best Threshold: {best_val_acc:.4f}")

    # ==========================================
    # 5. FULL REPORT: CLEVELAND (SOURCE)
    # ==========================================
    print("\n" + "="*30)
    print("  SOURCE DOMAIN: CLEVELAND METRICS")
    print("="*30)
    y_train_probs = xgb_model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_probs >= best_thr).astype(int)
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("\nDetailed Classification Report (Cleveland):")
    print(classification_report(y_train, y_train_pred))

    # ==========================================
    # 6. FULL REPORT: STATLOG (TARGET)
    # ==========================================
    print("\n" + "="*30)
    print("  TARGET DOMAIN: STATLOG METRICS")
    print("="*30)
    y_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= best_thr).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_probs):.4f}")
    print("\nDetailed Classification Report (Statlog):")
    print(classification_report(y_test, y_test_pred))

    # Save the model
    save_path = os.path.join(model_dir, 'xgboost_cpu_tuned.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(
            {
                'model': xgb_model,
                'threshold': best_thr,
                'best_cv_accuracy': best_cv_acc,
                'best_params': best_params
            },
            f
        )
    print(f"\nModel saved to: {save_path}")

if __name__ == "__main__":
    run_full_analysis()