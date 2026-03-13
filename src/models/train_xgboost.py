import pandas as pd
import numpy as np
import os
import pickle
import time
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)
XGB_SEARCH_ITERS = int(os.getenv('XGB_SEARCH_ITERS', '50'))


def find_best_threshold(y_true, y_probs, steps=201):
    """Find threshold that maximises the weighted F1 score."""
    thresholds = np.linspace(0.1, 0.9, steps)
    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        preds = (y_probs >= thr).astype(int)
        score = f1_score(y_true, preds, average='weighted', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr

    return float(best_thr), float(best_f1)


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
        'n_estimators': [100, 150, 200, 250, 350, 500],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        'max_depth': [2, 3, 4, 5],
        'min_child_weight': [2, 3, 5, 7, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'gamma': [0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
        'reg_alpha': [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
        'reg_lambda': [1.0, 3.0, 5.0, 10.0, 20.0],
        'scale_pos_weight': [0.8, 1.0, 1.2, 1.5]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=XGB_SEARCH_ITERS,
        scoring='f1_weighted',
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
    best_params, best_cv_f1 = tune_xgboost(X_train, y_train)
    print(f"Best CV F1 (weighted): {best_cv_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Tuning Time: {time.time() - tune_start:.2f} seconds")

    # ==========================================
    # 3. TRAIN FINAL MODEL ON FULL TRAINING DATA
    # ==========================================
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    print("\n--- Training on Cleveland Dataset ---")
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    print(f"Total Training Time: {time.time() - start_time:.4f} seconds")

    # ==========================================
    # 4. THRESHOLD OPTIMIZATION (F1-BASED VIA CV)
    # ==========================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            **best_params
        )
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]

    best_thr, best_val_f1 = find_best_threshold(y_train, oof_probs)
    print(f"Best CV Threshold: {best_thr:.3f}")
    print(f"CV F1 (weighted) @ Best Threshold: {best_val_f1:.4f}")

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
                'best_cv_f1': best_cv_f1,
                'best_params': best_params
            },
            f
        )
    print(f"\nModel saved to: {save_path}")

if __name__ == "__main__":
    run_full_analysis()