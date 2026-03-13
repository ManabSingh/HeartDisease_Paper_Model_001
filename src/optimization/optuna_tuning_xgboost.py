import pandas as pd
import os
import pickle
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
train_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
os.makedirs(model_dir, exist_ok=True)

print("--- Loading Cleveland Data for Tuning ---")
df = pd.read_csv(train_path)
X_train = df.drop('target', axis=1)
y_train = df['target']

# 5-Fold Cross Validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 2. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # Strict anti-overfitting ranges
    param = {
        'n_estimators': 100,
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    model = XGBClassifier(**param)
    
    # Calculate Cross-Validation Score
    scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        scores.append(accuracy_score(y_fold_val, preds))
        
    return sum(scores) / len(scores)

# ==========================================
# 3. RUN TUNING & SAVE BEST MODEL
# ==========================================
def run_optimization():
    print("\n" + "="*40)
    print(" STARTING OPTUNA OPTIMIZATION (XGBOOST)")
    print("="*40)
    
    # Run 50 smart trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print(f"\nBest CV Accuracy: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    
    # Train one final time on the FULL Cleveland dataset using the winning parameters
    best_params = study.best_params
    best_params['n_estimators'] = 100
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    best_params['n_jobs'] = -1
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Save the tuned model
    save_file = os.path.join(model_dir, 'xgboost_tuned.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(final_model, f)
        
    print("\n" + "="*50)
    print(f"SUCCESS: Best Tuned Model saved to --> {save_file}")
    print("="*50)

if __name__ == "__main__":
    run_optimization()