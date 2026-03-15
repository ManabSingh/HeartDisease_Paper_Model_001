import pandas as pd
import os
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
test_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

def run_meta_inference_on_target():
    print("--- Starting Meta-Inference on Statlog ---")
    
    # 1. Load Statlog Data
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('target', axis=1).astype(float)
    y_test = df_test['target']

    # 2. Load the Full Stacking Model
    # (The StackingClassifier object contains both the base models and the meta-classifier)
    try:
        with open(os.path.join(model_dir, 'stacking_ensemble.pkl'), 'rb') as f:
            stacking_model = pickle.load(f)
    except FileNotFoundError:
        print("Error: stacking_ensemble.pkl not found. Please train the ensemble first.")
        return

    # 3. Apply to Statlog
    print("Meta-Classifier is now processing Statlog predictions...")
    y_pred = stacking_model.predict(X_test)

    # 4. Final Results
    print("\n" + "="*50)
    print("   FINAL META-CLASSIFIER RESULTS: STATLOG")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*50)

if __name__ == "__main__":
    run_meta_inference_on_target()