"""
Flask Backend — AI Cardiology Assistant
Loads the actual trained .pkl models and serves real predictions via API.
Run with: python demo/server.py
Then open: http://localhost:5000
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# ==========================================
# 1. PATHS
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEMO_DIR = os.path.join(BASE_DIR, 'demo')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained_models')

# ==========================================
# 2. LOAD MODELS (once at startup)
# ==========================================
print("=" * 50)
print("  Loading Trained Models...")
print("=" * 50)

try:
    with open(os.path.join(MODEL_DIR, 'stacking_ensemble.pkl'), 'rb') as f:
        stacking_model = pickle.load(f)
    print("  [OK] Stacking Ensemble loaded")

    with open(os.path.join(MODEL_DIR, 'xgboost_tuned.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    print("  [OK] XGBoost loaded")

    with open(os.path.join(MODEL_DIR, 'lightgbm_tuned.pkl'), 'rb') as f:
        lgbm_model = pickle.load(f)
    print("  [OK] LightGBM loaded")

    with open(os.path.join(MODEL_DIR, 'rf_tuned.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    print("  [OK] Random Forest loaded")

    # ---> ADDED: Load the Scaler <---
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("  [OK] Scaler loaded")

    print("=" * 50)
    print("  All models loaded successfully!")
    print("=" * 50)
except FileNotFoundError as e:
    print(f"  [ERROR] Error loading models: {e}")
    print("  Make sure you have trained all models first.")
    exit(1)

# ==========================================
# 3. EXPECTED FEATURE COLUMNS
# ==========================================
EXPECTED_COLS = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang',
    'oldpeak', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0',
    'restecg_1', 'restecg_2', 'slope_0', 'slope_1', 'slope_2',
    'thal_1', 'thal_2', 'thal_3'
]

# ==========================================
# 4. FLASK APP
# ==========================================
app = Flask(__name__, static_folder=None)


@app.route('/')
def serve_index():
    """Serve the main HTML page."""
    return send_from_directory(DEMO_DIR, 'index.html')


@app.route('/figures/<path:filename>')
def serve_figures(filename):
    """Serve files from the figures/ directory."""
    return send_from_directory(os.path.join(BASE_DIR, 'figures'), filename)


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files from the results/ directory."""
    return send_from_directory(os.path.join(BASE_DIR, 'results'), filename)


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve CSS, JS, and other static files from demo/."""
    return send_from_directory(DEMO_DIR, filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accept patient data as JSON, run through ALL trained models,
    and return real predictions.
    """
    try:
        data = request.get_json()

        # Build one-hot encoded feature dictionary
        encoded = {col: 0.0 for col in EXPECTED_COLS}

        # Map continuous & binary variables
        encoded['age'] = float(data.get('age', 50))
        encoded['sex'] = float(data.get('sex', 1))
        encoded['trestbps'] = float(data.get('trestbps', 120))
        encoded['chol'] = float(data.get('chol', 200))
        encoded['fbs'] = float(data.get('fbs', 0))
        encoded['thalach'] = float(data.get('thalach', 150))
        encoded['exang'] = float(data.get('exang', 0))
        encoded['oldpeak'] = float(data.get('oldpeak', 1.0))
        encoded['ca'] = float(data.get('ca', 0))

        # One-hot encode categorical variables
        cp = int(data.get('cp', 0))
        if f'cp_{cp}' in encoded:
            encoded[f'cp_{cp}'] = 1.0

        restecg = int(data.get('restecg', 0))
        if f'restecg_{restecg}' in encoded:
            encoded[f'restecg_{restecg}'] = 1.0

        slope = int(data.get('slope', 0))
        if f'slope_{slope}' in encoded:
            encoded[f'slope_{slope}'] = 1.0

        thal = int(data.get('thal', 1))
        if f'thal_{thal}' in encoded:
            encoded[f'thal_{thal}'] = 1.0

        # Create DataFrame with exact column order
        patient_df = pd.DataFrame([encoded], columns=EXPECTED_COLS)

        # ---> ADDED: Apply the Scaler to continuous columns <---
        continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        patient_df[continuous_cols] = scaler.transform(patient_df[continuous_cols])

        # ---- Run ALL models ----
        # Stacking Ensemble (Meta-Classifier)
        ensemble_prob = float(stacking_model.predict_proba(patient_df)[0][1])
        ensemble_pred = int(stacking_model.predict(patient_df)[0])

        # Individual base models
        xgb_prob = float(xgb_model.predict_proba(patient_df)[0][1])
        lgbm_prob = float(lgbm_model.predict_proba(patient_df)[0][1])
        rf_prob = float(rf_model.predict_proba(patient_df)[0][1])

        return jsonify({
            'success': True,
            'ensemble': {
                'probability': ensemble_prob,
                'prediction': ensemble_pred,
                'label': 'Heart Disease Detected' if ensemble_pred == 1 else 'No Heart Disease'
            },
            'base_models': {
                'xgboost': xgb_prob,
                'lightgbm': lgbm_prob,
                'random_forest': rf_prob
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==========================================
# 5. RUN SERVER
# ==========================================
if __name__ == '__main__':
    print("\n[*] AI Cardiology Assistant -- Backend Server")
    print("    Open in browser: http://localhost:5000")
    print("    Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
