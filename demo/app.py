import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="AI Cardiology Assistant", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stAlert {border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🫀 Cross-Domain AI Cardiology Assistant")
st.markdown("*An intelligent diagnostic tool powered by a Stacking Ensemble (XGBoost, LightGBM, Random Forest).*")

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource 
def load_models():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')
    
    with open(os.path.join(model_dir, 'stacking_ensemble.pkl'), 'rb') as f:
        stacking = pickle.load(f)
    with open(os.path.join(model_dir, 'xgboost_tuned.pkl'), 'rb') as f:
        xgb = pickle.load(f)
    with open(os.path.join(model_dir, 'rf_tuned.pkl'), 'rb') as f:
        rf = pickle.load(f)
        
    return stacking, xgb, rf

try:
    stacking_model, xgb_model, rf_model = load_models()
except Exception as e:
    st.error(f"Error loading models. Details: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: PATIENT DATA INPUT
# ==========================================
st.sidebar.header("📋 Patient Clinical Profile")

def get_encoded_user_input():
    # Raw Inputs
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: ["Asymptomatic", "Atypical Angina", "Non-anginal Pain", "Typical Angina"][x])
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (chol)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (fbs)", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("ST Segment Slope (slope)", options=[0, 1, 2])
    ca = st.sidebar.slider("Major Vessels Colored (ca)", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversable Defect"][x])

    # EXPECTED COLUMNS BY THE MODEL (From the Error Log)
    expected_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 
                     'oldpeak', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0', 
                     'restecg_1', 'restecg_2', 'slope_0', 'slope_1', 'slope_2', 
                     'thal_1', 'thal_2', 'thal_3']

    # Initialize a dictionary with 0s for all expected columns
    encoded_data = {col: 0.0 for col in expected_cols}

    # Map Continuous and Binary variables
    encoded_data['age'] = float(age)
    encoded_data['sex'] = float(sex)
    encoded_data['trestbps'] = float(trestbps)
    encoded_data['chol'] = float(chol)
    encoded_data['fbs'] = float(fbs)
    encoded_data['thalach'] = float(thalach)
    encoded_data['exang'] = float(exang)
    encoded_data['oldpeak'] = float(oldpeak)
    encoded_data['ca'] = float(ca)

    # Apply One-Hot Encoding logic manually for categorical variables
    if f'cp_{cp}' in encoded_data: encoded_data[f'cp_{cp}'] = 1.0
    if f'restecg_{restecg}' in encoded_data: encoded_data[f'restecg_{restecg}'] = 1.0
    if f'slope_{slope}' in encoded_data: encoded_data[f'slope_{slope}'] = 1.0
    if f'thal_{thal}' in encoded_data: encoded_data[f'thal_{thal}'] = 1.0

    # Ensure output is a dataframe with the exact column order expected
    df = pd.DataFrame([encoded_data], columns=expected_cols)
    return df

patient_data = get_encoded_user_input()

# ==========================================
# 4. MAIN PANEL: DIAGNOSTICS
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Diagnostic Verdict (Meta-Classifier)")
    
    # Generate Predictions
    probability = stacking_model.predict_proba(patient_data)[0][1]

    # Display Results dynamically
    if probability > 0.50:
        st.error(f"🚨 **HIGH RISK of Heart Disease detected.**")
        st.metric(label="Risk Probability", value=f"{probability * 100:.1f}%")
        st.write("The ensemble model indicates a strong likelihood of cardiovascular pathology. Immediate clinical review is recommended.")
    else:
        st.success(f"✅ **LOW RISK of Heart Disease.**")
        st.metric(label="Risk Probability", value=f"{probability * 100:.1f}%")
        st.write("The ensemble model does not detect significant patterns of cardiovascular pathology based on the provided profile.")

with col2:
    st.subheader("The 'Board of Doctors'")
    st.write("Individual Base Model Opinions:")
    
    xgb_prob = xgb_model.predict_proba(patient_data)[0][1]
    rf_prob = rf_model.predict_proba(patient_data)[0][1]
    
    st.info(f"**Dr. XGBoost:** {xgb_prob * 100:.1f}%")
    st.info(f"**Dr. Random Forest:** {rf_prob * 100:.1f}%")
    st.caption("The Stacking Meta-Classifier weighs these individual predictions to make the final diagnostic verdict.")