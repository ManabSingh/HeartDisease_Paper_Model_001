import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle

BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
uci_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'cleveland_encoded.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_encoded.csv')
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned')

# Define the path to save the scaler
model_dir = os.path.join(BASE_DIR, 'models', 'trained_models')

def run_scaling():
    print("Starting Feature Scaling...")
    
    uci = pd.read_csv(uci_path)
    statlog = pd.read_csv(statlog_path)
    
    # We scale numerical features only (don't scale binary flags or target)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    
    scaler = StandardScaler()
    
    # Fit on UCI (Train Set) and transform both
    uci[num_cols] = scaler.fit_transform(uci[num_cols])
    statlog[num_cols] = scaler.transform(statlog[num_cols])
    
    # Save final versions
    uci.to_csv(os.path.join(output_dir, 'cleveland_final.csv'), index=False)
    statlog.to_csv(os.path.join(output_dir, 'statlog_final.csv'), index=False)
    
    # Save the scaler object using pickle
    os.makedirs(model_dir, exist_ok=True) # Ensure directory exists
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print("SUCCESS: Final scaled files saved as 'cleveland_final.csv' and 'statlog_final.csv'.")
    print(f"SUCCESS: Scaler successfully saved to: {scaler_path}")

if __name__ == "__main__":
    run_scaling()