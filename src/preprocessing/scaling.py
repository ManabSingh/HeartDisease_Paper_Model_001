import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
uci_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'cleveland_encoded.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_encoded.csv')
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned')

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
    
    print("SUCCESS: Final scaled files saved as 'cleveland_final.csv' and 'statlog_final.csv'.")

if __name__ == "__main__":
    run_scaling()