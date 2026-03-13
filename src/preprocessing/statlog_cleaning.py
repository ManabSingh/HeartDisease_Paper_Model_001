import pandas as pd
import os

# Paths
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
input_path = os.path.join(BASE_DIR, 'data', 'processed', 'harmonized', 'statlog', 'statlog_harmonized.csv')
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned')

os.makedirs(output_dir, exist_ok=True)

def run_statlog_cleaning():
    print("Step: Statlog Cleaning...")
    try:
        df = pd.read_csv(input_path)
        
        # Ensure categorical columns are integers
        cols_to_fix = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        output_file = os.path.join(output_dir, 'statlog_clean.csv')
        df.to_csv(output_file, index=False)
        print(f"SUCCESS: Created {output_file}")
    except Exception as e:
        print(f"Error cleaning Statlog: {e}")

if __name__ == "__main__":
    run_statlog_cleaning()