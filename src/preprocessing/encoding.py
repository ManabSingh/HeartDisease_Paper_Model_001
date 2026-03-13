import pandas as pd
import os

# Paths
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'
uci_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'cleveland_no_outliers.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_clean.csv')
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned')

def run_encoding():
    print("Starting One-Hot Encoding & Alignment...")
    
    try:
        # Load both datasets
        uci = pd.read_csv(uci_path)
        statlog = pd.read_csv(statlog_path)
        
        # 1. Define Categorical Columns
        categorical_cols = ['cp', 'restecg', 'slope', 'thal']

        # 2. Apply One-Hot Encoding
        uci_enc = pd.get_dummies(uci, columns=categorical_cols, prefix=categorical_cols)
        statlog_enc = pd.get_dummies(statlog, columns=categorical_cols, prefix=categorical_cols)

        # 3. ALIGNMENT (Crucial for Cross-Domain Research)
        # Ensure both datasets have the EXACT same binary columns
        all_columns = uci_enc.columns.union(statlog_enc.columns)
        
        uci_enc = uci_enc.reindex(columns=all_columns, fill_value=0)
        statlog_enc = statlog_enc.reindex(columns=all_columns, fill_value=0)

        # 4. Save
        uci_enc.to_csv(os.path.join(output_dir, 'cleveland_encoded.csv'), index=False)
        statlog_enc.to_csv(os.path.join(output_dir, 'statlog_encoded.csv'), index=False)
        
        print("-" * 30)
        print("SUCCESS: Cleveland and Statlog encoded and aligned.")
        print(f"Final feature count: {len(uci_enc.columns)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you ran 'statlog_cleaning.py' and 'outlier_removal.py'.")

if __name__ == "__main__":
    run_encoding()