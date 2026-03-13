import pandas as pd
import os
from imblearn.combine import SMOTEENN

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'

# Input: Use the encoded Cleveland file
input_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'cleveland_encoded.csv')

# Output: Save to a new 'balanced' folder
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'balanced')
os.makedirs(output_dir, exist_ok=True)

def run_smoteenn():
    print("Starting SMOTEENN Balancing on Cleveland dataset...")
    
    try:
        # Load the encoded data
        df = pd.read_csv(input_path)
        
        # 1. Separate Features (X) and Target (y)
        X = df.drop(columns=['target'])
        y = df['target']
        
        print(f"Original class distribution:\n{y.value_counts()}")

        # 2. Initialize SMOTEENN
        # sampling_strategy='auto' ensures the classes are balanced 50/50
        sme = SMOTEENN(random_state=42)
        
        # 3. Fit and Resample
        X_resampled, y_resampled = sme.fit_resample(X, y)

        # 4. Combine back into a single DataFrame
        df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced['target'] = y_resampled

        # 5. Export
        output_file = os.path.join(output_dir, 'cleveland_smoteenn.csv')
        df_balanced.to_csv(output_file, index=False)
        
        print("-" * 30)
        print("SUCCESS!")
        print(f"New class distribution:\n{y_resampled.value_counts()}")
        print(f"Balanced dataset saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during SMOTEENN: {e}")

if __name__ == "__main__":
    run_smoteenn()