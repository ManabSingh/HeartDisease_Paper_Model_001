import pandas as pd
import os

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'

# Input: The cleaned file from the imputation step
input_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'cleveland_clean.csv')

# Output: We save it back to 'cleaned' or a new folder if you prefer
# For this workflow, let's keep it in 'cleaned' as 'cleveland_no_outliers.csv'
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned')

def remove_outliers_iqr(df, columns):
    """Removes outliers using the IQR method (1.5 * IQR)"""
    df_final = df.copy()
    initial_count = len(df_final)
    
    for col in columns:
        Q1 = df_final[col].quantile(0.25)
        Q3 = df_final[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filtering the data
        df_final = df_final[(df_final[col] >= lower_bound) & (df_final[col] <= upper_bound)]
    
    print(f"Removed {initial_count - len(df_final)} outlier rows.")
    return df_final

def run_outlier_removal():
    print("Starting Outlier Removal (Cleveland)...")
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Run imputation first.")
        return

    # 2. SELECT CONTINUOUS COLUMNS
    # We only remove outliers from continuous medical measurements.
    # We DO NOT remove outliers from categorical columns like 'sex' or 'cp'.
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # 3. APPLY REMOVAL
    df_cleaned = remove_outliers_iqr(df, continuous_cols)

    # 4. EXPORT
    output_file = os.path.join(output_dir, 'cleveland_no_outliers.csv')
    df_cleaned.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Outliers removed. Final record count: {len(df_cleaned)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    run_outlier_removal()