import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'

# Path to the file created by the script above
input_path = os.path.join(BASE_DIR, 'data', 'processed', 'harmonized', 'cleveland', 'cleveland_harmonized.csv')

# Path to save the Cleaned version
output_dir = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned')
os.makedirs(output_dir, exist_ok=True)

def impute_data():
    try:
        df = pd.read_csv(input_path)
        print(f"Missing values before imputation: {df.isnull().sum().sum()}")
    except FileNotFoundError:
        print("Error: cleveland_harmonized.csv not found. Run Harmonization first.")
        return

    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    
    # Impute only features, keep target aside
    target = df['target']
    features = df.drop(columns=['target'])
    
    imputed_data = imputer.fit_transform(features)
    df_imputed = pd.DataFrame(imputed_data, columns=features.columns)
    df_imputed['target'] = target.values

    # Convert categorical columns back to integers
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in cat_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].round().astype(int)

    # Save to 'cleaned' folder
    out_path = os.path.join(output_dir, 'cleveland_clean.csv')
    df_imputed.to_csv(out_path, index=False)
    
    print(f"SUCCESS! Cleaned file saved to: {out_path}")
    print(f"Missing values after: {df_imputed.isnull().sum().sum()}")

if __name__ == "__main__":
    impute_data()