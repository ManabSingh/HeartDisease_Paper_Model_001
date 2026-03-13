import pandas as pd
import os

# ==========================================
# 1. SETUP PATHS (Based on your folder image)
# ==========================================
BASE_DIR = r'D:\Research\Model\HeartDisease_Paper_Model_001'

# Raw Data Paths
uci_path = os.path.join(BASE_DIR, 'data', 'raw', 'cleveland', 'heart_disease_uci.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'raw', 'statlog', 'heart_disease_statlog.csv')

# Output Directories (Matched to your Harmonized folder image)
cleveland_out_dir = os.path.join(BASE_DIR, 'data', 'processed', 'harmonized', 'cleveland')
statlog_out_dir = os.path.join(BASE_DIR, 'data', 'processed', 'harmonized', 'statlog')

os.makedirs(cleveland_out_dir, exist_ok=True)
os.makedirs(statlog_out_dir, exist_ok=True)

def run_harmonization():
    print("Starting Feature Harmonization...")

    try:
        uci = pd.read_csv(uci_path)
        statlog = pd.read_csv(statlog_path)
        
        # Strip spaces from column names to prevent hidden errors
        uci.columns = uci.columns.str.strip()
        statlog.columns = statlog.columns.str.strip()
        
        print(f"Successfully loaded {len(uci)} UCI records and {len(statlog)} Statlog records.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # ==========================================
    # 2. ALIGNMENT (Skip 'dataset' filter as it's not in your file)
    # ==========================================
    
    # Ensure UCI has the same column names as Statlog
    # Based on your file, UCI might have 'num' instead of 'target'
    if 'target' not in uci.columns and 'num' in uci.columns:
        uci = uci.rename(columns={'num': 'target'})
        uci['target'] = (uci['target'] > 0).astype(int)
        print("Converted 'num' to 'target'.")

    # Rename thalch if it exists
    if 'thalch' in uci.columns:
        uci = uci.rename(columns={'thalch': 'thalach'})

    # ==========================================
    # 3. SYNC AND EXPORT
    # ==========================================
    
    # Reorder UCI columns to match Statlog exactly
    final_cols = statlog.columns.tolist()
    
    # Check if all columns exist
    available_cols = [c for c in final_cols if c in uci.columns]
    uci_harmonized = uci[available_cols].copy()

    # Define final file paths
    uci_out = os.path.join(cleveland_out_dir, 'cleveland_harmonized.csv')
    statlog_out = os.path.join(statlog_out_dir, 'statlog_harmonized.csv')

    # Save
    uci_harmonized.to_csv(uci_out, index=False)
    statlog.to_csv(statlog_out, index=False)

    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Cleveland saved to: {cleveland_out_dir}")
    print(f"Statlog saved to: {statlog_out_dir}")
    print(f"Final columns: {list(uci_harmonized.columns)}")

if __name__ == "__main__":
    run_harmonization()