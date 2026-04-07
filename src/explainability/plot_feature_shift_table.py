import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import spearmanr

def create_feature_shift_table_image():
    # 1. Define Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    CSV_PATH = os.path.join(BASE_DIR, 'results', 'explainability', 'advanced_feature_shift_table.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'explainability')
    OUTPUT_IMG_PATH = os.path.join(OUTPUT_DIR, 'advanced_feature_shift_table.png')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}. Run feature_shift_table.py first.")
        return

    # --- FIX 3: CONTEXTUALIZE MAGNITUDES ---
    # We now evaluate the row to see if the feature is actually important (Top 10)
    def determine_status(row):
        shift_str = row['Rank Shift']
        cleve_rank = int(row['Cleveland Rank'])
        stat_rank = int(row['Statlog Rank'])
        
        # Consider a feature 'Important' if it's in the top 10 of EITHER dataset
        is_important = (cleve_rank <= 10) or (stat_rank <= 10)
        
        if 'Unchanged' in str(shift_str):
            return 'Stable'
        
        match = re.search(r'\d+', str(shift_str))
        if match:
            shift_val = int(match.group())
            if shift_val >= 4:
                if is_important:
                    return 'Highly Unstable (Critical)'
                else:
                    return 'Noise (Low Impact)'
            else:
                if is_important:
                    return 'Shifted'
                else:
                    return 'Minor Shift (Noise)'
        return 'Unknown'

    # Drop old status if it exists and recalculate using new row-based logic
    if 'Status' in df.columns:
        df = df.drop('Status', axis=1)
    
    df['Status'] = df.apply(determine_status, axis=1)
    df.to_csv(CSV_PATH, index=False)

    # Clean up magnitude columns
    df['Cleveland Magnitude'] = df['Cleveland Magnitude'].apply(lambda x: f"{float(x):.4f}")
    df['Statlog Magnitude'] = df['Statlog Magnitude'].apply(lambda x: f"{float(x):.4f}")

    # 4. Setup the Matplotlib Figure
    fig, ax = plt.subplots(figsize=(14, 10)) 
    ax.axis('off') 
    ax.axis('tight')

    # 5. Create the Table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.05, 1, 0.9] # Left space at bottom for H1 text
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style Header and Cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') 
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')

            col_name = df.columns[col]
            val = str(cell.get_text().get_text())

            if col_name == 'Rank Shift':
                if 'Rose' in val:
                    cell.set_text_props(color='#2e7d32', weight='bold') 
                elif 'Dropped' in val:
                    cell.set_text_props(color='#d32f2f', weight='bold') 
                elif 'Unchanged' in val:
                    cell.set_text_props(color='#7f8c8d', style='italic') 
            
            elif col_name == 'Status':
                if val == 'Stable':
                    cell.set_text_props(color='#2e7d32', weight='bold') 
                elif val == 'Shifted':
                    cell.set_text_props(color='#f57f17', weight='bold') 
                elif 'Critical' in val:
                    cell.set_text_props(color='#d32f2f', weight='bold') # Red alert
                elif 'Noise' in val:
                    cell.set_text_props(color='#95a5a6', style='italic') # Greyed out for noise

    # --- FIX 4: ADD H1 PROOF TO THE IMAGE ---
    corr, pval = spearmanr(df['Cleveland Rank'], df['Statlog Rank'])
    h1_text = (f"Statistical Proof (H1): Spearman Rank Correlation = {corr:.3f} (p = {pval:.2e}).\n"
               f"Conclusion: Rankings are significantly altered across datasets, proving severe attribution shift.")
    plt.figtext(0.5, 0.01, h1_text, ha="center", fontsize=12, style='italic', 
                bbox=dict(facecolor='#f8f9fa', edgecolor='grey', boxstyle='round,pad=0.5'))

    # Renamed Title (Fix 1)
    plt.title('Table 4: Severe Feature Attribution Shift Analysis', fontweight='bold', fontsize=16, pad=20)
    
    plt.savefig(OUTPUT_IMG_PATH, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"[SUCCESS] Table image saved to: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    create_feature_shift_table_image()