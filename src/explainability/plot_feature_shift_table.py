import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def create_feature_shift_table_image():
    # 1. Define Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    CSV_PATH = os.path.join(BASE_DIR, 'results', 'explainability', 'advanced_feature_shift_table.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'explainability')
    OUTPUT_IMG_PATH = os.path.join(OUTPUT_DIR, 'advanced_feature_shift_table.png')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Read the existing CSV data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}. Make sure the CSV exists.")
        return

    # 3. Dynamically calculate the 'Status' column if it doesn't exist yet
    def determine_status(shift_str):
        if 'Unchanged' in str(shift_str):
            return 'Stable'
        
        # Extract the number from strings like 'Dropped 1' or 'Rose 4'
        match = re.search(r'\d+', str(shift_str))
        if match:
            shift_val = int(match.group())
            # If a feature jumps or drops by 4 or more ranks, it's highly unstable
            if shift_val >= 4:
                return 'Highly Unstable'
            else:
                return 'Shifted'
        return 'Unknown'

    if 'Status' not in df.columns:
        df['Status'] = df['Rank Shift'].apply(determine_status)
        # Save the updated CSV with the Status column back to the file
        df.to_csv(CSV_PATH, index=False)
        print(f"[SUCCESS] Appended 'Status' column and saved CSV: {CSV_PATH}")

    # Clean up magnitude columns to show consistent 4 decimal places
    df['Cleveland Magnitude'] = df['Cleveland Magnitude'].apply(lambda x: f"{float(x):.4f}")
    df['Statlog Magnitude'] = df['Statlog Magnitude'].apply(lambda x: f"{float(x):.4f}")

    # 4. Setup the Matplotlib Figure (slightly wider to fit the new column)
    fig, ax = plt.subplots(figsize=(13, 10)) 
    ax.axis('off') 
    ax.axis('tight')

    # 5. Create the Table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1] # Expand table to fill the figure
    )

    # 6. Style the Table (Academic & Color-Coded Look)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style Header and Cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd') # Light grey borders
        
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') # Dark blue/grey header
        else:
            # Alternate row colors for readability
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')

            # --- DYNAMIC COLOR CODING FOR READABILITY ---
            col_name = df.columns[col]
            val = str(cell.get_text().get_text())

            # Highlight the "Rank Shift" column
            if col_name == 'Rank Shift':
                if 'Rose' in val:
                    cell.set_text_props(color='#2e7d32', weight='bold') # Dark Green
                elif 'Dropped' in val:
                    cell.set_text_props(color='#d32f2f', weight='bold') # Red
                elif 'Unchanged' in val:
                    cell.set_text_props(color='#7f8c8d', style='italic') # Grey italic
            
            # Highlight "Logic Inverted?" column
            elif col_name == 'Logic Inverted?':
                if val == 'Yes':
                    cell.set_text_props(color='#d32f2f', weight='bold') # Red alert
                else:
                    cell.set_text_props(color='#7f8c8d')
            
            # Highlight the new "Status" column
            elif col_name == 'Status':
                if val == 'Stable':
                    cell.set_text_props(color='#2e7d32', weight='bold') # Green
                elif val == 'Shifted':
                    cell.set_text_props(color='#f57f17', weight='bold') # Amber/Orange
                elif val == 'Highly Unstable':
                    cell.set_text_props(color='#d32f2f', weight='bold') # Red

    # 7. Add Title and Save
    plt.title('Table 4: Advanced Feature Attribution Shift Analysis', fontweight='bold', fontsize=16, pad=20)
    
    plt.savefig(OUTPUT_IMG_PATH, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"[SUCCESS] Table image saved to: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    create_feature_shift_table_image()