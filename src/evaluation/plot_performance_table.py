import pandas as pd
import matplotlib.pyplot as plt
import os

def create_table_image():
    # 1. Define Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    CSV_PATH = os.path.join(BASE_DIR, 'results', 'evaluation', 'cross_domain_comparison_table.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'evaluation')
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'cross_domain_performance_table.png')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Read the existing CSV data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}. Make sure the CSV exists.")
        return

    # 3. Setup the Matplotlib Figure
    fig, ax = plt.subplots(figsize=(10, 3.5)) # Adjust width/height as needed
    ax.axis('off') # Hide the axes
    ax.axis('tight')

    # 4. Create the Table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1] # Expand table to fill the figure
    )

    # 5. Style the Table (Academic Look)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style Header and Cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd') # Light grey borders
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') # Dark blue/grey header
        else:
            # Color the "Shift" column (Index 3) to highlight drops
            if col == 3:
                val = str(cell.get_text().get_text())
                if '-' in val:
                    cell.set_text_props(color='#d32f2f', weight='bold') # Red text for drops
            
            # Alternate row colors for readability
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')

    # 6. Add Title and Save
    plt.title('Table 3: Cross-Domain Performance Table', fontweight='bold', fontsize=14, pad=20)
    
    plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"[SUCCESS] Table image saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_table_image()