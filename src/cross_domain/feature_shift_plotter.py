import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cleveland_path = os.path.join(BASE_DIR, 'data', 'processed', 'balanced', 'cleveland_smoteenn.csv')
statlog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned', 'statlog_final.csv')

def plot_feature_shift():
    c_df = pd.read_csv(cleveland_path)
    s_df = pd.read_csv(statlog_path)
    
    # Add labels for comparison
    c_df['Dataset'] = 'Cleveland (Source)'
    s_df['Dataset'] = 'Statlog (Target)'
    
    combined = pd.concat([c_df, s_df])
    
    # Select top features to visualize shift (e.g., thalach, oldpeak, age)
    features_to_plot = ['thalach', 'oldpeak', 'age', 'trestbps']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features_to_plot):
        plt.subplot(2, 2, i+1)
        sns.kdeplot(data=combined, x=col, hue='Dataset', fill=True, common_norm=False, palette='viridis')
        plt.title(f'Distribution Shift: {col}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'results', 'feature_distribution_shift.png'))
    print("SUCCESS: Feature shift visualization saved to results folder.")

if __name__ == "__main__":
    plot_feature_shift()