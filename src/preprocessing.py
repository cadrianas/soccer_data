import pandas as pd
import numpy as np
from src.utils.paths import DATA_RAW, DATA_PROCESSED

def load_and_clean_data(filename='2022-2023 Football Team Stats.csv'):
    """Loads raw data, cleans columns, and forces numeric types."""
    file_path = DATA_RAW / filename
    df = pd.read_csv(file_path, sep=';', encoding='ISO-8859-1')

    # Drop duplicates
    df = df.drop_duplicates(subset=['Squad'])

    # Clean strings
    df['Country'] = df['Country'].astype(str).str.strip()
    df['Squad'] = df['Squad'].astype(str).str.strip()

    # Handle placeholders
    placeholders = ['N/A', 'n/a', '-', '—', 'None', 'null']
    df = df.replace(placeholders, np.nan)

    # Force numeric conversion
    numeric_cols = ['Rk', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP',
                    'xG', 'xGA', 'xGD', 'Attendance']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def engineer_features(df):
    """Calculates matchday rates and efficiency metrics."""
    # Rate normalization
    df['GF_per_game'] = df['GF'] / df['MP']
    df['GA_per_game'] = df['GA'] / df['MP']
    df['xG_per_game'] = df['xG'] / df['MP']
    df['xGA_per_game'] = df['xGA'] / df['MP']
    df['Pts_per_game'] = df['Pts/MP']

    # Finishing variance
    df['Finishing_Diff_per_game'] = (df['GF'] - df['xG']) / df['MP']
    
    return df

def generate_league_labels(group):
    """Generates intra-league quantile labels for classification."""
    # Binary Label: Elite Top 15% within this specific league boundary
    top_15_thresh = group['Rk'].quantile(0.15)
    group['Binary_Label'] = (group['Rk'] <= top_15_thresh).astype(int)

    # Multi-Class Label: Quantile boundaries split at 20% and 60%
    q20 = group['Rk'].quantile(0.20)
    q60 = group['Rk'].quantile(0.60)

    conditions = [
        (group['Rk'] <= q20),                 # Elite
        (group['Rk'] > q20) & (group['Rk'] <= q60), # Mid-Tier
        (group['Rk'] > q60)                   # Bottom Tier
    ]
    choices = [1, 2, 3]
    group['MultiClass_Label'] = np.select(conditions, choices, default=2)
    return group

def apply_labels(df):
    """Applies league-specific labels to the entire dataframe while preserving columns."""
    dfs = []
    for _, group in df.groupby('Country'):
        dfs.append(generate_league_labels(group))
    return pd.concat(dfs).sort_index()

def run_preprocessing_pipeline():
    """Execution wrapper to load, engineer, label, and save data."""
    df = load_and_clean_data()
    df = engineer_features(df)
    df = apply_labels(df)
    
    # Save to processed folder
    output_path = DATA_PROCESSED / "processed_football_stats.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully saved engineered and labeled data to: {output_path}")
    return df

if __name__ == "__main__":
    run_preprocessing_pipeline()
