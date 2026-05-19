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
    numeric_cols = ['MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP', 
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

def run_preprocessing_pipeline():
    """Execution wrapper to load, engineer, and save data."""
    df = load_and_clean_data()
    df = engineer_features(df)
    
    # Save to processed folder
    output_path = DATA_PROCESSED / "processed_football_stats.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully saved engineered data to: {output_path}")

if __name__ == "__main__":
    run_preprocessing_pipeline()