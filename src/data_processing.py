import pandas as pd
import numpy as np
import os

# Default paths based on the user's environment
DATA_DIR = r"C:\Users\rajas\Documents\ADS\SII\Big_Data_Bowl\data\114239_nfl_competition_files_published_analytics_final\combined"
INPUT_FILE = os.path.join(DATA_DIR, "group_input.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "group_output.csv")
SUPP_FILE = os.path.join(DATA_DIR, "supplementary_data.csv")

def load_data(input_path=INPUT_FILE, output_path=OUTPUT_FILE, supp_path=SUPP_FILE):
    """
    Loads the input, output, and supplementary dataframes.
    """
    print("Loading data...")
    try:
        df_input = pd.read_csv(input_path, low_memory=False)
        df_output = pd.read_csv(output_path)
        df_supp = pd.read_csv(supp_path)
        print(f"  - Input data shape: {df_input.shape}")
        print(f"  - Output data shape: {df_output.shape}")
        print(f"  - Supplementary data shape: {df_supp.shape}")
        return df_input, df_output, df_supp
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def normalize_coordinates(df):
    """
    Normalizes x and y coordinates so that all plays go from left to right.
    """
    print("Normalizing coordinates...")
    df['x_norm'] = df['x']
    df['y_norm'] = df['y']
    
    # Flip coordinates for plays going left
    mask_left = df['play_direction'] == 'left'
    df.loc[mask_left, 'x_norm'] = 120 - df.loc[mask_left, 'x']
    df.loc[mask_left, 'y_norm'] = 53.3 - df.loc[mask_left, 'y']
    
    # Normalize ball landing coordinates if present
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        df['ball_land_x_norm'] = df['ball_land_x']
        df['ball_land_y_norm'] = df['ball_land_y']
        df.loc[mask_left, 'ball_land_x_norm'] = 120 - df.loc[mask_left, 'ball_land_x']
        df.loc[mask_left, 'ball_land_y_norm'] = 53.3 - df.loc[mask_left, 'ball_land_y']
        
    print("Normalization complete.")
    return df

def filter_targeted_receivers(df_input):
    """
    Filters the input dataframe to include only the targeted receiver for each play.
    """
    print("Filtering for targeted receivers...")
    # Filter for the player we need to predict
    df_targeted = df_input[df_input['player_to_predict'] == True].copy()
    
    # Ensure we only have WRs (Wide Receivers) - Optional, based on notebook logic
    # The notebook filtered for WRs later, but we can check if that's needed.
    # For now, we'll stick to player_to_predict=True.
    
    print(f"  - Targeted receivers shape: {df_targeted.shape}")
    return df_targeted

def merge_pass_results(df, df_supp):
    """
    Merges pass results from supplementary data into the main dataframe.
    """
    print("Merging pass results...")
    if 'pass_result' not in df.columns:
        pass_outcomes = df_supp[['game_id', 'play_id', 'pass_result']].drop_duplicates()
        df = df.merge(pass_outcomes, on=['game_id', 'play_id'], how='left')
    return df
