import pandas as pd
import os

# Default path
DATA_DIR = r"C:\Users\rajas\Documents\ADS\SII\Big_Data_Bowl"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "wr_routes_embeddings.csv")

def load_route_embeddings(filepath=EMBEDDINGS_FILE):
    """
    Loads the pre-calculated route embeddings from Phase 1.
    """
    print(f"Loading route embeddings from {filepath}...")
    try:
        df_embeddings = pd.read_csv(filepath)
        print(f"  - Route embeddings shape: {df_embeddings.shape}")
        return df_embeddings
    except FileNotFoundError:
        print(f"Error: Route embeddings file not found at {filepath}")
        return None

def merge_route_embeddings(df_postthrow, df_embeddings):
    """
    Merges route embeddings into the post-throw dataframe.
    """
    print("Merging route embeddings...")
    
    # Select embedding columns
    # Assuming columns start with 'route_embed_'
    embed_cols = ['game_id', 'play_id'] + [col for col in df_embeddings.columns if col.startswith('route_embed_')]
    
    # Check if we have embeddings
    if len(embed_cols) <= 2:
        print("Warning: No route embedding columns found.")
        return df_postthrow
        
    route_embeds = df_embeddings[embed_cols].copy()
    
    # Merge
    df_merged = df_postthrow.merge(route_embeds, on=['game_id', 'play_id'], how='left')
    
    print(f"Route embeddings merged. Total columns: {len(df_merged.columns)}")
    return df_merged
