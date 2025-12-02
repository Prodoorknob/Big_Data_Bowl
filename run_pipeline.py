import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src import data_processing as dp
from src import feature_engineering as fe
from src import phase1_route_modeling as p1
from src import phase2_convergence as p2
from src import phase3_metrics as p3

def main():
    print("Starting Big Data Bowl 2026 Pipeline...")
    
    # 1. Load Data
    df_input, df_output, df_supp = dp.load_data()
    if df_input is None:
        return

    # 2. Normalize Coordinates
    df_input = dp.normalize_coordinates(df_input)
    
    # 3. Filter Targeted Receivers
    df_targeted = dp.filter_targeted_receivers(df_input)
    
    # 4. Merge Pass Results
    df_targeted = dp.merge_pass_results(df_targeted, df_supp)
    
    # 5. Calculate Geometry Features
    df_features = fe.calculate_geometry_features(df_targeted)
    
    # 6. Calculate Initial Separation
    df_features = fe.calculate_initial_separation(df_features)
    
    # 7. Calculate Convergence Rate
    df_features = fe.calculate_convergence_rate(df_features)
    
    # 8. Filter Training Data (Phase 2 Requirement)
    df_train_pool = p2.filter_training_data(df_features)
    
    # 9. Load Route Embeddings (Phase 1 Output)
    df_embeddings = p1.load_route_embeddings()
    
    # 10. Merge Route Embeddings
    if df_embeddings is not None:
        df_train_pool = p1.merge_route_embeddings(df_train_pool, df_embeddings)
        
    # 11. Build Sequences
    # Select features for LSTM
    # Using a subset of features for now
    feature_cols = ['x_norm', 'y_norm', 's', 'a', 'dir', 'o', 
                    'dist_to_land', 'bearing_to_land', 'heading_align_cos']
    
    # Add embedding cols if available
    if df_embeddings is not None:
        embed_cols = [c for c in df_embeddings.columns if c.startswith('route_embed_')]
        feature_cols.extend(embed_cols)
        
    # Ensure all columns exist
    feature_cols = [c for c in feature_cols if c in df_train_pool.columns]
    
    print(f"Using {len(feature_cols)} features for LSTM.")
    
    # Drop NaNs in feature columns
    initial_len = len(df_train_pool)
    df_train_pool = df_train_pool.dropna(subset=feature_cols)
    print(f"Dropped {initial_len - len(df_train_pool)} rows with NaNs in features.")
    
    if len(df_train_pool) == 0:
        print("Error: No data left after dropping NaNs.")
        return

    X, y, play_ids = p2.build_sequences(df_train_pool, feature_cols)
    
    # Split Train/Val
    # Simple split for now
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 12. Build and Train Model
    model = p2.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    p2.train_model(model, X_train, y_train, X_val, y_val, epochs=5) # Reduced epochs for testing
    
    # Save Model
    p2.save_model(model, "convergence_lstm_model.h5")
    
    # 13. Predict
    print("Generating predictions...")
    preds = model.predict(X)
    
    # Create predictions dataframe
    # We need to map back to frames. 
    # Since we have play_ids and fixed sequence length, it's a bit tricky to map 1:1 to original frames 
    # if we padded/truncated.
    # For metric calculation, we aggregated by play, so play-level predictions might be enough.
    # But ConvergenceIQ needs frame-level? No, the notebook aggregated.
    
    # Let's store mean prediction per play for now
    play_preds = []
    for i, (game_id, play_id) in enumerate(play_ids):
        # preds[i] is (25, 1)
        # Take mean of non-masked values (if we masked) or just mean
        mean_pred = np.mean(preds[i])
        play_preds.append({
            'game_id': game_id,
            'play_id': play_id,
            'predicted_convergence': mean_pred
        })
        
    df_preds = pd.DataFrame(play_preds)
    df_preds.to_csv("postthrow_predictions_new.csv", index=False)
    
    # 14. Calculate Metrics
    if df_embeddings is not None:
        # Calculate RouteExecIQ
        df_routes_iq = p3.calculate_route_exec_iq(df_embeddings)
        
        # Calculate ConvergenceIQ
        df_conv_iq = p3.calculate_convergence_iq(df_preds)
        
        # Merge
        df_metrics = df_routes_iq.merge(df_conv_iq, on=['game_id', 'play_id'], how='inner')
        
        # Calculate AirPlayIQ
        df_metrics = p3.calculate_air_play_iq(df_metrics)
        
        # Save
        output_file = "air_play_iq_metrics_final.csv"
        df_metrics.to_csv(output_file, index=False)
        print(f"Final metrics saved to {output_file}")
        
        # Display sample
        print(df_metrics[['game_id', 'play_id', 'RouteExecIQ', 'ConvergenceIQ_WR', 'AirPlayIQ_WR']].head())

if __name__ == "__main__":
    main()
