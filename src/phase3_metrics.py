import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_route_exec_iq(df_routes):
    """
    Calculates RouteExecIQ based on deviation from cluster centroids.
    """
    print("Calculating RouteExecIQ...")
    
    # Define features used for clustering (from Phase 1)
    # Assuming these columns exist in df_routes
    route_feature_cols = [
        'route_embed_0', 'route_embed_1', 'route_embed_2', 'route_embed_3',
        'route_embed_4', 'route_embed_5', 'route_embed_6', 'route_embed_7'
    ]
    # Adjust columns if necessary based on actual data
    available_cols = [c for c in df_routes.columns if c.startswith('route_embed_')]
    if not available_cols:
        print("Error: No route embedding columns found.")
        return df_routes
    
    # Use available embedding columns
    route_feature_cols = available_cols
    
    if 'route_cluster' not in df_routes.columns:
        print("Error: 'route_cluster' column missing.")
        return df_routes

    # Calculate centroids
    cluster_centroids = df_routes.groupby('route_cluster')[route_feature_cols].mean()
    
    # Calculate deviation
    def get_deviation(row):
        cluster = row['route_cluster']
        if cluster not in cluster_centroids.index:
            return np.nan
        centroid = cluster_centroids.loc[cluster].values
        features = row[route_feature_cols].values
        return euclidean(features, centroid)
        
    df_routes['route_deviation'] = df_routes.apply(get_deviation, axis=1)
    
    # Standardize within cluster
    # Lower deviation is better -> negate z-score
    def standardize(group):
        if len(group) < 2:
            group['RouteExecIQ'] = 0
            return group
        mean = group['route_deviation'].mean()
        std = group['route_deviation'].std()
        if std == 0:
            group['RouteExecIQ'] = 0
        else:
            group['RouteExecIQ'] = -(group['route_deviation'] - mean) / std
        return group
        
    df_routes = df_routes.groupby('route_cluster', group_keys=False).apply(standardize)
    
    print("RouteExecIQ calculated.")
    return df_routes

def calculate_convergence_iq(df_preds):
    """
    Calculates ConvergenceIQ from LSTM predictions.
    """
    print("Calculating ConvergenceIQ...")
    
    # Aggregate predictions by play
    # Assuming 'predicted_convergence' is the column
    if 'predicted_convergence' not in df_preds.columns:
        print("Error: 'predicted_convergence' column missing.")
        return df_preds
        
    # Mean convergence per play
    play_convergence = df_preds.groupby(['game_id', 'play_id'])['predicted_convergence'].mean().reset_index()
    play_convergence.rename(columns={'predicted_convergence': 'avg_convergence'}, inplace=True)
    
    # Standardize
    scaler = StandardScaler()
    play_convergence['ConvergenceIQ_WR'] = scaler.fit_transform(play_convergence[['avg_convergence']])
    
    print("ConvergenceIQ calculated.")
    return play_convergence

def calculate_air_play_iq(df_metrics):
    """
    Calculates AirPlayIQ using the weighted probabilistic formula.
    """
    print("Calculating AirPlayIQ...")
    
    # Normalize ConvergenceIQ to 0-1 for weighting
    scaler = MinMaxScaler()
    df_metrics['convergence_norm'] = scaler.fit_transform(df_metrics[['ConvergenceIQ_WR']])
    
    # Weight decays as convergence improves (or worsens? need to check logic)
    # "Linear decay model: weight = 1 - normalized_convergence"
    # Assuming higher convergence is better?
    # If convergence is high (1), weight is 0 -> AirPlayIQ depends on ConvergenceIQ?
    # Wait, the formula was:
    # AirPlayIQ = weight * RouteExecIQ + (1-weight) * ConvergenceIQ
    # If weight = 1 - conv, and conv is high (1), weight is 0.
    # Then AirPlayIQ = 0 * RouteExecIQ + 1 * ConvergenceIQ.
    # So if convergence is good, it dominates. That makes sense.
    
    df_metrics['weight'] = 1 - df_metrics['convergence_norm']
    
    df_metrics['AirPlayIQ_WR'] = (
        df_metrics['weight'] * df_metrics['RouteExecIQ'] +
        (1 - df_metrics['weight']) * df_metrics['ConvergenceIQ_WR']
    )
    
    print("AirPlayIQ calculated.")
    return df_metrics
