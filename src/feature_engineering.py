import numpy as np
import pandas as pd

def calculate_geometry_features(df):
    """
    Calculates geometry and motion features for each frame.
    """
    print("Calculating geometry features...")
    
    # Sort by play and frame
    df = df.sort_values(['game_id', 'play_id', 'frame_id']).reset_index(drop=True)
    
    # 1. Displacement
    df['dx'] = df.groupby(['game_id', 'play_id'])['x_norm'].diff().fillna(0)
    df['dy'] = df.groupby(['game_id', 'play_id'])['y_norm'].diff().fillna(0)
    
    # 2. Speed (yards per second) - assuming 0.1s per frame
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2) / 0.1
    
    # 3. Distance to landing
    df['dist_to_land'] = np.sqrt(
        (df['x_norm'] - df['ball_land_x_norm'])**2 + 
        (df['y_norm'] - df['ball_land_y_norm'])**2
    )
    
    # 4. Bearing to landing (angle from player to ball, in radians)
    df['bearing_to_land'] = np.arctan2(
        df['ball_land_y_norm'] - df['y_norm'],
        df['ball_land_x_norm'] - df['x_norm']
    )
    
    # 5. Heading (movement direction, in radians)
    df['heading'] = np.arctan2(df['dy'], df['dx'])
    
    # 6. Heading alignment (cosine of angle between heading and bearing)
    # heading_align_cos = 1.0 means moving directly toward target
    # heading_align_cos = -1.0 means moving directly away
    heading_error = df['bearing_to_land'] - df['heading']
    df['heading_align_cos'] = np.cos(heading_error)
    
    # 7. Temporal features
    df['frame_since_throw'] = df.groupby(['game_id', 'play_id']).cumcount()
    df['time_since_throw'] = df['frame_since_throw'] * 0.1
    
    print("Geometry features calculated.")
    return df

def calculate_initial_separation(df):
    """
    Calculates the distance from targeted WR to ball landing at the moment of throw (frame 1).
    """
    print("Calculating initial separation...")
    
    # Get distance at frame 1 for each play
    # Assuming frame_id starts at 1 or we use the first frame available
    initial_sep = df.groupby(['game_id', 'play_id']).first().reset_index()[['game_id', 'play_id', 'dist_to_land']]
    initial_sep = initial_sep.rename(columns={'dist_to_land': 'initial_separation'})
    
    # Merge back
    df = df.merge(initial_sep, on=['game_id', 'play_id'], how='left')
    
    print("Initial separation calculated.")
    return df

def calculate_convergence_rate(df):
    """
    Calculates the rate at which the player is closing distance to the ball landing location.
    converge_rate[t] = dist_to_land[t] - dist_to_land[t+1]
    """
    print("Calculating convergence rate...")
    
    # Shift dist_to_land by -1 to get next frame's distance
    df['dist_to_land_next'] = df.groupby(['game_id', 'play_id'])['dist_to_land'].shift(-1)
    
    # Positive = getting closer, Negative = moving away
    df['converge_rate'] = df['dist_to_land'] - df['dist_to_land_next']
    
    # Fill last frame with 0 (no next frame)
    df['converge_rate'] = df['converge_rate'].fillna(0)
    
    print("Convergence rate calculated.")
    return df
