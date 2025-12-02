import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

def filter_training_data(df):
    """
    Filters training data based on the user's criteria:
    Include routes with Open separation (>3 yards) OR a Catch (Pass==C).
    """
    print("Filtering training data...")
    
    # Check if required columns exist
    if 'pass_result' not in df.columns or 'initial_separation' not in df.columns:
        print("Error: 'pass_result' or 'initial_separation' columns missing.")
        return df
    
    # Apply filter: Catch OR Open Separation (> 3 yards)
    # Note: 'initial_separation' is in yards.
    condition = (df['pass_result'] == 'C') | (df['initial_separation'] > 3.0)
    
    df_filtered = df[condition].copy()
    
    pre_count = df.groupby(['game_id', 'play_id']).ngroups
    post_count = df_filtered.groupby(['game_id', 'play_id']).ngroups
    
    print(f"  - Original Plays: {pre_count}")
    print(f"  - Filtered Plays (Catch OR Sep > 3yds): {post_count}")
    print(f"  - Removed {pre_count - post_count} plays.")
    
    return df_filtered

def build_sequences(df, feature_cols, target_col='converge_rate', max_len=25):
    """
    Builds padded sequences for LSTM input.
    """
    print("Building LSTM sequences...")
    
    X_list = []
    y_list = []
    play_ids = []
    
    # Group by play
    # Note: This can be slow. For production, consider optimizing.
    for (game_id, play_id), group in df.groupby(['game_id', 'play_id']):
        # Sort by frame
        group = group.sort_values('frame_id').reset_index(drop=True)
        
        # Extract features and target
        X = group[feature_cols].values
        y = group[target_col].values.reshape(-1, 1)
        
        seq_len = len(X)
        
        # Pad or truncate
        if seq_len < max_len:
            # Pad with zeros
            X_padded = np.zeros((max_len, X.shape[1]))
            X_padded[:seq_len] = X
            
            y_padded = np.zeros((max_len, 1))
            y_padded[:seq_len] = y
        else:
            # Truncate
            X_padded = X[:max_len]
            y_padded = y[:max_len]
            
        X_list.append(X_padded)
        y_list.append(y_padded)
        play_ids.append((game_id, play_id))
        
    X_seq = np.array(X_list)
    y_seq = np.array(y_list)
    
    print(f"  - X_seq shape: {X_seq.shape}")
    print(f"  - y_seq shape: {y_seq.shape}")
    
    return X_seq, y_seq, play_ids

def build_lstm_model(input_shape):
    """
    Builds the LSTM model architecture.
    """
    print("Building LSTM model...")
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(32, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        TimeDistributed(Dense(1, activation='linear'))
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    """
    Trains the LSTM model.
    """
    print("Training model...")
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5, monitor='val_loss')
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history

def save_model(model, filepath):
    """
    Saves the trained model.
    """
    print(f"Saving model to {filepath}...")
    model.save(filepath)

def load_trained_model(filepath):
    """
    Loads a trained model.
    """
    print(f"Loading model from {filepath}...")
    return tf.keras.models.load_model(filepath)
