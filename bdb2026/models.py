from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _lazy_tf():
    # Lazy import so non-TF parts of the package can be used without TensorFlow installed.
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import Sequential  # type: ignore
    from tensorflow.keras.layers import Masking, LSTM, Dropout, TimeDistributed, Dense  # type: ignore
    return tf, keras, Sequential, Masking, LSTM, Dropout, TimeDistributed, Dense


@dataclass
class TrainResult:
    model: object
    history: object


def build_lstm(
    *,
    n_features: int,
    hidden_units: int = 64,
    dropout: float = 0.2,
    mask_value: float = 0.0,
    timesteps: Optional[int] = None,
    learning_rate: float = 0.001,
) -> object:
    """Build the convergence LSTM model in the exact structure you specified.

    Architecture (exact):
      Masking(mask_value=0.0)
      LSTM(64, return_sequences=True) + Dropout(0.3)
      LSTM(32, return_sequences=True) + Dropout(0.2)
      TimeDistributed(Dense(16, relu))
      TimeDistributed(Dense(1, linear))

    Notes:
      - Output shape is (batch, T, 1). If your y is (batch, T), train_lstm() will expand dims.
      - `hidden_units` and `dropout` are kept for API compatibility, but the layer sizes/dropouts
        are fixed to match the requested structure (64/32 units; 0.3/0.2 dropout).
    """
    tf, keras, Sequential, Masking, LSTM, Dropout, TimeDistributed, Dense = _lazy_tf()

    input_shape = (timesteps, n_features) if timesteps is not None else (None, n_features)

    model = Sequential([
        Masking(mask_value=mask_value, input_shape=input_shape),
        LSTM(64, return_sequences=True, name="lstm_1"),
        Dropout(0.3, name="dropout_1"),
        LSTM(32, return_sequences=True, name="lstm_2"),
        Dropout(0.2, name="dropout_2"),
        TimeDistributed(Dense(16, activation="relu"), name="dense_1"),
        TimeDistributed(Dense(1, activation="linear"), name="output"),
    ], name="convergence_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _ensure_y_shape(y: np.ndarray) -> np.ndarray:
    """Ensure y is (N, T, 1) to match TimeDistributed(Dense(1)) output."""
    y = np.asarray(y)
    if y.ndim == 2:
        return y[..., np.newaxis]
    if y.ndim == 3 and y.shape[-1] == 1:
        return y
    raise ValueError(f"y must have shape (N,T) or (N,T,1). Got {y.shape}.")


def _compute_timestep_weights(X: np.ndarray, mask_value: float = 0.0) -> np.ndarray:
    """Per-timestep weights: 1 for non-padded steps, 0 for padded (all mask_value)."""
    X = np.asarray(X)
    nonpad = ~(np.all(np.isclose(X, mask_value), axis=-1))
    return nonpad.astype(np.float32)


def train_lstm(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    reduce_lr_patience: int = 3,
    min_lr: float = 1e-6,
    mask_value: float = 0.0,
    use_sample_weights: bool = True,
    verbose: int = 1,
) -> TrainResult:
    """Train the model, aligning y shape to model output and masking padded timesteps."""
    tf, keras, *_ = _lazy_tf()

    y_train_ = _ensure_y_shape(y_train)
    sw = _compute_timestep_weights(X_train, mask_value=mask_value) if use_sample_weights else None

    monitor = "val_loss" if (X_val is not None and y_val is not None) else "loss"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=reduce_lr_patience, min_lr=min_lr, factor=0.5),
    ]

    if X_val is not None and y_val is not None:
        y_val_ = _ensure_y_shape(y_val)
        val_sw = _compute_timestep_weights(X_val, mask_value=mask_value) if use_sample_weights else None
        # When providing validation sample weights in Keras, include them as the 3rd element.
        validation_data = (X_val, y_val_, val_sw) if use_sample_weights else (X_val, y_val_)
        history = model.fit(
            X_train, y_train_,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            sample_weight=sw,
            validation_data=validation_data,
            verbose=verbose,
        )
    else:
        history = model.fit(
            X_train, y_train_,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            sample_weight=sw,
            verbose=verbose,
        )

    return TrainResult(model=model, history=history)
