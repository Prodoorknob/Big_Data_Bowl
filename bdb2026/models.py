"""Model definitions and training helpers.

This module keeps tensorflow/keras imports local so notebooks can import the
package even when TF isn't installed. (You'll only error when calling build/train.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TrainResult:
    model: Any
    history: Any


def _require_tf():
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # noqa: F401
        return tf
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for bdb2026.models. Install tensorflow (and a matching protobuf) to use LSTM utilities."
        ) from e


def build_lstm(
    *,
    n_features: int,
    hidden_units: int = 64,
    dropout: float = 0.2,
    mask_value: float = 0.0,
    learning_rate: float = 1e-3,
) -> Any:
    """Build a simple masked LSTM for per-timestep regression.

    Output shape: (batch, timesteps, 1)
    """
    tf = _require_tf()
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(None, n_features))
    x = layers.Masking(mask_value=mask_value)(inputs)
    x = layers.LSTM(hidden_units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)

    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_lstm(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 7,
    reduce_lr_patience: int = 3,
    min_lr: float = 1e-5,
    verbose: int = 1,
) -> TrainResult:
    """Train with EarlyStopping and ReduceLROnPlateau."""
    _require_tf()
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor="val_loss" if X_val is not None else "loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss" if X_val is not None else "loss", patience=reduce_lr_patience, factor=0.5, min_lr=min_lr, verbose=verbose),
    ]

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    return TrainResult(model=model, history=history)
