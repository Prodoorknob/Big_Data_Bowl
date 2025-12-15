"""Sequence building utilities for the LSTM phase.

This repo’s notebook-style usage expects:
    X, y, keys = build_sequences(df_features, feature_cols=..., target_col="converge_rate", max_len=25)

Design choice:
- Sequence by (game_id, play_id). This matches your stated approach and the assumption
  that you have exactly one modeled target receiver per play after filtering.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import GAME_ID, PLAY_ID, FRAME_ID


def build_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    target_col: str,
    max_len: int = 25,
    id_cols: Tuple[str, str] = (GAME_ID, PLAY_ID),
    frame_col: str = FRAME_ID,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build fixed-length sequences grouped by (game_id, play_id).

    Parameters
    ----------
    df:
        Frame-level feature table (one row per frame for the modeled player).
    feature_cols:
        Feature columns to include in X.
    target_col:
        Frame-level target column (e.g., 'converge_rate').
    max_len:
        Maximum sequence length. Longer sequences are truncated to the *last* max_len rows.
    id_cols:
        Sequence keys. Default (game_id, play_id).
    frame_col:
        Frame ordering column.
    pad_value:
        Value used to pad sequences shorter than max_len.

    Returns
    -------
    X:
        (N, max_len, F) float32 tensor
    y:
        (N, max_len) float32 tensor
    keys:
        DataFrame with one row per sequence and columns id_cols
    """
    required = [*id_cols, frame_col, target_col, *feature_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"build_sequences missing columns: {missing}")

    d = df.sort_values([*id_cols, frame_col]).copy()

    keys = d.loc[:, list(id_cols)].drop_duplicates().reset_index(drop=True)
    n_seq = len(keys)
    n_feat = len(feature_cols)

    X = np.full((n_seq, max_len, n_feat), pad_value, dtype=np.float32)
    y = np.full((n_seq, max_len), pad_value, dtype=np.float32)

    key_to_idx = {tuple(row): i for i, row in enumerate(keys.to_numpy())}

    for key, grp in d.groupby(list(id_cols), sort=False):
        # pandas gives key as tuple when len(id_cols) > 1
        if not isinstance(key, tuple):
            key = (key,)
        i = key_to_idx[key]

        # Keep the last max_len frames (post-throw window is typically at the end)
        grp = grp.tail(max_len)
        L = len(grp)

        X[i, :L, :] = grp[feature_cols].astype(float).to_numpy()
        y[i, :L] = grp[target_col].astype(float).to_numpy()

    return X, y, keys


# Backwards-compatible aliases (if your older notebooks call these)
build_lstm_tensors = build_sequences
