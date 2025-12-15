"""Sequence building utilities for the LSTM phase.
Key ideas:
- group by play (and usually by targeted receiver)
- for each play, build a fixed-length (max_len) tensor
- pad with mask_value for short plays; truncate long plays
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import GAME_ID, PLAY_ID, NFL_ID, FRAME_ID


PlayKey = Union[Tuple[int, int], Tuple[int, int, int]]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    max_len: int,
    id_cols: Sequence[str] = (GAME_ID, PLAY_ID),
    frame_col: str = FRAME_ID,
    mask_value: float = 0.0,
    dropna_target: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[PlayKey]]:
    """Build padded sequences.

    Parameters
    ----------
    df:
        Long-form frame-level table.
    feature_cols:
        Columns to use as input features.
    target_col:
        Column to use as supervised target per timestep.
    max_len:
        Timesteps to pad/truncate to.
    id_cols:
        Identifiers for the sequence. Default is (game_id, play_id).
        If you want per-player sequences, pass (game_id, play_id, nfl_id).
    mask_value:
        Value used for padding; choose consistently with the Masking layer.
    dropna_target:
        If True, drop rows where target_col is NaN before building sequences.

    Returns
    -------
    X: (n_seq, max_len, n_features)
    y: (n_seq, max_len, 1)
    keys: list of id tuples aligned to sequences
    """
    work = df.copy()
    if dropna_target:
        work = work.dropna(subset=[target_col])

    # Ensure deterministic order
    work = work.sort_values(list(id_cols) + [frame_col])

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    keys: List[PlayKey] = []

    for key, g in work.groupby(list(id_cols), sort=False):
        g = g.sort_values(frame_col)

        Xg = g[list(feature_cols)].astype(float).to_numpy()
        yg = g[[target_col]].astype(float).to_numpy()  # (t, 1)

        # truncate
        Xg = Xg[:max_len]
        yg = yg[:max_len]

        # pad
        t = Xg.shape[0]
        if t < max_len:
            pad_X = np.full((max_len - t, Xg.shape[1]), mask_value, dtype=float)
            pad_y = np.full((max_len - t, 1), mask_value, dtype=float)
            Xg = np.vstack([Xg, pad_X])
            yg = np.vstack([yg, pad_y])

        X_list.append(Xg)
        y_list.append(yg)
        keys.append(key if isinstance(key, tuple) else (key,))

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, max_len, len(feature_cols)))
    y = np.stack(y_list, axis=0) if y_list else np.empty((0, max_len, 1))

    return X, y, keys
