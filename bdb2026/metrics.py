from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import GAME_ID, PLAY_ID


Agg = Literal["mean", "median", "sum", "mae", "rmse"]


def compute_truespeed(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    id_cols: Sequence[str] = (GAME_ID, PLAY_ID),
    agg: Agg = "mean",
    out_col: str = "TrueSpeed",
) -> pd.DataFrame:
    """Compute TrueSpeed as an aggregation of residuals within each play.

    Definitions
    ----------
    residual_t = actual_t - pred_t

    Aggregation options:
    - mean: mean residual over timesteps (signed; positive = better-than-expected convergence)
    - median: median residual
    - sum: sum of residuals (scaled by timesteps)
    - mae: mean absolute error (unsigned; magnitude only)
    - rmse: root mean squared error (unsigned; magnitude only)

    Returns
    -------
    DataFrame with id_cols and out_col.
    """
    work = df.copy()
    residual = work[actual_col].astype(float) - work[pred_col].astype(float)

    if agg == "mean":
        score = residual.groupby(work[list(id_cols)].apply(tuple, axis=1)).mean()
    elif agg == "median":
        score = residual.groupby(work[list(id_cols)].apply(tuple, axis=1)).median()
    elif agg == "sum":
        score = residual.groupby(work[list(id_cols)].apply(tuple, axis=1)).sum()
    elif agg == "mae":
        score = residual.abs().groupby(work[list(id_cols)].apply(tuple, axis=1)).mean()
    elif agg == "rmse":
        score = (residual ** 2).groupby(work[list(id_cols)].apply(tuple, axis=1)).mean().pow(0.5)
    else:  # pragma: no cover
        raise ValueError(f"Unknown agg={agg}")

    out = pd.DataFrame(list(score.index), columns=list(id_cols))
    out[out_col] = score.to_numpy()
    return out


def calculate_route_deviation(
    df: pd.DataFrame,
    *,
    actual_x_col: str,
    actual_y_col: str,
    expected_x_col: str,
    expected_y_col: str,
    id_cols: Sequence[str] = (GAME_ID, PLAY_ID),
    agg: Agg = "mean",
    out_col: str = "route_deviation",
) -> pd.DataFrame:
    """Compute Euclidean deviation between two (x,y) trajectories and aggregate by play."""
    work = df.copy()
    dx = work[actual_x_col].astype(float) - work[expected_x_col].astype(float)
    dy = work[actual_y_col].astype(float) - work[expected_y_col].astype(float)
    dist = np.sqrt(dx ** 2 + dy ** 2)

    key = work[list(id_cols)].apply(tuple, axis=1)
    if agg == "mean":
        score = dist.groupby(key).mean()
    elif agg == "median":
        score = dist.groupby(key).median()
    elif agg == "sum":
        score = dist.groupby(key).sum()
    elif agg == "mae":
        score = dist.groupby(key).mean()  # same as mean for distances
    elif agg == "rmse":
        score = (dist ** 2).groupby(key).mean().pow(0.5)
    else:  # pragma: no cover
        raise ValueError(f"Unknown agg={agg}")

    out = pd.DataFrame(list(score.index), columns=list(id_cols))
    out[out_col] = score.to_numpy()
    return out


def standardize_within_cluster(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    metric_cols: Sequence[str],
    z_clip: Optional[float] = 3.0,
    suffix: str = "_z",
) -> pd.DataFrame:
    """Z-score standardization per cluster for radar charts / comparisons.

    Returns a copy with new columns {col}{suffix}.
    """
    out = df.copy()

    for col in metric_cols:
        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            if sd == 0 or np.isnan(sd):
                return pd.Series(np.zeros(len(s)), index=s.index)
            z = (s - mu) / sd
            if z_clip is not None:
                z = z.clip(-z_clip, z_clip)
            return z

        out[col + suffix] = out.groupby(cluster_col)[col].transform(_z)

    return out
