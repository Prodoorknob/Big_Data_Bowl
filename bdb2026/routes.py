"""Route modeling utilities (Phase 1 in the notebook).

These functions are intentionally generic: they operate on a tracking table
filtered to the targeted receiver (or any route runner) for the pre-throw window.

Core output:
- A per-(game_id, play_id, nfl_id) route feature table usable for clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import GAME_ID, PLAY_ID, NFL_ID, FRAME_ID, X_NORM, Y_NORM, SPEED


@dataclass(frozen=True)
class RouteClusterResult:
    """Results from k-means clustering over route feature vectors."""

    features: pd.DataFrame            # per-play feature table
    scaler: StandardScaler
    kmeans: KMeans
    assignments: pd.DataFrame         # (game_id, play_id, nfl_id, route_cluster)
    centroids: pd.DataFrame           # cluster centroids in original feature space

def engineer_route_features(
    df_routes: pd.DataFrame,
    *,
    id_cols: Tuple[str, str, str] = (GAME_ID, PLAY_ID, NFL_ID),
    frame_col: str = FRAME_ID,
    x_col: str = X_NORM,
    y_col: str = Y_NORM,
    speed_col: str = SPEED,
    accel_col: str = "a",
    dir_col: str = "dir",
) -> pd.DataFrame:
    """Engineer per-route (per play/player) features from pre-throw tracking rows.
    
    Creates 15 route features matching the notebook implementation:
    1. route_len (route_distance)
    2. delta_x (route_depth) 
    3. delta_y_abs (route_width - ABSOLUTE VALUE)
    4. x_end, y_end (final position)
    5. max_speed, mean_speed, std_speed
    6. max_accel, mean_accel (if available)
    7. direction_changes (>30 degree changes)
    8. route_frames, route_duration
    9. lateral_range (max-min Y)
    10. straightness (route_efficiency)
    """
    df = df_routes.copy()
    df = df.sort_values(list(id_cols) + [frame_col])

    g = df.groupby(list(id_cols), sort=False)

    # Start / end
    x0 = g[x_col].first()
    y0 = g[y_col].first()
    x1 = g[x_col].last()
    y1 = g[y_col].last()

    dx = x1 - x0
    dy = y1 - y0

    # Robust route length using shift
    df["_x_prev"] = g[x_col].shift(1)
    df["_y_prev"] = g[y_col].shift(1)
    df["_step_dist"] = np.sqrt((df[x_col] - df["_x_prev"]) ** 2 + (df[y_col] - df["_y_prev"]) ** 2)
    df["_step_dist"] = df["_step_dist"].fillna(0.0)

    route_len = g["_step_dist"].sum()

    # Straight-line distance + straightness
    direct_dist = np.sqrt(dx ** 2 + dy ** 2)
    straightness = (direct_dist / route_len.replace(0, np.nan)).fillna(0.0)

    # Speed summaries
    if speed_col in df.columns:
        mean_speed = g[speed_col].mean()
        max_speed = g[speed_col].max()
        std_speed = g[speed_col].std().fillna(0.0)
    else:
        mean_speed = pd.Series(np.nan, index=x0.index)
        max_speed = pd.Series(np.nan, index=x0.index)
        std_speed = pd.Series(np.nan, index=x0.index)

    # Acceleration summaries (if available)
    if accel_col in df.columns:
        mean_accel = g[accel_col].mean()
        max_accel = g[accel_col].max()
    else:
        mean_accel = pd.Series(0.0, index=x0.index)
        max_accel = pd.Series(0.0, index=x0.index)

    # Direction changes (>30 degrees)
    # This matches the notebook implementation
    if dir_col in df.columns:
        df["_dir_diff"] = g[dir_col].diff().abs()
        # Handle wrap-around: 359->1 is 2 degrees, not 358
        df["_dir_diff"] = df["_dir_diff"].apply(
            lambda x: min(x, 360 - x) if not pd.isna(x) else 0
        )
        direction_changes = g["_dir_diff"].apply(lambda x: (x > 30).sum())
    else:
        direction_changes = pd.Series(0, index=x0.index)

    # Lateral range (max Y - min Y)
    lateral_range = g[y_col].max() - g[y_col].min()

    # Route duration (frames * 0.1 seconds per frame)
    n_frames = g[frame_col].count()
    route_duration = n_frames * 0.1

    feats = pd.DataFrame({
        "route_frames": n_frames,
        "x_start": x0,
        "y_start": y0,
        "x_end": x1,
        "y_end": y1,
        "delta_x": dx,
        "delta_y": dy,
        "delta_y_abs": dy.abs(),  # CRITICAL: absolute value for route width
        "route_len": route_len,
        "direct_dist": direct_dist,
        "straightness": straightness,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "std_speed": std_speed,
        "mean_accel": mean_accel,
        "max_accel": max_accel,
        "direction_changes": direction_changes,
        "lateral_range": lateral_range,
        "route_duration": route_duration,
    }).reset_index()

    return feats
def cluster_routes_kmeans(
    route_features: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    n_clusters: int = 12,
    random_state: int = 42,
    label_col: str = "route_cluster",
) -> RouteClusterResult:
    """Cluster routes using k-means over engineered route features."""
    feats = route_features.copy()

    if feature_cols is None:
        # Default: exclude ID-like and start/end absolute positions which can dominate.
        candidate = [
            "route_frames", "delta_x", "delta_y", "route_len", "direct_dist",
            "straightness", "mean_speed", "max_speed", "std_speed"
        ]
        feature_cols = [c for c in candidate if c in feats.columns]

    Xdf = feats.loc[:, list(feature_cols)].copy()

    # Coerce to numeric; anything weird becomes NaN
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # Replace inf/-inf, then impute
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # Option A (recommended): median imputation
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    # If a column is all-NaN (median becomes NaN), fall back to 0
    Xdf = Xdf.fillna(0.0)

    X = Xdf.to_numpy(dtype=float)
    
    if np.isnan(X).any():
        nan_cols = Xdf.columns[Xdf.isna().any()].tolist()
        raise ValueError(f"NaNs remain in clustering features after imputation. Columns: {nan_cols}")
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(Xs)

    assignments = feats[[GAME_ID, PLAY_ID] + ([NFL_ID] if NFL_ID in feats.columns else [])].copy()
    assignments[label_col] = labels

    # Centroids in original feature space
    cent = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids = pd.DataFrame(cent, columns=list(feature_cols))
    centroids[label_col] = range(n_clusters)

    return RouteClusterResult(
        features=feats,
        scaler=scaler,
        kmeans=kmeans,
        assignments=assignments,
        centroids=centroids,
    )


def make_route_embedding_table(
    assignments: pd.DataFrame,
    *,
    label_col: str = "route_cluster",
    n_clusters: Optional[int] = None,
    prefix: str = "route_emb",
    key_cols: Sequence[str] = (GAME_ID, PLAY_ID, NFL_ID),
    encoding: str = "onehot",
    dtype: str = "float32",
) -> pd.DataFrame:
    """Create a per-play (optionally per-player) route-embedding table.

    Parameters
    ----------
    assignments:
        Output from `cluster_routes(...).assignments`, containing keys and `label_col`.
    label_col:
        Column holding integer cluster IDs.
    n_clusters:
        Total number of clusters. If None, inferred as max(label)+1.
    prefix:
        Prefix for embedding columns.
    key_cols:
        Key columns to keep/merge on. If some keys are missing in `assignments`,
        they are ignored.
    encoding:
        - "onehot": returns one-hot columns {prefix}_0..{prefix}_{K-1}.
        - "id": returns a single integer column `{prefix}_id`.
    dtype:
        dtype for the embedding columns.

    Returns
    -------
    DataFrame with key columns and embedding columns.
    """
    if label_col not in assignments.columns:
        raise ValueError(f"assignments must include '{label_col}'")

    keys = [c for c in key_cols if c in assignments.columns]
    if not keys:
        raise ValueError("No merge keys found in assignments; expected at least one of "
                         f"{list(key_cols)}")

    out = assignments[keys + [label_col]].drop_duplicates().copy()
    labels = out[label_col].astype(int).to_numpy()
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1 if labels.size else 0

    encoding = encoding.lower().strip()
    if encoding == "id":
        out[f"{prefix}_id"] = labels.astype("int32")
        return out.drop(columns=[label_col])

    if encoding != "onehot":
        raise ValueError("encoding must be one of {'onehot','id'}")

    # One-hot
    for k in range(n_clusters):
        out[f"{prefix}_{k}"] = (labels == k).astype(dtype)
    return out.drop(columns=[label_col])
