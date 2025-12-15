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
) -> pd.DataFrame:
    """Engineer per-route (per play) features.

    Expected input: tracking rows for the route runner during the pre-throw window.
    If you only have raw x/y, call `normalize_coordinates` and `add_derived_features` first.

    Returns
    -------
    DataFrame with one row per (game_id, play_id, nfl_id) and engineered features.
    """
    df = df_routes.copy()
    df = df.sort_values(list(id_cols) + [frame_col])

    # Basic deltas within route
    g = df.groupby(list(id_cols), sort=False)

    x0 = g[x_col].first()
    y0 = g[y_col].first()
    x1 = g[x_col].last()
    y1 = g[y_col].last()

    dx = x1 - x0
    dy = y1 - y0

    # Route length approximated by summing step distances
    step_dist = np.sqrt(g[x_col].diff() ** 2 + g[y_col].diff() ** 2)
    route_len = step_dist.groupby(level=list(range(len(id_cols)))).sum()

    # Straight-line distance
    direct_dist = np.sqrt(dx ** 2 + dy ** 2)

    # Straightness ratio (1 = perfectly straight)
    straightness = (direct_dist / route_len.replace(0, np.nan)).fillna(0)

    # Speed summaries (if available)
    if speed_col in df.columns:
        mean_speed = g[speed_col].mean()
        max_speed = g[speed_col].max()
        std_speed = g[speed_col].std().fillna(0)
    else:
        mean_speed = pd.Series(np.nan, index=x0.index)
        max_speed = pd.Series(np.nan, index=x0.index)
        std_speed = pd.Series(np.nan, index=x0.index)

    n_frames = g[frame_col].count()

    feats = pd.DataFrame({
        "route_frames": n_frames,
        "x_start": x0,
        "y_start": y0,
        "x_end": x1,
        "y_end": y1,
        "delta_x": dx,
        "delta_y": dy,
        "route_len": route_len,
        "direct_dist": direct_dist,
        "straightness": straightness,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "std_speed": std_speed,
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

    X = feats[list(feature_cols)].astype(float).to_numpy()
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
