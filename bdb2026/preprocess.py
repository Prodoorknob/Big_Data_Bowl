"""Data preparation utilities.

Key ideas reflected from the notebook:
- Normalize field direction so offense moves left-to-right (optional, controlled by args).
- Create engineered post-throw features used for sequence modeling.
- Provide lightweight filters for targeted WR route modeling.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Set

import numpy as np
import pandas as pd

from .config import (
    GAME_ID, PLAY_ID, FRAME_ID, NFL_ID,
    X, Y, X_NORM, Y_NORM,
    BALL_LAND_X, BALL_LAND_Y,
    DX, DY, SPEED,
    DIST_TO_LAND, BEARING_TO_LAND,
    HEADING, HEADING_ALIGN_COS,
    TIME_SINCE_THROW,
)


def normalize_to_100(x: pd.Series, *, x_min: float = 0.0, x_max: float = 120.0) -> pd.Series:
    """Normalize an x coordinate to a 0..100 scale.

    This helper mirrors common football-field normalization patterns used in notebooks.
    If your coordinate system already uses 0..120, leave defaults.
    """
    return 100.0 * (x - x_min) / (x_max - x_min)


def normalize_coordinates(
    df: pd.DataFrame,
    *,
    offense_left_to_right: bool = True,
    play_direction_col: str = "play_direction",
    x_col: str = X,
    y_col: str = Y,
    x_out: str = X_NORM,
    y_out: str = Y_NORM,
) -> pd.DataFrame:
    """Create normalized coordinates (x_norm, y_norm).

    If `offense_left_to_right=True` and the dataset includes a `play_direction` column
    with values like 'left'/'right', then plays going left are flipped so offense
    always advances to increasing x.

    Notes
    -----
    - This function is intentionally conservative: if `play_direction_col` is missing,
      it will simply copy x/y to x_norm/y_norm.
    - y is not flipped (the field width is symmetric); keep as-is unless you have a reason.
    """
    out = df.copy()
    out[y_out] = out[y_col].astype(float)

    if not offense_left_to_right or play_direction_col not in out.columns:
        out[x_out] = out[x_col].astype(float)
        return out

    direction = out[play_direction_col].astype(str).str.lower()
    x = out[x_col].astype(float)

    # Typical tracking coordinates: x from 0..120 (end zone to end zone). Flip for 'left'.
    out[x_out] = np.where(direction.eq("left"), 120.0 - x, x)
    return out


def add_derived_features(
    df: pd.DataFrame,
    *,
    group_cols: Tuple[str, str, str] = (GAME_ID, PLAY_ID, NFL_ID),
    frame_col: str = FRAME_ID,
    x_col: str = X_NORM,
    y_col: str = Y_NORM,
    dt_seconds: float = 0.1,
) -> pd.DataFrame:
    """Add dx, dy, speed based on frame-to-frame differences.

    Parameters
    ----------
    dt_seconds:
        Seconds per frame. Tracking is commonly 10 Hz -> 0.1s, but confirm for your artifacts.

    Returns
    -------
    DataFrame with DX, DY, SPEED appended.
    """
    out = df.copy()
    out = out.sort_values(list(group_cols) + [frame_col])

    out[DX] = out.groupby(list(group_cols))[x_col].diff() / dt_seconds
    out[DY] = out.groupby(list(group_cols))[y_col].diff() / dt_seconds
    out[SPEED] = np.sqrt(out[DX].fillna(0) ** 2 + out[DY].fillna(0) ** 2)

    return out


def _bearing(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Angle from (x1,y1) to (x2,y2) in radians."""
    return np.arctan2((y2 - y1), (x2 - x1))


def add_postthrow_features(
    df: pd.DataFrame,
    *,
    x_col: str = X_NORM,
    y_col: str = Y_NORM,
    land_x_col: str = BALL_LAND_X,
    land_y_col: str = BALL_LAND_Y,
    heading_deg_col: Optional[str] = "dir",
    time_since_throw_col: str = TIME_SINCE_THROW,
    dt_seconds: float = 0.1,
) -> pd.DataFrame:
    """Add engineered geometry features to support sequence modeling.

    Adds (if available / computable):
    - dist_to_land: Euclidean distance to ball landing point
    - bearing_to_land: angle from player to landing point (radians)
    - heading: movement heading from `dir` column (deg -> radians) if present
    - heading_align_cos: cos(angle_diff) between heading and bearing_to_land
    - dx, dy, speed (via `add_derived_features`)

    The function is robust to missing columns: it will compute what it can and leave the rest as NaN.
    """
    out = df.copy()

    # dx/dy/speed
    out = add_derived_features(out, x_col=x_col, y_col=y_col, dt_seconds=dt_seconds)

    # distance and bearing to landing point
    if land_x_col in out.columns and land_y_col in out.columns:
        x = out[x_col].astype(float).to_numpy()
        y = out[y_col].astype(float).to_numpy()
        lx = out[land_x_col].astype(float).to_numpy()
        ly = out[land_y_col].astype(float).to_numpy()

        out[DIST_TO_LAND] = np.sqrt((lx - x) ** 2 + (ly - y) ** 2)
        out[BEARING_TO_LAND] = _bearing(x, y, lx, ly)
    else:
        out[DIST_TO_LAND] = np.nan
        out[BEARING_TO_LAND] = np.nan

    # heading + alignment cosine
    if heading_deg_col and heading_deg_col in out.columns:
        heading = np.deg2rad(out[heading_deg_col].astype(float).to_numpy())
        out[HEADING] = heading
        if BEARING_TO_LAND in out.columns:
            diff = heading - out[BEARING_TO_LAND].astype(float).to_numpy()
            out[HEADING_ALIGN_COS] = np.cos(diff)
        else:
            out[HEADING_ALIGN_COS] = np.nan
    else:
        out[HEADING] = np.nan
        out[HEADING_ALIGN_COS] = np.nan

    # time_since_throw: if missing, approximate from frame index within play
    if time_since_throw_col not in out.columns:
        if FRAME_ID in out.columns:
            # Approx: compute relative frame within (game,play,nfl_id) sequence
            rel = out.groupby([GAME_ID, PLAY_ID, NFL_ID])[FRAME_ID].transform(lambda s: s - s.min())
            out[time_since_throw_col] = rel.astype(float) * dt_seconds
        else:
            out[time_since_throw_col] = np.nan

    return out


def merge_route_embeddings(
    postthrow_df: pd.DataFrame,
    route_embeddings: pd.DataFrame,
    *,
    how: str = "left",
    key_cols: Tuple[str, ...] = (GAME_ID, PLAY_ID, NFL_ID),
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Merge per-play route embeddings into the post-throw frame table.

    Route embeddings are typically static per play (and optionally per receiver),
    so after this merge each post-throw frame row will carry the same embedding
    values for that play/receiver.

    Notes
    -----
    - If `route_embeddings` is keyed only by (gameId, playId), the merge still works
      when NFL_ID is missing.
    - Any missing embedding values after merge are filled with `fill_value`
      (useful when a play was not clustered, or was filtered out).

    Parameters
    ----------
    postthrow_df:
        Frame-level post-throw table (the LSTM sequence rows).
    route_embeddings:
        Output of `routes.make_route_embedding_table(...)` or equivalent.
    how:
        Merge type (default "left").
    key_cols:
        Keys to join on. Only the intersection of available columns is used.
    fill_value:
        Value to fill for missing embedding columns after merge.

    Returns
    -------
    DataFrame with embedding columns appended.
    """
    left_keys = [c for c in key_cols if c in postthrow_df.columns and c in route_embeddings.columns]
    if not left_keys:
        raise ValueError("No common key columns between postthrow_df and route_embeddings.")

    out = postthrow_df.merge(route_embeddings, on=left_keys, how=how)

    # Fill embedding columns only (non-key, non-numeric safe fill)
    emb_cols = [c for c in route_embeddings.columns if c not in left_keys]
    for c in emb_cols:
        if c in out.columns:
            out[c] = out[c].fillna(fill_value)
    return out


def filter_targeted_wr_routes(
    df_supp: pd.DataFrame,
    *,
    position_col: str = "player_position",
    position_value: str = "WR",
    route_col: str = "route_of_targeted_receiver",
    drop_routes: Iterable[str] = ("SCREEN", "FLAT", "SHIELD", "ANGLE", "WHEEL"),
) -> pd.DataFrame:
    """Filter supplementary rows down to targeted WR routes and remove noisy route labels."""
    out = df_supp.copy()
    if position_col in out.columns:
        out = out[out[position_col] == position_value].copy()
    if route_col in out.columns and drop_routes:
        out = out[~out[route_col].isin(list(drop_routes))].copy()
    return out

def select_target_receiver_rows(
    df_in: pd.DataFrame,
    *,
    keys: Tuple[str, str, str] = ("game_id", "play_id", "nfl_id"),
    require_wr: bool = True,
) -> pd.DataFrame:
    """
    Select the targeted receiver rows for modeling.

    Operational definition for your dataset:
      - player_to_predict == True (post-throw tracked/prediction target)
      - player_side == offense
      - player_position == WR (optional but default True)
    """
    for col in ("player_to_predict", "player_side", "player_position"):
        if col not in df_in.columns:
            raise KeyError(f"df_in missing required column: {col}")
    for k in keys:
        if k not in df_in.columns:
            raise KeyError(f"df_in missing key: {k}")

    df = df_in[df_in["player_to_predict"] == True].copy()
    df = df[df["player_side"].astype(str).str.lower().eq("offense")].copy()
    if require_wr:
        df = df[df["player_position"].astype(str).str.upper().eq("WR")].copy()
    return df


def filter_to_completed_catches(
    df_in: pd.DataFrame,
    df_supp: pd.DataFrame,
    *,
    pass_result_col: str = "pass_result",
    completed_code: str = "C",
    keys: Tuple[str, str] = ("game_id", "play_id"),
) -> pd.DataFrame:
    """
    Keep only completed passes using df_supp.pass_result == 'C'.
    """
    for k in keys:
        if k not in df_in.columns or k not in df_supp.columns:
            raise KeyError(f"Missing join key '{k}' in df_in/df_supp")
    if pass_result_col not in df_supp.columns:
        raise KeyError(f"df_supp missing column: {pass_result_col}")

    plays_completed = (
        df_supp.loc[:, [*keys, pass_result_col]]
        .drop_duplicates(subset=list(keys))
    )
    plays_completed = plays_completed[
        plays_completed[pass_result_col].astype(str).str.upper().eq(completed_code.upper())
    ].loc[:, list(keys)]

    return df_in.merge(plays_completed, on=list(keys), how="inner")