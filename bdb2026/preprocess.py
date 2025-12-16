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
    BALL_LAND_X, BALL_LAND_Y, BALL_LAND_X_NORM, BALL_LAND_Y_NORM,
    DX, DY, SPEED,
    DIST_TO_LAND, BEARING_TO_LAND,
    HEADING, HEADING_ALIGN_COS,
    TIME_SINCE_THROW,
    CONVERGE_RATE,
    Y_TRUE_X, Y_TRUE_Y, Y_TRUE_X_NORM, Y_TRUE_Y_NORM,
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
    """Create normalized coordinates (x_norm, y_norm) and normalized landing coords.

    If `offense_left_to_right=True` and `play_direction` exists, plays going left are
    flipped so offense always advances to increasing x.

    Notes
    -----
    - y is not flipped (field width symmetric) unless you explicitly choose to.
    - If ball landing columns exist (ball_land_x/ball_land_y), this also creates
      ball_land_x_norm/ball_land_y_norm using the same x-flip logic.
    """
    out = df.copy()

    # Always normalize/cast y
    out[y_out] = out[y_col].astype(float)

    has_dir = play_direction_col in out.columns
    if offense_left_to_right and has_dir:
        direction = out[play_direction_col].astype(str).str.lower()
        x = out[x_col].astype(float)
        out[x_out] = np.where(direction.eq("left"), 120.0 - x, x)
    else:
        # No direction available, or caller opted out of flipping
        out[x_out] = out[x_col].astype(float)
        direction = None  # for landing flip logic

    # Normalize ball landing point to match x_norm/y_norm (if present)
    if BALL_LAND_X in out.columns and BALL_LAND_Y in out.columns:
        lx = out[BALL_LAND_X].astype(float)
        ly = out[BALL_LAND_Y].astype(float)
        out[BALL_LAND_Y_NORM] = ly
        if offense_left_to_right and has_dir:
            out[BALL_LAND_X_NORM] = np.where(direction.eq("left"), 120.0 - lx, lx)
        else:
            out[BALL_LAND_X_NORM] = lx

    return out

    direction = out[play_direction_col].astype(str).str.lower()
    x = out[x_col].astype(float)

    # Typical tracking coordinates: x from 0..120 (end zone to end zone). Flip for 'left'.
    out[x_out] = np.where(direction.eq("left"), 120.0 - x, x)

    # Normalize ball landing point to match x_norm/y_norm (if present)
    if BALL_LAND_X in out.columns and BALL_LAND_Y in out.columns:
        lx = out[BALL_LAND_X].astype(float)
        ly = out[BALL_LAND_Y].astype(float)
        out[BALL_LAND_Y_NORM] = ly
        out[BALL_LAND_X_NORM] = np.where(direction.eq("left"), 120.0 - lx, lx)
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
    # Prefer normalized landing columns if available
    lx_col = BALL_LAND_X_NORM if BALL_LAND_X_NORM in out.columns else land_x_col
    ly_col = BALL_LAND_Y_NORM if BALL_LAND_Y_NORM in out.columns else land_y_col

    if lx_col in out.columns and ly_col in out.columns:
        x = out[x_col].astype(float).to_numpy()
        y = out[y_col].astype(float).to_numpy()
        lx = out[lx_col].astype(float).to_numpy()
        ly = out[ly_col].astype(float).to_numpy()

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


def compute_initial_separation_at_throw(
    df_in_full: pd.DataFrame,
    *,
    play_keys: Tuple[str, str] = (GAME_ID, PLAY_ID),
    player_key: str = NFL_ID,
    frame_col: str = FRAME_ID,
    x_col: str = X_NORM,
    y_col: str = Y_NORM,
    side_col: str = "player_side",
    pos_col: str = "player_position",
    target_flag_col: str = "player_to_predict",
    offense_value: str = "offense",
    defense_value: str = "defense",
    wr_value: str = "WR",
    out_col: str = "initial_separation",
) -> pd.DataFrame:
    """Compute initial separation at throw for the targeted WR.

    Definition
    ----------
    - Throw moment: the last input frame for each (game_id, play_id) in df_input.
    - Targeted WR row: player_to_predict == True AND offense AND WR.
    - initial_separation: minimum Euclidean distance to any defender at the throw frame.

    Returns
    -------
    DataFrame keyed by (game_id, play_id, nfl_id) with one column: initial_separation.
    """
    req = [*play_keys, player_key, frame_col, x_col, y_col, side_col, pos_col, target_flag_col]
    missing = [c for c in req if c not in df_in_full.columns]
    if missing:
        raise KeyError(f"compute_initial_separation_at_throw missing columns: {missing}")

    df = df_in_full.copy()

    # Throw frame per play (last frame in input)
    throw_frames = (
        df.groupby(list(play_keys), as_index=False)[frame_col]
        .max()
        .rename(columns={frame_col: "throw_frame"})
    )

    df_throw = df.merge(throw_frames, on=list(play_keys), how="inner")
    df_throw = df_throw[df_throw[frame_col] == df_throw["throw_frame"]].copy()

    # Targeted WR at throw
    target_mask = (
        (df_throw[target_flag_col] == True)
        & (df_throw[side_col].astype(str).str.lower().eq(offense_value))
        & (df_throw[pos_col].astype(str).str.upper().eq(wr_value.upper()))
    )
    targets = df_throw.loc[target_mask, [*play_keys, player_key, x_col, y_col]].copy()

    # Defenders at throw
    def_mask = df_throw[side_col].astype(str).str.lower().eq(defense_value)
    defenders = df_throw.loc[def_mask, [*play_keys, x_col, y_col]].copy()

    if len(targets) == 0:
        return targets.assign(**{out_col: pd.Series(dtype=float)})

    # Join targets to all defenders in same play
    td = targets.merge(defenders, on=list(play_keys), how="left", suffixes=("", "_def"))
    td["_def_dist"] = np.sqrt(
        (td[x_col].astype(float) - td[f"{x_col}_def"].astype(float)) ** 2
        + (td[y_col].astype(float) - td[f"{y_col}_def"].astype(float)) ** 2
    )

    sep = (
        td.groupby([*play_keys, player_key], as_index=False)["_def_dist"]
        .min()
        .rename(columns={"_def_dist": out_col})
    )
    return sep


def add_converge_rate_from_labels(
    df_labeled: pd.DataFrame,
    *,
    group_cols: Tuple[str, str, str] = (GAME_ID, PLAY_ID, NFL_ID),
    frame_col: str = FRAME_ID,
    true_x_col: str = Y_TRUE_X_NORM,
    true_y_col: str = Y_TRUE_Y_NORM,
    land_x_col: str = BALL_LAND_X_NORM,
    land_y_col: str = BALL_LAND_Y_NORM,
    dt_seconds: float = 0.1,
    out_col: str = CONVERGE_RATE,
) -> pd.DataFrame:
    """Compute converge_rate from ground-truth (output) labels.

    converge_rate[t] = (dist[t-1] - dist[t]) / dt_seconds
    where dist[t] is distance from true position to the landing point at timestep t.

    Positive values mean the player is closing toward the landing point.
    """
    req = [*group_cols, frame_col, true_x_col, true_y_col, land_x_col, land_y_col]
    missing = [c for c in req if c not in df_labeled.columns]
    if missing:
        raise KeyError(f"add_converge_rate_from_labels missing columns: {missing}")

    out = df_labeled.copy()
    out = out.sort_values(list(group_cols) + [frame_col])

    dx = out[true_x_col].astype(float) - out[land_x_col].astype(float)
    dy = out[true_y_col].astype(float) - out[land_y_col].astype(float)
    out["_dist_true_to_land"] = np.sqrt(dx ** 2 + dy ** 2)

    prev = out.groupby(list(group_cols), sort=False)["_dist_true_to_land"].shift(1)
    out[out_col] = ((prev - out["_dist_true_to_land"]) / dt_seconds).fillna(0.0)

    return out.drop(columns=["_dist_true_to_land"], errors="ignore")

def compute_catch_separation(
    df_tracking: pd.DataFrame,
    *,
    id_cols: Tuple[str, str] = (GAME_ID, PLAY_ID),
    frame_col: str = FRAME_ID,
    nfl_id_col: str = NFL_ID,
    x_col: str = X_NORM,
    y_col: str = Y_NORM,
    side_col: str = "player_side",
) -> pd.DataFrame:
    """
    Calculates distance to nearest defender at catch point for ALL offensive players.
    
    The catch point is defined as the last frame of each play. This measures
    separation between receivers and nearest defenders to distinguish contested
    catches from scheme wins.
    
    Parameters
    ----------
    df_tracking:
        Full tracking data with all players (offense and defense).
    id_cols:
        Play identifier columns (game_id, play_id).
    frame_col:
        Frame ID column.
    nfl_id_col:
        Player ID column.
    x_col, y_col:
        Normalized coordinate columns.
    side_col:
        Column indicating 'Offense' vs 'Defense' (case-sensitive).
    
    Returns
    -------
    DataFrame with [game_id, play_id, nfl_id, sep_at_catch] for all offensive players.
    Wide open plays (no defenders) are capped at 10.0 yards.
    """
    # 1. Identify 'Catch Frame' (Last frame of the play)
    catch_frames = df_tracking.groupby(list(id_cols))[frame_col].max().reset_index()
    
    # 2. Filter tracking to just the catch frames
    df_catch = df_tracking.merge(catch_frames, on=list(id_cols) + [frame_col])

    # 3. Split Offense vs Defense
    df_offense = df_catch[df_catch[side_col] == 'Offense'].copy()
    df_defense = df_catch[df_catch[side_col] == 'Defense'].copy()

    # 4. Cartesian Product (Every Offense vs Every Defense on that play)
    pairs = df_offense.merge(
        df_defense,
        on=list(id_cols),
        suffixes=('_off', '_def')
    )

    # 5. Euclidean Distance
    pairs['dist'] = np.sqrt(
        (pairs[f"{x_col}_off"] - pairs[f"{x_col}_def"])**2 + 
        (pairs[f"{y_col}_off"] - pairs[f"{y_col}_def"])**2
    )

    # 6. Min Distance per Player (Group by game, play, AND OFFENSIVE PLAYER)
    sep_df = pairs.groupby(list(id_cols) + [f"{nfl_id_col}_off"])['dist'].min().reset_index()
    
    # 7. Rename for clean merge
    sep_df.rename(columns={
        f"{nfl_id_col}_off": nfl_id_col, 
        'dist': 'sep_at_catch'
    }, inplace=True)

    # Handle wide open / untracked (fill with cap)
    sep_df['sep_at_catch'] = sep_df['sep_at_catch'].fillna(10.0)

    return sep_df


def attach_output_labels(
    df_post: pd.DataFrame,
    df_out: pd.DataFrame,
    *,
    keys: Tuple[str, str, str, str] = ("game_id", "play_id", "nfl_id", "frame_id"),
    out_x: str = "x",
    out_y: str = "y",
    label_x: str = "y_true_x",
    label_y: str = "y_true_y",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Join df_output (ground-truth x,y) onto the post-throw feature table.

    Result columns:
      - y_true_x, y_true_y as label positions
    """
    for k in keys:
        if k not in df_post.columns:
            raise KeyError(f"df_post missing join key: {k}")
        if k not in df_out.columns:
            raise KeyError(f"df_out missing join key: {k}")
    for c in (out_x, out_y):
        if c not in df_out.columns:
            raise KeyError(f"df_out missing required column: {c}")

    out = df_out.loc[:, [*keys, out_x, out_y]].copy()
    out = out.rename(columns={out_x: label_x, out_y: label_y})

    merged = df_post.merge(out, on=list(keys), how=how)

    # Fail loudly if join dropped lots of rows (common silent bug)
    if how == "inner" and len(merged) == 0:
        raise ValueError("attach_output_labels() produced 0 rows. Check join keys/dtypes.")

    # Normalize label coordinates to match x_norm/y_norm so downstream targets are consistent.
    # We use play_direction from df_post (already present in merged) to flip x for leftward plays.
    if "play_direction" in merged.columns:
        direction = merged["play_direction"].astype(str).str.lower()
        yx = merged[label_x].astype(float)
        yy = merged[label_y].astype(float)
        merged[Y_TRUE_Y_NORM] = yy
        merged[Y_TRUE_X_NORM] = np.where(direction.eq("left"), 120.0 - yx, yx)
    else:
        merged[Y_TRUE_X_NORM] = merged[label_x].astype(float)
        merged[Y_TRUE_Y_NORM] = merged[label_y].astype(float)

    return merged