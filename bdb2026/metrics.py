from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union, Iterable, Set, Dict, List

import numpy as np
import pandas as pd

from .config import GAME_ID, PLAY_ID, NFL_ID, PLAYER_NAME

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

    g = residual.groupby([work[c] for c in id_cols], sort=False)

    if agg == "mean":
        score = g.mean()
    elif agg == "median":
        score = g.median()
    elif agg == "sum":
        score = g.sum()
    elif agg == "mae":
        score = g.apply(lambda s: s.abs().mean())
    elif agg == "rmse":
        score = g.apply(lambda s: np.sqrt(np.mean(np.square(s))))
    else:  # pragma: no cover
        raise ValueError(f"Unknown agg={agg}")

    out = score.reset_index()
    out.columns = list(id_cols) + [out_col]
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

    g = dist.groupby([work[c] for c in id_cols], sort=False)

    if agg == "mean":
        score = g.mean()
    elif agg == "median":
        score = g.median()
    elif agg == "sum":
        score = g.sum()
    elif agg == "mae":
        score = g.mean()  # same as mean for distances
    elif agg == "rmse":
        score = g.apply(lambda s: np.sqrt(np.mean(np.square(s))))
    else:  # pragma: no cover
        raise ValueError(f"Unknown agg={agg}")

    out = score.reset_index()
    out.columns = list(id_cols) + [out_col]
    return out


def standardize_within_cluster(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    metric_cols: Sequence[str],
    z_clip: Optional[float] = 3.0,
    suffix: str = "_z",
) -> pd.DataFrame:
    """Z-score standardization per cluster for radar charts / comparisons."""
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


# -----------------------------
# Scorecard utilities (Notebook-compatible)
# -----------------------------

def normalize_to_100(series: pd.Series, *, decimals: int = 1) -> pd.Series:
    """Min-max normalize a metric into [0, 100]. Matches the Kaggle notebook implementation."""
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(50.0, index=s.index).round(decimals)
    out = 100.0 * (s - mn) / (mx - mn)
    return out.round(decimals)

def compute_route_execution_score(
    route_features: pd.DataFrame,
    *,
    assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "route_cluster",
    id_cols: Sequence[str] = (GAME_ID, PLAY_ID),
    out_col: str = "RouteExecution",
) -> pd.DataFrame:
    """Compute per-play route deviation from cluster centroid and convert to a 0-100 score.

    This is a pragmatic proxy for "route execution / route deviation" from notebook 4.1/4.2:
    deviation = || route_feature_vector - centroid(route_cluster) ||_2

    Lower deviation => better execution => higher score.
    """
    feats = route_features.copy()
    asn = assignments.copy()

    # merge cluster label onto route_features
    merge_keys = [c for c in [GAME_ID, PLAY_ID, NFL_ID] if c in feats.columns and c in asn.columns]
    if not merge_keys:
        raise KeyError("compute_route_execution_score: no shared keys between route_features and assignments.")
    feats = feats.merge(asn[merge_keys + [label_col]], on=merge_keys, how="inner")

    # ensure feature cols exist and numeric
    cols = [c for c in feature_cols if c in feats.columns and c in centroids.columns]
    if not cols:
        raise ValueError("compute_route_execution_score: no usable feature_cols present in both feats and centroids.")

    X = feats.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # median impute to avoid KMeans-style NaN issues
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    cent = centroids.set_index(label_col).loc[feats[label_col], cols].reset_index(drop=True)
    cent = cent.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cent = cent.fillna(cent.median(numeric_only=True)).fillna(0.0)

    dev = np.sqrt(np.sum((X.to_numpy(dtype=float) - cent.to_numpy(dtype=float)) ** 2, axis=1))
    tmp = feats.loc[:, [c for c in id_cols]].copy()
    tmp["_dev"] = dev
    dev_play = tmp.groupby(list(id_cols), sort=False)["_dev"].mean().reset_index()

    # Invert: lower deviation = better execution = higher score
    # Negate the deviation before normalizing
    dev_play[out_col] = normalize_to_100(-dev_play["_dev"],  decimals=1)
    dev_play = dev_play.drop(columns=["_dev"])
    return dev_play


def _minmax_100(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    if pd.isna(rng) or rng == 0:
        return pd.Series(50.0, index=s.index)  # degenerate case
    return 100.0 * (s - s.min()) / rng


def build_truespeed_scorecard(
    *,
    truespeed_play: pd.DataFrame,
    route_exec_play: pd.DataFrame,
    df_input: pd.DataFrame,
    df_supp: pd.DataFrame,
    # keying / columns
    id_cols: Tuple[str, str] = ("game_id", "play_id"),
    player_name_col: str = "player_name",
    player_to_predict_col: str = "player_to_predict",
    player_side_col: str = "player_side",
    player_pos_col: str = "player_position",
    # supp columns
    route_label_col: str = "route_of_targeted_receiver",
    pass_result_col: str = "pass_result",
    yards_col: str = "yards_gained",
    epa_col: str = "expected_points_added",
    # logic
    junk_routes: Iterable[str] = ("SCREEN", "FLAT", "ANGLE", "WHEEL", "SHIELD"),
    volume_threshold_ratio: float = 0.50,
    explosive_yards: float = 20.0,
    # column names in play-level tables
    truespeed_col: str = "TrueSpeed",
    route_exec_col: str = "RouteExecution",
    normalize_scores_to_100: bool = True,
) -> pd.DataFrame:
    """
    Player scorecard builder for your modular pipeline.

    - Uses df_input to map (game_id, play_id) to the targeted WR player_name.
    - Uses df_supp for route label + production context.
    - Uses truespeed_play and route_exec_play for process metrics.
    - Applies junk route removal and a volume threshold (> max*ratio).
    """

    # --- validate ---
    for k in id_cols:
        if k not in truespeed_play.columns or k not in route_exec_play.columns:
            raise KeyError(f"Missing id col {k} in truespeed_play/route_exec_play")
        if k not in df_input.columns or k not in df_supp.columns:
            raise KeyError(f"Missing id col {k} in df_input/df_supp")

    if truespeed_col not in truespeed_play.columns:
        raise KeyError(f"truespeed_play missing column '{truespeed_col}'")
    if route_exec_col not in route_exec_play.columns:
        raise KeyError(f"route_exec_play missing column '{route_exec_col}'")

    # --- 1) identify the targeted WR per play from df_input ---
    # Definition consistent with your preprocessing:
    # player_to_predict == True AND offense AND WR
    required = [player_to_predict_col, player_side_col, player_pos_col, player_name_col]
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        raise KeyError(f"df_input missing required columns for player mapping: {missing}")

    df_player = df_input.copy()
    df_player = df_player[
        (df_player[player_to_predict_col] == True) &
        (df_player[player_side_col].astype(str).str.lower().eq("offense")) &
        (df_player[player_pos_col].astype(str).str.upper().eq("WR"))
    ].copy()

    # Ensure 1 targeted WR per play: keep last frame row per play if duplicates exist
    # (player_name is constant per nfl_id; this just stabilizes duplicate rows)
    df_player = (
        df_player.sort_values([*id_cols, "frame_id"] if "frame_id" in df_player.columns else list(id_cols))
                .drop_duplicates(subset=list(id_cols), keep="last")
        [[*id_cols, player_name_col]]
    )

    # --- 2) base play-context from df_supp ---
    supp_needed = [route_label_col, pass_result_col, yards_col, epa_col]
    missing = [c for c in supp_needed if c not in df_supp.columns]
    if missing:
        raise KeyError(f"df_supp missing required columns: {missing}")

    df_ctx = df_supp.loc[:, [*id_cols, route_label_col, pass_result_col, yards_col, epa_col]].drop_duplicates(subset=list(id_cols))

    # --- 3) merge everything to play-level ---
    df_play = df_ctx.merge(df_player, on=list(id_cols), how="left")

    df_play = df_play.merge(
        truespeed_play.loc[:, [*id_cols, truespeed_col]],
        on=list(id_cols),
        how="left",
    ).merge(
        route_exec_play.loc[:, [*id_cols, route_exec_col]],
        on=list(id_cols),
        how="left",
    )

    # --- 4) junk route filter (applied before scoring & production, per your notebook logic) ---
    if route_label_col in df_play.columns:
        df_play = df_play[~df_play[route_label_col].isin(list(junk_routes))].copy()

    # --- 5) "scored" subset for process metrics ---
    df_scored = df_play.dropna(subset=[truespeed_col]).copy()

    player_process = (
        df_scored.groupby(player_name_col, dropna=True)
        .agg(
            TrueSpeed=(truespeed_col, "mean"),
            RouteExecIQ=(route_exec_col, "mean"),  # Keep as RouteExecIQ before normalization
            Scored_Count=(id_cols[0], "count"),  # count of scored plays
        )
    )

    # --- 6) production metrics (context) ---
    df_play["is_catch"] = (df_play[pass_result_col] == "C").astype(int)
    df_play["is_explosive"] = (pd.to_numeric(df_play[yards_col], errors="coerce").fillna(0) >= explosive_yards).astype(int)
    df_play["is_successful"] = (pd.to_numeric(df_play[epa_col], errors="coerce").fillna(0) > 0).astype(int)

    player_production = (
        df_play.groupby(player_name_col, dropna=True)
        .agg(
            Total_Targets=(id_cols[0], "count"),
            Total_Yards=(yards_col, "sum"),
            Total_EPA=(epa_col, "sum"),
            Catch_Rate=("is_catch", "mean"),
            Explosive_Plays=("is_explosive", "sum"),
            Success_Rate=("is_successful", "mean"),
        )
    )
    player_production["Yards_Per_Target"] = player_production["Total_Yards"] / player_production["Total_Targets"].replace(0, np.nan)
    player_production["EPA_Per_Target"] = player_production["Total_EPA"] / player_production["Total_Targets"].replace(0, np.nan)

    # --- 7) combine ---
    df_final = player_process.join(player_production, how="inner").copy()

    # --- 8) volume threshold (> max * ratio) ---
    if len(df_final) > 0:
        threshold = df_final["Total_Targets"].max() * float(volume_threshold_ratio)
        df_final = df_final[df_final["Total_Targets"] > threshold].copy()

    # --- 9) normalize scores to 0–100 (optional) ---
    if normalize_scores_to_100 and len(df_final) > 0:
        # Preserve raw residual-based TrueSpeed for analyses like distributions.
        # (When normalization is on, TrueSpeed becomes a 0–100 index.)
        if "TrueSpeed_raw" not in df_final.columns:
            df_final["TrueSpeed_raw"] = df_final["TrueSpeed"]
        if "RouteExecIQ" in df_final.columns and "RouteExecIQ_raw" not in df_final.columns:
            df_final["RouteExecIQ_raw"] = df_final["RouteExecIQ"]

        df_final["TrueSpeed"] = _minmax_100(df_final["TrueSpeed"]).round(1)
        df_final["RouteExecution"] = _minmax_100(df_final["RouteExecIQ"]).round(1)
        # Drop the original RouteExecIQ column after creating RouteExecution
        df_final = df_final.drop(columns=["RouteExecIQ"])

    # optional ranking
    if len(df_final) > 0:
        df_final["Rank"] = df_final["TrueSpeed"].rank(ascending=False, method="min")

    # return with desired order (without forcing Rank if you don’t want it)
    df_final = df_final.reset_index()

    cols = [
        player_name_col, "TrueSpeed", "TrueSpeed_raw", "RouteExecution", "Scored_Count",
        "Total_Targets", "Total_Yards", "Total_EPA", "Catch_Rate",
        "Explosive_Plays", "Success_Rate", "Yards_Per_Target", "EPA_Per_Target",
    ]
    # If normalization wasn't applied, RouteExecIQ will still be present
    if not normalize_scores_to_100 and "RouteExecIQ" in df_final.columns:
        # Replace RouteExecution with RouteExecIQ in the column list
        cols = [c if c != "RouteExecution" else "RouteExecIQ" for c in cols]
    if "Rank" in df_final.columns:
        cols = ["Rank"] + cols

    # keep only columns that exist (defensive)
    cols = [c for c in cols if c in df_final.columns]
    return df_final.loc[:, cols]


def build_truespeed_scouting_scorecard(
    *,
    truespeed_play: pd.DataFrame,
    df_input: pd.DataFrame,
    df_supp: pd.DataFrame,
    # keying / columns
    id_cols: Tuple[str, str] = ("game_id", "play_id"),
    player_name_col: str = "player_name",
    player_to_predict_col: str = "player_to_predict",
    player_side_col: str = "player_side",
    player_pos_col: str = "player_position",
    # supplementary context columns
    route_label_col: str = "route_of_targeted_receiver",
    pass_result_col: str = "pass_result",
    pass_length_col: str = "pass_length",
    pass_location_col: str = "pass_location_type",
    coverage_zone_col: str = "team_coverage_man_zone",
    coverage_type_col: str = "team_coverage_type",
    yards_col: str = "yards_gained",
    epa_col: str = "expected_points_added",
    # logic
    junk_routes: Iterable[str] = ("SCREEN", "FLAT", "ANGLE", "WHEEL", "SHIELD"),
    volume_threshold_ratio: float = 0.50,
    explosive_yards: float = 20.0,
    # play-level metric column
    truespeed_col: str = "TrueSpeed",
    normalize_scores_to_100: bool = True,
) -> pd.DataFrame:
    """Build a WR scouting-oriented player scorecard using *only* TrueSpeed.

    This is a simplified variant of :func:`build_truespeed_scorecard` that:
      - does not require Route Execution inputs
      - keeps the focus on post-throw movement quality ("ball in the air")
      - produces context columns useful for scouting: targets, efficiency, and situation splits

    Notes
    -----
    - Player mapping comes from df_input using your operational definition:
        player_to_predict == True AND offense AND WR.
    - Junk route filtering is applied using df_supp.route_of_targeted_receiver,
      matching the notebook logic, but the output does not "lean" on routes.
    """

    # --- validate keys ---
    for k in id_cols:
        if k not in truespeed_play.columns:
            raise KeyError(f"truespeed_play missing id col {k}")
        if k not in df_input.columns or k not in df_supp.columns:
            raise KeyError(f"Missing id col {k} in df_input/df_supp")

    if truespeed_col not in truespeed_play.columns:
        raise KeyError(f"truespeed_play missing column '{truespeed_col}'")

    # --- 1) identify the targeted WR per play from df_input ---
    required = [player_to_predict_col, player_side_col, player_pos_col, player_name_col]
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        raise KeyError(f"df_input missing required columns for player mapping: {missing}")

    df_player = df_input.copy()
    df_player = df_player[
        (df_player[player_to_predict_col] == True)
        & (df_player[player_side_col].astype(str).str.lower().eq("offense"))
        & (df_player[player_pos_col].astype(str).str.upper().eq("WR"))
    ].copy()

    # One row per play for name mapping
    df_player = (
        df_player.sort_values([*id_cols, "frame_id"] if "frame_id" in df_player.columns else list(id_cols))
        .drop_duplicates(subset=list(id_cols), keep="last")
        .loc[:, [*id_cols, player_name_col]]
    )

    # --- 2) play context from supplementary ---
    ctx_cols = [route_label_col, pass_result_col, yards_col, epa_col]
    optional_cols = [pass_length_col, pass_location_col, coverage_zone_col, coverage_type_col]
    ctx_cols = [c for c in ctx_cols if c in df_supp.columns] + [c for c in optional_cols if c in df_supp.columns]
    if not set([pass_result_col, yards_col, epa_col]).issubset(df_supp.columns):
        raise KeyError(
            f"df_supp missing required columns for production context: "
            f"{[c for c in [pass_result_col, yards_col, epa_col] if c not in df_supp.columns]}"
        )

    df_ctx = df_supp.loc[:, [*id_cols, *ctx_cols]].drop_duplicates(subset=list(id_cols))

    # --- 3) merge play-level ---
    df_play = (
        df_ctx.merge(df_player, on=list(id_cols), how="left")
        .merge(truespeed_play.loc[:, [*id_cols, truespeed_col]], on=list(id_cols), how="left")
    )

    # --- 4) junk route filter ---
    if route_label_col in df_play.columns:
        df_play = df_play[~df_play[route_label_col].isin(list(junk_routes))].copy()

    # --- 5) scored subset (process metric) ---
    df_scored = df_play.dropna(subset=[truespeed_col]).copy()
    player_process = (
        df_scored.groupby(player_name_col, dropna=True)
        .agg(
            TrueSpeed=(truespeed_col, "mean"),
            Scored_Count=(id_cols[0], "count"),
        )
    )

    # --- 6) production metrics (context) ---
    df_play["is_catch"] = (df_play[pass_result_col] == "C").astype(int)
    df_play["is_explosive"] = (
        pd.to_numeric(df_play[yards_col], errors="coerce").fillna(0) >= float(explosive_yards)
    ).astype(int)
    df_play["is_successful"] = (pd.to_numeric(df_play[epa_col], errors="coerce").fillna(0) > 0).astype(int)

    player_production = (
        df_play.groupby(player_name_col, dropna=True)
        .agg(
            Total_Targets=(id_cols[0], "count"),
            Total_Yards=(yards_col, "sum"),
            Total_EPA=(epa_col, "sum"),
            Catch_Rate=("is_catch", "mean"),
            Explosive_Plays=("is_explosive", "sum"),
            Success_Rate=("is_successful", "mean"),
        )
    )
    player_production["Yards_Per_Target"] = (
        player_production["Total_Yards"] / player_production["Total_Targets"].replace(0, np.nan)
    )
    player_production["EPA_Per_Target"] = (
        player_production["Total_EPA"] / player_production["Total_Targets"].replace(0, np.nan)
    )

    # --- 7) combine ---
    df_final = player_process.join(player_production, how="inner").copy()

    # --- 8) volume threshold (> max * ratio) ---
    if len(df_final) > 0:
        threshold = df_final["Total_Targets"].max() * float(volume_threshold_ratio)
        df_final = df_final[df_final["Total_Targets"] > threshold].copy()

    # --- 9) normalize TrueSpeed to 0–100 (optional) ---
    if normalize_scores_to_100 and len(df_final) > 0:
        # Preserve raw residual-based TrueSpeed for analyses like distributions.
        if "TrueSpeed_raw" not in df_final.columns:
            df_final["TrueSpeed_raw"] = df_final["TrueSpeed"]
        df_final["TrueSpeed"] = _minmax_100(df_final["TrueSpeed"]).round(1)

    # optional ranking
    if len(df_final) > 0:
        df_final["Rank"] = df_final["TrueSpeed"].rank(ascending=False, method="min")

    df_final = df_final.reset_index()
    cols = [
        "Rank",
        player_name_col,
        "TrueSpeed",
        "TrueSpeed_raw",
        "Scored_Count",
        "Total_Targets",
        "Total_Yards",
        "Total_EPA",
        "Catch_Rate",
        "Explosive_Plays",
        "Success_Rate",
        "Yards_Per_Target",
        "EPA_Per_Target",
    ]
    cols = [c for c in cols if c in df_final.columns]
    return df_final.loc[:, cols]

