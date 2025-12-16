"""Visualization helpers (matplotlib + plotly).

These are notebook-centric helpers: they return figure objects rather than writing files.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def plot_minimal_bar(ax, series: pd.Series, title: str, *, c_max: str = "#66c2a5", c_min: str = "#fc8d62", c_neu: str = "#e0e0e0") -> None:
    """A minimal bar chart used in the notebook for quick EDA."""
    values = series.values
    colors: List[str] = []
    for v in values:
        if v == np.nanmax(values):
            colors.append(c_max)
        elif v == np.nanmin(values):
            colors.append(c_min)
        else:
            colors.append(c_neu)

    ax.bar(series.index.astype(str), values, color=colors)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)


def add_nfl_field_shapes(fig):
    """Add basic NFL field shapes to a plotly figure in tracking coordinates (x: 0..120, y: 0..53.3)."""
    import plotly.graph_objects as go  # local import

    shapes = []
    # End zones and field
    shapes.append(dict(type="rect", x0=0, y0=0, x1=10, y1=53.3, fillcolor="#385436", layer="below", line_width=0))
    shapes.append(dict(type="rect", x0=110, y0=0, x1=120, y1=53.3, fillcolor="#385436", layer="below", line_width=0))
    shapes.append(dict(type="rect", x0=10, y0=0, x1=110, y1=53.3, fillcolor="#517D4D", layer="below", line_width=0))

    # Yard lines
    for x in range(10, 111, 5):
        lw = 3 if x % 10 == 0 else 1
        shapes.append(dict(type="line", x0=x, y0=0, x1=x, y1=53.3, line=dict(color="white", width=lw), layer="below"))

    fig.update_layout(shapes=shapes)
    return fig


def get_highlight_role(row: pd.Series, *, player_role_col: str = "player_role", player_side_col: str = "player_side") -> str:
    """Heuristic used in animations to highlight QB and target."""
    role = str(row.get(player_role_col, ""))
    if role == "Passer":
        return "QB"
    if role == "Targeted Receiver":
        return "Target"
    return str(row.get(player_side_col, "Other"))


def animate_route_on_field(
    tracking: pd.DataFrame,
    *,
    title: str = "Play tracking",
    x_col: str = "x_norm",
    y_col: str = "y_norm",
    frame_col: str = "frame_id",
    id_col: str = "nfl_id",
    size_col: str = "s",
    color_col: str = "s",
    text_col: Optional[str] = None,
    hover_cols: Sequence[str] = ("s", "a", "dir"),
    height: int = 500,
    width: int = 1000,
    renderer: Optional[str] = None,
):
    """Animate player tracking on a field.

    Parameters
    ----------
    tracking:
        Must include frame_id, nfl_id, x_norm, y_norm. If you want QB/Target emphasis, add a `highlight_role` column.
    renderer:
        Pass e.g. 'iframe' for Kaggle/Colab-like environments.

    Returns
    -------
    plotly Figure
    """
    import plotly.express as px

    df = tracking.copy()
    df = df.sort_values(frame_col)

    # Plotly cannot accept NaNs in marker size. Tracking speed `s` can be missing
    # for some rows (e.g., football or incomplete joins), so coerce + fill.
    if size_col in df.columns:
        df[size_col] = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0)
        df[size_col] = df[size_col].clip(lower=0.0)

    use_text = text_col if (text_col and text_col in df.columns) else None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        animation_frame=frame_col,
        animation_group=id_col,
        size=size_col if size_col in df.columns else None,
        size_max=15,
        color=color_col if color_col in df.columns else None,
        text=use_text,
        hover_data=[c for c in hover_cols if c in df.columns],
        title=title,
        range_x=[0, 120],
        range_y=[0, 53.3],
    )

    if use_text:
        fig.update_traces(mode="markers+text", textposition="top center", textfont_size=10)
    fig = add_nfl_field_shapes(fig)
    fig.update_layout(plot_bgcolor="white", height=height, width=width)

    if renderer:
        fig.show(renderer=renderer)

    return fig


def animate_speed_comparison(
    df_input: pd.DataFrame,
    df_output: pd.DataFrame,
    *,
    game_id: int,
    play_id: int,
    player_name: str = "",
    target_nfl_id: Optional[int] = None,
    truespeed_score: Optional[float] = None,
    x_col: str = "x_norm",
    y_col: str = "y_norm",
    frame_col: str = "frame_id",
    title_prefix: str = "",
    show_only_defense_qb_target: bool = False,
    pred_df: Optional[pd.DataFrame] = None,
    pred_actual_col: str = "actual",
    pred_pred_col: str = "pred",
    annotation_xy: Tuple[float, float] = (6.0, 52.0),
    renderer: Optional[str] = None,
):
    """Convenience wrapper: concatenate pre-throw and post-throw tracking for a play and animate."""
    play_in = df_input[(df_input["game_id"] == game_id) & (df_input["play_id"] == play_id)].copy()
    play_out = df_output[(df_output["game_id"] == game_id) & (df_output["play_id"] == play_id)].copy()

    if len(play_in) == 0:
        raise ValueError(f"No input tracking rows for game_id={game_id}, play_id={play_id}")

    # Concatenate and ensure one row per (frame, player). If there are duplicates,
    # plotly's animation can appear to "bounce" because a player has multiple positions
    # in the same animation frame.
    play_in = play_in.assign(_phase=0)
    play_out = play_out.assign(_phase=1)
    full_play = pd.concat([play_in, play_out], ignore_index=True)
    full_play = full_play.sort_values([frame_col, "_phase"]) if frame_col in full_play.columns else full_play
    if frame_col in full_play.columns and "nfl_id" in full_play.columns:
        full_play = full_play.drop_duplicates(subset=[frame_col, "nfl_id"], keep="last")

    # Create a monotonic animation frame index (0..N-1) even if original frame ids skip.
    if frame_col in full_play.columns:
        _f = pd.to_numeric(full_play[frame_col], errors="coerce")
        # Fallback: preserve existing order if non-numeric
        if _f.isna().all():
            unique_frames = pd.Series(full_play[frame_col].astype(str).unique())
        else:
            unique_frames = pd.Series(_f.dropna().astype(int).unique()).sort_values()
        frame_map = {v: i for i, v in enumerate(unique_frames.tolist())}
        if _f.isna().all():
            full_play["_frame_anim"] = full_play[frame_col].astype(str).map(frame_map).astype(int)
        else:
            full_play["_frame_anim"] = _f.dropna().astype(int).map(frame_map)
            full_play["_frame_anim"] = full_play["_frame_anim"].fillna(method="ffill").fillna(0).astype(int)
    else:
        full_play["_frame_anim"] = 0

    # Build on-field text labels:
    # - Targeted receiver: show player_name (and position if available)
    # - Everyone else: show position (fallback: role/side)
    name_cols = ["displayName", "player_name", "player_display_name", "name"]
    pos_cols = ["position", "pos"]
    role_col = "player_role" if "player_role" in full_play.columns else None
    side_col = "player_side" if "player_side" in full_play.columns else None
    name_col = next((c for c in name_cols if c in full_play.columns), None)
    pos_col = next((c for c in pos_cols if c in full_play.columns), None)

    if target_nfl_id is None and role_col:
        mask = full_play[role_col].astype(str).eq("Targeted Receiver")
        if mask.any() and "nfl_id" in full_play.columns:
            try:
                target_nfl_id = int(full_play.loc[mask, "nfl_id"].dropna().iloc[0])
            except Exception:
                target_nfl_id = None

    if target_nfl_id is None and player_name and name_col and "nfl_id" in full_play.columns:
        mask = full_play[name_col].astype(str).str.contains(str(player_name), case=False, na=False)
        if mask.any():
            try:
                target_nfl_id = int(full_play.loc[mask, "nfl_id"].dropna().iloc[0])
            except Exception:
                target_nfl_id = None

    def _other_label(row: pd.Series) -> str:
        if pos_col and pd.notna(row.get(pos_col)):
            return str(row.get(pos_col))
        if role_col and pd.notna(row.get(role_col)):
            return str(row.get(role_col))
        if side_col and pd.notna(row.get(side_col)):
            return str(row.get(side_col))
        return ""

    if "nfl_id" in full_play.columns:
        labels = full_play.apply(_other_label, axis=1)
        if target_nfl_id is not None:
            target_mask = full_play["nfl_id"].astype("Int64").eq(target_nfl_id)
            if target_mask.any():
                target_pos = None
                if pos_col:
                    vals = full_play.loc[target_mask, pos_col].dropna().astype(str)
                    target_pos = vals.iloc[0] if len(vals) else None

                target_label = str(player_name).strip() or (
                    str(full_play.loc[target_mask, name_col].dropna().astype(str).iloc[0]).strip()
                    if name_col and full_play.loc[target_mask, name_col].notna().any()
                    else "Target"
                )
                if target_pos:
                    target_label = f"{target_label} ({target_pos})"
                labels = labels.mask(target_mask, target_label)
        full_play = full_play.assign(_label=labels)

    # Optional filtering: only show Defense + QB + Targeted Receiver
    if show_only_defense_qb_target:
        keep_mask = pd.Series(False, index=full_play.index)
        if side_col:
            keep_mask |= full_play[side_col].astype(str).eq("Defense")
        if role_col:
            keep_mask |= full_play[role_col].astype(str).eq("Passer")
            keep_mask |= full_play[role_col].astype(str).eq("Targeted Receiver")
        if pos_col:
            keep_mask |= full_play[pos_col].astype(str).eq("QB")
        if target_nfl_id is not None and "nfl_id" in full_play.columns:
            keep_mask |= full_play["nfl_id"].astype("Int64").eq(target_nfl_id)
        full_play = full_play.loc[keep_mask].copy()

        # Make QB label explicit
        if "_label" in full_play.columns and (role_col or pos_col):
            qb_mask = pd.Series(False, index=full_play.index)
            if role_col:
                qb_mask |= full_play[role_col].astype(str).eq("Passer")
            if pos_col:
                qb_mask |= full_play[pos_col].astype(str).eq("QB")
            full_play.loc[qb_mask, "_label"] = "QB"

    # Optional: add a fixed on-field annotation per frame for Actual/Pred/Residual.
    if pred_df is not None and len(pred_df) > 0 and "_frame_anim" in full_play.columns:
        work = pred_df.copy()
        if pred_actual_col not in work.columns or pred_pred_col not in work.columns:
            work = None
        else:
            work[pred_actual_col] = pd.to_numeric(work[pred_actual_col], errors="coerce")
            work[pred_pred_col] = pd.to_numeric(work[pred_pred_col], errors="coerce")

        if work is not None:
            frames = np.sort(full_play["_frame_anim"].unique())
            n_frames = len(frames)
            n_steps = len(work)
            if n_frames > 0 and n_steps > 0:
                # Map frame index -> timestep index
                idxs = np.round(np.linspace(0, n_steps - 1, n_frames)).astype(int)
                actual_vals = work[pred_actual_col].iloc[idxs].to_numpy()
                pred_vals = work[pred_pred_col].iloc[idxs].to_numpy()
                resid_vals = actual_vals - pred_vals

                ann = pd.DataFrame({
                    "_frame_anim": frames,
                    x_col: annotation_xy[0],
                    y_col: annotation_xy[1],
                    "nfl_id": -1,
                    "s": 0.0,
                    "_label": [
                        f"Actual: {a:.2f} | Pred: {p:.2f} | Resid: {r:+.2f}"
                        for a, p, r in zip(actual_vals, pred_vals, resid_vals)
                    ],
                })
                full_play = pd.concat([full_play, ann], ignore_index=True)

    title = title_prefix or f"{player_name} | game {game_id} play {play_id}"
    if truespeed_score is not None:
        title += f" | TrueSpeed: {truespeed_score:.3f}"

    return animate_route_on_field(
        full_play,
        title=title,
        x_col=x_col,
        y_col=y_col,
        frame_col="_frame_anim",
        text_col="_label" if "_label" in full_play.columns else None,
        hover_cols=("_label", "s", "a", "dir"),
        renderer=renderer,
    )


def get_representative_play(
    df_scores: pd.DataFrame,
    *,
    cluster_col: str = "route_cluster",
    score_col: str = "TrueSpeed",
    strategy: str = "median",
) -> pd.DataFrame:
    """Pick one representative play per cluster for visualization.

    strategy:
        - 'median': play with score closest to cluster median
        - 'max': highest score
        - 'min': lowest score
    """
    work = df_scores.copy()
    reps = []
    for cl, g in work.groupby(cluster_col):
        if strategy == "max":
            reps.append(g.sort_values(score_col, ascending=False).head(1))
        elif strategy == "min":
            reps.append(g.sort_values(score_col, ascending=True).head(1))
        else:
            med = g[score_col].median()
            g = g.assign(_dist=(g[score_col] - med).abs())
            reps.append(g.sort_values("_dist").head(1).drop(columns=["_dist"]))
    return pd.concat(reps, ignore_index=True) if reps else work.head(0)


def create_individual_radar_charts(
    df: pd.DataFrame,
    *,
    id_col: str,
    metric_cols: Sequence[str],
    label_col: Optional[str] = None,
    max_cols: int = 3,
):
    """Create small-multiple radar charts using matplotlib.

    Intended for comparing players (or clusters) across standardized metrics.
    """
    import matplotlib.pyplot as plt

    metrics = list(metric_cols)
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    items = df[id_col].tolist()
    n = len(items)
    n_cols = min(max_cols, n) if n else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, item in enumerate(items):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        row = df.iloc[idx]

        values = [float(row[m]) for m in metrics]
        values += values[:1]

        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.1)
        title = str(item)
        if label_col and label_col in df.columns:
            title = f"{title} ({row[label_col]})"
        ax.set_title(title, y=1.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)

    # hide unused axes
    for j in range(n, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig



import matplotlib.pyplot as plt


def plot_market_inefficiency_correlation(
    df: pd.DataFrame,
    *,
    x_col: str = "TrueSpeed",
    y_col: str = "EPA_Per_Target",
    label_col: Optional[str] = "player_name",
    title: str = "Market inefficiency correlation",
    annotate_top_n: int = 10,
) -> "plt.Figure":
    """
    Notebook 4.3-style correlation plot for market inefficiency.

    Default columns match the TrueSpeed.csv-style output:
      x_col: TrueSpeed (0-100)
      y_col: EPA_Per_Target

    Returns a matplotlib Figure.
    """
    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(work[x_col], work[y_col], alpha=0.75)

    # Trendline
    if len(work) >= 2:
        m, b = np.polyfit(work[x_col].to_numpy(), work[y_col].to_numpy(), 1)
        xs = np.linspace(work[x_col].min(), work[x_col].max(), 200)
        ax.plot(xs, m * xs + b, linewidth=2)

        r = np.corrcoef(work[x_col], work[y_col])[0, 1]
        ax.set_title(f"{title} (Pearson r={r:.2f})")
    else:
        ax.set_title(title)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Annotate top N by |residual| from trendline (inefficiency candidates)
    if label_col and label_col in work.columns and len(work) >= 3:
        # residuals from fitted line
        y_hat = m * work[x_col] + b
        resid = (work[y_col] - y_hat).abs()
        top = work.assign(_resid=resid).nlargest(annotate_top_n, "_resid")
        for _, row in top.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    fig.tight_layout()
    return fig


def plot_truespeed_distribution(
    df: pd.DataFrame,
    *,
    score_col: Optional[str] = None,
    title: str = "TrueSpeed distribution",
    bins: int = 25,
) -> "plt.Figure":
    """Histogram to communicate what is 'common' vs 'rare' in TrueSpeed.

    If the input contains a preserved raw column (``TrueSpeed_raw``), this
    defaults to plotting that residual-based value so the distribution is
    interpretable. Otherwise it falls back to ``TrueSpeed``.
    """
    work = df.copy()

    effective_col = score_col
    if effective_col is None:
        effective_col = "TrueSpeed_raw" if "TrueSpeed_raw" in work.columns else "TrueSpeed"
    if effective_col not in work.columns:
        raise KeyError(f"plot_truespeed_distribution: df missing score_col={effective_col!r}")

    work[effective_col] = pd.to_numeric(work[effective_col], errors="coerce")
    work = work.dropna(subset=[effective_col])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(work[effective_col].to_numpy(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(effective_col)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_truespeed_leaderboard(
    df_scorecard: pd.DataFrame,
    *,
    player_col: str = "player_name",
    score_col: str = "TrueSpeed",
    min_targets_col: str = "Total_Targets",
    min_targets: int = 15,
    top_n: int = 20,
    title: str = "Top TrueSpeed receivers (volume-qualified)",
) -> "plt.Figure":
    """Horizontal bar chart for quick scouting consumption."""
    work = df_scorecard.copy()
    if min_targets_col in work.columns:
        work = work[pd.to_numeric(work[min_targets_col], errors="coerce").fillna(0) >= int(min_targets)].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col])
    work = work.sort_values(score_col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, int(0.35 * len(work)))))
    ax.barh(work[player_col].astype(str), work[score_col].to_numpy())
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(score_col)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_truespeed_context_splits(
    df_play: pd.DataFrame,
    *,
    score_col: str = "TrueSpeed",
    split_col: str = "pass_length",
    title: Optional[str] = None,
    min_n: int = 200,
) -> "plt.Figure":
    """Average TrueSpeed by a context split (e.g., pass_length, coverage, location).

    This is intended to answer: *where does this metric matter most?*
    """
    work = df_play.copy()
    if split_col not in work.columns:
        raise KeyError(f"plot_truespeed_context_splits: df_play missing split_col={split_col!r}")

    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col, split_col])
    agg = (
        work.groupby(split_col)
        .agg(mean_true_speed=(score_col, "mean"), n=(score_col, "size"))
        .reset_index()
    )
    agg = agg[agg["n"] >= int(min_n)].sort_values("mean_true_speed", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(agg[split_col].astype(str), agg["mean_true_speed"].to_numpy())
    ax.set_title(title or f"Average {score_col} by {split_col} (n≥{min_n})")
    ax.set_xlabel(split_col)
    ax.set_ylabel(f"Mean {score_col}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def select_exemplar_plays_for_film(
    df_play: pd.DataFrame,
    *,
    id_cols: Tuple[str, str] = ("game_id", "play_id"),
    score_col: str = "TrueSpeed",
    player_col: str = "player_name",
    player_name: Optional[str] = None,
    n_each: int = 3,
) -> pd.DataFrame:
    """Return a compact set of exemplar plays for film review.

    Strategy:
      - Take top-N and bottom-N TrueSpeed plays (tails) for the player
      - Also take mid-percentile 'typical' plays to anchor expectations

    Output is a dataframe of plays, ready to feed into `animate_speed_comparison`.
    """
    work = df_play.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col])

    if player_name is not None and player_col in work.columns:
        work = work[work[player_col].astype(str) == str(player_name)].copy()

    if len(work) == 0:
        return work.head(0)

    work = work.sort_values(score_col)
    low = work.head(int(n_each))
    high = work.tail(int(n_each))

    # "Typical" plays: closest to median
    med = work[score_col].median()
    mid = work.assign(_dist=(work[score_col] - med).abs()).sort_values("_dist").head(int(n_each)).drop(columns=["_dist"])

    cols = [c for c in [*id_cols, player_col, score_col] if c in work.columns]
    out = pd.concat([high, mid, low], ignore_index=True).drop_duplicates(subset=list(id_cols))
    return out.loc[:, cols] if cols else out
