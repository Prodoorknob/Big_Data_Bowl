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

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        animation_frame=frame_col,
        animation_group=id_col,
        size=size_col if size_col in df.columns else None,
        size_max=15,
        color=color_col if color_col in df.columns else None,
        hover_data=[c for c in hover_cols if c in df.columns],
        title=title,
        range_x=[0, 120],
        range_y=[0, 53.3],
    )
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
    truespeed_score: Optional[float] = None,
    x_col: str = "x_norm",
    y_col: str = "y_norm",
    frame_col: str = "frame_id",
    title_prefix: str = "",
    renderer: Optional[str] = None,
):
    """Convenience wrapper: concatenate pre-throw and post-throw tracking for a play and animate."""
    play_in = df_input[(df_input["game_id"] == game_id) & (df_input["play_id"] == play_id)].copy()
    play_out = df_output[(df_output["game_id"] == game_id) & (df_output["play_id"] == play_id)].copy()

    if len(play_in) == 0:
        raise ValueError(f"No input tracking rows for game_id={game_id}, play_id={play_id}")

    full_play = pd.concat([play_in, play_out], ignore_index=True).sort_values(frame_col)

    title = title_prefix or f"{player_name} | game {game_id} play {play_id}"
    if truespeed_score is not None:
        title += f" | TrueSpeed: {truespeed_score:.3f}"

    return animate_route_on_field(full_play, title=title, x_col=x_col, y_col=y_col, frame_col=frame_col, renderer=renderer)


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
