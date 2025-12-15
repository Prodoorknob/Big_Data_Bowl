# bdb2026/data_io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class BDBPaths:
    train_dir: Path
    supplementary_csv: Path
    group_input_csv: Path
    group_output_csv: Path


def _find_weekly_files(train_dir: Path, prefix: str) -> List[Path]:
    """
    Find weekly files like:
      input_2023_w01.csv ... input_2023_w18.csv
      output_2023_w01.csv ... output_2023_w18.csv
    """
    files = sorted(train_dir.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found matching {prefix}_*.csv in {train_dir}")
    return files


def _concat_csvs(files: List[Path]) -> pd.DataFrame:
    """
    Concatenate CSVs in order. Keeps columns aligned.
    """
    dfs = []
    for fp in files:
        dfs.append(pd.read_csv(fp))
    return pd.concat(dfs, ignore_index=True, sort=False)


def prepare_group_input_output(
    train_dir: str | Path,
    *,
    group_input_name: str = "group_input.csv",
    group_output_name: str = "group_output.csv",
    input_prefix: str = "input",
    output_prefix: str = "output",
) -> Tuple[Path, Path]:
    """
    Ensure group_input.csv and group_output.csv exist inside train_dir by combining all weekly
    input_*.csv and output_*.csv files. If the group files already exist, do nothing.

    Returns:
      (group_input_path, group_output_path)
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir does not exist: {train_dir}")

    group_input_path = train_dir / group_input_name
    group_output_path = train_dir / group_output_name

    # If both already exist, skip combine
    if group_input_path.exists() and group_output_path.exists():
        return group_input_path, group_output_path

    # Combine input weekly files
    if not group_input_path.exists():
        input_files = _find_weekly_files(train_dir, input_prefix)
        df_in = _concat_csvs(input_files)
        df_in.to_csv(group_input_path, index=False)

    # Combine output weekly files
    if not group_output_path.exists():
        output_files = _find_weekly_files(train_dir, output_prefix)
        df_out = _concat_csvs(output_files)
        df_out.to_csv(group_output_path, index=False)

    return group_input_path, group_output_path


def load_bdb_csvs_from_kaggle_download(
    kaggle_root_dir: str | Path,
    supplementary_csv: str | Path,
    *,
    train_subdir: str = "train",
    force_recombine: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end loader for a typical Kaggle download layout:
      <kaggle_root_dir>/
        train/
          input_2023_w01.csv ... input_2023_w18.csv
          output_2023_w01.csv ... output_2023_w18.csv
          (optional cached) group_input.csv, group_output.csv
        supplementary_data.csv  (often at root, but you pass the path)

    Behavior:
      - If group_input.csv and group_output.csv exist, loads them
      - Else combines weekly files into group_* and then loads them
      - Supplementary is loaded directly (not combined)
      - If force_recombine=True, deletes and rebuilds group_* files
    """
    kaggle_root_dir = Path(kaggle_root_dir)
    train_dir = kaggle_root_dir / train_subdir

    supplementary_csv = Path(supplementary_csv)
    if not supplementary_csv.exists():
        raise FileNotFoundError(f"supplementary_csv not found: {supplementary_csv}")

    group_input = train_dir / "group_input.csv"
    group_output = train_dir / "group_output.csv"

    if force_recombine:
        if group_input.exists():
            group_input.unlink()
        if group_output.exists():
            group_output.unlink()

    group_input_path, group_output_path = prepare_group_input_output(train_dir)

    df_in = pd.read_csv(group_input_path)
    df_out = pd.read_csv(group_output_path)
    df_supp = pd.read_csv(supplementary_csv)

    return df_in, df_out, df_supp


# Backwards-compatible helper if you already use this in notebooks
def load_bdb_csvs(
    group_input_csv: str | Path,
    group_output_csv: str | Path,
    supplementary_csv: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Direct loader for already-combined group_input/group_output.
    """
    group_input_csv = Path(group_input_csv)
    group_output_csv = Path(group_output_csv)
    supplementary_csv = Path(supplementary_csv)

    if not group_input_csv.exists():
        raise FileNotFoundError(f"group_input_csv not found: {group_input_csv}")
    if not group_output_csv.exists():
        raise FileNotFoundError(f"group_output_csv not found: {group_output_csv}")
    if not supplementary_csv.exists():
        raise FileNotFoundError(f"supplementary_csv not found: {supplementary_csv}")

    return (
        pd.read_csv(group_input_csv),
        pd.read_csv(group_output_csv),
        pd.read_csv(supplementary_csv),
    )
