from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import pandas as pd


PathLike = Union[str, Path]


@dataclass(frozen=True)
class BDBData:
    """Container holding the three core tables used in the notebook."""

    df_input: pd.DataFrame
    df_output: pd.DataFrame
    df_supp: pd.DataFrame


def load_bdb_csvs(
    input_path: PathLike,
    output_path: PathLike,
    supplementary_path: PathLike,
    *,
    low_memory: bool = False,
    dtype: Optional[dict] = None,
) -> BDBData:
    """Load the notebook's CSV artifacts.

    Parameters
    ----------
    input_path, output_path, supplementary_path:
        Paths to `group_input.csv`, `group_output.csv`, and `supplementary_data.csv`.
    low_memory:
        Passed through to pandas.read_csv for the (often large) input file.
    dtype:
        Optional dtype mapping passed to read_csv.

    Returns
    -------
    BDBData
        A dataclass holding df_input, df_output, df_supp.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    supplementary_path = Path(supplementary_path)

    df_input = pd.read_csv(input_path, low_memory=low_memory, dtype=dtype)
    df_output = pd.read_csv(output_path, low_memory=low_memory, dtype=dtype)
    df_supp = pd.read_csv(supplementary_path, low_memory=low_memory, dtype=dtype)

    return BDBData(df_input=df_input, df_output=df_output, df_supp=df_supp)
