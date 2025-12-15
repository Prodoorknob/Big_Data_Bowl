"""bdb2026 - utilities for the 2026 Big Data Bowl workflow in this repo.

The modules are designed to be notebook-friendly: functions accept/return pandas
DataFrames / numpy arrays, and avoid hidden global state.
"""

from .data_io import load_bdb_csvs, load_bdb_csvs_from_kaggle_download
from .preprocess import (
    normalize_coordinates, add_derived_features, add_postthrow_features,
    filter_targeted_wr_routes, select_target_receiver_rows, filter_to_completed_catches, attach_output_labels
)
from .routes import engineer_route_features, cluster_routes_kmeans
from .sequences import build_sequences
from .models import build_lstm, train_lstm
from .metrics import compute_truespeed



__all__ = [
    "load_bdb_csvs",
    'load_bdb_csvs_from_kaggle_download',
    "normalize_coordinates",
    "add_derived_features",
    "add_postthrow_features",
    "filter_targeted_wr_routes",
    'select_target_receiver_rows',
    'filter_to_completed_catches',
    'attach_output_labels',
    "engineer_route_features",
    "cluster_routes_kmeans",
    "build_sequences",
    "build_lstm",
    "train_lstm",
    "compute_truespeed",
]

from .preprocess import merge_route_embeddings
from .routes import make_route_embedding_table
