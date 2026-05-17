"""Shared data-loading utilities for ML scripts in the menstrual-cycle NLP pipeline.

This module centralises the logic that was duplicated between script 12 (XGBoost)
and script 27 (EBM): loading the step-08b phase-labeled timeline, computing
per-user z-scores, filtering to the canonical phase set, and aggregating to
one feature profile per (user, phase).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.analysis import aggregate_to_phase_profiles, normalize_features_per_user_zscore
from src.constants import PHASE_ORDER
from src.io import find_latest_file


def load_phase_labeled_dataset(
    cfg: dict,
    no_anchors: bool = False,
    phase_file: str | None = None,
) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """Load the step-08b phase-labeled timeline and return ML-ready data structures.

    Steps performed internally:
      1. Locate the latest timeline_phase_labeled[_no_anchors]_*.csv in interim_dir.
      2. Compute per-user z-scores from the _mean columns via
         normalize_features_per_user_zscore().
      3. Filter rows to the canonical PHASE_ORDER phases.
      4. Aggregate day-level rows to one mean profile per (user, phase) via
         aggregate_to_phase_profiles().
      5. Fit a LabelEncoder on PHASE_ORDER so integer labels are consistent with
         the rest of the pipeline.

    The per-user z-score step must happen before aggregation so that each user's
    personal mean and std are computed over their complete set of labeled days,
    not just the days that survive per-phase grouping.

    Args:
        cfg:        Loaded config dict (from src.config.load_config).
        no_anchors: If True, load the _no_anchors variant of the phase-labeled file.
                    Requires running scripts/08b_label_phases.py --no-anchors first.

    Returns:
        df_profiles:   DataFrame with one row per (author, phase).  Columns are
                       "author", "phase", and all *_zscore feature columns.
        zscore_cols:   List of z-score column names (length == n_features).
        label_encoder: sklearn LabelEncoder fitted on PHASE_ORDER; use
                       label_encoder.transform(df_profiles["phase"]) to get y.

    Raises:
        FileNotFoundError: If no matching phase-labeled file is found.
        ValueError:        If the loaded file is missing the 'phase' column.
    """
    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg   = cfg["paths"]["files"]

    if phase_file is not None:
        path = Path(phase_file)
        if not path.exists():
            path = interim_dir / phase_file
        if not path.exists():
            raise FileNotFoundError(f"Phase-labeled file not found: {phase_file}")
    else:
        anchor_suffix = "_no_anchors" if no_anchors else ""
        pattern = files_cfg["phase_labeled"] + anchor_suffix + "_*.csv"
        path = find_latest_file(
            interim_dir,
            pattern,
            exclude=None if no_anchors else "_no_anchors",
        )
        if path is None:
            raise FileNotFoundError(
                f"No {pattern} found in {interim_dir}. "
                f"Run scripts/08b_label_phases.py"
                f"{'  --no-anchors' if no_anchors else ''} first."
            )

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"Loaded: {path.name}  ({len(df):,} rows, {df['author'].nunique():,} users)")

    if "phase" not in df.columns:
        raise ValueError("Missing 'phase' column. Re-run scripts/08b_label_phases.py.")

    # Compute per-user z-scores from the _mean columns produced by step 06.
    mean_cols = [
        c for c in df.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not mean_cols:
        raise RuntimeError(
            "No _mean columns found in the phase-labeled file. "
            "Check that the input is a step-06 daily-aggregated file."
        )
    df = normalize_features_per_user_zscore(df, mean_cols, user_col="author")
    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    logging.info(f"  {len(zscore_cols)} z-score features computed")

    # Restrict to the canonical four phases.
    df = df[df["phase"].isin(PHASE_ORDER)].copy()
    logging.info(f"  Phase counts:\n{df['phase'].value_counts().to_string()}")

    # Aggregate day-level rows to one mean profile per (user, phase).
    df_profiles = aggregate_to_phase_profiles(df, zscore_cols)
    logging.info(
        f"  Phase profiles: {len(df_profiles):,} rows "
        f"({df_profiles['author'].nunique():,} users x up to 4 phases)"
    )

    # Fit a label encoder consistent with the rest of the pipeline.
    label_encoder = LabelEncoder()
    label_encoder.fit(PHASE_ORDER)

    return df_profiles, zscore_cols, label_encoder
