#!/usr/bin/env python3
"""Standalone script: Feature Values by Menstrual Phase bar chart.

Loads:
  - data/interim/timeline_daily_aggregated_with_anchors_*.csv  (step 06)
  - data/interim/periodicity_results_*.csv                     (step 07)

Outputs:
  - reports/feature_values_by_phase_<timestamp>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.visualization import aggregate_features_by_phase, plot_phase_analysis


def main(config_path: str = "configs/base.yaml") -> None:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw timeline (aggregate_features_by_phase does its own daily agg) ---
    timeline_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    print(f"Loading timeline: {timeline_path.name}")
    timeline_df = pd.read_csv(timeline_path)
    print(f"  {len(timeline_df):,} posts, {timeline_df['author'].nunique():,} users")

    # --- Identify feature columns (exclude meta cols and zscore cols) ---
    meta_cols = {
        "author", "author_flair_text", "subreddit", "created_utc", "offset_from_cd1",
        "title", "selftext", "id", "url", "score", "num_comments", "permalink",
        "source", "text", "matched_phrase", "matched_phrase_norm",
        "pattern", "has_uncertainty", "anchor_date", "cd1_date",
        "post_date", "post_type",
    }
    features = sorted([
        c for c in timeline_df.columns
        if c not in meta_cols
        and not c.endswith("_zscore")
        and timeline_df[c].dtype in ["float64", "int64", "float32", "int32"]
    ])
    print(f"  {len(features)} feature columns found")

    # --- Load consensus periods (min 23 features agreeing, ~1042 users) ---
    consensus_path = find_latest_file(interim_dir, "consensus_periods_min23features_*.csv")
    print(f"Loading consensus periods: {consensus_path.name}")
    consensus_df = pd.read_csv(consensus_path)
    print(f"  {len(consensus_df):,} users with consensus period (≥23 features agreeing)")

    # aggregate_features_by_phase expects columns: user, period, method
    results_df = consensus_df[["user", "consensus_period"]].rename(
        columns={"consensus_period": "period"}
    )
    results_df["method"] = "fft_interpolation"

    # --- Aggregate features by phase ---
    print("\nAggregating features by phase...")
    phase_df = aggregate_features_by_phase(
        timeline_df=timeline_df,
        results_df=results_df,
        features=features,
        time_col="offset_from_cd1",
        user_col="author",
        method="fft_interpolation",
        normalize=True,
        average_per_user=True,
    )

    if len(phase_df) == 0:
        print("No phase data produced — check periodicity results.")
        return

    # Strip _mean suffix from feature names for display
    feature_order = sorted(phase_df["feature"].unique())

    # --- Plot ---
    output_path = reports_dir / f"feature_values_by_phase_{pd.Timestamp.now().strftime('%Y%m%dT%H%M%S')}.png"
    print(f"\nPlotting {len(feature_order)} features...")
    plot_phase_analysis(
        phase_df=phase_df,
        features=feature_order,
        output_path=output_path,
        error_bar_type="sem",
        average_per_user=True,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
