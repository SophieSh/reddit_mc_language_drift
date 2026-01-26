#!/usr/bin/env python3
"""Step 6: Aggregate posts by day and apply per-user z-score normalization.

Aggregates multiple posts per day into one data point by averaging features,
then applies per-user z-score normalization to make features comparable.

Input:
- Timeline with offsets from step 5: data/interim/timeline_with_offsets_*.csv

Output:
- data/interim/timeline_daily_aggregated_{timestamp}.csv
  (columns: author, offset_from_cd1, {feature}_mean, {feature}_zscore for each feature)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.analysis import (
    aggregate_all_features_by_day,
    normalize_features_per_user_zscore,
)
from src.utils import identify_feature_columns


def main(
    config_path: str = "configs/base.yaml",
    use_checkpoint: bool = True,
    force_recompute: bool = False,
):
    """Aggregate posts by day and apply per-user z-score normalization."""
    cfg = load_config(config_path)
    
    interim_dir = Path(cfg["paths"]["interim"])
    normalize = cfg.get("pipeline", {}).get("normalize_features", True)
    
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 6: Aggregate by Day & Normalize")
    print("=" * 60)
    print(f"  Normalization: {'Enabled' if normalize else 'Disabled'}")
    print()
    
    # Check for checkpoint
    checkpoint = find_latest_file(interim_dir, "timeline_daily_aggregated_*.csv")
    
    if use_checkpoint and not force_recompute and checkpoint:
        print(f"✓ Found checkpoint: {checkpoint.name}")
        print(f"  To recompute, use --force-recompute")
        return 0
    
    # Step 1: Load timeline with offsets
    print("[Step 6.1] Loading timeline with offsets...")
    timeline_file = find_latest_file(interim_dir, "timeline_with_offsets_*.csv")
    
    if not timeline_file:
        raise FileNotFoundError(
            f"No timeline file found in {interim_dir}. "
            "Please run scripts/05_build_timeline.py first."
        )
    
    timeline_df = pd.read_csv(timeline_file, encoding='utf-8-sig', low_memory=False)
    print(f"  ✓ Loaded {len(timeline_df):,} posts from {timeline_file.name}")
    print(f"  Users: {timeline_df['author'].nunique():,}")
    
    # Step 2: Identify feature columns
    print("\n[Step 6.2] Identifying feature columns...")
    feature_cols = identify_feature_columns(timeline_df, cfg)
    print(f"  ✓ Found {len(feature_cols)} feature columns")
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in timeline")
    
    # Step 3: Aggregate by day
    print("\n[Step 6.3] Aggregating posts by day...")
    daily_agg = aggregate_all_features_by_day(
        timeline_df=timeline_df,
        feature_cols=feature_cols,
        user_col='author',
        time_col='offset_from_cd1',
    )
    print(f"  ✓ Aggregated to {len(daily_agg):,} (user, day) combinations")
    
    # Step 4: Apply per-user z-score normalization
    if normalize:
        print("\n[Step 6.4] Applying per-user z-score normalization...")
        feature_mean_cols = [f"{feature}_mean" for feature in feature_cols if f"{feature}_mean" in daily_agg.columns]
        
        daily_agg = normalize_features_per_user_zscore(
            daily_agg_df=daily_agg,
            feature_mean_cols=feature_mean_cols,
            user_col='author',
        )
        print(f"  ✓ Normalization complete")
    else:
        print("\n[Step 6.4] Skipping normalization (disabled in config)")
    
    print(f"\n✓ Final: {len(daily_agg):,} (user, day) combinations")
    print(f"  Users: {daily_agg['author'].nunique():,}")
    print(f"  Days per user: {len(daily_agg) / daily_agg['author'].nunique():.1f} (average)")
    
    # Step 5: Save checkpoint
    print("\n[Step 6.5] Saving checkpoint...")
    output_file = save_with_timestamp(
        daily_agg,
        interim_dir,
        "timeline_daily_aggregated"
    )
    print(f"  ✓ Saved: {output_file.name}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6: Aggregate posts by day and apply per-user z-score normalization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint loading"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute even if checkpoint exists"
    )
    
    args = parser.parse_args()
    
    exit(main(
        config_path=args.config,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
    ))

