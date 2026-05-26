#!/usr/bin/env python3
"""Step 6: Aggregate posts by day.

Aggregates multiple posts per day into one data point by averaging raw features.
Normalization will be done later in the analysis step (inside analyze_user_timeline).

Input:
- Timeline with offsets from step 5: data/interim/timeline_with_offsets_*.csv

Output:
- data/interim/timeline_daily_aggregated_{timestamp}.csv
  (columns: author, offset_from_cd1, {feature}_mean for each feature)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.analysis import (
    aggregate_all_features_by_day,
)
from src.utils import identify_feature_columns


def main(
    config_path: str = "configs/base.yaml",
    use_checkpoint: bool = True,
    force_recompute: bool = False,
):
    cfg = load_config(config_path)
    
    interim_dir = Path(cfg["paths"]["interim"])
    
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 6: Aggregate by Day")
    print("=" * 60)
    print()
    
    files_cfg = cfg["paths"]["files"]

    # Check for checkpoint (with anchors version)
    checkpoint = find_latest_file(interim_dir, files_cfg["daily_aggregated_with_anchors"] + "_*.csv")
    
    if use_checkpoint and not force_recompute and checkpoint:
        print(f" Found checkpoint: {checkpoint.name}")
        print(f"  To recompute, use --force-recompute")
        return 0
    
    # Step 1: Load timeline with offsets (with anchors)
    print("[Step 6.1] Loading timeline with offsets...")
    timeline_file = find_latest_file(interim_dir, files_cfg["timeline_with_anchors"] + "_*.csv")
    
    if not timeline_file:
        raise FileNotFoundError(
            f"No timeline file found in {interim_dir}. "
            "Please run scripts/05_build_timeline.py first."
        )
    
    timeline_df = pd.read_csv(timeline_file, encoding='utf-8-sig', low_memory=False)
    print(f"   Loaded {len(timeline_df):,} posts from {timeline_file.name}")
    print(f"  Users: {timeline_df['author'].nunique():,}")
    
    # Step 2: Identify feature columns
    print("\n[Step 6.2] Identifying feature columns...")
    feature_cols = identify_feature_columns(timeline_df, cfg)
    print(f"   Found {len(feature_cols)} feature columns")
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in timeline")
    
    # Step 3: Aggregate raw features by day
    print("\n[Step 6.3] Aggregating posts by day...")
    daily_agg = aggregate_all_features_by_day(
        timeline_df=timeline_df,
        feature_cols=feature_cols,
        user_col='author',
        time_col='offset_from_cd1',
    )
    print(f"   Aggregated to {len(daily_agg):,} (user, day) combinations")
    
    print(f"\n Final: {len(daily_agg):,} (user, day) combinations")
    print(f"  Users: {daily_agg['author'].nunique():,}")
    print(f"  Days per user: {len(daily_agg) / daily_agg['author'].nunique():.1f} (average)")

    print(f"Appplying z-score normalization to features...")

    daily_agg[feature_cols] = (
        daily_agg.groupby('author')[feature_cols]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    # Step 4: Save checkpoint (with anchors)
    print("\n[Step 6.4] Saving checkpoint (with anchors)...")
    output_file = save_with_timestamp(
        daily_agg,
        interim_dir,
        files_cfg["daily_aggregated_with_anchors"],
    )
    print(f"   Saved (with anchors): {output_file.name}")
    
    # Optional: aggregate timeline WITHOUT anchors if available
    print("\n[Step 6.5] Aggregating timeline without anchors (if available)...")
    timeline_no_anchors_file = find_latest_file(interim_dir, files_cfg["timeline_no_anchors"] + "_*.csv")
    
    if timeline_no_anchors_file is None:
        print("   No 'timeline_with_offsets_no_anchors_*.csv' file found; skipping no-anchors aggregation.")
    else:
        print(f"   Using timeline without anchors: {timeline_no_anchors_file.name}")
        
        timeline_no_anchors_df = pd.read_csv(
            timeline_no_anchors_file,
            encoding='utf-8-sig',
            low_memory=False
        )
        print(f"   Loaded {len(timeline_no_anchors_df):,} posts from {timeline_no_anchors_file.name}")
        print(f"  Users: {timeline_no_anchors_df['author'].nunique():,}")
        
        # Identify feature columns again (same logic)
        print("\n[Step 6.5.1] Identifying feature columns for no-anchors timeline...")
        feature_cols_no = identify_feature_columns(timeline_no_anchors_df, cfg)
        print(f"   Found {len(feature_cols_no)} feature columns")
        
        if len(feature_cols_no) == 0:
            raise ValueError("No feature columns found in no-anchors timeline")
        
        # Aggregate by day for no-anchors timeline
        print("\n[Step 6.5.2] Aggregating posts by day (no anchors)...")
        daily_agg_no = aggregate_all_features_by_day(
            timeline_df=timeline_no_anchors_df,
            feature_cols=feature_cols_no,
            user_col='author',
            time_col='offset_from_cd1',
        )
        print(f"   Aggregated to {len(daily_agg_no):,} (user, day) combinations (no anchors)")
        print(f"\n Final (no anchors): {len(daily_agg_no):,} (user, day) combinations")
        print(f"  Users: {daily_agg_no['author'].nunique():,}")
        print(f"  Days per user: {len(daily_agg_no) / daily_agg_no['author'].nunique():.1f} (average)")

        print(f"Applying z-score normalization to features (no anchors)...")
        daily_agg_no[feature_cols_no] = (
            daily_agg_no.groupby('author')[feature_cols_no]
            .transform(lambda x: (x - x.mean()) / x.std())
        )

        # Save no-anchors aggregated timeline
        print("\n[Step 6.5.3] Saving no-anchors checkpoint...")
        output_file_no = save_with_timestamp(
            daily_agg_no,
            interim_dir,
            files_cfg["daily_aggregated_no_anchors"],
        )
        print(f"   Saved (no anchors): {output_file_no.name}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6: Aggregate posts by day (normalization happens later in analysis)"
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

