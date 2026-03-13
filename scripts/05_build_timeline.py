#!/usr/bin/env python3
"""Step 5: Build timeline by calculating offsets from anchor posts.

Calculates offset_from_cd1 for each post using anchor posts from user database.
Optionally filters by time window around anchor.

Input:
- Preprocessed posts from step 4: data/interim/posts_all_users_preprocessed_*.csv
- User database from step 3: data/processed/users_database_CD_*.csv

Output:
- data/interim/timeline_with_offsets_{timestamp}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.timeline import build_anchor_dict
from src.preprocess import add_offsets_from_anchors


def main(
    config_path: str = "configs/base.yaml",
    window_months: int | None = None,
    use_checkpoint: bool = True,
    force_recompute: bool = False,
):
    """Build timeline by calculating offsets from anchor posts."""
    cfg = load_config(config_path)
    
    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get window_months from config if not provided
    # Default to 3 months if not specified (matches old working version)
    if window_months is None:
        window_months = cfg.get("pipeline", {}).get("window_months", 3)
    
    print("=" * 60)
    print("Step 5: Build Timeline (Calculate Offsets from Anchors)")
    print("=" * 60)
    if window_months:
        print(f"  Time window: ±{window_months} months")
    print()
    
    # Check for checkpoint (with anchors timeline)
    checkpoint = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    
    if use_checkpoint and not force_recompute and checkpoint:
        print(f" Found checkpoint: {checkpoint.name}")
        print(f"  To recompute, use --force-recompute")
        return 0
    
    # Step 1: Load preprocessed posts (with anchors)
    print("[Step 5.1] Loading preprocessed posts...")
    posts_file = find_latest_file(interim_dir, "posts_all_users_preprocessed_with_anchors_*.csv")
    
    if not posts_file:
        raise FileNotFoundError(
            f"No preprocessed posts found in {interim_dir}. "
            "Please run scripts/04_filter_and_preprocess_posts.py first."
        )
    
    posts_df = pd.read_csv(posts_file, encoding='utf-8-sig', low_memory=False)
    print(f"   Loaded {len(posts_df):,} posts from {posts_file.name}")
    print(f"  Users: {posts_df['author'].nunique():,}")
    
    # Step 2: Load user database
    print("\n[Step 5.2] Loading user database...")
    users_db_file = find_latest_file(processed_dir, "users_database_CD_*.csv")
    
    if not users_db_file:
        raise FileNotFoundError(
            f"No user database found in {processed_dir}. "
            "Please run scripts/03_create_users_database.py first."
        )
    
    users_df = pd.read_csv(users_db_file, encoding='utf-8-sig')
    print(f"   Loaded {len(users_df):,} users from {users_db_file.name}")
    
    # Step 3: Build anchor dictionary
    print("\n[Step 5.3] Building anchor dictionary...")
    anchors = build_anchor_dict(users_df, pattern_type="cd")
    print(f"   Created anchor dictionary for {len(anchors):,} users")
    
    # Step 4: Calculate offsets
    print("\n[Step 5.4] Calculating offsets from anchors...")
    timeline_df = add_offsets_from_anchors(
        posts_df,
        anchors,
        author_col='author',
        timestamp_col='ts_utc',
        pattern_type='cd'
    )
    
    # Count posts with valid offsets
    valid_offsets = timeline_df['offset_from_cd1'].notna().sum()
    print(f"   Calculated offsets: {valid_offsets:,} posts with valid offsets (from {len(timeline_df):,})")
    
    # Filter to posts with valid offsets
    timeline_df = timeline_df[timeline_df['offset_from_cd1'].notna()].copy()
    
    # Step 5: Time window filtering (applied by default to match old working version)
        print(f"\n[Step 5.5] Filtering to ±{window_months} months around anchor...")
        from src.config import AVG_DAYS_PER_MONTH
        from src.preprocess import filter_posts_by_anchor_window
        
        before = len(timeline_df)
        timeline_df = filter_posts_by_anchor_window(
            timeline_df,
            users_df,
            window_months=window_months,
            user_col='author'
        )
        after = len(timeline_df)
    print(f"   Filtered to {after:,} posts (from {before:,})")
    
    print(f"\n Final timeline: {len(timeline_df):,} posts from {timeline_df['author'].nunique():,} users")
    
    # Step 6: Save checkpoint (with anchors)
    print("\n[Step 5.6] Saving checkpoint (with anchors)...")
    output_file = save_with_timestamp(
        timeline_df,
        interim_dir,
        "timeline_with_offsets_with_anchors"
    )
    print(f"   Saved (with anchors): {output_file.name}")
    
    # Step 7: Optionally build timeline WITHOUT anchors
    from src.io import find_latest_file as _find_latest  # reuse helper with local alias
    from src.preprocess import filter_posts_by_anchor_window
    
    print("\n[Step 5.7] Building timeline without anchors (if posts file exists)...")
    no_anchors_file = _find_latest(interim_dir, "posts_all_users_preprocessed_no_anchors_*.csv")
    
    if no_anchors_file is None:
        print("  No 'posts_all_users_preprocessed_no_anchors_*.csv' file found; skipping timeline_without_anchors.")
    else:
        print(f"  Using posts file without anchors: {no_anchors_file.name}")
        posts_no_anchors_df = pd.read_csv(no_anchors_file, encoding='utf-8-sig', low_memory=False)
        print(f"   Loaded {len(posts_no_anchors_df):,} posts from {posts_no_anchors_df['author'].nunique():,} users")
        
        # Calculate offsets from anchors
        print("  Calculating offsets from anchors (no anchors in posts)...")
        timeline_no_anchors_df = add_offsets_from_anchors(
            posts_no_anchors_df,
            anchors,
            author_col='author',
            timestamp_col='ts_utc',
            pattern_type='cd'
        )
        
        valid_offsets_no = timeline_no_anchors_df['offset_from_cd1'].notna().sum()
        print(f"   Calculated offsets (no anchors): {valid_offsets_no:,} posts with valid offsets "
              f"(from {len(timeline_no_anchors_df):,})")
        
        # Filter to posts with valid offsets
        timeline_no_anchors_df = timeline_no_anchors_df[
            timeline_no_anchors_df['offset_from_cd1'].notna()
        ].copy()
        
        # Apply the same time window filtering
        print(f"  Filtering to ±{window_months} months around anchor (no anchors)...")
        before_no = len(timeline_no_anchors_df)
        timeline_no_anchors_df = filter_posts_by_anchor_window(
            timeline_no_anchors_df,
            users_df,
            window_months=window_months,
            user_col='author'
        )
        after_no = len(timeline_no_anchors_df)
        print(f"   Filtered to {after_no:,} posts (from {before_no:,})")
        
        print(f"  Final timeline without anchors: {len(timeline_no_anchors_df):,} posts "
              f"from {timeline_no_anchors_df['author'].nunique():,} users")
        
        # Save checkpoint without anchors
        output_file_no = save_with_timestamp(
            timeline_no_anchors_df,
            interim_dir,
            "timeline_with_offsets_no_anchors"
        )
        print(f"   Saved (no anchors): {output_file_no.name}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 5: Build timeline by calculating offsets from anchor posts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=None,
        help="Time window in months around anchor (optional, default: no filtering)"
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
        window_months=args.window_months,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
    ))

