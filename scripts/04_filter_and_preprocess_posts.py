#!/usr/bin/env python3
"""Step 4: Filter and preprocess posts with precomputed features.

Loads moon1 and moon2 posts with precomputed features, filters to users in database,
removes users with ANY NaN in ANY feature, and preprocesses text.

Input:
- User database from step 3: data/processed/users_database_CD_*.csv
- Precomputed posts: data/raw/moon1_all_post_2015_2025_filtered_features.csv
- Precomputed posts: data/raw/moon2_all_post_2015_2025_filtered_features.csv

Output:
- data/interim/posts_all_users_preprocessed_{timestamp}.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.preprocess import (
    filter_users_with_nan_features,
    concatenate_title_selftext,
    add_timestamp_columns,
    fix_removed_posts_selftext,
    filter_deleted_authors,
)
from src.utils import identify_feature_columns


def load_posts_with_features(
    raw_dir: Path,
    files_cfg: dict,
    target_users: set[str],
    min_chars: int = 150,
) -> pd.DataFrame:
    """Load moon1 and moon2 posts with precomputed features.
    
    Args:
        raw_dir: Directory containing raw data files
        files_cfg: Configuration dictionary with file names
        target_users: Set of user IDs to filter to
        min_chars: Minimum text length (applied after concatenation)
        
    Returns:
        DataFrame with posts and features
    """
    posts_files = [
        files_cfg["moon1_posts"],
        files_cfg["moon2_posts"],
    ]
    
    all_posts = []
    
    for posts_file in posts_files:
        posts_path = raw_dir / posts_file
        
        if not posts_path.exists():
            print(f"  Warning: {posts_file} not found, skipping")
            continue
        
        print(f"  Loading {posts_file}...")
        df = pd.read_csv(posts_path, encoding='utf-8-sig', low_memory=False)
        print(f"    Loaded {len(df):,} posts")
        
        # Filter to target users
        if 'author' in df.columns:
            before = len(df)
            df = df[df['author'].isin(target_users)].copy()
            after = len(df)
            print(f"    Filtered to {after:,} posts from target users ({100 * after / before:.1f}%)")
        
        all_posts.append(df)
    
    if not all_posts:
        raise FileNotFoundError("No posts files found or loaded")
    
    combined = pd.concat(all_posts, ignore_index=True)
    
    # Remove duplicates if any
    original_len = len(combined)
    if 'id' in combined.columns:
        combined = combined.drop_duplicates(subset=['id'], keep='first')
        if len(combined) < original_len:
            print(f"  Removed {original_len - len(combined):,} duplicate posts")
    
    return combined


def main(
    config_path: str = "configs/base.yaml",
    use_checkpoint: bool = True,
    force_recompute: bool = False,
):
    """Filter and preprocess posts with precomputed features."""
    cfg = load_config(config_path)
    
    raw_dir = Path(cfg["paths"]["raw"])
    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    min_chars = cfg.get("pipeline", {}).get("min_chars", 150)
    
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 4: Filter and Preprocess Posts with Features")
    print("=" * 60)
    print()
    
    files_cfg = cfg["paths"]["files"]

    # Check for checkpoint (with anchors version)
    checkpoint = find_latest_file(interim_dir, files_cfg["posts_with_anchors"] + "_*.csv")

    if use_checkpoint and not force_recompute and checkpoint:
        print(f" Found checkpoint: {checkpoint.name}")
        print(f"  To recompute, use --force-recompute")
        return 0
    
    # Step 1: Load user database
    print("[Step 4.1] Loading user database...")
    users_db_file = find_latest_file(processed_dir, files_cfg["users_db_cd"])
    
    if not users_db_file:
        raise FileNotFoundError(
            f"No user database found in {processed_dir}. "
            "Please run scripts/03_create_users_database.py first."
        )
    
    users_df = pd.read_csv(users_db_file, encoding='utf-8-sig')
    target_users = set(users_df["user"].astype(str))
    print(f"   Loaded {len(target_users):,} users from {users_db_file.name}")
    
    # Step 2: Load posts with features
    print("\n[Step 4.2] Loading posts with precomputed features...")
    posts_df = load_posts_with_features(
        raw_dir=raw_dir,
        files_cfg=cfg["paths"]["files"],
        target_users=target_users,
        min_chars=min_chars,
    )
    print(f"   Total posts loaded: {len(posts_df):,}")
    
    # Step 3: Identify feature columns
    print("\n[Step 4.3] Identifying feature columns...")
    feature_cols = identify_feature_columns(posts_df, cfg)
    print(f"   Found {len(feature_cols)} feature columns")
    
    # Step 4: Remove users with ANY NaN in ANY feature
    print("\n[Step 4.4] Removing users with NaN in features...")
    before_users = posts_df['author'].nunique()
    posts_df, removed_users = filter_users_with_nan_features(
        posts_df,
        feature_cols,
        user_col='author'
    )
    after_users = posts_df['author'].nunique()
    print(f"  Removed {len(removed_users):,} users with NaN features")
    print(f"  Users remaining: {after_users:,} (from {before_users:,})")
    
    if len(posts_df) == 0:
        raise ValueError("No posts remaining after filtering")
    
    # Step 5: Preprocess posts
    print("\n[Step 4.5] Preprocessing posts...")
    
    # Filter deleted authors
    before = len(posts_df)
    posts_df = filter_deleted_authors(posts_df)
    after = len(posts_df)
    print(f"  Removed deleted authors: {before - after:,} posts")
    
    # Fix removed/deleted selftext
    posts_df = fix_removed_posts_selftext(posts_df)
    
    # Concatenate title + selftext
    posts_df = concatenate_title_selftext(posts_df)
    
    # Add timestamps
    if 'created_utc' in posts_df.columns:
        posts_df = add_timestamp_columns(posts_df, utc_col='created_utc')
    
    # Filter by minimum characters
    before = len(posts_df)
    posts_df = posts_df[posts_df['text'].str.len() >= min_chars].copy()
    after = len(posts_df)
    print(f"  Filtered by min_chars={min_chars}: {after:,} posts remaining (from {before:,})")
    
    print(f"\n Final (before anchor removal): {len(posts_df):,} posts from {posts_df['author'].nunique():,} users")
    
    # Step 5.5: Create version without anchor posts
    # Anchors are identified in the users database by (user, timestep);
    # after preprocessing we have (author, ts_utc). We remove any posts
    # whose (author, ts_utc) pair appears in the users database.
    posts_no_anchors = posts_df.copy()
    if 'timestep' in users_df.columns and 'ts_utc' in posts_df.columns:
        print("\n[Step 4.5b] Removing anchor posts based on user database...")
        
        anchor_keys = users_df[['user', 'timestep']].dropna().copy()
        anchor_keys['user'] = anchor_keys['user'].astype(str)
        anchor_keys['timestep'] = pd.to_datetime(anchor_keys['timestep'])
        
        posts_df['author'] = posts_df['author'].astype(str)
        posts_df['ts_utc'] = pd.to_datetime(posts_df['ts_utc'])
        
        # Build a boolean mask for anchors
        anchor_set = set(zip(anchor_keys['user'], anchor_keys['timestep']))
        is_anchor = [
            (u, t) in anchor_set
            for u, t in zip(posts_df['author'], posts_df['ts_utc'])
        ]
        is_anchor = pd.Series(is_anchor, index=posts_df.index)
        
        n_anchors = int(is_anchor.sum())
        print(f"  Identified {n_anchors:,} anchor posts")
        
        posts_no_anchors = posts_df[~is_anchor].copy()
        print(f"  Posts without anchors: {len(posts_no_anchors):,}")
    else:
        print("\n[Step 4.5b] Warning: timestep/ts_utc columns missing; cannot remove anchor posts.")
    
    # Step 6: Save checkpoints (with and without anchors)
    print("\n[Step 4.6] Saving checkpoints...")
    
    output_with = save_with_timestamp(
        posts_df,
        interim_dir,
        files_cfg["posts_with_anchors"],
    )
    print(f"   Saved (with anchors): {output_with.name}")

    output_no = save_with_timestamp(
        posts_no_anchors,
        interim_dir,
        files_cfg["posts_no_anchors"],
    )
    print(f"   Saved (no anchors): {output_no.name}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Filter and preprocess posts with precomputed features"
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

