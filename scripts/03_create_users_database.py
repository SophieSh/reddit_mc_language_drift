#!/usr/bin/env python3
"""Step 3: Create users database with latest anchor post details.

For each user:
- Filters out uncertain posts (keeps only has_uncertainty == False)
- Requires valid offset_from_cd1 (not null)
- If a user has multiple anchor posts, selects the most recent one (by timestamp)
- Output columns: user, timestep, text, offset_from_cd1, pattern_category

Creates two separate files:
1. Users with patterns 1, 2, 3, 4, 5, 6, 9
2. Users with pattern 7 (DPO); pattern_8 is excluded
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.utils import load_latest_preprocessed_file


def create_users_database(interim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create database with one row per user (most recent anchor post).

    Filters for non-uncertain posts with valid offset_from_cd1.
    For users with multiple anchor posts, selects the most recent one.
    Separates into two groups: patterns 1-6+9 and pattern_7 (DPO); pattern_8 is excluded.

    Returns:
        Tuple of (df_patterns_1_6_9, df_pattern_7) DataFrames.
    """
    pattern_files = {
        "moon1": "moon_with_uncertainty_*.csv",
        "moon2": "moon2_with_uncertainty_*.csv",
        "moon3": "moon3_with_uncertainty_*.csv",
    }

    all_anchors: list[pd.DataFrame] = []

    for moon_type, pattern in pattern_files.items():
        df = load_latest_preprocessed_file(interim_dir, pattern)
        if df is None:
            print(f"Warning: No preprocessed file found for {moon_type}")
            continue

        print(f"\n{moon_type}:")
        print(f"  Total posts: {len(df)}")

        if "has_uncertainty" not in df.columns or "offset_from_cd1" not in df.columns:
            print("  Warning: Required columns missing, skipping")
            continue

        if "author" not in df.columns:
            print("  Warning: author column missing, skipping")
            continue

        before = len(df)
        df_filtered = df[(~df["has_uncertainty"]) & (df["offset_from_cd1"].notna())].copy()
        after = len(df_filtered)
        print(f"  After filtering uncertain and null offsets: {after} ({100 * after / before:.1f}%)")

        if len(df_filtered) == 0:
            continue

        all_anchors.append(df_filtered)

    if not all_anchors:
        print("\nNo anchor posts found after filtering")
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_anchors, ignore_index=True)
    print(f"\nTotal anchor posts across all sources: {len(combined)}")

    if "ts_utc" not in combined.columns:
        print("Error: ts_utc column missing")
        return pd.DataFrame(), pd.DataFrame()

    combined["ts_utc"] = pd.to_datetime(combined["ts_utc"])

    # Exclude pattern_8 ("lmp was [Month]") - date parsing is less reliable than other patterns
    combined = combined[combined["regex_type"] != "pattern_8"].copy()

    patterns_1_6_9 = [
        "pattern_1",
        "pattern_2",
        "pattern_3",
        "pattern_4",
        "pattern_5",
        "pattern_6",
        "pattern_9",
    ]
    patterns_dpo = ["pattern_7"]

    df_patterns_1_6_9 = combined[combined["regex_type"].isin(patterns_1_6_9)].copy()
    df_pattern_7 = combined[combined["regex_type"].isin(patterns_dpo)].copy()

    print("\nSelecting most recent anchor post per user...")


    def select_latest_per_user(df_group: pd.DataFrame) -> pd.DataFrame:
        """Select latest anchor post per user from a pattern group."""
        if len(df_group) == 0:
            return pd.DataFrame()

        print(
            f"  {df_group['regex_type'].iloc[0] if len(df_group) > 0 else 'unknown'}: "
            f"{df_group['author'].nunique()} users before deduplication"
        )

        df_sorted = df_group.sort_values("ts_utc", ascending=False)
        df_unique = df_sorted.drop_duplicates(subset=["author"], keep="first").copy()

        print(f"  After selecting latest anchor: {len(df_unique)} users")

        output_cols = {
            "author": "user",
            "ts_utc": "timestep",
            "text": "text",
            "offset_from_cd1": "offset_from_cd1",
            "regex_type": "pattern_category",
        }

        result = df_unique[list(output_cols.keys())].copy()
        result = result.rename(columns=output_cols)
        result = result.sort_values("user").reset_index(drop=True)
        return result


    result_1_6_9 = select_latest_per_user(df_patterns_1_6_9)
    result_dpo = select_latest_per_user(df_pattern_7)

    if len(result_1_6_9) > 0 and len(result_dpo) > 0:
        users_1_6_9 = set(result_1_6_9["user"].unique())
        users_dpo = set(result_dpo["user"].unique())
        overlap = users_1_6_9 & users_dpo

        if len(overlap) > 0:
            # Prioritize patterns 1-6+9 (explicit period mentions) over pattern_7 (DPO)
            # DPO requires ovulation day assumption (default 14), while 1-6+9 are direct reports
            print(f"\nFound {len(overlap)} users in both groups")
            print("  Removing them from DPO (keeping in patterns 1-6+9)")
            result_dpo = result_dpo[~result_dpo["user"].isin(overlap)].copy()
            print(f"  DPO users after removal: {len(result_dpo)}")

    return result_1_6_9, result_dpo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create users database with latest anchor post details",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating users database with latest anchor posts")
    print("=" * 60)

    result_1_6_9, result_dpo = create_users_database(interim_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    if len(result_1_6_9) > 0:
        output_path_1_6_9 = output_dir / f"users_database_CD_{timestamp}.csv"
        result_1_6_9.to_csv(output_path_1_6_9, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 60)
        print("Patterns 1, 2, 3, 4, 5, 6, 9")
        print("=" * 60)
        print(f"Total unique users: {len(result_1_6_9)}")
        print("\nPattern distribution:")
        pattern_counts = result_1_6_9["pattern_category"].value_counts().sort_index()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count}")

        print("\nOffset statistics:")
        print(f"  Min: {result_1_6_9['offset_from_cd1'].min()}")
        print(f"  Max: {result_1_6_9['offset_from_cd1'].max()}")
        print(f"  Mean: {result_1_6_9['offset_from_cd1'].mean():.1f}")
        print(f"  Median: {result_1_6_9['offset_from_cd1'].median():.1f}")

        print(f"\nSaved to: {output_path_1_6_9}")
    else:
        print("\nNo users found for patterns 1, 2, 3, 4, 5, 6, 9")

    if len(result_dpo) > 0:
        output_path_dpo = output_dir / f"users_database_DPO_{timestamp}.csv"
        result_dpo.to_csv(output_path_dpo, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 60)
        print("Pattern 7 (DPO)")
        print("=" * 60)
        print(f"Total unique users: {len(result_dpo)}")

        print("\nOffset statistics:")
        print(f"  Min: {result_dpo['offset_from_cd1'].min()}")
        print(f"  Max: {result_dpo['offset_from_cd1'].max()}")
        print(f"  Mean: {result_dpo['offset_from_cd1'].mean():.1f}")
        print(f"  Median: {result_dpo['offset_from_cd1'].median():.1f}")

        print(f"\nSaved to: {output_path_dpo}")
    else:
        print("\nNo users found for pattern_7 (DPO)")


if __name__ == "__main__":
    main()

