#!/usr/bin/env python3
"""Step 07a: Filter step-06 timeline to eligible users.

Keeps only users who meet minimum activity criteria, then saves the filtered
daily-aggregated timeline.  Runs both the with-anchors and no-anchors variants
by default.

Note: step 05 already limits the timeline to ±3 months (~91d) around the anchor,
so no additional windowing is needed here.

Criteria:
  - at least min_post_days distinct posting days
  - span (max_offset - min_offset) >= min_span_days

Input:
  data/interim/timeline_daily_aggregated_with_anchors_*.csv  (step 06 output)
  data/interim/timeline_daily_aggregated_no_anchors_*.csv    (step 06 output)

Output:
  data/interim/eligible_users_timeline_minposts{N}_span{S}_*.csv
  data/interim/eligible_users_timeline_minposts{N}_span{S}_no_anchors_*.csv
  (same columns as step 06 output, rows restricted to eligible users)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp


def _filter_eligible(
    daily_agg: pd.DataFrame,
    min_post_days: int,
    min_span_days: int,
) -> pd.DataFrame:
    rows = []
    for user, grp in daily_agg.groupby("author"):
        n = len(grp)
        span = grp["offset_from_cd1"].max() - grp["offset_from_cd1"].min()
        if n >= min_post_days and span >= min_span_days:
            rows.append({"user": user, "n_post_days": n, "span_days": span})
    return pd.DataFrame(rows)


def _run_variant(
    label: str,
    timeline_file: Path,
    min_post_days: int,
    min_span_days: int,
    interim_dir: Path,
    out_stem: str,
) -> int:
    print(f"\n[{label}] Loading: {timeline_file.name}")
    daily_agg = pd.read_csv(timeline_file, encoding="utf-8-sig", low_memory=False)
    total_users = daily_agg["author"].nunique()
    print(f"  {len(daily_agg):,} (user, day) rows, {total_users:,} users")

    eligible_df = _filter_eligible(daily_agg, min_post_days, min_span_days)
    eligible_users = set(eligible_df["user"])
    print(f"  Eligible: {len(eligible_users):,} / {total_users:,} users "
          f"({100 * len(eligible_users) / max(total_users, 1):.1f}%)")

    if len(eligible_users) == 0:
        print("  No eligible users found.")
        return 1

    filtered = daily_agg[daily_agg["author"].isin(eligible_users)].copy()
    print(f"  Filtered table: {len(filtered):,} rows")

    output_file = save_with_timestamp(filtered, interim_dir, out_stem)
    print(f"  Saved: {output_file.name}")
    return 0


def main(
    config_path: str = "configs/base.yaml",
    min_post_days: int = 20,
    min_span_days: int = 100,
    no_anchors_only: bool = False,
    with_anchors_only: bool = False,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg = cfg["paths"]["files"]

    print("=" * 60)
    print("Step 07a: Select Eligible Users")
    print("=" * 60)
    print(f"  Min post days : {min_post_days}")
    print(f"  Min span      : {min_span_days} days")

    run_with = not no_anchors_only
    run_without = not with_anchors_only

    exit_code = 0

    if run_with:
        pattern = files_cfg["daily_aggregated_with_anchors"] + "_*.csv"
        timeline_file = find_latest_file(interim_dir, pattern, exclude="_no_anchors")
        if not timeline_file:
            print("\n  [with anchors] No timeline file found — skipping.")
            exit_code = 1
        else:
            rc = _run_variant(
                label="with anchors",
                timeline_file=timeline_file,
                min_post_days=min_post_days,
                min_span_days=min_span_days,
                interim_dir=interim_dir,
                out_stem=f"eligible_users_timeline_minposts{min_post_days}_span{min_span_days}",
            )
            exit_code = exit_code or rc

    if run_without:
        pattern_no = files_cfg["daily_aggregated_no_anchors"] + "_*.csv"
        timeline_file_no = find_latest_file(interim_dir, pattern_no)
        if not timeline_file_no:
            print("\n  [no anchors] No timeline file found — skipping.")
            exit_code = 1
        else:
            rc = _run_variant(
                label="no anchors",
                timeline_file=timeline_file_no,
                min_post_days=min_post_days,
                min_span_days=min_span_days,
                interim_dir=interim_dir,
                out_stem=f"eligible_users_timeline_minposts{min_post_days}_span{min_span_days}_no_anchors",
            )
            exit_code = exit_code or rc

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 07a: Select eligible users")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--min-post-days", type=int, default=20,
                        help="Min posting days within window (default: 20)")
    parser.add_argument("--min-span-days", type=int, default=100,
                        help="Min span in days within window (default: 100)")
    parser.add_argument("--no-anchors-only", action="store_true",
                        help="Only run the no-anchors variant")
    parser.add_argument("--with-anchors-only", action="store_true",
                        help="Only run the with-anchors variant")

    args = parser.parse_args()
    exit(main(
        config_path=args.config,
        min_post_days=args.min_post_days,
        min_span_days=args.min_span_days,
        no_anchors_only=args.no_anchors_only,
        with_anchors_only=args.with_anchors_only,
    ))
