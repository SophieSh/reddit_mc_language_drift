#!/usr/bin/env python3
"""Step 6: Aggregate posts by day and save raw + z-scored versions.

For each anchor variant (with anchors / without anchors) two files are saved:
  - raw:    author, offset_from_cd1, {feature}_mean   …
  - zscore: author, offset_from_cd1, {feature}_zscore … (per-user normalised)

Input:
  data/interim/timeline_with_offsets_with_anchors_*.csv  (step 05)
  data/interim/timeline_with_offsets_no_anchors_*.csv    (step 05, optional)

Output:
  data/interim/timeline_daily_aggregated_with_anchors_raw_*.csv
  data/interim/timeline_daily_aggregated_with_anchors_zscore_*.csv
  data/interim/timeline_daily_aggregated_no_anchors_raw_*.csv    (if no-anchors file exists)
  data/interim/timeline_daily_aggregated_no_anchors_zscore_*.csv (if no-anchors file exists)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.analysis import aggregate_all_features_by_day
from src.utils import identify_feature_columns

INDEX_COLS = ["author", "offset_from_cd1"]


def aggregate_and_save(
    timeline_path: Path,
    cfg: dict,
    interim_dir: Path,
    key_raw: str,
    key_zscore: str,
) -> None:
    """Aggregate one timeline file and save raw + z-scored outputs."""
    files_cfg = cfg["paths"]["files"]

    df = pd.read_csv(timeline_path, encoding="utf-8-sig", low_memory=False)
    print(f"   Loaded {len(df):,} posts  |  {df['author'].nunique():,} users")

    feature_cols = identify_feature_columns(df, cfg)
    if not feature_cols:
        raise ValueError(f"No feature columns found in {timeline_path.name}")
    print(f"   Features: {len(feature_cols)}")

    daily = aggregate_all_features_by_day(
        timeline_df=df,
        feature_cols=feature_cols,
        user_col="author",
        time_col="offset_from_cd1",
    )
    print(
        f"   Aggregated: {len(daily):,} (user, day) rows  |  "
        f"{daily['author'].nunique():,} users  |  "
        f"{len(daily) / daily['author'].nunique():.1f} days/user avg"
    )

    mean_cols = [c for c in daily.columns if c not in INDEX_COLS and c.endswith("_mean")]
    zscore_cols = [c.replace("_mean", "_zscore") for c in mean_cols]
    daily[zscore_cols] = (
        daily.groupby("author")[mean_cols]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    raw_out = save_with_timestamp(
        daily[INDEX_COLS + mean_cols],
        interim_dir,
        files_cfg[key_raw],
    )
    print(f"   Raw    → {raw_out.name}")

    zscore_out = save_with_timestamp(
        daily[INDEX_COLS + zscore_cols],
        interim_dir,
        files_cfg[key_zscore],
    )
    print(f"   Zscore → {zscore_out.name}")


def main(
    config_path: str = "configs/base.yaml",
    use_checkpoint: bool = True,
    force_recompute: bool = False,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    files_cfg = cfg["paths"]["files"]

    print("=" * 60)
    print("Step 6: Aggregate by Day")
    print("=" * 60)

    # ── WITH ANCHORS ──────────────────────────────────────────────────────────
    checkpoint = find_latest_file(
        interim_dir, files_cfg["daily_aggregated_with_anchors_zscore"] + "_*.csv"
    )
    if use_checkpoint and not force_recompute and checkpoint:
        print(f"\n Found checkpoint: {checkpoint.name}  (use --force-recompute to redo)")
    else:
        print("\n[6.1] Aggregating timeline WITH anchors…")
        timeline = find_latest_file(
            interim_dir, files_cfg["timeline_with_anchors"] + "_*.csv"
        )
        if not timeline:
            raise FileNotFoundError(
                f"No timeline_with_anchors file in {interim_dir}. Run script 05 first."
            )
        aggregate_and_save(
            timeline, cfg, interim_dir,
            key_raw="daily_aggregated_with_anchors_raw",
            key_zscore="daily_aggregated_with_anchors_zscore",
        )

    # ── WITHOUT ANCHORS (optional) ────────────────────────────────────────────
    timeline_no = find_latest_file(
        interim_dir, files_cfg["timeline_no_anchors"] + "_*.csv"
    )
    if timeline_no is None:
        print("\n[6.2] No no-anchors timeline found; skipping.")
    else:
        checkpoint_no = find_latest_file(
            interim_dir, files_cfg["daily_aggregated_no_anchors_zscore"] + "_*.csv"
        )
        if use_checkpoint and not force_recompute and checkpoint_no:
            print(f"\n Found no-anchors checkpoint: {checkpoint_no.name}  (use --force-recompute to redo)")
        else:
            print("\n[6.2] Aggregating timeline WITHOUT anchors…")
            aggregate_and_save(
                timeline_no, cfg, interim_dir,
                key_raw="daily_aggregated_no_anchors_raw",
                key_zscore="daily_aggregated_no_anchors_zscore",
            )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    exit(main(
        config_path=args.config,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
    ))
