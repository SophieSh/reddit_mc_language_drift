#!/usr/bin/env python3
"""Step 08b: Assign menstrual cycle phases to the daily-aggregated timeline.

Reads the consensus periods from step 08 and the daily-aggregated timeline from
step 06.  Filters to detected-period users and labels each user-day with its
adaptive phase (Menstrual / Follicular / Ovulation / Luteal).

Run this ONCE after step 08.  All downstream scripts (10, 12, ...) load the
output file instead of recomputing phase labels independently.

Input:
  data/interim/consensus_periods_*.csv          (step 08 output)
  data/interim/timeline_daily_aggregated_with_anchors_*.csv  (step 06 output)

Output:
  data/interim/timeline_phase_labeled_*.csv
    Columns: all columns from the daily-aggregated timeline + 'phase'

Usage:
  python scripts/08b_label_phases.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.analysis import assign_phases_to_timeline, compute_user_phase_definitions
from src.config import load_config
from src.io import find_latest_file, save_with_timestamp

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]


def main(config_path: str = "configs/base.yaml") -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg = cfg["paths"]["files"]

    # ── [1] Load consensus periods ──────────────────────────────────────────────
    consensus_path = find_latest_file(
        interim_dir,
        files_cfg["consensus_periods"] + "_*.csv",
        exclude="_no_anchors",
    )
    if consensus_path is None:
        logging.error(
            "No consensus_periods_*.csv found in data/interim/. "
            "Run scripts/08_consensus.py first."
        )
        return 1

    logging.info(f"[1/3] Loading consensus periods: {consensus_path.name}")
    consensus_df = pd.read_csv(consensus_path, encoding="utf-8-sig")
    if "consensus_period" not in consensus_df.columns or "user" not in consensus_df.columns:
        logging.error("Consensus file missing 'user' or 'consensus_period' columns.")
        return 1

    valid = consensus_df[consensus_df["consensus_period"].notna()]
    user_period_map = dict(
        zip(valid["user"].astype(str), valid["consensus_period"].astype(float))
    )
    logging.info(f"  {len(user_period_map):,} users with detected cycle lengths")

    # ── [2] Load daily-aggregated timeline ──────────────────────────────────────
    tl_path = find_latest_file(
        interim_dir,
        files_cfg["daily_aggregated_with_anchors"] + "_*.csv",
    )
    if tl_path is None:
        logging.error(
            "No timeline_daily_aggregated_with_anchors_*.csv found in data/interim/. "
            "Run scripts/06_aggregate_and_normalize.py first."
        )
        return 1

    logging.info(f"[2/3] Loading daily-aggregated timeline: {tl_path.name}")
    df = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users")

    # Filter to detected-period users
    df["author"] = df["author"].astype(str)
    df = df[df["author"].isin(user_period_map)].copy()
    logging.info(
        f"  After filtering to detected-period users: "
        f"{len(df):,} user-days, {df['author'].nunique():,} users"
    )

    # ── [3] Assign phases ───────────────────────────────────────────────────────
    logging.info("[3/3] Assigning adaptive phase labels…")
    upm_filtered = {u: p for u, p in user_period_map.items() if u in set(df["author"])}
    user_phase_df = compute_user_phase_definitions(upm_filtered)
    df = assign_phases_to_timeline(
        timeline_df=df,
        user_phase_df=user_phase_df,
        user_col="author",
        time_col="offset_from_cd1",
    )

    before = len(df)
    df = df[df["phase"].notna() & df["phase"].isin(PHASE_ORDER)].copy()
    if before - len(df):
        logging.warning(f"  Dropped {before - len(df):,} user-days with unassignable phase")

    logging.info(f"  Phase distribution:\n{df['phase'].value_counts().to_string()}")

    # ── Save ────────────────────────────────────────────────────────────────────
    out_path = save_with_timestamp(df, interim_dir, files_cfg["phase_labeled"])
    logging.info(f"  Saved → {out_path.name}")
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
