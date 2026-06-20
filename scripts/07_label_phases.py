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
  python scripts/08b_label_phases.py --config configs/base.yaml --fixed-period 29
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
from src.constants import PHASE_ORDER
from src.io import find_latest_file, save_with_timestamp


def main(config_path: str = "configs/base.yaml", no_anchors: bool = False, consensus_file: str | None = None, fixed_period: int | None = None, timeline_file: str | None = None, tag: str | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg = cfg["paths"]["files"]
    anchor_suffix = "_no_anchors" if no_anchors else ""
    period_suffix = f"_fixed{fixed_period}" if fixed_period is not None else ""

    # ── [1] Load consensus periods (skipped when --fixed-period is set) ─────────
    if fixed_period is not None:
        logging.info(f"[1/3] Fixed period mode: {fixed_period} days applied to all users.")
        user_period_map = None  # populated after the timeline is loaded
    else:
        if consensus_file is not None:
            consensus_path = Path(consensus_file)
            if not consensus_path.exists():
                logging.error(f"Consensus file not found: {consensus_file}")
                return 1
        else:
            cp_prefix = files_cfg["consensus_periods"]
            consensus_pattern = f"{cp_prefix}_*{anchor_suffix}_*.csv" if no_anchors else f"{cp_prefix}_*.csv"
            consensus_path = find_latest_file(
                interim_dir,
                consensus_pattern,
                exclude=None if no_anchors else "_no_anchors",
            )
            if consensus_path is None:
                logging.error(
                    f"No {consensus_pattern} found in data/interim/. "
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

    # ── [2] Load timeline ───────────────────────────────────────────────────────
    if timeline_file is not None:
        tl_path = Path(timeline_file)
        if not tl_path.exists():
            tl_path = interim_dir / timeline_file
        if not tl_path.exists():
            logging.error(f"Timeline file not found: {timeline_file}")
            return 1
    elif fixed_period is not None:
        # Prefer eligible-users timeline (step 08a output) if it exists; otherwise
        # fall back to the step-06 z-scored aggregated timeline.
        anchor_tag = "_no_anchors" if no_anchors else ""
        eligible_pattern = f"eligible_users_timeline_minposts*{anchor_tag}_*.csv"
        tl_path = find_latest_file(interim_dir, eligible_pattern,
                                   exclude=None if no_anchors else "_no_anchors")
        if tl_path is not None:
            logging.info(f"  Using eligible-users timeline (step 08a).")
        else:
            zscore_key = "daily_aggregated_no_anchors_zscore" if no_anchors else "daily_aggregated_with_anchors_zscore"
            tl_path = find_latest_file(
                interim_dir,
                files_cfg[zscore_key] + "_*.csv",
                exclude=None if no_anchors else "_no_anchors",
            )
            if tl_path is None:
                logging.error(
                    f"No step-06 zscore timeline found in data/interim/. "
                    "Run scripts/06_aggregate_and_normalize.py first, or pass --timeline-file."
                )
                return 1
            logging.info(f"  No eligible-users timeline found; falling back to step-06 zscore output.")
    else:
        tl_key = "daily_aggregated_no_anchors_zscore" if no_anchors else "daily_aggregated_with_anchors_zscore"
        tl_path = find_latest_file(interim_dir, files_cfg[tl_key] + "_*.csv",
                                   exclude=None if no_anchors else "_no_anchors")
        if tl_path is None:
            logging.error(
                f"No {files_cfg[tl_key]}_*.csv found in data/interim/. "
                "Run scripts/06_aggregate_and_normalize.py first."
            )
            return 1

    logging.info(f"[2/3] Loading daily-aggregated timeline: {tl_path.name}")
    df = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)

    required = {"author", "offset_from_cd1"}
    missing = required - set(df.columns)
    if missing:
        logging.error(f"Input file is missing required columns: {sorted(missing)}")
        return 1
    feature_cols = [c for c in df.columns if c not in ("author", "offset_from_cd1")]
    if not feature_cols:
        logging.error("Input file has no feature columns beyond 'author' and 'offset_from_cd1'.")
        return 1
    logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users  ({len(feature_cols)} feature columns)")

    df["author"] = df["author"].astype(str)

    if fixed_period is not None:
        # All users in the timeline get the fixed period — no filtering needed.
        user_period_map = {u: float(fixed_period) for u in df["author"].unique()}
        logging.info(f"  Assigned period={fixed_period} to {len(user_period_map):,} users")
    else:
        # Filter to detected-period users only.
        df = df[df["author"].isin(user_period_map)].copy()
        logging.info(
            f"  After filtering to detected-period users: "
            f"{len(df):,} user-days, {df['author'].nunique():,} users"
        )

    # ── [3] Assign phases ───────────────────────────────────────────────────────
    logging.info("[3/3] Assigning phase labels…")

    if fixed_period is not None:
        # Vectorized path: same boundaries for every user — no per-row Python loop needed.
        phase_def = compute_user_phase_definitions({" ": float(fixed_period)})
        phase_def = phase_def[phase_def["user"] == " "]
        boundaries = {
            row["phase"]: (int(row["start_day"]), int(row["end_day"]))
            for _, row in phase_def.iterrows()
        }
        day_mod = df["offset_from_cd1"].apply(
            lambda x: int(x) % fixed_period if pd.notna(x) else None
        )
        def _phase(d):
            if d is None:
                return None
            for phase, (s, e) in boundaries.items():
                if s <= d <= e:
                    return phase
            return None
        df["phase"] = day_mod.map(_phase)
    else:
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
    tag_suffix = f"_{tag}" if tag else ""
    out_path = save_with_timestamp(df, interim_dir, files_cfg["phase_labeled"] + period_suffix + tag_suffix + anchor_suffix)
    logging.info(f"  Saved → {out_path.name}")
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--no-anchors", action="store_true",
                        help="Use no-anchors timeline and consensus files.")
    parser.add_argument("--consensus-file", default=None,
                        help="Path to a specific consensus CSV (overrides auto-discovery).")
    parser.add_argument("--fixed-period", type=int, default=None,
                        help="Apply a single fixed cycle length (e.g. 29) to all users, "
                             "skipping consensus lookup. Auto-discovers the step-07a "
                             "eligible-users timeline unless --timeline-file is given.")
    parser.add_argument("--timeline-file", default=None,
                        help="Path or filename (in interim dir) of a specific timeline to use. "
                             "Overrides auto-discovery for both normal and --fixed-period runs.")
    parser.add_argument("--tag", default=None,
                        help="Optional label appended to output filename (e.g. 'bc' → timeline_phase_labeled_fixed28_bc_*.csv).")
    args = parser.parse_args()
    raise SystemExit(main(
        args.config,
        no_anchors=args.no_anchors,
        consensus_file=args.consensus_file,
        fixed_period=args.fixed_period,
        timeline_file=args.timeline_file,
        tag=args.tag,
    ))
