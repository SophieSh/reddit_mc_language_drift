#!/usr/bin/env python3
"""Step 11: Linguistic phase analysis for stable BC pill users.

Takes users identified as STABLE_USING in step 10, assumes a fixed 28-day cycle,
and assigns menstrual phases using offset relative to the BC mention post (not CD1).

The BC mention post is used as the local phase anchor (day 0). This avoids the problem
of CD1 anchors being years away from the BC mention, which would make phase assignment
arbitrary. Phase is assigned as (offset_from_cd1 - bc_mention_day) % 28.

If a user has multiple BC mention posts, the median offset_from_cd1 is used as anchor.

Input:
  data/interim/bc_users_stable_*.csv          -- STABLE_USING users from step 10
  data/interim/timeline_with_offsets_with_anchors_*.csv  -- full timeline with features

Output:
  data/interim/bc_stable_phase_stats_[ts].csv   -- phase statistics per feature
  reports/bc_stable_phase_plot_[ts].png         -- bar chart (one panel per feature)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file
from src.utils import identify_feature_columns
from src.visualization import aggregate_features_by_phase, plot_phase_analysis

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")


def main(config_path: str = "configs/base.yaml", period: float = 28.0,
         window: int = 90,
         bc_file: str | None = None, timeline_file: str | None = None) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 11: BC Stable Users — Linguistic Phase Analysis")
    print("=" * 60)
    print(f"  Cycle: {period} days  |  Window: ±{window} days around BC mention")
    print()

    # ---- Load BC stable users ------------------------------------------------
    if bc_file:
        bc_path = Path(bc_file)
    else:
        bc_path = find_latest_file(interim_dir, "bc_users_stable_*.csv")
    if not bc_path:
        raise FileNotFoundError(f"No bc_users_stable_*.csv in {interim_dir}. Run step 10 first.")
    print(f"[1] BC stable users: {bc_path.name}")
    bc_df = pd.read_csv(bc_path)

    # One anchor per user: median offset_from_cd1 of their BC mention posts
    bc_anchor = (
        bc_df.groupby("author")["offset_from_cd1"]
        .median()
        .reset_index()
        .rename(columns={"offset_from_cd1": "bc_anchor_day"})
    )
    bc_users = set(bc_anchor["author"])
    print(f"    {len(bc_users)} unique STABLE_USING users")
    print(f"    BC anchor = median offset_from_cd1 of their BC mention post(s)")

    # ---- Load timeline -------------------------------------------------------
    if timeline_file:
        tl_path = Path(timeline_file)
    else:
        tl_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    if not tl_path:
        raise FileNotFoundError(f"No timeline file in {interim_dir}. Run step 05 first.")
    print(f"\n[2] Timeline: {tl_path.name}")
    timeline = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)
    print(f"    {len(timeline):,} posts, {timeline['author'].nunique():,} users (full)")

    # ---- Filter to BC stable users and compute bc_relative_offset -----------
    timeline_bc = timeline[timeline["author"].isin(bc_users)].copy()
    timeline_bc = timeline_bc.merge(bc_anchor, on="author", how="inner")
    timeline_bc["bc_relative_offset"] = (
        timeline_bc["offset_from_cd1"] - timeline_bc["bc_anchor_day"]
    )

    # Keep only posts within ±window days of the BC mention
    timeline_bc = timeline_bc[
        timeline_bc["bc_relative_offset"].between(-window, window)
    ].copy()
    print(f"    {len(timeline_bc):,} posts, {timeline_bc['author'].nunique():,} users "
          f"(within ±{window} days of BC mention)")

    if timeline_bc.empty:
        print("  No posts found within window. Try increasing --window.")
        return 1

    # ---- Identify feature columns --------------------------------------------
    # Drop helper columns before feature detection so they aren't misidentified
    timeline_bc = timeline_bc.drop(columns=["bc_anchor_day"], errors="ignore")
    features = identify_feature_columns(timeline_bc, cfg)
    # Ensure the time axis column is never treated as a feature
    features = [f for f in features if f != "bc_relative_offset"]
    print(f"\n[3] Features: {len(features)} linguistic features")

    # ---- Phase aggregation using bc_relative_offset as time axis ------------
    print(f"\n[4] Aggregating features by phase (fixed {period}-day cycle, "
          f"anchored to BC mention)...")
    phase_df = aggregate_features_by_phase(
        timeline_df=timeline_bc,
        results_df=pd.DataFrame(),
        features=features,
        time_col="bc_relative_offset",
        user_col="author",
        fixed_cycle_length=period,
        normalize=True,
        average_per_user=True,
    )

    if phase_df.empty:
        print("  Phase aggregation returned no results.")
        return 1

    # ---- Save phase statistics -----------------------------------------------
    out_stats = interim_dir / f"bc_stable_phase_stats_{TIMESTAMP}.csv"
    phase_df.to_csv(out_stats, index=False)
    print(f"\n[5] Saved phase stats → {out_stats.name}")

    # ---- Plot ----------------------------------------------------------------
    print(f"\n[6] Plotting...")
    out_plot = reports_dir / f"bc_stable_phase_plot_{TIMESTAMP}.png"
    feature_names = sorted(phase_df["feature"].unique().tolist())
    plot_phase_analysis(
        phase_df=phase_df,
        features=feature_names,
        output_path=out_plot,
        error_bar_type="sem",
        average_per_user=True,
    )
    print(f"    Saved → {out_plot}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BC stable users linguistic phase analysis")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--period", type=float, default=28.0,
                    help="Assumed cycle length in days (default: 28)")
    ap.add_argument("--window", type=int, default=90,
                    help="Days before/after BC mention to include (default: 90)")
    ap.add_argument("--bc-file", default=None,
                    help="Path to bc_users_stable_*.csv (default: latest in interim)")
    ap.add_argument("--timeline-file", default=None,
                    help="Path to timeline CSV (default: latest with_anchors in interim)")
    args = ap.parse_args()
    exit(main(config_path=args.config, period=args.period, window=args.window,
              bc_file=args.bc_file, timeline_file=args.timeline_file))
