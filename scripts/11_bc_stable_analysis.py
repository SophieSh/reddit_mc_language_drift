#!/usr/bin/env python3
"""Step 11: Linguistic phase analysis for stable BC pill users.

Takes users identified as STABLE_USING in step 10, assumes a fixed 28-day cycle,
assigns menstrual phases using offset_from_cd1, and plots feature values by phase
(same visualization as the general population analysis).

Input:
  data/interim/bc_users_stable_*.csv          -- STABLE_USING users from step 10
  data/interim/timeline_with_offsets_with_anchors_*.csv  -- full timeline with features

Output:
  data/interim/bc_stable_phase_stats_[ts].csv   -- phase statistics per feature
  data/interim/bc_stable_phase_plot_[ts].png    -- bar chart (one panel per feature)
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
         bc_file: str | None = None, timeline_file: str | None = None) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    figures_dir = Path(cfg["paths"].get("figures", "data/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 11: BC Stable Users — Linguistic Phase Analysis")
    print("=" * 60)
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
    bc_users = set(bc_df["author"].unique())
    print(f"    {len(bc_users)} unique STABLE_USING users")

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

    # ---- Filter to BC stable users ------------------------------------------
    timeline_bc = timeline[timeline["author"].isin(bc_users)].copy()
    print(f"    {len(timeline_bc):,} posts, {timeline_bc['author'].nunique():,} users (BC stable)")

    if timeline_bc.empty:
        print("  No BC stable users found in timeline. Check that bc_users_stable file "
              "uses the same usernames as the timeline.")
        return 1

    # ---- Identify feature columns --------------------------------------------
    features = identify_feature_columns(timeline_bc, cfg)
    print(f"\n[3] Features: {len(features)} linguistic features")

    # ---- Phase aggregation with fixed 28-day cycle ---------------------------
    print(f"\n[4] Aggregating features by phase (fixed {period}-day cycle)...")
    phase_df = aggregate_features_by_phase(
        timeline_df=timeline_bc,
        results_df=pd.DataFrame(),   # not needed when fixed_cycle_length is set
        features=features,
        time_col="offset_from_cd1",
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
    out_plot = figures_dir / f"bc_stable_phase_plot_{TIMESTAMP}.png"
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
    ap.add_argument("--bc-file", default=None,
                    help="Path to bc_users_stable_*.csv (default: latest in interim)")
    ap.add_argument("--timeline-file", default=None,
                    help="Path to timeline CSV (default: latest with_anchors in interim)")
    args = ap.parse_args()
    exit(main(config_path=args.config, period=args.period,
              bc_file=args.bc_file, timeline_file=args.timeline_file))
