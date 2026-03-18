#!/usr/bin/env python3
"""Step 7: Run FFT periodicity detection and calculate SNR.

Runs FFT periodicity detection on all features using the old approach:
- analyze_user_timeline receives raw feature columns (ending in _mean from aggregation)
- It aggregates by day internally (if needed), normalizes, then runs FFT
- This matches the old working version's data flow

Input:
- Daily aggregated timeline from step 6: data/interim/timeline_daily_aggregated_with_anchors_*.csv
  (columns: author, offset_from_cd1, {feature}_mean for each feature)
  Note: FFT analysis always uses the timeline WITH anchors.

Output:
- data/interim/periodicity_results_{timestamp}.csv
  (columns: user, feature, method, period, power, peak_to_background, n_points, span_days)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import load_config, MIN_DATA_POINTS, DEFAULT_PERIOD_MIN, DEFAULT_PERIOD_MAX
from src.io import find_latest_file, save_with_timestamp
from src.analysis import (
    analyze_user_timeline,
)
from src.utils import identify_feature_columns


def main(
    config_path: str = "configs/base.yaml",
    method: str = "fft_interpolation",
    snr_threshold: float | None = None,
    use_checkpoint: bool = True,
    force_recompute: bool = False,
    no_anchors: bool = False,
):
    """Run FFT periodicity detection on all features."""
    cfg = load_config(config_path)

    interim_dir = Path(cfg["paths"]["interim"])
    analysis_cfg = cfg.get("analysis", {})

    # Get parameters from config
    period_min = analysis_cfg.get("period_min", DEFAULT_PERIOD_MIN)
    period_max = analysis_cfg.get("period_max", DEFAULT_PERIOD_MAX)
    if snr_threshold is None:
        snr_threshold = analysis_cfg.get("snr_threshold", 3.0)

    anchor_suffix = "_no_anchors" if no_anchors else ""

    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 7: Run FFT Periodicity Detection")
    print("=" * 60)
    print(f"  Method: {method}")
    print(f"  Period range: {period_min}-{period_max} days")
    print(f"  SNR threshold: {snr_threshold}")
    print(f"  Anchor posts: {'excluded' if no_anchors else 'included'}")
    print()

    # Check for checkpoint
    checkpoint_pattern = f"periodicity_results_{method}_snr{snr_threshold}{anchor_suffix}_*.csv"
    checkpoint = find_latest_file(interim_dir, checkpoint_pattern, exclude=None if no_anchors else "_no_anchors")

    if use_checkpoint and not force_recompute and checkpoint:
        print(f" Found checkpoint: {checkpoint.name}")
        print(f"  To recompute, use --force-recompute")
        return 0

    # Step 1: Load daily aggregated timeline
    timeline_pattern = (
        "timeline_daily_aggregated_no_anchors_*.csv" if no_anchors
        else "timeline_daily_aggregated_with_anchors_*.csv"
    )
    print(f"[Step 7.1] Loading daily aggregated timeline ({('no anchors' if no_anchors else 'with anchors')})...")
    timeline_file = find_latest_file(interim_dir, timeline_pattern)

    if not timeline_file:
        raise FileNotFoundError(
            f"No daily aggregated timeline ({timeline_pattern}) found in {interim_dir}. "
            "Please run scripts/06_aggregate_and_normalize.py first."
        )
    
    daily_agg = pd.read_csv(timeline_file, encoding='utf-8-sig', low_memory=False)
    print(f"   Loaded {len(daily_agg):,} (user, day) combinations from {timeline_file.name}")
    print(f"  Users: {daily_agg['author'].nunique():,}")
    
    # Step 2: Identify feature columns (ending in _mean from aggregation)
    print("\n[Step 7.2] Identifying feature columns...")
    all_cols = set(daily_agg.columns)
    metadata_cols = {'author', 'offset_from_cd1', 'text', 'ts_utc'}
    
    # Find _mean columns (from aggregation step)
    mean_cols = [col for col in all_cols if col.endswith('_mean') and col not in metadata_cols]
    
    if len(mean_cols) == 0:
        raise ValueError("No feature columns found (expected _mean columns from aggregation step)")
    
    # Get base feature names (remove _mean suffix)
    feature_base_names = [col.replace('_mean', '') for col in mean_cols]
    
    print(f"   Found {len(mean_cols)} features to analyze")
    
    # Step 3: Run FFT analysis for each user-feature combination
    # Use old approach: analyze_user_timeline handles aggregation and normalization internally
    print(f"\n[Step 7.3] Running {method} analysis...")
    print(f"  Using old approach: analyze_user_timeline handles normalization internally")
    print(f"  This may take several minutes...")
    
    results = []
    total_combinations = len(mean_cols) * daily_agg['author'].nunique()
    processed = 0
    
    # Determine which methods to run based on method parameter
    if method == "fft_interpolation_integer":
        methods_to_run = ["fft_interpolation"]  # Will use integer periods version if needed
    else:
        methods_to_run = [method]
    
    for feature_col, feature_name in zip(mean_cols, feature_base_names):
        for user, user_df in daily_agg.groupby('author'):
            processed += 1
            
            if processed % 100000 == 0:
                print(f"    Progress: {processed:,}/{total_combinations:,} ({100*processed/total_combinations:.1f}%)")
            
            # Use analyze_user_timeline (old version) which handles normalization internally
            # Pass the _mean column - analyze_user_timeline will aggregate (if needed) and normalize
            analysis = analyze_user_timeline(
                user_timeline=user_df,
                feature_col=feature_col,  # Pass column name with _mean suffix
                period_min=period_min,
                period_max=period_max,
                normalize_method="zscore",  # Normalize after aggregation (old approach)
                methods=methods_to_run,
            )
            
            if analysis is None:
                continue
            
            # Extract FFT interpolation results
            if "fft_interp_period" in analysis:
                peak_to_background = analysis.get("fft_interp_peak_to_background", 0.0)
                
                # Only include if SNR meets threshold
                if peak_to_background < snr_threshold:
                    continue
                
                results.append({
                    "user": user,
                    "feature": feature_name,
                    "method": method,
                    "period": analysis["fft_interp_period"],
                    "power": analysis["fft_interp_power"],
                    "peak_to_background": peak_to_background,
                    "n_points": analysis["n_points"],
                    "span_days": analysis["span_days"],
                })
    
    # Print final progress
    print(f"    Progress: {processed:,}/{total_combinations:,} (100.0%)")
    
    if len(results) == 0:
        print(f"\n  ⚠ No results found (all below SNR threshold {snr_threshold})")
        return 1
    
    results_df = pd.DataFrame(results)
    
    print(f"\n Analysis complete: {len(results_df):,} results")
    print(f"  Users with detections: {results_df['user'].nunique():,}")
    print(f"  Features with detections: {results_df['feature'].nunique()}")
    
    # Step 4: Save checkpoint
    print("\n[Step 7.4] Saving checkpoint...")
    output_file = save_with_timestamp(
        results_df,
        interim_dir,
        f"periodicity_results_{method}_snr{snr_threshold}{anchor_suffix}"
    )
    print(f"   Saved: {output_file.name}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  Mean period: {results_df['period'].mean():.1f} days")
    print(f"  Median period: {results_df['period'].median():.1f} days")
    print(f"  Mean SNR: {results_df['peak_to_background'].mean():.2f}")
    print(f"  Median SNR: {results_df['peak_to_background'].median():.2f}")
    
    # Print per-feature statistics
    print(f"\nPer-feature period statistics:")
    feature_stats = results_df.groupby('feature')['period'].agg(['mean', 'median', 'count']).round(1)
    feature_stats.columns = ['Mean Period', 'Median Period', 'N Detections']
    feature_stats = feature_stats.sort_values('Mean Period')
    
    # Print in a readable format
    print(f"  {'Feature':<40} {'Mean':>8} {'Median':>8} {'N':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8}")
    for feature, row in feature_stats.iterrows():
        print(f"  {feature:<40} {row['Mean Period']:>8.1f} {row['Median Period']:>8.1f} {int(row['N Detections']):>8}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 7: Run FFT periodicity detection and calculate SNR"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fft_interpolation",
        help="Analysis method: 'fft_interpolation' (default, uses old approach with internal normalization)"
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=None,
        help="SNR threshold (default: from config)"
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
    parser.add_argument(
        "--no-anchors",
        action="store_true",
        help="Use timeline without anchor posts (timeline_daily_aggregated_no_anchors_*.csv)"
    )

    args = parser.parse_args()

    exit(main(
        config_path=args.config,
        method=args.method,
        snr_threshold=args.snr_threshold,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
        no_anchors=args.no_anchors,
    ))

