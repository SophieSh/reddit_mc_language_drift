#!/usr/bin/env python3
"""Step 8: Apply X-feature consensus to assign final cycle length per user.

Takes periodicity results from step 7 and finds consensus period across features
using majority vote (within tolerance). Only assigns consensus if enough features agree.

Input:
- Periodicity results from step 7: data/interim/periodicity_results_*.csv

Output:
- data/interim/consensus_periods_*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.analysis import assign_consensus_period_by_majority
from src.timeline import load_users_database


def main(
    config_path: str = "configs/base.yaml",
    periodicity_file: str | None = None,
    min_features: int | None = None,
    tolerance: int | None = None,
    use_checkpoint: bool = True,
    force_recompute: bool = False,
    use_original_features_only: bool = False,
    pattern_1_only: bool = False,
    no_anchors: bool = False,
):
    """Apply consensus period assignment across features."""
    cfg = load_config(config_path)
    
    interim_dir = Path(cfg["paths"]["interim"])
    pipeline_cfg = cfg.get("pipeline", {})
    
    # Get parameters from config or arguments
    if min_features is None:
        min_features = pipeline_cfg.get("min_features_for_consensus", 5)
    if tolerance is None:
        tolerance = pipeline_cfg.get("consensus_tolerance", 1)
    
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 8: Apply X-Feature Consensus")
    print("=" * 60)
    anchor_suffix = "_no_anchors" if no_anchors else ""

    print(f"  Min features for consensus: {min_features}")
    print(f"  Tolerance: ±{tolerance} days")
    print(f"  Anchor posts: {'excluded' if no_anchors else 'included'}")
    print()

    # Check for checkpoint
    checkpoint = find_latest_file(interim_dir, f"consensus_periods_*{anchor_suffix}_*.csv")
    
    if use_checkpoint and not force_recompute and checkpoint:
        print(f" Found checkpoint: {checkpoint.name}")
        print(f"  To recompute, use --force-recompute")
        return 0
    
    # Step 1: Load periodicity results
    print("[Step 8.1] Loading periodicity results...")
    
    if periodicity_file:
        periodicity_path = interim_dir / periodicity_file
        if not periodicity_path.exists():
            raise FileNotFoundError(f"Periodicity file not found: {periodicity_path}")
    else:
        # Find latest periodicity results file (no_anchors variant if requested)
        search_pattern = "periodicity_results_*_no_anchors_*.csv" if no_anchors else "periodicity_results_*.csv"
        periodicity_path = find_latest_file(interim_dir, search_pattern)
        if not periodicity_path:
            raise FileNotFoundError(
                f"No periodicity results found in {interim_dir}. "
                "Please run scripts/07_run_fft_analysis.py first."
            )
    
    periodicity_df = pd.read_csv(periodicity_path, encoding='utf-8-sig', low_memory=False)
    print(f"   Loaded {len(periodicity_df):,} periodicity results from {periodicity_path.name}")
    
    # Original features used in biologically-aligned results
    ORIGINAL_FEATURES = [
        'negative_sentiment',           # Negative Sentiment
        'positive_sentiment',           # Positive Sentiment
        'num_words',                    # Word Count
        'avg_word_length',              # Avg. Word Length
        'num_sentences',                # Avg. Words/Sentence
        'unique_word_fraction',         # Unique Word Fraction
        'readability',                  # Flesch-Kincaid
        'spelling_errors_frac',         # Spelling Error Fraction
        'syntactic_complexity_subordination_index',  # Syntactic Complexity
        'cohesion_analysis_lexical_overlap',  # Cohesion
        # Note: MATTR not found in current features, using unique_word_fraction as lexical diversity measure
    ]
    
    # Filter to original features if requested
    if use_original_features_only:
        original_count = len(periodicity_df)
        periodicity_df = periodicity_df[periodicity_df['feature'].isin(ORIGINAL_FEATURES)].copy()
        filtered_count = len(periodicity_df)
        print(f"   Filtered to {filtered_count:,} results from {len(ORIGINAL_FEATURES)} original features (from {original_count:,} total)")
    
    # Filter to pattern_1 users only if requested (moon1 only, excluding moon2)
    if pattern_1_only:
        print(f"\n[Step 8.1.5] Filtering to pattern_1 users only (moon1)...")
        users_df = load_users_database(cfg, db_type="cd", pattern="pattern_1")
        pattern_1_users = set(users_df['user'].astype(str))
        
        before_count = len(periodicity_df)
        before_users = periodicity_df['user'].nunique()
        periodicity_df = periodicity_df[periodicity_df['user'].isin(pattern_1_users)].copy()
        after_count = len(periodicity_df)
        after_users = periodicity_df['user'].nunique()
        
        print(f"   Filtered to {after_count:,} results from {after_users:,} pattern_1 users (from {before_count:,} results, {before_users:,} users)")
    
    print(f"  Users: {periodicity_df['user'].nunique():,}")
    print(f"  Features: {periodicity_df['feature'].nunique()}")
    
    # Step 2: Apply consensus
    print(f"\n[Step 8.2] Applying consensus (min_features={min_features}, tolerance=±{tolerance})...")
    
    consensus_df = assign_consensus_period_by_majority(
        periodicity_df,
        min_features=min_features,
        tolerance=tolerance,
    )
    
    if len(consensus_df) == 0:
        print(f"  ⚠ No consensus found (insufficient features agreeing)")
        print(f"    Try reducing min_features or increasing tolerance")
        return 1
    
    print(f"   Consensus assigned to {len(consensus_df):,} users")
    
    # Step 3: Print statistics
    print(f"\n[Step 8.3] Consensus statistics:")
    print(f"  Users with consensus: {len(consensus_df):,}")
    print(f"  Mean consensus period: {consensus_df['consensus_period'].mean():.1f} days")
    print(f"  Median consensus period: {consensus_df['consensus_period'].median():.1f} days")
    print(f"  Mean features agreeing: {consensus_df['n_features_agreeing'].mean():.1f}")
    print(f"  Mean consensus percentage: {consensus_df['consensus_pct'].mean():.1f}%")
    
    # Period distribution
    print(f"\n  Consensus period distribution:")
    period_counts = consensus_df['consensus_period'].value_counts().sort_index()
    for period, count in period_counts.items():
        print(f"    {int(period)} days: {count:,} users")
    
    # Step 4: Save checkpoint
    print(f"\n[Step 8.4] Saving checkpoint...")
    output_file = save_with_timestamp(
        consensus_df,
        interim_dir,
        f"consensus_periods_min{min_features}features{anchor_suffix}"
    )
    print(f"   Saved: {output_file.name}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 8: Apply X-feature consensus to assign final cycle length per user"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--periodicity-file",
        type=str,
        default=None,
        help="Specific periodicity results file to use (default: find latest)"
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=None,
        help="Minimum features required for consensus (default: from config)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=None,
        help="Tolerance in days for consensus (±N days, default: from config)"
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
        "--original-features-only",
        action="store_true",
        help="Use only original 11 features for consensus (negative_sentiment, positive_sentiment, num_words, avg_word_length, num_sentences, unique_word_fraction, readability, spelling_errors_frac, syntactic_complexity_subordination_index, cohesion_analysis_lexical_overlap)"
    )
    parser.add_argument(
        "--pattern-1-only",
        action="store_true",
        help="Filter to only pattern_1 users (moon1 only, excluding moon2 and moon3)"
    )
    parser.add_argument(
        "--no-anchors",
        action="store_true",
        help="Use periodicity results from no-anchors FFT run (periodicity_results_*_no_anchors_*.csv)"
    )

    args = parser.parse_args()

    exit(main(
        config_path=args.config,
        periodicity_file=args.periodicity_file,
        min_features=args.min_features,
        tolerance=args.tolerance,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
        use_original_features_only=args.original_features_only,
        pattern_1_only=args.pattern_1_only,
        no_anchors=args.no_anchors,
    ))

