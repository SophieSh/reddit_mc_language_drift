#!/usr/bin/env python3
"""Step 9: Visualize cycle length distributions.

Creates histograms showing the distribution of detected cycle lengths,
both from individual features and from consensus periods.

Input:
- Periodicity results from step 7: data/interim/periodicity_results_*.csv
- Consensus results from step 8: data/interim/consensus_periods_*.csv (optional)

Output:
- reports/cycle_distributions_*.png
- reports/consensus_distribution_*.png (if consensus available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.io import find_latest_file
from src.visualization import plot_cycle_distributions
from src.timeline import load_users_database


def main(
    config_path: str = "configs/base.yaml",
    periodicity_file: str | None = None,
    consensus_file: str | None = None,
    use_checkpoint: bool = True,
    force_recompute: bool = False,
    use_original_features_only: bool = False,
    pattern_1_only: bool = False,
):
    """Visualize cycle length distributions."""
    cfg = load_config(config_path)
    
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 9: Visualize Cycle Length Distributions")
    print("=" * 60)
    print()
    
    # Step 1: Load periodicity results
    print("[Step 9.1] Loading periodicity results...")
    
    if periodicity_file:
        periodicity_path = interim_dir / periodicity_file
        if not periodicity_path.exists():
            raise FileNotFoundError(f"Periodicity file not found: {periodicity_path}")
    else:
        periodicity_path = find_latest_file(interim_dir, "periodicity_results_*.csv")
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
        print(f"  Filtering to pattern_1 users only (moon1)...")
        users_df = load_users_database(cfg, db_type="cd", pattern="pattern_1")
        pattern_1_users = set(users_df['user'].astype(str))
        
        before_count = len(periodicity_df)
        before_users = periodicity_df['user'].nunique()
        periodicity_df = periodicity_df[periodicity_df['user'].isin(pattern_1_users)].copy()
        after_count = len(periodicity_df)
        after_users = periodicity_df['user'].nunique()
        
        print(f"   Filtered to {after_count:,} results from {after_users:,} pattern_1 users (from {before_count:,} results, {before_users:,} users)")
    
    # Get unique features
    features = sorted(periodicity_df['feature'].unique())
    print(f"  Features: {len(features)}")
    print(f"  Methods: {sorted(periodicity_df['method'].unique())}")
    
    # Step 2: Plot individual feature distributions
    print(f"\n[Step 9.2] Plotting cycle length distributions by feature...")
    
    # Check for existing plot
    output_pattern = f"cycle_distributions_*.png"
    existing_plot = find_latest_file(reports_dir, output_pattern)
    
    if use_checkpoint and not force_recompute and existing_plot:
        print(f"   Found existing plot: {existing_plot.name}")
        print(f"  To recompute, use --force-recompute")
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        method_str = "_".join(sorted(periodicity_df['method'].unique()))
        output_path = reports_dir / f"cycle_distributions_{method_str}_{timestamp}.png"
        
        plot_cycle_distributions(
            periodicity_df,
            features,
            output_path,
            period_wide_max=None,  # Auto-detect from data
        )
        print(f"   Saved: {output_path.name}")
    
    # Step 3: Plot consensus distribution (if available)
    print(f"\n[Step 9.3] Checking for consensus results...")
    
    if consensus_file:
        consensus_path = interim_dir / consensus_file
        if not consensus_path.exists():
            print(f"  ⚠ Consensus file not found: {consensus_path}")
            consensus_path = None
    else:
        consensus_path = find_latest_file(interim_dir, "consensus_periods_*.csv")
    
    if consensus_path:
        print(f"   Found consensus results: {consensus_path.name}")
        consensus_df = pd.read_csv(consensus_path, encoding='utf-8-sig', low_memory=False)
        print(f"  Users with consensus: {len(consensus_df):,}")
        
        # Filter to pattern_1 users only if requested (moon1 only, excluding moon2)
        if pattern_1_only:
            print(f"  Filtering to pattern_1 users only (moon1)...")
            users_df = load_users_database(cfg, db_type="cd", pattern="pattern_1")
            pattern_1_users = set(users_df['user'].astype(str))
            
            before_count = len(consensus_df)
            consensus_df = consensus_df[consensus_df['user'].isin(pattern_1_users)].copy()
            after_count = len(consensus_df)
            
            print(f"   Filtered to {after_count:,} pattern_1 users (from {before_count:,} total)")
        
        # Create consensus distribution plot
        import matplotlib.pyplot as plt
        import numpy as np
        
        consensus_plot_pattern = "consensus_distribution_*.png"
        existing_consensus_plot = find_latest_file(reports_dir, consensus_plot_pattern)
        
        if use_checkpoint and not force_recompute and existing_consensus_plot:
            print(f"   Found existing consensus plot: {existing_consensus_plot.name}")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            periods = consensus_df['consensus_period'].values
            bins = range(int(periods.min()), int(periods.max()) + 2)
            
            period_mean = np.mean(periods)
            period_median = np.median(periods)
            
            ax.hist(periods, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(period_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {period_mean:.1f}d')
            ax.axvline(period_median, color='blue', linestyle='--', linewidth=2, 
                      label=f'Median: {period_median:.1f}d')
            
            ax.set_xlabel('Cycle Length (days)', fontsize=12)
            ax.set_ylabel('Number of Users', fontsize=12)
            ax.set_title(
                f'Cycle Length Distribution (≥{consensus_df["n_features_agreeing"].min()} features agreeing)\n'
                f'SNR≥3.0',
                fontsize=14,
                fontweight='bold'
            )
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            
            # Add statistics text box
            stats_text = (
                f"Mean: {np.mean(periods):.1f} days\n"
                f"Median: {np.median(periods):.1f} days\n"
                f"Std: {np.std(periods):.1f} days\n"
                f"N: {len(periods)} users"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            min_features = consensus_df['n_features_agreeing'].min()
            consensus_output_path = reports_dir / f"consensus_distribution_min{min_features}features_{timestamp}.png"
            plt.savefig(consensus_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved: {consensus_output_path.name}")
    else:
        print(f"  ⚠ No consensus results found. Run scripts/08_consensus.py first to generate consensus.")
    
    print(f"\n Visualization complete!")
    print(f"  Reports saved to: {reports_dir}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 9: Visualize cycle length distributions"
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
        "--consensus-file",
        type=str,
        default=None,
        help="Specific consensus results file to use (default: find latest)"
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
        help="Use only original 11 features for visualization (negative_sentiment, positive_sentiment, num_words, avg_word_length, num_sentences, unique_word_fraction, readability, spelling_errors_frac, syntactic_complexity_subordination_index, cohesion_analysis_lexical_overlap)"
    )
    parser.add_argument(
        "--pattern-1-only",
        action="store_true",
        help="Filter to only pattern_1 users (moon1 only, excluding moon2 and moon3)"
    )
    
    args = parser.parse_args()
    
    exit(main(
        config_path=args.config,
        periodicity_file=args.periodicity_file,
        consensus_file=args.consensus_file,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
        use_original_features_only=args.original_features_only,
        pattern_1_only=args.pattern_1_only,
    ))

