#!/usr/bin/env python3
"""Full pipeline for pattern_1 periodicity analysis.

Pipeline Overview:
1. Load pattern_1 users (those with "I got my period" anchor posts)
2. Load & filter posts and comments (min 150 chars by default)
3. Preprocess: clean text, add timestamps
4. Calculate offset_from_cd1 for each post/comment (days from anchor)
5. Filter to ±N months around anchor (reduces noise from long-term drift)
6. Combine posts + comments into unified timeline
7. Compute features:
   - Sentiment: VADER (compound, positive, negative), TextBlob (polarity, intensity)
   - Linguistic: word count, syntactic complexity
8. Run periodicity detection (Lomb-Scargle + 4 FFT variants) on ALL 7 features
   - Analyzes RAW features with zscore/minmax normalization
   - Compares: Does normalization method affect detected cycle?
9. Visualize cycle length distributions

Key Design Decisions:
- Normalization happens INSIDE periodicity analysis (per-user, on filtered data)
- Time window filter (±6 months) applied BEFORE analysis to focus on single cycles
- Checkpointing at each step for long-running pipeline
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp
from src.timeline import (
    select_users_by_pattern,
    load_posts_from_raw,
    load_comments_from_raw,
    preprocess_content,
    load_users_database,
    build_anchor_dict,
)
from src.preprocess import (
    add_offsets_from_anchors,
    filter_posts_by_anchor_window,
)
from src.features import (
    compute_vader_sentiment,
    compute_textblob_sentiment,
    compute_linguistic_features,
)
from src.analysis import analyze_all_users_with_normalizations
from src.visualization import plot_cycle_distributions


def main(
    config_path: str,
    run_config_path: str | None = None,
    patterns: list[str] | None = None,
    window_months: int = 6,
    min_chars: int = 150,
    use_checkpoints: bool = True,
    force_recompute: bool = False,
    posts_only: bool = False,
    output_subdir: str | None = None,
):
    """Run full pipeline for periodicity analysis (single or multiple patterns).
    
    Args:
        config_path: Path to base config YAML
        run_config_path: Optional path to run-specific config YAML
        patterns: Optional list of patterns to analyze (e.g., ["pattern_1", "pattern_2"])
        window_months: Months before/after anchor to include (default 6)
        min_chars: Minimum text length (default 150)
        use_checkpoints: Use existing files if found (default True)
        force_recompute: Ignore checkpoints and recompute (default False)
        posts_only: Skip comments, use posts only (default False)
        output_subdir: Optional subdirectory for this run's outputs (e.g., "run_20251218")
    """
    cfg = load_config(config_path)
    
    # Load run configuration if provided (otherwise use empty dict)
    run_cfg = {}
    if run_config_path:
        import yaml
        with open(run_config_path, 'r') as f:
            run_cfg = yaml.safe_load(f) or {}
    
    # Extract all parameters once (run config overrides command line, which overrides base config)
    patterns = patterns or run_cfg.get("patterns", ["pattern_1"])
    window_months = run_cfg.get("window_months", window_months)
    min_chars = run_cfg.get("min_chars", min_chars)
    posts_only = run_cfg.get("posts_only", posts_only)
    output_subdir = run_cfg.get("output_subdir", output_subdir)
    
    # Analysis configuration: run config → base config (required in base.yaml by design)
    analysis_cfg = cfg["analysis"]  # Will fail if missing - good, means base.yaml is incomplete
    analysis_methods = run_cfg.get("analysis_methods", analysis_cfg["methods"])
    normalization_method = run_cfg.get("normalization", analysis_cfg["normalization"])
    fap_threshold = run_cfg.get("fap_threshold", analysis_cfg["fap_threshold"])
    snr_threshold = run_cfg.get("snr_threshold", analysis_cfg["snr_threshold"])
    
    # Setup directories
    raw_dir = Path(cfg["paths"]["raw"])
    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    reports_dir = Path(cfg["paths"]["reports"])
    
    # Create optional subdirectory for organized outputs
    if output_subdir:
        interim_dir = interim_dir / output_subdir
        reports_dir = reports_dir / output_subdir
    
    interim_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    # Features to analyze for periodicity
    base_features = [
        # VADER sentiment features
        'sentiment_compound',      # VADER compound sentiment (-1 to +1)
        'sentiment_positive',      # VADER positive proportion (0-1)
        'sentiment_negative',      # VADER negative proportion (0-1)
        # TextBlob sentiment features
        'textblob_polarity',       # TextBlob polarity (-1 to +1)
        'textblob_subjectivity',   # TextBlob subjectivity (0=objective, 1=subjective)
        'textblob_intensity',      # TextBlob intensity (absolute polarity)
        # Linguistic features
        'word_count',              # Total number of words
        'avg_word_length',         # Average characters per word
        'avg_words_per_sentence',  # Average words per sentence
        'unique_word_fraction',    # Lexical diversity (0-1)
        'syntactic_complexity',    # Subordinate clauses per sentence
        'cohesion',                # Semantic similarity (0-1)
        'flesch_kincaid',          # Reading grade level
        'mattr',                   # MATTR - Moving Average Type-Token Ratio (lexical richness)
        'spelling_error_fraction', # Fraction of misspelled words (0-1)
    ]
    
    # Determine pattern type (CD or DPO)
    if 'pattern_type' in run_cfg:
        # Use explicit pattern_type from run config
        pattern_type = run_cfg["pattern_type"]
    else:
        # Infer from patterns (fallback when pattern_type not specified)
        pattern_types_map = cfg.get("pattern_types", {"cd": ["pattern_1"], "dpo": ["pattern_7"]})
        is_dpo = any(p in pattern_types_map.get("dpo", []) for p in patterns)
        pattern_type = "dpo" if is_dpo else "cd"
    
    # Determine which posts files to load
    if pattern_type == "cd":
        # For CD patterns: check which moon sources we need
        pattern_sources = cfg.get("pattern_sources", {})
        required_sources = set(pattern_sources.get(p, "moon1") for p in patterns)
        posts_files = [cfg["paths"]["files"][f"{source}_posts"] 
                       for source in required_sources 
                       if f"{source}_posts" in cfg["paths"]["files"]]
    else:  # dpo
        # For DPO patterns: always use moon3
        posts_files = [cfg["paths"]["files"]["moon3_posts"]]
    
    print("="*70)
    print(f"FULL PIPELINE: Periodicity Analysis")
    print(f"  Patterns: {', '.join(patterns)}")
    print(f"  Pattern type: {pattern_type.upper()}")
    print(f"  Posts files: {', '.join(posts_files) if posts_files else 'None'}")
    print(f"  Window: ±{window_months} months")
    print(f"  Min chars: {min_chars}")
    print(f"  Content: {'Posts only' if posts_only else 'Posts + Comments'}")
    print(f"  Analysis methods: {', '.join(analysis_methods)}")
    print(f"  Normalization: {normalization_method}")
    print(f"  SNR threshold: {snr_threshold}")
    print(f"  FAP threshold: {fap_threshold}")
    print(f"  Checkpoints: {'Enabled' if use_checkpoints and not force_recompute else 'Disabled'}")
    if output_subdir:
        print(f"  Output subdir: {output_subdir}")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load users database
    # ========================================================================
    print("\n[Step 1/10] Loading users database...")
    users_df = load_users_database(cfg, db_type=pattern_type)
    
    # Filter to specified patterns if needed
    if patterns:
        users_df = users_df[users_df["pattern_category"].isin(patterns)].copy()
        print(f"  Filtered to patterns {patterns}: {len(users_df)} users")
    
    target_users = set(users_df["user"].astype(str))
    print(f"  ✓ Loaded {len(target_users):,} users")
    
    # ========================================================================
    # STEP 2: Load & filter posts
    # ========================================================================
    print(f"\n[Step 2/10] Loading & filtering posts (min_chars={min_chars})...")
    
    patterns_str = "_".join(patterns)
    posts_checkpoint = find_latest_file(interim_dir, f"posts_{patterns_str}_filtered_minchars{min_chars}_*.csv")
    
    if use_checkpoints and not force_recompute and posts_checkpoint:
        print(f"  ✓ Found checkpoint: {posts_checkpoint.name}")
        posts_df = pd.read_csv(posts_checkpoint, encoding='utf-8-sig', low_memory=False)
        print(f"  ✓ Loaded {len(posts_df):,} posts")
    else:
        # Load posts from all required source files
        all_posts = []
        for posts_filename in posts_files:
            posts_path = raw_dir / posts_filename
            if posts_path.exists():
                print(f"  Loading {posts_filename}...")
                posts_df_chunk = load_posts_from_raw(posts_path, target_users, min_chars=min_chars)
                all_posts.append(posts_df_chunk)
                print(f"    ✓ Loaded {len(posts_df_chunk):,} posts")
            else:
                print(f"  ⚠️  Warning: {posts_filename} not found, skipping")
        
        if all_posts:
            posts_df = pd.concat(all_posts, ignore_index=True)
            # Remove duplicates if any (in case users appear in multiple files)
            posts_df = posts_df.drop_duplicates(subset=["id"], keep="first")
            print(f"  ✓ Total: {len(posts_df):,} posts after deduplication")
        else:
            raise FileNotFoundError("No posts files found for specified patterns")
        
        posts_checkpoint = save_with_timestamp(
            posts_df, interim_dir, f"posts_{patterns_str}_filtered_minchars{min_chars}"
        )
        print(f"  ✓ Saved: {posts_checkpoint.name}")
    
    # ========================================================================
    # STEP 3: Load & filter comments (optional)
    # ========================================================================
    if posts_only:
        print(f"\n[Step 3/10] Skipping comments (--posts-only flag)")
        comments_df = None
    else:
        print(f"\n[Step 3/10] Loading & filtering comments (min_chars={min_chars})...")
        print("  ⚠️  This may take 10-20 minutes for 40GB file...")
        
        comments_checkpoint = find_latest_file(interim_dir, f"comments_pattern1_filtered_minchars{min_chars}_*.csv")
        
        if use_checkpoints and not force_recompute and comments_checkpoint:
            print(f"  ✓ Found checkpoint: {comments_checkpoint.name}")
            comments_df = pd.read_csv(comments_checkpoint, encoding='utf-8-sig', low_memory=False)
            print(f"  ✓ Loaded {len(comments_df):,} comments")
        else:
            comments_filename = cfg["paths"]["files"].get("comments_all", "moon_all_comments_2009_2025.tsv")
            comments_path = raw_dir / comments_filename
            comments_df = load_comments_from_raw(comments_path, target_users, min_chars=min_chars)
            print(f"  ✓ Loaded {len(comments_df):,} comments")
            
            comments_checkpoint = save_with_timestamp(
                comments_df, interim_dir, f"comments_pattern1_filtered_minchars{min_chars}"
            )
            print(f"  ✓ Saved: {comments_checkpoint.name}")
    
    # ========================================================================
    # STEP 4: Preprocess posts & comments
    # ========================================================================
    print(f"\n[Step 4/10] Preprocessing posts & comments...")
    
    content_type_suffix = "postsonly" if posts_only else "all"
    posts_prep_checkpoint = find_latest_file(interim_dir, f"posts_pattern1_preprocessed_{content_type_suffix}_*.csv")
    comments_prep_checkpoint = find_latest_file(interim_dir, f"comments_pattern1_preprocessed_{content_type_suffix}_*.csv") if not posts_only else None
    
    if use_checkpoints and not force_recompute and posts_prep_checkpoint and (posts_only or comments_prep_checkpoint):
        print(f"  ✓ Found checkpoints")
        posts_df = pd.read_csv(posts_prep_checkpoint, encoding='utf-8-sig', low_memory=False)
        if not posts_only:
            comments_df = pd.read_csv(comments_prep_checkpoint, encoding='utf-8-sig', low_memory=False)
    else:
        posts_df = preprocess_content(posts_df)
        if not posts_only:
            comments_df = preprocess_content(comments_df)
        
        save_with_timestamp(posts_df, interim_dir, f"posts_pattern1_preprocessed_{content_type_suffix}")
        if not posts_only:
            save_with_timestamp(comments_df, interim_dir, f"comments_pattern1_preprocessed_{content_type_suffix}")
        print(f"  ✓ Saved preprocessed data")
    
    if posts_only:
        print(f"  ✓ Posts: {len(posts_df):,}")
    else:
        print(f"  ✓ Posts: {len(posts_df):,}, Comments: {len(comments_df):,}")
    
    # ========================================================================
    # STEP 5: Add offsets from CD1
    # ========================================================================
    print(f"\n[Step 5/10] Adding offsets from CD1...")
    
    posts_offset_checkpoint = find_latest_file(interim_dir, f"posts_pattern1_with_offsets_{content_type_suffix}_*.csv")
    comments_offset_checkpoint = find_latest_file(interim_dir, f"comments_pattern1_with_offsets_{content_type_suffix}_*.csv") if not posts_only else None
    
    if use_checkpoints and not force_recompute and posts_offset_checkpoint and (posts_only or comments_offset_checkpoint):
        print(f"  ✓ Found checkpoints")
        posts_df = pd.read_csv(posts_offset_checkpoint, encoding='utf-8-sig', low_memory=False)
        if not posts_only:
            comments_df = pd.read_csv(comments_offset_checkpoint, encoding='utf-8-sig', low_memory=False)
    else:
        anchors = build_anchor_dict(users_df)
        posts_df = add_offsets_from_anchors(posts_df, anchors, author_col='author', timestamp_col='ts_utc')
        if not posts_only:
            comments_df = add_offsets_from_anchors(comments_df, anchors, author_col='author', timestamp_col='ts_utc')
        
        save_with_timestamp(posts_df, interim_dir, f"posts_pattern1_with_offsets_{content_type_suffix}")
        if not posts_only:
            save_with_timestamp(comments_df, interim_dir, f"comments_pattern1_with_offsets_{content_type_suffix}")
        print(f"  ✓ Saved with offsets")
    
    if posts_only:
        print(f"  ✓ Posts: {len(posts_df):,}")
    else:
        print(f"  ✓ Posts: {len(posts_df):,}, Comments: {len(comments_df):,}")
    
    # ========================================================================
    # STEP 6: Filter by anchor window
    # ========================================================================
    print(f"\n[Step 6/10] Filtering to ±{window_months} months of anchor...")
    print(f"  ⚠️  This is KEY: analyzing shorter windows reduces noise from long-term drift")
    
    posts_window_checkpoint = find_latest_file(interim_dir, f"posts_pattern1_{window_months}mo_{content_type_suffix}_*.csv")
    comments_window_checkpoint = find_latest_file(interim_dir, f"comments_pattern1_{window_months}mo_{content_type_suffix}_*.csv") if not posts_only else None
    
    if use_checkpoints and not force_recompute and posts_window_checkpoint and (posts_only or comments_window_checkpoint):
        print(f"  ✓ Found checkpoints")
        posts_df = pd.read_csv(posts_window_checkpoint, encoding='utf-8-sig', low_memory=False)
        if not posts_only:
            comments_df = pd.read_csv(comments_window_checkpoint, encoding='utf-8-sig', low_memory=False)
    else:
        posts_df = filter_posts_by_anchor_window(posts_df, users_df, window_months=window_months, user_col='author')
        if not posts_only:
            comments_df = filter_posts_by_anchor_window(comments_df, users_df, window_months=window_months, user_col='author')
        
        save_with_timestamp(posts_df, interim_dir, f"posts_pattern1_{window_months}mo_{content_type_suffix}")
        if not posts_only:
            save_with_timestamp(comments_df, interim_dir, f"comments_pattern1_{window_months}mo_{content_type_suffix}")
        print(f"  ✓ Saved filtered data")
    
    if posts_only:
        print(f"  ✓ Posts: {len(posts_df):,}")
    else:
        print(f"  ✓ Posts: {len(posts_df):,}, Comments: {len(comments_df):,}")
    
    # ========================================================================
    # STEP 7: Combine into timeline
    # ========================================================================
    if posts_only:
        print(f"\n[Step 7/10] Creating timeline (posts only)...")
    else:
        print(f"\n[Step 7/10] Combining posts + comments into timeline...")
    
    timeline_checkpoint = find_latest_file(interim_dir, f"timeline_pattern1_{window_months}mo_{content_type_suffix}_*.csv")
    
    if use_checkpoints and not force_recompute and timeline_checkpoint:
        print(f"  ✓ Found checkpoint: {timeline_checkpoint.name}")
        timeline_df = pd.read_csv(timeline_checkpoint, encoding='utf-8-sig', low_memory=False)
    else:
        posts_df['content_type'] = 'post'
        
        if posts_only:
            timeline_df = posts_df.copy()
        else:
            comments_df['content_type'] = 'comment'
            timeline_df = pd.concat([posts_df, comments_df], ignore_index=True)
        
        timeline_df = timeline_df.sort_values(['author', 'ts_utc'])
        
        timeline_checkpoint = save_with_timestamp(
            timeline_df, interim_dir, f"timeline_pattern1_{window_months}mo_{content_type_suffix}"
        )
        print(f"  ✓ Saved: {timeline_checkpoint.name}")
    
    print(f"  ✓ Timeline: {len(timeline_df):,} entries from {timeline_df['author'].nunique():,} users")
    if 'content_type' in timeline_df.columns:
        print(f"    Posts: {(timeline_df['content_type']=='post').sum():,}, Comments: {(timeline_df['content_type']=='comment').sum():,}")
    
    # ========================================================================
    # STEP 8: Compute sentiment & linguistic features
    # ========================================================================
    print(f"\n[Step 8/10] Computing sentiment & linguistic features...")
    print("  ⚠️  This may take 10-15 minutes...")
    
    features_checkpoint = find_latest_file(interim_dir, f"timeline_pattern1_{window_months}mo_{content_type_suffix}_with_features_*.csv")
    
    if use_checkpoints and not force_recompute and features_checkpoint:
        print(f"  ✓ Found checkpoint: {features_checkpoint.name}")
        timeline_df = pd.read_csv(features_checkpoint, encoding='utf-8-sig', low_memory=False)
    else:
        print("    Computing VADER sentiment (compound, positive, negative)...")
        timeline_df = compute_vader_sentiment(timeline_df, text_column='text')
        
        print("    Computing TextBlob sentiment (polarity, subjectivity, intensity)...")
        timeline_df = compute_textblob_sentiment(timeline_df, text_column='text')
        
        print("    Computing linguistic features (word_count, syntactic_complexity)...")
        timeline_df = compute_linguistic_features(timeline_df, text_column='text')
        
        features_checkpoint = save_with_timestamp(
            timeline_df, interim_dir, f"timeline_pattern1_{window_months}mo_{content_type_suffix}_with_features"
        )
        print(f"  ✓ Saved: {features_checkpoint.name}")
    
    # ========================================================================
    # STEP 9: Run periodicity detection
    # ========================================================================
    print(f"\n[Step 9/10] Running periodicity detection...")
    print(f"  ⚠️  This may take 10-15 minutes...")
    print(f"  Methods: {', '.join(analysis_methods)}")
    print(f"  Normalization: {normalization_method} (per-user)")
    
    methods_str = "_".join(analysis_methods)
    # periodicity_checkpoint = find_latest_file(
    #     interim_dir, 
    #     f"periodicity_results_{patterns_str}_{window_months}mo_{content_type_suffix}_{normalization_method}_{methods_str}_snr{snr_threshold}_*.csv"
    # )
    periodicity_checkpoint = None
    if use_checkpoints and not force_recompute and periodicity_checkpoint:
        print(f"  ✓ Found checkpoint: {periodicity_checkpoint.name}")
        results_df = pd.read_csv(periodicity_checkpoint, encoding='utf-8-sig')
    else:
        print(f"    Analyzing {len(base_features)} features with {normalization_method} normalization...")
        print(f"    Using methods: {', '.join(analysis_methods)}")
        
        # Run analysis with configured methods and normalization
        results_df = analyze_all_users_with_normalizations(
            timeline_df=timeline_df,
            base_features=base_features,
            normalizations=[normalization_method],
            user_col='author',
            period_min=24.0,
            period_max=35.0,
            filter_range=False,
            fap_threshold=fap_threshold,
            snr_threshold=snr_threshold,
            methods=analysis_methods,
        )
        
        periodicity_checkpoint = save_with_timestamp(
            results_df, interim_dir, 
            f"periodicity_results_{patterns_str}_{window_months}mo_{content_type_suffix}_{normalization_method}_{methods_str}_snr{snr_threshold}"
        )
        print(f"  ✓ Saved: {periodicity_checkpoint.name}")
    
    print(f"  ✓ {len(results_df):,} analyses for {results_df['user'].nunique():,} users")
    
    # ========================================================================
    # STEP 10: Visualize results
    # ========================================================================
    print(f"\n[Step 10/10] Visualizing cycle length distributions...")
    
    if len(results_df) == 0:
        print(f"  ⚠ No results to visualize!")
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = reports_dir / f"cycle_distributions_{patterns_str}_{normalization_method}_{methods_str}_{window_months}mo_{content_type_suffix}_snr{snr_threshold}_{timestamp}.png"
        
        plot_cycle_distributions(results_df, base_features, output_path)
        print(f"  ✓ Saved: {output_path.name}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults summary:")
    print(f"  Total analyses: {len(results_df):,}")
    print(f"  Users analyzed: {results_df['user'].nunique():,}")
    print(f"  Features analyzed: {len(results_df['feature'].unique())}")
    print(f"  Methods: {', '.join(results_df['method'].unique())}")
    print(f"  Normalization: {normalization_method}")
    
    print(f"\nCycle length statistics (all features):")
    print(f"  Mean: {results_df['period'].mean():.1f} days")
    print(f"  Median: {results_df['period'].median():.1f} days")
    print(f"  Std: {results_df['period'].std():.1f} days")
    
    print(f"\nBy feature:")
    for feature in results_df['feature'].unique():
        feature_periods = results_df[results_df['feature'] == feature]['period']
        print(f"  {feature}:")
        print(f"    n={len(feature_periods):,}, Mean: {feature_periods.mean():.1f}d, Median: {feature_periods.median():.1f}d")
        if len(feature_periods.mode()) > 0:
            print(f"    Mode: {feature_periods.mode()[0]:.0f}d")
    
    print(f"\nOutputs saved to:")
    print(f"  Interim: {interim_dir}")
    print(f"  Reports: {reports_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline for periodicity analysis (single or multiple patterns)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run with a run configuration file
  python scripts/07_full_pipeline_pattern1_clean.py --run-config configs/runs/pattern1_only.yaml
  
  # Run with multiple patterns
  python scripts/07_full_pipeline_pattern1_clean.py --patterns pattern_1 pattern_2 pattern_3
  
  # Run with posts only (faster, less memory)
  python scripts/07_full_pipeline_pattern1_clean.py --run-config configs/runs/all_cd_patterns.yaml --posts-only
  
  # Run with organized output folder
  python scripts/07_full_pipeline_pattern1_clean.py --output-subdir run_20251218
  
  # Recompute everything from scratch
  python scripts/07_full_pipeline_pattern1_clean.py --force-recompute
        """
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base config file")
    parser.add_argument("--run-config", type=str, default=None, help="Path to run-specific config file")
    parser.add_argument("--patterns", nargs="+", default=None, help="List of patterns to analyze (e.g., pattern_1 pattern_2)")
    parser.add_argument("--window-months", type=int, default=6, help="Months before/after anchor (default: 6)")
    parser.add_argument("--min-chars", type=int, default=150, help="Minimum text length (default: 150)")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint loading")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute all steps, ignore checkpoints")
    parser.add_argument("--posts-only", action="store_true", help="Use posts only, skip comments")
    parser.add_argument("--output-subdir", type=str, default=None, help="Subdirectory for this run's outputs (e.g., 'run_20251218')")
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        run_config_path=args.run_config,
        patterns=args.patterns,
        window_months=args.window_months,
        min_chars=args.min_chars,
        use_checkpoints=not args.no_checkpoints,
        force_recompute=args.force_recompute,
        posts_only=args.posts_only,
        output_subdir=args.output_subdir,
    )

