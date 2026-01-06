#!/usr/bin/env python3
"""Analyze which cycle phase PMDD users post in depression/suicide subreddits.

Uses processed timeline data with cycle phases already calculated.
Shows posting statistics by menstrual phase for PMDD users.
"""

from typing import Any


import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_1samp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io import find_latest_file, parse_jsonl_file
from src.visualization import create_adaptive_phases, assign_phase_to_day
from src.analysis import assign_consensus_period_by_majority

# Subreddit groups
PMDD_SUBREDDITS = {"PMDD", "PMDDxADHD", "PMDDSharing"}
DEPRESSION_SUBREDDITS = {"depression", "mentalhealth", "Anxiety", "socialanxiety", "AnxietyDepression"}
SUICIDE_SUBREDDITS = {"SuicideWatch"}


def main(split_luteal: bool = True, force_recompute_pmdd: bool = False):
    """Main analysis pipeline.
    
    Args:
        split_luteal: If True, split luteal phase into Early Luteal (7 days) and Late Luteal (7 days)
                      and analyze Late Luteal specifically. Default: True
        force_recompute_pmdd: If True, recompute PMDD users identification even if checkpoint exists. Default: False
    """
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    interim_dir = data_dir / "interim"
    raw_dir = data_dir / "raw"
    output_dir = interim_dir / "pmdd_phase_posting"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PMDD Users: Mental Health Posting by Cycle Phase")
    if split_luteal:
        print("Mode: Luteal phase split into Early (7d) and Late (7d) - analyzing Late Luteal")
    print("="*70)
    
    # ========================================================================
    # Step 1: Load processed timeline 
    # ========================================================================
    print("\n[Step 1/6] Loading processed timeline data...")
    
    timeline_dir = interim_dir / "cd_patterns_1to6_clean_run"
    timeline_file = find_latest_file(
        timeline_dir, 
        "timeline_pattern1_*.csv"
    )
    
    if not timeline_file:
        print("  ❌ Timeline file not found!")
        return
    
    print(f"  Loading: {timeline_file.name}")
    timeline_df = pd.read_csv(timeline_file, encoding='utf-8-sig', low_memory=False)
    print(f"  ✓ Loaded {len(timeline_df):,} posts from {timeline_df['author'].nunique():,} users")
    
    # Filter to ±3 months (±90 days)
    timeline_df = timeline_df[
        (timeline_df['offset_from_cd1'] >= -90) & 
        (timeline_df['offset_from_cd1'] <= 90)
    ].copy()
    print(f"  ✓ Filtered to ±3 months: {len(timeline_df):,} posts from {timeline_df['author'].nunique():,} users")
    
    # ========================================================================
    # Step 2: Load PMDD users list (with checkpoint)
    # ========================================================================
    print("\n[Step 2/6] Loading PMDD users...")
    
    # Load users database - pattern_1 + pattern_2
    users_db_path = data_dir / "processed" / "users_database_CD_20251215T171047.csv"
    users_db = pd.read_csv(users_db_path, encoding='utf-8-sig')
    users_db = users_db[users_db["pattern_category"].isin(["pattern_1", "pattern_2"])].copy()
    all_cd_users = set[Any](users_db["user"].astype(str).unique())
    print(f"  ✓ Loaded {len(all_cd_users):,} CD users from database (pattern_1 + pattern_2)")
    
    # Checkpoint for PMDD users identification
    checkpoint_dir = interim_dir / "pmdd_phase_posting"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pmdd_users_checkpoint = checkpoint_dir / "pmdd_users_pattern1_pattern2_checkpoint.csv"
    pmdd_posts_checkpoint = checkpoint_dir / "pmdd_users_posts_pattern1_pattern2_checkpoint.csv"
    
    if not force_recompute_pmdd and pmdd_users_checkpoint.exists() and pmdd_posts_checkpoint.exists():
        print(f"  ✓ Loading PMDD users from checkpoint: {pmdd_users_checkpoint.name}")
        pmdd_users_df = pd.read_csv(pmdd_users_checkpoint, encoding='utf-8-sig')
        pmdd_users = set(pmdd_users_df['user'].astype(str).unique())
        print(f"  ✓ Loaded {len(pmdd_users):,} PMDD users from checkpoint")
        
        print(f"  ✓ Loading PMDD posts from checkpoint: {pmdd_posts_checkpoint.name}")
        pmdd_posts_df = pd.read_csv(pmdd_posts_checkpoint, encoding='utf-8-sig')
        print(f"  ✓ Loaded {len(pmdd_posts_df):,} PMDD posts from {pmdd_posts_df['author'].nunique():,} users")
    else:
        if force_recompute_pmdd:
            print(f"  ⚠️ Force recompute flag set - scanning moon files...")
        else:
            print(f"  ⚠️ Checkpoint not found - scanning moon files...")
        
        # Scan moon1 (pattern_1) + moon2 (pattern_2) to find which users posted in PMDD subreddits
        post_files = [
            raw_dir / "moon1_all_post_2015_2025.filtered.tsv",
            raw_dir / "moon2_all_posts_2010_2025.tsv"
        ]
        
        pmdd_users = set()
        pmdd_posts_list = []
        
        for posts_file in post_files:
            print(f"  Scanning {posts_file.name} for PMDD posts...")
            
            for data in parse_jsonl_file(posts_file, progress_interval=2_000_000):
                author = data.get("author")
                subreddit = data.get("subreddit")
                
                if author in all_cd_users and subreddit in PMDD_SUBREDDITS:
                    pmdd_users.add(author)
                    # Store post data
                    pmdd_posts_list.append({
                        'author': author,
                        'id': data.get('id'),
                        'subreddit': subreddit,
                        'created_utc': data.get('created_utc'),
                        'text': data.get('selftext') or data.get('body', ''),
                    })
        
        print(f"  ✓ Found {len(pmdd_users):,} CD users who ever posted in PMDD subreddits (pattern_1 + pattern_2)")
        
        # Save checkpoint
        print(f"  Saving checkpoint...")
        pmdd_users_df = pd.DataFrame([{'user': u} for u in pmdd_users])
        pmdd_users_df.to_csv(pmdd_users_checkpoint, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved PMDD users checkpoint: {pmdd_users_checkpoint.name}")
        
        pmdd_posts_df = pd.DataFrame(pmdd_posts_list)
        pmdd_posts_df.to_csv(pmdd_posts_checkpoint, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved PMDD posts checkpoint: {pmdd_posts_checkpoint.name} ({len(pmdd_posts_df):,} posts)")
    
    # ========================================================================
    # Step 3: Identify mental health posts from PMDD users
    # ========================================================================
    print("\n[Step 3/6] Identifying mental health posts from PMDD users in depression/suicide subreddits...")
    
    mental_health_subs = DEPRESSION_SUBREDDITS | SUICIDE_SUBREDDITS
    
    # Filter to PMDD users in timeline first
    pmdd_users_in_timeline = set(timeline_df['author'].unique()) & pmdd_users
    print(f"  PMDD users in timeline: {len(pmdd_users_in_timeline):,}")
    
    # Filter timeline to PMDD users' posts in mental health subreddits
    mh_posts = timeline_df[
        (timeline_df['author'].isin(pmdd_users_in_timeline)) &
        (timeline_df['subreddit'].isin(mental_health_subs))
    ].copy()
    print(f"  ✓ Found {len(mh_posts):,} mental health posts from {mh_posts['author'].nunique():,} PMDD users")
    
    # ========================================================================
    # Step 4: Load FFT periodicity results and compute consensus periods
    # ========================================================================
    print("\n[Step 4/6] Loading FFT periodicity results and computing consensus periods...")
    
    # Look for files that contain fft_interpolation method (not just fft_interpolation_wide)
    # Files with both methods: "fft_interpolation_fft_interpolation_wide" 
    # Files with only wide: "fft_interpolation_wide" (no standalone fft_interpolation)
    all_files = list(timeline_dir.glob("periodicity_results_pattern_1_pattern_2_*snr3.0*.csv"))
    
    # Prefer files that have standalone "fft_interpolation" (files with both methods)
    # These files contain results for both fft_interpolation and fft_interpolation_wide
    preferred_files = [f for f in all_files if "fft_interpolation_fft_interpolation_wide" in f.name]
    
    if preferred_files:
        # Sort by modification time and take latest
        results_file = max(preferred_files, key=lambda f: f.stat().st_mtime)
    else:
        # No files with both methods, try any file (will check method inside)
        results_file = max(all_files, key=lambda f: f.stat().st_mtime) if all_files else None
    
    if not results_file:
        print("  ⚠️ Periodicity results not found! Using standard 29-day cycle.")
        user_periods = None
    else:
        print(f"  Loading: {results_file.name}")
        results_df = pd.read_csv(results_file, encoding='utf-8-sig')
        
        snr_threshold = 3.0
        
        print(f"  Total results in file: {len(results_df):,} rows")
        print(f"    Users: {results_df['user'].nunique():,}")
        print(f"    Features: {results_df['feature'].unique() if 'feature' in results_df.columns else 'N/A'}")
        print(f"    Methods: {results_df['method'].unique() if 'method' in results_df.columns else 'N/A'}")
        
        # Filter to SNR >= threshold first (for all features and methods)
        if 'peak_to_background' in results_df.columns:
            filtered_results = results_df[
                (results_df['period'].notna()) &
                (results_df['peak_to_background'] >= snr_threshold)
            ].copy()
            print(f"  After filtering by SNR >= {snr_threshold}: {len(filtered_results):,} results")
        else:
            # If no peak_to_background column, just filter out NaN periods
            filtered_results = results_df[results_df['period'].notna()].copy()
            print(f"  After filtering NaN periods: {len(filtered_results):,} results (no SNR filtering available)")
        
        if len(filtered_results) == 0:
            print("  ⚠️ No valid periodicity results after filtering!")
            print("     Using 29-day fallback for all.")
            user_periods = None
        else:
            # Use consensus period function to get most popular period per user across all features
            print(f"  Computing consensus periods across all features...")
            consensus_df = assign_consensus_period_by_majority(
                filtered_results,
                user_col='user',
                period_col='period',
                feature_col='feature'
            )
            
            print(f"  ✓ Consensus periods computed for {len(consensus_df):,} users")
            if len(consensus_df) > 0:
                # Filter to users with at least 5 features agreeing on consensus period
                min_features_agreeing = 5
                consensus_df_filtered = consensus_df[
                    consensus_df['n_features_agreeing'] >= min_features_agreeing
                ].copy()
                
                print(f"  After filtering (>= {min_features_agreeing} features agreeing): {len(consensus_df_filtered):,} users")
                print(f"    Dropped: {len(consensus_df) - len(consensus_df_filtered):,} users with < {min_features_agreeing} features agreeing")
                
                if len(consensus_df_filtered) > 0:
                    # Create user -> period mapping from filtered consensus results
                    user_periods = dict(zip(consensus_df_filtered['user'], consensus_df_filtered['consensus_period']))
                    
                    # Show statistics
                    print(f"    Mean consensus period: {np.mean(consensus_df_filtered['consensus_period']):.1f} days")
                    print(f"    Period range: {np.min(consensus_df_filtered['consensus_period']):.1f} - {np.max(consensus_df_filtered['consensus_period']):.1f} days")
                    print(f"    Mean features agreeing: {consensus_df_filtered['n_features_agreeing'].mean():.1f} features")
                    print(f"    Min features agreeing: {consensus_df_filtered['n_features_agreeing'].min()}")
                    print(f"    Max features agreeing: {consensus_df_filtered['n_features_agreeing'].max()}")
                else:
                    print(f"  ⚠️ No users with >= {min_features_agreeing} features agreeing!")
                    print("     Using 29-day fallback for all.")
                    user_periods = None
            else:
                user_periods = None
    
    # ========================================================================
    # Step 5: Match mental health posts to timeline & assign phases
    # ========================================================================
    print("\n[Step 5/6] Assigning cycle phases to mental health posts...")
    
    # Add subreddit category
    mh_posts['mh_category'] = mh_posts['subreddit'].map(
        lambda x: 'Depression' if x in DEPRESSION_SUBREDDITS else ('Suicide' if x in SUICIDE_SUBREDDITS else None)
    )
    
    # Assign phases using user-specific detected cycle lengths from FFT
    def assign_phase_for_post(row):
        user = row['author']
        day = row['offset_from_cd1']
        
        if pd.isna(day):
            return None
        
        # Use user's detected period, or fall back to 29 days (mean of 24-35)
        if user_periods and user in user_periods:
            period = user_periods[user]
        else:
            period = 29  # Mean cycle length as fallback
        
        phases = create_adaptive_phases(period, split_luteal=split_luteal)
        return assign_phase_to_day(day, phases)
    
    mh_posts['detected_period'] = mh_posts['author'].map(
        lambda u: user_periods.get(u, 29) if user_periods else 29
    )
    mh_posts['has_fft_period'] = mh_posts['author'].map(
        lambda u: u in user_periods if user_periods else False
    )
    mh_posts['phase'] = mh_posts.apply(assign_phase_for_post, axis=1)
    
    mh_posts = mh_posts[mh_posts['phase'].notna()].copy()
    
    # Show how many users had detected periods
    users_with_detected = mh_posts[mh_posts['has_fft_period']]['author'].nunique()
    users_with_fallback = mh_posts[~mh_posts['has_fft_period']]['author'].nunique()
    total_mh_users = mh_posts['author'].nunique()
    
    print(f"  ✓ Assigned phases to {len(mh_posts):,} posts from {total_mh_users} users")
    print(f"    - {users_with_detected} users with FFT-detected periods (SNR >= 3.0)")
    print(f"    - {users_with_fallback} users using 29-day fallback (no valid FFT period)")
    
    # ========================================================================
    # Step 6: Calculate statistics by phase
    # ========================================================================
    print("\n[Step 6/6] Calculating statistics by phase...")
    
    n_pmdd_users_all_patterns = len(pmdd_users)
    n_mh_posts_in_timeline = len(mh_posts)  # MH posts in timeline from PMDD users
    n_pmdd_users_posted_mh_window = mh_posts['author'].nunique()  # PMDD users with MH posts in timeline
    
    # Get pattern_1 + pattern_2 specific stats
    timeline_users = set(timeline_df['author'].unique())
    
    print(f"\n{'='*70}")
    print("FULL STATISTICS FUNNEL")
    print(f"{'='*70}")
    print(f"\nPATTERN_1 + PATTERN_2:")
    print(f"1. Pattern_1 + Pattern_2 CD users with cycle anchors: {len(all_cd_users):,}")
    print(f"2. ↓ Posted in PMDD subreddits: {n_pmdd_users_all_patterns:,} ({n_pmdd_users_all_patterns/len(all_cd_users)*100:.1f}%)")
    print(f"\nFILTERED TO PATTERN_1 + PATTERN_2 (have timeline+periodicity data):")
    print(f"3. Pattern_1 + Pattern_2 users in timeline: {len(timeline_users):,}")
    print(f"4. ↓ PMDD users in timeline: {len(pmdd_users_in_timeline):,} ({len(pmdd_users_in_timeline)/len(timeline_users)*100:.1f}% of timeline)")
    print(f"5. ↓ Posted in MH subs within timeline window: {n_pmdd_users_posted_mh_window} users, {n_mh_posts_in_timeline:,} posts")
    print(f"\nFFT period detection (consensus across all features, SNR >= 3.0):")
    print(f"   - Users with detected periods: {users_with_detected}")
    print(f"   - Users using 29-day fallback: {users_with_fallback}")
    print(f"   - Total users in analysis: {n_pmdd_users_posted_mh_window}")
    
    print(f"\n{'='*70}")
    print("PHASE ANALYSIS")
    print(f"{'='*70}")
    
    # Overall breakdown by category
    print(f"\nBy Subreddit Category (All Users):")
    print("-" * 70)
    category_counts = mh_posts.groupby('mh_category').agg({
        'id': 'count',
        'author': 'nunique'
    }).reset_index()
    category_counts.columns = ['Category', 'Posts', 'Unique Users']
    print(category_counts.to_string(index=False))
    
    if split_luteal:
        phase_order = ['Menstrual', 'Follicular', 'Ovulation', 'Early Luteal', 'Late Luteal']
    else:
        phase_order = ['Menstrual', 'Follicular', 'Ovulation', 'Luteal']
    
    # Function to calculate phase stats for a subset
    def calculate_phase_stats(df_subset, label):
        print(f"\n{label}:")
        print("-" * 70)
        
        phase_stats = []
        for phase in phase_order:
            phase_posts = df_subset[df_subset['phase'] == phase]
            n_posts = len(phase_posts)
            n_users = phase_posts['author'].nunique()
            
            depression_posts = phase_posts[phase_posts['mh_category'] == 'Depression']
            suicide_posts = phase_posts[phase_posts['mh_category'] == 'Suicide']
            
            # Count unique users for each category
            depression_users = depression_posts['author'].nunique()
            suicide_users = suicide_posts['author'].nunique()
            
            avg_sentiment = phase_posts['sentiment_positive'].mean() if 'sentiment_positive' in phase_posts.columns else None
            
            phase_stats.append({
                'Phase': phase,
                'Total Posts': n_posts,
                'Users': n_users,
                'Depression Posts': len(depression_posts),
                'Depression Users': depression_users,
                'Suicide Posts': len(suicide_posts),
                'Suicide Users': suicide_users,
                'Avg Positive Sentiment': f"{avg_sentiment:.3f}" if avg_sentiment is not None else "N/A"
            })
        
        phase_df = pd.DataFrame(phase_stats)
        total_posts = phase_df['Total Posts'].sum()
        phase_df['% of Total'] = (phase_df['Total Posts'] / total_posts * 100).round(1)
        
        print(phase_df.to_string(index=False))
        return phase_df
    
    # Calculate for users with FFT periods only (subset who have MH posts)
    mh_posts_fft = mh_posts[mh_posts['has_fft_period']].copy()
    phase_df_fft = calculate_phase_stats(
        mh_posts_fft, 
        f"By Menstrual Cycle Phase (FFT-detected periods only, n={users_with_detected} users with MH posts)"
    )
    
    # Calculate for all users (FFT + fallback)
    phase_df_all = calculate_phase_stats(
        mh_posts,
        f"By Menstrual Cycle Phase (All users: FFT + 29-day fallback, n={n_pmdd_users_posted_mh_window} users)"
    )
    
    # ========================================================================
    # Normalized Posting Rate Analysis (FFT-detected periods only)
    # ========================================================================
    print(f"\n{'='*70}")
    print("NORMALIZED POSTING RATE ANALYSIS (FFT-detected periods only)")
    print(f"{'='*70}")
    print("\nMethod: Calculate posts/day per phase, normalize within each user, then aggregate")
    
    if not user_periods or len(user_periods) == 0:
        print("  ⚠️ No users with FFT-detected periods available for normalized rate analysis")
    else:
        # Filter to only FFT users with MH posts
        mh_posts_fft_only = mh_posts[mh_posts['has_fft_period']].copy()
        
        if len(mh_posts_fft_only) == 0:
            print("  ⚠️ No mental health posts from users with FFT-detected periods")
        else:
            # Helper function to get phase length in days
            def get_phase_length(phase_name: str, cycle_length: float, split_luteal: bool) -> float:
                """Get phase length in days based on cycle length."""
                phases = create_adaptive_phases(cycle_length, split_luteal=split_luteal)
                if phase_name in phases:
                    start, end = phases[phase_name]
                    return float(end - start + 1)  # +1 because inclusive
                return 0.0
            
            # Helper function to analyze normalized rates for a subset of posts
            def analyze_normalized_rates(posts_subset, category_name):
                """Calculate normalized posting rates for a category of posts."""
                if len(posts_subset) == 0:
                    print(f"\n  ⚠️ No {category_name} posts available")
                    return None, None
                
                # Calculate posting rates per user per phase
                user_phase_rates = []
                
                fft_users = posts_subset['author'].unique()
                print(f"\n  Processing {len(fft_users):,} users with {category_name} posts...")
                
                for user in fft_users:
                    user_posts = posts_subset[posts_subset['author'] == user].copy()
                    cycle_length = user_periods[user]
                    
                    # Count posts per phase
                    phase_counts = user_posts['phase'].value_counts().to_dict()
                    
                    # Calculate rates (posts/day) for each phase
                    phase_rates = {}
                    for phase in phase_order:
                        count = phase_counts.get(phase, 0)
                        phase_len = get_phase_length(phase, cycle_length, split_luteal)
                        if phase_len > 0:
                            rate = count / phase_len
                        else:
                            rate = 0.0
                        phase_rates[phase] = rate
                    
                    # Store rates for this user
                    user_phase_rates.append({
                        'user': user,
                        'cycle_length': cycle_length,
                        **phase_rates
                    })
                
                user_rates_df = pd.DataFrame(user_phase_rates)
                
                if len(user_rates_df) == 0:
                    return None, None
                
                # Normalize rates within each user (z-score normalization)
                print(f"  Normalizing rates within each user (z-score)...")
                normalized_rates = []
                
                for phase in phase_order:
                    if phase in user_rates_df.columns:
                        rates = user_rates_df[phase].values
                        # Z-score: (rate - mean) / std
                        user_means = user_rates_df[phase_order].mean(axis=1).values
                        user_stds = user_rates_df[phase_order].std(axis=1).values
                        # Handle division by zero (if all phases have same rate for a user)
                        user_stds = np.where(user_stds < 1e-10, 1.0, user_stds)
                        normalized = (rates - user_means) / user_stds
                        normalized_rates.append(pd.Series(normalized, name=phase))
                
                normalized_df = pd.concat(normalized_rates, axis=1)
                normalized_df['user'] = user_rates_df['user']
                
                # Aggregate: mean normalized rate per phase
                print(f"  Aggregating normalized rates across {len(normalized_df):,} users...")
                agg_stats = []
                
                for phase in phase_order:
                    if phase in normalized_df.columns:
                        phase_normalized = normalized_df[phase].dropna()
                        if len(phase_normalized) > 0:
                            mean_norm_rate = phase_normalized.mean()
                            std_norm_rate = phase_normalized.std()
                            sem_norm_rate = std_norm_rate / np.sqrt(len(phase_normalized))
                            n_users = len(phase_normalized)
                            
                            agg_stats.append({
                                'Phase': phase,
                                'Mean Normalized Rate': mean_norm_rate,
                                'SEM': sem_norm_rate,
                                'Std Dev': std_norm_rate,
                                'N Users': n_users
                            })
                
                agg_df = pd.DataFrame(agg_stats)
                
                print(f"\n  {category_name.upper()} POSTS:")
                print(f"  {'Phase':<20} {'Mean Norm Rate':>18} {'SEM':>12} {'N Users':>10}")
                print("  " + "-" * 68)
                for _, row in agg_df.iterrows():
                    print(f"  {row['Phase']:<20} {row['Mean Normalized Rate']:>18.3f} {row['SEM']:>12.3f} {int(row['N Users']):>10}")
                
                # Statistical test: Is Follicular significantly higher?
                target_phase = 'Follicular'
                if target_phase in normalized_df.columns:
                    target_normalized = normalized_df[target_phase].dropna()
                    
                    if len(target_normalized) > 1:
                        # One-sample t-test: Is mean significantly different from 0?
                        t_stat, p_value = ttest_1samp(target_normalized, 0.0)
                        
                        print(f"\n  Statistical test for {target_phase} ({category_name}):")
                        print(f"    Mean normalized rate: {target_normalized.mean():.3f}")
                        print(f"    One-sample t-test (H0: mean = 0):")
                        print(f"      t-statistic: {t_stat:.3f}")
                        print(f"      p-value: {p_value:.6f}")
                        
                        if p_value < 0.05:
                            if target_normalized.mean() > 0:
                                print(f"    ✅ SIGNIFICANT (p < 0.05): Users post MORE {category_name} posts in {target_phase} relative to their own average")
                            else:
                                print(f"    ✅ SIGNIFICANT (p < 0.05): Users post LESS {category_name} posts in {target_phase} relative to their own average")
                        else:
                            print(f"    ❌ NOT SIGNIFICANT (p >= 0.05): No significant difference from user's own average")
                
                return normalized_df, agg_df
            
            # Analyze Depression posts
            depression_posts = mh_posts_fft_only[mh_posts_fft_only['mh_category'] == 'Depression'].copy()
            norm_df_dep, agg_df_dep = analyze_normalized_rates(depression_posts, "Depression")
            
            # Analyze Suicide posts
            suicide_posts = mh_posts_fft_only[mh_posts_fft_only['mh_category'] == 'Suicide'].copy()
            norm_df_sui, agg_df_sui = analyze_normalized_rates(suicide_posts, "Suicide")
            
            # Save normalized rates
            if norm_df_dep is not None or norm_df_sui is not None:
                print(f"\n[Saving Normalized Rates - PMDD Users]")
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                
                if norm_df_dep is not None:
                    normalized_rates_file = output_dir / f"normalized_posting_rates_depression_pmdd_{timestamp}.csv"
                    norm_df_dep.to_csv(normalized_rates_file, index=False, encoding='utf-8-sig')
                    print(f"  ✓ Saved user-level normalized rates (Depression, PMDD): {normalized_rates_file.name}")
                    
                    agg_rates_file = output_dir / f"aggregated_normalized_rates_depression_pmdd_{timestamp}.csv"
                    agg_df_dep.to_csv(agg_rates_file, index=False, encoding='utf-8-sig')
                    print(f"  ✓ Saved aggregated normalized rates (Depression, PMDD): {agg_rates_file.name}")
                
                if norm_df_sui is not None:
                    normalized_rates_file = output_dir / f"normalized_posting_rates_suicide_pmdd_{timestamp}.csv"
                    norm_df_sui.to_csv(normalized_rates_file, index=False, encoding='utf-8-sig')
                    print(f"  ✓ Saved user-level normalized rates (Suicide, PMDD): {normalized_rates_file.name}")
                    
                    agg_rates_file = output_dir / f"aggregated_normalized_rates_suicide_pmdd_{timestamp}.csv"
                    agg_df_sui.to_csv(agg_rates_file, index=False, encoding='utf-8-sig')
                    print(f"  ✓ Saved aggregated normalized rates (Suicide, PMDD): {agg_rates_file.name}")
    
    # ========================================================================
    # Normalized Posting Rate Analysis - CONTROL GROUP (FFT-detected periods only)
    # ========================================================================
    print(f"\n{'='*70}")
    print("NORMALIZED POSTING RATE ANALYSIS - CONTROL GROUP (FFT-detected periods only)")
    print(f"{'='*70}")
    print("\nMethod: Calculate posts/day per phase, normalize within each user, then aggregate")
    print("Control group: Users with cycle anchors who NEVER posted in PMDD subreddits")
    
    if not user_periods or len(user_periods) == 0:
        print("  ⚠️ No users with FFT-detected periods available for control group analysis")
    else:
        # Define control users: users with cycle anchors who never posted in PMDD
        timeline_users_all = set(timeline_df['author'].unique())
        control_users = timeline_users_all - pmdd_users_in_timeline
        control_users_with_fft = control_users & set(user_periods.keys())
        
        print(f"  Control users in timeline: {len(control_users):,}")
        print(f"  Control users with FFT-detected periods: {len(control_users_with_fft):,}")
        
        if len(control_users_with_fft) == 0:
            print("  ⚠️ No control users with FFT-detected periods available")
        else:
            # Get mental health posts from control users
            control_mh_posts = timeline_df[
                (timeline_df['author'].isin(control_users_with_fft)) &
                (timeline_df['subreddit'].isin(mental_health_subs))
            ].copy()
            
            if len(control_mh_posts) == 0:
                print("  ⚠️ No mental health posts from control users with FFT-detected periods")
            else:
                # Add subreddit category
                control_mh_posts['mh_category'] = control_mh_posts['subreddit'].map(
                    lambda x: 'Depression' if x in DEPRESSION_SUBREDDITS else ('Suicide' if x in SUICIDE_SUBREDDITS else None)
                )
                
                # Assign phases to control users' posts
                def assign_phase_for_control_post(row):
                    user = row['author']
                    day = row['offset_from_cd1']
                    
                    if pd.isna(day):
                        return None
                    
                    if user in user_periods:
                        period = user_periods[user]
                    else:
                        return None  # Shouldn't happen since we filtered
                    
                    phases = create_adaptive_phases(period, split_luteal=split_luteal)
                    return assign_phase_to_day(day, phases)
                
                control_mh_posts['phase'] = control_mh_posts.apply(assign_phase_for_control_post, axis=1)
                control_mh_posts = control_mh_posts[control_mh_posts['phase'].notna()].copy()
                
                print(f"  ✓ Found {len(control_mh_posts):,} mental health posts from {control_mh_posts['author'].nunique():,} control users")
                
                # Analyze Depression posts (Control)
                control_depression_posts = control_mh_posts[control_mh_posts['mh_category'] == 'Depression'].copy()
                control_norm_df_dep, control_agg_df_dep = analyze_normalized_rates(control_depression_posts, "Depression (Control)")
                
                # Analyze Suicide posts (Control)
                control_suicide_posts = control_mh_posts[control_mh_posts['mh_category'] == 'Suicide'].copy()
                control_norm_df_sui, control_agg_df_sui = analyze_normalized_rates(control_suicide_posts, "Suicide (Control)")
                
                # Save normalized rates for control group
                if control_norm_df_dep is not None or control_norm_df_sui is not None:
                    print(f"\n[Saving Normalized Rates - Control Users]")
                    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                    
                    if control_norm_df_dep is not None:
                        normalized_rates_file = output_dir / f"normalized_posting_rates_depression_control_{timestamp}.csv"
                        control_norm_df_dep.to_csv(normalized_rates_file, index=False, encoding='utf-8-sig')
                        print(f"  ✓ Saved user-level normalized rates (Depression, Control): {normalized_rates_file.name}")
                        
                        agg_rates_file = output_dir / f"aggregated_normalized_rates_depression_control_{timestamp}.csv"
                        control_agg_df_dep.to_csv(agg_rates_file, index=False, encoding='utf-8-sig')
                        print(f"  ✓ Saved aggregated normalized rates (Depression, Control): {agg_rates_file.name}")
                    
                    if control_norm_df_sui is not None:
                        normalized_rates_file = output_dir / f"normalized_posting_rates_suicide_control_{timestamp}.csv"
                        control_norm_df_sui.to_csv(normalized_rates_file, index=False, encoding='utf-8-sig')
                        print(f"  ✓ Saved user-level normalized rates (Suicide, Control): {normalized_rates_file.name}")
                        
                        agg_rates_file = output_dir / f"aggregated_normalized_rates_suicide_control_{timestamp}.csv"
                        control_agg_df_sui.to_csv(agg_rates_file, index=False, encoding='utf-8-sig')
                        print(f"  ✓ Saved aggregated normalized rates (Suicide, Control): {agg_rates_file.name}")
    
    # ========================================================================
    # Chi-square test (2x2): Are MH posts more likely in Follicular phase?
    # ========================================================================
    target_phase = 'Follicular'
    print(f"\n{'='*70}")
    print(f"CHI-SQUARE TEST (2x2): Mental Health Posts vs {target_phase} Phase")
    print(f"{'='*70}")
    
    # Filter to only users with FFT-detected periods
    if not user_periods or len(user_periods) == 0:
        print("  ⚠️ No users with FFT-detected periods available for chi-square test")
        print("  Skipping chi-square test...")
    else:
        users_with_fft_periods = set(user_periods.keys())
        pmdd_users_with_fft = pmdd_users_in_timeline & users_with_fft_periods
        print(f"  Filtering to {len(pmdd_users_with_fft):,} PMDD users with FFT-detected periods")
        print(f"  Note: Includes all FFT users (even without MH posts) to enable MH vs non-MH comparison")
        print(f"       Phase stats above used {users_with_detected} users (only those with MH posts)")
        
        # Get all posts from PMDD users with FFT periods (not just MH posts) and assign phases
        pmdd_posts_all = timeline_df[timeline_df['author'].isin(pmdd_users_with_fft)].copy()
        
        # Assign phases to all posts (using FFT-detected periods only)
        def assign_phase_for_post_fft(row):
            user = row['author']
            day = row['offset_from_cd1']
            
            if pd.isna(day):
                return None
            
            # All users in this dataset have FFT periods
            period = user_periods[user]
            phases = create_adaptive_phases(period, split_luteal=split_luteal)
            return assign_phase_to_day(day, phases)
        
        pmdd_posts_all['phase'] = pmdd_posts_all.apply(assign_phase_for_post_fft, axis=1)
        pmdd_posts_all = pmdd_posts_all[pmdd_posts_all['phase'].notna()].copy()
        
        # Mark which posts are mental health posts
        mental_health_subs = DEPRESSION_SUBREDDITS | SUICIDE_SUBREDDITS
        pmdd_posts_all['is_mh_post'] = pmdd_posts_all['subreddit'].isin(mental_health_subs)
        pmdd_posts_all['is_target_phase'] = pmdd_posts_all['phase'] == target_phase
        
        # Build 2x2 contingency table
        contingency_table = pd.crosstab(
            pmdd_posts_all['is_mh_post'], 
            pmdd_posts_all['is_target_phase'],
            margins=True
        )
        
        print(f"\n2x2 Contingency Table (FFT-detected periods only):")
        print(f"                     Is {target_phase}?")
        print("                 No          Yes        Total")
        print(f"Is MH Post?")
        print(f"  No          {contingency_table.loc[False, False]:6,}    {contingency_table.loc[False, True]:6,}    {contingency_table.loc[False, 'All']:6,}")
        print(f"  Yes         {contingency_table.loc[True, False]:6,}    {contingency_table.loc[True, True]:6,}    {contingency_table.loc[True, 'All']:6,}")
        print(f"  Total       {contingency_table.loc['All', False]:6,}    {contingency_table.loc['All', True]:6,}    {contingency_table.loc['All', 'All']:6,}")
        
        # Calculate percentages
        pct_mh_in_target = contingency_table.loc[True, True] / contingency_table.loc[True, 'All'] * 100
        pct_mh_in_non_target = contingency_table.loc[True, False] / contingency_table.loc[True, 'All'] * 100
        pct_non_mh_in_target = contingency_table.loc[False, True] / contingency_table.loc[False, 'All'] * 100
        pct_non_mh_in_non_target = contingency_table.loc[False, False] / contingency_table.loc[False, 'All'] * 100
        
        print(f"\nPercentages:")
        print(f"  MH posts in {target_phase}: {pct_mh_in_target:.1f}%")
        print(f"  MH posts in Non-{target_phase}: {pct_mh_in_non_target:.1f}%")
        print(f"  Non-MH posts in {target_phase}: {pct_non_mh_in_target:.1f}%")
        print(f"  Non-MH posts in Non-{target_phase}: {pct_non_mh_in_non_target:.1f}%")
        
        # Calculate theoretical phase proportions based on average cycle length of FFT users
        avg_period = np.mean(list(user_periods.values()))
        menstrual_len = 4
        ovulation_len = 3
        luteal_len = 14
        follicular_len = avg_period - menstrual_len - ovulation_len - luteal_len
        
        theoretical_target_pct = follicular_len / avg_period * 100
        if split_luteal:
            late_luteal_len = 7
            early_luteal_len = 7
            theoretical_non_target_pct = (menstrual_len + ovulation_len + luteal_len) / avg_period * 100
            
            print(f"\nTheoretical phase proportions (based on avg cycle length {avg_period:.1f} days):")
            print(f"  Menstrual: {menstrual_len} days ({menstrual_len/avg_period*100:.1f}%)")
            print(f"  Follicular: {follicular_len:.1f} days ({theoretical_target_pct:.1f}%)")
            print(f"  Ovulation: {ovulation_len} days ({ovulation_len/avg_period*100:.1f}%)")
            print(f"  Early Luteal: {early_luteal_len} days ({early_luteal_len/avg_period*100:.1f}%)")
            print(f"  Late Luteal: {late_luteal_len} days ({late_luteal_len/avg_period*100:.1f}%)")
            print(f"  Non-Follicular (M+O+L): {theoretical_non_target_pct:.1f}%")
        else:
            theoretical_non_target_pct = (menstrual_len + ovulation_len + luteal_len) / avg_period * 100
            
            print(f"\nTheoretical phase proportions (based on avg cycle length {avg_period:.1f} days):")
            print(f"  Menstrual: {menstrual_len} days ({menstrual_len/avg_period*100:.1f}%)")
            print(f"  Follicular: {follicular_len:.1f} days ({theoretical_target_pct:.1f}%)")
            print(f"  Ovulation: {ovulation_len} days ({ovulation_len/avg_period*100:.1f}%)")
            print(f"  Luteal: {luteal_len} days ({luteal_len/avg_period*100:.1f}%)")
            print(f"  Non-Follicular (M+O+L): {theoretical_non_target_pct:.1f}%")
        
        # Observed phase proportions
        obs_target_pct = (contingency_table.loc['All', True] / contingency_table.loc['All', 'All']) * 100
        obs_non_target_pct = (contingency_table.loc['All', False] / contingency_table.loc['All', 'All']) * 100
        
        print(f"\nObserved phase distribution in posts:")
        print(f"  {target_phase}: {obs_target_pct:.1f}% of all posts")
        print(f"  Non-{target_phase}: {obs_non_target_pct:.1f}% of all posts")
        print(f"  Note: Chi-square test uses observed proportions in expected value calculation")
        
        # Run chi-square test
        contingency_2x2 = pd.crosstab(pmdd_posts_all['is_mh_post'], pmdd_posts_all['is_target_phase'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_2x2)
        
        print(f"\nChi-square test results:")
        print(f"  Expected values (based on observed marginals):")
        print(f"    Non-MH, Non-{target_phase}: {expected[0,0]:.1f}")
        print(f"    Non-MH, {target_phase}: {expected[0,1]:.1f}")
        print(f"    MH, Non-{target_phase}: {expected[1,0]:.1f}")
        print(f"    MH, {target_phase}: {expected[1,1]:.1f}")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  ✅ SIGNIFICANT (p < 0.05): Mental health posts are significantly associated with {target_phase}")
            if pct_mh_in_target > pct_non_mh_in_target:
                print(f"     → MH posts are MORE likely in {target_phase} than non-MH posts")
                print(f"     → This suggests biological factors, not just phase length")
            else:
                print(f"     → MH posts are LESS likely in {target_phase} than non-MH posts")
        else:
            print(f"  ❌ NOT SIGNIFICANT (p >= 0.05): No significant association")
            print(f"     → The concentration in {target_phase} may be due to phase length, not biological factors")
            print(f"     → MH posts follow the same distribution as non-MH posts across phases")
    
    # Additional analysis: Critical symptom window (last 4 days luteal + first 2 days menstrual)
    # print(f"\nCritical Symptom Window Analysis (All users):")
    # print("-" * 70)
    
    # def assign_symptom_window(row):
    #     """Assign posts to critical symptom window vs other days."""
    #     day = row['offset_from_cd1']
    #     period = row['detected_period']
        
    #     if pd.isna(day) or pd.isna(period):
    #         return None
        
    #     # Normalize day to cycle
    #     day_normalized = day % period
        
    #     # Critical window: last 4 days of luteal + first 2 days of menstrual
    #     # Luteal ends at period-1, so last 4 days are: period-4, period-3, period-2, period-1
    #     # Menstrual is days 0-1 (first 2 days)
        
    #     luteal_end = int(period - 1)
    #     last_4_luteal_start = max(14, luteal_end - 3)  # Luteal starts around day 14
        
    #     if (last_4_luteal_start <= day_normalized <= luteal_end) or (0 <= day_normalized <= 1):
    #         return "Critical Window (Late Luteal + Early Menstrual)"
    #     else:
    #         return "Other Days"
    
    # mh_posts['symptom_window'] = mh_posts.apply(assign_symptom_window, axis=1)
    
    # symptom_stats = []
    # for window in ["Critical Window (Late Luteal + Early Menstrual)", "Other Days"]:
    #     window_posts = mh_posts[mh_posts['symptom_window'] == window]
    #     n_posts = len(window_posts)
    #     n_users = window_posts['author'].nunique()
        
    #     depression_posts = window_posts[window_posts['mh_category'] == 'Depression']
    #     suicide_posts = window_posts[window_posts['mh_category'] == 'Suicide']
        
    #     avg_sentiment = window_posts['textblob_polarity'].mean() if 'textblob_polarity' in window_posts.columns else None
        
    #     symptom_stats.append({
    #         'Window': window,
    #         'Total Posts': n_posts,
    #         'Users': n_users,
    #         'Depression': len(depression_posts),
    #         'Suicide': len(suicide_posts),
    #         'Avg Sentiment': f"{avg_sentiment:.3f}" if avg_sentiment is not None else "N/A"
    #     })
    
    # symptom_df = pd.DataFrame(symptom_stats)
    # total_posts = symptom_df['Total Posts'].sum()
    # symptom_df['% of Total'] = (symptom_df['Total Posts'] / total_posts * 100).round(1)
    
    # print(symptom_df.to_string(index=False))
    
    # ========================================================================
    # Save results
    # ========================================================================
    print(f"\n[Saving Results]")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    # Save detailed post-level data
    posts_out = output_dir / f"mh_posts_by_phase_{timestamp}.csv"
    save_cols = ['author', 'id', 'subreddit', 'mh_category', 'offset_from_cd1', 
                 'detected_period', 'has_fft_period', 'phase', 'created_utc']
    if 'text' in mh_posts.columns:
        save_cols.append('text')
    if 'sentiment_positive' in mh_posts.columns:
        save_cols.append('sentiment_positive')
    mh_posts[save_cols].to_csv(posts_out, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved post-level data: {posts_out.name}")
    
    # Save phase statistics (both FFT-only and combined)
    stats_fft_out = output_dir / f"phase_statistics_fft_only_{timestamp}.csv"
    phase_df_fft.to_csv(stats_fft_out, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved phase statistics (FFT only): {stats_fft_out.name}")
    
    stats_all_out = output_dir / f"phase_statistics_all_{timestamp}.csv"
    phase_df_all.to_csv(stats_all_out, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved phase statistics (all users): {stats_all_out.name}")
    
    # symptom_window_out = output_dir / f"symptom_window_statistics_{timestamp}.csv"
    # symptom_df.to_csv(symptom_window_out, index=False, encoding='utf-8-sig')
    # print(f"  ✓ Saved symptom window statistics: {symptom_window_out.name}")
    
    print(f"\n{'='*70}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze PMDD users' mental health posting by cycle phase")
    parser.add_argument(
        "--force-recompute-pmdd",
        action="store_true",
        help="Force recomputation of PMDD users (skip checkpoint)"
    )
    parser.add_argument(
        "--no-split-luteal",
        action="store_true",
        help="Don't split luteal phase into Early/Late (use single Luteal phase)"
    )
    
    args = parser.parse_args()
    
    main(
        split_luteal=not args.no_split_luteal,
        force_recompute_pmdd=args.force_recompute_pmdd
    )

