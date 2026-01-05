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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io import find_latest_file, parse_jsonl_file
from src.visualization import create_adaptive_phases, assign_phase_to_day

# Subreddit groups
PMDD_SUBREDDITS = {"PMDD", "PMDDxADHD", "PMDDSharing"}
DEPRESSION_SUBREDDITS = {"depression", "mentalhealth", "Anxiety", "socialanxiety", "AnxietyDepression"}
SUICIDE_SUBREDDITS = {"SuicideWatch"}


def main():
    """Main analysis pipeline."""
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    interim_dir = data_dir / "interim"
    raw_dir = data_dir / "raw"
    output_dir = interim_dir / "pmdd_phase_posting"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PMDD Users: Mental Health Posting by Cycle Phase")
    print("="*70)
    
    # ========================================================================
    # Step 1: Load processed timeline 
    # ========================================================================
    print("\n[Step 1/5] Loading processed timeline data...")
    
    timeline_dir = interim_dir / "test_pipeline_verify_28.12"
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
    
    # # Filter to ±3 months (±90 days) as requested
    # timeline_df = timeline_df[
    #     (timeline_df['offset_from_cd1'] >= -90) & 
    #     (timeline_df['offset_from_cd1'] <= 90)
    # ].copy()
    # print(f"  ✓ Filtered to ±3 months: {len(timeline_df):,} posts")
    
    # ========================================================================
    # Step 2: Load PMDD users list
    # ========================================================================
    print("\n[Step 2/5] Loading PMDD users...")
    
    # Load users database - pattern_1 only
    users_db_path = data_dir / "processed" / "users_database_CD_20251215T171047.csv"
    users_db = pd.read_csv(users_db_path, encoding='utf-8-sig')
    users_db = users_db[users_db["pattern_category"] == "pattern_1"].copy()
    all_cd_users = set[Any](users_db["user"].astype(str).unique())
    print(f"  ✓ Loaded {len(all_cd_users):,} CD users from database (pattern_1 only)")
    
    # Scan moon1 + moon2 to find which users posted in PMDD subreddits
    # posts_files = [
    #     raw_dir / "moon1_all_post_2015_2025.filtered.tsv",
    #     raw_dir / "moon2_all_posts_2010_2025.tsv"
    # ]
    post_files = [raw_dir / "moon1_all_post_2015_2025.filtered.tsv"]
    
    pmdd_users_all = set()
    for posts_file in post_files:
        print(f"  Scanning {posts_file.name} for PMDD posts...")
        
        for data in parse_jsonl_file(posts_file, progress_interval=2_000_000):
            author = data.get("author")
            subreddit = data.get("subreddit")
            
            if author in all_cd_users and subreddit in PMDD_SUBREDDITS:
                pmdd_users_all.add(author)
    
    print(f"  ✓ Found {len(pmdd_users_all):,} CD users who posted in PMDD subreddits (pattern_1 only)")
    
    # Filter to those in pattern_1 timeline for phase analysis
    pmdd_users_in_timeline = set(timeline_df['author'].unique()) & pmdd_users_all
    print(f"  ✓ {len(pmdd_users_in_timeline):,} of them are in pattern_1 timeline (for phase analysis)")
    
    # Filter timeline to PMDD users
    timeline_pmdd = timeline_df[timeline_df['author'].isin(pmdd_users_in_timeline)].copy()
    print(f"  ✓ Timeline filtered to PMDD users: {len(timeline_pmdd):,} posts")
    
    # ========================================================================
    # Step 3: Identify mental health posts
    # ========================================================================
    print("\n[Step 3/5] Identifying mental health posts in depression/suicide subreddits...")
    
    mental_health_subs = DEPRESSION_SUBREDDITS | SUICIDE_SUBREDDITS
    post_subreddit_map = {}
    
    for posts_file in post_files:
        print(f"  Scanning {posts_file.name} for mental health posts...")
        
        for data in parse_jsonl_file(posts_file, progress_interval=2_000_000):
            author = data.get("author")
            post_id = data.get("id")
            subreddit = data.get("subreddit")
            
            if author in pmdd_users_all and subreddit in mental_health_subs:
                post_subreddit_map[post_id] = {
                    'subreddit': subreddit,
                    'category': 'Depression' if subreddit in DEPRESSION_SUBREDDITS else 'Suicide'
                }
    
    print(f"  ✓ Found {len(post_subreddit_map):,} mental health posts from PMDD users (all time, pattern_1 only)")
    
    # ========================================================================
    # Step 4: Load FFT periodicity results for detected cycle lengths
    # ========================================================================
    print("\n[Step 4/6] Loading FFT periodicity results (sentiment negative)...")
    
    results_file = find_latest_file(
        timeline_dir,
        "periodicity_results_pattern_1_*.csv"
    )
    
    if not results_file:
        print("  ⚠️ Periodicity results not found! Using standard 28-day cycle.")
        user_periods = None
    else:
        print(f"  Loading: {results_file.name}")
        results_df = pd.read_csv(results_file, encoding='utf-8-sig')
        
        snr_threshold = 3.0
        
        # Filter to sentiment negatie feature + FFT interpolation method + SNR threshold
        fft_results = results_df[
            (results_df['feature'] == 'sentiment_negative') &
            (results_df['method'] == 'fft_interpolation')
        ].copy()
        
        if len(fft_results) == 0:
            print("  ⚠️ No sentiment negative results found! Using 29-day fallback for all.")
            user_periods = None
        else:
            print(f"  Before filtering: {len(fft_results):,} users (pattern_1)")
            
            # IMPORTANT: Only use users who are in the pattern_1 timeline we're analyzing
            timeline_users = set(timeline_df['author'].unique())
            fft_results = fft_results[fft_results['user'].isin(timeline_users)]
            print(f"  After filtering to timeline users: {len(fft_results):,} users")
            
            # Filter to SNR >= threshold
            fft_results = fft_results[
                (fft_results['period'].notna()) &
                (fft_results['peak_to_background'] >= snr_threshold)
            ]
            
            print(f"  After SNR >= {snr_threshold}: {len(fft_results):,} users")
            
            # Get user -> period mapping
            user_periods = dict(zip(fft_results['user'], fft_results['period']))
            print(f"  ✓ Loaded sentiment negative periods for {len(user_periods):,} users (passed all filters)")
            if len(user_periods) > 0:
                print(f"    Mean detected period: {np.mean(list(user_periods.values())):.1f} days")
                print(f"    Period range: {np.min(list(user_periods.values())):.1f} - {np.max(list(user_periods.values())):.1f} days")
    
    # ========================================================================
    # Step 5: Match mental health posts to timeline & assign phases
    # ========================================================================
    print("\n[Step 5/6] Assigning cycle phases to mental health posts...")
    
    # Filter timeline to only mental health posts
    timeline_pmdd['post_id'] = timeline_pmdd['id']
    mh_posts = timeline_pmdd[timeline_pmdd['post_id'].isin(post_subreddit_map.keys())].copy()
    
    print(f"  ✓ Matched {len(mh_posts):,} mental health posts in timeline")
    
    if len(mh_posts) == 0:
        print("  ⚠️ No mental health posts found in timeline window!")
        return
    
    # Add subreddit category
    mh_posts['mh_category'] = mh_posts['post_id'].map(
        lambda x: post_subreddit_map[x]['category'] if x in post_subreddit_map else None
    )
    mh_posts['subreddit'] = mh_posts['post_id'].map(
        lambda x: post_subreddit_map[x]['subreddit'] if x in post_subreddit_map else None
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
        
        phases = create_adaptive_phases(period)
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
    print(f"    - {users_with_detected} users with FFT-detected periods (SNR >= 2.0)")
    print(f"    - {users_with_fallback} users using 29-day fallback (no valid FFT period)")
    
    # ========================================================================
    # Step 6: Calculate statistics by phase
    # ========================================================================
    print("\n[Step 6/6] Calculating statistics by phase...")
    
    n_pmdd_users_all_patterns = len(pmdd_users_all)
    n_pmdd_users_in_timeline = len(pmdd_users_in_timeline)
    n_mh_posts_alltime = len(post_subreddit_map)  # Total MH posts ever
    n_pmdd_users_posted_mh_window = mh_posts['author'].nunique()  # In ±3mo window
    
    # Get pattern_1 + pattern_2 specific stats
    timeline_users = set(timeline_df['author'].unique())
    
    print(f"\n{'='*70}")
    print("FULL STATISTICS FUNNEL")
    print(f"{'='*70}")
    print(f"\nPATTERN_1 ONLY:")
    print(f"1. Pattern_1 CD users with cycle anchors: {len(all_cd_users):,}")
    print(f"2. ↓ Posted in PMDD subreddits: {n_pmdd_users_all_patterns:,} ({n_pmdd_users_all_patterns/len(all_cd_users)*100:.1f}%)")
    print(f"3. ↓ Posted in MH subs (depression/suicide) - all time: {n_mh_posts_alltime:,} posts")
    print(f"\nFILTERED TO PATTERN_1 (have timeline+periodicity data):")
    print(f"4. Pattern_1 users in timeline: {len(timeline_users):,}")
    print(f"5. ↓ PMDD users in timeline: {n_pmdd_users_in_timeline:,} ({n_pmdd_users_in_timeline/len(timeline_users)*100:.1f}% of timeline)")
    print(f"6. ↓ Posted in MH subs within ±3 months: {n_pmdd_users_posted_mh_window} users, {len(mh_posts):,} posts")
    print(f"5. FFT period detection (textblob_polarity, SNR >= 2.0):")
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
        'post_id': 'count',
        'author': 'nunique'
    }).reset_index()
    category_counts.columns = ['Category', 'Posts', 'Unique Users']
    print(category_counts.to_string(index=False))
    
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
            
            avg_sentiment = phase_posts['textblob_polarity'].mean() if 'textblob_polarity' in phase_posts.columns else None
            
            phase_stats.append({
                'Phase': phase,
                'Total Posts': n_posts,
                'Users': n_users,
                'Depression': len(depression_posts),
                'Suicide': len(suicide_posts),
                'Avg Sentiment': f"{avg_sentiment:.3f}" if avg_sentiment is not None else "N/A"
            })
        
        phase_df = pd.DataFrame(phase_stats)
        total_posts = phase_df['Total Posts'].sum()
        phase_df['% of Total'] = (phase_df['Total Posts'] / total_posts * 100).round(1)
        
        print(phase_df.to_string(index=False))
        return phase_df
    
    # Calculate for users with FFT periods only
    mh_posts_fft = mh_posts[mh_posts['has_fft_period']].copy()
    phase_df_fft = calculate_phase_stats(
        mh_posts_fft, 
        f"By Menstrual Cycle Phase (FFT-detected periods only, n={users_with_detected} users)"
    )
    
    # Calculate for all users (FFT + fallback)
    phase_df_all = calculate_phase_stats(
        mh_posts,
        f"By Menstrual Cycle Phase (All users: FFT + 29-day fallback, n={n_pmdd_users_posted_mh_window} users)"
    )
    
    # Additional analysis: Critical symptom window (last 4 days luteal + first 2 days menstrual)
    print(f"\nCritical Symptom Window Analysis (All users):")
    print("-" * 70)
    
    def assign_symptom_window(row):
        """Assign posts to critical symptom window vs other days."""
        day = row['offset_from_cd1']
        period = row['detected_period']
        
        if pd.isna(day) or pd.isna(period):
            return None
        
        # Normalize day to cycle
        day_normalized = day % period
        
        # Critical window: last 4 days of luteal + first 2 days of menstrual
        # Luteal ends at period-1, so last 4 days are: period-4, period-3, period-2, period-1
        # Menstrual is days 0-1 (first 2 days)
        
        luteal_end = int(period - 1)
        last_4_luteal_start = max(14, luteal_end - 3)  # Luteal starts around day 14
        
        if (last_4_luteal_start <= day_normalized <= luteal_end) or (0 <= day_normalized <= 1):
            return "Critical Window (Late Luteal + Early Menstrual)"
        else:
            return "Other Days"
    
    mh_posts['symptom_window'] = mh_posts.apply(assign_symptom_window, axis=1)
    
    symptom_stats = []
    for window in ["Critical Window (Late Luteal + Early Menstrual)", "Other Days"]:
        window_posts = mh_posts[mh_posts['symptom_window'] == window]
        n_posts = len(window_posts)
        n_users = window_posts['author'].nunique()
        
        depression_posts = window_posts[window_posts['mh_category'] == 'Depression']
        suicide_posts = window_posts[window_posts['mh_category'] == 'Suicide']
        
        avg_sentiment = window_posts['textblob_polarity'].mean() if 'textblob_polarity' in window_posts.columns else None
        
        symptom_stats.append({
            'Window': window,
            'Total Posts': n_posts,
            'Users': n_users,
            'Depression': len(depression_posts),
            'Suicide': len(suicide_posts),
            'Avg Sentiment': f"{avg_sentiment:.3f}" if avg_sentiment is not None else "N/A"
        })
    
    symptom_df = pd.DataFrame(symptom_stats)
    total_posts = symptom_df['Total Posts'].sum()
    symptom_df['% of Total'] = (symptom_df['Total Posts'] / total_posts * 100).round(1)
    
    print(symptom_df.to_string(index=False))
    
    # ========================================================================
    # Save results
    # ========================================================================
    print(f"\n[Saving Results]")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    # Save detailed post-level data
    posts_out = output_dir / f"mh_posts_by_phase_{timestamp}.csv"
    mh_posts[['author', 'post_id', 'subreddit', 'mh_category', 'offset_from_cd1', 
              'detected_period', 'has_fft_period', 'phase', 'textblob_polarity', 'created_utc']].to_csv(
        posts_out, index=False, encoding='utf-8-sig'
    )
    print(f"  ✓ Saved post-level data: {posts_out.name}")
    
    # Save phase statistics (both FFT-only and combined)
    stats_fft_out = output_dir / f"phase_statistics_fft_only_{timestamp}.csv"
    phase_df_fft.to_csv(stats_fft_out, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved phase statistics (FFT only): {stats_fft_out.name}")
    
    stats_all_out = output_dir / f"phase_statistics_all_{timestamp}.csv"
    phase_df_all.to_csv(stats_all_out, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved phase statistics (all users): {stats_all_out.name}")
    
    symptom_window_out = output_dir / f"symptom_window_statistics_{timestamp}.csv"
    symptom_df.to_csv(symptom_window_out, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved symptom window statistics: {symptom_window_out.name}")
    
    print(f"\n{'='*70}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

