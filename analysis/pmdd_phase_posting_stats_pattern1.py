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
from scipy.stats import chi2_contingency

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
    print("\n[Step 1/6] Loading processed timeline data...")
    
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
    print("\n[Step 2/6] Loading PMDD users...")
    
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
    
    pmdd_users = set()
    for posts_file in post_files:
        print(f"  Scanning {posts_file.name} for PMDD posts...")
        
        for data in parse_jsonl_file(posts_file, progress_interval=2_000_000):
            author = data.get("author")
            subreddit = data.get("subreddit")
            
            if author in all_cd_users and subreddit in PMDD_SUBREDDITS:
                pmdd_users.add(author)
    
    print(f"  ✓ Found {len(pmdd_users):,} CD users who ever posted in PMDD subreddits (pattern_1 only)")
    
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
    print(f"\nPATTERN_1 ONLY:")
    print(f"1. Pattern_1 CD users with cycle anchors: {len(all_cd_users):,}")
    print(f"2. ↓ Posted in PMDD subreddits: {n_pmdd_users_all_patterns:,} ({n_pmdd_users_all_patterns/len(all_cd_users)*100:.1f}%)")
    print(f"\nFILTERED TO PATTERN_1 (have timeline+periodicity data):")
    print(f"3. Pattern_1 users in timeline: {len(timeline_users):,}")
    print(f"4. ↓ PMDD users in timeline: {len(pmdd_users_in_timeline):,} ({len(pmdd_users_in_timeline)/len(timeline_users)*100:.1f}% of timeline)")
    print(f"5. ↓ Posted in MH subs within timeline window: {n_pmdd_users_posted_mh_window} users, {n_mh_posts_in_timeline:,} posts")
    print(f"\nFFT period detection (sentiment_negative, SNR >= 3.0):")
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
            
            avg_sentiment = phase_posts['sentiment_negative'].mean() if 'sentiment_negative' in phase_posts.columns else None
            
            phase_stats.append({
                'Phase': phase,
                'Total Posts': n_posts,
                'Users': n_users,
                'Depression': len(depression_posts),
                'Suicide': len(suicide_posts),
                'Avg Negative Sentiment': f"{avg_sentiment:.3f}" if avg_sentiment is not None else "N/A"
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
    # Chi-square test (2x2): Are MH posts more likely in Luteal phase?
    # ========================================================================
    print(f"\n{'='*70}")
    print("CHI-SQUARE TEST (2x2): Mental Health Posts vs Luteal Phase")
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
            phases = create_adaptive_phases(period)
            return assign_phase_to_day(day, phases)
        
        pmdd_posts_all['phase'] = pmdd_posts_all.apply(assign_phase_for_post_fft, axis=1)
        pmdd_posts_all = pmdd_posts_all[pmdd_posts_all['phase'].notna()].copy()
        
        # Mark which posts are mental health posts
        mental_health_subs = DEPRESSION_SUBREDDITS | SUICIDE_SUBREDDITS
        pmdd_posts_all['is_mh_post'] = pmdd_posts_all['subreddit'].isin(mental_health_subs)
        pmdd_posts_all['is_luteal'] = pmdd_posts_all['phase'] == 'Luteal'
        
        # Build 2x2 contingency table
        contingency_table = pd.crosstab(
            pmdd_posts_all['is_mh_post'], 
            pmdd_posts_all['is_luteal'],
            margins=True
        )
        
        print(f"\n2x2 Contingency Table (FFT-detected periods only):")
        print("                     Is Luteal?")
        print("                 No          Yes        Total")
        print(f"Is MH Post?")
        print(f"  No          {contingency_table.loc[False, False]:6,}    {contingency_table.loc[False, True]:6,}    {contingency_table.loc[False, 'All']:6,}")
        print(f"  Yes         {contingency_table.loc[True, False]:6,}    {contingency_table.loc[True, True]:6,}    {contingency_table.loc[True, 'All']:6,}")
        print(f"  Total       {contingency_table.loc['All', False]:6,}    {contingency_table.loc['All', True]:6,}    {contingency_table.loc['All', 'All']:6,}")
        
        # Calculate percentages
        pct_mh_in_luteal = contingency_table.loc[True, True] / contingency_table.loc[True, 'All'] * 100
        pct_mh_in_non_luteal = contingency_table.loc[True, False] / contingency_table.loc[True, 'All'] * 100
        pct_non_mh_in_luteal = contingency_table.loc[False, True] / contingency_table.loc[False, 'All'] * 100
        pct_non_mh_in_non_luteal = contingency_table.loc[False, False] / contingency_table.loc[False, 'All'] * 100
        
        print(f"\nPercentages:")
        print(f"  MH posts in Luteal: {pct_mh_in_luteal:.1f}%")
        print(f"  MH posts in Non-Luteal: {pct_mh_in_non_luteal:.1f}%")
        print(f"  Non-MH posts in Luteal: {pct_non_mh_in_luteal:.1f}%")
        print(f"  Non-MH posts in Non-Luteal: {pct_non_mh_in_non_luteal:.1f}%")
        
        # Calculate theoretical phase proportions based on average cycle length of FFT users
        # Phase lengths: Menstrual=4, Follicular=variable, Ovulation=3, Luteal=14
        avg_period = np.mean(list(user_periods.values()))
        menstrual_len = 4
        ovulation_len = 3
        luteal_len = 14
        follicular_len = avg_period - menstrual_len - ovulation_len - luteal_len
        
        theoretical_luteal_pct = luteal_len / avg_period * 100
        theoretical_non_luteal_pct = (menstrual_len + follicular_len + ovulation_len) / avg_period * 100
        
        print(f"\nTheoretical phase proportions (based on avg cycle length {avg_period:.1f} days):")
        print(f"  Menstrual: {menstrual_len} days ({menstrual_len/avg_period*100:.1f}%)")
        print(f"  Follicular: {follicular_len:.1f} days ({follicular_len/avg_period*100:.1f}%)")
        print(f"  Ovulation: {ovulation_len} days ({ovulation_len/avg_period*100:.1f}%)")
        print(f"  Luteal: {luteal_len} days ({theoretical_luteal_pct:.1f}%)")
        print(f"  Non-Luteal (M+F+O): {theoretical_non_luteal_pct:.1f}%")
        
        # Observed phase proportions
        obs_luteal_pct = (contingency_table.loc['All', True] / contingency_table.loc['All', 'All']) * 100
        obs_non_luteal_pct = (contingency_table.loc['All', False] / contingency_table.loc['All', 'All']) * 100
        
        print(f"\nObserved phase distribution in posts:")
        print(f"  Luteal: {obs_luteal_pct:.1f}% of all posts")
        print(f"  Non-Luteal: {obs_non_luteal_pct:.1f}% of all posts")
        print(f"  Note: Chi-square test uses observed proportions in expected value calculation")
        
        # Run chi-square test
        contingency_2x2 = pd.crosstab(pmdd_posts_all['is_mh_post'], pmdd_posts_all['is_luteal'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_2x2)
        
        print(f"\nChi-square test results:")
        print(f"  Expected values (based on observed marginals):")
        print(f"    Non-MH, Non-Luteal: {expected[0,0]:.1f}")
        print(f"    Non-MH, Luteal: {expected[0,1]:.1f}")
        print(f"    MH, Non-Luteal: {expected[1,0]:.1f}")
        print(f"    MH, Luteal: {expected[1,1]:.1f}")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  ✅ SIGNIFICANT (p < 0.05): Mental health posts are significantly associated with luteal phase")
            if pct_mh_in_luteal > pct_non_mh_in_luteal:
                print(f"     → MH posts are MORE likely in luteal phase than non-MH posts")
                print(f"     → This suggests biological factors, not just phase length")
            else:
                print(f"     → MH posts are LESS likely in luteal phase than non-MH posts")
        else:
            print(f"  ❌ NOT SIGNIFICANT (p >= 0.05): No significant association")
            print(f"     → The concentration in luteal phase may be due to phase length, not biological factors")
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
    if 'sentiment_negative' in mh_posts.columns:
        save_cols.append('sentiment_negative')
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
    main()

