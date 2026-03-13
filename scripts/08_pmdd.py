#!/usr/bin/env python3
"""PMDD analysis pipeline.

Identifies PMDD users, analyzes their posting patterns by cycle phase,
and compares to control groups.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import load_config
from src.timeline import load_users_database, filter_timeline_by_offset_days
from src.utils import extract_users_from_subreddits
from src.analysis import assign_phases_to_timeline, calculate_phase_statistics, compute_user_phase_definitions
from src.io import find_latest_file, save_with_timestamp, find_periodicity_results


def main(
    config_path: str,
    pattern: str | None = None,
    timeline_dir: Path | None = None,
    days_window: int = 90,
    use_all_users: bool = False,
    fallback_period: int = 29,
    output_subdir: str | None = None,
):
    """Run PMDD analysis pipeline.
    
    Args:
        config_path: Path to config YAML file
        pattern: Pattern to filter (e.g., "pattern_1", default: all CD patterns)
        timeline_dir: Directory with timeline files (auto-detect if None)
        days_window: Days before/after anchor to include (default: 90)
        use_all_users: If True, use fallback_period for all users. If False, only use users with valid detected periods from periodicity results.
        fallback_period: Standard period to use when use_all_users=True or when periodicity results unavailable (default: 29)
        output_subdir: Subdirectory for outputs (default: "pmdd_analysis")
    """
    cfg = load_config(config_path)
    
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    raw_dir = Path(cfg["paths"]["raw"])
    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    analysis_dir = Path(cfg["paths"].get("analysis_dir", interim_dir / "test_pipeline_verify_28.12"))
    
    if output_subdir:
        output_dir = interim_dir / output_subdir
    else:
        output_dir = interim_dir / "pmdd_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PMDD Analysis Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Config: {config_path}")
    print(f"  Pattern filter: {pattern or 'all CD patterns'}")
    print(f"  Days window: ±{days_window} days")
    print(f"  Use all users: {use_all_users}")
    print(f"  Fallback period: {fallback_period} days")
    print(f"  Output directory: {output_dir}")
    
    # ========================================================================
    # Step 1: Load users database
    # ========================================================================
    print("\n[Step 1/9] Loading users database...")
    
    users_df = load_users_database(cfg, db_type="cd", pattern=pattern)
    all_cycle_users = set(users_df["user"].astype(str).unique())
    
    print(f"  Loaded {len(all_cycle_users):,} users with cycle anchors")
    if pattern:
        print(f"    Pattern: {pattern}")
    else:
        pattern_counts = users_df["pattern_category"].value_counts().sort_index()
        print(f"    Pattern breakdown:")
        for pat, count in pattern_counts.items():
            print(f"      - {pat}: {count:,} users")
    
    # ========================================================================
    # Step 2: Identify PMDD users
    # ========================================================================
    print("\n[Step 2/9] Identifying PMDD users...")
    
    pmdd_subreddits = set(cfg["subreddits"]["pmdd"])
    print(f"  PMDD subreddits: {', '.join(sorted(pmdd_subreddits))}")
    
    posts_files = [
        raw_dir / cfg["paths"]["files"]["moon1_posts"],
        raw_dir / cfg["paths"]["files"]["moon2_posts"],
    ]
    
    existing_posts_files = [pf for pf in posts_files if pf.exists()]
    if not existing_posts_files:
        print("  ERROR: No posts files found.")
        return
    
    missing_files = [pf for pf in posts_files if pf not in existing_posts_files]
    if missing_files:
        print(f"  WARNING: {len(missing_files)} posts file(s) not found (continuing with available files)")
    
    pmdd_posts_df = extract_users_from_subreddits(
        posts_files=existing_posts_files,
        target_subreddits=pmdd_subreddits,
        user_db_or_path=users_df,
    )
    
    if len(pmdd_posts_df) == 0:
        print("  ERROR: No PMDD users found.")
        return
    
    pmdd_users = set(pmdd_posts_df["author"].astype(str).unique())
    print(f"  Found {len(pmdd_users):,} PMDD users ({len(pmdd_posts_df):,} total posts)")
    
    pmdd_users_df = users_df[users_df["user"].isin(pmdd_users)].copy()
    pmdd_db_path = save_with_timestamp(
        pmdd_users_df,
        processed_dir,
        "pmdd_users",
    )
    print(f"  Saved PMDD users database: {pmdd_db_path.name}")
    
    # ========================================================================
    # Step 3: Define control group
    # ========================================================================
    print("\n[Step 3/9] Defining control group...")
    
    control_users = all_cycle_users - pmdd_users
    print(f"  Control group: {len(control_users):,} users (never posted in PMDD subreddits)")
    print(f"    PMDD users: {len(pmdd_users):,}")
    print(f"    Total cycle users: {len(all_cycle_users):,}")
    
    # ========================================================================
    # Step 4: Find all mental health posts (all time, from moon1/moon2)
    # ========================================================================
    print("\n[Step 4/9] Finding all mental health posts (all time)...")
    
    mental_health_subreddits = set(cfg["subreddits"]["mental_health"])
    depression_subreddits = set(cfg["subreddits"]["depression"])
    suicide_subreddits = set(cfg["subreddits"]["suicide"])
    adhd_subreddits = set(cfg["subreddits"]["adhd"])
    
    print(f"  Mental health subreddits: {', '.join(sorted(mental_health_subreddits))}")
    print(f"  ADHD subreddits: {', '.join(sorted(adhd_subreddits))}")
    
    # Create a DataFrame with target users for filtering (all_cycle_users = pmdd_users | control_users)
    target_users_df = pd.DataFrame({"user": list(all_cycle_users)})
    
    # Extract all mental health posts (including ADHD) from moon1/moon2 (all time), filtered to our target users
    # Combine mental_health_subreddits with adhd_subreddits for extraction
    all_subreddits_for_extraction = mental_health_subreddits | adhd_subreddits
    all_mh_posts_df = extract_users_from_subreddits(
        posts_files=existing_posts_files,
        target_subreddits=all_subreddits_for_extraction,
        user_db_or_path=target_users_df,
        progress_interval=2_000_000,
    )
    
    if len(all_mh_posts_df) == 0:
        print("  WARNING: No mental health posts found in posts files.")
        all_mh_posts_df = pd.DataFrame()
    else:
        print(f"  Found {len(all_mh_posts_df):,} mental health posts (all time) from target users")
    
    # Calculate statistics: how many users ever posted in depression/suicide subreddits
    if len(all_mh_posts_df) > 0:
        # Calculate statistics: how many users ever posted in depression/suicide subreddits
        pmdd_mh_posts = all_mh_posts_df[all_mh_posts_df["author"].astype(str).isin(pmdd_users)].copy()
        control_mh_posts = all_mh_posts_df[all_mh_posts_df["author"].astype(str).isin(control_users)].copy()
        
        # Count unique users by subreddit type (case-sensitive matching)
        pmdd_depression_users_alltime = set(pmdd_mh_posts[
            pmdd_mh_posts["subreddit"].isin(depression_subreddits)
        ]["author"].astype(str).unique())
        pmdd_suicide_users_alltime = set(pmdd_mh_posts[
            pmdd_mh_posts["subreddit"].isin(suicide_subreddits)
        ]["author"].astype(str).unique())
        pmdd_adhd_users_alltime = set(pmdd_mh_posts[
            pmdd_mh_posts["subreddit"].isin(adhd_subreddits)
        ]["author"].astype(str).unique())
        
        control_depression_users_alltime = set(control_mh_posts[
            control_mh_posts["subreddit"].isin(depression_subreddits)
        ]["author"].astype(str).unique())
        control_suicide_users_alltime = set(control_mh_posts[
            control_mh_posts["subreddit"].isin(suicide_subreddits)
        ]["author"].astype(str).unique())
        control_adhd_users_alltime = set(control_mh_posts[
            control_mh_posts["subreddit"].isin(adhd_subreddits)
        ]["author"].astype(str).unique())
        
        print("\n  All-time posting statistics (from moon1/moon2 files):")
        print(f"    PMDD users:")
        print(f"      Ever posted in depression subreddits: {len(pmdd_depression_users_alltime):,} / {len(pmdd_users):,} ({100*len(pmdd_depression_users_alltime)/len(pmdd_users):.1f}%)")
        print(f"      Ever posted in suicide subreddits: {len(pmdd_suicide_users_alltime):,} / {len(pmdd_users):,} ({100*len(pmdd_suicide_users_alltime)/len(pmdd_users):.1f}%)")
        print(f"      Ever posted in ADHD subreddits: {len(pmdd_adhd_users_alltime):,} / {len(pmdd_users):,} ({100*len(pmdd_adhd_users_alltime)/len(pmdd_users):.1f}%)")
        print(f"    Control users:")
        print(f"      Ever posted in depression subreddits: {len(control_depression_users_alltime):,} / {len(control_users):,} ({100*len(control_depression_users_alltime)/len(control_users):.1f}%)")
        print(f"      Ever posted in suicide subreddits: {len(control_suicide_users_alltime):,} / {len(control_users):,} ({100*len(control_suicide_users_alltime)/len(control_users):.1f}%)")
        print(f"      Ever posted in ADHD subreddits: {len(control_adhd_users_alltime):,} / {len(control_users):,} ({100*len(control_adhd_users_alltime)/len(control_users):.1f}%)")
    else:
        print("  WARNING: No mental health posts found, skipping statistics.")
    
    # ========================================================================
    # Step 5: Load timeline and filter to mental health posts
    # ========================================================================
    print("\n[Step 5/9] Loading timeline and filtering to mental health posts...")
    
    if timeline_dir is None:
        timeline_dir = interim_dir  # Use interim_dir instead of analysis_dir for new pipeline
    
    # Look for new pipeline timeline files WITH ANCHORS (required)
    # Priority: 1) with_anchors (required), 2) old pattern files for backwards compatibility
    timeline_path = find_latest_file(timeline_dir, "timeline_with_offsets_with_anchors_*.csv")
    # Backwards compatibility: old pattern files (these typically include anchors)
    if timeline_path is None:
        timeline_path = find_latest_file(timeline_dir, "timeline_pattern1_*.csv")
    if timeline_path is None:
        timeline_path = find_latest_file(timeline_dir, "timeline_*.csv")
    
    if timeline_path is None:
        print("  ERROR: Timeline file not found.")
        return
    
    print(f"  Loading: {timeline_path.name}")
    timeline_df = pd.read_csv(timeline_path, encoding="utf-8-sig")
    print(f"  Loaded {len(timeline_df):,} posts")
    
    user_col = "user" if "user" in timeline_df.columns else "author"
    if user_col not in timeline_df.columns:
        print(f"  ERROR: Timeline missing user column (expected 'user' or 'author').")
        return
    
    # Check if timeline is pattern_1 specific and filter users accordingly
    is_pattern1_timeline = "pattern1" in timeline_path.name.lower() or "pattern_1" in timeline_path.name.lower()
    
    if is_pattern1_timeline:
        print(f"  WARNING: Timeline appears to be pattern_1 specific, filtering users to pattern_1 only")
        # Filter users database to pattern_1
        pattern1_users_df = users_df[users_df["pattern_category"] == "pattern_1"].copy()
        pattern1_users = set(pattern1_users_df["user"].astype(str).unique())
        
        # Filter PMDD and control users to only pattern_1 users
        pmdd_users = pmdd_users & pattern1_users
        control_users = control_users & pattern1_users
        all_cycle_users = pattern1_users  # Update all_cycle_users for timeline filtering
        
        print(f"    Filtered to pattern_1 users: PMDD {len(pmdd_users):,}, Control {len(control_users):,}")
    
    timeline_df = timeline_df[timeline_df[user_col].astype(str).isin(all_cycle_users)].copy()
    print(f"  Filtered to target users: {len(timeline_df):,} posts")
    
    timeline_df = filter_timeline_by_offset_days(
        timeline_df,
        days_before=days_window,
        days_after=days_window,
        offset_col="offset_from_cd1",
    )
    
    # Filter timeline to only mental health posts (including ADHD)
    if "subreddit" not in timeline_df.columns:
        print("  ERROR: Timeline missing 'subreddit' column.")
        return
    
    # Include both mental health and ADHD subreddits for timeline filtering
    all_subreddits_for_timeline = mental_health_subreddits | adhd_subreddits
    mental_health_df = timeline_df[timeline_df["subreddit"].isin(all_subreddits_for_timeline)].copy()
    print(f"  Found {len(mental_health_df):,} mental health posts in timeline")
    
    pmdd_mental_health_df = mental_health_df[
        mental_health_df[user_col].astype(str).isin(pmdd_users)
    ].copy()
    control_mental_health_df = mental_health_df[
        mental_health_df[user_col].astype(str).isin(control_users)
    ].copy()
    
    print(f"  PMDD group: {len(pmdd_mental_health_df):,} posts from {pmdd_mental_health_df[user_col].nunique():,} users")
    print(f"  Control group: {len(control_mental_health_df):,} posts from {control_mental_health_df[user_col].nunique():,} users")
    
    # ========================================================================
    # Step 6: Load periodicity results and create user->period mapping
    # ========================================================================
    print("\n[Step 6/9] Loading periodicity results...")
    
    user_period_map = {}
    use_fallback = use_all_users
    
    if not use_fallback:
        # Use config pattern if specified, otherwise use default search
        periodicity_pattern = cfg["paths"].get("periodicity_file", None)
        if periodicity_pattern:
            periodicity_files = list(analysis_dir.glob(periodicity_pattern))
            if periodicity_files:
                periodicity_path = max(periodicity_files, key=lambda p: p.stat().st_mtime)
            else:
                periodicity_path = None
        else:
            periodicity_path = find_periodicity_results(analysis_dir)
        
        if periodicity_path is None:
            print(f"  WARNING: No periodicity results found.")
            use_fallback = True
        else:
            print(f"  Loading periodicity results: {periodicity_path.name}")
            periodicity_df = pd.read_csv(periodicity_path, encoding="utf-8-sig")
            
            if "user" not in periodicity_df.columns or "consensus_period" not in periodicity_df.columns:
                print(f"  ERROR: Periodicity results missing required columns (user, consensus_period).")
                use_fallback = True
            else:
                # Filter out rows with missing periods
                periodicity_df = periodicity_df[periodicity_df["consensus_period"].notna()].copy()
                if len(periodicity_df) == 0:
                    print(f"  WARNING: No valid periods found in results.")
                    use_fallback = True
                else:
                    user_period_map = dict(zip(
                        periodicity_df["user"].astype(str),
                        periodicity_df["consensus_period"].astype(float)
                    ))
                    
                    users_with_period = set(user_period_map.keys())
                    
                    print(f"  Found valid periods for {len(user_period_map):,} users")
                
                pmdd_users_before = len(pmdd_users)
                control_users_before = len(control_users)
                pmdd_users = pmdd_users & users_with_period
                control_users = control_users & users_with_period
                print(f"  PMDD users with valid periods: {len(pmdd_users):,}")
                print(f"  Control users with valid periods: {len(control_users):,}")
                
                pmdd_mental_health_df = pmdd_mental_health_df[
                    pmdd_mental_health_df[user_col].astype(str).isin(pmdd_users)
                ].copy()
                control_mental_health_df = control_mental_health_df[
                    control_mental_health_df[user_col].astype(str).isin(control_users)
                ].copy()
                print(f"  Filtered mental health posts: PMDD {len(pmdd_mental_health_df):,} posts, Control {len(control_mental_health_df):,} posts")
    
    if use_fallback:
        print(f"  Using fallback period ({fallback_period} days) for all users")
        user_period_map = {user: fallback_period for user in all_cycle_users}
        print(f"  Assigned {fallback_period}-day period to {len(user_period_map):,} users")
    
    # Pre-compute phase definitions for all users (used in Steps 7 and 8)
    print("\n  Pre-computing phase definitions for all users...")
    user_phase_df = compute_user_phase_definitions(user_period_map)
    print(f"  Computed phase definitions for {user_phase_df['user'].nunique():,} users ({len(user_phase_df):,} user-phase combinations)")
    
    # ========================================================================
    # Step 7: Assign phases
    # ========================================================================
    print("\n[Step 7/9] Assigning phases to posts...")
    
    pmdd_mental_health_df = assign_phases_to_timeline(
        pmdd_mental_health_df,
        user_phase_df,
        user_col=user_col,
        time_col="offset_from_cd1",
    )
    control_mental_health_df = assign_phases_to_timeline(
        control_mental_health_df,
        user_phase_df,
        user_col=user_col,
        time_col="offset_from_cd1",
    )
    
    # ========================================================================
    # Step 8: Calculate statistics
    # ========================================================================
    print("\n[Step 8/9] Calculating phase statistics...")
    
    suicide_subreddits = set(cfg["subreddits"]["suicide"])
    depression_subreddits = set(cfg["subreddits"]["depression"])
    adhd_subreddits = set(cfg["subreddits"]["adhd"])
    
    pmdd_stats = calculate_phase_statistics(
        pmdd_mental_health_df,
        "PMDD",
        user_phase_df,
        user_col=user_col,
        suicide_subreddits=suicide_subreddits,
        depression_subreddits=depression_subreddits,
        adhd_subreddits=adhd_subreddits,
    )
    control_stats = calculate_phase_statistics(
        control_mental_health_df,
        "Control",
        user_phase_df,
        user_col=user_col,
        suicide_subreddits=suicide_subreddits,
        depression_subreddits=depression_subreddits,
        adhd_subreddits=adhd_subreddits,
    )
    
    # Combine into final table
    if len(pmdd_stats) > 0 and len(control_stats) > 0:
        stats_table = pd.concat([pmdd_stats, control_stats], ignore_index=True)
        
        # Reorder columns
        stats_table = stats_table[[
            "group", "phase", 
            "suicide_volume", "depression_volume", "adhd_volume",
            "suicide_raw", "depression_raw", "adhd_raw",
            "n_users", "n_users_suicide", "n_users_depression", "n_users_adhd",
            "suicide_zscore", "depression_zscore", "adhd_zscore"
        ]]
        
        print(f"  Created statistics table: {len(stats_table)} rows × {len(stats_table.columns)} columns")
    else:
        stats_table = pd.DataFrame()
        print("  WARNING: Could not create statistics table (missing data)")
    
    # ========================================================================
    # Step 9: Save results
    # ========================================================================
    print("\n[Step 9/9] Saving results...")
    
    if len(stats_table) > 0:
        stats_path = save_with_timestamp(stats_table, output_dir, "pmdd_phase_statistics")
        print(f"  Saved statistics table: {stats_path.name}")
    else:
        print("  WARNING: No statistics to save")
    
    if len(user_phase_df) > 0:
        phase_def_path = save_with_timestamp(user_phase_df, output_dir, "user_phase_definitions")
        print(f"  Saved phase definitions: {phase_def_path.name}")
    
    print(f"\n{'=' * 70}")
    print("PMDD Analysis Pipeline Complete")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {output_dir}")
    if len(stats_table) > 0:
        print(f"  Statistics table: {len(stats_table)} rows")
    print(f"  PMDD users: {len(pmdd_users):,}")
    print(f"  Control users: {len(control_users):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PMDD analysis pipeline - identify users and analyze posting patterns"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Pattern to filter (e.g., 'pattern_1', default: all CD patterns)",
    )
    parser.add_argument(
        "--timeline-dir",
        type=Path,
        default=None,
        help="Directory with timeline files (auto-detect if not provided)",
    )
    parser.add_argument(
        "--days-window",
        type=int,
        default=90,
        help="Days before/after anchor to include (default: 90)",
    )
    parser.add_argument(
        "--use-all-users",
        action="store_true",
        help="Use all users with fallback period, not just those with detected periods",
    )
    parser.add_argument(
        "--fallback-period",
        type=int,
        default=29,
        help="Standard period for users without detected periods (default: 29)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Subdirectory for outputs (default: 'pmdd_analysis')",
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        pattern=args.pattern,
        timeline_dir=args.timeline_dir,
        days_window=args.days_window,
        use_all_users=args.use_all_users,
        fallback_period=args.fallback_period,
        output_subdir=args.output_subdir,
    )

