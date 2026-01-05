#!/usr/bin/env python3
"""Compare PMDD users vs control users (crossposting to mental health subreddits).

Research Question:
Do PMDD users post more frequently in depression/suicide subreddits compared to 
control users (women who never mentioned PMDD)?

Groups:
- PMDD users: Posted in PMDD subreddits at least once + have cycle anchors
- Control users: Never posted in PMDD subreddits + have cycle anchors

This ensures both groups are:
- Women who post about menstrual cycles on Reddit
- Trackable (we know their cycle timing)
But differ in PMDD status.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io import parse_jsonl_file
from src.timeline import load_users_database
from src.config import load_config

# Subreddit groups
PMDD_SUBREDDITS = {"PMDD", "PMDDxADHD", "PMDDSharing"}
DEPRESSION_SUBREDDITS = {"depression", "mentalhealth", "Anxiety", "socialanxiety", "AnxietyDepression"}
SUICIDE_SUBREDDITS = {"SuicideWatch"}

ALL_MENTAL_HEALTH = DEPRESSION_SUBREDDITS | SUICIDE_SUBREDDITS


def get_user_subreddit_posts(posts_files: list, target_users: set, target_subreddits: set) -> dict:
    """Get posts from specific users in specific subreddits across multiple files.
    
    Args:
        posts_files: List of paths to posts JSONL files
        target_users: Set of usernames to look for
        target_subreddits: Set of subreddits to extract
        
    Returns:
        Dictionary mapping username -> list of posts
    """
    print(f"  Scanning {len(posts_files)} files for {len(target_users):,} users...")
    print(f"    Target subreddits: {', '.join(sorted(target_subreddits))}")
    
    user_posts = {}
    total_found = 0
    
    for posts_file in posts_files:
        print(f"    → {posts_file.name}")
        
        for data in parse_jsonl_file(posts_file, progress_interval=2_000_000):
            author = data.get("author")
            subreddit = data.get("subreddit")
            
            if author in target_users and subreddit in target_subreddits:
                if author not in user_posts:
                    user_posts[author] = []
                    
                user_posts[author].append({
                    "author": author,
                    "post_id": data.get("id"),
                    "subreddit": subreddit,
                    "created_utc": data.get("created_utc"),
                })
                total_found += 1
                
                if total_found % 500 == 0:
                    print(f"      Found {total_found:,} posts...", end="\r")
    
    print(f"\n  ✓ Found {total_found:,} posts from {len(user_posts):,} users")
    
    return user_posts


def calculate_posting_rates(user_posts: dict, total_users_in_group: int, group_name: str) -> dict:
    """Calculate posting statistics for a group.
    
    Args:
        user_posts: Dictionary mapping username -> list of posts (only users who posted)
        total_users_in_group: Total number of users in the group (including those who didn't post)
        group_name: Name of group (for display)
        
    Returns:
        Dictionary with statistics
    """
    users_with_posts = len(user_posts)
    total_posts = sum(len(posts) for posts in user_posts.values())
    
    stats = {
        "group": group_name,
        "total_users": total_users_in_group,
        "users_who_posted": users_with_posts,
        "percentage_who_posted": users_with_posts / total_users_in_group * 100 if total_users_in_group > 0 else 0,
        "total_posts": total_posts,
        "posts_per_user_mean": total_posts / total_users_in_group if total_users_in_group > 0 else 0,
    }
    
    return stats


def main():
    """Main comparison pipeline."""
    # Setup
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "interim" / "pmdd_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config(project_dir / "configs" / "base.yaml")
    
    # Use both moon1 and moon2 for all CD patterns
    posts_files = [
        raw_dir / "moon1_all_post_2015_2025.filtered.tsv",  # pattern_1
        raw_dir / "moon2_all_posts_2010_2025.tsv",  # patterns 2-6, 9
    ]
    
    print("="*70)
    print("PMDD vs Control: Mental Health Crossposting Comparison")
    print("="*70)
    print(f"\nInput files:")
    for pf in posts_files:
        print(f"  - {pf.name}")
    print(f"Output directory: {output_dir}")
    
    # ========================================================================
    # Step 1: Load users database - ALL CD PATTERNS
    # ========================================================================
    print("\n[Step 1/5] Loading users database (all CD patterns)...")
    
    # Use the larger Dec 15 database (15,474 users) instead of Dec 22 (323 users)
    users_db_path = data_dir / "processed" / "users_database_CD_20251215T171047.csv"
    users_df = pd.read_csv(users_db_path, encoding='utf-8-sig')
    print(f"Loaded users database: {users_db_path.name} ({len(users_df):,} rows)")
    
    all_cycle_users = set(users_df["user"].astype(str).unique())
    print(f"  ✓ Found {len(all_cycle_users):,} users with cycle anchors (all patterns)")
    
    # Show breakdown by pattern
    pattern_counts = users_df["pattern_category"].value_counts().sort_index()
    print(f"  Pattern breakdown:")
    for pattern, count in pattern_counts.items():
        print(f"    - {pattern}: {count:,} rows")
    
    # ========================================================================
    # Step 2: Identify PMDD users (scan all files)
    # ========================================================================
    print("\n[Step 2/5] Identifying PMDD users (scanning all files)...")
    
    pmdd_user_posts = {}
    for posts_file in posts_files:
        print(f"  → Scanning {posts_file.name}...")
        
        for data in parse_jsonl_file(posts_file, progress_interval=2_000_000):
            author = data.get("author")
            subreddit = data.get("subreddit")
            
            if subreddit in PMDD_SUBREDDITS and author in all_cycle_users:
                if author not in pmdd_user_posts:
                    pmdd_user_posts[author] = []
                pmdd_user_posts[author].append({
                    "post_id": data.get("id"),
                    "subreddit": subreddit,
                    "created_utc": data.get("created_utc"),
                })
    
    pmdd_users = set(pmdd_user_posts.keys())
    print(f"  ✓ Found {len(pmdd_users):,} PMDD users (posted in PMDD subreddits)")
    
    # ========================================================================
    # Step 3: Define control group
    # ========================================================================
    print("\n[Step 3/5] Defining control group...")
    control_users = all_cycle_users - pmdd_users
    print(f"  ✓ Control group: {len(control_users):,} users (never posted in PMDD)")
    
    # ========================================================================
    # Step 4: Analyze mental health posting for both groups (all files)
    # ========================================================================
    print("\n[Step 4/5] Analyzing mental health subreddit posting...")
    
    # 4a: Depression subreddits
    print("\n  [4a] Depression subreddits...")
    pmdd_depression = get_user_subreddit_posts(posts_files, pmdd_users, DEPRESSION_SUBREDDITS)
    control_depression = get_user_subreddit_posts(posts_files, control_users, DEPRESSION_SUBREDDITS)
    
    # 4b: Suicide subreddits
    print("\n  [4b] Suicide subreddits...")
    pmdd_suicide = get_user_subreddit_posts(posts_files, pmdd_users, SUICIDE_SUBREDDITS)
    control_suicide = get_user_subreddit_posts(posts_files, control_users, SUICIDE_SUBREDDITS)
    
    # 4c: Any mental health
    print("\n  [4c] Any mental health subreddits...")
    pmdd_any_mh = get_user_subreddit_posts(posts_files, pmdd_users, ALL_MENTAL_HEALTH)
    control_any_mh = get_user_subreddit_posts(posts_files, control_users, ALL_MENTAL_HEALTH)
    
    # ========================================================================
    # Step 5: Calculate statistics
    # ========================================================================
    print("\n[Step 5/5] Calculating statistics...")
    
    stats_depression = [
        calculate_posting_rates(pmdd_depression, len(pmdd_users), "PMDD - Depression"),
        calculate_posting_rates(control_depression, len(control_users), "Control - Depression"),
    ]
    
    stats_suicide = [
        calculate_posting_rates(pmdd_suicide, len(pmdd_users), "PMDD - Suicide"),
        calculate_posting_rates(control_suicide, len(control_users), "Control - Suicide"),
    ]
    
    stats_any_mh = [
        calculate_posting_rates(pmdd_any_mh, len(pmdd_users), "PMDD - Any Mental Health"),
        calculate_posting_rates(control_any_mh, len(control_users), "Control - Any Mental Health"),
    ]
    
    # Combine all stats
    all_stats = stats_depression + stats_suicide + stats_any_mh
    stats_df = pd.DataFrame(all_stats)
    
    # ========================================================================
    # Display Results
    # ========================================================================
    print("\n" + "="*70)
    print("RESULTS: PMDD vs Control Comparison")
    print("="*70)
    
    print(f"\n{'Group':<30} {'Total Users':>12} {'Posted':>12} {'% Posted':>12}")
    print("-"*70)
    for _, row in stats_df.iterrows():
        print(f"{row['group']:<30} {row['total_users']:>12,} {row['users_who_posted']:>12,} {row['percentage_who_posted']:>11.1f}%")
    
    # Calculate ratios
    pmdd_dep_pct = stats_df[stats_df['group'] == 'PMDD - Depression']['percentage_who_posted'].iloc[0]
    control_dep_pct = stats_df[stats_df['group'] == 'Control - Depression']['percentage_who_posted'].iloc[0]
    depression_ratio = pmdd_dep_pct / control_dep_pct if control_dep_pct > 0 else float('inf')
    
    pmdd_sui_pct = stats_df[stats_df['group'] == 'PMDD - Suicide']['percentage_who_posted'].iloc[0]
    control_sui_pct = stats_df[stats_df['group'] == 'Control - Suicide']['percentage_who_posted'].iloc[0]
    suicide_ratio = pmdd_sui_pct / control_sui_pct if control_sui_pct > 0 else float('inf')
    
    # ========================================================================
    # Chi-square tests: Are PMDD users more likely to post in MH subreddits?
    # ========================================================================
    print("\n" + "="*70)
    print("CHI-SQUARE TESTS: PMDD vs Control (Mental Health Posting)")
    print("="*70)
    
    def run_chi_square_test(group1_posted: int, group1_total: int, 
                            group2_posted: int, group2_total: int,
                            group1_name: str, group2_name: str,
                            test_name: str):
        """Run chi-square test of independence for posting rates."""
        group1_not_posted = group1_total - group1_posted
        group2_not_posted = group2_total - group2_posted
        
        # Build 2x2 contingency table
        contingency_table = np.array([
            [group1_posted, group1_not_posted],  # PMDD: Posted, Didn't post
            [group2_posted, group2_not_posted],  # Control: Posted, Didn't post
        ])
        
        # Calculate percentages
        pct_group1_posted = group1_posted / group1_total * 100
        pct_group2_posted = group2_posted / group2_total * 100
        
        # Run chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n{test_name}:")
        print("-" * 70)
        print(f"2x2 Contingency Table:")
        print(f"                     Posted    Didn't Post    Total")
        print(f"{group1_name:20s} {group1_posted:8,} {group1_not_posted:12,} {group1_total:8,}")
        print(f"{group2_name:20s} {group2_posted:8,} {group2_not_posted:12,} {group2_total:8,}")
        print(f"\nPercentages:")
        print(f"  {group1_name}: {pct_group1_posted:.1f}% posted")
        print(f"  {group2_name}: {pct_group2_posted:.1f}% posted")
        print(f"\nChi-square test results:")
        print(f"  Expected values:")
        print(f"    {group1_name} Posted: {expected[0,0]:.1f}")
        print(f"    {group1_name} Didn't Post: {expected[0,1]:.1f}")
        print(f"    {group2_name} Posted: {expected[1,0]:.1f}")
        print(f"    {group2_name} Didn't Post: {expected[1,1]:.1f}")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  ✅ SIGNIFICANT (p < 0.05): {group1_name} users are significantly more likely to post")
            if pct_group1_posted > pct_group2_posted:
                print(f"     → {group1_name} users post MORE than {group2_name} users")
            else:
                print(f"     → {group1_name} users post LESS than {group2_name} users")
        else:
            print(f"  ❌ NOT SIGNIFICANT (p >= 0.05): No significant difference in posting rates")
        return chi2, p_value
    
    # Depression subreddits
    pmdd_dep_posted = stats_df[stats_df['group'] == 'PMDD - Depression']['users_who_posted'].iloc[0]
    pmdd_dep_total = stats_df[stats_df['group'] == 'PMDD - Depression']['total_users'].iloc[0]
    control_dep_posted = stats_df[stats_df['group'] == 'Control - Depression']['users_who_posted'].iloc[0]
    control_dep_total = stats_df[stats_df['group'] == 'Control - Depression']['total_users'].iloc[0]
    
    chi2_dep, p_dep = run_chi_square_test(
        pmdd_dep_posted, pmdd_dep_total,
        control_dep_posted, control_dep_total,
        "PMDD", "Control",
        "Depression Subreddits"
    )
    
    # Suicide subreddits
    pmdd_sui_posted = stats_df[stats_df['group'] == 'PMDD - Suicide']['users_who_posted'].iloc[0]
    pmdd_sui_total = stats_df[stats_df['group'] == 'PMDD - Suicide']['total_users'].iloc[0]
    control_sui_posted = stats_df[stats_df['group'] == 'Control - Suicide']['users_who_posted'].iloc[0]
    control_sui_total = stats_df[stats_df['group'] == 'Control - Suicide']['total_users'].iloc[0]
    
    chi2_sui, p_sui = run_chi_square_test(
        pmdd_sui_posted, pmdd_sui_total,
        control_sui_posted, control_sui_total,
        "PMDD", "Control",
        "Suicide Subreddits"
    )
    
    # Any mental health
    pmdd_mh_posted = stats_df[stats_df['group'] == 'PMDD - Any Mental Health']['users_who_posted'].iloc[0]
    pmdd_mh_total = stats_df[stats_df['group'] == 'PMDD - Any Mental Health']['total_users'].iloc[0]
    control_mh_posted = stats_df[stats_df['group'] == 'Control - Any Mental Health']['users_who_posted'].iloc[0]
    control_mh_total = stats_df[stats_df['group'] == 'Control - Any Mental Health']['total_users'].iloc[0]
    
    chi2_mh, p_mh = run_chi_square_test(
        pmdd_mh_posted, pmdd_mh_total,
        control_mh_posted, control_mh_total,
        "PMDD", "Control",
        "Any Mental Health Subreddits"
    )
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(f"Depression posting:  PMDD {pmdd_dep_pct:.1f}% vs Control {control_dep_pct:.1f}% → {depression_ratio:.2f}x higher (p={p_dep:.4f})")
    print(f"Suicide posting:     PMDD {pmdd_sui_pct:.1f}% vs Control {control_sui_pct:.1f}% → {suicide_ratio:.2f}x higher (p={p_sui:.4f})")
    
    pmdd_any_pct = stats_df[stats_df['group'] == 'PMDD - Any Mental Health']['percentage_who_posted'].iloc[0]
    control_any_pct = stats_df[stats_df['group'] == 'Control - Any Mental Health']['percentage_who_posted'].iloc[0]
    any_mh_ratio = pmdd_any_pct / control_any_pct if control_any_pct > 0 else float('inf')
    print(f"Any MH posting:       PMDD {pmdd_any_pct:.1f}% vs Control {control_any_pct:.1f}% → {any_mh_ratio:.2f}x higher (p={p_mh:.4f})")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n[Saving Results]")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    stats_file = output_dir / f"comparison_stats_{timestamp}.csv"
    stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved statistics: {stats_file.name}")
    
    # Save chi-square test results
    chi2_results = pd.DataFrame([
        {
            'category': 'Depression',
            'chi2': chi2_dep,
            'p_value': p_dep,
            'significant': p_dep < 0.05
        },
        {
            'category': 'Suicide',
            'chi2': chi2_sui,
            'p_value': p_sui,
            'significant': p_sui < 0.05
        },
        {
            'category': 'Any Mental Health',
            'chi2': chi2_mh,
            'p_value': p_mh,
            'significant': p_mh < 0.05
        }
    ])
    chi2_file = output_dir / f"chi2_test_results_{timestamp}.csv"
    chi2_results.to_csv(chi2_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved chi-square test results: {chi2_file.name}")
    
    # Save user lists
    pmdd_list = pd.DataFrame([{"user": u, "group": "PMDD"} for u in pmdd_users])
    control_list = pd.DataFrame([{"user": u, "group": "Control"} for u in control_users])
    users_file = output_dir / f"user_groups_{timestamp}.csv"
    pd.concat([pmdd_list, control_list]).to_csv(users_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved user groups: {users_file.name}")
    
    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

