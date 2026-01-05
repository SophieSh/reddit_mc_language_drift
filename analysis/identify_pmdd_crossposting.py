#!/usr/bin/env python3
"""Identify PMDD users and their crossposting to mental health subreddits.

Research Question:
Do users who post in PMDD subreddits also post in depression/suicide subreddits?

Subreddit Groups:
- PMDD: PMDD, PMDDxADHD, PMDDSharing
- Depression: depression, mentalhealth, Anxiety, socialanxiety
- Suicide: SuicideWatch

Output:
1. PMDD users with all their posts (CSV)
2. Crossposting statistics (CSV)
3. Users who posted in both PMDD and mental health (CSV)
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io import parse_jsonl_file

# Subreddit groups
PMDD_SUBREDDITS = {"PMDD", "PMDDxADHD", "PMDDSharing"}
DEPRESSION_SUBREDDITS = {"depression", "mentalhealth", "Anxiety", "socialanxiety", "AnxietyDepression"}
SUICIDE_SUBREDDITS = {"SuicideWatch"}

ALL_TARGET_SUBREDDITS = PMDD_SUBREDDITS | DEPRESSION_SUBREDDITS | SUICIDE_SUBREDDITS


def extract_pmdd_users_posts(posts_file: Path) -> pd.DataFrame:
    """Extract all posts from PMDD users.
    
    Args:
        posts_file: Path to posts JSONL file
        
    Returns:
        DataFrame with columns: author, post_id, subreddit, title, text, created_utc
    """
    print(f"\n[Step 1/3] Extracting PMDD user posts from: {posts_file.name}")
    print(f"  Target subreddits: {', '.join(sorted(ALL_TARGET_SUBREDDITS))}")
    
    rows = []
    total_processed = 0
    
    for data in parse_jsonl_file(posts_file, progress_interval=500_000):
        total_processed += 1
        subreddit = data.get("subreddit")
        
        if subreddit in ALL_TARGET_SUBREDDITS:
            # Combine title + selftext
            title = data.get("title", "")
            selftext = data.get("selftext", "")
            text = (title + " " + selftext).strip()
            
            rows.append({
                "author": data.get("author"),
                "post_id": data.get("id"),
                "subreddit": subreddit,
                "title": title,
                "text": text,
                "created_utc": data.get("created_utc"),
            })
            
            if len(rows) % 1000 == 0:
                print(f"    Found {len(rows):,} posts from target subreddits...", end="\r")
    
    print(f"\n  ✓ Found {len(rows):,} posts from target subreddits ({len(rows)/total_processed*100:.3f}%)")
    
    df = pd.DataFrame(rows)
    
    if len(df) > 0:
        # Convert timestamp
        df["ts_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        df = df.sort_values(["author", "ts_utc"]).reset_index(drop=True)
        
        print(f"  ✓ Posts by subreddit:")
        for sub in sorted(ALL_TARGET_SUBREDDITS):
            count = (df["subreddit"] == sub).sum()
            if count > 0:
                print(f"    - {sub}: {count:,} posts")
    
    return df


def identify_pmdd_users(posts_df: pd.DataFrame) -> set:
    """Identify users who posted in PMDD subreddits.
    
    Args:
        posts_df: DataFrame with posts
        
    Returns:
        Set of PMDD user IDs
    """
    print(f"\n[Step 2/3] Identifying PMDD users...")
    
    pmdd_posts = posts_df[posts_df["subreddit"].isin(PMDD_SUBREDDITS)]
    pmdd_users = set(pmdd_posts["author"].unique())
    
    print(f"  ✓ Found {len(pmdd_users):,} unique users who posted in PMDD subreddits")
    print(f"    Total PMDD posts: {len(pmdd_posts):,}")
    
    return pmdd_users


def analyze_crossposting(posts_df: pd.DataFrame, pmdd_users: set) -> dict:
    """Analyze crossposting behavior of PMDD users.
    
    Args:
        posts_df: DataFrame with all posts
        pmdd_users: Set of PMDD user IDs
        
    Returns:
        Dictionary with crossposting statistics
    """
    print(f"\n[Step 3/3] Analyzing crossposting behavior...")
    
    # Filter to PMDD users only
    pmdd_user_posts = posts_df[posts_df["author"].isin(pmdd_users)].copy()
    
    # Categorize posts
    pmdd_user_posts["category"] = pmdd_user_posts["subreddit"].apply(
        lambda x: "PMDD" if x in PMDD_SUBREDDITS
        else "Depression" if x in DEPRESSION_SUBREDDITS
        else "Suicide" if x in SUICIDE_SUBREDDITS
        else "Other"
    )
    
    # Count users by posting pattern
    user_categories = pmdd_user_posts.groupby("author")["category"].apply(set).to_dict()
    
    posted_depression = sum(1 for cats in user_categories.values() if "Depression" in cats)
    posted_suicide = sum(1 for cats in user_categories.values() if "Suicide" in cats)
    posted_both_mh = sum(1 for cats in user_categories.values() if "Depression" in cats or "Suicide" in cats)
    
    stats = {
        "total_pmdd_users": len(pmdd_users),
        "posted_depression": posted_depression,
        "posted_suicide": posted_suicide,
        "posted_any_mental_health": posted_both_mh,
        "pmdd_only": len(pmdd_users) - posted_both_mh,
    }
    
    print(f"\n  === Crossposting Statistics ===")
    print(f"  Total PMDD users: {stats['total_pmdd_users']:,}")
    print(f"  Posted in depression subs: {stats['posted_depression']:,} ({stats['posted_depression']/stats['total_pmdd_users']*100:.1f}%)")
    print(f"  Posted in suicide subs: {stats['posted_suicide']:,} ({stats['posted_suicide']/stats['total_pmdd_users']*100:.1f}%)")
    print(f"  Posted in ANY mental health: {stats['posted_any_mental_health']:,} ({stats['posted_any_mental_health']/stats['total_pmdd_users']*100:.1f}%)")
    print(f"  PMDD only (no mental health): {stats['pmdd_only']:,} ({stats['pmdd_only']/stats['total_pmdd_users']*100:.1f}%)")
    
    # Create user-level summary
    user_summary = []
    for author, posts in pmdd_user_posts.groupby("author"):
        categories = posts["category"].unique()
        
        user_summary.append({
            "author": author,
            "posted_pmdd": "PMDD" in categories,
            "posted_depression": "Depression" in categories,
            "posted_suicide": "Suicide" in categories,
            "n_pmdd_posts": (posts["category"] == "PMDD").sum(),
            "n_depression_posts": (posts["category"] == "Depression").sum(),
            "n_suicide_posts": (posts["category"] == "Suicide").sum(),
            "total_posts": len(posts),
            "first_post_date": posts["ts_utc"].min(),
            "last_post_date": posts["ts_utc"].max(),
        })
    
    user_summary_df = pd.DataFrame(user_summary)
    
    return stats, user_summary_df, pmdd_user_posts


def main():
    """Main analysis pipeline."""
    # Setup
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "interim" / "pmdd_crossposting"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    posts_file = raw_dir / "moon1_all_post_2015_2025.filtered.tsv"
    
    print("="*70)
    print("PMDD Crossposting Analysis")
    print("="*70)
    print(f"\nInput file: {posts_file}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Extract posts from target subreddits
    all_posts = extract_pmdd_users_posts(posts_file)
    
    if len(all_posts) == 0:
        print("\n❌ No posts found in target subreddits!")
        return
    
    # Step 2: Identify PMDD users
    pmdd_users = identify_pmdd_users(all_posts)
    
    # Step 3: Analyze crossposting
    stats, user_summary, pmdd_user_posts = analyze_crossposting(all_posts, pmdd_users)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    print(f"\n[Saving Results]")
    
    # 1. All posts from PMDD users
    posts_file_out = output_dir / f"pmdd_users_all_posts_{timestamp}.csv"
    pmdd_user_posts.to_csv(posts_file_out, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved PMDD user posts: {posts_file_out.name}")
    print(f"    ({len(pmdd_user_posts):,} posts from {len(pmdd_users):,} users)")
    
    # 2. User-level summary
    summary_file_out = output_dir / f"pmdd_users_summary_{timestamp}.csv"
    user_summary.to_csv(summary_file_out, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved user summary: {summary_file_out.name}")
    print(f"    ({len(user_summary):,} users)")
    
    # 3. Statistics
    stats_file_out = output_dir / f"crossposting_stats_{timestamp}.csv"
    pd.DataFrame([stats]).to_csv(stats_file_out, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved statistics: {stats_file_out.name}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

