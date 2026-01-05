"""Utility functions for file I/O and data loading."""
import pandas as pd
from pathlib import Path

from src.io import find_latest_file, parse_jsonl_file


def load_latest_preprocessed_file(interim_dir: Path, pattern: str) -> pd.DataFrame | None:
    """Load the most recent preprocessed file matching the pattern.
    
    Args:
        interim_dir: Directory to search for files
        pattern: Glob pattern to match files (e.g., "moon_with_uncertainty_*.csv")
    
    Returns:
        DataFrame from the most recent matching file, or None if no files found
    """
    latest_file = find_latest_file(interim_dir, pattern)
    if latest_file is None:
        return None
    return pd.read_csv(latest_file, encoding='utf-8-sig')


def count_subreddits_in_file(
    file_path: Path | str,
    min_posts: int = 1,
    top_n: int | None = None,
    progress_interval: int = 1_000_000,
) -> pd.DataFrame:
    """Count posts per subreddit in a posts file.
    
    Efficiently streams through large JSONL files to count posts by subreddit.
    Useful for exploratory analysis to identify which subreddits are well-represented.
    
    Args:
        file_path: Path to posts file (JSONL format: row_id:{json})
        min_posts: Only return subreddits with at least this many posts (default: 1)
        top_n: If specified, return only top N subreddits by post count
        progress_interval: Print progress every N lines (default: 1M)
        
    Returns:
        DataFrame with columns:
        - subreddit: Subreddit name
        - post_count: Number of posts in that subreddit
        - percentage: Percentage of total posts
        Sorted by post_count descending
        
    Example:
        >>> from src.utils import count_subreddits_in_file
        >>> stats = count_subreddits_in_file(
        ...     "data/raw/moon1_all_post_2015_2025.filtered.tsv",
        ...     min_posts=100,
        ...     top_n=20
        ... )
        >>> print(stats.head())
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Counting subreddits in: {file_path.name}")
    print(f"  Streaming through file...")
    
    subreddit_counts = {}
    total_posts = 0
    
    for data in parse_jsonl_file(file_path, progress_interval=progress_interval):
        subreddit = data.get("subreddit")
        if subreddit:
            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
            total_posts += 1
    
    print(f"  ✓ Processed {total_posts:,} posts from {len(subreddit_counts):,} subreddits")
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {"subreddit": sub, "post_count": count}
        for sub, count in subreddit_counts.items()
    ])
    
    # Filter by minimum posts
    if min_posts > 1:
        before = len(df)
        df = df[df["post_count"] >= min_posts].copy()
        print(f"  Filtered to {len(df):,} subreddits with >={min_posts} posts (removed {before - len(df):,})")
    
    # Calculate percentage
    df["percentage"] = (df["post_count"] / total_posts * 100).round(2)
    
    # Sort by post count
    df = df.sort_values("post_count", ascending=False).reset_index(drop=True)
    
    # Limit to top N if specified
    if top_n is not None:
        df = df.head(top_n)
        print(f"  Showing top {len(df)} subreddits")
    
    return df


def filter_posts_by_subreddits(
    file_path: Path | str,
    target_subreddits: set[str] | list[str],
    output_path: Path | str | None = None,
    progress_interval: int = 1_000_000,
) -> pd.DataFrame:
    """Filter posts file to only include specific subreddits.
    
    Useful for extracting posts from mental health subreddits for analysis.
    
    Args:
        file_path: Path to posts file (JSONL format)
        target_subreddits: Set or list of subreddit names to keep
        output_path: If specified, save filtered posts to this file
        progress_interval: Print progress every N lines
        
    Returns:
        DataFrame with filtered posts
        
    Example:
        >>> mental_health_subs = {"PMDD", "depression", "SuicideWatch", "anxiety"}
        >>> filtered = filter_posts_by_subreddits(
        ...     "data/raw/moon1_all_post_2015_2025.filtered.tsv",
        ...     mental_health_subs
        ... )
    """
    file_path = Path(file_path)
    target_subreddits = set(target_subreddits)
    
    print(f"Filtering posts from {len(target_subreddits)} target subreddits...")
    print(f"  Target subreddits: {', '.join(sorted(target_subreddits))}")
    
    rows = []
    total_processed = 0
    
    for data in parse_jsonl_file(file_path, progress_interval=progress_interval):
        total_processed += 1
        subreddit = data.get("subreddit")
        
        if subreddit in target_subreddits:
            rows.append(data)
            
            if len(rows) % 10000 == 0:
                print(f"    Found {len(rows):,} matching posts...", end="\r")
    
    print(f"\n  ✓ Found {len(rows):,} posts from target subreddits ({len(rows)/total_processed*100:.2f}%)")
    
    df = pd.DataFrame(rows)
    
    if output_path and len(df) > 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved to: {output_path}")
    
    return df

