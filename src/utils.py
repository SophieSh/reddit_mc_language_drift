"""Utility functions for file I/O and data loading."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import find_latest_file, parse_jsonl_file

log = logging.getLogger(__name__)


def compute_family_pca(df: pd.DataFrame, families_path: Path, per_user_normalize: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Reduce features to one PC1 per attribute family (MATLAB-matching implementation).

    Matches runner_compute_frequencies_grouped.m (supervisor version):
      0. Per-user z-score every feature column (omitnan mean/std per user) — removes
         between-user variance so PCA captures within-user cycle structure
      Then for each family:
      1. Build pairwise Pearson correlation matrix (NaN-safe per pair)
      2. Drop columns with <80% finite off-diagonal correlations
      3. Eigendecompose correlation matrix; last eigenvector = PC1 (largest eigenvalue)
      4. Fix PC1 sign so sum of loadings is positive
      5. Z-score each feature independently (NaN-safe)
      6. Project z-scored data onto PC1 → family composite score (NaN propagates)

    Args:
        per_user_normalize: if True (default), apply per-user z-scoring before PCA,
                            matching the supervisor's MATLAB script.

    Returns (df_with_family_cols, list_of_family_col_names).
    """
    families = pd.read_excel(families_path)
    families["Attribute"] = families["Attribute"] + "_mean"

    df = df.copy()

    # Step 0: per-user z-score all feature columns (MATLAB lines 8-21)
    if per_user_normalize and "author" in df.columns:
        feat_cols = [c for c in families["Attribute"].tolist() if c in df.columns]
        for col in feat_cols:
            if df[col].var(skipna=True) > 0:
                mu  = df.groupby("author")[col].transform(lambda v: v.mean())
                sig = df.groupby("author")[col].transform(lambda v: v.std())
                sig = sig.fillna(1.0).replace(0.0, 1.0)
                df[col] = (df[col] - mu) / sig
        log.info("  Per-user z-scoring applied to %d feature columns", len(feat_cols))

    family_cols: list[str] = []

    for family, grp in families.groupby("Family"):
        attrs = [a for a in grp["Attribute"].tolist() if a in df.columns]
        if len(attrs) < 2:
            log.warning("Family %s: only %d feature(s) found in data, skipping", family, len(attrs))
            continue

        cur = df[attrs].values.astype(float)
        n_var = cur.shape[1]

        # Pairwise correlation matrix (NaN-safe, matching MATLAB corr() with 'type','Pearson')
        c = np.eye(n_var)
        for i1 in range(n_var):
            for i2 in range(i1 + 1, n_var):
                v1, v2 = cur[:, i1], cur[:, i2]
                both = np.isfinite(v1) & np.isfinite(v2)
                if both.sum() >= 2:
                    r = np.corrcoef(v1[both], v2[both])[0, 1]
                    c[i1, i2] = c[i2, i1] = r if np.isfinite(r) else 0.0
                else:
                    c[i1, i2] = c[i2, i1] = 0.0

        keep = np.sum(np.isfinite(c), axis=0) >= 0.8 * n_var
        attrs_k = [a for a, k in zip(attrs, keep) if k]
        c_k = c[np.ix_(keep, keep)]

        if len(attrs_k) < 2:
            log.warning("Family %s: fewer than 2 features survive keep_vars filter, skipping", family)
            continue

        eigenvalues, eigenvectors = np.linalg.eigh(c_k)
        pc1 = eigenvectors[:, -1]
        var_explained = eigenvalues[-1] / eigenvalues.sum() * 100

        if pc1.sum() < 0:
            pc1 = -pc1

        log.info("  Family %-15s: %2d features, PC1 = %.1f%% variance", family, len(attrs_k), var_explained)

        cur_k = df[attrs_k].values.astype(float)
        zdata = np.full_like(cur_k, np.nan)
        for i1 in range(cur_k.shape[1]):
            v = cur_k[:, i1]
            fin = np.isfinite(v)
            if fin.sum() > 1:
                mu, sd = v[fin].mean(), v[fin].std(ddof=1)
                zdata[fin, i1] = (v[fin] - mu) / sd if sd > 0 else 0.0

        col = f"family_{family.lower().replace(' ', '_')}"
        df[col] = zdata @ pc1
        family_cols.append(col)

    log.info("Family PCA: %d composites created: %s", len(family_cols), family_cols)
    return df, family_cols


def identify_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Identify feature columns (exclude metadata columns and non-numeric columns).

    Args:
        df: DataFrame with posts and features
        cfg: Configuration dictionary

    Returns:
        List of numeric feature column names
    """
    metadata_cols = set(cfg.get("reddit_metadata_columns", []))

    # Also exclude common non-feature columns that might not be in metadata list
    additional_exclude = {
        'offset_from_cd1', 'ts_utc', 'ts_date', 'id', 'source', 'source_cut',
        'created_utc', 'permalink', 'author'
    }

    all_cols = set(df.columns)
    # Filter: exclude metadata columns, additional exclude columns, and non-numeric columns
    feature_cols = [
        col for col in all_cols
        if col not in metadata_cols
        and col not in additional_exclude
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_cols


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


def extract_users_from_subreddits(
    posts_files: list[Path | str],
    target_subreddits: set[str] | list[str],
    user_db_or_path: Path | str | pd.DataFrame | None = None,
    user_col: str = "user",
    progress_interval: int = 2_000_000,
) -> pd.DataFrame:
    """Find all users who posted in specific subreddits, return DataFrame with their posts.
    
    Searches through posts files to find posts from target subreddits.
    If user_db_or_path is provided, only searches for posts from those users.
    
    Args:
        posts_files: List of paths to posts files (CSV or TSV/JSONL format)
        target_subreddits: Set or list of subreddit names to search
        user_db_or_path: Optional users database (DataFrame or path to CSV). If provided, only search posts from these users.
        user_col: Column name for user identifier in user_db (default: "user")
        progress_interval: Print progress every N lines for JSONL files (default: 2M)
    
    Returns:
        DataFrame with posts from target subreddits. Columns include: author, subreddit, created_utc, title, selftext, etc.
    
    Example:
        >>> pmdd_subs = {"PMDD", "PMDDxADHD"}
        >>> users_df = pd.read_csv("data/processed/users_database_CD_*.csv")
        >>> posts = extract_users_from_subreddits(
        ...     posts_files=["data/raw/moon1_posts.csv"],
        ...     target_subreddits=pmdd_subs,
        ...     user_db_or_path=users_df
        ... )
    """
    target_subreddits = set(target_subreddits)
    target_users = None
    
    # Load target users if user_db_or_path provided
    if user_db_or_path is not None:
        if isinstance(user_db_or_path, pd.DataFrame):
            target_users = set(user_db_or_path[user_col].astype(str).unique())
        else:
            # Load from file
            user_db_path = Path(user_db_or_path)
            if not user_db_path.exists():
                raise FileNotFoundError(f"User database not found: {user_db_path}")
            user_db_df = pd.read_csv(user_db_path, encoding="utf-8-sig")
            target_users = set(user_db_df[user_col].astype(str).unique())
        
        print(f"  Filtering to {len(target_users):,} target users")
    
    all_posts = []
    
    for posts_file in posts_files:
        posts_path = Path(posts_file)
        if not posts_path.exists():
            print(f"  WARNING: File not found: {posts_path.name}, skipping")
            continue
        
        print(f"  Processing: {posts_path.name}")
        
        # Determine file format and read accordingly
        if posts_path.suffix == ".csv":
            # CSV format (DataFrame)
            print(f"    Reading CSV file...")
            df = pd.read_csv(posts_path, encoding="utf-8-sig", low_memory=False)
            print(f"    Loaded {len(df):,} posts")
            
            # Filter by subreddit (case-sensitive matching)
            df = df[df["subreddit"].isin(target_subreddits)].copy()
            print(f"    Found {len(df):,} posts in target subreddits")
            
            # Filter by users if provided
            if target_users is not None:
                before = len(df)
                df = df[df["author"].astype(str).isin(target_users)].copy()
                print(f"    Filtered to target users: {before:,} -> {len(df):,} posts")
            
            if len(df) > 0:
                all_posts.append(df)
        
        elif posts_path.suffix in [".tsv", ".txt"]:
            # TSV/JSONL format (streaming)
            print(f"    Streaming through JSONL file...")
            rows = []
            total_processed = 0
            
            for data in parse_jsonl_file(posts_path, progress_interval=progress_interval):
                total_processed += 1
                subreddit = data.get("subreddit")
                author = data.get("author")
                
                # Filter by subreddit (case-sensitive)
                if subreddit is None or subreddit not in target_subreddits:
                    continue
                
                # Filter by users if provided
                if target_users is not None and author not in target_users:
                    continue
                
                rows.append(data)
                
                if len(rows) % 10000 == 0:
                    print(f"      Found {len(rows):,} matching posts...", end="\r")
            
            print(f"\n    Found {len(rows):,} matching posts")
            
            if len(rows) > 0:
                df = pd.DataFrame(rows)
                all_posts.append(df)
        
        else:
            print(f"    WARNING: Unsupported file format: {posts_path.suffix}, skipping")
            continue
    
    if not all_posts:
        print("  No matching posts found")
        return pd.DataFrame()
    
    # Combine all DataFrames
    result_df = pd.concat(all_posts, ignore_index=True)
    print(f"  Total matching posts: {len(result_df):,} from {result_df['author'].nunique():,} users")
    
    return result_df

