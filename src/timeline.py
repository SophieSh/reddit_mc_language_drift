"""Reusable functions for building user timelines from comments and posts."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.preprocess import (
    add_offsets_from_anchors,
    add_timestamp_columns,
    filter_deleted_authors,
    concatenate_title_selftext_text,
)
from src.io import find_latest_file, parse_jsonl_file


def select_users_by_pattern(
    users_df: pd.DataFrame,
    pattern: str,
    n_users: int = 50,
    seed: int | None = None,
) -> pd.DataFrame:
    """Select N random users from users database for a specific pattern.
    
    Note: The users database should already contain only non-uncertain users
    (created via create_users_database.py which filters out has_uncertainty=True).
    
    Args:
        users_df: Users database DataFrame with columns: user, timestep, offset_from_cd1, pattern_category
        pattern: Pattern name to filter by (e.g., "pattern_1", "pattern_7")
        n_users: Number of users to select
        seed: Random seed (should be passed from config for reproducibility)
    
    Returns:
        DataFrame with selected users
    """
    if "pattern_category" not in users_df.columns:
        raise ValueError("users_df must have 'pattern_category' column")
    
    # Filter to specified pattern
    df_pattern = users_df[users_df["pattern_category"] == pattern].copy()
    
    if len(df_pattern) == 0:
        raise ValueError(f"No users found for pattern: {pattern}")
    
    if len(df_pattern) < n_users:
        print(f"Warning: Only {len(df_pattern)} users available for {pattern}, using all")
        return df_pattern
    
    if seed is not None:
        random.seed(seed)
    selected_indices = random.sample(range(len(df_pattern)), n_users)
    selected = df_pattern.iloc[selected_indices].copy()
    
    seed_msg = f" (seed={seed})" if seed is not None else ""
    print(f"Selected {len(selected)} users for {pattern}{seed_msg}")
    return selected


def _load_content_from_raw(
    file_path: Path,
    target_users: set[str],
    content_type: str = "comments",
    min_chars: int = 150,
    progress_interval: int = 1_000_000,
) -> pd.DataFrame:
    """Load content for target users from raw file (format: row_id:{json}).
    
    Filters by text length FIRST (cheap), then by user membership (expensive).
    
    Outputs unified structure: both comments and posts have 'text' column.
    
    Args:
        file_path: Path to raw data file
        target_users: Set of user IDs to filter for
        content_type: "comments" or "posts" (for display messages and source column)
        min_chars: Minimum text length to keep (default 150)
        progress_interval: Print progress every N lines
    
    Returns:
        DataFrame with unified structure: author, created_utc, text, subreddit, id, source
    """
    rows = []
    matching_count = 0
    
    print(f"Streaming through {content_type} file, filtering by length>={min_chars} then users...")
    
    for data in parse_jsonl_file(file_path, progress_interval=progress_interval):
        # FILTER 1: Check text length FIRST (cheap operation)
        # For posts, check combined length without concatenating (faster)
        if content_type == "comments":
            text_length = len(data.get("body", ""))
        else:  # posts
            title = data.get("title") or ""
            selftext = data.get("selftext") or ""
            text_length = len(title) + len(selftext)
        
        if text_length < min_chars:
            continue
        
        # FILTER 2: Check user membership SECOND (expensive operation)
        author = data.get("author")
        if author not in target_users:
            continue
        
        matching_count += 1
        if matching_count % 10000 == 0:
            print(f"  Found {matching_count:,} matching {content_type}...", end="\r")
        
        # Now concatenate for posts (only for rows that passed filters)
        if content_type == "comments":
            text = data.get("body", "")
        else:  # posts
            text = concatenate_title_selftext_text(data.get("title"), data.get("selftext"))
        
        # Unified structure: both comments and posts have 'text' column
        filtered_data = {
            "author": author,
            "created_utc": data.get("created_utc"),
            "text": text.strip(),
            "subreddit": data.get("subreddit"),
            "id": data.get("id"),
            "source": content_type.rstrip("s"),  # "comments" -> "comment", "posts" -> "post"
        }
        rows.append(filtered_data)
    
    print(f"\n  Finished: found {len(rows):,} matching {content_type}")
    
    if not rows:
        print(f"  Warning: No matching users found in {file_path.name}")
        return pd.DataFrame()
    
    print(f"  Converting {len(rows):,} rows to DataFrame (may take 30-60 seconds)...")
    df = pd.DataFrame(rows)
    print(f"  ✓ DataFrame created: {len(df):,} rows × {len(df.columns)} columns")
    return df


def load_comments_from_raw(
    comments_path: Path,
    target_users: set[str],
    min_chars: int = 150,
) -> pd.DataFrame:
    """Load comments for target users from raw comments file (JSON-per-line format).
    
    Args:
        comments_path: Path to raw comments file
        target_users: Set of user IDs to filter for
        min_chars: Minimum body length to keep (default 150)
    
    Returns:
        DataFrame with raw comment data (not preprocessed)
    """
    return _load_content_from_raw(comments_path, target_users, "comments", min_chars=min_chars, progress_interval=1_000_000)


def load_posts_from_raw(
    posts_path: Path,
    target_users: set[str],
    min_chars: int = 150,
) -> pd.DataFrame:
    """Load posts for target users from raw posts file (JSON-per-line format).
    
    Filters by text length first (drops ~60%), then by user membership.
    
    Args:
        posts_path: Path to raw posts file
        target_users: Set of user IDs to filter for
        min_chars: Minimum title+selftext length to keep (default 150)
    
    Returns:
        DataFrame with raw post data (not preprocessed)
    """
    return _load_content_from_raw(posts_path, target_users, "posts", min_chars=min_chars, progress_interval=100_000)


def preprocess_content(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess content DataFrame: convert timestamps and clean up.
    
    Expects data already filtered by min_chars and target_users (from _load_content_from_raw).
    Only performs essential transformations: timestamp conversion and cleanup of Reddit markers.
    
    Args:
        df: DataFrame with unified structure from _load_content_from_raw
            (columns: author, created_utc, text, subreddit, id, source)
    
    Returns:
        Preprocessed DataFrame with 'ts_utc' column added
    """
    if "author" not in df.columns:
        raise KeyError("'author' column not found in DataFrame")
    
    if "text" not in df.columns:
        raise KeyError("'text' column not found - expected unified structure from _load_content_from_raw")
    
    if "created_utc" not in df.columns:
        raise KeyError("DataFrame must contain 'created_utc' column")
    
    print(f"  Preprocessing {len(df):,} entries...")
    
    # Safety check: filter deleted authors (defensive programming, unlikely but possible)
    before = len(df)
    df = filter_deleted_authors(df)
    if len(df) < before:
        print(f"    [1/3] Filtered deleted authors: {before:,} → {len(df):,}")
    else:
        print(f"    [1/3] No deleted authors found")
    
    # Remove Reddit markers for removed/deleted content (entire text is just marker)
    before = len(df)
    df = df[~df["text"].isin(["[removed]", "[deleted]", ""])].copy()
    df = df[df["text"].notna()].copy()
    if len(df) < before:
        print(f"    [2/3] Removed Reddit markers: {before:,} → {len(df):,}")
    else:
        print(f"    [2/3] No Reddit markers found")
    
    # Convert timestamps (essential transformation)
    print(f"    [3/3] Converting timestamps...")
    df = add_timestamp_columns(df, utc_col="created_utc", add_date_string=False)
    print(f"      ✓ Done")
    
    print(f"  ✓ Preprocessing complete: {len(df):,} entries")
    return df


def build_user_timeline(
    users_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    posts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build unified timeline for selected users from comments and posts DataFrames.
    
    Args:
        users_df: DataFrame with selected users (columns: user, timestep, offset_from_cd1)
        comments_df: DataFrame with all comments (columns: author, ts_utc, text, ...)
        posts_df: DataFrame with all posts (columns: author, ts_utc, text, ...)
    
    Returns:
        Combined DataFrame with columns: author, ts_utc, text, offset_from_cd1, source
    """
    target_users = set(users_df["user"].astype(str))
    
    # Build anchor lookup
    anchors = {}
    for _, row in users_df.iterrows():
        user = str(row["user"])
        anchor_ts = pd.to_datetime(row["timestep"])
        anchor_offset = int(row["offset_from_cd1"])
        anchors[user] = (anchor_ts, anchor_offset)
    
    print(f"\nBuilding timeline for {len(target_users)} users...")
    
    # Filter comments to target users
    if len(comments_df) > 0:
        df_comments = comments_df[comments_df["author"].isin(target_users)].copy()
        df_comments["source"] = "comment"
        print(f"  Comments: {len(df_comments):,} rows")
    else:
        df_comments = pd.DataFrame()
        print("  Comments: 0 rows")
    
    # Filter posts to target users
    if len(posts_df) > 0:
        df_posts = posts_df[posts_df["author"].isin(target_users)].copy()
        df_posts["source"] = "post"
        print(f"  Posts: {len(df_posts):,} rows")
    else:
        df_posts = pd.DataFrame()
        print("  Posts: 0 rows")
    
    # Filter to ±1 year of anchor
    def in_anchor_window(row: pd.Series, anchors_dict: dict) -> bool:
        user = str(row["author"])
        if user not in anchors_dict:
            return False
        anchor_ts, _ = anchors_dict[user]
        post_ts = pd.to_datetime(row["ts_utc"])
        days_diff = (post_ts.normalize() - anchor_ts.normalize()).days
        return abs(days_diff) <= 365
    
    if len(df_comments) > 0:
        before = len(df_comments)
        df_comments = df_comments[df_comments.apply(lambda r: in_anchor_window(r, anchors), axis=1)].copy()
        print(f"  Comments after ±1 year filter: {before:,} → {len(df_comments):,}")
    
    if len(df_posts) > 0:
        before = len(df_posts)
        df_posts = df_posts[df_posts.apply(lambda r: in_anchor_window(r, anchors), axis=1)].copy()
        print(f"  Posts after ±1 year filter: {before:,} → {len(df_posts):,}")
    
    # Calculate offsets
    if len(df_comments) > 0:
        df_comments = add_offsets_from_anchors(df_comments, anchors)
    
    if len(df_posts) > 0:
        df_posts = add_offsets_from_anchors(df_posts, anchors)
    
    # Combine
    common_cols = ["author", "ts_utc", "text", "offset_from_cd1", "source"]
    
    if len(df_comments) > 0 and len(df_posts) > 0:
        df_combined = pd.concat([
            df_comments[common_cols],
            df_posts[common_cols]
        ], ignore_index=True)
    elif len(df_comments) > 0:
        df_combined = df_comments[common_cols].copy()
    elif len(df_posts) > 0:
        df_combined = df_posts[common_cols].copy()
    else:
        raise ValueError("No data found for selected users!")
    
    # Sort by user and timestamp
    df_combined = df_combined.sort_values(["author", "ts_utc"]).reset_index(drop=True)
    
    print(f"\n  Total timeline entries: {len(df_combined):,}")
    print(f"  Users with data: {df_combined['author'].nunique()}")
    
    return df_combined


def load_users_database(
    cfg: dict,
    db_type: str = "cd",
    pattern: str | None = None,
) -> pd.DataFrame:
    """Load latest users database file.
    
    Args:
        cfg: Config dictionary with paths
        db_type: Database type - "cd" or "dpo" (default: "cd")
        pattern: Optional pattern name to filter (e.g., "pattern_1")
    
    Returns:
        DataFrame with users database (columns: user, timestep, text, offset_from_cd1, pattern_category)
    """
    processed_dir = Path(cfg["paths"]["processed"])
    db_pattern = cfg["paths"]["files"][f"users_db_{db_type}"]
    
    db_file = find_latest_file(processed_dir, db_pattern)
    if db_file is None:
        raise FileNotFoundError(f"No {db_pattern} found in {processed_dir}")
    
    df = pd.read_csv(db_file, encoding="utf-8-sig")
    print(f"Loaded users database: {db_file.name} ({len(df)} users)")
    
    if pattern is not None:
        df = df[df["pattern_category"] == pattern].copy()
        print(f"Filtered to {pattern}: {len(df)} users")
    
    return df


def build_anchor_dict(users_df: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, int]]:
    """Build anchor lookup dictionary from users database.
    
    Args:
        users_df: Users database DataFrame with columns: user, timestep, offset_from_cd1
    
    Returns:
        Dictionary mapping user -> (anchor_timestamp, anchor_offset_from_cd1)
    """
    anchors = {}
    for _, row in users_df.iterrows():
        user = str(row["user"])
        anchor_ts = pd.to_datetime(row["timestep"])
        anchor_offset = int(row["offset_from_cd1"])
        anchors[user] = (anchor_ts, anchor_offset)
    
    print(f"Built anchor dictionary for {len(anchors)} users")
    return anchors


def load_posts_for_users(
    posts_path: Path,
    users_df: pd.DataFrame,
    user_col: str = "user",
) -> pd.DataFrame:
    """Load posts for users in users database.
    
    Args:
        posts_path: Path to posts file (Excel or CSV)
        users_df: Users database DataFrame
        user_col: Column name for user identifier
    
    Returns:
        DataFrame with posts (not yet preprocessed)
    """
    print(f"Loading posts from {posts_path.name}...")
    
    if posts_path.suffix == ".xlsx":
        df = pd.read_excel(posts_path)
    elif posts_path.suffix == ".csv":
        df = pd.read_csv(posts_path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {posts_path.suffix}")
    
    print(f"  Total posts: {len(df):,}")
    
    target_users = set(users_df[user_col].astype(str))
    df = df[df["author"].isin(target_users)].copy()
    
    print(f"  Posts from target users: {len(df):,}")
    return df

