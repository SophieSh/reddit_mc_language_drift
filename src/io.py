"""File I/O utilities for consistent file handling."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Find the most recently modified file matching pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "timeline_*.csv")
    
    Returns:
        Path to latest file, or None if no files found
    """
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def save_with_timestamp(
    df: pd.DataFrame,
    directory: Path,
    prefix: str,
    suffix: str = ".csv",
    encoding: str = "utf-8-sig",
) -> Path:
    """Save DataFrame with timestamp in filename.
    
    Args:
        df: DataFrame to save
        directory: Output directory
        suffix: File extension (default: ".csv")
        encoding: File encoding (default: "utf-8-sig" for Excel compatibility)
    
    Returns:
        Path to saved file
    """
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"{prefix}_{timestamp}{suffix}"
    output_path = directory / filename
    df.to_csv(output_path, index=False, encoding=encoding)
    return output_path


def load_timeline_with_features(interim_dir: Path) -> pd.DataFrame | None:
    """Load timeline file with precomputed features if available.
    
    Prefers timeline_with_features_*.csv, falls back to timeline_*.csv.
    
    Args:
        interim_dir: Directory to search
    
    Returns:
        DataFrame or None if no timeline found
    """
    # Try timeline with features first
    timeline_with_features = find_latest_file(interim_dir, "timeline_with_features_*.csv")
    if timeline_with_features:
        return pd.read_csv(timeline_with_features, encoding="utf-8-sig")
    
    # Fall back to regular timeline
    timeline = find_latest_file(interim_dir, "timeline_*.csv")
    if timeline:
        return pd.read_csv(timeline, encoding="utf-8-sig")
    
    return None


def find_periodicity_results(interim_dir: Path, prefer_long_format: bool = True) -> Path | None:
    """Find latest periodicity results file.
    
    Args:
        interim_dir: Directory to search
        prefer_long_format: If True, prefer long format over table format
    
    Returns:
        Path to results file or None
    """
    all_files = list(interim_dir.glob("feature_periodicity_*.csv"))
    
    if prefer_long_format:
        long_format = [f for f in all_files if 'table' not in f.name]
        if long_format:
            return max(long_format, key=lambda p: p.stat().st_mtime)
    
    if all_files:
        return max(all_files, key=lambda p: p.stat().st_mtime)
    
    return None


def parse_jsonl_file(file_path: Path, progress_interval: int = 1_000_000):
    """Parse JSON-per-line file format (row_id:{json}).
    
    Streams through file line-by-line to handle large files efficiently.
    Yields parsed JSON objects, skipping malformed lines.
    
    Args:
        file_path: Path to JSONL file
        progress_interval: Print progress every N lines
    
    Yields:
        Parsed JSON dictionaries
    """
    lines_processed = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            lines_processed += 1
            if lines_processed % progress_interval == 0:
                print(f"  Processed {lines_processed:,} lines...", end="\r")
            
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            colon_pos = line.find(":")
            json_part = line[colon_pos + 1 :].strip()
            
            try:
                data = json.loads(json_part)
                yield data
            except (json.JSONDecodeError, AttributeError):
                continue
    
    print(f"\n  Finished parsing: {lines_processed:,} lines")

