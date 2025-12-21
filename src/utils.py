"""Utility functions for file I/O and data loading."""
import pandas as pd
from pathlib import Path

from src.io import find_latest_file


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

