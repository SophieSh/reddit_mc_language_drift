#!/usr/bin/env python3
"""Step 1: Validate precomputed feature files.

This script validates that the precomputed feature files exist and have the expected structure.
Features are already computed on all posts from moon1 and moon2, so this is just a validation step.
"""

from pathlib import Path
import pandas as pd
import sys

from src.config import load_config


def validate_feature_files(cfg: dict) -> tuple[bool, str]:
    """Validate that precomputed feature files exist and have feature columns.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, message)
    """
    raw_dir = Path(cfg["paths"]["raw"])
    files_cfg = cfg["paths"]["files"]
    
    moon1_file = raw_dir / files_cfg["moon1_posts"]
    moon2_file = raw_dir / files_cfg["moon2_posts"]
    
    # Check if files exist
    if not moon1_file.exists():
        return False, f"moon1 posts file not found: {moon1_file}"
    
    if not moon2_file.exists():
        return False, f"moon2 posts file not found: {moon2_file}"
    
    # Try to read and check columns exist
    try:
        # Read just the first row to check structure
        df1 = pd.read_csv(moon1_file, nrows=1, encoding='utf-8-sig')
        df2 = pd.read_csv(moon2_file, nrows=1, encoding='utf-8-sig')
        
        if len(df1.columns) == 0:
            return False, f"moon1 file has no columns: {moon1_file}"
        
        if len(df2.columns) == 0:
            return False, f"moon2 file has no columns: {moon2_file}"
        
        return True, f"✓ Validated {len(df1.columns)} columns in moon1, {len(df2.columns)} columns in moon2"
        
    except Exception as e:
        return False, f"Error reading feature files: {e}"


def main(config_path: str):
    """Validate precomputed feature files."""
    cfg = load_config(config_path)
    
    print("=" * 60)
    print("Step 1: Validating Precomputed Feature Files")
    print("=" * 60)
    print()
    
    is_valid, message = validate_feature_files(cfg)
    
    if is_valid:
        print(message)
        print("\n✓ Feature files validated successfully")
        return 0
    else:
        print(f"✗ Validation failed: {message}")
        return 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate precomputed feature files")
    ap.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    args = ap.parse_args()
    sys.exit(main(args.config))
