"""Analyze user overlap between patterns.

Checks overlap of users between different patterns.
"""
import pandas as pd
from pathlib import Path
import argparse
from src.config import load_config


def analyze_overlap(df: pd.DataFrame, pattern1: str, pattern2: str) -> dict:
    """Analyze overlap between two patterns."""
    p1_users = set(df[df['regex_type'] == pattern1]['author'].unique())
    p2_users = set(df[df['regex_type'] == pattern2]['author'].unique())
    
    overlap = p1_users & p2_users
    p1_only = p1_users - p2_users
    p2_only = p2_users - p1_users
    
    return {
        'pattern1': pattern1,
        'pattern2': pattern2,
        'pattern1_count': len(p1_users),
        'pattern2_count': len(p2_users),
        'overlap_count': len(overlap),
        'pattern1_only_count': len(p1_only),
        'pattern2_only_count': len(p2_only),
        'overlap_pct_of_p1': 100 * len(overlap) / len(p1_users) if len(p1_users) > 0 else 0,
        'overlap_pct_of_p2': 100 * len(overlap) / len(p2_users) if len(p2_users) > 0 else 0,
        'overlap_users': list(overlap)
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze user overlap between patterns")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML file")
    parser.add_argument("--pattern1", type=str, default="pattern_1", help="First pattern to compare")
    parser.add_argument("--pattern2", type=str, default="pattern_2", help="Second pattern to compare")
    parser.add_argument("--input", type=str, default="data/interim/unique_users_patterns_1_2_3_4_5_6_9.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for overlap users (optional)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    if 'regex_type' not in df.columns or 'author' not in df.columns:
        print("Error: Required columns 'regex_type' and 'author' not found")
        return
    
    if args.pattern1 not in df['regex_type'].values:
        print(f"Warning: {args.pattern1} not found in data")
    
    if args.pattern2 not in df['regex_type'].values:
        print(f"Warning: {args.pattern2} not found in data")
    
    result = analyze_overlap(df, args.pattern1, args.pattern2)
    
    print(f"\n{'='*60}")
    print(f"Overlap Analysis: {args.pattern1} vs {args.pattern2}")
    print(f"{'='*60}")
    print(f"{args.pattern1} unique users: {result['pattern1_count']:,}")
    print(f"{args.pattern2} unique users: {result['pattern2_count']:,}")
    print(f"\nOverlap (users in both): {result['overlap_count']:,}")
    print(f"{args.pattern1} only: {result['pattern1_only_count']:,}")
    print(f"{args.pattern2} only: {result['pattern2_only_count']:,}")
    print(f"\nOverlap percentage (of {args.pattern1}): {result['overlap_pct_of_p1']:.2f}%")
    print(f"Overlap percentage (of {args.pattern2}): {result['overlap_pct_of_p2']:.2f}%")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        overlap_df = pd.DataFrame({
            'author': result['overlap_users']
        })
        overlap_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved overlap users list to {output_path}")


if __name__ == "__main__":
    main()


