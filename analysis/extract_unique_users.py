"""Extract unique users for patterns 1,2,3,4,5,6,9 and pattern_8 separately.

Loads preprocessed files and creates two lists:
1. Unique users for patterns 1, 2, 3, 4, 5, 6, 9
2. Unique users for pattern 8 only
"""
import pandas as pd
from pathlib import Path
import argparse
from src.config import load_config
from src.utils import load_latest_preprocessed_file


def extract_users_for_patterns(df: pd.DataFrame, moon_type: str, target_patterns: list[str]) -> list[dict]:
    """Extract users for specified patterns."""
    users = []
    
    if 'regex_type' not in df.columns or 'author' not in df.columns:
        return users
    
    df_filtered = df[df['regex_type'].isin(target_patterns)].copy()
    
    if len(df_filtered) == 0:
        return users
    
    unique_users = df_filtered.groupby('regex_type')['author'].unique().reset_index()
    unique_users['moon_type'] = moon_type
    
    for _, row in unique_users.iterrows():
        for user in row['author']:
            users.append({
                'moon_type': moon_type,
                'regex_type': row['regex_type'],
                'author': user
            })
    
    return users


def main():
    parser = argparse.ArgumentParser(description="Extract unique users for patterns 1,2,3,4,5,6,9 and pattern_8")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    pattern_files = {
        "moon1": "moon_with_uncertainty*.csv",
        "moon2": "moon2_with_uncertainty*.csv",
        "moon3": "moon3_with_uncertainty*.csv",
    }
    
    patterns_1_2_3_4_5_6_9 = ['pattern_1', 'pattern_2', 'pattern_3', 'pattern_4', 'pattern_5', 'pattern_6', 'pattern_9']
    pattern_8 = ['pattern_8']
    
    users_patterns_1_6_9 = []
    users_pattern_8 = []
    
    for moon_type, pattern in pattern_files.items():
        df = load_latest_preprocessed_file(interim_dir, pattern)
        if df is None:
            print(f"Warning: No preprocessed file found for {moon_type}")
            continue
        
        if 'regex_type' not in df.columns:
            print(f"Warning: regex_type column missing in {moon_type}, skipping")
            continue
        
        if 'author' not in df.columns:
            print(f"Warning: author column missing in {moon_type}, skipping")
            continue
        
        users_1_6_9 = extract_users_for_patterns(df, moon_type, patterns_1_2_3_4_5_6_9)
        users_8 = extract_users_for_patterns(df, moon_type, pattern_8)
        
        users_patterns_1_6_9.extend(users_1_6_9)
        users_pattern_8.extend(users_8)
        
        if users_1_6_9:
            df_filtered = df[df['regex_type'].isin(patterns_1_2_3_4_5_6_9)].copy()
            pattern_counts = df_filtered.groupby('regex_type')['author'].nunique().reset_index()
            pattern_counts.columns = ['regex_type', 'unique_users']
            pattern_counts['moon_type'] = moon_type
            print(f"\n{moon_type} - patterns 1,2,3,4,5,6,9:")
            print(pattern_counts.to_string(index=False))
        
        if users_8:
            df_filtered = df[df['regex_type'] == 'pattern_8'].copy()
            pattern_counts = df_filtered.groupby('regex_type')['author'].nunique().reset_index()
            pattern_counts.columns = ['regex_type', 'unique_users']
            pattern_counts['moon_type'] = moon_type
            print(f"\n{moon_type} - pattern_8:")
            print(pattern_counts.to_string(index=False))
    
    output_path_1_6_9 = interim_dir / "unique_users_patterns_1_2_3_4_5_6_9.csv"
    output_path_8 = interim_dir / "unique_users_pattern_8.csv"
    
    if users_patterns_1_6_9:
        users_df_1_6_9 = pd.DataFrame(users_patterns_1_6_9)
        unique_users_all_1_6_9 = users_df_1_6_9['author'].unique()
        users_df_1_6_9.to_csv(output_path_1_6_9, index=False, encoding='utf-8-sig')
        
        summary_df = users_df_1_6_9.groupby(['moon_type', 'regex_type']).size().reset_index(name='count')
        summary_df = summary_df.sort_values(['moon_type', 'regex_type'])
        
        print(f"\n{'='*60}")
        print(f"Patterns 1,2,3,4,5,6,9:")
        print(f"Total unique users: {len(unique_users_all_1_6_9)}")
        print(f"Total rows: {len(users_df_1_6_9)}")
        print(f"\nSummary by pattern:")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to {output_path_1_6_9}")
    else:
        print("\nNo users found for patterns 1,2,3,4,5,6,9")
    
    if users_pattern_8:
        users_df_8 = pd.DataFrame(users_pattern_8)
        unique_users_all_8 = users_df_8['author'].unique()
        users_df_8.to_csv(output_path_8, index=False, encoding='utf-8-sig')
        
        summary_df = users_df_8.groupby(['moon_type', 'regex_type']).size().reset_index(name='count')
        summary_df = summary_df.sort_values(['moon_type', 'regex_type'])
        
        print(f"\n{'='*60}")
        print(f"Pattern 8:")
        print(f"Total unique users: {len(unique_users_all_8)}")
        print(f"Total rows: {len(users_df_8)}")
        print(f"\nSummary by pattern:")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to {output_path_8}")
    else:
        print("\nNo users found for pattern_8")


if __name__ == "__main__":
    main()

