"""Count posts per pattern from preprocessed data.

Loads preprocessed files and counts how many posts match each pattern type.
"""
import pandas as pd
from pathlib import Path
import argparse
import yaml
from src.preprocess import count_posts_by_pattern
from src.utils import load_latest_preprocessed_file


def main():
    parser = argparse.ArgumentParser(description="Count posts per pattern from preprocessed data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output", type=str, default="data/validation/pattern_counts.xlsx", help="Output file path")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    interim_dir = Path(config["paths"]["interim"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_counts = []
    
    pattern_files = {
        "moon1": "moon_with_uncertainty_*.csv",
        "moon2": "moon2_with_uncertainty_*.csv",
        "moon3": "moon3_with_uncertainty_*.csv",
    }
    
    for moon_type, pattern in pattern_files.items():
        df = load_latest_preprocessed_file(interim_dir, pattern)
        if df is None:
            print(f"Warning: No preprocessed file found for {moon_type}")
            continue
        
        counts_df = count_posts_by_pattern(df)
        if len(counts_df) > 0:
            counts_df.insert(0, 'moon_type', moon_type)
            all_counts.append(counts_df)
            print(f"{moon_type}: Found {len(counts_df)} patterns, {counts_df['count'].sum()} total posts")
    
    if not all_counts:
        print("No pattern counts computed")
        return
    
    combined_counts = pd.concat(all_counts, ignore_index=True)
    combined_counts.to_excel(output_path, index=False)
    print(f"\nSaved pattern counts to {output_path}")
    print("\nPattern counts:")
    print(combined_counts.to_string(index=False))


if __name__ == "__main__":
    main()

