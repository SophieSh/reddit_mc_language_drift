from pathlib import Path
from datetime import datetime, timezone
from src.config import load_config
from src.preprocess import (
    filter_deleted_authors,
    add_timestamp_columns,
    fix_removed_posts_selftext,
    concatenate_title_selftext,
    add_matched_pattern_moon3,
    add_normalized_phrase,
    add_uncertainty_flag,
    add_offset_from_cd1_by_pattern,
    sample_posts_by_phrase,
)
import pandas as pd


IRRELEVANT_SUBREDDITS_MOON3 = {
    'Reduction',
    'tummytucksurgery',
    'PlasticSurgery',
    'TopSurgery',
    'hysterectomy',
    'prebreastreduction',
    'tummytuckinfo',
    'SonoBello',
}


def main(cfg_path: str, test_sample: int | None = None):
    cfg = load_config(cfg_path)
    
    raw_dir = Path(cfg["paths"]["raw"])
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    moon3_patterns = cfg["patterns"]["moon3"]
    moon3_anchors = cfg["paths"]["files"]["moon3_anchors"]
    
    # Create filtered file if it doesn't exist
    filtered_path = interim_dir / "moon3_filtered.csv"
    if not filtered_path.exists():
        print("Filtered file not found. Creating it from raw data...")
        print(f"Loading {moon3_anchors}...")
        df = pd.read_csv(raw_dir / moon3_anchors, encoding='latin-1')
        print(f"Loaded {len(df)} posts")
        
        df = filter_deleted_authors(df)
        print(f"After removing deleted authors: {len(df)} posts")
        
        before = len(df)
        df = df[~df['subreddit'].isin(IRRELEVANT_SUBREDDITS_MOON3)]
        after = len(df)
        print(f"Filtered out {before - after} posts from irrelevant surgery subreddits")
        print(f"Remaining: {after} posts")
        
        df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
        print(f"Saved filtered data to {filtered_path}\n")
    else:
        print(f"Using existing filtered file: {filtered_path}")
    
    # Read from filtered file
    print(f"Loading filtered data from {filtered_path}...")
    df = pd.read_csv(filtered_path, encoding='utf-8-sig')
    print(f"Loaded {len(df)} posts")
    
    # For testing: sample a subset
    if test_sample is not None:
        print(f"\nSampling {test_sample} posts for testing...")
        df = df.sample(n=min(test_sample, len(df)), random_state=cfg['seed']).reset_index(drop=True)
        print(f"Using {len(df)} posts for preprocessing")
    df = add_timestamp_columns(df, add_date_string=True)
    df = fix_removed_posts_selftext(df)
    df = concatenate_title_selftext(df)
    df = add_matched_pattern_moon3(df, moon3_patterns)
    df = add_normalized_phrase(df)
    df = add_uncertainty_flag(df)
    df = add_offset_from_cd1_by_pattern(df)
    # Temporarily commented to debug: df = df[df['offset_from_cd1'].notna()]
    
    df['source_cut'] = 'moon3'
    
    df_output = df.drop(columns=['matched_phrase'], errors='ignore')
    
    files_cfg = cfg["paths"]["files"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = interim_dir / f"{files_cfg['moon3_preprocessed']}_{timestamp}.csv"
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    total_uncertain = df['has_uncertainty'].sum()
    phrase_summary = df.groupby('matched_phrase_norm').agg({
        'matched_phrase_norm': 'size',
        'has_uncertainty': 'sum'
    }).rename(columns={'matched_phrase_norm': 'total', 'has_uncertainty': 'uncertain'})
    phrase_summary = phrase_summary.sort_values('total', ascending=False)
    
    sample_df = sample_posts_by_phrase(
        df,
        phrase_col='regex_type',
        sample_size=30,
        random_state=cfg['seed'],
    )
    if not sample_df.empty:
        sample_cols = [
            'regex_type',
            'matched_phrase_norm',
            'sample_index',
            'has_uncertainty',
            'matched_sentence',
            'title',
            'text',
            'author',
            'subreddit',
            'permalink',
            'ts_date',
            'offset_from_cd1',
        ]
        sample_path = interim_dir / f"moon3_validation_sample_{timestamp}.csv"
        sample_df[sample_cols].to_csv(sample_path, index=False, encoding='utf-8-sig')
    else:
        sample_path = None

    print(f"\nProcessed {len(df)} posts from {moon3_anchors} (after filtering)")
    print(f"Found {len(phrase_summary)} unique matched phrases")
    print(f"Total uncertain posts: {total_uncertain} ({100*total_uncertain/len(df):.1f}%)")
    print(f"\nTop 10 phrases by count:")
    print(phrase_summary.head(10).to_string())
    print(f"\nSaved to {output_path}")
    if sample_path:
        print(f"Saved validation sample ({len(sample_df)} rows) to {sample_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--test-sample", type=int, default=None, 
                    help="Sample N posts for testing (default: use all)")
    args = ap.parse_args()
    main(args.config, test_sample=args.test_sample)

