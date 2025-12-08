from pathlib import Path
from datetime import datetime
from src.config import load_config
from src.preprocess import (
    filter_deleted_authors,
    add_timestamp_columns,
    fix_removed_posts_selftext,
    concatenate_title_selftext,
    add_matched_pattern_moon2,
    add_normalized_phrase,
    add_uncertainty_flag,
    add_offset_from_cd1_by_pattern,
    sample_posts_by_phrase,
)
import pandas as pd


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    
    raw_dir = Path(cfg["paths"]["raw"])
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    moon2_patterns = cfg["patterns"]["moon2"]
    
    df = pd.read_excel(raw_dir / "moon2.xlsx")
    df = filter_deleted_authors(df)
    df = add_timestamp_columns(df)
    df = fix_removed_posts_selftext(df)
    df = concatenate_title_selftext(df)
    df = add_matched_pattern_moon2(df, moon2_patterns)
    df = add_normalized_phrase(df)
    df = add_uncertainty_flag(df)
    df = add_offset_from_cd1_by_pattern(df)
    # Temporarily commented to debug: df = df[df['offset_from_cd1'].notna()]
    
    df['source_cut'] = 'moon2'
    
    df_output = df.drop(columns=['matched_phrase'], errors='ignore')
    
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    output_path = interim_dir / f"moon2_with_uncertainty_{timestamp}.csv"
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
        sample_path = interim_dir / f"moon2_validation_sample_{timestamp}.csv"
        sample_df[sample_cols].to_csv(sample_path, index=False, encoding='utf-8-sig')
    else:
        sample_path = None

    print(f"Processed {len(df)} posts from moon2.xlsx")
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
    args = ap.parse_args()
    main(args.config)

