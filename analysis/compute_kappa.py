"""Compute Cohen's kappa for inter-annotator agreement.

Loads validation files with labels and computes kappa statistics per pattern type.
"""
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
import numpy as np


def compute_kappa_for_pattern(df: pd.DataFrame, pattern: str) -> dict | None:
    """Compute Cohen's kappa for a specific pattern.
    
    Args:
        df: DataFrame with original_label and eyt_label columns
        pattern: Pattern name (e.g., 'pattern_1')
    
    Returns:
        Dictionary with pattern, n_samples, kappa, or None if insufficient data
    """
    pattern_df = df[df['regex_type'] == pattern].copy()
    
    if len(pattern_df) < 2:
        return None
    
    original_labels = pattern_df['original_label']
    eyt_labels = pattern_df['eyt_label']
    
    observed_agreement = (original_labels == eyt_labels).sum() / len(original_labels)
    
    try:
        kappa = cohen_kappa_score(original_labels, eyt_labels)
        total = len(original_labels)
        all_labels = set(original_labels) | set(eyt_labels)
        expected_agreement = sum(
            (original_labels == label).sum() * (eyt_labels == label).sum() 
            for label in all_labels
        ) / (total * total)
    except ValueError as e:
        print(f"  Warning: Could not compute kappa for {pattern}: {e}")
        return None
    
    disagreements = (original_labels != eyt_labels).sum()
    
    disagreement_pairs = []
    if disagreements > 0:
        disagree_mask = original_labels != eyt_labels
        for orig, eyt in zip(original_labels[disagree_mask], eyt_labels[disagree_mask]):
            disagreement_pairs.append(f"{orig} vs {eyt}")
    
    return {
        'pattern': pattern,
        'n_samples': len(pattern_df),
        'kappa': kappa,
        'observed_agreement': observed_agreement,
        'expected_agreement': expected_agreement,
        'disagreements': disagreements,
        'disagreement_pairs': disagreement_pairs,
    }


def compute_kappa_for_file(file_path: Path) -> pd.DataFrame:
    """Load validation file and compute kappa for each pattern.
    
    Returns:
        DataFrame with columns: pattern, n_samples, kappa
    """
    print(f"Loading {file_path.name}...")
    df = pd.read_excel(file_path)
    
    if 'regex_type' not in df.columns:
        print(f"  Warning: regex_type column missing")
        return pd.DataFrame()
    
    if 'original_label' not in df.columns or 'eyt_label' not in df.columns:
        print(f"  Warning: label columns missing")
        return pd.DataFrame()
    
    patterns = sorted(df['regex_type'].dropna().unique())
    print(f"  Found {len(patterns)} patterns: {patterns}")
    
    results = []
    for pattern in patterns:
        result = compute_kappa_for_pattern(df, pattern)
        if result is not None:
            results.append(result)
            print(f"\n  Pattern: {pattern}")
            print(f"    Samples: {result['n_samples']}")
            print(f"    Disagreements: {result['disagreements']}")
            if result['disagreements'] > 0:
                from collections import Counter
                pair_counts = Counter(result['disagreement_pairs'])
                print(f"    Disagreement breakdown:")
                for pair, count in pair_counts.most_common():
                    print(f"      {pair}: {count}")
            print(f"    Observed agreement: {result['observed_agreement']:.3f}")
            print(f"    Expected agreement (by chance): {result['expected_agreement']:.3f}")
            print(f"    Cohen's κ: {result['kappa']:.3f}")
            if result['kappa'] < 0.2 and result['observed_agreement'] > 0.8:
                print(f"    Note: Low κ despite high agreement - likely due to class imbalance")
                print(f"         (both annotators favor the same label, making chance agreement high)")
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for validation labels")
    parser.add_argument("--validation-dir", type=str, default="data/validation", help="Directory with validation files")
    parser.add_argument("--pattern-counts", type=str, default="data/validation/pattern_counts.xlsx", help="Path to pattern counts Excel file")
    parser.add_argument("--output", type=str, default="data/validation/kappa_results.xlsx", help="Output file path")
    args = parser.parse_args()
    
    validation_dir = Path(args.validation_dir)
    pattern_counts_path = Path(args.pattern_counts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_label_files = [f for f in validation_dir.glob("*_validation_with_labels*.xlsx") if not f.name.startswith("~$")]
    
    label_files = []
    for moon_type in ["moon1", "moon2", "moon3"]:
        moon_files = [f for f in all_label_files if moon_type in f.name]
        if moon_files:
            most_recent = max(moon_files, key=lambda x: x.stat().st_mtime)
            label_files.append(most_recent)
            print(f"Using most recent {moon_type} file: {most_recent.name}")
    
    if len(label_files) == 0:
        print(f"No validation label files found in {validation_dir}")
        return
    
    all_results = []
    
    for file_path in label_files:
        if "moon1" in file_path.name:
            moon_type = "moon1"
        elif "moon2" in file_path.name:
            moon_type = "moon2"
        elif "moon3" in file_path.name:
            moon_type = "moon3"
        else:
            continue
        
        results_df = compute_kappa_for_file(file_path)
        if len(results_df) > 0:
            results_df.insert(0, 'moon_type', moon_type)
            all_results.append(results_df)
        print()
    
    if not all_results:
        print("No kappa results computed")
        return
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    if pattern_counts_path.exists():
        pattern_counts_df = pd.read_excel(pattern_counts_path)
        combined_results = combined_results.rename(columns={'pattern': 'regex_type'})
        merge_cols = ['moon_type', 'regex_type', 'count']
        if 'unique_users' in pattern_counts_df.columns:
            merge_cols.append('unique_users')
        combined_results = combined_results.merge(
            pattern_counts_df[merge_cols],
            on=['moon_type', 'regex_type'],
            how='left'
        )
        combined_results = combined_results.rename(columns={'count': 'total_posts'})
    else:
        print(f"Warning: Pattern counts file not found at {pattern_counts_path}")
        print("  Run analysis/count_patterns.py to generate it")
    
    results_for_excel = combined_results.drop(columns=['disagreement_pairs'], errors='ignore').copy()
    
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    output_path_with_timestamp = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"
    results_for_excel.to_excel(output_path_with_timestamp, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY BY PATTERN")
    print("="*60)
    for _, row in combined_results.iterrows():
        print(f"\n{row['moon_type']} - {row['regex_type']}:")
        if pd.notna(row.get('total_posts')):
            print(f"  Total posts: {int(row['total_posts'])}")
        if pd.notna(row.get('unique_users')):
            print(f"  Unique users: {int(row['unique_users'])}")
        print(f"  Validation samples: {row['n_samples']}")
        print(f"  Disagreements: {row['disagreements']}")
        if row['disagreements'] > 0 and 'disagreement_pairs' in row and row['disagreement_pairs']:
            from collections import Counter
            pair_counts = Counter(row['disagreement_pairs'])
            print(f"  Disagreement breakdown:")
            for pair, count in pair_counts.most_common():
                print(f"    {pair}: {count}")
        print(f"  Observed agreement: {row['observed_agreement']:.3f}")
        print(f"  Expected agreement (by chance): {row['expected_agreement']:.3f}")
        print(f"  Cohen's κ: {row['kappa']:.3f}")
        if row['kappa'] < 0.2 and row['observed_agreement'] > 0.8:
            print(f"  Note: Low κ despite high agreement - likely due to class imbalance")
            print(f"       (both annotators favor the same label, making chance agreement high)")
    
    print(f"\nSaved kappa results to {output_path_with_timestamp.name}")


if __name__ == "__main__":
    main()

