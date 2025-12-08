"""Create validation labels from EYT-annotated Excel files.

Loads validation samples, normalizes offsets, and creates original_label and eyt_label
columns for inter-annotator agreement analysis.
"""
import pandas as pd
from pathlib import Path
import argparse
import yaml
from datetime import datetime
from src.preprocess import extract_dpo_from_phrase


def normalize_offsets_in_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize offsets for 'last night' and 'a couple of days ago' to 1."""
    df = df.copy()
    
    if 'matched_phrase_norm' not in df.columns:
        return df
    
    mask_last_night = df['matched_phrase_norm'].astype(str).str.contains('last night', case=False, na=False)
    mask_couple_days = df['matched_phrase_norm'].astype(str).str.contains('a couple of days ago', case=False, na=False)
    
    df.loc[mask_last_night, 'offset_from_cd1'] = 1
    df.loc[mask_couple_days, 'offset_from_cd1'] = 1
    
    return df


def make_original_label(row: pd.Series, regex_type: str) -> int:
    """Create original_label from offset_from_cd1 or DPO.
    
    Returns:
        int: Clean, known label (offset or DPO number), or -1 for uncertain/ambiguous
    """
    has_uncertainty = row.get('has_uncertainty', False)
    
    if regex_type == "pattern_7":
        matched_phrase_norm = row.get('matched_phrase_norm')
        dpo = extract_dpo_from_phrase(matched_phrase_norm)
        
        if dpo is not None and not has_uncertainty:
            return int(dpo)
        else:
            return -1
    else:
        offset = row.get('offset_from_cd1')
        
        if pd.notna(offset) and not has_uncertainty:
            try:
                return int(float(offset))
            except (ValueError, TypeError):
                return -1
        else:
            return -1


def parse_date_with_year(date_str: str) -> datetime | None:
    """Parse date string in MM/DD/YYYY format.
    
    Returns datetime with actual year, or None if can't parse.
    """
    if pd.isna(date_str):
        return None
    
    try:
        if isinstance(date_str, pd.Timestamp):
            return date_str.to_pydatetime()
        
        date_str_clean = str(date_str).strip()
        dt = pd.to_datetime(date_str_clean, format='%m/%d/%Y', errors='coerce')
        if pd.notna(dt):
            return dt.to_pydatetime()
        return None
    except (ValueError, TypeError, AttributeError):
        return None


def parse_date_ignore_year(date_str: str) -> datetime | None:
    """Parse date string in MM/DD/YYYY format, ignoring year.
    
    Returns datetime with year=2000 (placeholder), or None if can't parse.
    """
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    try:
        if isinstance(date_str, pd.Timestamp):
            dt = date_str.to_pydatetime()
            return datetime(year=2000, month=dt.month, day=dt.day)
        
        dt = datetime.strptime(date_str, '%m/%d/%Y')
        return datetime(year=2000, month=dt.month, day=dt.day)
    except (ValueError, TypeError, AttributeError):
        return None


def compute_offset_from_date(post_date: pd.Timestamp, eyt_date: datetime) -> int | None:
    """Compute offset in days from EYT's date to post date, ignoring year.
    
    If EYT's date is later in the calendar year than post date, assumes EYT's date
    is from the previous year.
    
    Args:
        post_date: Post publication timestamp
        eyt_date: EYT's date (with year=2000 placeholder)
    
    Returns:
        Days difference (always positive), or None if can't compute
    """
    if post_date is None or pd.isna(post_date) or eyt_date is None:
        return None
    
    post_dt = post_date.to_pydatetime().replace(tzinfo=None)
    post_date_only = datetime(year=2000, month=post_dt.month, day=post_dt.day)
    
    days_diff = (post_date_only.date() - eyt_date.date()).days
    
    if days_diff < 0:
        days_diff += 365
    
    if days_diff < 0 or days_diff > 365:
        return None
    
    return days_diff


def make_eyt_label(row: pd.Series, original_label: int, regex_type: str) -> int:
    """Create eyt_label from EYT's annotation.
    
    For pattern_3: Parses EYT_raw as date (MM/DD/YYYY) and computes offset from post date.
    For other patterns: Uses EYT_raw as integer value.
    
    Returns:
        int: Clean, known label (offset or DPO number), or -1 for uncertain/ambiguous/unknown
    """
    eyt_raw = row.get('EYT_raw')
    
    try:
        if isinstance(eyt_raw, pd.Series):
            eyt_raw = eyt_raw.iloc[0] if len(eyt_raw) > 0 else None
        
        if eyt_raw is None or pd.isna(eyt_raw):
            return original_label
        
        eyt_raw_str = str(eyt_raw).strip()
    except (ValueError, AttributeError, TypeError):
        return original_label
    
    if eyt_raw_str == '' or eyt_raw_str == 'nan' or eyt_raw_str.lower() == 'none':
        return original_label
    
    if eyt_raw_str == '?':
        return -1
    
    if regex_type == "pattern_3":
        try:
            date_val = pd.to_datetime(eyt_raw_str, format='%Y-%m-%d', errors='coerce')
            if pd.isna(date_val):
                date_val = pd.to_datetime(eyt_raw_str, format='%m/%d/%Y', errors='coerce')
            if pd.isna(date_val):
                date_val = pd.to_datetime(eyt_raw_str, errors='coerce')
            
            if pd.notna(date_val):
                post_timestamp = row.get('ts_date') or row.get('ts_utc')
                
                if post_timestamp is not None and pd.notna(post_timestamp):
                    try:
                        if isinstance(post_timestamp, pd.Series):
                            post_timestamp = post_timestamp.iloc[0] if len(post_timestamp) > 0 else None
                        if post_timestamp is None or pd.isna(post_timestamp):
                            return -1
                        if hasattr(post_timestamp, 'to_pydatetime'):
                            post_dt = post_timestamp.to_pydatetime().replace(tzinfo=None)
                        else:
                            post_dt = pd.to_datetime(post_timestamp).to_pydatetime().replace(tzinfo=None)
                        date_dt = date_val.to_pydatetime()
                        days_diff = (post_dt.date() - date_dt.date()).days
                        if days_diff >= 0:
                            return days_diff
                    except (ValueError, TypeError, AttributeError):
                        return -1
        except (ValueError, TypeError, AttributeError):
            pass
    
    try:
        eyt_num = float(eyt_raw)
        if eyt_num.is_integer():
            return int(eyt_num)
        else:
            return -1
    except (ValueError, TypeError, AttributeError):
        return -1


def process_validation_file(
    file_path: Path,
    moon_type: str,
    output_dir: Path,
    selected_patterns: list[str],
) -> None:
    """Process a single validation file and create labels."""
    print(f"Loading {file_path.name}...")
    df = pd.read_excel(file_path)
    
    print(f"  Original shape: {df.shape}")
    
    df = normalize_offsets_in_validation(df)
    
    if selected_patterns is None:
        df_filtered = df.copy()
        print(f"  No pattern filtering (all rows kept)")
    elif 'regex_type' not in df.columns:
        print(f"  Warning: regex_type column missing, skipping pattern filtering")
        df_filtered = df
    else:
        df_filtered = df[df['regex_type'].isin(selected_patterns)].copy()
        print(f"  After pattern filtering: {df_filtered.shape}")
    
    if len(df_filtered) == 0:
        print(f"  Warning: No rows match selected patterns")
        return
    
    df_filtered['original_label'] = df_filtered.apply(
        lambda row: make_original_label(row, row.get('regex_type', '')),
        axis=1
    )
    
    df_filtered['eyt_label'] = df_filtered.apply(
        lambda row: make_eyt_label(row, row['original_label'], row.get('regex_type', '')),
        axis=1
    )
    
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    output_file = output_dir / f"{moon_type}_validation_with_labels_{timestamp}.xlsx"
    df_filtered.to_excel(output_file, index=False)
    print(f"  Saved to {output_file.name}")
    print(f"  Final shape: {df_filtered.shape}")


def main():
    parser = argparse.ArgumentParser(description="Create validation labels from EYT annotations")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--validation-dir", type=str, default="data/validation", help="Directory with validation files")
    parser.add_argument("--output-dir", type=str, default="data/validation", help="Output directory for labeled files")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eyt_files = sorted([f for f in validation_dir.glob("*_EYT.xlsx") if not f.name.startswith("~$")])
    
    if len(eyt_files) == 0:
        print(f"No EYT files found in {validation_dir}")
        return
    
    pattern_selection = {
        "moon1": ["pattern_1"],
        "moon2": ["pattern_2", "pattern_3", "pattern_4", "pattern_5", "pattern_6"],
        "moon3": ["pattern_7", "pattern_9"],
    }
    
    for file_path in eyt_files:
        print(f"\nProcessing {file_path.name}...")
        if "moon1" in file_path.name:
            moon_type = "moon1"
            selected_patterns = pattern_selection["moon1"]
        elif "moon2" in file_path.name:
            moon_type = "moon2"
            selected_patterns = pattern_selection["moon2"]
        elif "moon3" in file_path.name:
            moon_type = "moon3"
            selected_patterns = pattern_selection["moon3"]
        else:
            print(f"  Warning: Could not determine moon type for {file_path.name}, skipping")
            continue
        
        try:
            process_validation_file(file_path, moon_type, output_dir, selected_patterns)
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

