"""Functions for FFT and Lomb-Scargle periodogram analysis."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

from src.config import (
    MIN_DATA_POINTS,
    EPSILON,
    DEFAULT_PERIOD_MIN,
    DEFAULT_PERIOD_MAX,
    AVG_DAYS_PER_MONTH,
)



def normalize_values(
    values: np.ndarray, 
    method: str = "zscore",
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Normalize values using specified method.
    
    Args:
        values: Array of values to normalize
        method: Normalization method ("zscore", "minmax", or "none")
        epsilon: Small epsilon for numerical stability (default: 1e-10)
        
    Returns:
        Normalized array
    """
    # Ensure epsilon is float64 for numpy operations
    epsilon_val = np.float64(epsilon)
    
    if method == "zscore":
        return (values - np.mean(values)) / (np.std(values) + epsilon_val)
    elif method == "minmax":
        v_min, v_max = np.min(values), np.max(values)
        return (values - v_min) / (v_max - v_min + epsilon_val)
    elif method == "none":
        return values
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def validate_timeline_data(
    offsets: np.ndarray,
    values: np.ndarray,
    min_points: int = MIN_DATA_POINTS,
    min_unique_days: int = 10,
    min_span_days: int = 21,
    epsilon: float = EPSILON,
) -> tuple[bool, str | None]:
    """Validate timeline data quality.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be numeric)
        min_points: Minimum number of data points required (default: 10)
        min_unique_days: Minimum number of unique days required (default: 10)
        min_span_days: Minimum time span required (days, default: 24)
        epsilon: Small epsilon for numerical stability (default: 1e-10)
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if len(offsets) < min_points:
        return False, f"Insufficient data points: {len(offsets)} < {min_points}"
    
    if len(np.unique(offsets)) < min_unique_days:
        return False, f"Insufficient unique days: {len(np.unique(offsets))} < {min_unique_days}"
    
    span = offsets.max() - offsets.min()
    if span < min_span_days:
        return False, f"Time span too short: {span} < {min_span_days} days"
    
    # Ensure values are numeric (convert if needed to handle dtype issues)
    # This handles cases where values come from pandas DataFrame with object dtype
    if not np.issubdtype(values.dtype, np.number):
        # Convert object/string array to numeric
        values = pd.to_numeric(values, errors='coerce')
        values = np.asarray(values, dtype=np.float64)
    else:
        # Ensure it's float64 for consistency
        values = np.asarray(values, dtype=np.float64)
    
    # Filter out NaN values before checking std
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return False, "All values are NaN"
    
    if len(valid_values) < 2:
        return False, "Insufficient valid values for std calculation"
    
    # Ensure epsilon is float64 for comparison
    epsilon_val = np.float64(epsilon)
    if np.std(valid_values) < epsilon_val:
        return False, "Constant values (no variation)"
    
    return True, None


def is_periodicity_significant(
    period: float | None,
    snr: float,
    span_days: float,
    min_snr: float = 1.5,
    min_cycles: float = 1.5,
) -> tuple[bool, str | None]:
    """Check if detected periodicity is statistically significant.
    
    Args:
        period: Detected period (days) or None
        snr: Signal-to-noise ratio
        span_days: Total time span of data (days)
        min_snr: Minimum SNR threshold (default 1.5)
        min_cycles: Minimum number of cycles required (default 1.5)
        
    Returns:
        Tuple of (is_significant, reason_if_not)
    """
    if period is None:
        return False, "No period detected"
    
    if snr < min_snr:
        return False, f"SNR too low: {snr:.2f} < {min_snr:.2f}"
    
    n_cycles = span_days / period
    if n_cycles < min_cycles:
        return False, f"Insufficient cycles: {n_cycles:.2f} < {min_cycles:.2f}"
    
    return True, None


def aggregate_by_day(
    df: pd.DataFrame,
    offset_col: str = "offset_from_cd1",
    value_col: str = "_feature_value",
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Aggregate feature values by day (offset).
    
    Args:
        df: DataFrame with offset and value columns
        offset_col: Name of offset column
        value_col: Name of value column to aggregate
        agg_func: Aggregation function ("mean", "median", etc.)
        
    Returns:
        DataFrame with columns: offset_col, feature_mean, n_entries
    """
    daily_agg = df.groupby(offset_col)[value_col].agg([agg_func, "count"]).reset_index()
    daily_agg.columns = [offset_col, "feature_mean", "n_entries"]
    
    # Filter out NaN values
    daily_agg = daily_agg[daily_agg["feature_mean"].notna()].copy()
    
    return daily_agg


def aggregate_all_features_by_day(
    timeline_df: pd.DataFrame,
    feature_cols: list[str],
    user_col: str = "author",
    time_col: str = "offset_from_cd1",
) -> pd.DataFrame:
    """Aggregate multiple posts per day into one data point by averaging features.
    
    Creates a NEW DataFrame where:
    - Input: Multiple posts per day per user (original timeline with all posts)
    - Output: One row per (user, day) with averaged feature values
    
    Args:
        timeline_df: DataFrame with multiple posts per day, columns include feature_cols
        feature_cols: List of feature column names to aggregate
        user_col: Column name for user identifier
        time_col: Column name for day/offset
    
    Returns:
        NEW DataFrame with columns:
        - user_col, time_col
        - {feature}_mean for each feature (averaged across posts per day)
    
    Example:
        Input timeline: user='Alice', day=5, feature_1=[0.5, 0.6, 0.7] (3 posts on same day)
        Output: NEW row: user='Alice', day=5, feature_1_mean=0.6
    """
    df = timeline_df.copy()
    
    print(f"Aggregating {len(feature_cols)} features by (user, day)...")
    
    # Filter feature columns to only those that exist, are not grouping columns, and are numeric
    valid_feature_cols = []
    for f in feature_cols:
        if f not in df.columns:
            continue
        if f in [user_col, time_col]:
            continue
        # Only include numeric columns (can be aggregated with mean)
        if pd.api.types.is_numeric_dtype(df[f]):
            valid_feature_cols.append(f)
    
    if len(valid_feature_cols) == 0:
        raise ValueError(
            f"No numeric feature columns found in DataFrame. "
            f"Available columns: {df.columns.tolist()}\n"
            f"Feature columns passed: {feature_cols[:10]}..."
        )
    
    # Select only columns we need: grouping columns + feature columns
    # This avoids trying to aggregate non-numeric columns like timestamps, text, etc.
    cols_to_keep = [user_col, time_col] + valid_feature_cols
    
    # Remove duplicates (in case grouping columns were in feature_cols)
    cols_to_keep = list(dict.fromkeys(cols_to_keep))  # Preserves order while removing duplicates
    
    df_subset = df[cols_to_keep].copy()
    
    # Group by user and time, compute mean for each feature
    agg_dict = {feature: "mean" for feature in valid_feature_cols}
    
    if len(agg_dict) == 0:
        raise ValueError(f"None of the feature columns found in DataFrame. Available columns: {df.columns.tolist()}")
    
    daily_agg = df_subset.groupby([user_col, time_col]).agg(agg_dict).reset_index()
    
    # Ensure all aggregated columns are numeric (defensive check)
    for col in daily_agg.columns:
        if col not in [user_col, time_col] and not pd.api.types.is_numeric_dtype(daily_agg[col]):
            print(f"  Warning: Aggregated column {col} is not numeric (dtype: {daily_agg[col].dtype}), converting...")
            daily_agg[col] = pd.to_numeric(daily_agg[col], errors='coerce')
    
    # Rename feature columns to {feature}_mean
    rename_dict = {}
    for feature in feature_cols:
        if feature in daily_agg.columns:
            rename_dict[feature] = f"{feature}_mean"
    
    daily_agg = daily_agg.rename(columns=rename_dict)
    
    print(f"  ✓ Aggregated to {len(daily_agg)} (user, day) combinations")
    
    return daily_agg


def normalize_features_per_user_zscore(
    daily_agg_df: pd.DataFrame,
    feature_mean_cols: list[str],
    user_col: str = "author",
) -> pd.DataFrame:
    """Apply per-user z-score normalization to daily aggregated features.
    
    For each user, for each feature, calculates z-score across all their days.
    Adds {feature}_zscore columns to the DataFrame.
    
    Args:
        daily_agg_df: DataFrame with daily aggregated features (from aggregate_all_features_by_day)
        feature_mean_cols: List of feature column names ending in _mean (e.g., ['feature_1_mean', 'feature_2_mean'])
        user_col: Column name for user identifier
    
    Returns:
        DataFrame with added {feature}_zscore columns
    
    Example:
        Input: user='Alice', day=5, feature_1_mean=0.6
        Output: user='Alice', day=5, feature_1_mean=0.6, feature_1_zscore = (0.6 - mean_Alice_feature1) / std_Alice_feature1
    """
    df = daily_agg_df.copy()
    
    logging.debug(f"Applying per-user z-score normalization to {len(feature_mean_cols)} features...")
    
    for mean_col in feature_mean_cols:
        if mean_col not in df.columns:
            logging.debug(f"  Warning: {mean_col} not found, skipping")
            continue
        
        # Ensure column is numeric (convert if needed)
        original_dtype = df[mean_col].dtype
        if not pd.api.types.is_numeric_dtype(df[mean_col]):
            logging.debug(f"  Warning: {mean_col} is not numeric (dtype: {original_dtype}), converting...")
            df[mean_col] = pd.to_numeric(df[mean_col], errors='coerce')
        
        zscore_col = mean_col.replace("_mean", "_zscore")
        
        # Vectorized per-user normalization using groupby.transform
        user_means = df.groupby(user_col)[mean_col].transform('mean')
        user_stds = df.groupby(user_col)[mean_col].transform('std')
        
        # Debug: Check dtypes before conversion
        if user_stds.dtype == 'object' or str(user_stds.dtype).startswith('<U'):
            logging.debug(f"  ERROR: {mean_col} -> user_stds has dtype {user_stds.dtype}, converting...")
            logging.debug(f"    Sample values: {user_stds.head(5).tolist()}")
            logging.debug(f"    mean_col dtype: {df[mean_col].dtype}")
        
        # Ensure both are numeric Series (defensive check) - convert BEFORE arithmetic
        # Convert to numpy arrays with explicit float64 dtype to avoid pandas dtype issues
        # Use np.float64 explicitly to ensure proper conversion even if Series has object dtype
        user_means_series = pd.to_numeric(user_means, errors='coerce')
        user_stds_series = pd.to_numeric(user_stds, errors='coerce')
        mean_col_series = pd.to_numeric(df[mean_col], errors='coerce')
        
        # Convert to numpy arrays with explicit float64 dtype
        user_means_arr = np.asarray(user_means_series, dtype=np.float64)
        user_stds_arr = np.asarray(user_stds_series, dtype=np.float64)
        mean_col_arr = np.asarray(mean_col_series, dtype=np.float64)
        
        # Ensure EPSILON is float64 for the arithmetic operation
        epsilon_val = np.float64(EPSILON)
        
        # Calculate z-score using numpy arrays (handles dtype issues)
        # Match old behavior: z = (x - μ) / (σ + ε) where ε is small epsilon
        # This matches the original implementation that worked correctly
        with np.errstate(divide='ignore', invalid='ignore'):
            # Old formula: (x - μ) / (σ + ε) - matches original working code
            zscore_arr = (mean_col_arr - user_means_arr) / (user_stds_arr + epsilon_val)
            # Replace non-finite values (inf, -inf, nan) with NaN
            zscore_arr = np.where(np.isfinite(zscore_arr), zscore_arr, np.nan)
        
        df[zscore_col] = zscore_arr
        
        # Set to NaN where std was effectively zero (constant values)
        # This handles cases where user has constant values (std ≈ 0)
        df.loc[user_stds_series < epsilon_val, zscore_col] = np.nan
        
        valid_count = df[zscore_col].notna().sum()
        logging.debug(f"  {mean_col} -> {zscore_col}: {valid_count} valid values")
    
    return df


def run_lombscargle(
    offsets: np.ndarray,
    values: np.ndarray,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    period_resolution: int = 1,
) -> dict:
    """Run Lomb-Scargle periodogram analysis.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_min: Minimum period to search (days)
        period_max: Maximum period to search (days)
        period_resolution: Resolution for period search (days, default 1)
        
    Returns:
        dict with best_period, best_power, peak_to_background_ratio, periods, powers
    """
    periods = np.arange(period_min, period_max + period_resolution, period_resolution)
    frequencies = 1.0 / periods
    
    power = signal.lombscargle(offsets, values, frequencies, normalize=True)
    
    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    best_power = power[best_idx]
    
    # Peak-to-background ratio (higher = stronger signal relative to noise)
    # Note: This is NOT a true SNR, but useful for ranking results
    # TODO: Implement proper statistical significance testing (FAP or permutation tests)
    power_without_peak = np.concatenate([power[:best_idx], power[best_idx+1:]])
    if len(power_without_peak) > 0:
        median_power_background = np.median(power_without_peak)
        # Avoid division by extremely small values that produce huge SNR
        # Cap SNR at reasonable maximum (1000) to prevent extreme outliers
        if median_power_background < EPSILON:
            peak_to_background = min(best_power / EPSILON, 1000.0)
        else:
            peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        # Only one period in range - cannot compute meaningful SNR
        # Cap at reasonable value to avoid extreme outliers
        peak_to_background = min(best_power / EPSILON, 1000.0)
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
        "periods": periods,
        "powers": power,
    }


def run_fft_interpolation(
    offsets: np.ndarray,
    values: np.ndarray,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
) -> dict:
    """Run FFT on interpolated data (no zero-padding).
    
    Interpolates irregular data to daily grid, then runs FFT.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_min: Minimum period to search (days)
        period_max: Maximum period to search (days)
        
    Returns:
        dict with best_period, best_power, peak_to_background, periods, powers
    """
    if len(offsets) < 2:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    offset_min = int(np.floor(offsets.min()))
    offset_max = int(np.ceil(offsets.max()))
    regular_offsets = np.arange(offset_min, offset_max + 1)
    regular_values = np.interp(regular_offsets, offsets, values)
    regular_values = regular_values - np.mean(regular_values)
    
    n = len(regular_values)
    fft_vals = fft(regular_values)
    freqs = fftfreq(n, d=1.0)
    
    positive_freqs = freqs[: n // 2]
    periods = 1.0 / positive_freqs[1:]
    powers = np.abs(fft_vals[: n // 2])[1:] ** 2
    
    mask = (periods >= period_min) & (periods <= period_max)
    periods_filtered = periods[mask]
    powers_filtered = powers[mask]
    
    if len(powers_filtered) == 0:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    best_idx = np.argmax(powers_filtered)
    best_period = round(periods_filtered[best_idx])  # Round to integer days
    best_power = powers_filtered[best_idx]
    
    powers_without_peak = np.concatenate([powers_filtered[:best_idx], powers_filtered[best_idx+1:]])
    if len(powers_without_peak) > 0:
        median_power_background = np.median(powers_without_peak)
        peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        peak_to_background = best_power / EPSILON
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
        "periods": periods_filtered,
        "powers": powers_filtered,
    }


def run_fft_interpolation_integer_periods(
    offsets: np.ndarray,
    values: np.ndarray,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
) -> dict:
    """Run FFT on interpolated data, then evaluate power at exact integer periods.
    
    Similar to run_fft_interpolation, but instead of rounding FFT periods,
    interpolates power at exact integer periods [24, 25, 26, ..., 35].
    This is more biologically appropriate (cycles are integer days) and
    more accurate (no rounding error).
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_min: Minimum period to search (days)
        period_max: Maximum period to search (days)
        
    Returns:
        dict with best_period, best_power, peak_to_background, periods, powers
        periods and powers are at exact integer periods
    """
    if len(offsets) < 2:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    # Step 1: Same as run_fft_interpolation - get FFT spectrum
    offset_min = int(np.floor(offsets.min()))
    offset_max = int(np.ceil(offsets.max()))
    regular_offsets = np.arange(offset_min, offset_max + 1)
    regular_values = np.interp(regular_offsets, offsets, values)
    regular_values = regular_values - np.mean(regular_values)
    
    n = len(regular_values)
    fft_vals = fft(regular_values)
    freqs = fftfreq(n, d=1.0)
    
    positive_freqs = freqs[: n // 2]
    periods_fft = 1.0 / positive_freqs[1:]  # All FFT periods
    powers_fft = np.abs(fft_vals[: n // 2])[1:] ** 2  # All FFT powers
    
    # Step 2: Create integer periods in range [period_min, period_max]
    period_min_int = int(np.ceil(period_min))
    period_max_int = int(np.floor(period_max))
    integer_periods = np.arange(period_min_int, period_max_int + 1, dtype=int)
    
    if len(integer_periods) == 0:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    # Step 3: Interpolate power at exact integer periods
    # Need to handle edge case: periods_fft must be sorted for interpolation
    # Also need to handle case where integer period is outside FFT range
    sort_idx = np.argsort(periods_fft)
    periods_fft_sorted = periods_fft[sort_idx]
    powers_fft_sorted = powers_fft[sort_idx]
    
    # Filter to periods that are within reasonable range for interpolation
    # (interpolate only if integer period is within or near FFT range)
    valid_mask = (
        (integer_periods >= periods_fft_sorted.min()) & 
        (integer_periods <= periods_fft_sorted.max())
    )
    integer_periods_valid = integer_periods[valid_mask]
    
    if len(integer_periods_valid) == 0:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    # Interpolate power at integer periods
    powers_at_integers = np.interp(
        integer_periods_valid,
        periods_fft_sorted,
        powers_fft_sorted
    )
    
    # Step 4: Find maximum among integer periods
    best_idx = np.argmax(powers_at_integers)
    best_period = int(integer_periods_valid[best_idx])  # Already an integer!
    best_power = powers_at_integers[best_idx]
    
    # Step 5: Calculate SNR (peak-to-background)
    powers_without_peak = np.concatenate([
        powers_at_integers[:best_idx],
        powers_at_integers[best_idx+1:]
    ])
    if len(powers_without_peak) > 0:
        median_power_background = np.median(powers_without_peak)
        # Avoid division by extremely small values that produce huge SNR
        # Cap SNR at reasonable maximum (1000) to prevent extreme outliers
        if median_power_background < EPSILON:
            peak_to_background = min(best_power / EPSILON, 1000.0)
        else:
            peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        # Only one period in range - cannot compute meaningful SNR
        # Cap at reasonable value to avoid extreme outliers
        peak_to_background = min(best_power / EPSILON, 1000.0)
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
        "periods": integer_periods_valid.astype(int),
        "powers": powers_at_integers,
    }


def run_fft_zeropad(
    offsets: np.ndarray,
    values: np.ndarray,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
) -> dict:
    """Run FFT with zero-padding for higher frequency resolution.
    
    Interpolates to daily grid, then zero-pads to next power of 2.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_min: Minimum period to search (days)
        period_max: Maximum period to search (days)
        
    Returns:
        dict with best_period, best_power, peak_to_background, periods, powers
    """
    if len(offsets) < 2:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    offset_min = int(np.floor(offsets.min()))
    offset_max = int(np.ceil(offsets.max()))
    regular_offsets = np.arange(offset_min, offset_max + 1)
    regular_values = np.interp(regular_offsets, offsets, values)
    regular_values = regular_values - np.mean(regular_values)
    
    n_original = len(regular_values)
    n_padded = 2 ** int(np.ceil(np.log2(n_original)))
    regular_values_padded = np.pad(regular_values, (0, n_padded - n_original), mode='constant')
    
    n = len(regular_values_padded)
    fft_vals = fft(regular_values_padded)
    freqs = fftfreq(n, d=1.0)
    
    positive_freqs = freqs[: n // 2]
    periods = 1.0 / positive_freqs[1:]
    powers = np.abs(fft_vals[: n // 2])[1:] ** 2
    
    mask = (periods >= period_min) & (periods <= period_max)
    periods_filtered = periods[mask]
    powers_filtered = powers[mask]
    
    if len(powers_filtered) == 0:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    best_idx = np.argmax(powers_filtered)
    best_period = round(periods_filtered[best_idx])  # Round to integer days
    best_power = powers_filtered[best_idx]
    
    powers_without_peak = np.concatenate([powers_filtered[:best_idx], powers_filtered[best_idx+1:]])
    if len(powers_without_peak) > 0:
        median_power_background = np.median(powers_without_peak)
        # Avoid division by extremely small values that produce huge SNR
        # Cap SNR at reasonable maximum (1000) to prevent extreme outliers
        if median_power_background < EPSILON:
            peak_to_background = min(best_power / EPSILON, 1000.0)
        else:
            peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        # Only one period in range - cannot compute meaningful SNR
        # Cap at reasonable value to avoid extreme outliers
        peak_to_background = min(best_power / EPSILON, 1000.0)
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
        "periods": periods_filtered,
        "powers": powers_filtered,
    }


def analyze_user_timeline(
    user_timeline: pd.DataFrame,
    feature_col: str = "text",
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    period_wide_min: float = 10.0,
    period_wide_max: float = 50.0,
    normalize_method: str = "zscore",
    methods: list[str] | None = None,
) -> dict | None:
    """Analyze a single user's timeline with configurable periodicity detection methods.
    
    Available methods:
    - lomb_scargle: Astropy LS with FAP (searches 10-90 days)
    - fft_interpolation: FFT with linear interpolation (searches 24-35 days)
    - fft_zeropad: FFT with zero-padding (searches 24-35 days)
    - fft_interpolation_wide: FFT interpolation wide search (configurable range)
    - fft_zeropad_wide: FFT zero-pad wide search (configurable range)
    
    Args:
        user_timeline: DataFrame with columns including 'offset_from_cd1' and feature_col
        feature_col: Column name for feature values (default: 'text' uses text length)
        period_min: Minimum period for biological relevance (days, default 24)
        period_max: Maximum period for biological relevance (days, default 35)
        period_wide_min: Minimum period for wide FFT search (days, default 10)
        period_wide_max: Maximum period for wide FFT search (days, default 50)
        normalize_method: Method to normalize feature values ("zscore", "minmax", or "none")
        methods: List of method names to run (default: all methods)
        
    Returns:
        dict with analysis results including:
        - n_points, span_days: Data quality metrics
        - method-specific fields (e.g., ls_period, fft_interp_period, etc.)
        Returns None if insufficient data
    """
    if not isinstance(user_timeline, pd.DataFrame):
        raise TypeError("user_timeline must be a pandas DataFrame")
    
    if len(user_timeline) < MIN_DATA_POINTS:
        return None
    
    # Create a copy to avoid mutating the input
    user_timeline = user_timeline.copy()
    
    # Aggregate by offset (day) - take mean of feature values for each day
    if feature_col == "text":
        user_timeline["_feature_value"] = user_timeline["text"].str.len()
    else:
        user_timeline["_feature_value"] = user_timeline[feature_col]
    
    # Determine which time column to use (dpo_days for DPO patterns, offset_from_cd1 for CD patterns)
    time_col = "dpo_days" if "dpo_days" in user_timeline.columns and user_timeline["dpo_days"].notna().any() else "offset_from_cd1"
    
    # Aggregate by day
    daily_agg = aggregate_by_day(
        user_timeline,
        offset_col=time_col,
        value_col="_feature_value",
        agg_func="mean"
    )
    
    if len(daily_agg) < MIN_DATA_POINTS:
        return None
    
    # Check for constant values
    if daily_agg["feature_mean"].nunique() == 1:
        return None  # Constant values, no signal
    
    # Get the time column name from daily_agg (should match time_col used above)
    time_col_name = "dpo_days" if "dpo_days" in daily_agg.columns else "offset_from_cd1"
    offsets = daily_agg[time_col_name].values
    values = daily_agg["feature_mean"].values
    
    # Normalize feature values per user to make amplitudes comparable
    values_norm = normalize_values(values, method=normalize_method)
    
    # Validate data quality
    is_valid, error_msg = validate_timeline_data(offsets, values, min_span_days=period_min)
    if not is_valid:
        return None
    
    # Default to fft_interpolation if not specified (matches old version's behavior when only fft_interpolation is used)
    if methods is None:
        methods = ["fft_interpolation"]
    
    # Base result dict
    span_days = float(offsets.max() - offsets.min())
    result = {
        "n_points": len(offsets),
        "span_days": span_days,
    }
    
    # Run only specified methods
    if "lomb_scargle" in methods:
        # Only run if astropy is available
        try:
            from astropy.timeseries import LombScargle
            ls_result = run_lombscargle_astropy(offsets, values_norm, period_search_min=10.0, period_search_max=90.0)
            result.update({
                "ls_period": ls_result["best_period"],
                "ls_power": ls_result["best_power"],
                "ls_fap": ls_result["fap"],
            })
        except ImportError:
            pass
    
    if "fft_interpolation" in methods:
        fft_interp_result = run_fft_interpolation(offsets, values_norm, period_min, period_max)
        result.update({
            "fft_interp_period": fft_interp_result["best_period"],
            "fft_interp_power": fft_interp_result["best_power"],
            "fft_interp_peak_to_background": fft_interp_result["peak_to_background"],
        })
    
    if "fft_zeropad" in methods:
        fft_zeropad_result = run_fft_zeropad(offsets, values_norm, period_min, period_max)
        result.update({
            "fft_zeropad_period": fft_zeropad_result["best_period"],
            "fft_zeropad_power": fft_zeropad_result["best_power"],
            "fft_zeropad_peak_to_background": fft_zeropad_result["peak_to_background"],
        })
    
    if "fft_interpolation_wide" in methods:
        fft_interp_wide_result = run_fft_interpolation_wide(
            offsets, values_norm, 
            period_search_min=period_wide_min, 
            period_search_max=period_wide_max
        )
        result.update({
            "fft_interp_wide_period": fft_interp_wide_result["best_period"],
            "fft_interp_wide_power": fft_interp_wide_result["best_power"],
            "fft_interp_wide_peak_to_background": fft_interp_wide_result["peak_to_background"],
        })
    
    if "fft_zeropad_wide" in methods:
        fft_zeropad_wide_result = run_fft_zeropad_wide(offsets, values_norm, period_search_min=period_wide_min, period_search_max=period_wide_max)
        result.update({
            "fft_zeropad_wide_period": fft_zeropad_wide_result["best_period"],
            "fft_zeropad_wide_power": fft_zeropad_wide_result["best_power"],
            "fft_zeropad_wide_peak_to_background": fft_zeropad_wide_result["peak_to_background"],
        })
    
    return result


def run_lombscargle_astropy(
    offsets: np.ndarray,
    values: np.ndarray,
    period_search_min: float = 10.0,
    period_search_max: float = 90.0,
    period_resolution: float = 0.1,
) -> dict:
    """Run Lomb-Scargle periodogram using astropy (with FAP and proper detrending).
    
    This is the improved version that searches a wider range and calculates
    False Alarm Probability. Post-filter results to 24-35 days after getting them.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_search_min: Minimum period to search (days, default 10)
        period_search_max: Maximum period to search (days, default 90)
        period_resolution: Resolution for period search (days, default 0.1)
        
    Returns:
        dict with:
            - best_period: detected period (days)
            - best_power: power at best period
            - fap: False Alarm Probability (lower = more significant)
            - periods: array of periods tested
            - powers: array of power values
    """
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        # Fallback if astropy not available
        return {
            "best_period": None,
            "best_power": 0.0,
            "fap": 1.0,
            "periods": np.array([]),
            "powers": np.array([]),
        }
    
    # Astropy's LombScargle handles mean-centering internally
    ls = LombScargle(offsets, values, center_data=True, fit_mean=True)
    
    # Create frequency grid (astropy uses frequency, not period)
    periods = np.arange(period_search_min, period_search_max + period_resolution, period_resolution)
    frequencies = 1.0 / periods
    
    # Calculate power
    power = ls.power(frequencies)
    
    # Find best period
    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    best_power = power[best_idx]
    
    # Calculate False Alarm Probability for the best peak
    # This tells us: "What's the probability this peak is just noise?"
    fap = ls.false_alarm_probability(best_power)
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "fap": fap,
        "periods": periods,
        "powers": power,
    }


def run_fft_interpolation_wide(
    offsets: np.ndarray,
    values: np.ndarray,
    period_search_min: float = 10.0,
    period_search_max: float = 50.0,
) -> dict:
    """Run FFT on interpolated data with WIDE search window (configurable range).
    
    Apply "widen and filter" strategy: search broadly, filter to 24-35 in post-processing.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_search_min: Minimum period to search (days, default 10)
        period_search_max: Maximum period to search (days, default 50)
        
    Returns:
        dict with best_period, best_power, peak_to_background (best in 10-50 range)
    """
    if len(offsets) < 2:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
        }
    
    offset_min = int(np.floor(offsets.min()))
    offset_max = int(np.ceil(offsets.max()))
    regular_offsets = np.arange(offset_min, offset_max + 1)
    regular_values = np.interp(regular_offsets, offsets, values)
    regular_values = regular_values - np.mean(regular_values)
    
    n = len(regular_values)
    fft_vals = fft(regular_values)
    freqs = fftfreq(n, d=1.0)
    
    positive_freqs = freqs[: n // 2]
    periods = 1.0 / positive_freqs[1:]
    powers = np.abs(fft_vals[: n // 2])[1:] ** 2
    
    mask = (periods >= period_search_min) & (periods <= period_search_max)
    periods_filtered = periods[mask]
    powers_filtered = powers[mask]
    
    if len(powers_filtered) == 0:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
        }
    
    best_idx = np.argmax(powers_filtered)
    best_period = round(periods_filtered[best_idx])
    best_power = powers_filtered[best_idx]
    
    powers_without_peak = np.concatenate([powers_filtered[:best_idx], powers_filtered[best_idx+1:]])
    if len(powers_without_peak) > 0:
        median_power_background = np.median(powers_without_peak)
        peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        peak_to_background = best_power / EPSILON
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
    }


def run_fft_zeropad_wide(
    offsets: np.ndarray,
    values: np.ndarray,
    period_search_min: float = 10.0,
    period_search_max: float = 50.0,
) -> dict:
    """Run FFT with zero-padding and WIDE search window (10-50 days).
    
    Apply "widen and filter" strategy: search broadly, filter to 24-35 in post-processing.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset (should be pre-normalized per user)
        period_search_min: Minimum period to search (days, default 10)
        period_search_max: Maximum period to search (days, default 50)
        
    Returns:
        dict with best_period, best_power, peak_to_background (best in 10-50 range)
    """
    if len(offsets) < 2:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
        }
    
    offset_min = int(np.floor(offsets.min()))
    offset_max = int(np.ceil(offsets.max()))
    regular_offsets = np.arange(offset_min, offset_max + 1)
    regular_values = np.interp(regular_offsets, offsets, values)
    regular_values = regular_values - np.mean(regular_values)
    
    n_original = len(regular_values)
    n_padded = 2 ** int(np.ceil(np.log2(n_original)))
    regular_values_padded = np.pad(regular_values, (0, n_padded - n_original), mode='constant')
    
    n = len(regular_values_padded)
    fft_vals = fft(regular_values_padded)
    freqs = fftfreq(n, d=1.0)
    
    positive_freqs = freqs[: n // 2]
    periods = 1.0 / positive_freqs[1:]
    powers = np.abs(fft_vals[: n // 2])[1:] ** 2
    
    mask = (periods >= period_search_min) & (periods <= period_search_max)
    periods_filtered = periods[mask]
    powers_filtered = powers[mask]
    
    if len(powers_filtered) == 0:
        return {
            "best_period": None,
            "best_power": 0.0,
            "peak_to_background": 0.0,
        }
    
    best_idx = np.argmax(powers_filtered)
    best_period = round(periods_filtered[best_idx])
    best_power = powers_filtered[best_idx]
    
    powers_without_peak = np.concatenate([powers_filtered[:best_idx], powers_filtered[best_idx+1:]])
    if len(powers_without_peak) > 0:
        median_power_background = np.median(powers_without_peak)
        peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        peak_to_background = best_power / EPSILON
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
    }


def analyze_all_users_periodicity(
    timeline_df: pd.DataFrame,
    features: list[str],
    user_col: str = "author",
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
) -> pd.DataFrame:
    """Run periodicity detection for all users across multiple features.
    
    Args:
        timeline_df: DataFrame with columns: user_col, offset_from_cd1, and feature columns
        features: List of feature column names to analyze (should already be normalized)
        user_col: Column name for user identifier (default: "author")
        period_min: Minimum period to search (days)
        period_max: Maximum period to search (days)
    
    Returns:
        Long-format DataFrame with columns:
        - user, feature, method (ls/fft_interp/fft_zeropad),
          period, power, peak_to_background, n_points, span_days
    """
    print(f"Analyzing {len(features)} features for {timeline_df[user_col].nunique()} users...")
    
    results = []
    
    for feature in features:
        if feature not in timeline_df.columns:
            print(f"  Warning: {feature} not found, skipping")
            continue
        
        print(f"  Processing {feature}...")
        
        for user, user_df in timeline_df.groupby(user_col):
            analysis = analyze_user_timeline(
                user_df,
                feature_col=feature,
                period_min=period_min,
                period_max=period_max,
                normalize_method="none",  # Features already normalized per user
            )
            
            if analysis is None:
                continue
            
            # Lomb-Scargle results
            results.append({
                "user": user,
                "feature": feature,
                "method": "lombscargle",
                "period": analysis["ls_period"],
                "power": analysis["ls_power"],
                "peak_to_background": analysis["ls_peak_to_background"],
                "n_points": analysis["n_points"],
                "span_days": analysis["span_days"],
            })
            
            # FFT interpolation results
            results.append({
                "user": user,
                "feature": feature,
                "method": "fft_interpolation",
                "period": analysis["fft_interp_period"],
                "power": analysis["fft_interp_power"],
                "peak_to_background": analysis["fft_interp_peak_to_background"],
                "n_points": analysis["n_points"],
                "span_days": analysis["span_days"],
            })
            
            # FFT zero-padding results
            results.append({
                "user": user,
                "feature": feature,
                "method": "fft_zeropad",
                "period": analysis["fft_zeropad_period"],
                "power": analysis["fft_zeropad_power"],
                "peak_to_background": analysis["fft_zeropad_peak_to_background"],
                "n_points": analysis["n_points"],
                "span_days": analysis["span_days"],
            })
    
    results_df = pd.DataFrame(results)
    print(f"✓ Analysis complete: {len(results_df)} results")
    return results_df


def analyze_all_users_with_normalizations(
    timeline_df: pd.DataFrame,
    base_features: list[str],
    normalizations: list[str],
    user_col: str = "author",
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    period_wide_min: float = 10.0,
    period_wide_max: float = 50.0,
    filter_range: bool = False,
    fap_threshold: float = 0.1,
    snr_threshold: float = 3.0,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Run periodicity detection with multiple normalization methods for comparison.
    
    This function analyzes BASE features (e.g., sentiment_compound) and applies
    different normalizations (zscore, minmax) to create a proper comparison.
    
    Args:
        timeline_df: DataFrame with columns: user_col, offset_from_cd1, and base feature columns
        base_features: List of BASE feature names (e.g., ['sentiment_compound', 'textblob_polarity'])
        normalizations: List of normalization methods (e.g., ['zscore', 'minmax'])
        user_col: Column name for user identifier
        period_min: Min period (used for narrow FFT, not for filtering)
        period_max: Max period (used for narrow FFT, not for filtering)
        period_wide_min: Min period for wide FFT search (default: 10.0)
        period_wide_max: Max period for wide FFT search (default: 50.0)
        filter_range: If True, filter results to period_min-period_max (default: False)
        fap_threshold: False Alarm Probability threshold for LS filtering (default: 0.1)
        snr_threshold: Peak-to-background threshold for FFT filtering (default: 3.0)
    
    Returns:
        Long-format DataFrame with columns:
        - user, feature, normalization, method, period, power, fap, peak_to_background, n_points, span_days
    """
    print(f"Analyzing {len(base_features)} base features × {len(normalizations)} normalizations for {timeline_df[user_col].nunique()} users...")
    
    results = []
    
    for base_feature in base_features:
        if base_feature not in timeline_df.columns:
            print(f"  Warning: {base_feature} not found, skipping")
            continue
        
        for normalization in normalizations:
            print(f"  Processing {base_feature} with {normalization} normalization...")
            
            for user, user_df in timeline_df.groupby(user_col):
                analysis = analyze_user_timeline(
                    user_df,
                    feature_col=base_feature,
                    period_min=period_min,
                    period_max=period_max,
                    period_wide_min=period_wide_min,
                    period_wide_max=period_wide_max,
                    normalize_method=normalization,
                    methods=methods,
                )
                
                if analysis is None:
                    continue
                
                # Extract results for each method that was run (conditional on presence in analysis dict)
                if "ls_period" in analysis:
                    results.append({
                        "user": user,
                        "feature": base_feature,
                        "normalization": normalization,
                        "method": "lombscargle",
                        "period": analysis["ls_period"],
                        "power": analysis["ls_power"],
                        "fap": analysis["ls_fap"],
                        "n_points": analysis["n_points"],
                        "span_days": analysis["span_days"],
                    })
                
                if "fft_interp_period" in analysis:
                    results.append({
                        "user": user,
                        "feature": base_feature,
                        "normalization": normalization,
                        "method": "fft_interpolation",
                        "period": analysis["fft_interp_period"],
                        "power": analysis["fft_interp_power"],
                        "peak_to_background": analysis["fft_interp_peak_to_background"],
                        "n_points": analysis["n_points"],
                        "span_days": analysis["span_days"],
                    })
                
                if "fft_zeropad_period" in analysis:
                    results.append({
                        "user": user,
                        "feature": base_feature,
                        "normalization": normalization,
                        "method": "fft_zeropad",
                        "period": analysis["fft_zeropad_period"],
                        "power": analysis["fft_zeropad_power"],
                        "peak_to_background": analysis["fft_zeropad_peak_to_background"],
                        "n_points": analysis["n_points"],
                        "span_days": analysis["span_days"],
                    })
                
                if "fft_interp_wide_period" in analysis:
                    results.append({
                        "user": user,
                        "feature": base_feature,
                        "normalization": normalization,
                        "method": "fft_interp_wide",
                        "period": analysis["fft_interp_wide_period"],
                        "power": analysis["fft_interp_wide_power"],
                        "peak_to_background": analysis["fft_interp_wide_peak_to_background"],
                        "n_points": analysis["n_points"],
                        "span_days": analysis["span_days"],
                    })
                
                if "fft_zeropad_wide_period" in analysis:
                    results.append({
                        "user": user,
                        "feature": base_feature,
                        "normalization": normalization,
                        "method": "fft_zeropad_wide",
                        "period": analysis["fft_zeropad_wide_period"],
                        "power": analysis["fft_zeropad_wide_power"],
                        "peak_to_background": analysis["fft_zeropad_wide_peak_to_background"],
                        "n_points": analysis["n_points"],
                        "span_days": analysis["span_days"],
                    })
    
    results_df = pd.DataFrame(results)
    print(f"✓ Analysis complete: {len(results_df)} results (before filtering)")
    
    # Apply FAP filtering to LS results (only if fap column exists)
    ls_before = (results_df["method"] == "lombscargle").sum()
    if ls_before > 0 and "fap" in results_df.columns:
        ls_filtered = results_df[
            (results_df["method"] == "lombscargle") & 
            (results_df["fap"] < fap_threshold)
        ]
        ls_after = len(ls_filtered)
    else:
        ls_filtered = results_df[results_df["method"] == "lombscargle"]
        ls_after = len(ls_filtered)
    
    # Apply peak_to_background filtering to FFT results (only if column exists)
    fft_results = results_df[results_df["method"] != "lombscargle"]
    fft_before = len(fft_results)
    if fft_before > 0 and "peak_to_background" in results_df.columns:
        fft_filtered = fft_results[fft_results["peak_to_background"] > snr_threshold]
        fft_after = len(fft_filtered)
    else:
        fft_filtered = fft_results
        fft_after = len(fft_filtered)
    
    # Combine
    results_df = pd.concat([ls_filtered, fft_filtered], ignore_index=True)
    
    if ls_before > 0 and "fap" in results_df.columns:
        print(f"  ✓ LS filtered by FAP < {fap_threshold}: {ls_before} → {ls_after} ({ls_before - ls_after} dropped)")
    elif ls_before > 0:
        print(f"  ✓ LS results: {ls_before} (no FAP column, skipping filter)")
    if fft_before > 0 and "peak_to_background" in results_df.columns:
        print(f"  ✓ FFT filtered by peak_to_background > {snr_threshold}: {fft_before} → {fft_after} ({fft_before - fft_after} dropped)")
    elif fft_before > 0:
        print(f"  ✓ FFT results: {fft_before} (no peak_to_background column, skipping filter)")
    
    if filter_range:
        # Optional range filtering (applied to all methods)
        print(f"  Applying range filter [{period_min}-{period_max}] days...")
        before_range = len(results_df)
        results_df = results_df[
            (results_df["period"] >= period_min) & 
            (results_df["period"] <= period_max)
        ].copy()
        print(f"  ✓ After range filtering: {before_range} → {len(results_df)} results")
    
    return results_df


def create_adaptive_phases(cycle_length: float) -> dict[str, tuple[int, int]]:
    """Create phase definitions adapted to cycle length.
    
    Biological fact: Cycle length variation comes primarily from FOLLICULAR phase.
    - Menstrual phase: ~4 days (relatively fixed)
    - Follicular phase: 7-15 days (VARIABLE - where cycle differences occur)
    - Ovulation: ~3 days (relatively fixed)
    - Luteal phase: ~14 days (relatively fixed, though can vary slightly)
    
    For 28-day cycle: M(0-3), F(4-10=7d), O(11-13), L(14-27=14d)
    For 32-day cycle: M(0-3), F(4-14=11d), O(15-17), L(18-31=14d) <- +4 days to follicular
    
    Args:
        cycle_length: Detected cycle length in days (24-35)
    
    Returns:
        Dictionary of {phase_name: (start_day, end_day)}
    """
    menstrual_len = 4
    ovulation_len = 3
    luteal_len = 14
    
    follicular_len = cycle_length - menstrual_len - ovulation_len - luteal_len
    follicular_len = max(3, follicular_len)
    
    menstrual_end = menstrual_len - 1
    follicular_start = menstrual_len
    follicular_end = follicular_start + follicular_len - 1
    ovulation_start = follicular_end + 1
    ovulation_end = ovulation_start + ovulation_len - 1
    luteal_start = ovulation_end + 1
    luteal_end = int(cycle_length - 1)
    
    return {
        'Menstrual': (0, menstrual_end),
        'Follicular': (follicular_start, follicular_end),
        'Ovulation': (ovulation_start, ovulation_end),
        'Luteal': (luteal_start, luteal_end),
    }


def assign_phase_to_day(day: float, phase_definition: dict[str, tuple[int, int]]) -> str | None:
    """Assign phase to a day based on phase definition.
    
    Handles negative days (before anchor) and days beyond cycle length (wrapping).
    Works for both CD (offset_from_cd1) and DPO (dpo_days) patterns.
    
    Args:
        day: Day value (offset_from_cd1 for CD patterns, dpo_days for DPO patterns)
        phase_definition: Dictionary of {phase_name: (start_day, end_day)}
    
    Returns:
        Phase name or None
    """
    cycle_length = max(end for _, end in phase_definition.values()) + 1
    
    # Wrap days using modulo (Python's % already handles negatives correctly)
    day_normalized = day % cycle_length
    
    for phase_name, (start, end) in phase_definition.items():
        if start <= day_normalized <= end:
            return phase_name
    
    return None


def compute_user_phase_definitions(
    user_period_map: dict[str, float]
) -> pd.DataFrame:
    """Pre-compute phase definitions for all users.
    
    Args:
        user_period_map: Dictionary mapping user -> period length (days)
    
    Returns:
        DataFrame with columns: user, phase, start_day, end_day, length_days
        One row per user-phase combination (4 rows per user)
    """
    rows = []
    for user, period in user_period_map.items():
        phase_def = create_adaptive_phases(period)
        for phase, (start, end) in phase_def.items():
            length_days = end - start + 1
            rows.append({
                "user": user,
                "phase": phase,
                "start_day": start,
                "end_day": end,
                "length_days": length_days,
            })
    
    return pd.DataFrame(rows)


def assign_phases_to_timeline(
    timeline_df: pd.DataFrame,
    user_phase_df: pd.DataFrame,
    user_col: str = "author",
    time_col: str = "offset_from_cd1",
) -> pd.DataFrame:
    """Assign phases to timeline posts based on pre-computed phase definitions.
    
    Args:
        timeline_df: DataFrame with posts and time column
        user_phase_df: DataFrame with columns: user, phase, start_day, end_day, length_days
        user_col: Column name for user identifier (default: "author")
        time_col: Column name for time offset (default: "offset_from_cd1")
    
    Returns:
        DataFrame with added 'phase' column
    """
    timeline_df = timeline_df.copy()
    
    # Create lookup dict for faster access: {user: {phase: (start, end)}}
    user_phase_lookup = {}
    for user in user_phase_df["user"].unique():
        user_phases = user_phase_df[user_phase_df["user"] == user]
        user_phase_lookup[user] = {
            row["phase"]: (row["start_day"], row["end_day"])
            for _, row in user_phases.iterrows()
        }
    
    def assign_phase_vectorized(row):
        user = str(row[user_col])
        if user not in user_phase_lookup:
            return None
        
        day = row.get(time_col, None)
        if day is None or pd.isna(day):
            return None
        
        phase_def = user_phase_lookup[user]
        return assign_phase_to_day(day, phase_def)
    
    timeline_df["phase"] = timeline_df.apply(assign_phase_vectorized, axis=1)
    return timeline_df


def aggregate_to_phase_profiles(df: pd.DataFrame, zscore_cols: list[str]) -> pd.DataFrame:
    """Collapse day-level rows to one averaged feature vector per (user, phase).

    Instead of predicting from a single noisy day, each sample becomes a user's
    mean linguistic profile across all days in a given phase.  This removes
    within-phase day-to-day noise and matches exactly what bar charts display.

    Result: ~N_users × 4 rows (one per user-phase pair that has data).
    GroupKFold must still be keyed on author so all phase rows for a user
    stay together in either train or test.

    Args:
        df: Day-level DataFrame with 'author', 'phase', and zscore_cols.
        zscore_cols: Feature columns to average (typically {feature}_zscore columns).

    Returns:
        DataFrame with one row per (author, phase), columns = author + phase + zscore_cols.
    """
    profiles = (
        df.groupby(["author", "phase"])[zscore_cols]
        .mean()
        .reset_index()
    )
    # Fill NaN means with 0 (= user's own mean, since features are z-scored)
    profiles[zscore_cols] = profiles[zscore_cols].fillna(0)
    return profiles


def calculate_phase_statistics(
    df: pd.DataFrame,
    group_name: str,
    user_phase_df: pd.DataFrame,
    user_col: str = "author",
    suicide_subreddits: set[str] | None = None,
    depression_subreddits: set[str] | None = None,
    adhd_subreddits: set[str] | None = None,
) -> pd.DataFrame:
    """Calculate posting volume by phase with cycle-length and z-score normalization.
    
    For each woman:
    1. Count posts per phase (separate suicide, depression, and ADHD)
    2. Normalize for cycle length (divide by phase length in days)
    3. Z-score normalize per woman (across 4 phases)
    
    Then aggregate by group and phase.
    
    Args:
        df: DataFrame with posts, must have 'phase', 'subreddit', and user_col columns
        group_name: Name of group (e.g., "PMDD" or "Control")
        user_phase_df: DataFrame with columns: user, phase, start_day, end_day, length_days
        user_col: Column name for user identifier (default: "author")
        suicide_subreddits: Set of suicide subreddit names (default: None, uses empty set)
        depression_subreddits: Set of depression subreddit names (default: None, uses empty set)
        adhd_subreddits: Set of ADHD subreddit names (default: None, uses empty set)
    
    Returns:
        DataFrame with columns: group, phase, suicide_volume, depression_volume, adhd_volume,
        suicide_raw, depression_raw, adhd_raw, suicide_zscore, depression_zscore, adhd_zscore,
        n_users, n_users_suicide, n_users_depression, n_users_adhd
    """
    df = df.copy()
    
    if suicide_subreddits is None:
        suicide_subreddits = set()
    if depression_subreddits is None:
        depression_subreddits = set()
    if adhd_subreddits is None:
        adhd_subreddits = set()
    
    # Filter to posts with assigned phases
    df = df[df["phase"].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame()
    
    # Classify posts by subreddit type (case-sensitive matching)
    df["is_suicide"] = df["subreddit"].isin(suicide_subreddits)
    df["is_depression"] = df["subreddit"].isin(depression_subreddits)
    df["is_adhd"] = df["subreddit"].isin(adhd_subreddits)
    
    # Identify users who posted in suicide/depression/ADHD subreddits within timeline window
    # (used for z-score aggregation - include all these users even if 0 posts in a specific phase)
    users_with_suicide_posts = set(df[df["is_suicide"]][user_col].astype(str).unique())
    users_with_depression_posts = set(df[df["is_depression"]][user_col].astype(str).unique())
    users_with_adhd_posts = set(df[df["is_adhd"]][user_col].astype(str).unique())
    
    # Create lookup dict for phase lengths: {user: {phase: length}}
    user_phase_lengths = {}
    for user in user_phase_df["user"].unique():
        user_phases = user_phase_df[user_phase_df["user"] == user]
        user_phase_lengths[user] = {
            row["phase"]: row["length_days"]
            for _, row in user_phases.iterrows()
        }
    
    # Count posts per user per phase
    # IMPORTANT: Create rows for ALL phases for each user, even if they have 0 posts
    # This ensures users who didn't post in a specific phase are included in z-score calculations
    user_phase_counts = []
    for user in df[user_col].unique():
        user_str = str(user)
        if user_str not in user_phase_lengths:
            continue
        
        user_df = df[df[user_col] == user]
        
        # Create rows for ALL 4 phases for this user
        for phase in ["Menstrual", "Follicular", "Ovulation", "Luteal"]:
            if phase not in user_phase_lengths[user_str]:
                continue
            
            # Get posts for this user in this phase (may be empty)
            phase_df = user_df[user_df["phase"] == phase]
            suicide_posts = phase_df["is_suicide"].sum()
            depression_posts = phase_df["is_depression"].sum()
            adhd_posts = phase_df["is_adhd"].sum()
            
            phase_length = user_phase_lengths[user_str][phase]
            
            user_phase_counts.append({
                "user": user_str,
                "phase": phase,
                "suicide_posts": suicide_posts,
                "depression_posts": depression_posts,
                "adhd_posts": adhd_posts,
                "phase_length": phase_length,
                "suicide_volume": suicide_posts / phase_length if phase_length > 0 else 0,
                "depression_volume": depression_posts / phase_length if phase_length > 0 else 0,
                "adhd_volume": adhd_posts / phase_length if phase_length > 0 else 0,
            })
    
    counts_df = pd.DataFrame(user_phase_counts)
    if len(counts_df) == 0:
        return pd.DataFrame()
    
    # Z-score normalize per user across phases
    zscore_df = []
    for user in counts_df["user"].unique():
        user_df = counts_df[counts_df["user"] == user].copy()
        
        # Calculate z-scores across phases for this user
        suicide_mean = user_df["suicide_volume"].mean()
        suicide_std = user_df["suicide_volume"].std()
        depression_mean = user_df["depression_volume"].mean()
        depression_std = user_df["depression_volume"].std()
        adhd_mean = user_df["adhd_volume"].mean()
        adhd_std = user_df["adhd_volume"].std()
        
        # Handle division by zero (if all phases have same rate for a user) - match old script
        # Set std to 1.0 if < 1e-10, then calculate z-score normally
        if suicide_std < 1e-10:
            suicide_std = 1.0
        if depression_std < 1e-10:
            depression_std = 1.0
        if adhd_std < 1e-10:
            adhd_std = 1.0
        
        user_df["suicide_zscore"] = (user_df["suicide_volume"] - suicide_mean) / suicide_std
        user_df["depression_zscore"] = (user_df["depression_volume"] - depression_mean) / depression_std
        user_df["adhd_zscore"] = (user_df["adhd_volume"] - adhd_mean) / adhd_std
        
        zscore_df.append(user_df)
    
    zscore_df = pd.concat(zscore_df, ignore_index=True)
    
    # Aggregate by phase
    phase_stats = []
    for phase in ["Menstrual", "Follicular", "Ovulation", "Luteal"]:
        phase_df = zscore_df[zscore_df["phase"] == phase]
        
        if len(phase_df) == 0:
            continue
        
        # Count unique users from original posts DataFrame
        phase_posts = df[df["phase"] == phase]
        n_users_total = len(phase_posts[user_col].unique())
        
        # Count users who posted in suicide subreddits in this phase
        suicide_posts_in_phase = phase_posts[phase_posts["is_suicide"]]
        n_users_suicide = len(suicide_posts_in_phase[user_col].unique()) if len(suicide_posts_in_phase) > 0 else 0
        
        # Count users who posted in depression subreddits in this phase
        depression_posts_in_phase = phase_posts[phase_posts["is_depression"]]
        n_users_depression = len(depression_posts_in_phase[user_col].unique()) if len(depression_posts_in_phase) > 0 else 0
        
        # Count users who posted in ADHD subreddits in this phase
        adhd_posts_in_phase = phase_posts[phase_posts["is_adhd"]]
        n_users_adhd = len(adhd_posts_in_phase[user_col].unique()) if len(adhd_posts_in_phase) > 0 else 0
        
        # For volume calculations, only include users who posted
        phase_df_with_posts = phase_df[
            (phase_df["suicide_posts"] > 0) | (phase_df["depression_posts"] > 0) | (phase_df["adhd_posts"] > 0)
        ]
        suicide_volume_mean = phase_df_with_posts["suicide_volume"].mean() if len(phase_df_with_posts) > 0 else 0.0
        depression_volume_mean = phase_df_with_posts["depression_volume"].mean() if len(phase_df_with_posts) > 0 else 0.0
        adhd_volume_mean = phase_df_with_posts["adhd_volume"].mean() if len(phase_df_with_posts) > 0 else 0.0
        
        # For z-score, calculate separately for suicide and depression (like old script)
        # Suicide z-score: include ALL users who posted in suicide subreddits (within timeline window)
        # even if they have 0 posts in this specific phase
        suicide_users_in_phase = phase_df[phase_df["user"].astype(str).isin(users_with_suicide_posts)]
        if len(suicide_users_in_phase) > 0:
            suicide_zscore_mean = suicide_users_in_phase["suicide_zscore"].mean()
        else:
            suicide_zscore_mean = 0.0
        
        # Depression z-score: include ALL users who posted in depression subreddits (within timeline window)
        # even if they have 0 posts in this specific phase
        depression_users_in_phase = phase_df[phase_df["user"].astype(str).isin(users_with_depression_posts)]
        if len(depression_users_in_phase) > 0:
            depression_zscore_mean = depression_users_in_phase["depression_zscore"].mean()
        else:
            depression_zscore_mean = 0.0
        
        # ADHD z-score: include ALL users who posted in ADHD subreddits (within timeline window)
        # even if they have 0 posts in this specific phase
        adhd_users_in_phase = phase_df[phase_df["user"].astype(str).isin(users_with_adhd_posts)]
        if len(adhd_users_in_phase) > 0:
            adhd_zscore_mean = adhd_users_in_phase["adhd_zscore"].mean()
        else:
            adhd_zscore_mean = 0.0
        
        phase_stats.append({
            "group": group_name,
            "phase": phase,
            "suicide_volume": suicide_volume_mean,
            "depression_volume": depression_volume_mean,
            "adhd_volume": adhd_volume_mean,
            "suicide_raw": phase_df["suicide_posts"].sum(),
            "depression_raw": phase_df["depression_posts"].sum(),
            "adhd_raw": phase_df["adhd_posts"].sum(),
            "suicide_zscore": suicide_zscore_mean,
            "depression_zscore": depression_zscore_mean,
            "adhd_zscore": adhd_zscore_mean,
            "n_users": n_users_total,
            "n_users_suicide": n_users_suicide,
            "n_users_depression": n_users_depression,
            "n_users_adhd": n_users_adhd,
        })
    
    return pd.DataFrame(phase_stats)


def calculate_posts_before_after_ratio(
    timeline_df: pd.DataFrame,
    offset_col: str = 'offset_from_cd1',
    user_col: str = 'author',
) -> dict[str, float]:
    """Calculate ratio of posts after anchor to posts before anchor for each user.
    
    For each user, computes:
        ratio = posts_after_anchor / posts_before_anchor
    
    Where:
        - posts_before_anchor: posts with offset < 0
        - posts_after_anchor: posts with offset > 0
        - Anchor day (offset = 0) is excluded from both counts
    
    Args:
        timeline_df: Timeline DataFrame with offset column
        offset_col: Column name for offset from anchor (default: 'offset_from_cd1')
        user_col: Column name for user identifier (default: 'author')
    
    Returns:
        Dictionary with:
            - 'user_ratios': Series of ratios per user (indexed by user)
            - 'mean_ratio': Average ratio across users
            - 'median_ratio': Median ratio across users
            - 'n_users': Number of users with valid ratios
            - 'n_users_before_only': Number of users with only before-anchor posts
            - 'n_users_after_only': Number of users with only after-anchor posts
    """
    timeline_df = timeline_df.copy()
    
    # Filter to users with valid offsets
    valid_df = timeline_df[
        timeline_df[offset_col].notna() & 
        timeline_df[user_col].notna()
    ].copy()
    
    if len(valid_df) == 0:
        return {
            'user_ratios': pd.Series(dtype=float),
            'mean_ratio': np.nan,
            'median_ratio': np.nan,
            'n_users': 0,
            'n_users_before_only': 0,
            'n_users_after_only': 0,
        }
    
    # Count posts before and after anchor for each user
    user_counts = []
    
    for user in valid_df[user_col].unique():
        user_data = valid_df[valid_df[user_col] == user]
        
        # Exclude anchor day (offset = 0)
        before = len(user_data[user_data[offset_col] < 0])
        after = len(user_data[user_data[offset_col] > 0])
        
        if before == 0 and after == 0:
            continue  # Skip users with no posts before or after
        
        user_counts.append({
            user_col: user,
            'posts_before': before,
            'posts_after': after,
        })
    
    counts_df = pd.DataFrame(user_counts)
    
    # Calculate ratio (posts_after / posts_before)
    # Handle edge cases:
    # - before = 0, after > 0: ratio = inf (user only posts after anchor)
    # - before > 0, after = 0: ratio = 0 (user only posts before anchor)
    # - before > 0, after > 0: ratio = after / before
    counts_df['ratio'] = np.where(
        counts_df['posts_before'] == 0,
        np.inf,  # Only after-anchor posts
        counts_df['posts_after'] / counts_df['posts_before']
    )
    
    # Separate users with valid ratios (both before and after)
    valid_ratios = counts_df[
        (counts_df['posts_before'] > 0) & (counts_df['posts_after'] > 0)
    ]['ratio']
    
    before_only = len(counts_df[(counts_df['posts_before'] > 0) & (counts_df['posts_after'] == 0)])
    after_only = len(counts_df[(counts_df['posts_before'] == 0) & (counts_df['posts_after'] > 0)])
    
    # Calculate statistics on valid ratios only (exclude inf and 0 from edge cases)
    mean_ratio = valid_ratios.mean() if len(valid_ratios) > 0 else np.nan
    median_ratio = valid_ratios.median() if len(valid_ratios) > 0 else np.nan
    
    return {
        'user_ratios': counts_df.set_index(user_col)['ratio'],
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'n_users': len(valid_ratios),
        'n_users_before_only': before_only,
        'n_users_after_only': after_only,
        'counts_df': counts_df,  # Include full counts for detailed analysis
    }


def assign_consensus_period_by_majority(
    periodicity_results: pd.DataFrame,
    min_features: int = 5,
    tolerance: int = 1,
) -> pd.DataFrame:
    """Assign consensus period based on majority vote across features.
    
    For each user, finds the period that the majority of features agree on
    (within tolerance). Requires at least min_features to agree.
    
    Args:
        periodicity_results: DataFrame with columns: user, feature, period, snr (or similar)
        min_features: Minimum number of features required for consensus (default: 5)
        tolerance: ±N days tolerance for "agreement" (default: 1)
    
    Returns:
        DataFrame with columns: user, consensus_period, n_features_agreeing, consensus_pct
    """
    if len(periodicity_results) == 0:
        return pd.DataFrame(columns=["user", "consensus_period", "n_features_agreeing", "consensus_pct"])
    
    # Required columns
    required_cols = ["user", "feature", "period"]
    missing_cols = [col for col in required_cols if col not in periodicity_results.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    consensus_results = []
    
    for user in periodicity_results["user"].unique():
        user_data = periodicity_results[periodicity_results["user"] == user].copy()
        
        # Filter out NaN periods
        user_data = user_data[user_data["period"].notna()].copy()
        
        if len(user_data) < min_features:
            continue  # Skip users with insufficient features
        
        periods = user_data["period"].values
        
        # Find the period that most features agree on (within tolerance)
        best_period = None
        best_count = 0
        
        # Try each unique period as a candidate
        for candidate_period in np.unique(periods):
            # Count how many features agree (within tolerance)
            within_tolerance = np.sum(np.abs(periods - candidate_period) <= tolerance)
            
            if within_tolerance > best_count:
                best_count = within_tolerance
                best_period = candidate_period
        
        # Only assign consensus if enough features agree
        if best_count >= min_features:
            consensus_pct = (best_count / len(periods)) * 100
            consensus_results.append({
                "user": user,
                "consensus_period": best_period,
                "n_features_agreeing": best_count,
                "consensus_pct": consensus_pct,
            })
    
    if len(consensus_results) == 0:
        return pd.DataFrame(columns=["user", "consensus_period", "n_features_agreeing", "consensus_pct"])
    
    consensus_df = pd.DataFrame(consensus_results)
    return consensus_df.sort_values("user").reset_index(drop=True)


def calculate_feature_phase_range(
    phase_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    phase_col: str = "phase",
) -> pd.DataFrame:
    """Calculate (max - min) feature value across phases for each feature.
    
    This metric measures how much each feature varies across menstrual cycle phases.
    Features with larger (max-min) differences are more likely to show cyclical patterns.
    
    Args:
        phase_df: DataFrame with phase assignments and feature values
            Should have columns: phase_col, and feature columns (or {feature}_zscore columns)
        feature_cols: List of feature column names to analyze
            If None, auto-detect feature columns (those ending in _zscore or _mean)
        phase_col: Column name for phase assignments (default: "phase")
    
    Returns:
        DataFrame with columns: feature, max_min_diff, max_phase, min_phase, mean_max, mean_min
        Sorted by max_min_diff descending (largest phase differences first)
    """
    phase_df = phase_df.copy()
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        # Look for columns ending in _zscore or _mean
        all_cols = set(phase_df.columns)
        exclude_cols = {phase_col, "user", "author", "offset_from_cd1", "day", "period"}
        feature_cols = [
            col for col in all_cols 
            if col not in exclude_cols and (col.endswith("_zscore") or col.endswith("_mean"))
        ]
    
    if len(feature_cols) == 0:
        return pd.DataFrame(columns=["feature", "max_min_diff", "max_phase", "min_phase", "mean_max", "mean_min"])
    
    results = []
    
    for feature in feature_cols:
        if feature not in phase_df.columns:
            continue
        
        # Calculate mean value per phase for this feature
        phase_means = phase_df.groupby(phase_col)[feature].mean()
        
        if len(phase_means) == 0:
            continue
        
        # Find max and min phases
        max_val = phase_means.max()
        min_val = phase_means.min()
        max_min_diff = max_val - min_val
        
        max_phase = phase_means.idxmax()
        min_phase = phase_means.idxmin()
        
        results.append({
            "feature": feature,
            "max_min_diff": max_min_diff,
            "max_phase": max_phase,
            "min_phase": min_phase,
            "mean_max": max_val,
            "mean_min": min_val,
        })
    
    if len(results) == 0:
        return pd.DataFrame(columns=["feature", "max_min_diff", "max_phase", "min_phase", "mean_max", "mean_min"])
    
    result_df = pd.DataFrame(results)
    return result_df.sort_values("max_min_diff", ascending=False).reset_index(drop=True)


def analyze_signal_fading(
    timeline_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    feature_cols: list[str],
    time_col: str = "offset_from_cd1",
    user_col: str = "author",
    months_bins: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Analyze how (max-min) feature values reduce over time.
    
    For each feature, calculates (max - min) across phases, binned by months from anchor.
    Shows how signal strength (phase difference) decreases as time from anchor increases,
    which is expected due to cycle length shifts (~2 days per month).
    
    Args:
        timeline_df: DataFrame with posts and time offsets
        phase_df: DataFrame with phase assignments (columns: user, phase, feature values)
        feature_cols: List of feature column names to analyze (should be zscore normalized)
        time_col: Column name for time offset (default: "offset_from_cd1")
        user_col: Column name for user identifier (default: "author")
        months_bins: List of (start_month, end_month) tuples for binning (default: auto)
    
    Returns:
        DataFrame with columns: feature, months_bin, max_min_diff, n_users, mean_max_min
    """
    if months_bins is None:
        # Default bins: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6 months
        months_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    
    # Convert time offsets to months from anchor (assuming anchor is at offset=0)
    timeline_df = timeline_df.copy()
    timeline_df["months_from_anchor"] = timeline_df[time_col] / AVG_DAYS_PER_MONTH
    
    # Assign months bin to each post
    def assign_months_bin(months: float) -> str | None:
        """Assign months bin label."""
        for start, end in months_bins:
            if start <= months < end:
                return f"{start}-{end}"
        return None
    
    timeline_df["months_bin"] = timeline_df["months_from_anchor"].apply(assign_months_bin)
    
    # For each feature, calculate (max - min) per phase, per months bin
    fading_results = []
    
    for feature in feature_cols:
        # Use zscore column if available, otherwise use mean
        feature_col = f"{feature}_zscore" if f"{feature}_zscore" in phase_df.columns else f"{feature}_mean"
        
        if feature_col not in phase_df.columns:
            continue
        
        # For each months bin, calculate phase statistics
        for months_bin_label in [f"{start}-{end}" for start, end in months_bins]:
            # Filter to posts in this months bin
            bin_timeline = timeline_df[timeline_df["months_bin"] == months_bin_label].copy()
            
            if len(bin_timeline) == 0:
                continue
            
            # Get users in this bin
            users_in_bin = set(bin_timeline[user_col].unique())
            
            # For each user, calculate (max - min) across phases
            user_max_mins = []
            
            for user in users_in_bin:
                user_phase_data = phase_df[
                    (phase_df[user_col] == user) & 
                    (phase_df[feature_col].notna())
                ].copy()
                
                if len(user_phase_data) == 0:
                    continue
                
                # Calculate max and min across phases for this user
                max_val = user_phase_data[feature_col].max()
                min_val = user_phase_data[feature_col].min()
                max_min_diff = max_val - min_val
                
                user_max_mins.append(max_min_diff)
            
            if len(user_max_mins) > 0:
                fading_results.append({
                    "feature": feature,
                    "months_bin": months_bin_label,
                    "max_min_diff": np.mean(user_max_mins),  # Mean across users
                    "n_users": len(user_max_mins),
                    "std_max_min": np.std(user_max_mins),
                })
    
    if len(fading_results) == 0:
        return pd.DataFrame(columns=["feature", "months_bin", "max_min_diff", "n_users", "std_max_min"])
    
    fading_df = pd.DataFrame(fading_results)
    return fading_df.sort_values(["feature", "months_bin"]).reset_index(drop=True)

