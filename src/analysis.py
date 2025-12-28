"""Functions for FFT and Lomb-Scargle periodogram analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from astropy.timeseries import LombScargle

# Constants
MIN_DATA_POINTS = 10
EPSILON = 1e-10
DEFAULT_PERIOD_MIN = 24
DEFAULT_PERIOD_MAX = 35


def normalize_values(values: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize values using specified method.
    
    Args:
        values: Array of values to normalize
        method: Normalization method ("zscore", "minmax", or "none")
        
    Returns:
        Normalized array
    """
    if method == "zscore":
        return (values - np.mean(values)) / (np.std(values) + EPSILON)
    elif method == "minmax":
        v_min, v_max = np.min(values), np.max(values)
        return (values - v_min) / (v_max - v_min + EPSILON)
    elif method == "none":
        return values
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def validate_timeline_data(
    offsets: np.ndarray,
    values: np.ndarray,
    min_points: int = MIN_DATA_POINTS,
    min_unique_days: int = MIN_DATA_POINTS,
    min_span_days: int = DEFAULT_PERIOD_MIN,
) -> tuple[bool, str | None]:
    """Validate timeline data quality.
    
    Args:
        offsets: Time offsets (days from CD1)
        values: Feature values at each offset
        min_points: Minimum number of data points required
        min_unique_days: Minimum number of unique days required
        min_span_days: Minimum time span required (days)
        
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
    
    if np.std(values) < EPSILON:
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
        peak_to_background = best_power / (median_power_background + EPSILON)
    else:
        peak_to_background = best_power / EPSILON
    
    return {
        "best_period": best_period,
        "best_power": best_power,
        "peak_to_background": peak_to_background,
        "periods": periods,
        "powers": power,
    }


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
    
    # Default to all methods if not specified
    if methods is None:
        methods = ["lomb_scargle", "fft_interpolation", "fft_zeropad", "fft_interpolation_wide", "fft_zeropad_wide"]
    
    # Base result dict
    span_days = float(offsets.max() - offsets.min())
    result = {
        "n_points": len(offsets),
        "span_days": span_days,
    }
    
    # Run only specified methods
    if "lomb_scargle" in methods:
        ls_result = run_lombscargle_astropy(offsets, values_norm, period_search_min=10.0, period_search_max=90.0)
        result.update({
            "ls_period": ls_result["best_period"],
            "ls_power": ls_result["best_power"],
            "ls_fap": ls_result["fap"],
        })
    
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
        # Debug: verify parameters are being used
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


def analyze_all_users_periodicity(
    timeline_df: pd.DataFrame,
    features: list[str],
    user_col: str = "author",
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Run periodicity detection for all users across multiple features.
    
    Available methods:
    - lomb_scargle: searches 10-90 days, filters to 24-35 with FAP < 0.1
    - fft_interpolation: search 24-35 days directly
    - fft_zeropad: search 24-35 days with zero-padding
    - fft_interpolation_wide: search 10-50 days, filter to 24-35
    - fft_zeropad_wide: search 10-50 days with zero-padding, filter to 24-35
    
    Args:
        timeline_df: DataFrame with columns: user_col, offset_from_cd1, and feature columns
        features: List of feature column names to analyze (should already be normalized)
        user_col: Column name for user identifier (default: "author")
        period_min: Minimum period for biological relevance filter (days, default 24)
        period_max: Maximum period for biological relevance filter (days, default 35)
        methods: List of method names to run (default: all methods)
    
    Returns:
        Long-format DataFrame with columns:
        - user, feature, method
        - period, power, fap (LS only), peak_to_background (FFT only)
        - n_points, span_days
        Note: Results are post-filtered to period_min-period_max range
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
                methods=methods,
            )
            
            if analysis is None:
                continue
            
            # Extract results for each method that was run
            if "ls_period" in analysis:
                results.append({
                    "user": user,
                    "feature": feature,
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
                    "feature": feature,
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
                    "feature": feature,
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
                    "feature": feature,
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
                    "feature": feature,
                    "method": "fft_zeropad_wide",
                    "period": analysis["fft_zeropad_wide_period"],
                    "power": analysis["fft_zeropad_wide_power"],
                    "peak_to_background": analysis["fft_zeropad_wide_peak_to_background"],
                    "n_points": analysis["n_points"],
                    "span_days": analysis["span_days"],
                })
    
    results_df = pd.DataFrame(results)
    print(f"✓ Analysis complete: {len(results_df)} results (before filtering)")
    
    # Filter LS results: only keep periods in 24-35 days range with FAP < 0.1
    ls_mask = (
        (results_df["method"] == "lombscargle") &
        (results_df["period"] >= period_min) &
        (results_df["period"] <= period_max) &
        (results_df["fap"] < 0.1)
    )
    
    # Keep narrow FFT results (they already search 24-35)
    fft_narrow_mask = results_df["method"].isin(["fft_interpolation", "fft_zeropad"])
    
    # Filter wide FFT results: searched 10-50, keep only 24-35
    fft_wide_mask = (
        results_df["method"].isin(["fft_interp_wide", "fft_zeropad_wide"]) &
        (results_df["period"] >= period_min) &
        (results_df["period"] <= period_max)
    )
    
    # Combine filters: LS (with FAP), narrow FFT (no filter), wide FFT (filtered)
    filtered_df = results_df[ls_mask | fft_narrow_mask | fft_wide_mask].copy()
    
    print(f"✓ After filtering: {len(filtered_df)} results")
    print(f"  LS: {ls_mask.sum()} kept, {(~ls_mask & (results_df['method'] == 'lombscargle')).sum()} dropped (FAP/range)")
    print(f"  FFT narrow: {fft_narrow_mask.sum()} kept (all)")
    print(f"  FFT wide: {fft_wide_mask.sum()} kept, {(~fft_wide_mask & results_df['method'].isin(['fft_interp_wide', 'fft_zeropad_wide'])).sum()} dropped (range)")
    
    return filtered_df


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

