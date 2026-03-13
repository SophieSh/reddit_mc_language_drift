"""Visualization functions for periodicity analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analysis import aggregate_by_day


def create_adaptive_phases(cycle_length: float, split_luteal: bool = False) -> dict[str, tuple[int, int]]:
    """Create phase definitions adapted to cycle length.
    
    Uses 0-indexed offsets where offset 0 = CD1 (first day of menstruation/anchor day).
    Note: In medical convention, CD1 = Day 1, but we use 0-indexing for offsets.
    
    Biological fact: Cycle length variation comes primarily from FOLLICULAR phase.
    - Menstrual phase: ~4 days (relatively fixed)
    - Follicular phase: 7-15 days (VARIABLE - where cycle differences occur)
    - Ovulation: ~3 days (relatively fixed)
    - Luteal phase: ~14 days (relatively fixed, though can vary slightly)
    
    For 28-day cycle: M(0-3)=CD1-4, F(4-10)=CD5-11, O(11-13)=CD12-14, L(14-27)=CD15-28
    For 32-day cycle: M(0-3)=CD1-4, F(4-14)=CD5-15, O(15-17)=CD16-18, L(18-31)=CD19-32
    
    Args:
        cycle_length: Detected cycle length in days (24-35)
        split_luteal: If True, split luteal phase into Early Luteal (7 days) and Late Luteal (7 days)
    
    Returns:
        Dictionary of {phase_name: (start_day, end_day)} where days are 0-indexed offsets
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
    
    phases = {
        'Menstrual': (0, menstrual_end),
        'Follicular': (follicular_start, follicular_end),
        'Ovulation': (ovulation_start, ovulation_end),
    }
    
    if split_luteal:
        early_luteal_end = luteal_start + 6  # First 7 days (0-6 = 7 days)
        late_luteal_start = early_luteal_end + 1
        phases['Early Luteal'] = (luteal_start, early_luteal_end)
        phases['Late Luteal'] = (late_luteal_start, luteal_end)
    else:
        phases['Luteal'] = (luteal_start, luteal_end)
    
    return phases


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


def fit_sine_wave(offsets: np.ndarray, values: np.ndarray, period: float) -> tuple[np.ndarray, dict]:
    """Fit a sine wave with given period to the data.
    
    Returns:
        fitted_values: Fitted sine wave values
        params: Dictionary with amplitude, phase, offset
    """
    mean_val = np.mean(values)
    std_val = np.std(values) + 1e-10
    values_norm = (values - mean_val) / std_val
    
    def sine_func(x, amplitude, phase, offset):
        return amplitude * np.sin(2 * np.pi * x / period + phase) + offset
    
    try:
        popt, _ = curve_fit(
            sine_func,
            offsets,
            values_norm,
            p0=[1.0, 0.0, 0.0],
            maxfev=10000
        )
        amplitude, phase, offset = popt
        
        x_fit = np.linspace(offsets.min(), offsets.max(), 1000)
        y_fit_norm = sine_func(x_fit, amplitude, phase, offset)
        y_fit = y_fit_norm * std_val + mean_val
        
        params = {
            'amplitude': amplitude * std_val,
            'phase': phase,
            'offset': offset * std_val + mean_val
        }
        
        return y_fit, params
    except:
        y_fit = np.full(1000, mean_val)
        params = {'amplitude': 0, 'phase': 0, 'offset': mean_val}
        return y_fit, params


def plot_user_timeline(
    user: str,
    feature: str,
    timeline_df: pd.DataFrame,
    ls_period: float | None,
    fft_period: float | None,
    output_path: Path,
    figsize: tuple[int, int] = (30, 6),
    dpi: int = 300,
) -> None:
    """Plot data points and fitted curves for a single user.
    
    Args:
        user: User ID
        feature: Feature name
        timeline_df: Timeline DataFrame for this user
        ls_period: Detected period from Lomb-Scargle (or None)
        fft_period: Detected period from FFT (or None)
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        dpi: Resolution for saved plot
    """
    user_df = timeline_df[timeline_df['author'] == user].copy()
    
    if feature not in user_df.columns:
        print(f"  Warning: Feature {feature} not found for user {user}")
        return
    
    user_df['_feature_value'] = user_df[feature]
    daily_agg = aggregate_by_day(
        user_df,
        offset_col='offset_from_cd1',
        value_col='_feature_value',
        agg_func='mean'
    )
    
    if len(daily_agg) < 10:
        print(f"  Warning: Insufficient data for user {user}")
        return
    
    offsets = daily_agg['offset_from_cd1'].values
    values = daily_agg['feature_mean'].values
    
    valid_mask = ~(np.isnan(offsets) | np.isnan(values))
    offsets = offsets[valid_mask]
    values = values[valid_mask]
    
    if len(offsets) < 10:
        return
    
    x_fit = np.linspace(offsets.min(), offsets.max(), 1000)
    
    if ls_period is not None and not np.isnan(ls_period) and ls_period > 0:
        y_ls, _ = fit_sine_wave(offsets, values, ls_period)
    else:
        y_ls = None
    
    if fft_period is not None and not np.isnan(fft_period) and fft_period > 0:
        y_fft, _ = fit_sine_wave(offsets, values, fft_period)
    else:
        y_fft = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sorted_idx = np.argsort(offsets)
    ax.plot(offsets[sorted_idx], values[sorted_idx], 'o-', alpha=0.7, markersize=4, 
            linewidth=1, color='gray', label='Data points (connected)', zorder=3)
    
    if y_ls is not None:
        ax.plot(x_fit, y_ls, 'r-', linewidth=2, 
                label=f'Lomb-Scargle fit (period={ls_period:.1f} days)', zorder=2)
    
    if y_fft is not None:
        ax.plot(x_fit, y_fft, 'b--', linewidth=2, 
                label=f'FFT fit (period={fft_period:.1f} days)', zorder=2)
    
    ax.set_xlabel('Days from CD1', fontsize=12)
    ax.set_ylabel(f'{feature}', fontsize=12)
    ax.set_title(f'User: {user}\n{feature} - Data points and fitted curves', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, zorder=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_cycle_distributions(
    results_df: pd.DataFrame,
    features: list[str],
    output_path: Path,
    period_wide_max: float | None = None,
) -> None:
    """Plot histograms of cycle length distributions for specified features.
    
    Args:
        results_df: DataFrame with periodicity results (long or table format)
        features: List of feature names to plot
        output_path: Path to save the plot
        period_wide_max: Maximum period for wide FFT methods (if None, auto-detect from data)
    """
    df_filtered = results_df[results_df['feature'].isin(features)].copy()
    
    if len(df_filtered) == 0:
        print(f"  ⚠ Warning: No data found for features {features}")
        return
    
    # Get unique methods
    methods = sorted(df_filtered['method'].unique())
    n_methods = len(methods)
    n_features = len(features)
    
    fig, axes = plt.subplots(n_features, n_methods, figsize=(7 * n_methods, 5 * n_features))
    
    # Handle single feature or single method case
    if n_features == 1 and n_methods == 1:
        axes = np.array([[axes]])
    elif n_features == 1:
        axes = axes.reshape(1, -1)
    elif n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle("Cycle Length Distributions", fontsize=16, fontweight="bold")
    
    method_names = {
        'lombscargle': 'Lomb-Scargle',
        'fft_interpolation': 'FFT (Interpolation)',
        'fft_zeropad': 'FFT (Zero-pad)',
        'fft_interp_wide': 'FFT (Interp-Wide)',
        'fft_zeropad_wide': 'FFT (Zero-pad-Wide)'
    }
    
    colors = {
        'lombscargle': 'steelblue',
        'fft_interpolation': 'darkorange',
        'fft_zeropad': 'forestgreen',
        'fft_interp_wide': 'coral',
        'fft_zeropad_wide': 'limegreen'
    }
    
    for feat_idx, feature in enumerate(features):
        for method_idx, method in enumerate(methods):
            ax = axes[feat_idx, method_idx]
            
            # Filter data for this feature and method
            data = df_filtered[
                (df_filtered['feature'] == feature) & 
                (df_filtered['method'] == method)
            ]['period'].dropna()
            
            if len(data) > 0:
                # Determine bin range based on method
                if 'wide' in method:
                    # For wide methods, use period_wide_max if provided, otherwise auto-detect from data
                    if period_wide_max is not None:
                        max_period = int(np.ceil(period_wide_max))
                    else:
                        # Auto-detect: use max of data or 50, whichever is larger
                        max_period = max(int(np.ceil(data.max())), 50)
                    bins = range(10, max_period + 2)
                    xlim = (9.5, max_period + 1.5)
                elif method == 'lombscargle':
                    bins = range(10, 92)  # LS searches 10-90
                    xlim = (9.5, 91.5)
                else:
                    bins = range(10, 37)  # Narrow FFT shows 10-35 to see edges
                    xlim = (9.5, 36.5)
                
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, 
                       color=colors.get(method, 'gray'))
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {data.mean():.1f}d')
                ax.axvline(data.median(), color='blue', linestyle='--', linewidth=2, 
                          label=f'Median: {data.median():.1f}d')
                ax.set_xlabel('Period (days)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{feature}\n{method_names.get(method, method)} (n={len(data)})')
                ax.legend()
                ax.grid(alpha=0.3)
                ax.set_xlim(xlim)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{feature}\n{method_names.get(method, method)} (no data)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def aggregate_features_by_phase(
    timeline_df: pd.DataFrame,
    results_df: pd.DataFrame,
    features: list[str],
    time_col: str = 'offset_from_cd1',
    user_col: str = 'author',
    method: str = 'fft_interpolation',
    normalize: bool = True,
    time_window: tuple[float, float] | None = None,
    fixed_cycle_length: float | None = None,
    average_per_user: bool = True,
) -> pd.DataFrame:
    """Aggregate features by phase for users with detected cycles.
    
    Uses the OLD EXPERIMENT APPROACH that produces small error bars:
    1. Filter to users with detected cycles
    2. Optional: filter to time window (e.g., 0-30 days)
    3. DAILY AGGREGATION FIRST (per user per day)
    4. Per-user normalization on daily aggregates (optional)
    5. Phase assignment based on detected cycle length
    6. Aggregate by phase using user-day means
    7. Error bars = SEM of user-day means
    
    Args:
        timeline_df: Timeline DataFrame with features and time column
        results_df: Periodicity results DataFrame
        features: List of features to analyze
        time_col: Time column name ('offset_from_cd1' or 'dpo_days')
        user_col: User column name
        method: Method to use for cycle length (default: 'fft_interpolation')
        normalize: Whether to apply per-user normalization (default: True)
        time_window: Optional (min_day, max_day) to filter timeline (e.g., (0, 30))
        fixed_cycle_length: If not None, override per-user detected period with this
            fixed cycle length in days (e.g., 28) when assigning phases.
        average_per_user: If True (default), average per user per phase first, then aggregate
            across users (SEM from user count). If False, use all user-days directly
            (SEM from user-day count). Default: True.
    
    Returns:
        DataFrame with columns: feature, phase, mean, sem, n_user_days, n_users
    """
    # If fixed_cycle_length is provided, use all users from timeline (no periodicity filtering)
    if fixed_cycle_length is not None:
        print(f"  Using fixed cycle length for all users: {fixed_cycle_length} days")
        print(f"  Including all users from timeline (no periodicity filtering)...")
        
        # Get all unique users from timeline
        all_users = set(timeline_df[user_col].unique())
        
        # Create user_cycles DataFrame with fixed cycle length for all users
        user_cycles = pd.DataFrame({
            user_col: list(all_users),
            'period': float(fixed_cycle_length)
        })
        
        timeline_valid = timeline_df.copy()
        print(f"  Processing {len(all_users)} users from timeline...")
    else:
        # Filter to users with detected cycles for the specified method
        # Note: No period range filter needed - periodicity detection already enforces
        # the search range (21-35 days from config) during detection
        valid_results = results_df[
            (results_df['method'] == method) & 
            (results_df['period'].notna())
        ].copy()
        
        if len(valid_results) == 0:
            print(f"  ⚠ No valid cycle detections for method {method}")
            return pd.DataFrame()
        
        # Results DataFrame uses 'user' column, timeline uses user_col (typically 'author')
        results_user_col = 'user' if 'user' in valid_results.columns else user_col
        
        # Get unique users with their cycle lengths
        user_cycles = valid_results[[results_user_col, 'period']].drop_duplicates()
        valid_users = set(user_cycles[results_user_col])
        
        print(f"  Processing {len(valid_users)} users with detected cycles...")
        
        # Pre-filter timeline to only valid users
        timeline_valid = timeline_df[timeline_df[user_col].isin(valid_users)].copy()
    
    if len(timeline_valid) == 0:
        print(f"  ⚠ No timeline data for valid users")
        return pd.DataFrame()
    
    # Step 1: DAILY AGGREGATION on FULL timeline (for valid users)
    # This removes within-user-day variance
    print(f"  Step 1: Daily aggregation on full timeline (per user per day)...")
    
    agg_dict = {f: 'mean' for f in features if f in timeline_valid.columns}
    if not agg_dict:
        print(f"  ⚠ No valid features found in timeline")
        return pd.DataFrame()
    
    daily_full = timeline_valid.groupby([user_col, time_col]).agg(agg_dict).reset_index()
    print(f"    → {len(daily_full)} user-days from {daily_full[user_col].nunique()} users (full timeline)")
    
    # Step 2: Per-user normalization on FULL daily aggregates (if enabled)
    # This ensures consistent baseline regardless of time window
    if normalize:
        print(f"  Step 2: Applying per-user normalization on full timeline daily aggregates...")
        for feature in features:
            if feature not in daily_full.columns:
                continue
            
            # Full z-score normalization on FULL timeline
            user_means = daily_full.groupby(user_col)[feature].transform('mean')
            user_stds = daily_full.groupby(user_col)[feature].transform('std')
            daily_full[f'{feature}_zscore'] = (daily_full[feature] - user_means) / (user_stds + 1e-10)
            daily_full[f'{feature}_zscore'] = daily_full[f'{feature}_zscore'].replace([np.inf, -np.inf], np.nan)
        
        # Use z-score versions for analysis
        features_to_analyze = [f'{f}_zscore' for f in features if f'{f}_zscore' in daily_full.columns]
    else:
        features_to_analyze = features
    
    # Step 3: Apply time window filter AFTER normalization (if specified)
    if time_window is not None:
        min_day, max_day = time_window
        daily_agg = daily_full[
            (daily_full[time_col] >= min_day) &
            (daily_full[time_col] <= max_day)
        ].copy()
        print(f"  Filtered to {time_col} in [{min_day}, {max_day}]: {len(daily_agg)} user-days")
    else:
        daily_agg = daily_full.copy()
    
    # Step 4: Merge cycle lengths and assign phases
    print(f"  Step 4: Assigning phases based on cycle lengths...")
    # Ensure user_cycles has the correct column name for merging
    if fixed_cycle_length is None:
        # When using periodicity results, rename 'user' to user_col if needed
        if 'user' in user_cycles.columns and user_col != 'user':
            user_cycles = user_cycles.rename(columns={'user': user_col})
    
    daily_agg = daily_agg.merge(
        user_cycles,
        on=user_col,
        how='left'
    )
    
    def assign_phase_vectorized(row):
        cycle_length = row['period']
        day = row[time_col]
        if pd.isna(cycle_length) or pd.isna(day):
            return None
        phase_definition = create_adaptive_phases(cycle_length)
        return assign_phase_to_day(day, phase_definition)
    
    daily_agg['phase'] = daily_agg.apply(assign_phase_vectorized, axis=1)
    daily_agg = daily_agg[daily_agg['phase'].notna()].copy()
    
    if len(daily_agg) == 0:
        print(f"  ⚠ No user-days assigned to phases")
        return pd.DataFrame()
    
    print(f"    → {len(daily_agg)} user-days assigned to phases")
    
    # Print diagnostic: users and user-days per phase (after timeframe filtering)
    print(f"\n  Phase distribution (after timeframe requirements):")
    phase_summary = daily_agg.groupby('phase').agg({
        user_col: 'nunique',
        time_col: 'count'
    }).reset_index()
    phase_summary.columns = ['phase', 'n_users', 'n_user_days']
    phase_order = ['Menstrual', 'Follicular', 'Ovulation', 'Luteal']
    for phase in phase_order:
        phase_row = phase_summary[phase_summary['phase'] == phase]
        if len(phase_row) > 0:
            print(f"    {phase}: {phase_row['n_users'].iloc[0]:,} users, {phase_row['n_user_days'].iloc[0]:,} user-days")
    print()
    
    # Step 4: Aggregate by phase from daily data
    if average_per_user:
        # First average per user per phase, then aggregate across users
        # This gives equal weight to each user regardless of posting volume
        print(f"  Step 4: Aggregating by phase (averaging per user per phase first)...")
    else:
        # Use all user-days directly (users with more posts contribute more)
        print(f"  Step 4: Aggregating by phase (using all user-days directly)...")
    phase_results = []
    
    for feature in features_to_analyze:
        if feature not in daily_agg.columns:
            continue
        
        feature_data = daily_agg[[user_col, 'phase', feature]].copy()
        feature_data = feature_data[feature_data[feature].notna()].copy()
        
        if len(feature_data) == 0:
            continue
        
        if average_per_user:
            # Step 4a: Average per user per phase (one value per user per phase)
            # This removes within-user, within-phase variance
            user_phase_means = feature_data.groupby([user_col, 'phase'])[feature].mean().reset_index()
            # The column name after reset_index() is the feature name itself
            
            # Step 4b: Aggregate across users (treating each user as one observation per phase)
            phase_stats = user_phase_means.groupby('phase')[feature].agg(['mean', 'std', 'count']).reset_index()
            phase_stats.columns = ['phase', 'mean', 'std', 'n_users']
            
            # Calculate SEM from user means (n = number of users)
            phase_stats['sem'] = phase_stats['std'] / np.sqrt(phase_stats['n_users'])
            
            # Count user-days for reference (but SEM is computed from users, not user-days)
            user_days_per_phase = feature_data.groupby('phase').size().reset_index()
            user_days_per_phase.columns = ['phase', 'n_user_days']
            phase_stats = phase_stats.merge(user_days_per_phase, on='phase')
        else:
            # Use all user-days directly (no per-user averaging)
            phase_stats = feature_data.groupby('phase')[feature].agg(['mean', 'std', 'count']).reset_index()
            phase_stats.columns = ['phase', 'mean', 'std', 'n_user_days']
            
            # Calculate SEM from user-days (n = number of user-days)
            phase_stats['sem'] = phase_stats['std'] / np.sqrt(phase_stats['n_user_days'])
            
            # Count users for reference (but SEM is computed from user-days, not users)
            users_per_phase = feature_data.groupby('phase')[user_col].nunique().reset_index()
            users_per_phase.columns = ['phase', 'n_users']
            phase_stats = phase_stats.merge(users_per_phase, on='phase')
        
        # Get original feature name (remove normalization suffix if present)
        original_feature = feature.replace('_zscore', '').replace('_centered', '')
        
        for _, row in phase_stats.iterrows():
            phase_results.append({
                'feature': original_feature,
                'phase': row['phase'],
                'mean': row['mean'],
                'std': row['std'],
                'sem': row['sem'],
                'n_user_days': int(row['n_user_days']),
                'n_users': int(row['n_users']),
            })
    
    if len(phase_results) == 0:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(phase_results)
    
    # Print diagnostic: final user and user-day counts per phase (from aggregated data)
    print(f"\n  Final phase distribution (from aggregated data):")
    phase_final = result_df.groupby('phase').agg({
        'n_users': 'first',  # n_users should be same for all features per phase
        'n_user_days': 'first'  # n_user_days should be same for all features per phase
    }).reset_index()
    phase_order = ['Menstrual', 'Follicular', 'Ovulation', 'Luteal']
    for phase in phase_order:
        phase_row = phase_final[phase_final['phase'] == phase]
        if len(phase_row) > 0:
            print(f"    {phase}: {phase_row['n_users'].iloc[0]:,} users, {phase_row['n_user_days'].iloc[0]:,} user-days")
    print()
    
    print(f"  ✓ Aggregation complete")
    return result_df


def plot_phase_analysis(
    phase_df: pd.DataFrame,
    features: list[str],
    output_path: Path,
    error_bar_type: str = 'sem',
    average_per_user: bool = True,
) -> None:
    """Plot bar charts showing features across menstrual phases.
    
    Args:
        phase_df: Aggregated phase DataFrame (from aggregate_features_by_phase)
        features: List of features to plot
        output_path: Path to save the plot
        error_bar_type: Type of error bars to use:
            - 'sem': Standard Error of the Mean (std / sqrt(n)) - shows precision of mean estimate
            - 'std': Standard deviation - shows raw variability in the data
            Default: 'sem'
        average_per_user: If True, display n_users (number of users). If False, display n_user_days
            (total user-days). Should match the aggregation mode used. Default: True
    """
    if len(phase_df) == 0:
        print(f"  ⚠ No phase data to visualize")
        return
    
    phase_order = ['Menstrual', 'Follicular', 'Ovulation', 'Luteal']
    phase_colors = {
        'Menstrual': '#d62728',
        'Follicular': '#ff7f0e',
        'Ovulation': '#2ca02c',
        'Luteal': '#1f77b4',
    }
    # Bar widths relative to Ovulation (1x baseline)
    # Ovulation: 1x, Menstrual: 1.5x, Follicular: 2x, Luteal: 4.5x
    phase_widths = {
        'Menstrual': 1.5,
        'Follicular': 2.0,
        'Ovulation': 1.0,
        'Luteal': 4.5,
    }
    
    n_features = len(features)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    error_label = 'SEM' if error_bar_type == 'sem' else 'Std Dev'
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f"Feature Values by Menstrual Phase (Error bars = {error_label})", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        feature_data = phase_df[phase_df['feature'] == feature]
        
        # Clean feature name for display: remove _mean suffix and replace underscores
        feature_display = feature.replace('_mean', '').replace('_', ' ').title()
        
        if len(feature_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_display)
            continue
        
        means = []
        errors = []
        n_list = []
        colors_list = []
        
        for phase in phase_order:
            phase_row = feature_data[feature_data['phase'] == phase]
            if len(phase_row) > 0:
                means.append(phase_row.iloc[0]['mean'])
                # Choose error bar type
                if error_bar_type == 'std':
                    errors.append(phase_row.iloc[0]['std'])
                else:  # default to sem
                    errors.append(phase_row.iloc[0]['sem'])
                # Use n_users when averaging per user, n_user_days when using all user-days
                if average_per_user:
                    n_list.append(int(phase_row.iloc[0]['n_users']))
                else:
                    n_list.append(int(phase_row.iloc[0]['n_user_days']))
                colors_list.append(phase_colors[phase])
            else:
                means.append(0)
                errors.append(0)
                n_list.append(0)
                colors_list.append('#cccccc')
        
        # Calculate bar widths based on phase lengths
        widths = [phase_widths[phase] for phase in phase_order]
        # Calculate x positions to center bars with different widths
        # Start at 0, then position each bar's center based on cumulative width
        x_positions = []
        current_pos = 0
        spacing = 0.2  # Spacing between bars
        for width in widths:
            x_positions.append(current_pos + width / 2)
            current_pos += width + spacing
        
        bars = ax.bar(x_positions, means, width=widths, yerr=errors, capsize=5, 
                     color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add n labels (users or user-days depending on aggregation mode)
        for bar, n in zip(bars, n_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'n={n}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(phase_order, rotation=45, ha='right')
        # Set x-axis limits to show all bars with some padding
        ax.set_xlim(-0.5, current_pos + 0.5)
        ax.set_ylabel('Mean Value (Normalized)')
        ax.set_title(feature_display, fontweight='bold')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle at top
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

