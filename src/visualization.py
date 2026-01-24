"""Visualization functions for periodicity analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analysis import aggregate_by_day


from src.analysis import create_adaptive_phases, assign_phase_to_day


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
    
    Returns:
        DataFrame with columns: feature, phase, mean, sem, n_user_days, n_users
    """
    # Filter to users with detected cycles for the specified method
    valid_results = results_df[
        (results_df['method'] == method) & 
        (results_df['period'].notna()) &
        (results_df['period'] >= 24) &
        (results_df['period'] <= 35)
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
    timeline_filtered = timeline_df[timeline_df[user_col].isin(valid_users)].copy()
    
    if len(timeline_filtered) == 0:
        print(f"  ⚠ No timeline data for valid users")
        return pd.DataFrame()
    
    # Merge cycle lengths into timeline
    timeline_filtered = timeline_filtered.merge(
        user_cycles.rename(columns={results_user_col: user_col}),
        on=user_col,
        how='left'
    )
    
    # Vectorized phase assignment
    def assign_phase_vectorized(row):
        cycle_length = row['period']
        day = row[time_col]
        if pd.isna(cycle_length) or pd.isna(day):
            return None
        phase_definition = create_adaptive_phases(cycle_length)
        return assign_phase_to_day(day, phase_definition)
    
    print(f"  Assigning phases to {len(timeline_filtered)} posts...")
    timeline_filtered['phase'] = timeline_filtered.apply(assign_phase_vectorized, axis=1)
    timeline_filtered = timeline_filtered[timeline_filtered['phase'].notna()].copy()
    
    if len(timeline_filtered) == 0:
        print(f"  ⚠ No posts assigned to phases")
        return pd.DataFrame()
    
    # Now aggregate by feature, user, and phase
    phase_results = []
    
    for feature in features:
        if feature not in timeline_filtered.columns:
            continue
        
        feature_df = timeline_filtered[[user_col, time_col, 'phase', feature]].copy()
        feature_df = feature_df[feature_df[feature].notna()].copy()
        
        if len(feature_df) == 0:
            continue
        
        # Aggregate by user, day, and phase first (avoid multiple posts per day bias)
        daily_agg = feature_df.groupby([user_col, time_col, 'phase']).agg({
            feature: 'mean'
        }).reset_index()
        
        # Then aggregate by user and phase
        user_phase_agg = daily_agg.groupby([user_col, 'phase']).agg({
            feature: ['mean', 'std', 'count']
        }).reset_index()
        
        user_phase_agg.columns = [user_col, 'phase', 'mean', 'std', 'n_days']
        
        for _, row in user_phase_agg.iterrows():
            phase_results.append({
                'feature': feature,
                'phase': row['phase'],
                'user': row[user_col],
                'mean': row['mean'],
                'std': row['std'] if not pd.isna(row['std']) else 0.0,
                'n_days': int(row['n_days']),
            })
    
    if len(phase_results) == 0:
        return pd.DataFrame()
    
    phase_df = pd.DataFrame(phase_results)
    
    # Aggregate across users
    aggregated = phase_df.groupby(['feature', 'phase']).agg({
        'mean': ['mean', 'std'],
        'n_days': 'sum',
        'user': 'nunique'
    }).reset_index()
    
    aggregated.columns = ['feature', 'phase', 'mean', 'std', 'n_days', 'n_users']
    
    return aggregated


def plot_phase_analysis(
    phase_df: pd.DataFrame,
    features: list[str],
    output_path: Path,
) -> None:
    """Plot bar charts showing features across menstrual phases.
    
    Args:
        phase_df: Aggregated phase DataFrame (from aggregate_features_by_phase)
        features: List of features to plot
        output_path: Path to save the plot
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
    
    n_features = len(features)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Feature Values by Menstrual Phase", fontsize=16, fontweight='bold')
    
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        feature_data = phase_df[phase_df['feature'] == feature]
        
        if len(feature_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature.replace('_', ' ').title())
            continue
        
        means = []
        stds = []
        n_users_list = []
        colors_list = []
        
        for phase in phase_order:
            phase_row = feature_data[feature_data['phase'] == phase]
            if len(phase_row) > 0:
                means.append(phase_row.iloc[0]['mean'])
                stds.append(phase_row.iloc[0]['std'])
                n_users_list.append(int(phase_row.iloc[0]['n_users']))
                colors_list.append(phase_colors[phase])
            else:
                means.append(0)
                stds.append(0)
                n_users_list.append(0)
                colors_list.append('#cccccc')
        
        x = np.arange(len(phase_order))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_list, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add n_users labels
        for bar, n in zip(bars, n_users_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'n={n}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

