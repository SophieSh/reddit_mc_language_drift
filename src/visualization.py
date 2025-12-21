"""Visualization functions for periodicity analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analysis import aggregate_by_day


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
) -> None:
    """Plot histograms of cycle length distributions for specified features.
    
    Args:
        results_df: DataFrame with periodicity results (long or table format)
        features: List of feature names to plot
        output_path: Path to save the plot
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
                # Determine bin range based on method (wide methods show 10-50, narrow show 10-35)
                if 'wide' in method:
                    bins = range(10, 52)
                    xlim = (9.5, 51.5)
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

