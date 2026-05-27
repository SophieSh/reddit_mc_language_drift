#!/usr/bin/env python3
"""Step 34: SVD-based spectrum analysis to identify dominant periodic structure.

For each user the script:
  1. Interpolates each feature onto a regular daily grid.
  2. Computes an FFT power spectrum evaluated at 100 period points in [period_min, period_max].
  3. Stacks spectra into a (n_variables x 100) matrix and runs SVD.
  4. Derives a per-user cycling score (variance in first component) and variable importance.

Globally it identifies the top-k most important variables, re-scores users on those
variables only, selects the top-frac% cycling users, and plots the mean dominant spectrum.

Output files (in outputs/):
  34_variable_importance.png
  34_dominant_spectrum.png

Output CSV (in data/interim/):
  svd_cycling_scores_{timestamp}.csv   [columns: user, cycling_score, in_top50]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend as scipy_detrend
from tqdm import tqdm

from src.io import find_latest_file, save_with_timestamp

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
_MIN_OBS_PER_VARIABLE = 10   # minimum non-NaN observations to include a variable for a user
_EPSILON = 1e-12              # numerical guard for z-score normalisation


def _parabolic_peak(spectrum: np.ndarray, period_grid: np.ndarray) -> float:
    """Refine argmax with parabolic interpolation — avoids grid-point bias on broad peaks."""
    idx = int(np.argmax(spectrum))
    if idx == 0 or idx == len(spectrum) - 1:
        return float(period_grid[idx])
    d = float(period_grid[1] - period_grid[0])
    y0, y1, y2 = float(spectrum[idx - 1]), float(spectrum[idx]), float(spectrum[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-15:
        return float(period_grid[idx])
    return float(period_grid[idx]) + 0.5 * d * (y0 - y2) / denom


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class UserSVDResult(NamedTuple):
    """SVD decomposition result for a single user."""

    user: str
    cycling_score: float          # S[0]^2 / sum(S^2)
    s0: float                     # S[0] — first singular value, used as averaging weight
    dominant_spectrum: np.ndarray  # shape (n_periods,), Vt[0,:]²
    var_importances: np.ndarray    # shape (n_included_vars,), |U[:,0]| / n_included_vars
    included_var_indices: list[int]  # indices into the global variable list


# ---------------------------------------------------------------------------
# Step 1 helper: per-(user, variable) power spectrum
# ---------------------------------------------------------------------------

def _power_spectrum_at_periods(
    offsets: np.ndarray,
    values: np.ndarray,
    period_grid: np.ndarray,
) -> np.ndarray | None:
    """Compute FFT power interpolated onto ``period_grid`` (in days).

    Parameters
    ----------
    offsets:
        Integer day offsets for a single user/feature (irregular, no NaNs).
    values:
        Corresponding feature values (no NaNs, same length as offsets).
    period_grid:
        1-D array of period values (days) at which to evaluate power; must be
        strictly positive and in ascending order of corresponding frequencies.

    Returns
    -------
    Power array of shape ``(len(period_grid),)`` or ``None`` if the signal is
    too short to support the requested period range.
    """
    offsets = offsets.astype(int)
    day_min, day_max = int(offsets.min()), int(offsets.max())
    span = day_max - day_min

    # Need at least 2 full cycles of the longest requested period to get
    # meaningful spectral resolution, but we relax this to allow users with
    # moderate data.  A hard floor of period_max * 1.5 is pragmatic.
    if span < float(period_grid.max()) * 1.5:
        return None

    # Build regular grid and interpolate
    grid_days = np.arange(day_min, day_max + 1, dtype=float)
    n = len(grid_days)

    # Sort by offset before interpolation (input may be unsorted)
    sort_idx = np.argsort(offsets)
    interp_values = np.interp(grid_days, offsets[sort_idx].astype(float), values[sort_idx])

    # Z-score time series so spectra are comparable across variables
    ts_std = interp_values.std()
    if ts_std > _EPSILON:
        interp_values = (interp_values - interp_values.mean()) / ts_std
    else:
        interp_values = interp_values - interp_values.mean()

    # Direct DFT evaluated at exactly the 100 requested period points
    target_freqs = 1.0 / period_grid  # (n_periods,) cycles per day
    phases = 2.0 * np.pi * np.outer(target_freqs, grid_days)
    dft = interp_values @ np.exp(-1j * phases).T  # (n_periods,)
    return np.abs(dft) ** 2


# ---------------------------------------------------------------------------
# Step 2 helper: per-user SVD
# ---------------------------------------------------------------------------

def _user_svd(
    spectrum_matrix: np.ndarray,
    included_var_indices: list[int],
    I_minus_H: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """SVD on per-variable log-power spectra after log-T detrending.

    Log-T detrend: fit log(P) as a linear function of log(T) and subtract.
    This removes a pure power-law red-noise trend (P ∝ T^α) exactly, leaving
    only deviations from the power law.  Unlike linear detrend in period space,
    the log-T detrend has no boundary-dependent concavity artifact — moving
    period_min from 20 to 24 does not shift the detected peak.

    Row z-score equalises remaining amplitude differences between variables.
    Sign convention: flip so sum(U[:,0]) > 0.
    Importances: raw projection of the un-z-scored residual onto the dominant
    direction, preserving amplitude differences across variables.
    """
    log_matrix = np.log1p(spectrum_matrix)
    # Apply log-T detrend via precomputed projection matrix (I - H)
    detrended = log_matrix @ I_minus_H if I_minus_H is not None else log_matrix

    # Row z-score so no single variable dominates the SVD direction
    row_stds = detrended.std(axis=1, keepdims=True) + _EPSILON
    normalized_matrix = detrended / row_stds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, S, Vt = np.linalg.svd(normalized_matrix, full_matrices=False)

    cycling_score = float(S[0] ** 2 / (np.sum(S ** 2) + _EPSILON))
    s0 = float(S[0])

    dominant_spectrum = Vt[0, :]
    if U[:, 0].sum() < 0:
        dominant_spectrum = -dominant_spectrum

    # Raw (un-z-scored) projection preserves amplitude → non-uniform importances
    raw_projections = detrended @ dominant_spectrum
    var_importances = np.abs(raw_projections) / (np.abs(raw_projections).sum() + _EPSILON)

    return cycling_score, s0, dominant_spectrum, var_importances


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_spectrum_cube(
    df: pd.DataFrame,
    feature_cols: list[str],
    period_grid: np.ndarray,
    min_obs: int = _MIN_OBS_PER_VARIABLE,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the 3-D spectrum array (n_vars x n_users x n_periods).

    Missing (user, variable) combinations are filled with NaN rows so the
    array stays rectangular.

    Parameters
    ----------
    df:
        Input DataFrame with columns ``author``, ``offset_from_cd1``, and all
        feature columns.
    feature_cols:
        Ordered list of ``_mean`` column names.
    period_grid:
        1-D array of period values at which to evaluate power.
    min_obs:
        Minimum number of non-NaN observations for a variable/user pair to be
        included.

    Returns
    -------
    cube : np.ndarray
        Shape (n_vars, n_users, n_periods); NaN where no valid spectrum exists.
    users : list[str]
        Ordered list of user identifiers (axis 1 of cube).
    variables : list[str]
        Ordered list of feature column names (axis 0 of cube).
    """
    users = sorted(df["author"].unique())
    n_vars = len(feature_cols)
    n_users = len(users)
    n_periods = len(period_grid)

    cube = np.full((n_vars, n_users, n_periods), np.nan, dtype=np.float32)

    user_index = {u: i for i, u in enumerate(users)}

    for vi, feat in enumerate(tqdm(feature_cols, desc="Building spectrum cube", unit="var", disable=not show_progress)):
        sub = df[["author", "offset_from_cd1", feat]].dropna(subset=[feat])
        for user, grp in sub.groupby("author", sort=False):
            if len(grp) < min_obs:
                continue
            ui = user_index[user]
            offsets = grp["offset_from_cd1"].values
            values = grp[feat].values
            spectrum = _power_spectrum_at_periods(offsets, values, period_grid)
            if spectrum is not None:
                cube[vi, ui, :] = spectrum.astype(np.float32)

    return cube, users, feature_cols


def compute_svd_per_user(
    cube: np.ndarray,
    users: list[str],
    feature_cols: list[str],
    var_subset: list[int] | None = None,
    I_minus_H: np.ndarray | None = None,
    show_progress: bool = True,
) -> list[UserSVDResult]:
    """Run per-user SVD over the spectrum cube.

    Parameters
    ----------
    cube:
        Shape (n_vars, n_users, n_periods).
    users:
        User identifiers, length n_users.
    feature_cols:
        Feature column names, length n_vars.
    var_subset:
        If provided, use only these variable indices (0-based into n_vars axis).

    Returns
    -------
    List of UserSVDResult, one per user.  Users for whom fewer than 2 variable
    spectra are available are skipped.
    """
    if var_subset is None:
        var_subset = list(range(cube.shape[0]))

    results: list[UserSVDResult] = []

    for ui, user in enumerate(tqdm(users, desc="Per-user SVD", unit="user", disable=not show_progress)):
        # Identify variables with a valid spectrum for this user
        valid_vi = [vi for vi in var_subset if not np.any(np.isnan(cube[vi, ui, :]))]

        if len(valid_vi) < 2:
            # Cannot run SVD with fewer than 2 variables
            continue

        M = cube[valid_vi, ui, :].astype(np.float64)  # (n_valid_vars, n_periods)

        cycling_score, s0, dominant_spectrum, var_importances = _user_svd(M, valid_vi,
                                                                           I_minus_H=I_minus_H)

        results.append(
            UserSVDResult(
                user=user,
                cycling_score=cycling_score,
                s0=s0,
                dominant_spectrum=dominant_spectrum,
                var_importances=var_importances,
                included_var_indices=valid_vi,
            )
        )

    return results


def compute_mean_variable_importance(
    svd_results: list[UserSVDResult],
    n_vars: int,
) -> np.ndarray:
    """Average |U[:,0]| loadings across users, accounting for per-user missingness.

    Parameters
    ----------
    svd_results:
        Output of ``compute_svd_per_user``.
    n_vars:
        Total number of variables.

    Returns
    -------
    mean_importance : np.ndarray
        Shape (n_vars,); NaN for variables not observed in any user.
    """
    accumulator = np.zeros(n_vars, dtype=np.float64)
    counts = np.zeros(n_vars, dtype=np.int64)

    for res in svd_results:
        for local_i, global_vi in enumerate(res.included_var_indices):
            accumulator[global_vi] += res.var_importances[local_i]
            counts[global_vi] += 1

    with np.errstate(invalid="ignore"):
        mean_importance = np.where(counts > 0, accumulator / counts, np.nan)

    return mean_importance


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def plot_diagnostic_spectra(
    cube: np.ndarray,
    period_grid: np.ndarray,
    feature_cols: list[str],
    output_path: Path,
    dataset_name: str,
) -> None:
    """Plot mean log-power spectrum across users — per variable and grand mean.

    Two panels:
      Top: one faint line per variable (mean across users who have valid spectra),
           plus a thick line for the grand mean across all variables.
      Bottom: grand mean only, with a vertical marker at the peak.
    """
    n_vars, n_users, n_periods = cube.shape

    # Mean over users (axis=1), ignoring NaNs → shape (n_vars, n_periods)
    with np.errstate(invalid="ignore"):
        per_var_mean = np.nanmean(cube.astype(np.float64), axis=1)

    # Grand mean across variables → shape (n_periods,)
    grand_mean = np.nanmean(per_var_mean, axis=0)
    peak_idx = int(np.argmax(grand_mean))
    peak_period = float(period_grid[peak_idx])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Panel 1: all variables ---
    ax = axes[0]
    for vi in range(n_vars):
        ax.plot(period_grid, per_var_mean[vi], color="#aec6e8", linewidth=0.6, alpha=0.6)
    ax.plot(period_grid, grand_mean, color="#d7191c", linewidth=2.5, label="Grand mean")
    ax.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Mean log(1 + power)")
    ax.set_title(
        f"Diagnostic: raw mean spectra — {dataset_name}\n"
        f"Faint lines = {n_vars} variables; red = grand mean across variables"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel 2: grand mean only ---
    ax2 = axes[1]
    ax2.plot(period_grid, grand_mean, color="#2c7bb6", linewidth=2)
    ax2.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.5,
                label=f"Peak: {peak_period:.2f} d")
    ax2.annotate(
        f"{peak_period:.2f} d",
        xy=(peak_period, float(grand_mean[peak_idx])),
        xytext=(peak_period + 0.5, float(grand_mean[peak_idx]) * 0.98),
        fontsize=10, color="#d7191c",
        arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.2),
    )
    ax2.set_xlabel("Period (days)")
    ax2.set_ylabel("Mean log(1 + power)")
    ax2.set_title("Grand mean spectrum (no SVD)")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved diagnostic: {output_path}")
    print(f"  Diagnostic grand-mean peak: {peak_period:.2f} days")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_variable_importance(
    mean_importance: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int,
    dataset_name: str,
    n_users: int,
) -> None:
    """Bar chart of mean variable importance, sorted descending.

    Parameters
    ----------
    mean_importance:
        Array of shape (n_vars,) with mean |U[:,0]| per variable.
    feature_names:
        Human-readable names corresponding to each index.
    output_path:
        File path to save the figure.
    top_k:
        Number of top variables to highlight.
    dataset_name:
        Short string used in the figure title.
    n_users:
        Number of users processed (for title annotation).
    """
    valid_mask = ~np.isnan(mean_importance)
    imp = mean_importance.copy()
    imp[~valid_mask] = 0.0

    sort_idx = np.argsort(imp)[::-1]
    sorted_imp = imp[sort_idx]
    sorted_names = [feature_names[i].replace("_mean", "") for i in sort_idx]

    n_shown = min(len(sorted_names), 40)  # cap at 40 bars for readability
    colors = [
        "#2c7bb6" if i < top_k else "#d7e8f5"
        for i in range(n_shown)
    ]

    fig, ax = plt.subplots(figsize=(max(14, n_shown * 0.35), 6))
    ax.bar(range(n_shown), sorted_imp[:n_shown], color=colors, edgecolor="none")
    ax.set_xticks(range(n_shown))
    ax.set_xticklabels(sorted_names[:n_shown], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean |U[:,0]| (variable loading on first SVD component)")
    ax.set_xlabel("Feature variable")
    ax.set_title(
        f"SVD Variable Importance — {dataset_name}\n"
        f"n={n_users} users | top-{top_k} highlighted in blue"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_dominant_spectrum(
    mean_spectrum: np.ndarray,
    period_grid: np.ndarray,
    output_path: Path,
    n_top_users: int,
    n_total_users: int,
    dataset_name: str,
) -> float:
    """Plot mean dominant spectrum and return the peak period.

    Parameters
    ----------
    mean_spectrum:
        Shape (n_periods,).  Averaged dominant spectrum across top-cycling users.
    period_grid:
        Corresponding period values in days.
    output_path:
        File path to save the figure.
    n_top_users:
        Number of users included in the average.
    n_total_users:
        Total users before the top-frac filter.
    dataset_name:
        Short string used in the figure title.

    Returns
    -------
    peak_period : float
        Period (days) at which mean_spectrum is maximised.
    """
    peak_period = _parabolic_peak(mean_spectrum, period_grid)
    peak_idx = int(np.argmax(mean_spectrum))   # for annotation y-position

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(period_grid, mean_spectrum, color="#2c7bb6", linewidth=2, label="Mean dominant spectrum")
    ax.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.5,
               label=f"Peak: {peak_period:.2f} d (parabolic)")
    ax.annotate(
        f"{peak_period:.2f} d",
        xy=(peak_period, float(mean_spectrum[peak_idx])),
        xytext=(peak_period + 0.4, float(mean_spectrum[peak_idx]) * 0.95),
        fontsize=10,
        color="#d7191c",
        arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.2),
    )
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Mean Vt[0,:]² (squared first right singular vector)")
    ax.set_title(
        f"Dominant Spectral Component — {dataset_name}\n"
        f"top-cycling users: {n_top_users}/{n_total_users}"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return peak_period


# ---------------------------------------------------------------------------
# Surrogate test
# ---------------------------------------------------------------------------

def run_surrogate_test(
    cube: np.ndarray,
    users: list[str],
    feature_cols: list[str],
    period_grid: np.ndarray,
    top_vars: int,
    top_frac: float,
    I_minus_H: np.ndarray,
    n_surrogates: int,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Null distribution by independently permuting user indices per variable.

    For each surrogate iteration, each variable's row of user spectra is
    independently re-ordered.  This breaks the cross-variable coherence that
    SVD exploits, while preserving per-(user,variable) spectral structure.
    Under the null there is no shared periodic signal — any detected peak is
    due to chance co-occurrence of spectral shapes across variables.

    Returns
    -------
    surrogate_peaks : np.ndarray, shape (n_surrogates,)
        Parabolic-interpolated peak period for each surrogate.
    surrogate_mean_spectra : list of np.ndarray, each shape (n_periods,)
    """
    rng = np.random.default_rng(seed)
    n_vars, n_users, _ = cube.shape
    surrogate_peaks: list[float] = []
    surrogate_mean_spectra: list[np.ndarray] = []

    shuffled_cube = np.empty_like(cube)

    for _ in tqdm(range(n_surrogates), desc="Surrogate iterations", unit="iter"):
        for vi in range(n_vars):
            shuffled_cube[vi] = cube[vi, rng.permutation(n_users), :]

        svd_all = compute_svd_per_user(
            shuffled_cube, users, feature_cols,
            I_minus_H=I_minus_H, show_progress=False,
        )
        if len(svd_all) < 2:
            continue

        mean_imp = compute_mean_variable_importance(svd_all, len(feature_cols))
        valid_imp = np.where(np.isnan(mean_imp), -np.inf, mean_imp)
        top_vi = [int(i) for i in np.argsort(valid_imp)[::-1][:top_vars]]

        svd_top = compute_svd_per_user(
            shuffled_cube, users, feature_cols,
            var_subset=top_vi, I_minus_H=I_minus_H, show_progress=False,
        )
        if len(svd_top) < 2:
            continue

        scores = np.array([r.cycling_score for r in svd_top])
        thresh = float(np.quantile(scores, 1.0 - top_frac))
        top_res = [r for r in svd_top if r.cycling_score >= thresh]
        if not top_res:
            continue

        specs = np.stack([r.dominant_spectrum for r in top_res])
        w = np.array([r.s0 for r in top_res])
        w = w / (w.sum() + _EPSILON)
        mean_spec = (specs * w[:, np.newaxis]).sum(axis=0)

        surrogate_peaks.append(_parabolic_peak(mean_spec, period_grid))
        surrogate_mean_spectra.append(mean_spec)

    return np.array(surrogate_peaks), surrogate_mean_spectra


def plot_surrogate_results(
    real_spectrum: np.ndarray,
    real_peak_period: float,
    surrogate_peaks: np.ndarray,
    surrogate_mean_spectra: list[np.ndarray],
    period_grid: np.ndarray,
    output_path: Path,
    dataset_name: str,
) -> float:
    """Two-panel figure: null peak distribution + real vs. null spectra.

    Test statistic: height of the mean dominant spectrum at the real peak
    argmax index.  p-value = fraction of surrogates whose height exceeds the
    real height at that index.

    Returns p-value.
    """
    n_surr = len(surrogate_mean_spectra)
    real_peak_idx = int(np.argmax(real_spectrum))
    real_height = float(real_spectrum[real_peak_idx])
    surr_heights = np.array([float(s[real_peak_idx]) for s in surrogate_mean_spectra])
    p_value = float(np.mean(surr_heights >= real_height))

    surr_stack = np.stack(surrogate_mean_spectra)
    surr_mean = surr_stack.mean(axis=0)
    surr_lo = np.percentile(surr_stack, 2.5, axis=0)
    surr_hi = np.percentile(surr_stack, 97.5, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(surrogate_peaks, bins=30, color="#aec6e8", edgecolor="none",
            density=True, label=f"Null  (n={n_surr})")
    ax.axvline(real_peak_period, color="#d7191c", linewidth=2,
               label=f"Real peak: {real_peak_period:.2f} d")
    ax.set_xlabel("Peak period (days)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Null distribution of peak periods\n"
        f"Real = {real_peak_period:.2f} d  |  p = {p_value:.3f}"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.fill_between(period_grid, surr_lo, surr_hi,
                     color="#aec6e8", alpha=0.5, label="Null 2.5–97.5 %")
    ax2.plot(period_grid, surr_mean, color="#74a9d8",
             linewidth=1.5, linestyle="--", label="Null mean")
    ax2.plot(period_grid, real_spectrum, color="#d7191c",
             linewidth=2, label="Real data")
    ax2.axvline(real_peak_period, color="#d7191c", linestyle=":", linewidth=1)
    ax2.set_xlabel("Period (days)")
    ax2.set_ylabel("Mean dominant spectrum")
    ax2.set_title(f"Real vs. null spectra — {dataset_name}")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Surrogate test  ({n_surr} iterations)  —  p = {p_value:.3f}\n{dataset_name}",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    print(f"  p-value (height at peak):  {p_value:.4f}")
    print(f"  Real height : {real_height:.4f}")
    print(f"  Null mean   : {surr_heights.mean():.4f}  |  max: {surr_heights.max():.4f}")

    return p_value


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    input_path: Path | None,
    top_vars: int,
    top_frac: float,
    n_periods: int,
    period_min: float,
    period_max: float,
    output_dir: Path,
    interim_dir: Path,
    n_surrogates: int = 0,
) -> int:
    """Full SVD spectrum pipeline.

    Parameters
    ----------
    input_path:
        Path to the eligible-users timeline CSV, or None for auto-detection.
    top_vars:
        Number of top variables by importance to use in the final analysis.
    top_frac:
        Fraction of top-cycling users to include in the mean spectrum (0 < top_frac <= 1).
    n_periods:
        Number of points in the period evaluation grid.
    period_min:
        Minimum period (days) for the evaluation grid.
    period_max:
        Maximum period (days) for the evaluation grid.
    output_dir:
        Directory for PNG outputs.
    interim_dir:
        Directory for CSV outputs (uses save_with_timestamp).

    Returns
    -------
    Exit code (0 = success, 1 = error).
    """
    # ------------------------------------------------------------------
    # Resolve input file
    # ------------------------------------------------------------------
    if input_path is None:
        input_path = find_latest_file(interim_dir, "eligible_users_timeline_*.csv")
        if input_path is None:
            print(f"ERROR: No eligible_users_timeline_*.csv found in {interim_dir}.")
            return 1
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    dataset_name = input_path.stem

    print("=" * 60)
    print("Step 34: SVD Spectrum Analysis")
    print("=" * 60)
    print(f"  Input : {input_path.name}")
    print(f"  Period range: {period_min}–{period_max} days ({n_periods} points)")
    print(f"  Top variables : {top_vars}")
    print(f"  Top-cycling fraction: {top_frac:.0%}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("[Step 34.1] Loading data...")
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    print(f"  Rows: {len(df):,}  |  Users: {df['author'].nunique():,}")

    metadata_cols = {"author", "offset_from_cd1"}
    feature_cols = [
        c for c in df.columns
        if c.endswith("_mean") and c not in metadata_cols
    ]
    if not feature_cols:
        print("ERROR: No _mean feature columns found in input file.")
        return 1
    print(f"  Feature columns: {len(feature_cols)}")

    # Period evaluation grid
    period_grid = np.linspace(period_min, period_max, n_periods)

    # ------------------------------------------------------------------
    # Step 1: Build 3-D spectrum cube
    # ------------------------------------------------------------------
    print("\n[Step 34.2] Building per-(user, variable) power spectra...")
    cube, users, _ = build_spectrum_cube(df, feature_cols, period_grid)
    n_users_total = len(users)
    print(f"  Spectrum cube shape: {cube.shape}  (vars x users x periods)")

    # Diagnostic: raw mean spectrum across users — no SVD involved
    plot_diagnostic_spectra(
        cube=cube,
        period_grid=period_grid,
        feature_cols=feature_cols,
        output_path=output_dir / "34_diagnostic_mean_spectrum.png",
        dataset_name=dataset_name,
    )

    # ------------------------------------------------------------------
    # Step 1b: Precompute log-T detrend projection matrix.
    # Detrending log(power) against log(T) removes a pure power-law trend
    # (P ∝ T^α) without any boundary-dependent concavity artifact.
    # The residual is boundary-invariant: changing period_min/max shifts
    # which points are evaluated but not where a genuine peak appears.
    # ------------------------------------------------------------------
    log_T = np.log(period_grid)                    # shape (n_periods,)
    X_logT = np.column_stack([log_T, np.ones_like(log_T)])  # design matrix
    # I_minus_H projects out the log-T linear fit from each spectrum row
    I_minus_H = np.eye(len(period_grid)) - X_logT @ np.linalg.solve(
        X_logT.T @ X_logT, X_logT.T
    )

    # ------------------------------------------------------------------
    # Step 2: Per-user SVD (all variables)
    # ------------------------------------------------------------------
    print("\n[Step 34.3] Running per-user SVD (all variables)...")
    svd_results_all = compute_svd_per_user(cube, users, feature_cols,
                                           I_minus_H=I_minus_H)
    print(f"  Users with valid SVD: {len(svd_results_all):,} / {n_users_total:,}")

    if len(svd_results_all) < 2:
        print("ERROR: Fewer than 2 users with valid SVD decompositions.")
        return 1

    # ------------------------------------------------------------------
    # Step 3: Global variable importance + top-k selection
    # ------------------------------------------------------------------
    print("\n[Step 34.4] Computing mean variable importance...")
    mean_importance = compute_mean_variable_importance(svd_results_all, len(feature_cols))

    # Sort descending (ignore NaNs)
    valid_importance = np.where(np.isnan(mean_importance), -np.inf, mean_importance)
    sorted_vi = np.argsort(valid_importance)[::-1]
    top_vi: list[int] = [int(i) for i in sorted_vi[:top_vars]]
    top_feature_names = [feature_cols[i] for i in top_vi]

    print(f"  Top-{top_vars} variables:")
    for rank, vi in enumerate(top_vi, 1):
        fname = feature_cols[vi].replace("_mean", "")
        print(f"    {rank:2d}. {fname}  (importance={mean_importance[vi]:.4f})")

    # Plot variable importance
    vi_plot_path = output_dir / "34_variable_importance.png"
    plot_variable_importance(
        mean_importance=mean_importance,
        feature_names=feature_cols,
        output_path=vi_plot_path,
        top_k=top_vars,
        dataset_name=dataset_name,
        n_users=len(svd_results_all),
    )

    # ------------------------------------------------------------------
    # Step 4: Re-run SVD restricted to top variables
    # ------------------------------------------------------------------
    print(f"\n[Step 34.5] Re-running per-user SVD with top-{top_vars} variables...")
    svd_results_top = compute_svd_per_user(cube, users, feature_cols, var_subset=top_vi,
                                           I_minus_H=I_minus_H)
    print(f"  Users with valid SVD (top-{top_vars} vars): {len(svd_results_top):,}")

    if len(svd_results_top) < 2:
        print("ERROR: Fewer than 2 users have valid spectra for the top variables.")
        return 1

    # Collect cycling scores
    user_scores = {r.user: r.cycling_score for r in svd_results_top}
    scores_array = np.array([r.cycling_score for r in svd_results_top])

    # Top-frac threshold (use median when top_frac == 0.5)
    threshold = float(np.quantile(scores_array, 1.0 - top_frac))
    top_users_results = [r for r in svd_results_top if r.cycling_score >= threshold]

    print(f"  Cycling-score threshold ({top_frac:.0%}): {threshold:.4f}")
    print(f"  Top-cycling users: {len(top_users_results):,}")

    # ------------------------------------------------------------------
    # Mean dominant spectrum — weighted by S[0] so strong-cycling users
    # contribute more than noise-dominated users.
    # ------------------------------------------------------------------
    dominant_spectra = np.stack([r.dominant_spectrum for r in top_users_results])  # (n_top, n_periods)
    s0_weights = np.array([r.s0 for r in top_users_results])
    s0_weights = s0_weights / (s0_weights.sum() + _EPSILON)
    mean_spectrum = (dominant_spectra * s0_weights[:, np.newaxis]).sum(axis=0)

    # Plot
    ds_plot_path = output_dir / "34_dominant_spectrum.png"
    peak_period = plot_dominant_spectrum(
        mean_spectrum=mean_spectrum,
        period_grid=period_grid,
        output_path=ds_plot_path,
        n_top_users=len(top_users_results),
        n_total_users=len(svd_results_top),
        dataset_name=dataset_name,
    )

    # ------------------------------------------------------------------
    # Step 5 (optional): Surrogate test
    # ------------------------------------------------------------------
    if n_surrogates > 0:
        print(f"\n[Step 34.5b] Running surrogate test ({n_surrogates} iterations)...")
        print("  Null hypothesis: no shared periodic signal across variables.")
        surrogate_peaks, surrogate_mean_spectra = run_surrogate_test(
            cube=cube,
            users=users,
            feature_cols=feature_cols,
            period_grid=period_grid,
            top_vars=top_vars,
            top_frac=top_frac,
            I_minus_H=I_minus_H,
            n_surrogates=n_surrogates,
        )
        if len(surrogate_mean_spectra) > 0:
            plot_surrogate_results(
                real_spectrum=mean_spectrum,
                real_peak_period=peak_period,
                surrogate_peaks=surrogate_peaks,
                surrogate_mean_spectra=surrogate_mean_spectra,
                period_grid=period_grid,
                output_path=output_dir / "34_surrogate_test.png",
                dataset_name=dataset_name,
            )
        else:
            print("  WARNING: No valid surrogate iterations completed.")

    # ------------------------------------------------------------------
    # Step 6: Save results CSV
    # ------------------------------------------------------------------
    print("\n[Step 34.6] Saving cycling scores CSV...")
    in_top50_set = {r.user for r in top_users_results}

    # Include all users that had a valid top-variable SVD
    scores_rows = [
        {
            "user": r.user,
            "cycling_score": float(r.cycling_score),
            "in_top50": r.user in in_top50_set,
        }
        for r in svd_results_top
    ]
    scores_df = pd.DataFrame(scores_rows)

    output_csv = save_with_timestamp(
        scores_df,
        interim_dir,
        "svd_cycling_scores",
    )
    print(f"  Saved: {output_csv.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total users (valid SVD on top-{top_vars} vars) : {len(svd_results_top):,}")
    print(f"  Top {top_frac:.0%} cycling users               : {len(top_users_results):,}")
    print(f"  Peak period (parabolic, dominant spectrum)    : {peak_period:.2f} days")
    if n_surrogates > 0:
        print(f"  Surrogate test                               : {n_surrogates} iterations")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 34: SVD-based spectrum analysis for menstrual-cycle NLP pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to eligible_users_timeline CSV. "
            "If omitted, the latest eligible_users_timeline_*.csv in data/interim/ is used."
        ),
    )
    parser.add_argument(
        "--top-vars",
        type=int,
        default=10,
        metavar="INT",
        help="Number of top variables by SVD importance to retain (default: 10).",
    )
    parser.add_argument(
        "--top-frac",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Fraction of users with highest cycling score to include in mean spectrum (default: 0.5).",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=100,
        metavar="INT",
        help="Number of evaluation points on the period grid (default: 100).",
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=20.0,
        metavar="FLOAT",
        help="Minimum period in days for the evaluation grid (default: 20).",
    )
    parser.add_argument(
        "--period-max",
        type=float,
        default=35.0,
        metavar="FLOAT",
        help="Maximum period in days for the evaluation grid (default: 35).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        metavar="PATH",
        help="Directory for PNG outputs (default: outputs/).",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=0,
        metavar="INT",
        help=(
            "Number of surrogate iterations for significance testing (default: 0 = skip). "
            "200 is fast; 500 gives better p-value resolution."
        ),
    )

    args = parser.parse_args()

    # Resolve directories relative to this script's project root
    project_root = Path(__file__).resolve().parent.parent
    interim_dir = project_root / "data" / "interim"
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else project_root / args.output_dir
    )
    input_path = args.input
    if input_path is not None and not input_path.is_absolute():
        input_path = project_root / input_path

    raise SystemExit(
        main(
            input_path=input_path,
            top_vars=args.top_vars,
            top_frac=args.top_frac,
            n_periods=args.n_periods,
            period_min=args.period_min,
            period_max=args.period_max,
            output_dir=output_dir,
            interim_dir=interim_dir,
            n_surrogates=args.n_surrogates,
        )
    )
