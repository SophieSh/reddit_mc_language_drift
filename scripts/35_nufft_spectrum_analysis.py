#!/usr/bin/env python3
"""Step 35: NUFFT-based spectrum analysis.

Replaces the interpolate-then-FFT approach of script 34 with a direct
Non-Uniform DFT on the original irregular time samples — equivalent to
MATLAB's nufft(f, t, k).

Key difference from script 34:
  - No grid interpolation; DFT evaluated directly at observed day offsets.
  - Time-series values are NOT z-scored before DFT (natural amplitudes
    preserved), so high-variance variables (e.g. syntactic complexity)
    contribute more — matching the supervisor's non-uniform importances.
  - Row z-scoring of spectra is off by default for the same reason.
  - Log-T detrend is kept (controls red-noise slope); disable with
    --no-detrend to see the raw spectral shape.

Output files (in outputs/):
  35_variable_importance.png
  35_dominant_spectrum.png
  35_diagnostic_mean_spectrum.png

Output CSV (in data/interim/):
  nufft_cycling_scores_{timestamp}.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import find_latest_file, save_with_timestamp

_MIN_OBS = 10
_EPSILON = 1e-12

# One representative per semantic cluster (domain-expert curated)
REPRESENTATIVE_FEATURES = [
    "Syntactic_phrase_distribution_SBAR_mean",
    "syntactic_complexity_subordination_index_mean",
    "pos_distribution_PRON_mean",
    "avg_concreteness_mean",
    "valence_dict_average_mean",
    "hedging_modal_verbs_mean",
    "cohesion_analysis_lexical_overlap_mean",
    "idea_density_cpidr_density_mean",
    "unique_word_fraction_mean",
    "spelling_errors_frac_mean",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class UserSVDResult(NamedTuple):
    user: str
    cycling_score: float
    s0: float
    dominant_spectrum: np.ndarray
    var_importances: np.ndarray
    included_var_indices: list[int]


# ---------------------------------------------------------------------------
# Step 1: per-(user, variable) power spectrum via direct NUFFT
# ---------------------------------------------------------------------------

def _power_spectrum_nufft(
    offsets: np.ndarray,
    values: np.ndarray,
    period_grid: np.ndarray,
    ts_zscore: bool = False,
) -> np.ndarray | None:
    """Direct Non-Uniform DFT on original irregular time samples.

    Mathematically equivalent to finufft.nufft1d3() and MATLAB nufft(f,t,k),
    but faster for our problem size (~50 obs/user, 100 target freqs) because
    it avoids per-call library overhead.

    Computes  F(k) = sum_j v_j · exp(-2πi · freq_k · t_j)
    at each target frequency freq_k = 1/period_grid[k].
    No grid interpolation.

    ts_zscore: if True, standardise the time series to mean=0 std=1 before DFT
               so all variables contribute equal power regardless of raw scale.
    """
    if len(offsets) < _MIN_OBS:
        return None
    span = float(offsets.max() - offsets.min())
    if span < float(period_grid.max()) * 1.5:
        return None

    t = offsets.astype(float)
    v = values.astype(float)
    v = v - v.mean()
    if ts_zscore:
        std = v.std()
        if std > _EPSILON:
            v = v / std

    target_freqs = 1.0 / period_grid
    phases = 2.0 * np.pi * np.outer(target_freqs, t)
    dft = v @ np.exp(-1j * phases).T
    return np.abs(dft) ** 2 / len(v)


# ---------------------------------------------------------------------------
# Step 2: per-user SVD
# ---------------------------------------------------------------------------

def _parabolic_peak(spectrum: np.ndarray, period_grid: np.ndarray) -> float:
    """Parabolic interpolation around argmax — reduces grid-point bias."""
    idx = int(np.argmax(spectrum))
    if idx == 0 or idx == len(spectrum) - 1:
        return float(period_grid[idx])
    d = float(period_grid[1] - period_grid[0])
    y0, y1, y2 = float(spectrum[idx - 1]), float(spectrum[idx]), float(spectrum[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-15:
        return float(period_grid[idx])
    return float(period_grid[idx]) + 0.5 * d * (y0 - y2) / denom


def _user_svd(
    spectrum_matrix: np.ndarray,
    included_var_indices: list[int],
    I_minus_H: np.ndarray | None,
    row_zscore: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """SVD on per-variable log-power spectra.

    Parameters
    ----------
    I_minus_H : log-T detrend projection matrix, or None to skip detrending.
    row_zscore : if True, equalise row amplitudes before SVD (suppresses
                 high-amplitude variables but stabilises peak location).
                 If False, high-variance variables dominate naturally.
    """
    log_matrix = np.log1p(spectrum_matrix)

    detrended = log_matrix @ I_minus_H if I_minus_H is not None else log_matrix

    if row_zscore:
        row_stds = detrended.std(axis=1, keepdims=True) + _EPSILON
        svd_input = detrended / row_stds
    else:
        svd_input = detrended

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, S, Vt = np.linalg.svd(svd_input, full_matrices=False)

    cycling_score = float(S[0] ** 2 / (np.sum(S ** 2) + _EPSILON))
    s0 = float(S[0])

    dominant_spectrum = Vt[0, :]
    if U[:, 0].sum() < 0:
        dominant_spectrum = -dominant_spectrum

    # Importances from raw (un-z-scored) projections — preserves amplitude info
    raw_proj = detrended @ dominant_spectrum
    var_importances = np.abs(raw_proj) / (np.abs(raw_proj).sum() + _EPSILON)

    return cycling_score, s0, dominant_spectrum, var_importances


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_spectrum_cube(
    df: pd.DataFrame,
    feature_cols: list[str],
    period_grid: np.ndarray,
    ts_zscore: bool = False,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_vars × n_users × n_periods) cube via NUFFT."""
    users = sorted(df["author"].unique())
    n_vars, n_users, n_periods = len(feature_cols), len(users), len(period_grid)
    cube = np.full((n_vars, n_users, n_periods), np.nan, dtype=np.float32)
    user_index = {u: i for i, u in enumerate(users)}

    for vi, feat in enumerate(tqdm(feature_cols, desc="NUFFT cube", unit="var",
                                   disable=not show_progress)):
        sub = df[["author", "offset_from_cd1", feat]].dropna(subset=[feat])
        for user, grp in sub.groupby("author", sort=False):
            if len(grp) < _MIN_OBS:
                continue
            spectrum = _power_spectrum_nufft(
                grp["offset_from_cd1"].values,
                grp[feat].values,
                period_grid,
                ts_zscore=ts_zscore,
            )
            if spectrum is not None:
                cube[vi, user_index[user], :] = spectrum.astype(np.float32)

    return cube, users


def compute_svd_per_user(
    cube: np.ndarray,
    users: list[str],
    feature_cols: list[str],
    var_subset: list[int] | None = None,
    I_minus_H: np.ndarray | None = None,
    row_zscore: bool = False,
    show_progress: bool = True,
) -> list[UserSVDResult]:
    if var_subset is None:
        var_subset = list(range(cube.shape[0]))

    results: list[UserSVDResult] = []
    for ui, user in enumerate(tqdm(users, desc="Per-user SVD", unit="user",
                                   disable=not show_progress)):
        valid_vi = [vi for vi in var_subset if not np.any(np.isnan(cube[vi, ui, :]))]
        if len(valid_vi) < 2:
            continue
        M = cube[valid_vi, ui, :].astype(np.float64)
        cycling_score, s0, dom_spec, var_imp = _user_svd(
            M, valid_vi, I_minus_H, row_zscore,
        )
        results.append(UserSVDResult(
            user=user,
            cycling_score=cycling_score,
            s0=s0,
            dominant_spectrum=dom_spec,
            var_importances=var_imp,
            included_var_indices=valid_vi,
        ))
    return results


def compute_mean_variable_importance(
    svd_results: list[UserSVDResult],
    n_vars: int,
) -> np.ndarray:
    accumulator = np.zeros(n_vars)
    counts = np.zeros(n_vars, dtype=np.int64)
    for res in svd_results:
        for local_i, global_vi in enumerate(res.included_var_indices):
            accumulator[global_vi] += res.var_importances[local_i]
            counts[global_vi] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, accumulator / counts, np.nan)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_variable_importance(
    mean_importance: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int,
    dataset_name: str,
    n_users: int,
) -> None:
    imp = mean_importance.copy()
    imp[np.isnan(imp)] = 0.0
    sort_idx = np.argsort(imp)[::-1]
    n_shown = min(len(feature_names), 40)
    colors = ["#2c7bb6" if i < top_k else "#d7e8f5" for i in range(n_shown)]

    fig, ax = plt.subplots(figsize=(max(14, n_shown * 0.35), 6))
    ax.bar(range(n_shown), imp[sort_idx[:n_shown]], color=colors, edgecolor="none")
    ax.set_xticks(range(n_shown))
    ax.set_xticklabels(
        [feature_names[i].replace("_mean", "") for i in sort_idx[:n_shown]],
        rotation=60, ha="right", fontsize=7,
    )
    ax.set_ylabel("Mean variable importance")
    ax.set_title(f"NUFFT Variable Importance — {dataset_name}\nn={n_users} | top-{top_k} in blue")
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
    peak_period = _parabolic_peak(mean_spectrum, period_grid)
    peak_idx = int(np.argmax(mean_spectrum))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(period_grid, mean_spectrum, color="#2c7bb6", linewidth=2)
    ax.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.5,
               label=f"Peak: {peak_period:.2f} d (parabolic)")
    ax.annotate(
        f"{peak_period:.2f} d",
        xy=(peak_period, float(mean_spectrum[peak_idx])),
        xytext=(peak_period + 0.4, float(mean_spectrum[peak_idx]) * 0.95),
        fontsize=10, color="#d7191c",
        arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.2),
    )
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Mean dominant spectrum (Vt[0,:])")
    ax.set_title(
        f"NUFFT Dominant Spectral Component — {dataset_name}\n"
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


def plot_diagnostic(
    cube: np.ndarray,
    period_grid: np.ndarray,
    feature_cols: list[str],
    output_path: Path,
    dataset_name: str,
) -> None:
    with np.errstate(invalid="ignore"):
        per_var_mean = np.nanmean(cube.astype(np.float64), axis=1)
    grand_mean = np.nanmean(per_var_mean, axis=0)
    peak_period = _parabolic_peak(grand_mean, period_grid)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    for vi in range(per_var_mean.shape[0]):
        ax.plot(period_grid, per_var_mean[vi], color="#aec6e8", linewidth=0.6, alpha=0.6)
    ax.plot(period_grid, grand_mean, color="#d7191c", linewidth=2.5, label="Grand mean")
    ax.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Mean NUFFT power (log1p)")
    ax.set_title(f"Diagnostic: raw NUFFT mean spectra — {dataset_name}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.plot(period_grid, grand_mean, color="#2c7bb6", linewidth=2)
    ax2.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.5,
                label=f"Peak: {peak_period:.2f} d")
    ax2.set_xlabel("Period (days)")
    ax2.set_ylabel("Mean NUFFT power (log1p)")
    ax2.set_title("Grand mean spectrum (no SVD)")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved diagnostic: {output_path}")
    print(f"  Grand-mean NUFFT peak (no SVD): {peak_period:.2f} days")


# ---------------------------------------------------------------------------
# Surrogate test (day-shuffle null)
# ---------------------------------------------------------------------------

def run_surrogate_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    top_vi: list[int],
    period_grid: np.ndarray,
    ts_zscore: bool,
    I_minus_H: np.ndarray | None,
    row_zscore: bool,
    top_frac: float,
    real_mean_spectrum: np.ndarray,
    n_surrogates: int,
    rng: np.random.Generator,
) -> dict:
    """Day-shuffle surrogate test (correct null hypothesis).

    For each iteration: independently shuffle observed VALUES within each
    (user, variable) pair, keeping timestamps fixed. This destroys any
    temporal periodicity while preserving observation counts and timing.
    Only the top-k variables are reshuffled and recomputed for speed.
    """
    real_peak_idx = int(np.argmax(real_mean_spectrum))
    real_peak_height = float(real_mean_spectrum[real_peak_idx])

    top_feat = [feature_cols[vi] for vi in top_vi]
    null_heights: list[float] = []
    null_peak_periods: list[float] = []

    for _it in tqdm(range(n_surrogates), desc="Day-shuffle surrogates"):
        df_s = df.copy()
        for feat in top_feat:
            for user, grp in df_s.groupby("author", sort=False):
                idx = grp.index
                vals = df_s.loc[idx, feat].values.copy()
                rng.shuffle(vals)
                df_s.loc[idx, feat] = vals

        cube_s, users_s = build_spectrum_cube(
            df_s, top_feat, period_grid, ts_zscore=ts_zscore, show_progress=False
        )
        local_vi = list(range(len(top_vi)))
        svd_s = compute_svd_per_user(
            cube_s, users_s, top_feat,
            var_subset=local_vi, I_minus_H=I_minus_H, row_zscore=row_zscore,
            show_progress=False,
        )
        if len(svd_s) < 2:
            continue

        scores_s = np.array([r.cycling_score for r in svd_s])
        thresh_s = float(np.quantile(scores_s, 1.0 - top_frac))
        top_s = [r for r in svd_s if r.cycling_score >= thresh_s]
        if not top_s:
            continue

        specs_s = np.stack([r.dominant_spectrum for r in top_s])
        w_s = np.array([r.s0 for r in top_s])
        w_s /= w_s.sum() + _EPSILON
        mean_s = (specs_s * w_s[:, np.newaxis]).sum(axis=0)

        null_heights.append(float(mean_s[real_peak_idx]))
        null_peak_periods.append(_parabolic_peak(mean_s, period_grid))

    arr = np.array(null_heights)
    p_value = float(np.mean(arr >= real_peak_height)) if len(arr) else float("nan")
    return {
        "p_value": p_value,
        "real_peak_height": real_peak_height,
        "real_peak_idx": real_peak_idx,
        "null_heights": arr,
        "null_peak_periods": null_peak_periods,
    }


def plot_surrogate_results(
    null_heights: np.ndarray,
    real_peak_height: float,
    p_value: float,
    null_peak_periods: list[float],
    real_peak_period: float,
    output_path: Path,
    dataset_name: str,
    n_surrogates: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(null_heights, bins=max(5, n_surrogates // 3), color="#aec6e8", edgecolor="white")
    ax.axvline(real_peak_height, color="#d7191c", linewidth=2,
               label=f"Real: {real_peak_height:.4f}")
    ax.set_xlabel("Spectrum height at real peak index")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Day-shuffle surrogate null (n={n_surrogates})\n"
        f"p = {p_value:.3f}  |  {dataset_name}"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.hist(null_peak_periods, bins=max(5, n_surrogates // 3),
             color="#aec6e8", edgecolor="white")
    ax2.axvline(real_peak_period, color="#d7191c", linewidth=2,
                label=f"Real: {real_peak_period:.2f} d")
    ax2.set_xlabel("Peak period (days)")
    ax2.set_ylabel("Count")
    ax2.set_title("Null peak period distribution")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
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
    ts_zscore: bool,
    detrend: bool,
    row_zscore: bool,
    tag: str = "",
    n_surrogates: int = 0,
    weighted_mean: bool = False,
    raw_mean: bool = False,
    use_representatives: bool = False,
) -> int:
    # ------------------------------------------------------------------
    # Resolve input
    # ------------------------------------------------------------------
    if input_path is None:
        input_path = find_latest_file(interim_dir, "eligible_users_timeline_*.csv")
        if input_path is None:
            print(f"ERROR: No eligible_users_timeline_*.csv in {interim_dir}")
            return 1
    if not input_path.exists():
        print(f"ERROR: {input_path}")
        return 1

    dataset_name = input_path.stem
    prefix = f"35_{tag}_" if tag else "35_"

    print("=" * 60)
    print(f"Step 35: NUFFT Spectrum Analysis  [{tag or 'default'}]")
    print("=" * 60)
    print(f"  Input         : {input_path.name}")
    print(f"  Period range  : {period_min}–{period_max} days ({n_periods} points)")
    print(f"  Top vars      : {top_vars}  |  Top frac: {top_frac:.0%}")
    print(f"  TS z-score    : {ts_zscore}")
    print(f"  Detrend       : {detrend}")
    print(f"  Row z-score   : {row_zscore}")
    print()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("[35.1] Loading data...")
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    print(f"  Rows: {len(df):,}  |  Users: {df['author'].nunique():,}")

    feature_cols = [
        c for c in df.columns
        if c.endswith("_mean") and c not in {"author", "offset_from_cd1"}
    ]
    if not feature_cols:
        print("ERROR: no _mean feature columns found")
        return 1
    print(f"  Features: {len(feature_cols)}")

    period_grid = np.linspace(period_min, period_max, n_periods)

    # ------------------------------------------------------------------
    # Build spectrum cube (NUFFT)
    # ------------------------------------------------------------------
    print("\n[35.2] Building NUFFT spectrum cube...")
    cube, users = build_spectrum_cube(df, feature_cols, period_grid, ts_zscore=ts_zscore)
    print(f"  Cube shape: {cube.shape}  (vars × users × periods)")

    plot_diagnostic(cube, period_grid, feature_cols,
                    output_dir / f"{prefix}diagnostic_mean_spectrum.png", dataset_name)

    # ------------------------------------------------------------------
    # Detrend projection matrix (optional)
    # ------------------------------------------------------------------
    I_minus_H: np.ndarray | None = None
    if detrend:
        log_T = np.log(period_grid)
        X = np.column_stack([log_T, np.ones_like(log_T)])
        I_minus_H = np.eye(len(period_grid)) - X @ np.linalg.solve(X.T @ X, X.T)
        print("  Log-T detrend: ON")
    else:
        print("  Log-T detrend: OFF")

    if use_representatives:
        # ------------------------------------------------------------------
        # Fixed cluster representatives — skip SVD-based importance selection
        # ------------------------------------------------------------------
        print("\n[35.3] Using domain-expert cluster representatives (skipping importance step)...")
        missing = [r for r in REPRESENTATIVE_FEATURES if r not in feature_cols]
        if missing:
            print(f"ERROR: missing representative features: {missing}")
            return 1
        top_vi = [feature_cols.index(r) for r in REPRESENTATIVE_FEATURES]
        print(f"  Fixed {len(top_vi)} representatives:")
        for rank, vi in enumerate(top_vi, 1):
            print(f"    {rank:2d}. {feature_cols[vi].replace('_mean', '')}")
    else:
        # ------------------------------------------------------------------
        # SVD — all variables
        # ------------------------------------------------------------------
        print("\n[35.3] Per-user SVD (all variables)...")
        svd_all = compute_svd_per_user(cube, users, feature_cols,
                                        I_minus_H=I_minus_H, row_zscore=row_zscore)
        print(f"  Valid SVD: {len(svd_all):,} / {len(users):,}")

        if len(svd_all) < 2:
            print("ERROR: fewer than 2 users with valid SVD")
            return 1

        # ------------------------------------------------------------------
        # Variable importance + top-k
        # ------------------------------------------------------------------
        print("\n[35.4] Variable importance...")
        mean_imp = compute_mean_variable_importance(svd_all, len(feature_cols))
        valid_imp = np.where(np.isnan(mean_imp), -np.inf, mean_imp)
        top_vi = [int(i) for i in np.argsort(valid_imp)[::-1][:top_vars]]

        print(f"  Top-{top_vars} variables:")
        for rank, vi in enumerate(top_vi, 1):
            print(f"    {rank:2d}. {feature_cols[vi].replace('_mean','')}  "
                  f"(importance={mean_imp[vi]:.4f})")

        plot_variable_importance(mean_imp, feature_cols,
                                  output_dir / f"{prefix}variable_importance.png",
                                  top_k=top_vars, dataset_name=dataset_name,
                                  n_users=len(svd_all))

    # ------------------------------------------------------------------
    # SVD — top variables
    # ------------------------------------------------------------------
    print(f"\n[35.5] Re-running SVD on top-{top_vars} variables...")
    svd_top = compute_svd_per_user(cube, users, feature_cols,
                                    var_subset=top_vi,
                                    I_minus_H=I_minus_H, row_zscore=row_zscore)
    print(f"  Valid SVD (top vars): {len(svd_top):,}")

    if len(svd_top) < 2:
        print("ERROR: fewer than 2 users with valid top-variable SVD")
        return 1

    scores = np.array([r.cycling_score for r in svd_top])
    threshold = float(np.quantile(scores, 1.0 - top_frac))
    top_users = [r for r in svd_top if r.cycling_score >= threshold]
    print(f"  Threshold ({top_frac:.0%}): {threshold:.4f}  |  Top users: {len(top_users):,}")

    # Mean spectrum: detrended log-power mean across top vars+users, or SVD Vt[0,:] mean
    if raw_mean:
        # Supervisor's approach: for each top user, apply log1p+detrend to their
        # (top_vars × 100) slice, mean across vars → (100,), then mean across users.
        user_name_to_idx = {u: i for i, u in enumerate(users)}
        top_user_indices = [user_name_to_idx[r.user] for r in top_users]
        per_user_spectra = []
        for ui in top_user_indices:
            M = cube[top_vi, ui, :].astype(np.float64)   # (10 × 100)
            if np.any(np.isnan(M)):
                continue
            log_M = np.log1p(M)                          # no detrend — matches supervisor's positive y-axis
            per_user_spectra.append(log_M.mean(axis=0))  # mean across vars → (100,)
        mean_spectrum = np.mean(per_user_spectra, axis=0)  # mean across users → (100,)
    else:
        specs = np.stack([r.dominant_spectrum for r in top_users])
        if weighted_mean:
            w = np.array([r.s0 for r in top_users])
            w = w / (w.sum() + _EPSILON)
            mean_spectrum = (specs * w[:, np.newaxis]).sum(axis=0)
        else:
            mean_spectrum = specs.mean(axis=0)

    peak_period = plot_dominant_spectrum(
        mean_spectrum, period_grid,
        output_dir / f"{prefix}dominant_spectrum.png",
        n_top_users=len(top_users),
        n_total_users=len(svd_top),
        dataset_name=dataset_name,
    )

    # ------------------------------------------------------------------
    # Surrogate test
    # ------------------------------------------------------------------
    if n_surrogates > 0:
        print(f"\n[35.6] Day-shuffle surrogate test ({n_surrogates} iterations)...")
        rng = np.random.default_rng(42)
        surr = run_surrogate_test(
            df=df,
            feature_cols=feature_cols,
            top_vi=top_vi,
            period_grid=period_grid,
            ts_zscore=ts_zscore,
            I_minus_H=I_minus_H,
            row_zscore=row_zscore,
            top_frac=top_frac,
            real_mean_spectrum=mean_spectrum,
            n_surrogates=n_surrogates,
            rng=rng,
        )
        print(f"  p-value          : {surr['p_value']:.3f}")
        print(f"  Real height      : {surr['real_peak_height']:.4f}")
        if len(surr["null_heights"]):
            print(f"  Null mean height : {surr['null_heights'].mean():.4f}")
            print(f"  Null max height  : {surr['null_heights'].max():.4f}")
        plot_surrogate_results(
            null_heights=surr["null_heights"],
            real_peak_height=surr["real_peak_height"],
            p_value=surr["p_value"],
            null_peak_periods=surr["null_peak_periods"],
            real_peak_period=peak_period,
            output_path=output_dir / f"{prefix}surrogate_test.png",
            dataset_name=dataset_name,
            n_surrogates=n_surrogates,
        )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    print("\n[35.7] Saving cycling scores...")
    in_top_set = {r.user for r in top_users}
    scores_df = pd.DataFrame([
        {"user": r.user, "cycling_score": float(r.cycling_score),
         "in_top50": r.user in in_top_set}
        for r in svd_top
    ])
    out_csv = save_with_timestamp(scores_df, interim_dir, f"nufft_cycling_scores_{tag}" if tag else "nufft_cycling_scores")
    print(f"  Saved: {out_csv.name}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Users (valid top-var SVD) : {len(svd_top):,}")
    print(f"  Top {top_frac:.0%} cycling users  : {len(top_users):,}")
    print(f"  Peak period (parabolic)   : {peak_period:.2f} days")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 35: NUFFT-based spectrum analysis"
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--top-vars", type=int, default=10)
    parser.add_argument("--top-frac", type=float, default=0.5)
    parser.add_argument("--n-periods", type=int, default=100)
    parser.add_argument("--period-min", type=float, default=20.0)
    parser.add_argument("--period-max", type=float, default=35.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ts-zscore", action="store_true",
                        help="Z-score each time series before DFT (equalises raw feature amplitudes)")
    parser.add_argument("--no-detrend", action="store_true",
                        help="Skip log-T detrending")
    parser.add_argument("--row-zscore", action="store_true",
                        help="Row z-score spectra before SVD (equalises spectral shapes)")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag appended to output filenames to distinguish runs")
    parser.add_argument("--n-surrogates", type=int, default=0,
                        help="Day-shuffle surrogate iterations (0 = skip)")
    parser.add_argument("--weighted-mean", action="store_true",
                        help="S[0]-weight the mean spectrum (default: unweighted, matching supervisor)")
    parser.add_argument("--raw-mean", action="store_true",
                        help="Average raw power across top vars then top users (supervisor's likely approach)")
    parser.add_argument("--use-representatives", action="store_true",
                        help="Use 10 domain-expert cluster representatives instead of SVD importance selection")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    interim_dir = project_root / "data" / "interim"
    output_dir = (args.output_dir if args.output_dir.is_absolute()
                  else project_root / args.output_dir)
    input_path = args.input
    if input_path is not None and not input_path.is_absolute():
        input_path = project_root / input_path

    raise SystemExit(main(
        input_path=input_path,
        top_vars=args.top_vars,
        top_frac=args.top_frac,
        n_periods=args.n_periods,
        period_min=args.period_min,
        period_max=args.period_max,
        output_dir=output_dir,
        interim_dir=interim_dir,
        ts_zscore=args.ts_zscore,
        detrend=not args.no_detrend,
        row_zscore=args.row_zscore,
        tag=args.tag,
        n_surrogates=args.n_surrogates,
        weighted_mean=args.weighted_mean,
        raw_mean=args.raw_mean,
        use_representatives=args.use_representatives,
    ))
