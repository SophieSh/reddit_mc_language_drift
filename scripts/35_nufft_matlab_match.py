#!/usr/bin/env python3
"""Step 35m: NUFFT spectrum analysis — exact Python match of runner_compute_frequencies.m.

Differences from 35_nufft_spectrum_analysis.py that faithfully match MATLAB:
  - Global filter |offset_from_cd1| <= 90 before NUFFT
  - Per-series preprocessing: mean-subtract then linear detrend (scipy.signal.detrend)
  - Stores magnitude/N (not power/N); no log1p, no spectral detrend
  - SVD always row-centers then row-z-scores (ddof=1, MATLAB std convention)
  - unified_spectrum = abs(Vt[0,:])   (absolute value, no sign flip)
  - variable_importance = abs(U[:,0]) normalised per user by its max, mean across users
  - Selects top-10 vars, re-runs SVD, plots mean of top-50% confidence users
  - Peak = argmax grid point (no parabolic interpolation)

Output files (in outputs/):
  35m_variable_importance.png
  35m_dominant_spectrum.png

Output CSV (in data/interim/):
  nufft_matlab_cycling_scores_{timestamp}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import finufft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend as linear_detrend
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import find_latest_file, save_with_timestamp

_MIN_OBS = 4      # minimum valid observations per (user, variable) after filtering
_EPSILON = 1e-12

# MATLAB defaults
_MAX_OFFSET = 90.0
_PERIOD_MIN = 20.0
_PERIOD_MAX = 40.0
_NSELECT = 10
_TOP_FRAC = 0.50


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def make_grids(period_min: float, period_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate MATLAB grids.

    num_freq_points = 5 * (period_max - period_min)  →  100 for 20–40
    f_grid          = linspace(1/period_max, 1/period_min, num_freq_points)
    w_grid          = 2*pi * f_grid
    periods_grid    = 1 / f_grid
    """
    n = int(5 * (period_max - period_min))
    f_grid = np.linspace(1.0 / period_max, 1.0 / period_min, n)
    w_grid = 2.0 * np.pi * f_grid
    periods_grid = 1.0 / f_grid
    return f_grid, w_grid, periods_grid



# ---------------------------------------------------------------------------
# NUFFT
# ---------------------------------------------------------------------------

def _nufft_mag(t: np.ndarray, x: np.ndarray, w_grid: np.ndarray) -> np.ndarray:
    """Type-3 NUFFT via finufft.nufft1d3, matching MATLAB nufft(x_detrended, t, w_grid).

    MATLAB nufft(x, t, f) formula: X(k) = sum_n x(n) * exp(-2*pi*j * f(k) * t(n))
    MATLAB passes w_grid = 2*pi*f_grid as the 'f' argument, so MATLAB computes:
        X(k) = sum_n x(n) * exp(-2*pi*j * w_grid(k) * t(n))

    finufft.nufft1d3(x_pts, c, s_pts, isign=-1) computes:
        f[k] = sum_j c[j] * exp(-i * s[k] * x[j])

    To match MATLAB we need: -i * s[k] * x[j] = -2*pi*i * w_grid[k] * t[j]
    => s[k] * x[j] = 2*pi * w_grid[k] * t[j]

    We keep the finufft source-point constraint |x| < pi by scaling:
        x_pts = t * pi/T_max  (maps t into [-pi, pi))
    Then we must set:
        s_scaled = 2*pi * w_grid * T_max/pi
    so that s_scaled[k] * x_pts[j] = 2*pi * w_grid[k] * t[j]  ✓

    Returns abs(X) / len(x)  (magnitude normalised by N, matching MATLAB).
    """
    T_max = float(np.max(np.abs(t)))
    if T_max == 0:
        return np.zeros(len(w_grid))
    t_scaled = t * (np.pi / T_max)
    s_scaled = 2.0 * np.pi * w_grid * (T_max / np.pi)
    X = finufft.nufft1d3(
        t_scaled.astype(np.float64),
        x.astype(np.complex128),
        s_scaled.astype(np.float64),
        isign=-1,
        eps=1e-9,
    )
    return np.abs(X) / len(x)


def build_cube(
    df: pd.DataFrame,
    feature_cols: list[str],
    w_grid: np.ndarray,
    max_offset: float,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_vars × n_users × n_freqs) cube matching MATLAB all_spectra.

    Initialised to zeros (matching MATLAB zeros(...)). Entries for (user, variable)
    pairs with < _MIN_OBS valid observations remain zero.
    Only users with at least one observation within |offset_from_cd1| <= max_offset
    are included (matching MATLAB's numUsers definition).
    """
    df_in = df[np.abs(df["offset_from_cd1"]) <= max_offset]
    users = sorted(df_in["author"].unique())
    n_vars, n_users, n_freqs = len(feature_cols), len(users), len(w_grid)

    cube = np.zeros((n_vars, n_users, n_freqs), dtype=np.float32)
    user_idx = {u: i for i, u in enumerate(users)}

    for vi, feat in enumerate(tqdm(feature_cols, desc="NUFFT cube", unit="var",
                                   disable=not show_progress)):
        sub = df_in[["author", "offset_from_cd1", feat]].dropna(subset=[feat])
        for user, grp in sub.groupby("author", sort=False):
            if len(grp) < _MIN_OBS:
                continue
            t = grp["offset_from_cd1"].values.astype(float)
            x = grp[feat].values.astype(float)
            # MATLAB: x_detrended = detrend(x - mean(x))
            x_det = linear_detrend(x - x.mean())
            cube[vi, user_idx[user], :] = _nufft_mag(t, x_det, w_grid).astype(np.float32)

    return cube, users


# ---------------------------------------------------------------------------
# SVD pipeline
# ---------------------------------------------------------------------------

def svd_pipeline(
    cube: np.ndarray,
    var_subset: list[int],
    users: list[str],
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-user SVD exactly matching MATLAB.

    Operates on the (vars × freqs) spectrum slice for each user:
      1. Row-center (subtract row mean)
      2. Row z-score (ddof=1, matching MATLAB std(x, 0, 2))
      3. SVD → unified_spectrum = abs(Vt[0,:]), cycling_confidence = S[0]^2/sum(S^2)
              → raw_importance = abs(U[:,0])  (before per-user max-normalisation)

    Users with fewer than 2 non-zero variable rows are skipped (NaN in output).
    """
    n_users = len(users)
    n_freqs = cube.shape[2]
    n_sub = len(var_subset)

    unified = np.full((n_users, n_freqs), np.nan)
    conf = np.full(n_users, np.nan)
    raw_imp = np.full((n_users, n_sub), np.nan)

    for ui in tqdm(range(n_users), desc="Per-user SVD", unit="user",
                   disable=not show_progress):
        M = cube[var_subset, ui, :].astype(np.float64)   # (n_sub × n_freqs)

        # Skip if fewer than 2 rows have any signal (all-zero = no data)
        if (M != 0).any(axis=1).sum() < 2:
            continue

        # Row-center
        M = M - M.mean(axis=1, keepdims=True)

        # Row z-score (MATLAB std(x, 0, 2) uses ddof=1)
        row_stds = M.std(axis=1, ddof=1)
        row_stds[row_stds == 0] = 1.0
        M = M / row_stds[:, np.newaxis]

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        unified[ui, :] = np.abs(Vt[0, :])
        conf[ui] = S[0] ** 2 / (np.sum(S ** 2) + _EPSILON)
        raw_imp[ui, :] = np.abs(U[:, 0])

    return unified, conf, raw_imp


# ---------------------------------------------------------------------------
# Variable importance
# ---------------------------------------------------------------------------

def compute_mean_importance(raw_imp: np.ndarray) -> np.ndarray:
    """Replicate MATLAB mean_actual_weights.

    mean(variable_importance ./ max(variable_importance, [], 2), axis=0)
    i.e. for each user normalise importance by that user's max, then average across users.
    """
    with np.errstate(invalid="ignore"):
        row_max = np.nanmax(raw_imp, axis=1, keepdims=True)
    row_max = np.where(np.isnan(row_max) | (row_max < _EPSILON), 1.0, row_max)
    normed = raw_imp / row_max
    normed[np.isnan(raw_imp)] = np.nan
    return np.nanmean(normed, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_importance(
    mean_imp: np.ndarray,
    feature_names: list[str],
    top_vi: list[int],
    output_path: Path,
    dataset_name: str,
    n_users: int,
) -> None:
    sort_idx = np.argsort(mean_imp)        # ascending, matches MATLAB [m,I]=sort(m)
    n_shown = min(len(feature_names), 40)
    sort_shown = sort_idx[-n_shown:]
    top_set = set(top_vi)

    fig, ax = plt.subplots(figsize=(max(14, n_shown * 0.35), 6))
    vals = mean_imp[sort_shown]
    colors = ["#2c7bb6" if int(sort_shown[i]) in top_set else "#d7e8f5"
              for i in range(len(sort_shown))]
    ax.bar(range(len(sort_shown)), vals, color=colors, edgecolor="none")
    ax.set_xticks(range(len(sort_shown)))
    ax.set_xticklabels(
        [feature_names[int(j)].replace("_mean", "") for j in sort_shown],
        rotation=60, ha="right", fontsize=7,
    )
    ax.set_ylabel("Mean variable importance (MATLAB match)")
    ax.set_title(f"Variable Contribution — {dataset_name}  n={n_users}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spectrum(
    periods_grid: np.ndarray,
    mean_spec: np.ndarray,
    peak_period: float,
    n_top: int,
    n_total: int,
    dataset_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(periods_grid, mean_spec, color="#2c7bb6", linewidth=2)
    ax.axvline(peak_period, color="#d7191c", linestyle="--", linewidth=1.5,
               label=f"Peak: {peak_period:.2f} d (argmax)")
    ax.set_xlabel("Number of days per cycle")
    ax.set_ylabel("Spectral power")
    ax.set_title(
        f"NUFFT dominant spectrum (MATLAB match) — {dataset_name}\n"
        f"top-cycling users: {n_top}/{n_total}"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
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
    max_offset: float,
    period_min: float,
    period_max: float,
    n_select: int,
    top_frac: float,
    output_dir: Path,
    interim_dir: Path,
    tag: str = "",
) -> int:
    # Resolve input
    if input_path is None:
        input_path = find_latest_file(interim_dir, "eligible_users_timeline_*.csv")
        if input_path is None:
            print(f"ERROR: No eligible_users_timeline_*.csv in {interim_dir}")
            return 1
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        return 1

    dataset_name = input_path.stem
    prefix = f"35m_{tag}_" if tag else "35m_"

    print("=" * 60)
    print(f"Step 35m: NUFFT Spectrum Analysis (MATLAB match) [{tag or 'default'}]")
    print("=" * 60)
    print(f"  Input          : {input_path.name}")
    print(f"  Period range   : {period_min}–{period_max} days")
    print(f"  |offset| filter: <= {max_offset} days")
    print(f"  Top vars       : {n_select}  |  Top frac: {top_frac:.0%}")
    print()

    # Load
    print("[35m.1] Loading data...")
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    print(f"  Rows (raw): {len(df):,}  |  Users (raw): {df['author'].nunique():,}")

    feature_cols = [
        c for c in df.columns
        if c.endswith("_mean") and c not in {"author", "offset_from_cd1"}
    ]
    if not feature_cols:
        print("ERROR: no _mean feature columns found")
        return 1
    print(f"  Features: {len(feature_cols)}")

    df_in = df[np.abs(df["offset_from_cd1"]) <= max_offset]
    print(f"  Rows after |offset|<={max_offset:.0f} filter: {len(df_in):,}  "
          f"| Users: {df_in['author'].nunique():,}")

    # Grids
    f_grid, w_grid, periods_grid = make_grids(period_min, period_max)
    print(f"  Freq grid: {len(f_grid)} points  "
          f"({periods_grid[-1]:.1f}–{periods_grid[0]:.1f} days)")

    # Build NUFFT spectrum cube
    print("\n[35m.2] Building NUFFT spectrum cube...")
    cube, users = build_cube(df, feature_cols, w_grid, max_offset)
    print(f"  Cube shape: {cube.shape}  (vars × users × freqs)")

    # SVD — all variables (for importance ranking)
    print("\n[35m.3] Per-user SVD (all variables)...")
    _, conf_all, raw_imp_all = svd_pipeline(cube, list(range(len(feature_cols))), users)
    valid_all = ~np.isnan(conf_all)
    print(f"  Valid SVD: {valid_all.sum():,} / {len(users):,}")

    if valid_all.sum() < 2:
        print("ERROR: fewer than 2 users with valid SVD")
        return 1

    # Variable importance + top-k selection
    print("\n[35m.4] Computing variable importance...")
    mean_imp = compute_mean_importance(raw_imp_all)
    sort_idx = np.argsort(mean_imp)
    top_vi = [int(i) for i in sort_idx[-n_select:]]   # top-k, highest importance last

    print(f"  Top-{n_select} variables (highest → lowest importance):")
    for rank, vi in enumerate(reversed(top_vi), 1):
        print(f"    {rank:2d}. {feature_cols[vi].replace('_mean', '')}  "
              f"(importance={mean_imp[vi]:.4f})")

    plot_importance(
        mean_imp, feature_cols, top_vi,
        output_dir / f"{prefix}variable_importance.png",
        dataset_name=dataset_name,
        n_users=int(valid_all.sum()),
    )

    # Re-run SVD on top-k variables only
    print(f"\n[35m.5] Re-running SVD on top-{n_select} variables...")
    unified_top, conf_top, _ = svd_pipeline(cube, top_vi, users)
    valid_top = ~np.isnan(conf_top)
    print(f"  Valid SVD (top vars): {valid_top.sum():,}")

    if valid_top.sum() < 2:
        print("ERROR: fewer than 2 users with valid top-variable SVD")
        return 1

    # Final spectrum: mean of users above cycling_confidence median
    # Matches MATLAB: mean(new_spectra(new_confidence > prctile(new_confidence, 50), :))
    conf_valid = conf_top[valid_top]
    threshold = float(np.percentile(conf_valid, 50.0))
    top_mask = valid_top & (conf_top > threshold)
    print(f"  Cycling confidence median: {threshold:.4f}  |  Top users: {top_mask.sum():,}")

    mean_spec = unified_top[top_mask, :].mean(axis=0)
    peak_idx = int(np.argmax(mean_spec))
    peak_period = float(periods_grid[peak_idx])
    print(f"  Peak period (argmax): {peak_period:.2f} days")

    plot_spectrum(
        periods_grid, mean_spec, peak_period,
        n_top=int(top_mask.sum()),
        n_total=int(valid_top.sum()),
        dataset_name=dataset_name,
        output_path=output_dir / f"{prefix}dominant_spectrum.png",
    )

    # Save cycling scores CSV
    print("\n[35m.6] Saving cycling scores...")
    in_top_set = {users[ui] for ui in np.where(top_mask)[0]}
    scores_df = pd.DataFrame([
        {"user": users[ui],
         "cycling_score": float(conf_top[ui]),
         "in_top50": users[ui] in in_top_set}
        for ui in range(len(users)) if valid_top[ui]
    ])
    csv_stem = f"nufft_matlab_cycling_scores_{tag}" if tag else "nufft_matlab_cycling_scores"
    out_csv = save_with_timestamp(scores_df, interim_dir, csv_stem)
    print(f"  Saved: {out_csv.name}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Users (valid top-var SVD) : {valid_top.sum():,}")
    print(f"  Top 50% cycling users     : {top_mask.sum():,}")
    print(f"  Peak period (argmax)      : {peak_period:.2f} days")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 35m: NUFFT spectrum analysis (exact MATLAB match)"
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--max-offset", type=float, default=_MAX_OFFSET,
                        help=f"|offset_from_cd1| <= this (default {_MAX_OFFSET})")
    parser.add_argument("--period-min", type=float, default=_PERIOD_MIN)
    parser.add_argument("--period-max", type=float, default=_PERIOD_MAX)
    parser.add_argument("--n-select", type=int, default=_NSELECT,
                        help=f"Top-k variables to select (default {_NSELECT})")
    parser.add_argument("--top-frac", type=float, default=_TOP_FRAC,
                        help="Fraction of users for final spectrum plot (default 0.50)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tag", type=str, default="")

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
        max_offset=args.max_offset,
        period_min=args.period_min,
        period_max=args.period_max,
        n_select=args.n_select,
        top_frac=args.top_frac,
        output_dir=output_dir,
        interim_dir=interim_dir,
        tag=args.tag,
    ))
