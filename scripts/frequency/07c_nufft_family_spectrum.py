#!/usr/bin/env python3
"""NUFFT spectrum per attribute family — no SVD.

Stage-1 of runner_compute_frequencies_grouped_norm.m only: reduce the 69
features to 5 family composites (per-user z-score + family PCA), then compute
a per-user NUFFT spectrum for each family and average across users to read off
each family's dominant cyclic period. The SVD family-fusion / importance step
of the MATLAB script is intentionally omitted — here we want each family's own
signal, not a unified cross-family rhythm.

Per-user preprocessing matches the MATLAB grouped loop:
    x_detrended = detrend(x - mean(x))          # mean + linear-trend removal
    X = nufft(x_detrended, t, w_grid); abs(X)/N
With --zscore-series the detrended series is additionally divided by its SD
(the 07b convention), which equalises each user's contribution to the mean.

Outputs (in --output-dir):
  family_spectrum_<Family>.png   -- mean spectrum + peak, per family
  family_spectrum_peaks.csv      -- peak period (days) per family
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import compute_family_pca

_MIN_OBS = 4
_EPS = 1e-12


def make_grids(period_min: float, period_max: float):
    """MATLAB grids: n = 5*(pmax-pmin); f = linspace(1/pmax, 1/pmin, n)."""
    n = int(5 * (period_max - period_min))
    f_grid = np.linspace(1.0 / period_max, 1.0 / period_min, n)
    w_grid = 2.0 * np.pi * f_grid
    periods_grid = 1.0 / f_grid
    return w_grid, periods_grid


def nufft_mag(t: np.ndarray, x: np.ndarray, w_grid: np.ndarray) -> np.ndarray:
    """abs(nufft(x, t, w_grid)) / len(x), matching MATLAB nufft conventions."""
    T_max = float(np.max(np.abs(t)))
    if T_max == 0:
        return np.zeros(len(w_grid))
    t_scaled = t * (np.pi / T_max)
    s_scaled = 2.0 * np.pi * w_grid * (T_max / np.pi)
    X = finufft.nufft1d3(t_scaled.astype(np.float64), x.astype(np.complex128),
                         s_scaled.astype(np.float64), isign=-1, eps=1e-9)
    return np.abs(X) / len(x)


def family_spectra(df: pd.DataFrame, feat: str, w_grid: np.ndarray,
                   zscore_series: bool) -> np.ndarray:
    """Per-user NUFFT magnitude spectra for one family composite -> (n_users x n_freqs)."""
    sub = df[["author", "offset_from_cd1", feat]].dropna(subset=[feat])
    rows = []
    for _, grp in sub.groupby("author", sort=False):
        if len(grp) < _MIN_OBS:
            continue
        t = grp["offset_from_cd1"].values.astype(float)
        x = grp[feat].values.astype(float)
        x = linear_detrend(x - x.mean())               # MATLAB: detrend(x - mean(x))
        if zscore_series:
            sd = x.std(ddof=1)
            if sd > 0:
                x = x / sd
        rows.append(nufft_mag(t, x, w_grid))
    return np.array(rows) if rows else np.empty((0, len(w_grid)))


def plot_family(periods, mean_all, peak_all, mean_top, peak_top, family, n_users, n_top, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(periods, mean_all, color="#2c7bb6", lw=2, label=f"all users (n={n_users})")
    ax.axvline(peak_all, color="#2c7bb6", ls="--", lw=1.2, label=f"peak {peak_all:.1f} d")
    ax.plot(periods, mean_top, color="#d7191c", lw=2, alpha=0.8,
            label=f"top-50% concentrated (n={n_top})")
    ax.axvline(peak_top, color="#d7191c", ls=":", lw=1.2, label=f"peak {peak_top:.1f} d")
    ax.set_xlabel("Number of days per cycle")
    ax.set_ylabel("Mean spectral magnitude")
    ax.set_title(f"NUFFT family spectrum (no SVD): {family}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=ROOT / "data/interim/eligible_users_timeline_minposts20_span100_no_anchors_20260517T134609.csv")
    ap.add_argument("--families", type=Path, default=ROOT / "data/raw/Attribute_families_cycles.xlsx")
    ap.add_argument("--period-min", type=float, default=20.0)
    ap.add_argument("--period-max", type=float, default=40.0)
    ap.add_argument("--max-offset", type=float, default=90.0)
    ap.add_argument("--top-frac", type=float, default=0.50,
                    help="Fraction of most spectrally-concentrated users for the second curve")
    ap.add_argument("--zscore-series", action="store_true",
                    help="Divide each detrended user series by its SD (07b convention)")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "nufft_family")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {args.input.name}")
    df = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    df = df[np.abs(df["offset_from_cd1"]) <= args.max_offset].copy()
    print(f"Rows after |offset|<={args.max_offset:.0f}: {len(df):,}  users: {df['author'].nunique():,}")

    # Stage 1a: per-user z-score + family PCA (grouped_norm: per_user_normalize=True, no sign fix)
    df, families = compute_family_pca(df, args.families, per_user_normalize=True, sign_fix=False)

    w_grid, periods = make_grids(args.period_min, args.period_max)
    print(f"Freq grid: {len(w_grid)} points ({periods[-1]:.1f}-{periods[0]:.1f} days)\n")

    peaks = []
    for feat in families:
        specs = family_spectra(df, feat, w_grid, args.zscore_series)
        if len(specs) < 2:
            print(f"{feat:24s}  too few users, skipped")
            continue

        mean_all = specs.mean(axis=0)
        peak_all = float(periods[int(np.argmax(mean_all))])

        # "concentration" per user = peak magnitude / mean magnitude (peaky = cycling)
        concentration = specs.max(axis=1) / (specs.mean(axis=1) + _EPS)
        k = max(2, int(len(specs) * args.top_frac))
        top_idx = np.argsort(concentration)[-k:]
        mean_top = specs[top_idx].mean(axis=0)
        peak_top = float(periods[int(np.argmax(mean_top))])

        name = feat.replace("family_", "").replace("_", " ").title()
        out = args.output_dir / f"family_spectrum_{name.replace(' ', '_')}.png"
        plot_family(periods, mean_all, peak_all, mean_top, peak_top, name, len(specs), k, out)
        peaks.append({"family": name, "n_users": len(specs),
                      "peak_all_days": peak_all, "peak_top50_days": peak_top})
        print(f"{name:18s}  n={len(specs):4d}  peak(all)={peak_all:5.1f} d   peak(top50)={peak_top:5.1f} d   -> {out.name}")

    pd.DataFrame(peaks).to_csv(args.output_dir / "family_spectrum_peaks.csv", index=False)
    print(f"\nSaved peaks -> {args.output_dir / 'family_spectrum_peaks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
