"""Phase variance diagnostic: signal vs. measurement noise.

For each phase and feature, decomposes the observed variance of per-user phase
means into two components:

  Var(phase_mean across users) = true biological variance
                                 + mean(within-user day variance / n_days)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   measurement noise floor

If the observed variance ≈ noise floor → what you see in the distribution plot
is entirely estimation noise driven by how many days per phase each user has.
If observed >> noise floor → real biological signal exists.

Key output:
  - SNR = observed_var / noise_floor   (> 1 means signal above noise)
  - Heatmap of SNR across features × phases
  - Bar chart comparing observed SD vs expected noise SD for top features
  - Table sorted by SNR for the Menstrual phase (most interesting biologically)

Usage:
  python scripts/45_phase_variance_diagnostic.py \\
    --phase-labeled data/interim/timeline_phase_labeled_fixed28_raw_*.csv \\
    --eligible     data/interim/eligible_users_timeline_minposts20_span100_minphase3_*.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.io import find_latest_file

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _resolve_features(columns: list[str]) -> tuple[list[str], str]:
    non_feature = {"author", "offset_from_cd1", "phase"}
    for suffix in ("_mean", "_zscore"):
        selected = sorted(c for c in columns if c.endswith(suffix) and c not in non_feature)
        if selected:
            return selected, suffix
    return [], ""


def _clean(col: str) -> str:
    return col.replace("_zscore", "").replace("_mean", "").replace("_", " ")


def _assign_phase(offset: float, period: int, boundaries: dict) -> str | None:
    day = int(offset) % period
    for phase, (start, end) in boundaries.items():
        if start <= day <= end:
            return phase
    return None


def compute_variance_decomposition(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    For each (phase, feature):
      - observed_var  = Var of per-user phase means across users
      - noise_floor   = mean across users of (within-phase day variance / n_days)
      - signal_var    = observed_var - noise_floor  (can be negative = all noise)
      - snr           = observed_var / noise_floor
    Returns a DataFrame indexed by (phase, feature).
    """
    records = []
    for phase in PHASE_ORDER:
        phase_rows = df[df["phase"] == phase]
        for author, grp in phase_rows.groupby("author"):
            n = len(grp)
            for feat in features:
                vals = grp[feat].dropna().values
                if len(vals) < 2:
                    continue
                records.append({
                    "author": author,
                    "phase": phase,
                    "feature": feat,
                    "phase_mean": vals.mean(),
                    "within_var": vals.var(ddof=1),
                    "n_days": len(vals),
                })

    per_user = pd.DataFrame(records)

    rows = []
    for (phase, feat), grp in per_user.groupby(["phase", "feature"]):
        observed_var = grp["phase_mean"].var(ddof=1)
        # Expected noise: E[within_var / n_days] across users
        noise_floor  = (grp["within_var"] / grp["n_days"]).mean()
        signal_var   = observed_var - noise_floor
        snr          = observed_var / noise_floor if noise_floor > 0 else np.nan
        avg_n_days   = grp["n_days"].mean()
        rows.append({
            "phase":        phase,
            "feature":      feat,
            "observed_sd":  np.sqrt(max(observed_var, 0)),
            "noise_sd":     np.sqrt(max(noise_floor,  0)),
            "signal_sd":    np.sqrt(max(signal_var,   0)),
            "snr":          snr,
            "avg_n_days":   avg_n_days,
            "n_users":      len(grp),
        })

    return pd.DataFrame(rows)


def plot_snr_heatmap(decomp: pd.DataFrame, output_dir: Path) -> None:
    pivot = decomp.pivot(index="feature", columns="phase", values="snr")[PHASE_ORDER]
    pivot.index = [_clean(f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(8, max(10, len(pivot) * 0.28)))
    vmax = min(decomp["snr"].quantile(0.95), 5.0)

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=1.0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(PHASE_ORDER)))
    ax.set_xticklabels(PHASE_ORDER, fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    # Annotate each cell
    for i in range(len(pivot)):
        for j in range(len(PHASE_ORDER)):
            v = pivot.values[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6, color="black" if v < vmax * 0.8 else "white")

    plt.colorbar(im, ax=ax, label="SNR = observed var / noise floor  (1 = all noise)")
    ax.set_title(
        "Signal-to-noise ratio per phase × feature\n"
        "SNR = 1 → distribution width is entirely measurement noise\n"
        "SNR > 1 → real biological variance above the noise floor",
        fontsize=10,
    )
    fig.tight_layout()
    path = output_dir / "snr_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_observed_vs_noise(decomp: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """Bar chart: observed SD vs noise SD for the most informative features."""
    # Rank by max SNR across phases
    max_snr = decomp.groupby("feature")["snr"].max().sort_values(ascending=False)
    top_features = max_snr.head(top_n).index.tolist()

    sub = decomp[decomp["feature"].isin(top_features)].copy()
    sub["label"] = sub["feature"].map(_clean)

    ncols = len(PHASE_ORDER)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, max(6, top_n * 0.4)),
                             sharey=True)

    for ax, phase in zip(axes, PHASE_ORDER):
        ph = sub[sub["phase"] == phase].sort_values("snr", ascending=True)
        y = range(len(ph))
        ax.barh(y, ph["observed_sd"], color=PHASE_COLORS[phase], alpha=0.85,
                label="observed SD")
        ax.barh(y, ph["noise_sd"], color="lightgray", alpha=0.9,
                label="noise floor SD")
        ax.set_yticks(list(y))
        ax.set_yticklabels(ph["label"], fontsize=7)
        ax.set_title(f"{phase}\n(avg {ph['avg_n_days'].mean():.1f} days/user)",
                     fontsize=9, color=PHASE_COLORS[phase])
        ax.set_xlabel("SD of phase means", fontsize=8)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Observed SD vs noise floor SD — top {top_n} features by max SNR\n"
        "Gray bar = expected width from measurement noise alone.  "
        "Coloured bar = what you actually see in the distribution plots.",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "observed_vs_noise_sd.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def print_table(decomp: pd.DataFrame) -> None:
    print("\nSNR table — sorted by Menstrual SNR descending")
    print("=" * 90)
    menstrual = decomp[decomp["phase"] == "Menstrual"][
        ["feature", "snr", "observed_sd", "noise_sd", "avg_n_days"]
    ].sort_values("snr", ascending=False)
    print(f"{'Feature':<45} {'SNR':>6} {'ObsSD':>8} {'NoiseSD':>8} {'AvgDays':>8}")
    print("-" * 90)
    for _, row in menstrual.iterrows():
        flag = "  <<< signal" if row["snr"] > 1.5 else ""
        print(f"{_clean(row['feature']):<45} {row['snr']:>6.2f} "
              f"{row['observed_sd']:>8.4f} {row['noise_sd']:>8.4f} "
              f"{row['avg_n_days']:>8.1f}{flag}")

    print("\nPhase avg_n_days summary:")
    print(decomp.groupby("phase")["avg_n_days"].mean().reindex(PHASE_ORDER).to_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    interim = ROOT / "data" / "interim"
    parser.add_argument("--phase-labeled", type=Path, default=None)
    parser.add_argument("--eligible", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "phase_variance_diagnostic")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--menstrual-days", type=int, default=4)
    parser.add_argument("--follicular-days", type=int, default=11)
    parser.add_argument("--ovulation-days", type=int, default=3)
    args = parser.parse_args()

    if args.phase_labeled is None:
        candidates = sorted(interim.glob("timeline_phase_labeled_fixed28_*.csv"))
        pool = [f for f in candidates if "_raw_" in f.name]
        args.phase_labeled = max(pool or candidates, key=lambda p: p.stat().st_mtime)

    if args.eligible is None:
        el = [f for f in sorted(interim.glob(
              "eligible_users_timeline_minposts20_span100_minphase3_*.csv"))
              if not any(t in f.name for t in ("_no_anchors_", "_bc_"))]
        if el:
            pl_mtime = args.phase_labeled.stat().st_mtime
            after = [f for f in el if f.stat().st_mtime >= pl_mtime]
            args.eligible = min(after, key=lambda f: f.stat().st_mtime) if after else el[-1]
        else:
            args.eligible = find_latest_file(
                interim, "eligible_users_timeline_minposts20_span100_minphase3_*.csv")

    return args


def main() -> None:
    args = _parse_args()

    log.info("Eligible: %s", args.eligible.name)
    eligible_authors = set(pd.read_csv(args.eligible, usecols=["author"])["author"].unique())

    log.info("Phase-labeled: %s", args.phase_labeled.name)
    df = pd.read_csv(args.phase_labeled)
    df = df[df["author"].isin(eligible_authors)].copy()

    m, f, o = args.menstrual_days, args.follicular_days, args.ovulation_days
    l = args.period - m - f - o
    boundaries = {
        "Menstrual":  (0,           m - 1),
        "Follicular": (m,           m + f - 1),
        "Ovulation":  (m + f,       m + f + o - 1),
        "Luteal":     (m + f + o,   args.period - 1),
    }
    log.info("Phase model: period=%d  M=%d F=%d O=%d L=%d", args.period, m, f, o, l)
    df["phase"] = df["offset_from_cd1"].apply(
        lambda x: _assign_phase(x, args.period, boundaries))
    df = df[df["phase"].isin(PHASE_ORDER)].copy()

    if args.window:
        df = df[df["offset_from_cd1"].between(-args.window, args.window)].copy()
        log.info("±%d day window → %d rows", args.window, len(df))

    log.info("Phase counts:\n%s", df["phase"].value_counts().reindex(PHASE_ORDER).to_string())

    features, suffix = _resolve_features(list(df.columns))
    log.info("%d features (suffix=%s)", len(features), suffix)

    log.info("Computing variance decomposition…")
    decomp = compute_variance_decomposition(df, features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_snr_heatmap(decomp, args.output_dir)
    plot_observed_vs_noise(decomp, args.output_dir)
    print_table(decomp)


if __name__ == "__main__":
    main()
