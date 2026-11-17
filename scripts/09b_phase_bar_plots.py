#!/usr/bin/env python3
"""Phase bar plots — one subplot per feature, bars = mean ± SEM per phase.

Accepts any CSV that already has a 'phase' column (or computes phases from
offset_from_cd1 using a fixed period), plus feature columns.

Compatible with both 69 raw _mean features and 5 family_* PCA composites.

Usage:
  # From a phase-labeled file (phase column already present):
  python scripts/09b_phase_bar_plots.py \
    --input data/interim/timeline_phase_labeled_fixed28_raw_no_anchors_*.csv \
    --output-dir reports/phase_bars

  # From a file without phases — compute from offset_from_cd1 with fixed period:
  python scripts/09b_phase_bar_plots.py \
    --input data/interim/eligible_users_timeline_minposts20_span100_no_anchors_*.csv \
    --period 28 \
    --output-dir reports/phase_bars

  # With family PCA (add --family-pca to reduce 69 features to 5 composites):
  python scripts/09b_phase_bar_plots.py \
    --input data/interim/timeline_phase_labeled_fixed28_raw_no_anchors_*.csv \
    --family-pca data/raw/Attribute_families_cycles.xlsx \
    --output-dir reports/phase_bars/family_pca
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.utils import compute_family_pca

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _assign_phases_fixed(offset: pd.Series, period: int) -> pd.Series:
    """Assign phase from offset_from_cd1 using a fixed cycle period."""
    m = max(1, round(period * 4 / 28))
    f = max(1, round(period * 7 / 28))
    o = max(1, round(period * 3 / 28))

    day = offset % period

    def _phase(d):
        if d < m:               return "Menstrual"
        if d < m + f:           return "Follicular"
        if d < m + f + o:       return "Ovulation"
        return "Luteal"

    return day.apply(_phase)


def _resolve_features(df: pd.DataFrame) -> list[str]:
    skip = {"author", "offset_from_cd1", "phase", "bc", "pill_type", "bc_status"}
    family = [c for c in df.columns if c.startswith("family_")]
    if family:
        return family
    mean_cols = [c for c in df.columns if c.endswith("_mean") and c not in skip]
    if mean_cols:
        return mean_cols
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _clean(col: str) -> str:
    if col.startswith("family_"):
        return col[len("family_"):].replace("_", " ").title() + " PC1"
    return col.replace("_mean", "").replace("_zscore", "").replace("_", " ")


def _phase_stats(df: pd.DataFrame, features: list[str], zscore: bool = False) -> pd.DataFrame:
    """Per-user mean-centered phase means → cross-user mean ± SEM. Returns long DataFrame."""
    rows = []
    for feat in features:
        sub = df[["author", "phase", feat]].dropna().copy()
        if sub.empty:
            continue
        # Within-user normalization: remove each user's global baseline
        user_mean = sub.groupby("author")[feat].transform("mean")
        if zscore:
            user_std = sub.groupby("author")[feat].transform("std")
            sub[feat] = (sub[feat] - user_mean) / (user_std + 1e-10)
        else:
            sub[feat] = sub[feat] - user_mean
        user_means = sub.groupby(["author", "phase"])[feat].mean().reset_index()
        for phase, grp in user_means.groupby("phase"):
            vals = grp[feat].values
            n = len(vals)
            if n < 2:
                continue
            rows.append({
                "feature": feat,
                "phase": phase,
                "mean": vals.mean(),
                "sem": vals.std(ddof=1) / np.sqrt(n),
                "n": n,
            })
    return pd.DataFrame(rows)


def _kruskal_p(df: pd.DataFrame, feat: str) -> float | None:
    groups = [grp[feat].dropna().values
              for _, grp in df.groupby("phase")
              if grp[feat].notna().sum() >= 3]
    if len(groups) < 2:
        return None
    try:
        return sp_stats.kruskal(*groups).pvalue
    except Exception:
        return None


def plot_phase_bars(
    df: pd.DataFrame,
    features: list[str],
    output_path: Path,
    n_cols: int = 4,
    zscore: bool = False,
) -> None:
    stats = _phase_stats(df, features, zscore=zscore)
    if stats.empty:
        log.warning("No phase stats computed — check input data.")
        return

    n_feat = len(features)
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for idx, feat in enumerate(features):
        ax = axes[idx // n_cols][idx % n_cols]
        feat_df = stats[stats["feature"] == feat]
        if feat_df.empty:
            ax.set_visible(False)
            continue

        means, sems, colors, labels = [], [], [], []
        for phase in PHASE_ORDER:
            row = feat_df[feat_df["phase"] == phase]
            if row.empty:
                means.append(0); sems.append(0)
            else:
                means.append(float(row["mean"].iloc[0]))
                sems.append(float(row["sem"].iloc[0]))
            colors.append(PHASE_COLORS[phase])
            labels.append(phase)

        x = np.arange(len(PHASE_ORDER))
        bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                      alpha=0.8, width=0.6, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([p[:3] for p in PHASE_ORDER], fontsize=8)
        ax.set_title(_clean(feat), fontsize=9, fontweight="bold")
        ax.set_ylabel("z-score ± SEM" if zscore else "mean ± SEM", fontsize=7)
        ax.grid(axis="y", alpha=0.25)

        # Kruskal-Wallis p on subplot
        p = _kruskal_p(df, feat)
        if p is not None:
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.set_xlabel(f"KW p={p:.3g} {sig}", fontsize=7)

    for idx in range(n_feat, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    n_users = df["author"].nunique()
    norm_label = "within-user z-score" if zscore else "mean-centered"
    fig.suptitle(f"Feature Values by Phase  (n={n_users:,} users, {norm_label}, ± SEM)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Input CSV with feature columns and either a 'phase' column "
                             "or 'offset_from_cd1' (requires --period).")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "phase_bars")
    parser.add_argument("--period", type=int, default=None,
                        help="Fixed cycle period (days) used to assign phases from "
                             "offset_from_cd1. Required if input has no 'phase' column.")
    parser.add_argument("--family-pca", type=Path, default=None,
                        help="Path to Attribute_families_cycles.xlsx to reduce features "
                             "to 5 family PC1 composites before plotting.")
    parser.add_argument("--window", type=int, default=None,
                        help="Restrict to offset_from_cd1 in [-window, +window] before plotting.")
    parser.add_argument("--eligible", type=Path, default=None,
                        help="Optional eligible-users CSV; filters input to those authors.")
    parser.add_argument("--n-cols", type=int, default=4,
                        help="Number of subplot columns (default: 4).")
    parser.add_argument("--zscore", action="store_true",
                        help="Within-user z-score (divide by user std) instead of mean-centering only.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.input.name)
    df = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    log.info("  %d rows, %d users", len(df), df["author"].nunique())

    if args.window is not None:
        before = len(df)
        df = df[df["offset_from_cd1"].between(-args.window, args.window)].copy()
        log.info("  Window ±%d days: %d → %d rows, %d users",
                 args.window, before, len(df), df["author"].nunique())

    if args.eligible is not None:
        eligible_authors = set(pd.read_csv(args.eligible, usecols=["author"])["author"])
        df = df[df["author"].isin(eligible_authors)].copy()
        log.info("  After eligible filter: %d rows, %d users", len(df), df["author"].nunique())

    if "phase" not in df.columns:
        if args.period is None:
            parser.error("Input has no 'phase' column — provide --period to compute phases.")
        log.info("  Assigning phases (fixed %d-day cycle) ...", args.period)
        df["phase"] = _assign_phases_fixed(df["offset_from_cd1"], args.period)

    if args.family_pca:
        log.info("Computing family PCA from %s ...", args.family_pca.name)
        df, _ = compute_family_pca(df, args.family_pca)

    features = sorted(_resolve_features(df), key=_clean)
    log.info("Plotting %d features ...", len(features))

    tag = "family_pca" if args.family_pca else "all_features"
    norm_tag = "_zscore" if args.zscore else ""
    out = args.output_dir / f"phase_bars_{tag}{norm_tag}_{pd.Timestamp.now().strftime('%Y%m%dT%H%M%S')}.png"
    plot_phase_bars(df, features, out, n_cols=args.n_cols, zscore=args.zscore)


if __name__ == "__main__":
    main()
