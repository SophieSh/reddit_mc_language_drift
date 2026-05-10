"""Step 10 — Phase Feature Distributions: Population vs Minority Check
=======================================================================
Validates whether group-level phase peaks in linguistic features reflect
population-wide effects or are driven by a strong minority.

For each key feature:
  1. Per-user, compute mean z-score in each phase (adaptive boundaries,
     detected-period users only — same setup as script 12).
  2. Auto-detect the peak phase: phase with the largest absolute deviation
     of the group-level mean z-score from zero.
  3. Compute pct_consistent = % of users whose z-score in the peak phase
     has the same sign as the group-level peak (i.e., goes in the same
     direction as the population average).
  4. Plot: box plot + individual user scatter dots per phase, with the
     peak phase highlighted and pct_consistent annotated.

Interpretation guide:
  pct_consistent > 65%  → broad population effect
  pct_consistent 50-65% → mild majority effect, real but noisy
  pct_consistent < 50%  → minority-driven: a strong subgroup pulls the mean

Pipeline:
  [1] Load daily aggregated timeline (step 06)
  [2] Build user→period map (step 07, detected-period users only)
  [3] Label days with adaptive phase boundaries (same logic as script 12)
  [4] Compute per-user z-scores from _mean columns
  [5] Per-user per-phase mean z-score aggregation
  [6] For each feature: detect peak phase + compute pct_consistent
  [7] Save box/strip plots + summary CSV

Outputs (reports/phase_distributions/):
  {feature}_distribution_{timestamp}.png
  phase_peak_summary_{timestamp}.csv

Usage:
  python scripts/10_phase_feature_distributions.py                  # all features, skip plots
  python scripts/10_phase_feature_distributions.py --plots          # all features + plots
  python scripts/10_phase_feature_distributions.py --features readability negative_sentiment
  python scripts/10_phase_feature_distributions.py --features readability --plots
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import normalize_features_per_user_zscore
from src.config import load_config
from src.constants import PHASE_ORDER, PHASE_COLORS
from src.io import find_latest_file

warnings.filterwarnings("ignore", category=UserWarning)

# Key features to analyse (base names without _mean suffix)
DEFAULT_FEATURES = [
    "readability",
    "negative_sentiment",
    "positive_sentiment",
    "valence_dict_average",
    "hedging_epistemic_phrases",
    "idea_density_depid_density",
    "arousal_average",
]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument(
        "--features", nargs="+", default=None,
        help="Base feature names (without _mean). Defaults to ALL available features.",
    )
    p.add_argument("--plots", action="store_true",
                   help="Generate PNG plots. Off by default (slow for many features).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_phase_labeled(interim_dir: Path, pattern: str) -> pd.DataFrame:
    path = find_latest_file(interim_dir, pattern)
    if path is None:
        raise FileNotFoundError(
            f"No {pattern} found in data/interim/. "
            "Run scripts/08b_label_phases.py first."
        )
    logging.info(f"Loading phase-labeled timeline: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users")
    return df


# ── Analysis ───────────────────────────────────────────────────────────────────

def compute_user_phase_means(df: pd.DataFrame, mean_cols: list[str]) -> pd.DataFrame:
    """Return per-user per-phase mean z-scores."""
    logging.info(f"  Computing per-user z-scores for {len(mean_cols)} features…")
    df = normalize_features_per_user_zscore(df, mean_cols, user_col="author")
    # normalize_features_per_user_zscore produces {feature}_zscore from {feature}_mean
    # e.g. readability_mean → readability_zscore  (NOT readability_mean_zscore)
    zscore_cols = [c.replace("_mean", "_zscore") for c in mean_cols
                   if c.replace("_mean", "_zscore") in df.columns]
    if not zscore_cols:
        raise RuntimeError(
            "No _zscore columns produced — check normalize_features_per_user_zscore output."
        )
    user_phase = (
        df.groupby(["author", "phase"])[zscore_cols].mean().reset_index()
    )
    return user_phase, zscore_cols


def peak_phase_stats(user_phase: pd.DataFrame, zscore_col: str) -> dict:
    """For one feature: find peak phase + compute pct_consistent for ALL phases.

    Returns a dict with:
      - peak_phase, group_mean_zscore, direction, pct_consistent, n_users  (peak phase)
      - per_phase: dict of {phase: {mean, pct_consistent, n}}  (all phases)
    """
    phase_means = (
        user_phase.groupby("phase")[zscore_col]
        .mean()
        .reindex(PHASE_ORDER)
        .dropna()
    )
    if phase_means.empty:
        return {}

    # Per-phase breakdown
    per_phase = {}
    for phase in PHASE_ORDER:
        data = user_phase[user_phase["phase"] == phase][zscore_col].dropna()
        if len(data) == 0:
            continue
        grp_mean = phase_means.get(phase, float("nan"))
        direction = "positive" if grp_mean > 0 else "negative"
        n_cons = (data > 0).sum() if direction == "positive" else (data < 0).sum()
        per_phase[phase] = {
            "mean": round(grp_mean, 4),
            "pct_consistent": round(n_cons / len(data) * 100, 1),
            "n": len(data),
        }

    # Peak = phase with largest absolute group-level mean z-score
    peak_phase = phase_means.abs().idxmax()
    peak = per_phase[peak_phase]

    return {
        "peak_phase": peak_phase,
        "group_mean_zscore": peak["mean"],
        "direction": "positive" if peak["mean"] > 0 else "negative",
        "pct_consistent": peak["pct_consistent"],
        "n_users": peak["n"],
        "per_phase": per_phase,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_feature_distribution(
    user_phase: pd.DataFrame,
    zscore_col: str,
    feature_label: str,
    stats: dict,
    output_path: Path,
):
    """Box plot + individual user scatter per phase, with peak phase highlighted."""
    fig, ax = plt.subplots(figsize=(9, 5))

    phase_data = [
        user_phase[user_phase["phase"] == ph][zscore_col].dropna().values
        for ph in PHASE_ORDER
    ]

    # Box plots
    bp = ax.boxplot(
        phase_data,
        positions=range(len(PHASE_ORDER)),
        patch_artist=True,
        widths=0.4,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    peak_idx = PHASE_ORDER.index(stats["peak_phase"]) if stats else None
    for i, (patch, phase) in enumerate(zip(bp["boxes"], PHASE_ORDER)):
        color = PHASE_COLORS[phase]
        alpha = 1.0 if i == peak_idx else 0.4
        patch.set_facecolor(color)
        patch.set_alpha(alpha)

    # Individual user scatter
    rng = np.random.default_rng(42)
    for i, (data, phase) in enumerate(zip(phase_data, PHASE_ORDER)):
        if len(data) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        alpha = 0.6 if i == peak_idx else 0.2
        ax.scatter(
            np.full(len(data), i) + jitter, data,
            color=PHASE_COLORS[phase], s=8, alpha=alpha, zorder=2,
        )

    # Annotate peak phase
    if stats:
        peak_idx_plot = PHASE_ORDER.index(stats["peak_phase"])
        ymax = ax.get_ylim()[1]
        ax.annotate(
            f"peak phase\n{stats['pct_consistent']:.0f}% consistent\n(n={stats['n_users']})",
            xy=(peak_idx_plot, ymax * 0.88),
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(PHASE_ORDER)))
    ax.set_xticklabels(PHASE_ORDER, fontsize=11)
    ax.set_ylabel("Per-user mean z-score", fontsize=10)
    ax.set_title(
        f"{feature_label}\nPer-user phase distribution  |  peak={stats.get('peak_phase', '?')}  "
        f"group_mean={stats.get('group_mean_zscore', 0):+.3f}  "
        f"({stats.get('direction', '')})",
        fontsize=10,
    )

    legend_patches = [
        mpatches.Patch(facecolor=PHASE_COLORS[ph], label=ph,
                       alpha=1.0 if ph == stats.get("peak_phase") else 0.4)
        for ph in PHASE_ORDER
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    interim_dir = ROOT / cfg["paths"]["interim"]
    output_dir = ROOT / cfg["paths"]["reports"] / "phase_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # [1] Load phase-labeled timeline (step 08b)
    files_cfg = cfg["paths"]["files"]
    df = load_phase_labeled(interim_dir, files_cfg["phase_labeled"] + "_*.csv")

    # [2] Resolve feature names
    NON_FEATURE = {"author", "offset_from_cd1", "phase", "subreddit",
                   "created_utc", "post_id", "permalink", "title", "selftext"}
    if args.features:
        feature_bases = args.features
        mean_cols = [f"{f}_mean" for f in feature_bases if f"{f}_mean" in df.columns]
        missing = [f for f in feature_bases if f"{f}_mean" not in df.columns]
        if missing:
            logging.warning(f"  Features not found in data (skipped): {missing}")
    else:
        # All available _mean columns that are numeric and not metadata
        mean_cols = [
            c for c in df.columns
            if c.endswith("_mean")
            and c.split("_mean")[0] not in NON_FEATURE
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().mean() > 0.1
        ]
        feature_bases = [c.replace("_mean", "") for c in mean_cols]
        logging.info(f"  Using all {len(mean_cols)} available _mean features")
    if not mean_cols:
        raise RuntimeError("No feature columns found in data.")

    # [3] Per-user z-scores + per-phase aggregation
    logging.info("\n[3] Computing z-scores and per-user phase means…")
    user_phase, zscore_cols = compute_user_phase_means(df, mean_cols)

    # [4] Per-feature: detect peak phase + pct_consistent
    logging.info("\n[4] Computing peak phase statistics…")
    summary_rows = []
    for feat_base, zscore_col in zip(feature_bases, zscore_cols):
        if zscore_col not in user_phase.columns:
            logging.warning(f"  {feat_base}: zscore column not found, skipping")
            continue

        stats = peak_phase_stats(user_phase, zscore_col)
        if not stats:
            logging.warning(f"  {feat_base}: insufficient data, skipping")
            continue

        label = (
            "majority-driven" if stats["pct_consistent"] >= 65
            else ("mild majority" if stats["pct_consistent"] >= 50
                  else "minority-driven")
        )
        # Build flat row with per-phase columns for CSV
        row = {"feature": feat_base, "peak_phase": stats["peak_phase"],
               "peak_group_mean": stats["group_mean_zscore"],
               "peak_pct_consistent": stats["pct_consistent"],
               "peak_n": stats["n_users"], "label": label}
        for ph in PHASE_ORDER:
            pp = stats["per_phase"].get(ph, {})
            row[f"{ph}_mean"] = pp.get("mean", float("nan"))
            row[f"{ph}_pct_consistent"] = pp.get("pct_consistent", float("nan"))
            row[f"{ph}_n"] = pp.get("n", 0)
        logging.info(
            f"  {feat_base:40s}  peak={stats['peak_phase']:12s}  "
            f"grp_mean={stats['group_mean_zscore']:+.4f}  "
            f"pct={stats['pct_consistent']:5.1f}%  n={stats['n_users']:4d}  [{label}]"
        )
        summary_rows.append(row)

        if args.plots:
            plot_path = output_dir / f"{feat_base}_distribution_{timestamp}.png"
            plot_feature_distribution(user_phase, zscore_col, feat_base, stats, plot_path)
            logging.info(f"    → {plot_path.name}")

    # [5] Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"phase_peak_summary_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False)
    logging.info(f"\nSummary → {csv_path}")

    # Print summary table — peak phase + per-phase pct_consistent
    ph_w = 7  # column width per phase
    header_phases = "  ".join(f"{ph[:ph_w]:>{ph_w}}" for ph in PHASE_ORDER)
    print("\n" + "=" * 110)
    print(f"PHASE PEAK SUMMARY  (pct_consistent = % users going in same direction as group mean)")
    print(f"{'feature':40s}  {'peak':12s}  {'grp_mean':>8}  {header_phases}  label")
    print("-" * 110)
    for row in summary_rows:
        phase_cols = "  ".join(
            f"{row.get(f'{ph}_pct_consistent', float('nan')):>{ph_w}.1f}"
            if not pd.isna(row.get(f'{ph}_pct_consistent', float('nan'))) else f"{'—':>{ph_w}}"
            for ph in PHASE_ORDER
        )
        print(
            f"{row['feature']:40s}  {row['peak_phase']:12s}  "
            f"{row['peak_group_mean']:>+8.4f}  {phase_cols}  {row['label']}"
        )
    print("=" * 110)
    print(f"\nDone. Outputs → {output_dir}/")


if __name__ == "__main__":
    main()
