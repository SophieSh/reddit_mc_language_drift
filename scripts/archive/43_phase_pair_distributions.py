"""Pairwise phase distribution comparison plots for menstrual cycle phases.

For each of the 6 pairwise combinations of the 4 phases, produces a figure with
one KDE subplot per feature, overlapping distributions for the two phases.
Also prints a Mann-Whitney U summary table (Bonferroni-corrected within each pair).

Usage:
  python scripts/43_phase_pair_distributions.py               # raw values (default)
  python scripts/43_phase_pair_distributions.py --zscore      # z-scored values
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.io import find_latest_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PAIRS: list[tuple[str, str]] = list(combinations(PHASE_ORDER, 2))

CANDIDATE_FEATURES: list[str] = [
    "valence_dict_average",
    "positive_sentiment",
    "negative_sentiment",
    "dominance_average",
    "arousal_average",
    "imaginability_average",
    "readability",
    "spelling_errors_frac",
    "num_words",
    "avg_word_length",
    "hedging_modal_verbs",
    "idea_density_cpidr_density",
    "word_frequency",
    "unique_word_fraction",
    "concrete_abstract_ratio",
]


def _resolve_features(columns: list[str]) -> tuple[list[str], str]:
    """Return (all feature column names, suffix) — no cap.

    Tries '_mean' first (raw file), then '_zscore'.
    Excludes metadata columns (author, offset_from_cd1, phase).
    """
    non_feature = {"author", "offset_from_cd1", "phase"}
    for suffix in ("_mean", "_zscore"):
        selected = sorted(
            c for c in columns if c.endswith(suffix) and c not in non_feature
        )
        if selected:
            return selected, suffix
    return [], ""


def _clean_label(col: str) -> str:
    return col.replace("_zscore", "").replace("_mean", "").replace("_", " ")


def _plot_pair(
    df: pd.DataFrame,
    phase_a: str,
    phase_b: str,
    features: list[str],
    suffix: str,
    output_dir: Path,
) -> None:
    sub_a = df[df["phase"] == phase_a]
    sub_b = df[df["phase"] == phase_b]
    n_a, n_b = len(sub_a), len(sub_b)

    ncols = 6
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes_flat = np.array(axes).flatten()

    color_a = PHASE_COLORS[phase_a]
    color_b = PHASE_COLORS[phase_b]
    xlabel = "deviation from user mean" if suffix == "_mean" else "z-score (centered)"

    for ax, col in zip(axes_flat[:len(features)], features):
        vals_a = sub_a[col].dropna().values
        vals_b = sub_b[col].dropna().values

        all_vals = np.concatenate([v for v in [vals_a, vals_b] if len(v) >= 2])
        if len(all_vals) == 0:
            continue
        # clip extremes so rare outliers don't flatten the KDE visually
        lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        x_grid = np.linspace(lo, hi, 300)

        for vals, color, label, ls in [
            (vals_a, color_a, phase_a, "-"),
            (vals_b, color_b, phase_b, "--"),
        ]:
            vals_clipped = vals[(vals >= lo) & (vals <= hi)]
            if len(vals_clipped) < 2:
                continue
            try:
                kernel = stats.gaussian_kde(vals_clipped, bw_method="scott")
                kde_vals = kernel(x_grid)
                ax.plot(x_grid, kde_vals, color=color, linewidth=2.5, linestyle=ls, label=label)
            except np.linalg.LinAlgError:
                # Degenerate (near-constant) feature — skip KDE, just mark median
                pass

            med = np.median(vals_clipped)
            ax.axvline(med, color=color, linewidth=1.5, linestyle=ls, alpha=0.9)

        ax.set_title(_clean_label(col), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, framealpha=0.6)

    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Phase Distribution Comparison: {phase_a} vs {phase_b}"
        f"\n(n={n_a} {phase_a} users, n={n_b} {phase_b} users — vertical lines = median)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    fname = f"dist_{phase_a.replace(' ', '_')}_vs_{phase_b.replace(' ', '_')}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_dir / fname)


def _print_summary(df: pd.DataFrame, features: list[str]) -> None:
    n_tests = len(features) * len(PAIRS)  # 12 features × 6 pairs = 72
    print("\t".join(["Pair", "Feature", "Mean_A±SD_A", "Mean_B±SD_B", "MW-U p (raw)", "p (Bonf)"]))
    print("-" * 100)
    for phase_a, phase_b in PAIRS:
        sub_a = df[df["phase"] == phase_a]
        sub_b = df[df["phase"] == phase_b]
        for col in features:
            vals_a = sub_a[col].dropna().values
            vals_b = sub_b[col].dropna().values
            mean_a, std_a = vals_a.mean(), vals_a.std()
            mean_b, std_b = vals_b.mean(), vals_b.std()
            if len(vals_a) >= 2 and len(vals_b) >= 2:
                _, p_raw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            else:
                p_raw = float("nan")
            p_bonf = min(p_raw * n_tests, 1.0)
            print(
                f"{phase_a} vs {phase_b}\t{_clean_label(col)}\t"
                f"{mean_a:.3f}±{std_a:.3f}\t{mean_b:.3f}±{std_b:.3f}\t"
                f"{p_raw:.4g}\t{p_bonf:.4g}"
            )
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise phase distribution comparison plots.")
    interim = ROOT / "data" / "interim"
    parser.add_argument("--phase-labeled", type=Path, default=None)
    parser.add_argument("--eligible", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "phase_distributions")
    parser.add_argument("--zscore", action="store_true",
                        help="Use z-scored file instead of raw values (default: raw)")
    parser.add_argument("--window", type=int, default=30,
                        help="Restrict to ±N days from CD1 (default: 30, 0 = no filter)")
    parser.add_argument("--per-user-means", action="store_true", default=True,
                        help="Aggregate to one value per user per phase before plotting (default: on)")
    parser.add_argument("--no-per-user-means", dest="per_user_means", action="store_false")
    parser.add_argument("--period", type=int, default=30,
                        help="Cycle length in days used to re-label phases (default: 30)")
    parser.add_argument("--menstrual-days", type=int, default=4)
    parser.add_argument("--follicular-days", type=int, default=11)
    parser.add_argument("--ovulation-days", type=int, default=3)
    args = parser.parse_args()

    if args.phase_labeled is None:
        candidates = sorted(interim.glob("timeline_phase_labeled_fixed28_*.csv"))
        if args.zscore:
            pool = [f for f in candidates if not any(t in f.name for t in ("_bc_", "_no_anchors_", "_raw_"))]
        else:
            pool = [f for f in candidates if "_raw_" in f.name]
        if not pool:
            pool = candidates
        args.phase_labeled = max(pool, key=lambda p: p.stat().st_mtime) if pool else None
        if args.phase_labeled is None:
            parser.error("Could not auto-detect phase-labeled CSV in data/interim/")

    if args.eligible is None:
        # Pick the eligible-users file whose timestamp is closest to (and after) the phase-labeled file.
        el_candidates = [
            f for f in sorted(interim.glob("eligible_users_timeline_minposts20_span100_minphase3_*.csv"))
            if not any(t in f.name for t in ("_no_anchors_", "_bc_"))
        ]
        if el_candidates:
            # prefer the file created just after the phase-labeled file
            pl_mtime = args.phase_labeled.stat().st_mtime
            after = [f for f in el_candidates if f.stat().st_mtime >= pl_mtime]
            args.eligible = min(after, key=lambda f: f.stat().st_mtime) if after else el_candidates[-1]
        else:
            args.eligible = find_latest_file(interim, "eligible_users_timeline_minposts20_span100_minphase3_*.csv")
        if args.eligible is None:
            parser.error("Could not auto-detect eligible users CSV in data/interim/")

    if args.zscore:
        args.output_dir = args.output_dir.parent / (args.output_dir.name + "_zscore")

    return args


def main() -> None:
    args = _parse_args()

    log.info("Loading eligible users from %s", args.eligible)
    eligible_authors = set(pd.read_csv(args.eligible, usecols=["author"])["author"].unique())
    log.info("Eligible authors: %d", len(eligible_authors))

    log.info("Loading phase-labeled timeline from %s", args.phase_labeled)
    phase_df = pd.read_csv(args.phase_labeled)
    before = len(phase_df)
    phase_df = phase_df[phase_df["author"].isin(eligible_authors)].copy()
    log.info("Filtered %d → %d rows (eligible authors only)", before, len(phase_df))

    # Recompute phase labels from offset_from_cd1 using the requested cycle model.
    m  = args.menstrual_days
    f  = args.follicular_days
    o  = args.ovulation_days
    l  = args.period - m - f - o
    if l <= 0:
        log.error("Luteal length = %d (period %d - %d - %d - %d). Check --period and phase-day args.",
                  l, args.period, m, f, o)
        sys.exit(1)
    boundaries = {
        "Menstrual":  (0,       m - 1),
        "Follicular": (m,       m + f - 1),
        "Ovulation":  (m + f,   m + f + o - 1),
        "Luteal":     (m + f + o, args.period - 1),
    }
    log.info("Phase model: period=%d  M=%d F=%d O=%d L=%d  boundaries=%s",
             args.period, m, f, o, l, boundaries)

    def _assign_phase(offset: float) -> str | None:
        day = int(offset) % args.period
        for phase, (start, end) in boundaries.items():
            if start <= day <= end:
                return phase
        return None

    phase_df["phase"] = phase_df["offset_from_cd1"].apply(_assign_phase)
    phase_df = phase_df[phase_df["phase"].isin(set(PHASE_ORDER))].copy()

    if args.window:
        before = len(phase_df)
        phase_df = phase_df[phase_df["offset_from_cd1"].between(-args.window, args.window)].copy()
        log.info("±%d day window: %d → %d rows", args.window, before, len(phase_df))

    log.info("Phase counts:\n%s", phase_df["phase"].value_counts().to_string())

    features, suffix = _resolve_features(list(phase_df.columns))
    if not features:
        log.error("No expected feature columns found. Aborting.")
        sys.exit(1)
    log.info("Using suffix=%s, %d features: %s", suffix, len(features), [_clean_label(f) for f in features])

    if args.per_user_means:
        # One mean per user per phase using ALL available days — no downsampling.
        # Ovulation will have a wider distribution than Luteal because it genuinely
        # has fewer days; that is honest, not a bug.
        valid_users = [
            author for author, grp in phase_df.groupby("author")
            if all(p in grp["phase"].values for p in PHASE_ORDER)
        ]
        phase_df = phase_df[phase_df["author"].isin(valid_users)]
        phase_df = phase_df.groupby(["author", "phase"])[features].mean().reset_index()
        log.info("Aggregated to per-user phase means: %d rows, %d users",
                 len(phase_df), phase_df["author"].nunique())

        # Center AFTER aggregation using the equal-weighted grand mean across the 4
        # phase means (not the day-count-weighted daily mean, which is Luteal-dominated).
        grand_means = phase_df.groupby("author")[features].transform("mean")
        phase_df = phase_df.copy()
        phase_df[features] = phase_df[features] - grand_means
        log.info("Applied within-user centering on equal-weighted 4-phase grand mean.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for phase_a, phase_b in PAIRS:
        log.info("Plotting: %s vs %s", phase_a, phase_b)
        _plot_pair(phase_df, phase_a, phase_b, features, suffix, args.output_dir)

    _print_summary(phase_df, features)


if __name__ == "__main__":
    main()
