"""Pairwise phase difference distributions.

For each of the 6 phase pairs and each feature, computes the per-user difference
(Phase A mean − Phase B mean) and plots its distribution. One plot per pair.

Why differences instead of overlapping KDEs:
  With Cohen's d ≈ 0.14 the two raw distributions have ~94% overlap and look
  identical. The difference distribution has only one curve; the question
  "does the median sit left or right of zero?" is immediately readable.

Pipeline:
  1. Load phase-labeled raw timeline, filter to eligible users.
  2. Apply ±window day filter (default ±30).
  3. One mean per user per phase using ALL available days.
  4. Center by equal-weighted 4-phase grand mean per user.
  5. For each pair, compute diff = phase_A_mean − phase_B_mean per user.
  6. Plot KDE of the differences with:
       - vertical line at 0 (null)
       - vertical line at the median (coloured)
       - shaded 95% bootstrap CI of the median
  7. Print Wilcoxon signed-rank test table (paired, Bonferroni-corrected).

Usage:
  python scripts/44_phase_difference_distributions.py \\
    --phase-labeled data/interim/timeline_phase_labeled_fixed28_raw_*.csv \\
    --eligible     data/interim/eligible_users_timeline_minposts20_span100_minphase3_*.csv
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
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.io import find_latest_file

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PAIRS: list[tuple[str, str]] = list(combinations(PHASE_ORDER, 2))

CANDIDATE_FEATURES: list[str] = [
    "valence_dict_average", "positive_sentiment", "negative_sentiment",
    "dominance_average", "arousal_average", "imaginability_average",
    "readability", "spelling_errors_frac", "num_words", "avg_word_length",
    "hedging_modal_verbs", "idea_density_cpidr_density",
    "word_frequency", "unique_word_fraction", "concrete_abstract_ratio",
]


def _resolve_features(columns: list[str]) -> tuple[list[str], str]:
    present = set(columns)
    for suffix in ("_mean", "_zscore"):
        selected = [f"{f}{suffix}" for f in CANDIDATE_FEATURES if f"{f}{suffix}" in present][:12]
        if selected:
            return selected, suffix
    return [], ""


def _clean_label(col: str) -> str:
    return col.replace("_zscore", "").replace("_mean", "").replace("_", " ")


def _bootstrap_median_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot_medians = np.array([
        np.median(rng.choice(x, size=len(x), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_medians, 2.5)), float(np.percentile(boot_medians, 97.5))


def _plot_pair_differences(
    diff_df: pd.DataFrame,          # columns: feature columns, index = author
    phase_a: str,
    phase_b: str,
    features: list[str],
    output_dir: Path,
) -> None:
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes_flat = np.array(axes).flatten()

    # Blend the two phase colours for the difference curve
    color_a = np.array(plt.matplotlib.colors.to_rgb(PHASE_COLORS[phase_a]))
    color_b = np.array(plt.matplotlib.colors.to_rgb(PHASE_COLORS[phase_b]))
    diff_color = tuple((color_a + color_b) / 2)

    n_users = len(diff_df)

    for ax, col in zip(axes_flat[:len(features)], features):
        diffs = diff_df[col].dropna().values

        lo, hi = np.percentile(diffs, 1), np.percentile(diffs, 99)
        diffs_clipped = diffs[(diffs >= lo) & (diffs <= hi)]
        x_grid = np.linspace(lo, hi, 300)

        kernel = stats.gaussian_kde(diffs_clipped, bw_method="scott")
        kde_vals = kernel(x_grid)

        ax.plot(x_grid, kde_vals, color=diff_color, linewidth=2.5)
        ax.fill_between(x_grid, kde_vals, alpha=0.15, color=diff_color)

        # Null reference
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="zero (null)")

        # Median + 95% bootstrap CI
        med = float(np.median(diffs_clipped))
        ci_lo, ci_hi = _bootstrap_median_ci(diffs_clipped)
        ax.axvline(med, color=diff_color, linewidth=2.0, linestyle="-",
                   label=f"median={med:.4f}")
        ax.axvspan(ci_lo, ci_hi, alpha=0.25, color=diff_color, label="95% CI")

        ax.set_title(_clean_label(col), fontsize=9)
        ax.set_xlabel(f"{phase_a} − {phase_b} (deviation from user mean)", fontsize=7)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, framealpha=0.6)

    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-user difference: {phase_a} − {phase_b}  (n={n_users} users)\n"
        f"Black dashed = 0 (null).  Coloured line = median.  Shaded = 95% bootstrap CI of median.",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    fname = f"diff_{phase_a.replace(' ', '_')}_minus_{phase_b.replace(' ', '_')}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_dir / fname)


def _print_summary(wide: pd.DataFrame, features: list[str]) -> None:
    """Wilcoxon signed-rank test on per-user differences, Bonferroni-corrected."""
    n_tests = len(features) * len(PAIRS)
    print("\t".join(["Pair", "Feature", "Median diff", "Wilcoxon p (raw)", "p (Bonf)", "Sig"]))
    print("-" * 110)
    for phase_a, phase_b in PAIRS:
        col_a = f"{phase_a}__{phase_b}"
        for feat in features:
            diffs = wide[feat][col_a]
            diffs = diffs[~np.isnan(diffs)]
            med = np.median(diffs)
            if len(diffs) >= 10 and not np.all(diffs == 0):
                _, p_raw = wilcoxon(diffs, alternative="two-sided")
            else:
                p_raw = float("nan")
            p_bonf = min(p_raw * n_tests, 1.0)
            sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else ""))
            print(f"{phase_a} vs {phase_b}\t{_clean_label(feat)}\t"
                  f"{med:.4f}\t{p_raw:.4g}\t{p_bonf:.4g}\t{sig}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-user phase difference distributions.")
    interim = ROOT / "data" / "interim"
    parser.add_argument("--phase-labeled", type=Path, default=None)
    parser.add_argument("--eligible", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "phase_differences")
    parser.add_argument("--window", type=int, default=30,
                        help="Restrict to ±N days from CD1 (default: 30)")
    args = parser.parse_args()

    if args.phase_labeled is None:
        candidates = sorted(interim.glob("timeline_phase_labeled_fixed28_*.csv"))
        pool = [f for f in candidates if "_raw_" in f.name]
        args.phase_labeled = max(pool or candidates, key=lambda p: p.stat().st_mtime)

    if args.eligible is None:
        el = [
            f for f in sorted(interim.glob("eligible_users_timeline_minposts20_span100_minphase3_*.csv"))
            if not any(t in f.name for t in ("_no_anchors_", "_bc_"))
        ]
        if el:
            pl_mtime = args.phase_labeled.stat().st_mtime
            after = [f for f in el if f.stat().st_mtime >= pl_mtime]
            args.eligible = min(after, key=lambda f: f.stat().st_mtime) if after else el[-1]
        else:
            args.eligible = find_latest_file(interim, "eligible_users_timeline_minposts20_span100_minphase3_*.csv")

    if args.phase_labeled is None or args.eligible is None:
        parser.error("Could not auto-detect input files.")
    return args


def main() -> None:
    args = _parse_args()

    log.info("Eligible users: %s", args.eligible.name)
    eligible_authors = set(pd.read_csv(args.eligible, usecols=["author"])["author"].unique())
    log.info("  %d authors", len(eligible_authors))

    log.info("Phase-labeled: %s", args.phase_labeled.name)
    df = pd.read_csv(args.phase_labeled)
    df = df[df["author"].isin(eligible_authors) & df["phase"].isin(PHASE_ORDER)].copy()

    if args.window:
        df = df[df["offset_from_cd1"].between(-args.window, args.window)].copy()
        log.info("±%d day window → %d rows", args.window, len(df))

    features, suffix = _resolve_features(list(df.columns))
    if not features:
        log.error("No feature columns found.")
        sys.exit(1)
    log.info("Features (%s): %s", suffix, [_clean_label(f) for f in features])

    # One mean per user per phase (all available days — no downsampling)
    valid_users = [
        author for author, grp in df.groupby("author")
        if all(p in grp["phase"].values for p in PHASE_ORDER)
    ]
    df = df[df["author"].isin(valid_users)]
    phase_means = df.groupby(["author", "phase"])[features].mean().reset_index()
    log.info("Per-user phase means: %d rows, %d users",
             len(phase_means), phase_means["author"].nunique())

    # Equal-weighted centering: subtract mean of the 4 phase means per user
    grand = phase_means.groupby("author")[features].transform("mean")
    phase_means = phase_means.copy()
    phase_means[features] = phase_means[features] - grand
    log.info("Applied equal-weighted within-user centering.")

    # Pivot to wide: one row per user, one column per phase per feature
    wide = phase_means.pivot(index="author", columns="phase", values=features)
    # wide.columns = MultiIndex (feature, phase)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-pair difference DataFrames and plot
    for phase_a, phase_b in PAIRS:
        diff_cols = {}
        for feat in features:
            diff_cols[feat] = wide[feat][phase_a] - wide[feat][phase_b]
        diff_df = pd.DataFrame(diff_cols)
        log.info("Plotting diff: %s − %s  (n=%d)", phase_a, phase_b, len(diff_df.dropna()))
        _plot_pair_differences(diff_df, phase_a, phase_b, features, args.output_dir)

    # Stats table — rebuild per-pair diff series into the format _print_summary expects
    # Re-use wide but store diffs in a nested dict keyed by pair label and feature
    import collections
    wide_for_stats = collections.defaultdict(dict)
    for feat in features:
        for phase_a, phase_b in PAIRS:
            key = f"{phase_a}__{phase_b}"
            wide_for_stats[feat][key] = (wide[feat][phase_a] - wide[feat][phase_b]).dropna().values

    class _FakeWide:
        """Minimal shim so _print_summary can index wide[feat][pair_key]."""
        def __init__(self, d): self._d = d
        def __getitem__(self, feat):
            class _Col:
                def __init__(self, d): self._d = d
                def __getitem__(self, key): return self._d[key]
            return _Col(self._d[feat])

    _print_summary(_FakeWide(wide_for_stats), features)


if __name__ == "__main__":
    main()
