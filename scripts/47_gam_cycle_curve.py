"""Method 2: GAM smooth curve of linguistic features over the cycle.

Instead of the 4-box phase model, fits a smooth curve of each feature
as a function of cycle-normalized day using a Generalized Additive Model.

  feature_value ~ s(cycle_day_norm, n_splines=N)

  Features are within-user mean-centered before fitting, which removes
  inter-user baseline differences and lets the smooth capture only the
  within-user cyclic variation.

cycle_day_norm = (offset_from_cd1 % period) / period x 28
  -- maps every post to a position on a standardized 28-day scale using
     a single fixed period applied to all users.

With --two-cycles, the raw offset_from_cd1 is used as the x-axis so
that the previous cycle (negative offsets) and current cycle (positive
offsets) are visible side by side, with CD1 at x=0.

Outputs (per feature with significant smooth):
  <output-dir>/<feature>.png     -- fitted curve + 95% CI band
  <output-dir>/gam_summary.csv  -- EDF, p-values and BH-FDR q-values

Usage:
  python scripts/47_gam_cycle_curve.py \\
    --phase-labeled data/interim/timeline_phase_labeled_fixed28_raw_*.csv \\
    --eligible      data/interim/eligible_users_timeline_minposts20_span100_minphase3_*.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.io import find_latest_file

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Phase boundaries on standardized 28-day scale for single-cycle annotation.
# Each tuple is (axvspan_start, axvspan_end) on the continuous x-axis.
# Gaps between phases (e.g. day 4, day 13, day 16) are intentional.
PHASE_SPANS_28 = {
    "Menstrual":  (0,  4),
    "Follicular": (5,  13),
    "Ovulation":  (14, 16),
    "Luteal":     (17, 28),
}


def _phase_spans_two_cycles(period: int, m: int, f: int, o: int) -> list[tuple[str, int, int]]:
    """Return (phase, start_offset, end_offset) for two cycles around CD1.

    Previous cycle occupies [-period, 0); current cycle occupies [0, period).
    """
    luteal = period - m - f - o
    spans = []
    for phase, start, length in [
        ("Menstrual",  0,         m),
        ("Follicular", m,         f),
        ("Ovulation",  m + f,     o),
        ("Luteal",     m + f + o, luteal),
    ]:
        spans.append((phase, start - period, start - period + length))
    for phase, start, length in [
        ("Menstrual",  0,         m),
        ("Follicular", m,         f),
        ("Ovulation",  m + f,     o),
        ("Luteal",     m + f + o, luteal),
    ]:
        spans.append((phase, start, start + length))
    return spans


def _resolve_features(columns: list[str]) -> tuple[list[str], str]:
    non_feature = {"author", "offset_from_cd1", "phase"}
    for suffix in ("_mean", "_zscore"):
        selected = sorted(c for c in columns if c.endswith(suffix) and c not in non_feature)
        if selected:
            return selected, suffix
    return [], ""


def _clean(col: str) -> str:
    return col.replace("_zscore", "").replace("_mean", "").replace("_", " ")


def fit_gam(X: np.ndarray, y: np.ndarray, n_splines: int = 10) -> LinearGAM | None:
    """Fit LinearGAM with a smooth on cycle_day_norm (0-28)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam = LinearGAM(s(0, n_splines=n_splines, spline_order=3)).fit(X, y)
        return gam
    except Exception:
        return None


def plot_curve(gam: LinearGAM, feature: str, output_dir: Path,
               two_cycles: bool = False, period: int = 28,
               m: int = 4, f: int = 8, o: int = 3) -> None:
    XX = gam.generate_X_grid(term=0, n=300)
    preds = gam.predict(XX)
    ci = gam.confidence_intervals(XX, width=0.95)
    lo, hi = ci[:, 0], ci[:, 1]

    fig, ax = plt.subplots(figsize=(12 if two_cycles else 9, 4))

    if two_cycles:
        spans = _phase_spans_two_cycles(period, m, f, o)
        labeled = set()
        for phase, start, end in spans:
            label = phase if phase not in labeled else "_nolegend_"
            ax.axvspan(start, end, alpha=0.10, color=PHASE_COLORS[phase], label=label)
            labeled.add(phase)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.4,
                   label="CD1 (anchor)")
        xlabel = "Days from CD1 (anchor post)"
    else:
        for phase, (start, end) in PHASE_SPANS_28.items():
            ax.axvspan(start, end, alpha=0.08, color=PHASE_COLORS[phase], label=phase)
        xlabel = "Cycle day (standardized to 28-day scale)"

    ax.plot(XX[:, 0], preds, color="black", linewidth=2, label="GAM fit")
    ax.fill_between(XX[:, 0], lo, hi, alpha=0.25, color="steelblue", label="95% CI")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Feature value (deviation from user mean)", fontsize=10)
    ax.set_title(f"GAM cycle curve: {_clean(feature)}", fontsize=11)

    handles = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.3, label=p) for p in PHASE_ORDER]
    handles += [plt.Line2D([0], [0], color="black", lw=2, label="GAM fit"),
                plt.Line2D([0], [0], color="steelblue", lw=6, alpha=0.3, label="95% CI")]
    ax.legend(handles=handles, fontsize=8, loc="best")

    fig.tight_layout()
    fname = _clean(feature).replace(" ", "_") + ".png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    interim = ROOT / "data" / "interim"
    parser.add_argument("--phase-labeled", type=Path, default=None)
    parser.add_argument("--eligible", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "gam_cycle_curves")
    parser.add_argument("--window", type=int, default=28,
                        help="Restrict data to offset_from_cd1 in [-window, window] (default: 28)")
    parser.add_argument("--eligibility-window", type=int, default=None,
                        help="Wider window for eligibility check. Default: same as --window")
    parser.add_argument("--min-posts", type=int, default=None,
                        help="Min posts in eligibility window (overrides --eligible file)")
    parser.add_argument("--min-span", type=int, default=None,
                        help="Min span in days in eligibility window")
    parser.add_argument("--max-gap", type=int, default=None,
                        help="Max allowed gap between consecutive posts in connected chain from anchor")
    parser.add_argument("--period", type=int, default=28,
                        help="Cycle length used for normalization (default: 28)")
    parser.add_argument("--n-splines", type=int, default=10,
                        help="Number of spline basis functions (default: 10)")
    parser.add_argument("--plot-all", action="store_true",
                        help="Plot curves for all features, not just significant ones")
    parser.add_argument("--two-cycles", action="store_true",
                        help="Use raw offset_from_cd1 as x-axis (two cycles visible)")
    parser.add_argument("--exclude-anchor", action="store_true",
                        help="Exclude posts at offset_from_cd1 == 0 (cycle day 1 posts)")
    parser.add_argument("--menstrual-days", type=int, default=4)
    parser.add_argument("--follicular-days", type=int, default=8)
    parser.add_argument("--ovulation-days", type=int, default=3)
    args = parser.parse_args()

    if args.phase_labeled is None:
        candidates = sorted(interim.glob("timeline_phase_labeled_fixed28_*.csv"))
        pool = [p for p in candidates if "_raw_" in p.name]
        args.phase_labeled = max(pool or candidates, key=lambda p: p.stat().st_mtime)
    if args.eligible is None:
        el = [p for p in sorted(interim.glob(
              "eligible_users_timeline_minposts20_span100_minphase3_*.csv"))
              if not any(t in p.name for t in ("_no_anchors_", "_bc_"))]
        if el:
            after = [p for p in el if p.stat().st_mtime >= args.phase_labeled.stat().st_mtime]
            args.eligible = min(after, key=lambda p: p.stat().st_mtime) if after else el[-1]
        else:
            args.eligible = find_latest_file(
                interim, "eligible_users_timeline_minposts20_span100_minphase3_*.csv")
    return args


def main() -> None:
    args = _parse_args()

    if args.min_posts is not None or args.min_span is not None or args.max_gap is not None:
        el_window = args.eligibility_window or args.window
        all_df = pd.read_csv(args.phase_labeled, usecols=["author", "offset_from_cd1"])
        within = all_df[all_df["offset_from_cd1"].between(-el_window, el_window)]
        eligible_authors = set()
        for author, grp in within.groupby("author"):
            days = sorted(grp["offset_from_cd1"].unique())
            n    = len(days)
            span = days[-1] - days[0] if days else 0

            if args.max_gap is not None:
                before_days = [d for d in days if d <= 0]
                after_days  = [d for d in days if d >= 0]
                start_day = before_days[-1] if before_days else (after_days[0] if after_days else None)
                if start_day is None:
                    continue
                si = days.index(start_day)
                right = start_day
                for i in range(si + 1, len(days)):
                    if days[i] - days[i - 1] <= args.max_gap:
                        right = days[i]
                    else:
                        break
                left = start_day
                for i in range(si - 1, -1, -1):
                    if days[i + 1] - days[i] <= args.max_gap:
                        left = days[i]
                    else:
                        break
                span = right - left

            crosses_anchor = left < 0 and right > 0
            if args.two_cycles and not crosses_anchor:
                continue
            if (args.min_posts is None or n >= args.min_posts) and \
               (args.min_span  is None or span >= args.min_span):
                eligible_authors.add(author)

        log.info("Eligible authors (min_posts=%s, min_span=%s, max_gap=%s, el_window=+/-%d): %d",
                 args.min_posts, args.min_span, args.max_gap, el_window, len(eligible_authors))
    else:
        eligible_authors = set(pd.read_csv(args.eligible, usecols=["author"])["author"].unique())
        log.info("Eligible authors: %d", len(eligible_authors))

    df = pd.read_csv(args.phase_labeled)
    df = df[df["author"].isin(eligible_authors)].copy()

    if args.window:
        df = df[df["offset_from_cd1"].between(-args.window, args.window)].copy()

    if args.exclude_anchor:
        before = len(df)
        df = df[df["offset_from_cd1"] != 0].copy()
        log.info("Excluded cycle-day-1 posts (offset=0): %d -> %d rows", before, len(df))

    log.info("Rows after window filter: %d", len(df))

    if args.two_cycles:
        df["cycle_day_norm"] = df["offset_from_cd1"].astype(float)
    else:
        df["cycle_day_norm"] = (df["offset_from_cd1"] % args.period) / args.period * 28
        df = df[df["cycle_day_norm"].between(0, 28)].copy()

    features, suffix = _resolve_features(list(df.columns))
    log.info("Fitting GAMs for %d features...", len(features))

    user_means = df.groupby("author")[features].transform("mean")
    df[features] = df[features] - user_means

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, feat in enumerate(features):
        if (i + 1) % 10 == 0:
            log.info("  %d / %d", i + 1, len(features))

        sub = df[["cycle_day_norm", feat]].dropna()
        if len(sub) < 50 or sub[feat].std() < 1e-10:
            summary_rows.append({"feature": feat, "edf": np.nan, "p_value": np.nan})
            continue

        X = sub[["cycle_day_norm"]].values
        y = sub[feat].values

        gam = fit_gam(X, y, n_splines=args.n_splines)
        if gam is None:
            summary_rows.append({"feature": feat, "edf": np.nan, "p_value": np.nan})
            continue

        stats = gam.statistics_
        p_val = float(stats["p_values"][0]) if "p_values" in stats else np.nan
        edf   = float(stats["edof_per_coef"].sum()) if "edof_per_coef" in stats else np.nan
        summary_rows.append({"feature": feat, "edf": edf, "p_value": p_val, "n_obs": len(sub)})

        if args.plot_all:
            plot_curve(gam, feat, args.output_dir,
                       two_cycles=args.two_cycles, period=args.period,
                       m=args.menstrual_days, f=args.follicular_days,
                       o=args.ovulation_days)

    summary = pd.DataFrame(summary_rows)

    mask = summary["p_value"].notna()
    if mask.sum() > 0:
        _, qvals, _, _ = multipletests(summary.loc[mask, "p_value"], method="fdr_bh")
        summary.loc[mask, "q_value"] = qvals

    summary["label"] = summary["feature"].map(_clean)
    summary = summary.sort_values("p_value")

    out_csv = args.output_dir / "gam_summary.csv"
    summary.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    if "q_value" in summary.columns:
        sig_features = summary[summary["q_value"] < 0.05]["feature"].tolist()
    else:
        sig_features = []
    if not sig_features and not args.plot_all:
        sig_features = summary.dropna(subset=["p_value"]).head(15)["feature"].tolist()
        log.info("No FDR-significant features; plotting top 15 by p-value.")

    for feat in sig_features:
        sub = df[["cycle_day_norm", feat]].dropna()
        X = sub[["cycle_day_norm"]].values
        y = sub[feat].values
        gam = fit_gam(X, y, n_splines=args.n_splines)
        if gam:
            plot_curve(gam, feat, args.output_dir,
                       two_cycles=args.two_cycles, period=args.period,
                       m=args.menstrual_days, f=args.follicular_days,
                       o=args.ovulation_days)

    log.info("Plotted %d feature curves.", len(sig_features))

    print("\n=== GAM results (top 20 by p-value) ===")
    print(f"{'Feature':<45} {'EDF':>6} {'p_value':>10} {'q_value':>10}")
    print("-" * 75)
    for _, row in summary.dropna(subset=["p_value"]).head(20).iterrows():
        q = row.get("q_value", np.nan)
        sig = " ***" if q < 0.001 else (" **" if q < 0.01 else (" *" if q < 0.05 else ""))
        print(f"{_clean(row['feature']):<45} {row['edf']:>6.2f} "
              f"{row['p_value']:>10.4g} {q:>10.4g}{sig}")


if __name__ == "__main__":
    main()
