"""Method 2: GAM smooth curve of linguistic features over the cycle.

Fits a smooth curve of each feature as a function of cycle day using a
Generalized Additive Model:

  feature_value ~ s(cycle_day_norm, n_splines=N)

Features are within-user mean-centered before fitting, which removes
inter-user baseline differences and lets the smooth capture only the
within-user cyclic variation.

The x-axis is the raw offset_from_cd1, so the previous cycle (negative
offsets) and current cycle (positive offsets) are visible side by side,
with CD1 at x=0.

Eligible users come from the --eligible file (produced by
08a_select_eligible_users.py); this script does not recompute eligibility.

Outputs:
  <output-dir>/<feature>.png    -- fitted curve + 95% CI band (one per feature)
  <output-dir>/gam_summary.csv  -- features fitted and their observation counts

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.io import find_latest_file
from src.utils import compute_family_pca

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _phase_spans_two_cycles(period: int, m: int, f: int, o: int) -> list[tuple[str, int, int]]:
    """Return (phase, start_offset, end_offset) for two cycles around CD1.

    Previous cycle occupies [-period, 0); current cycle occupies [0, period).
    """
    luteal = period - m - f - o
    layout = [
        ("Menstrual",  0,         m),
        ("Follicular", m,         f),
        ("Ovulation",  m + f,     o),
        ("Luteal",     m + f + o, luteal),
    ]
    spans = []
    for shift in (-period, 0):
        for phase, start, length in layout:
            spans.append((phase, start + shift, start + shift + length))
    return spans


def _resolve_features(columns: list[str]) -> tuple[list[str], str]:
    non_feature = {"author", "offset_from_cd1", "phase"}
    for suffix in ("_mean", "_zscore"):
        selected = sorted(c for c in columns if c.endswith(suffix) and c not in non_feature)
        if selected:
            return selected, suffix
    return [], ""


def _clean(col: str) -> str:
    if col.startswith("family_"):
        return col[len("family_"):].replace("_", " ").title() + " PC1"
    return col.replace("_zscore", "").replace("_mean", "").replace("_", " ")


def fit_gam(X: np.ndarray, y: np.ndarray, n_splines: int = 10) -> LinearGAM | None:
    """Fit LinearGAM with a smooth on cycle_day_norm."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam = LinearGAM(s(0, n_splines=n_splines, spline_order=3)).fit(X, y)
        return gam
    except Exception:
        return None


def plot_curve(gam: LinearGAM, feature: str, output_dir: Path,
               period: int = 28, m: int = 4, f: int = 8, o: int = 3) -> None:
    XX = gam.generate_X_grid(term=0, n=300)
    preds = gam.predict(XX)
    ci = gam.confidence_intervals(XX, width=0.95)
    lo, hi = ci[:, 0], ci[:, 1]

    fig, ax = plt.subplots(figsize=(12, 4))

    labeled = set()
    for phase, start, end in _phase_spans_two_cycles(period, m, f, o):
        label = phase if phase not in labeled else "_nolegend_"
        ax.axvspan(start, end, alpha=0.10, color=PHASE_COLORS[phase], label=label)
        labeled.add(phase)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.4, label="CD1 (anchor)")

    ax.plot(XX[:, 0], preds, color="black", linewidth=2, label="GAM fit")
    ax.fill_between(XX[:, 0], lo, hi, alpha=0.25, color="steelblue", label="95% CI")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

    ax.set_xlabel("Days from CD1 (anchor post)", fontsize=10)
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
    parser.add_argument("--eligible", type=Path, default=None,
                        help="Eligible-users file (from 08a). Provides the author set.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "gam_cycle_curves")
    parser.add_argument("--window", type=int, default=28,
                        help="Restrict data to offset_from_cd1 in [-window, window] (default: 28)")
    parser.add_argument("--period", type=int, default=28,
                        help="Cycle length used for phase-band annotation (default: 28)")
    parser.add_argument("--n-splines", type=int, default=10,
                        help="Number of spline basis functions (default: 10)")
    parser.add_argument("--exclude-anchor", action="store_true",
                        help="Exclude posts at offset_from_cd1 == 0 (cycle day 1 posts)")
    parser.add_argument("--menstrual-days", type=int, default=4)
    parser.add_argument("--follicular-days", type=int, default=8)
    parser.add_argument("--ovulation-days", type=int, default=3)
    parser.add_argument("--family-pca", type=Path, default=None,
                        help="Path to Attribute_families_cycles.xlsx; reduces 69 features to "
                             "one PC1 per family before GAM fitting")
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

    # Inputs are logged explicitly: the auto-discovery picks the latest matching
    # files, and silently landing on the wrong dataset is easy to miss otherwise.
    log.info("Phase-labeled: %s", args.phase_labeled.name)
    log.info("Eligible:      %s", args.eligible.name)

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

    df["cycle_day_norm"] = df["offset_from_cd1"].astype(float)

    if args.family_pca:
        log.info("Computing family PCA composites from %s ...", args.family_pca.name)
        # per_user_normalize=True so PC1 aligns to *within-user* cyclic structure
        # (the composites are per-user-centered below, so otherwise PC1 captures
        # between-user baselines and the wave washes out). sign_fix=True gives a
        # deterministic curve orientation. Both differ from the MATLAB-faithful
        # default (False/False) used by the spectral/NUFFT scripts.
        df, features = compute_family_pca(df, args.family_pca,
                                          per_user_normalize=True, sign_fix=True)
    else:
        features, _ = _resolve_features(list(df.columns))
    log.info("Fitting GAMs for %d features...", len(features))

    user_means = df.groupby("author")[features].transform("mean")
    df[features] = df[features] - user_means

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, feat in enumerate(features):
        if (i + 1) % 10 == 0:
            log.info("  %d / %d", i + 1, len(features))

        sub = df[["cycle_day_norm", feat]].dropna()
        if len(sub) < 50 or sub[feat].std() < 1e-10:
            continue

        gam = fit_gam(sub[["cycle_day_norm"]].values, sub[feat].values, n_splines=args.n_splines)
        if gam is None:
            continue

        plot_curve(gam, feat, args.output_dir, period=args.period,
                   m=args.menstrual_days, f=args.follicular_days, o=args.ovulation_days)
        records.append({"feature": feat, "label": _clean(feat), "n_obs": len(sub)})

    pd.DataFrame(records).to_csv(args.output_dir / "gam_summary.csv", index=False)
    log.info("Plotted %d feature curves.", len(records))


if __name__ == "__main__":
    main()
