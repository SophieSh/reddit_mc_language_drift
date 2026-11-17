"""Combined GAM family-panel figure for the paper.

Fits the same within-user family-PCA GAM as 47_gam_cycle_curve.py, but instead of
one PNG per family it lays ALL FIVE family curves into a single figure (one panel
per family, shared cycle x-axis). Then it emits a second, identical figure with every
curve multiplied by -1 (PC1 sign is arbitrary; the mirror lets the reader orient each
family to the hypothesised hormonal direction).

No titles on either figure (paper style). Family name sits as the panel y-label.

Canonical input (paper Figure 2; Yngve-corrected features):
  data/interim/eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125_yngvefix_20260906T140542.csv
(711 users, 14,834 user-days). The uncorrected file without "_yngvefix" gives
slightly different curves and does NOT match the paper.

Usage (no arguments reproduces the paper Figure 2 byte-for-byte):
  python scripts/gam/50_gam_family_panel.py
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
from src.utils import compute_family_pca

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CANONICAL = ("data/interim/"
             "eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125"
             "_yngvefix_20260906T140542.csv")

# Panel order requested for the paper.
FAMILY_ORDER = [
    ("family_affective",      "Affective"),
    ("family_lexical_access", "Lexical access"),
    ("family_syntactic",      "Syntactic"),
    ("family_general",        "General"),
    ("family_pragmatic",      "Pragmatic"),
]


def _phase_spans_two_cycles(period, m, f, o):
    luteal = period - m - f - o
    layout = [("Menstrual", 0, m), ("Follicular", m, f),
              ("Ovulation", m + f, o), ("Luteal", m + f + o, luteal)]
    spans = []
    for shift in (-period, 0):
        for phase, start, length in layout:
            spans.append((phase, start + shift, start + shift + length))
    return spans


def fit_gam(x, y, n_splines=10):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LinearGAM(s(0, n_splines=n_splines, spline_order=3)).fit(x, y)


def draw_panel(ax, XX, preds, lo, hi, label, period, m, f, o, caption_y=-0.14):
    for phase, start, end in _phase_spans_two_cycles(period, m, f, o):
        ax.axvspan(start, end, alpha=0.10, color=PHASE_COLORS[phase])
    ax.axvline(0, color="black", linewidth=1.1, linestyle="-", alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.plot(XX, preds, color="black", linewidth=2)
    ax.fill_between(XX, lo, hi, alpha=0.25, color="steelblue")
    # family name as a caption UNDER the panel (not a top title, not a y-label)
    if label:
        ax.text(0.5, caption_y, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=12, fontweight="bold")
    ax.margins(x=0)

    # Scale y to the GAM CURVE's own range (with headroom), not the CI. The CI
    # explodes near the sparse window edges; if we scaled to it the sleeve would
    # fill the panel and hide the wiggle. Instead the curve dominates and the CI
    # shows as a sleeve, clipping only at the extreme tails.
    crange = float(preds.max() - preds.min())
    pad = 0.45 * crange
    ax.set_ylim(preds.min() - pad, preds.max() + pad)


def render(fits, out_path, period, m, f, o, invert=False):
    n = len(fits)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.9 * n), sharex=True)
    if n == 1:
        axes = [axes]
    sgn = -1.0 if invert else 1.0
    for i, (ax, (label, XX, preds, lo, hi)) in enumerate(zip(axes, fits)):
        # mirror flips the curve AND its CI band (lo/hi swap under negation)
        p, l, h = sgn * preds, sgn * hi, sgn * lo
        # bottom panel carries the shared x-tick labels, so drop its caption lower
        cap_y = -0.30 if i == n - 1 else -0.14
        draw_panel(ax, XX, p, l, h, label, period, m, f, o, caption_y=cap_y)

    handles = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.3, label=p) for p in PHASE_ORDER]
    handles += [plt.Line2D([0], [0], color="black", lw=2, label="GAM fit"),
                plt.Line2D([0], [0], color="steelblue", lw=6, alpha=0.3, label="95% CI")]
    # legend in its own row ABOVE the top panel so it never overlaps a curve
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.905), ncol=6, framealpha=0.9)

    # single shared x-axis label at the very bottom, clear of the panel captions
    fig.supxlabel("Days from CD1 (anchor post)", fontsize=11)
    fig.subplots_adjust(hspace=0.45, top=0.9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def render_single(fit, out_path, period, m, f, o, invert=False):
    """One family on its own axes (original or sign-inverted), same styling as the panels."""
    label, XX, preds, lo, hi = fit
    sgn = -1.0 if invert else 1.0
    p, l, h = sgn * preds, sgn * hi, sgn * lo
    fig, ax = plt.subplots(figsize=(9, 3.6))
    draw_panel(ax, XX, p, l, h, "", period, m, f, o)   # no inline caption on single plots
    ax.set_xlabel("Days from CD1 (anchor post)", fontsize=11)

    handles = [mpatches.Patch(color=PHASE_COLORS[ph], alpha=0.3, label=ph) for ph in PHASE_ORDER]
    handles += [plt.Line2D([0], [0], color="black", lw=2, label="GAM fit"),
                plt.Line2D([0], [0], color="steelblue", lw=6, alpha=0.3, label="95% CI")]
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.9), ncol=6, framealpha=0.9)
    # family name as a bold caption BELOW the x-axis label; mark the mirrored version
    caption = f"{label} (inverted)" if invert else label
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.24)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / CANONICAL)
    ap.add_argument("--family-pca", type=Path, default=ROOT / "data/raw/Attribute_families_cycles.xlsx")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "gam_cycle_curves_yngvefix")
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--period", type=int, default=28)
    ap.add_argument("--n-splines", type=int, default=10)
    ap.add_argument("--menstrual-days", type=int, default=4)
    ap.add_argument("--follicular-days", type=int, default=8)
    ap.add_argument("--ovulation-days", type=int, default=3)
    args = ap.parse_args()

    log.info("Input: %s", args.input.name)
    df = pd.read_csv(args.input)
    if args.window:
        df = df[df["offset_from_cd1"].between(-args.window, args.window)].copy()
    log.info("Rows after window +/-%d: %d (%d users)", args.window, len(df), df["author"].nunique())

    df, family_cols = compute_family_pca(df, args.family_pca,
                                         per_user_normalize=True, sign_fix=True)
    df["cycle_day_norm"] = df["offset_from_cd1"].astype(float)
    # within-user mean-center so the smooth captures only within-user cyclic variation
    df[family_cols] = df[family_cols] - df.groupby("author")[family_cols].transform("mean")

    fits = []
    for col, label in FAMILY_ORDER:
        if col not in df.columns:
            log.warning("family column %s missing; skipping", col)
            continue
        sub = df[["cycle_day_norm", col]].dropna()
        gam = fit_gam(sub[["cycle_day_norm"]].values, sub[col].values, args.n_splines)
        XX = gam.generate_X_grid(term=0, n=300)
        preds = gam.predict(XX)
        ci = gam.confidence_intervals(XX, width=0.95)
        fits.append((label, XX[:, 0], preds, ci[:, 0], ci[:, 1]))
        log.info("  fit %s (n=%d)", label, len(sub))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    m, f, o = args.menstrual_days, args.follicular_days, args.ovulation_days
    render(fits, args.output_dir / "gam_families.png", args.period, m, f, o, invert=False)
    render(fits, args.output_dir / "gam_families_inverted.png", args.period, m, f, o, invert=True)

    # per-family single plots: one original + one inverted each
    per_family_dir = args.output_dir / "per_family"
    per_family_dir.mkdir(parents=True, exist_ok=True)
    for fit in fits:
        slug = fit[0].lower().replace(" ", "_")
        render_single(fit, per_family_dir / f"gam_{slug}.png", args.period, m, f, o, invert=False)
        render_single(fit, per_family_dir / f"gam_{slug}_inverted.png", args.period, m, f, o, invert=True)


if __name__ == "__main__":
    main()
