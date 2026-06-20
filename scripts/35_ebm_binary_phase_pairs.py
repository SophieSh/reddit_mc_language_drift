"""
Step 35 — Binary EBM Classifiers for Consecutive Phase Pairs
=============================================================
Trains one EBM per pairwise phase combination (all C(4,2)=6 pairs).

Why binary pairs instead of 4-class?
  The 4-class EBM must separate all phases simultaneously.  Binary classifiers
  let the model focus on the specific linguistic shift at each transition,
  which is more interpretable and clinically meaningful.

Design
  • StratifiedGroupKFold (5 folds, grouped by author) — no user leakage
  • sample_weight="balanced" — minority phases not drowned out
  • Pure additive EBM (interactions=0) — every feature has a shape function

Output (reports/ebm/pairs/)
  ebm_pair_<A>_vs_<B>_importance_<ts>.png
  ebm_pair_<A>_vs_<B>_shapes_<ts>.png
  ebm_pairs_oof_summary_<ts>.txt

Usage
  python scripts/35_ebm_binary_phase_pairs.py --input-file data/interim/prepared.csv
  python scripts/35_ebm_binary_phase_pairs.py --input-file data/interim/prepared.csv --top-n 10
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.ebm_utils import (
    clean_name,
    make_ebm,
    run_cv,
    shape_functions,
    signed_importance,
)

warnings.filterwarnings("ignore", category=UserWarning)

PAIRS = [
    ("Menstrual",  "Follicular"),
    ("Menstrual",  "Ovulation"),
    ("Menstrual",  "Luteal"),
    ("Follicular", "Ovulation"),
    ("Follicular", "Luteal"),
    ("Ovulation",  "Luteal"),
]


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_dataset(path: str, equalize_phase_posts: bool = False, seed: int = 42,
                 window_days: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV, aggregate to one mean profile per (author, phase)."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"Loaded {Path(path).name}: {len(df):,} rows, {df['author'].nunique():,} users")

    if window_days is not None:
        before = len(df)
        df = df[df["offset_from_cd1"].between(-window_days, window_days)]
        logging.info(f"  Window ±{window_days} days: {before:,} → {len(df):,} rows, {df['author'].nunique():,} users")

    if any(c.endswith("_zscore") for c in df.columns):
        feat_cols = [c for c in df.columns if c.endswith("_zscore")]
        logging.info(f"  {len(feat_cols)} z-score features")
    elif any(c.endswith("_mean") for c in df.columns):
        feat_cols = [c for c in df.columns if c.endswith("_mean")]
        logging.info(f"  {len(feat_cols)} raw mean features")
    else:
        raise ValueError("No *_zscore or *_mean feature columns found. Check input file.")

    if equalize_phase_posts:
        chunks = []
        ns = []
        for author, grp in df.groupby("author"):
            phase_counts = grp.groupby("phase").size()
            n = int(phase_counts.min())
            ns.append(n)
            for phase, phase_grp in grp.groupby("phase"):
                chunks.append(phase_grp.sample(n=n, random_state=seed, replace=False))
        df = pd.concat(chunks, ignore_index=True)
        import numpy as _np
        logging.info(f"  After equalizing: {len(df):,} rows  (per-user min: mean={_np.mean(ns):.1f} median={_np.median(ns):.0f} min={min(ns)} max={max(ns)})")

    profiles = df.groupby(["author", "phase"])[feat_cols].mean().reset_index()
    logging.info(f"  Profiles: {len(profiles):,}  ({profiles['author'].nunique():,} users × up to 4 phases)")
    return profiles, feat_cols


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_importance(
    signed_imp: np.ndarray,
    feature_names: list[str],
    class_a: str,
    class_b: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Signed horizontal bar chart.  Positive (right) = toward class_b."""
    sorted_idx = np.argsort(np.abs(signed_imp))
    top_idx    = sorted_idx[-top_n:]

    vals   = [signed_imp[i] for i in top_idx]
    labels = [clean_name(feature_names[i]) for i in top_idx]
    color_b = PHASE_COLORS.get(class_b, "steelblue")
    color_a = PHASE_COLORS.get(class_a, "tomato")
    colors  = [color_b if v >= 0 else color_a for v in vals]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(top_idx)), vals, color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Density-weighted mean log-odds (signed)")
    ax.set_title(
        f"EBM Feature Importances — {class_a} vs {class_b} (top {top_n})\n"
        f"Positive (right) → {class_b}  |  Negative (left) → {class_a}",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(handles=[
        mpatches.Patch(color=color_b, label=class_b),
        mpatches.Patch(color=color_a, label=class_a),
    ], fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance plot → {output_path.name}")


def plot_shape_grid(
    shapes: dict[str, dict],
    feature_names: list[str],
    class_a: str,
    class_b: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Grid of binary log-odds shape plots for top_n features."""
    # shapes keys are feature names (from shape_functions); we want top_n by |weighted_sum|
    feats  = list(shapes.keys())[:top_n]
    n_feat = len(feats)
    ncols  = 2
    nrows  = (n_feat + ncols - 1) // ncols
    color_a = PHASE_COLORS.get(class_a, "tomato")
    color_b = PHASE_COLORS.get(class_b, "steelblue")

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(feats):
        ax  = axes[i]
        # Binary shape_functions returns per-class dict; class_b is class 1 (positive)
        d   = shapes[feat][class_b]
        x, y = d["x"], d["y"]
        valid = ~np.isnan(x)
        xv, yv = x[valid], y[valid]

        for j in range(len(xv) - 1):
            color = color_b if yv[j] >= 0 else color_a
            ax.axvspan(xv[j], xv[j + 1], alpha=0.12, color=color, linewidth=0)

        ax.step(xv, yv, where="mid", color="black", linewidth=1.8)
        ax.fill_between(xv, yv, 0, step="mid", where=yv >= 0,
                        alpha=0.35, color=color_b, label=class_b)
        ax.fill_between(xv, yv, 0, step="mid", where=yv < 0,
                        alpha=0.35, color=color_a, label=class_a)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.5, linestyle=":")
        ax.set_title(f"{clean_name(feat)}\nstrength={d['weighted_sum']:.3f}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("z-score vs user mean", fontsize=8)
        ax.set_ylabel("Log-odds", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"EBM Shape Functions — {class_a}  →  {class_b}  (top {n_feat} features)\n"
        f"Above 0 → {class_b}  |  Below 0 → {class_a}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape grid → {output_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True,
                   help="CSV with author, phase, *_zscore columns.")
    p.add_argument("--top-n",   type=int, default=10)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--equalize-phase-posts", action="store_true",
                   help="Sample equal posts per phase per user before computing mean profiles.")
    p.add_argument("--window-days", type=int, default=None,
                   help="Restrict to offset_from_cd1 ∈ [-N, N] before aggregating.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = args.seed
    np.random.seed(seed)

    output_dir = ROOT / "reports" / "ebm" / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load ──────────────────────────────────────────────────────────────
    logging.info("[1] Loading data…")
    df, feature_names = load_dataset(args.input_file,
                                      equalize_phase_posts=args.equalize_phase_posts,
                                      seed=seed,
                                      window_days=args.window_days)
    logging.info(f"  Profiles: {len(df):,}  Users: {df['author'].nunique():,}  Features: {len(feature_names)}")

    oof_lines = [
        f"Binary EBM Phase-Pair OOF Report  ({ts})",
        f"StratifiedGroupKFold n_folds={args.n_folds}  |  sample_weight=balanced",
        "=" * 65,
    ]

    # ── [2] One classifier per pair ───────────────────────────────────────────
    t_global = time.time()
    for phase_a, phase_b in PAIRS:
        pair_label = f"{phase_a}_vs_{phase_b}"
        logging.info(f"\n{'='*60}\n  Pair: {phase_a}  →  {phase_b}\n{'='*60}")

        mask    = df["phase"].isin([phase_a, phase_b])
        df_pair = df[mask].copy()
        df_pair["y"] = (df_pair["phase"] == phase_b).astype(int)

        X      = df_pair[feature_names].values.astype(np.float64)
        y      = df_pair["y"].values
        groups = df_pair["author"].values
        n_a, n_b = (y == 0).sum(), (y == 1).sum()
        logging.info(f"  {phase_a}: {n_a:,}  {phase_b}: {n_b:,}  total: {len(y):,}")

        # ── Cross-validation ─────────────────────────────────────────────────
        if not args.skip_cv:
            logging.info(f"  GroupKFold CV ({args.n_folds} folds)…")
            y_pred, y_proba, _fold_aucs = run_cv(X, y, groups, n_classes=2,
                                               seed=seed, n_splits=args.n_folds)
            auc      = roc_auc_score(y, y_proba[:, 1])
            bal_acc  = balanced_accuracy_score(y, y_pred)
            acc      = accuracy_score(y, y_pred)
            logging.info(f"  OOF AUC={auc:.4f}  Balanced-Acc={bal_acc:.4f}  Acc={acc:.4f}")
            oof_lines += [
                f"\n{phase_a} vs {phase_b}",
                f"  N: {phase_a}={n_a}  {phase_b}={n_b}",
                f"  OOF AUC            : {auc:.4f}",
                f"  OOF Balanced-Acc   : {bal_acc:.4f}",
                f"  OOF Accuracy       : {acc:.4f}",
            ]
        else:
            logging.info("  CV skipped (--skip-cv).")

        # ── Final fit ─────────────────────────────────────────────────────────
        logging.info("  Fitting final EBM…")
        t0 = time.time()
        ebm = make_ebm(seed)
        ebm.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        logging.info(f"  Fitted in {time.time() - t0:.1f}s")

        # ── Signed importance (BUG 3 fix: density-weighted log-odds) ──────────
        sim = signed_importance(ebm, feature_names)

        top_idx = np.argsort(np.abs(sim))[-args.top_n:][::-1]
        print(f"\n  Top {args.top_n} features — {phase_a} vs {phase_b}:")
        for rank, fi in enumerate(top_idx, 1):
            direction = f"→ {phase_b}" if sim[fi] >= 0 else f"→ {phase_a}"
            print(f"    {rank:2d}. {feature_names[fi]:<55s}  {sim[fi]:>+.4f}  {direction}")

        # ── Shapes: only top_n features, ordered by |sim| ────────────────────
        top_feat_names = [feature_names[fi] for fi in top_idx]
        shapes_all = shape_functions(ebm, feature_names, [phase_a, phase_b])
        shapes = {feat: shapes_all[feat] for feat in top_feat_names}

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_importance(
            sim, feature_names, phase_a, phase_b,
            output_dir / f"ebm_pair_{pair_label}_importance_{ts}.png",
            top_n=args.top_n,
        )
        plot_shape_grid(
            shapes, top_feat_names, phase_a, phase_b,
            output_dir / f"ebm_pair_{pair_label}_shapes_{ts}.png",
            top_n=args.top_n,
        )

    # ── OOF summary ───────────────────────────────────────────────────────────
    if not args.skip_cv:
        summary_path = output_dir / f"ebm_pairs_oof_summary_{ts}.txt"
        summary_path.write_text("\n".join(oof_lines))
        logging.info(f"\n  OOF summary → {summary_path.name}")

    logging.info(f"\nAll done. Total time: {time.time() - t_global:.1f}s")
    logging.info(f"Outputs → {output_dir}/")


if __name__ == "__main__":
    main()
