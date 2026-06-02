"""
Step 42 — EBM Binary BC Classifier
=====================================
Binary classifier: BC users (label=1) vs non-BC users (label=0).

Input
  A CSV with columns:
    author      — user ID
    phase       — Menstrual | Follicular | Ovulation | Luteal
    <label-col> — binary label per user (1 = BC, 0 = non-BC)
    *_zscore    — per-user z-scored linguistic features

Design
  • Aggregates day-level rows to one mean profile per (author, phase); columns
    become <feature>_<Phase> (e.g. negative_sentiment_zscore_Menstrual).
    Users with no posts in a phase get NaN for those columns — EBM handles NaN.
  • One row per user after pivot; BC label is constant per user.
  • StratifiedGroupKFold (5 folds, grouped by author) — no user leakage
  • sample_weight="balanced" — corrects class imbalance (~1500 non-BC vs ~400 BC)
  • Pure additive EBM (interactions=0)

Outputs (reports/ebm/bc/)
  ebm_bc_importance_<ts>.png    — signed importance bar chart
  ebm_bc_shapes_<ts>.png        — shape grid
  ebm_bc_confusion_<ts>.png     — OOF confusion matrix
  ebm_bc_oof_report_<ts>.txt    — OOF metrics

Usage
  python scripts/42_ebm_bc_classifier.py --input-file data/interim/bc_labeled.csv
  python scripts/42_ebm_bc_classifier.py --input-file data/interim/bc_labeled.csv --label-col bc
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ebm_utils import (
    clean_name,
    make_ebm,
    run_cv,
    shape_functions,
    signed_importance,
)

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = ROOT / "reports" / "ebm" / "bc"

CLASS_NAMES = ["non-BC", "BC"]


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_importance(
    signed_imp: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Signed horizontal bar chart.  Positive (right) = toward BC (class 1)."""
    sorted_idx = np.argsort(np.abs(signed_imp))
    top_idx    = sorted_idx[-top_n:]

    vals   = [signed_imp[i] for i in top_idx]
    labels = [clean_name(feature_names[i]) for i in top_idx]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]   # red=BC, blue=non-BC

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(top_idx)), vals, color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Density-weighted mean log-odds (signed)")
    ax.set_title(
        f"EBM Feature Importances — non-BC vs BC (top {top_n})\n"
        "Positive (right) → BC  |  Negative (left) → non-BC",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(handles=[
        mpatches.Patch(color="#d62728", label="BC"),
        mpatches.Patch(color="#1f77b4", label="non-BC"),
    ], fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance plot → {output_path.name}")


def plot_shape_grid(
    shapes: dict[str, dict],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Binary shape grid: above 0 pushes toward BC, below 0 toward non-BC."""
    feats  = list(shapes.keys())[:top_n]
    n_feat = len(feats)
    ncols  = 2
    nrows  = (n_feat + ncols - 1) // ncols
    color_bc     = "#d62728"   # red
    color_non_bc = "#1f77b4"   # blue

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(feats):
        ax    = axes[i]
        d     = shapes[feat]["BC"]
        x, y  = d["x"], d["y"]
        valid = ~np.isnan(x)
        xv, yv = x[valid], y[valid]

        for j in range(len(xv) - 1):
            color = color_bc if yv[j] >= 0 else color_non_bc
            ax.axvspan(xv[j], xv[j + 1], alpha=0.12, color=color, linewidth=0)

        ax.step(xv, yv, where="mid", color="black", linewidth=1.8)
        ax.fill_between(xv, yv, 0, step="mid", where=yv >= 0,
                        alpha=0.35, color=color_bc, label="BC")
        ax.fill_between(xv, yv, 0, step="mid", where=yv < 0,
                        alpha=0.35, color=color_non_bc, label="non-BC")

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.5, linestyle=":")
        ax.set_title(f"{clean_name(feat)}\nstrength={d['weighted_sum']:.3f}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("mean z-score in phase vs user baseline", fontsize=8)
        ax.set_ylabel("Log-odds", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "EBM Shape Functions — non-BC vs BC\n"
        "Above 0 → BC  |  Below 0 → non-BC  |  x=0 → user at personal mean",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape grid → {output_path.name}")


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Normalised OOF confusion matrix."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of true class")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_norm[i, j] > 0.6 else "black")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("EBM BC Classifier — OOF Confusion Matrix\n(count  /  row fraction)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Confusion matrix → {output_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regular-users-file", required=True,
                   help="CSV with author, features")
    p.add_argument("--bc-users-file", required=True,
                   help="CSV with author, features")
    p.add_argument("--top-n",   type=int, default=15)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip-cv", action="store_true")
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── [1] Load & build user-level feature matrix ────────────────────────────
    logging.info("[1] Loading data…")
    df = pd.read_csv(args.input_file, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} rows, {df['author'].nunique():,} users")

    for col in ("phase", args.label_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    if not zscore_cols:
        raise ValueError("No *_zscore columns found. Check input file.")

    # Mean z-score per (author, phase) — one profile per user per phase.
    profiles = df.groupby(["author", "phase"])[zscore_cols].mean().reset_index()

    # Pivot to wide: columns become <feature>_<Phase> (e.g. sent_zscore_Menstrual).
    # Users missing a phase get NaN for those columns; EBM handles NaN natively.
    profiles_wide = profiles.pivot(index="author", columns="phase", values=zscore_cols)
    profiles_wide.columns = [f"{feat}_{phase}" for feat, phase in profiles_wide.columns]
    profiles_wide = profiles_wide.reset_index()

    # BC label is constant per user — take first occurrence.
    bc_labels = (
        df.groupby("author")[args.label_col]
        .first()
        .round()
        .astype(int)
        .reset_index()
    )

    df_users = profiles_wide.merge(bc_labels, on="author", how="left")

    feature_names = [c for c in df_users.columns if c not in ("author", args.label_col)]
    X      = df_users[feature_names].values.astype(np.float64)
    y      = df_users[args.label_col].values
    groups = df_users["author"].values

    n_bc, n_non = (y == 1).sum(), (y == 0).sum()
    logging.info(
        f"  users={len(df_users):,}  BC={n_bc:,}  non-BC={n_non:,}  "
        f"features={len(feature_names)} ({len(zscore_cols)} base × up to 4 phases)"
    )

    # ── [2] Cross-validation ──────────────────────────────────────────────────
    y_pred = None
    if not args.skip_cv:
        logging.info(f"\n[2] GroupKFold CV ({args.n_folds} folds)…")
        y_pred, y_proba = run_cv(X, y, groups, n_classes=2,
                                 seed=seed, n_splits=args.n_folds)

        auc     = roc_auc_score(y, y_proba[:, 1])
        bal_acc = balanced_accuracy_score(y, y_pred)
        acc     = accuracy_score(y, y_pred)
        logging.info(f"  OOF AUC={auc:.4f}  Balanced-Acc={bal_acc:.4f}  Acc={acc:.4f}")

        oof_text = (
            f"EBM BC Classifier — OOF Report ({ts})\n"
            f"StratifiedGroupKFold  n_folds={args.n_folds}  "
            f"interactions=0  max_bins=256  sample_weight=balanced\n"
            f"{'='*60}\n"
            f"AUC (ROC)    : {auc:.4f}\n"
            f"balanced_acc : {bal_acc:.4f}\n"
            f"accuracy     : {acc:.4f}\n"
            f"N(BC)        : {n_bc}\n"
            f"N(non-BC)    : {n_non}\n"
        )
        (OUT_DIR / f"ebm_bc_oof_report_{ts}.txt").write_text(oof_text)
        logging.info("  OOF report saved.")
    else:
        logging.info("\n[2] CV skipped (--skip-cv).")

    # ── [3] Final fit ─────────────────────────────────────────────────────────
    logging.info("\n[3] Final fit on full data…")
    ebm = make_ebm(seed)
    ebm.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    logging.info("  Fitted.")

    # ── [4] Signed importance ─────────────────────────────────────────────────
    logging.info("\n[4] Computing signed importance…")
    sim = signed_importance(ebm, feature_names)

    top_idx = np.argsort(np.abs(sim))[-args.top_n:][::-1]
    print(f"\nTop {args.top_n} features — non-BC vs BC:")
    for rank, fi in enumerate(top_idx, 1):
        direction = "→ BC" if sim[fi] >= 0 else "→ non-BC"
        print(f"  {rank:2d}. {feature_names[fi]:<60s}  {sim[fi]:>+.4f}  {direction}")

    plot_importance(sim, feature_names,
                    OUT_DIR / f"ebm_bc_importance_{ts}.png",
                    top_n=args.top_n)

    # ── [5] Shape grid ────────────────────────────────────────────────────────
    logging.info("\n[5] Shape grid…")
    top_feat_names = [feature_names[fi] for fi in top_idx]
    shapes_all = shape_functions(ebm, feature_names, CLASS_NAMES)
    shapes = {feat: shapes_all[feat] for feat in top_feat_names}

    plot_shape_grid(shapes,
                    OUT_DIR / f"ebm_bc_shapes_{ts}.png",
                    top_n=args.top_n)

    # ── [6] Confusion matrix ──────────────────────────────────────────────────
    if y_pred is not None:
        logging.info("\n[6] Confusion matrix…")
        plot_confusion(y, y_pred, CLASS_NAMES,
                       OUT_DIR / f"ebm_bc_confusion_{ts}.png")

    logging.info(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
