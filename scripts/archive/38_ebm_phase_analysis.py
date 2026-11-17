"""
Step 38 — EBM Phase Analysis
=============================
4-class Explainable Boosting Machine on menstrual cycle phases.

Input
  A single CSV with columns:
    author        — user ID
    phase         — Menstrual | Follicular | Ovulation | Luteal
    *_zscore      — per-user z-scored linguistic features (ready to use)

Design
  • StratifiedGroupKFold (5 folds, grouped by author) — no user leakage
  • sample_weight="balanced" — corrects Luteal/Ovulation class imbalance
  • Pure additive EBM (interactions=0) — every feature has a shape function
  • Aggregates day-level rows to one mean profile per (author, phase)

Outputs (reports/ebm/)
  ebm_importance_heatmap_<ts>.png   — top-N features × 4 phases heatmap
  ebm_shapes_<ts>.png               — overlay shape grid (all phases per axes)
  ebm_confusion_<ts>.png            — OOF normalised confusion matrix
  ebm_oof_report_<ts>.txt           — metrics + classification report

Usage
  python scripts/38_ebm_phase_analysis.py --input-file data/interim/prepared.csv
  python scripts/38_ebm_phase_analysis.py --input-file data/interim/prepared.csv --top-n 10
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.ebm_utils import (
    clean_name,
    global_importance,
    make_ebm,
    run_cv,
    shape_functions,
)

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = ROOT / "reports" / "ebm"


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_dataset(path: str, equalize_phase_posts: bool = False, seed: int = 42,
                 window_days: int | None = None) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """Read CSV, aggregate to one mean profile per (author, phase), return X/y/groups."""
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
        # Per user: sample the same number of posts from every phase (= min across phases)
        # Removes estimation noise caused by unequal phase window lengths (3d Ovulation vs 14d Luteal)
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

    le = LabelEncoder().fit(PHASE_ORDER)
    return profiles, feat_cols, le


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_importance_heatmap(
    df_imp: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Heatmap of mean-absolute log-odds importance: features × phases."""
    top    = df_imp.head(top_n)
    matrix = top[[f"{c}_imp" for c in class_names]].values
    labels = [clean_name(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Mean |log-odds|")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"EBM Feature Importance — top {top_n}\nMean |log-odds| per phase",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance heatmap → {output_path.name}")


def plot_shape_overlay_grid(
    ebm,
    features: list[str],
    feature_names: list[str],
    class_names: list[str],
    colors: dict[str, str],
    output_path: Path,
) -> None:
    """One axes per feature; all phase curves overlaid on the same axes."""
    shapes = shape_functions(ebm, feature_names, class_names)
    n_feat = len(features)
    ncols  = 2
    nrows  = (n_feat + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        ax        = axes[i]
        feat_data = shapes[feat]

        # Background shading: dominant phase at each x interval.
        all_x = next(iter(feat_data.values()))["x"]
        valid = ~np.isnan(all_x)
        x_v   = all_x[valid]
        if len(x_v) > 1:
            y_stack  = np.array([feat_data[c]["y"][valid] for c in class_names])
            dominant = np.argmax(y_stack, axis=0)
            for j in range(len(x_v) - 1):
                ax.axvspan(x_v[j], x_v[j + 1],
                           alpha=0.08, color=colors.get(class_names[dominant[j]], "gray"),
                           linewidth=0)

        for cls, d in feat_data.items():
            color = colors.get(cls, "gray")
            xv, yv = d["x"][valid], d["y"][valid]
            ax.step(xv, yv, where="mid", color=color, linewidth=1.8,
                    label=f"{cls} ({d['weighted_sum']:.2f})")
            ax.fill_between(xv, yv, 0, step="mid", alpha=0.15, color=color)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.5, linestyle=":")
        ax.set_title(clean_name(feat), fontsize=8, fontweight="bold")
        ax.set_xlabel("z-score vs user mean", fontsize=7)
        ax.set_ylabel("Log-odds", fontsize=7)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(axis="y", alpha=0.2)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("EBM Shape Functions — all phases overlaid\n"
                 "Above 0 → pushes toward phase  |  x=0 → user at personal mean",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape overlay grid → {output_path.name}")


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Normalised OOF confusion matrix."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of true class")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.6 else "black")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("EBM — OOF Confusion Matrix\n(count  /  row fraction)", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Confusion matrix → {output_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True,
                   help="CSV with author, phase, *_zscore columns.")
    p.add_argument("--top-n",   type=int, default=15)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--equalize-phase-posts", action="store_true",
                   help="Sample equal posts per phase per user (= min across phases) before "
                        "computing mean profiles. Removes estimation noise from unequal phase lengths.")
    p.add_argument("--window-days", type=int, default=None,
                   help="Restrict to offset_from_cd1 ∈ [-N, N] before aggregating.")
    p.add_argument("--feature", default=None,
                   help="Plot shape for a single named feature instead of top-N.")
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

    # ── [1] Load ──────────────────────────────────────────────────────────────
    logging.info("[1] Loading data…")
    df, zscore_cols, le = load_dataset(args.input_file,
                                        equalize_phase_posts=args.equalize_phase_posts,
                                        seed=seed,
                                        window_days=args.window_days)
    # Drop zero-variance features — EBM's density histogram breaks on all-constant columns
    variances = df[zscore_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        logging.warning(f"  Dropping {len(zero_var)} zero-variance features: {zero_var}")
        zscore_cols = [c for c in zscore_cols if c not in zero_var]

    X             = df[zscore_cols].values.astype(np.float64)
    y             = le.transform(df["phase"])
    groups        = df["author"].values
    class_names   = list(le.classes_)
    feature_names = zscore_cols
    logging.info(f"  X: {X.shape}  classes: {class_names}  users: {len(np.unique(groups)):,}")

    # ── [2] Cross-validation ──────────────────────────────────────────────────
    if not args.skip_cv:
        logging.info(f"\n[2] GroupKFold CV ({args.n_folds} folds)…")
        y_pred, y_proba, _fold_aucs = run_cv(X, y, groups, n_classes=len(class_names),
                                            seed=seed, n_splits=args.n_folds)

        bal_acc = balanced_accuracy_score(y, y_pred)
        acc     = accuracy_score(y, y_pred)
        auc     = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        logging.info(f"  OOF  balanced_acc={bal_acc:.4f}  acc={acc:.4f}  AUC(OvR)={auc:.4f}")

        oof_report = (
            f"EBM Phase Analysis — OOF Report ({ts})\n"
            f"StratifiedGroupKFold  n_folds={args.n_folds}  "
            f"interactions=0  max_bins=256  sample_weight=balanced\n"
            f"{'='*60}\n"
            f"balanced_acc : {bal_acc:.4f}\n"
            f"accuracy     : {acc:.4f}\n"
            f"AUC (OvR)    : {auc:.4f}\n\n"
            + classification_report(y, y_pred, target_names=class_names)
        )
        (OUT_DIR / f"ebm_oof_report_{ts}.txt").write_text(oof_report)
        logging.info(f"  OOF report saved.")
    else:
        logging.info("\n[2] CV skipped (--skip-cv).")
        y_pred = None

    # ── [3] Final fit ─────────────────────────────────────────────────────────
    logging.info("\n[3] Final fit on full data…")
    ebm = make_ebm(seed)
    ebm.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    logging.info("  Fitted.")

    # ── [4] Importance heatmap ────────────────────────────────────────────────
    logging.info("\n[4] Extracting importance…")
    df_imp = global_importance(ebm, feature_names, class_names)

    print("\nTop 10 features by global weighted importance:")
    print(df_imp[["feature", "global_avg"]].head(10).to_string(index=False))

    plot_importance_heatmap(
        df_imp, class_names,
        OUT_DIR / f"ebm_importance_heatmap_{ts}.png",
        top_n=min(args.top_n, len(df_imp)),
    )

    # ── [5] Shape overlay grid ────────────────────────────────────────────────
    logging.info("\n[5] Shape overlay grid…")
    if args.feature:
        if args.feature not in feature_names:
            logging.error(f"  Feature '{args.feature}' not found.")
            sys.exit(1)
        features_to_plot = [args.feature]
    else:
        features_to_plot = df_imp["feature"].head(args.top_n).tolist()

    plot_shape_overlay_grid(
        ebm, features_to_plot, feature_names, class_names, PHASE_COLORS,
        OUT_DIR / f"ebm_shapes_{ts}.png",
    )

    # ── [6] Confusion matrix ──────────────────────────────────────────────────
    if y_pred is not None:
        logging.info("\n[6] Confusion matrix…")
        plot_confusion(y, y_pred, class_names, OUT_DIR / f"ebm_confusion_{ts}.png")

    logging.info(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
