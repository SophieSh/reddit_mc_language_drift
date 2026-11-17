"""
Step 39 — Fixed-Effects EBM Phase Classifier
=============================================
4-class Explainable Boosting Machine on menstrual cycle phases using
within-user demeaning (the "fixed effect") instead of aggregating posts
to one mean profile per (author, phase).

Motivation
  Script 38 aggregates each user's posts to a single mean per phase.  That
  compresses a 14-day Luteal window and a 3-day Ovulation window to the same
  single data point, so the model never sees within-person variation AND the
  Ovulation mean is far noisier (fewer observations).  Here each posting day
  stays as its own row.  We remove the user-level baseline by subtracting each
  user's grand mean across all their days:

      x_demeaned = x_post − user_mean_across_all_days

  The EBM then sees only within-person linguistic variation — the part that
  actually varies with cycle phase.

Design
  • Within-user demeaning replaces the z-score / equalization trick.
  • StratifiedGroupKFold (5 folds, grouped by author) — no user leakage.
  • sample_weight="balanced" — corrects Luteal / Ovulation class imbalance.
  • Pure additive EBM (interactions=0) — every feature has a shape function.

Outputs (reports/ebm/fixed_effects/)
  fe_phase_oof_report_<ts>.txt       — per-fold + overall metrics
  fe_phase_importance_<ts>.png       — top-N feature importance heatmap

Usage
  python scripts/39_ebm_fixed_effects_phase.py --input-file data/interim/prepared.csv
  python scripts/39_ebm_fixed_effects_phase.py --input-file data/interim/prepared.csv --top-n 20 --window-days 14
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
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_ORDER
from src.ebm_utils import clean_name, global_importance, make_ebm, run_cv

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = ROOT / "reports" / "ebm" / "fixed_effects"


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def detect_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names: prefer *_zscore, fall back to *_mean."""
    exclude = {"author", "phase", "offset_from_cd1"}
    if any(c.endswith("_zscore") for c in df.columns):
        cols = [c for c in df.columns if c.endswith("_zscore")]
        logging.info(f"  Feature source: *_zscore  ({len(cols)} columns)")
        return cols
    if any(c.endswith("_mean") for c in df.columns):
        cols = [c for c in df.columns if c.endswith("_mean")]
        logging.info(f"  Feature source: *_mean  ({len(cols)} columns)")
        return cols
    raise ValueError("No *_zscore or *_mean feature columns found. Check input file.")


def demean_within_user(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Subtract each user's grand mean (across all their posting days) from
    every feature column.  Returns a copy with demeaned values in-place."""
    user_means = df.groupby("author")[feat_cols].transform("mean")
    df = df.copy()
    df[feat_cols] = df[feat_cols].values - user_means.values
    return df


def load_dataset(
    path: str,
    window_days: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """Read CSV, apply optional time window, demean within user.

    Returns
    -------
    df        : demeaned row-level DataFrame (one row = one posting day)
    feat_cols : feature column names (after zero-variance drop)
    le        : fitted LabelEncoder for PHASE_ORDER
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    logging.info(
        f"Loaded {Path(path).name}: {len(df):,} rows, "
        f"{df['author'].nunique():,} users"
    )

    if window_days is not None:
        before = len(df)
        df = df[df["offset_from_cd1"].between(-window_days, window_days)]
        logging.info(
            f"  Window ±{window_days} days: {before:,} → {len(df):,} rows, "
            f"{df['author'].nunique():,} users"
        )

    # Keep only rows whose phase label is recognised.
    before = len(df)
    df = df[df["phase"].isin(PHASE_ORDER)].reset_index(drop=True)
    if len(df) < before:
        logging.warning(f"  Dropped {before - len(df):,} rows with unknown phase labels.")

    feat_cols = detect_feature_cols(df)

    df = demean_within_user(df, feat_cols)
    logging.info(f"  Within-user demeaning applied.  Rows: {len(df):,}")

    # Drop zero-variance features — EBM bin_density crashes on all-constant columns.
    variances = df[feat_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        logging.warning(f"  Dropping {len(zero_var)} zero-variance features: {zero_var}")
        feat_cols = [c for c in feat_cols if c not in zero_var]

    le = LabelEncoder().fit(PHASE_ORDER)
    return df, feat_cols, le


# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot_importance_heatmap(
    df_imp: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Heatmap of density-weighted mean |log-odds| importance: features × phases."""
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
    ax.set_title(
        f"Fixed-Effects EBM — Feature Importance (top {top_n})\n"
        "Mean |log-odds| per phase  |  within-user demeaned features",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance heatmap → {output_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True,
                   help="CSV with author, phase, offset_from_cd1, *_zscore or *_mean columns.")
    p.add_argument("--top-n",      type=int, default=15,
                   help="Number of top features to show in the importance heatmap.")
    p.add_argument("--n-folds",    type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--window-days", type=int, default=None,
                   help="Restrict to offset_from_cd1 ∈ [-N, N] before processing.")
    p.add_argument("--skip-cv",    action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
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

    # ── [1] Load + demean ─────────────────────────────────────────────────────
    logging.info("[1] Loading and demeaning data…")
    df, feat_cols, le = load_dataset(
        args.input_file, window_days=args.window_days, seed=seed
    )

    X           = df[feat_cols].values.astype(np.float64)
    y           = le.transform(df["phase"])
    groups      = df["author"].values
    class_names = list(le.classes_)
    logging.info(
        f"  X: {X.shape}  classes: {class_names}  "
        f"users: {len(np.unique(groups)):,}"
    )

    # ── [2] Cross-validation ──────────────────────────────────────────────────
    if not args.skip_cv:
        logging.info(f"\n[2] StratifiedGroupKFold CV ({args.n_folds} folds)…")
        y_pred, y_proba, _fold_aucs = run_cv(
            X, y, groups,
            n_classes=len(class_names),
            seed=seed,
            n_splits=args.n_folds,
        )

        bal_acc = balanced_accuracy_score(y, y_pred)
        auc     = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        logging.info(
            f"  OOF  balanced_acc={bal_acc:.4f}  AUC(OvR macro)={auc:.4f}"
        )

        # Per-fold summary is already logged inside run_cv; collect overall here.
        oof_report = (
            f"Fixed-Effects EBM Phase Classifier — OOF Report ({ts})\n"
            f"Within-user demeaning  |  StratifiedGroupKFold  n_folds={args.n_folds}\n"
            f"interactions=0  max_bins=256  sample_weight=balanced\n"
            f"{'='*60}\n"
            f"balanced_acc : {bal_acc:.4f}\n"
            f"AUC (OvR)    : {auc:.4f}\n\n"
            + classification_report(y, y_pred, target_names=class_names)
        )
        report_path = OUT_DIR / f"fe_phase_oof_report_{ts}.txt"
        report_path.write_text(oof_report)
        logging.info(f"  OOF report → {report_path.name}")
    else:
        logging.info("\n[2] CV skipped (--skip-cv).")

    # ── [3] Final fit on full data ────────────────────────────────────────────
    logging.info("\n[3] Final fit on full data…")
    ebm = make_ebm(seed)
    ebm.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    logging.info("  Fitted.")

    # ── [4] Importance heatmap ────────────────────────────────────────────────
    logging.info("\n[4] Extracting global importance…")
    df_imp = global_importance(ebm, feat_cols, class_names)

    print("\nTop 10 features by global weighted importance:")
    print(df_imp[["feature", "global_avg"]].head(10).to_string(index=False))

    plot_importance_heatmap(
        df_imp, class_names,
        OUT_DIR / f"fe_phase_importance_{ts}.png",
        top_n=min(args.top_n, len(df_imp)),
    )

    logging.info(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
