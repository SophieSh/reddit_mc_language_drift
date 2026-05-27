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
  python scripts/38_ebm_phase_analysis.py --input-file data/interim/prepared.csv --feature valence_dict_average_zscore
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
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER

warnings.filterwarnings("ignore", category=UserWarning)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

OUT_DIR = ROOT / "reports" / "ebm"


# ── DATA LOADING ──────────────────────────────────────────────────────────────
# we don't need this, just put placeholder for a file, assume that it has z-scored columns and phase column
def load_dataset(path: str) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """Read CSV, aggregate to one mean profile per (author, phase), return X/y/groups."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"Loaded {Path(path).name}: {len(df):,} rows, {df['author'].nunique():,} users")

    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    if not zscore_cols:
        raise ValueError("No *_zscore columns found. Check input file.")
    logging.info(f"  {len(zscore_cols)} z-score features")

    # One mean profile per (author, phase)
    profiles = df.groupby(["author", "phase"])[zscore_cols].mean().reset_index()
    logging.info(f"  Profiles: {len(profiles):,}  ({profiles['author'].nunique():,} users × up to 4 phases)")

    le = LabelEncoder().fit(PHASE_ORDER)
    return profiles, zscore_cols, le


# ── EBM FACTORY ───────────────────────────────────────────────────────────────

def _make_ebm(seed: int) -> ExplainableBoostingClassifier:
    return ExplainableBoostingClassifier(
        max_bins=256,
        max_interaction_bins=64,
        interactions=0,
        learning_rate=0.01,
        max_rounds=5000,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


# ── SHAPE HELPERS ─────────────────────────────────────────────────────────────

def _parse_bin_midpoints(bin_labels: list) -> np.ndarray:
    """Convert EBM bin-edge strings ('a to b', '>a', '<=a') to numeric midpoints."""
    vals = []
    for label in bin_labels:
        label = str(label)
        try:
            if " to " in label:
                lo, hi = label.split(" to ")
                vals.append((float(lo) + float(hi)) / 2)
            elif label.startswith(">"):
                vals.append(float(label[1:].strip()))
            elif label.startswith("<="):
                vals.append(float(label[2:].strip()))
            else:
                vals.append(float(label))
        except ValueError:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float64)


def _extract_density(data: dict, n_bins: int) -> np.ndarray:
    """Return normalised density array (length n_bins) aligned to score bins by x value."""
    density_raw = data.get("density")
    if not isinstance(density_raw, dict):
        return np.ones(n_bins) / n_bins
    dens_counts = np.asarray(density_raw.get("scores", []), dtype=float)
    dens_edges  = np.asarray(density_raw.get("names",  []), dtype=float)
    if len(dens_counts) == 0 or len(dens_edges) < 2:
        return np.ones(n_bins) / n_bins
    total = dens_counts.sum()
    if total == 0:
        return np.ones(n_bins) / n_bins
    dens_frac = dens_counts / total
    if len(dens_counts) == n_bins:
        return dens_frac
    score_edges = np.asarray(data.get("names", []), dtype=float)
    if len(score_edges) < n_bins + 1:
        return np.ones(n_bins) / n_bins
    midpoints   = (score_edges[:n_bins] + score_edges[1:n_bins + 1]) / 2
    bin_indices = np.clip(np.searchsorted(dens_edges[1:], midpoints, side="left"), 0, len(dens_frac) - 1)
    out = dens_frac[bin_indices]
    s   = out.sum()
    return out / s if s > 0 else np.ones(n_bins) / n_bins


def _clean(name: str) -> str:
    return name.replace("_zscore", "").replace("_", " ")


# ── CROSS-VALIDATION ──────────────────────────────────────────────────────────

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    le: LabelEncoder,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold OOF evaluation of an EBM.

    Why StratifiedGroupKFold?
      All phase-profiles from the same user (up to 4 rows) share the same writing
      style, baseline vocabulary, and cycle trajectory.  They are NOT independent
      samples — GroupKFold keeps all rows for a given author in the same fold.
      Stratified additionally preserves the phase-class distribution per fold,
      which matters because Follicular has more data for more users than Ovulation.

      Classes (y): These are the 4 phases (Menstrual, Follicular, Ovulation, and Luteal).
      These are the labels the model is trying to predict.
      Groups: These are the Authors (the users).
      There are many unique groups (one for each unique user in your CSV).

    Returns:
        y_pred  : (n_samples,) integer hard predictions (OOF)
        y_proba : (n_samples, n_classes) probability matrix (OOF)
    """
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    #encoding = [Menstrual, Follicular, Ovulation, and Luteal] for model usage
    n_classes = len(le.classes_)
    # number of users x 4 (up to 4 averaged values of the phase profile for each user)
    y_pred  = np.empty_like(y)
    # matrix of probabilities for each class for each sample
    y_proba = np.zeros((len(y), n_classes), dtype=np.float64)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        logging.info(f"  Fold {fold}/{n_splits}  train={len(tr):,}  test={len(te):,}")
        # EBM is fitted fresh on each fold (clean, no memory)
        # n_jobs=-1 parallelises the boosting rounds across features.
        ebm = _make_ebm(seed)
        ebm.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        y_pred[te]  = ebm.predict(X[te])
        y_proba[te] = ebm.predict_proba(X[te])

    return y_pred, y_proba


# ── IMPORTANCE ────────────────────────────────────────────────────────────────

def extract_global_importance(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
    le: LabelEncoder,
) -> pd.DataFrame:
    """Density-weighted mean |log-odds| per feature per phase, sorted by global average."""
    global_exp  = ebm.explain_global()
    phase_names = list(le.classes_)
    rows = []

    for fi, feat in enumerate(feature_names):
        data    = global_exp.data(fi)
        scores  = np.asarray(data["scores"])
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]

        density_raw = data.get("density", np.ones(scores.shape[0]))
        if isinstance(density_raw, dict):
            density_raw = density_raw.get("scores", np.ones(scores.shape[0]))
        density = np.asarray(density_raw, dtype=float)
        if len(density) != scores.shape[0]:
            density = np.ones(scores.shape[0])
        density /= density.sum()

        weighted_imp = np.sum(np.abs(scores) * density[:, np.newaxis], axis=0)
        row = {"feature": feat}
        for i, phase in enumerate(phase_names):
            row[f"{phase}_imp"] = float(weighted_imp[i])
        row["global_avg"] = float(np.mean(weighted_imp))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("global_avg", ascending=False).reset_index(drop=True)


def extract_shape_function(
    ebm: ExplainableBoostingClassifier,
    feature_name: str,
    feature_names: list[str],
    le: LabelEncoder,
) -> dict[str, dict]:
    """Log-odds shape curves for one feature across all 4 phases."""
    fi   = feature_names.index(feature_name)
    data = ebm.explain_global().data(fi)

    scores = np.asarray(data["scores"])
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    x = _parse_bin_midpoints(data["names"])
    n = min(len(x), scores.shape[0])
    x, scores = x[:n], scores[:n]
    density   = _extract_density(data, n)

    result = {}
    for ci, phase in enumerate(le.classes_):
        if ci >= scores.shape[1]:
            continue
        y = scores[:, ci]
        result[phase] = {
            "x":            x,
            "y":            y,
            "density":      density,
            "weighted_sum": float(np.sum(np.abs(y) * density)),
        }
    return result


# ── FEATURE INTERACTIONS ──────────────────────────────────────────────────────

def print_feature_interactions(
    ebm: ExplainableBoostingClassifier,
    X_train: np.ndarray,
    feature_names: list[str],
    le: LabelEncoder,
    top_features_idx: np.ndarray,
) -> None:
    """Print log-odds combinations for Low/High of the top-2 features."""
    if len(top_features_idx) < 2:
        return
    fi1, fi2 = top_features_idx[0], top_features_idx[1]
    f1, f2   = feature_names[fi1], feature_names[fi2]
    print(f"\nFeature interactions — top 2: '{_clean(f1)}'  ×  '{_clean(f2)}'")

    lo1, hi1 = np.percentile(X_train[:, fi1], [25, 75])
    lo2, hi2 = np.percentile(X_train[:, fi2], [25, 75])
    baseline = np.mean(X_train, axis=0)

    global_exp  = ebm.explain_global()
    phase_names = list(le.classes_)

    for desc, v1, v2 in [("Low–Low", lo1, lo2), ("Low–High", lo1, hi2),
                          ("High–Low", hi1, lo2), ("High–High", hi1, hi2)]:
        sample = baseline.copy()
        sample[fi1], sample[fi2] = v1, v2
        proba = ebm.predict_proba(sample.reshape(1, -1))[0]
        top   = phase_names[np.argmax(proba)]
        print(f"  {desc:10s} → {top:12s}  probs: " +
              "  ".join(f"{p}={proba[i]:.2f}" for i, p in enumerate(phase_names)))


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_importance_heatmap(
    df_imp: pd.DataFrame,
    le: LabelEncoder,
    output_path: Path,
    top_n: int = 20,
) -> None:
    top         = df_imp.head(top_n)
    phase_names = list(le.classes_)
    matrix      = top[[f"{p}_imp" for p in phase_names]].values
    feat_labels = [_clean(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Mean |log-odds|")
    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_title(f"EBM Feature Importance — top {top_n}\nMean |log-odds| per phase",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance heatmap → {output_path.name}")


def plot_shape_overlay_grid(
    ebm: ExplainableBoostingClassifier,
    features_to_plot: list[str],
    feature_names: list[str],
    le: LabelEncoder,
    output_path: Path,
) -> None:
    """One axes per feature; all 4 phase curves overlaid on the same axes."""
    n_feat = len(features_to_plot)
    ncols  = 2
    nrows  = (n_feat + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features_to_plot):
        ax         = axes[i]
        shape_data = extract_shape_function(ebm, feat, feature_names, le)

        # Background: shade region dominated by each phase
        # (phase with highest log-odds at each x-interval)
        all_x = None
        all_y = {}
        for phase, d in shape_data.items():
            valid = ~np.isnan(d["x"])
            if all_x is None:
                all_x = d["x"][valid]
            all_y[phase] = d["y"][valid]

        if all_x is not None and len(all_x) > 1:
            y_stack = np.array([all_y.get(p, np.zeros_like(all_x))
                                for p in le.classes_])
            dominant = np.argmax(y_stack, axis=0)
            for j in range(len(all_x) - 1):
                phase = le.classes_[dominant[j]]
                ax.axvspan(all_x[j], all_x[j + 1],
                           alpha=0.08, color=PHASE_COLORS.get(phase, "gray"), linewidth=0)

        # Overlay phase curves
        for phase, d in shape_data.items():
            color = PHASE_COLORS.get(phase, "gray")
            valid = ~np.isnan(d["x"])
            x_v, y_v = d["x"][valid], d["y"][valid]
            ax.step(x_v, y_v, where="mid", color=color, linewidth=1.8,
                    label=f"{phase} ({d['weighted_sum']:.2f})")
            ax.fill_between(x_v, y_v, 0, step="mid", alpha=0.15, color=color)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.5, linestyle=":")
        ax.set_title(_clean(feat), fontsize=8, fontweight="bold")
        ax.set_xlabel("z-score vs user mean", fontsize=7)
        ax.set_ylabel("Log-odds", fontsize=7)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(axis="y", alpha=0.2)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("EBM Shape Functions — all phases overlaid\n"
                 "Above 0 → pushes toward phase  |  x=0 → user at personal mean  |"
                 "  Legend: phase (weighted |log-odds|)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape overlay grid → {output_path.name}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
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
    p.add_argument("--top-n",   type=int, default=15,
                   help="Features to include in plots (default 15).")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip-cv", action="store_true",
                   help="Skip cross-validation, go straight to final fit.")
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
    df, zscore_cols, le = load_dataset(args.input_file)
    X      = df[zscore_cols].values.astype(np.float64)
    y      = le.transform(df["phase"])
    groups = df["author"].values
    class_names  = list(le.classes_)
    feature_names = zscore_cols
    logging.info(f"  X: {X.shape}  classes: {class_names}  users: {len(np.unique(groups)):,}")

    # ── [2] Cross-validation ──────────────────────────────────────────────────
    if not args.skip_cv:
        logging.info(f"\n[2] GroupKFold CV ({args.n_folds} folds)…")
        y_pred, y_proba = run_cv(X, y, groups, le=le, seed=seed, n_splits=args.n_folds)

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
        rpt_path = OUT_DIR / f"ebm_oof_report_{ts}.txt"
        rpt_path.write_text(oof_report)
        logging.info(f"  OOF report → {rpt_path.name}")
    else:
        logging.info("\n[2] CV skipped (--skip-cv).")
        y_pred = None

    # ── [3] Final fit ─────────────────────────────────────────────────────────
    logging.info("\n[3] Final fit on full data…")
    ebm = _make_ebm(seed)
    ebm.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    logging.info("  Fitted.")

    # ── [4] Importance heatmap ────────────────────────────────────────────────
    logging.info("\n[4] Extracting importance…")
    df_imp = extract_global_importance(ebm, feature_names, le)

    top10 = df_imp[["feature", "global_avg"]].head(10)
    print("\nTop 10 features by global weighted importance:")
    print(top10.to_string(index=False))

    plot_importance_heatmap(
        df_imp, le,
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
        ebm, features_to_plot, feature_names, le,
        OUT_DIR / f"ebm_shapes_{ts}.png",
    )

    # ── [6] Confusion matrix ──────────────────────────────────────────────────
    if y_pred is not None:
        logging.info("\n[6] Confusion matrix…")
        plot_confusion_matrix(y, y_pred, class_names,
                              OUT_DIR / f"ebm_confusion_{ts}.png")

    # ── [7] Feature interactions (console only) ───────────────────────────────
    top2_idx = np.argsort([df_imp.index[df_imp["feature"] == f].item()
                           for f in features_to_plot[:2]])[:2]
    top2_feat_idx = np.array([feature_names.index(f) for f in features_to_plot[:2]])
    print_feature_interactions(ebm, X, feature_names, le, top2_feat_idx)

    logging.info(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
