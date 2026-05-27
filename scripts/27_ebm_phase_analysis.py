"""
Step 27 — Explainable Boosting Machine (EBM) Phase Analysis
=============================================================
Goal: Feature Discovery — identify which linguistic features are characteristic
of each menstrual cycle phase, and in which *direction* they move.

Why   over XGBoost for this goal?
  XGBoost is a black-box ensemble: feature importance is a global summary score
  (mean |SHAP|), but you cannot see the *shape* of the relationship — whether a
  feature monotonically increases, peaks at a threshold, or has a non-linear curve.

  EBM (Explainable Boosting Machine, a.k.a. GA²M) learns an explicit shape function
  for every feature:

      P(phase | x) ∝ f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ)

  Each fᵢ is a piecewise-constant function over the feature's value range.  The
  y-axis of that function is in *log-odds* units: positive = pushes toward the
  class, negative = pushes away.  This is exactly what we want for biological
  interpretation:

    "During Menstrual phase, valence_zscore at −1.5 contributes +0.8 log-odds
     (i.e. a user scoring 1.5 SD below their personal mean in valence is ~2.2×
     more likely to be classified as Menstrual)."

  EBMs also handle class imbalance well and are competitive with gradient boosting
  on tabular data.

Pipeline
  [1] Load step-08b phase-labeled timeline (timeline_phase_labeled_*.csv),
      compute per-user z-scores from _mean columns, aggregate to one profile
      per (user, phase).
  [2] StratifiedGroupKFold (5 folds, grouped by author) OOF evaluation.
  [3] Final fit on full dataset.
  [4] Extract global explanation: per-feature importance by phase.
  [5] Extract local shape functions: bin edges + log-odds curve for any feature.
  [6] Save summary CSVs and PNG plots for every feature × phase combination.

Usage
  python scripts/27_ebm_phase_analysis.py --config configs/base.yaml
  python scripts/27_ebm_phase_analysis.py --config configs/base.yaml --feature valence_dict_average_zscore
  python scripts/27_ebm_phase_analysis.py --config configs/base.yaml --top-n 10

Output (reports/ebm/)
  ebm_global_importance_<timestamp>.csv   — ranked feature importance per phase
  ebm_shape_<feature>_<timestamp>.png     — shape function plot per top feature
  ebm_oof_report_<timestamp>.txt          — OOF AUC scores
"""

import argparse
import logging
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import PHASE_COLORS, PHASE_ORDER
from src.ml_data import load_phase_labeled_dataset
from src.ml_eval import log_oof_metrics, run_phase_statistical_gauntlet

warnings.filterwarnings("ignore", category=UserWarning)

# Class order must be consistent with LabelEncoder throughout the script.
# LabelEncoder sorts alphabetically, so this is the canonical mapping:
#   0 = Follicular, 1 = Luteal, 2 = Menstrual, 3 = Ovulation
# PHASE_ORDER and PHASE_COLORS are imported from src.constants.


def _make_ebm(seed: int) -> ExplainableBoostingClassifier:
    """Construct a canonical EBM with the project-standard hyperparameters.

    All EBM instances in this script must be created via this factory so that
    hyperparameters are defined in a single place and folds / final fits are
    guaranteed to use identical settings.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        An unfitted ExplainableBoostingClassifier.
    """
    return ExplainableBoostingClassifier(
        max_bins=256,            # resolution of shape functions
        max_interaction_bins=64,
        interactions=0,          # pure additive model — easier to interpret
        learning_rate=0.01,
        max_rounds=5000,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def _clean_feat_name(name: str) -> str:
    """Strip z-score suffix and replace underscores with spaces for display.

    Args:
        name: Raw feature column name (e.g. "valence_dict_average_zscore").

    Returns:
        Human-readable label (e.g. "valence dict average").
    """
    return name.replace("_zscore", "").replace("_", " ")


# load_phase_labeled_dataset is imported from src.ml_data


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_encoder: LabelEncoder,
    n_splits: int = 5,
    seed: int = 42,
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
    n_classes = len(label_encoder.classes_)
    # number of users x 4 (up to 4 averaged values of the phase profile for each user)
    y_pred  = np.empty_like(y)
    # matrix of probabilities for each class for each sample
    y_proba = np.zeros((len(y), n_classes), dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        logging.info(f"  Fold {fold}/{n_splits} — "
                     f"train={len(train_idx):,} test={len(test_idx):,}")

        # EBM is fitted fresh on each fold (clean, no memory)
        # n_jobs=-1 parallelises the boosting rounds across features.
        ebm_fold = _make_ebm(seed)
        ebm_fold.fit(X[train_idx], y[train_idx])

        y_pred[test_idx]  = ebm_fold.predict(X[test_idx])
        y_proba[test_idx] = ebm_fold.predict_proba(X[test_idx])

    return y_pred, y_proba

# log_oof_metrics and run_phase_statistical_gauntlet are imported from src.ml_eval


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_density(data: dict, n_bins: int) -> np.ndarray:
    """Return a normalized density array of length n_bins, aligned with score bins by x value.

    EBM stores the training-data density histogram at a coarser resolution than the score
    bins (e.g. 20 density bins vs 254 score bins).  Direct length-matching therefore fails.
    Instead we map each score bin to its corresponding density bin by midpoint x value, then
    re-normalize so the returned array sums to 1.
    Falls back to uniform weights if any required field is missing or malformed.
    """
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

    # Fast path: density and score bins already match.
    if len(dens_counts) == n_bins:
        return dens_frac

    # General path: map score bin midpoints → density bins via shared x-axis edges.
    score_edges = np.asarray(data.get("names", []), dtype=float)
    if len(score_edges) < n_bins + 1:
        return np.ones(n_bins) / n_bins

    score_midpoints = (score_edges[:n_bins] + score_edges[1 : n_bins + 1]) / 2
    # searchsorted against density right-edges finds the bin each midpoint falls in.
    bin_indices = np.searchsorted(dens_edges[1:], score_midpoints, side="left")
    bin_indices = np.clip(bin_indices, 0, len(dens_frac) - 1)

    density_per_score_bin = dens_frac[bin_indices]
    s = density_per_score_bin.sum()
    return density_per_score_bin / s if s > 0 else np.ones(n_bins) / n_bins


def extract_global_importance(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    global_exp = ebm.explain_global()
    phase_names = list(label_encoder.classes_)
    rows = []

    for feat_idx, feat_name in enumerate(feature_names):
        data = global_exp.data(feat_idx)
        scores = np.asarray(data["scores"])

        # 1. Correct 2D Reshaping (Ensures bins are rows, phases are columns)
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]

        # 2. Normalized Density (Ensures weights sum to 1 for fair comparison)
        # data["density"] is a dict in current interpret versions — unwrap to raw counts first.
        density_raw = data.get("density", np.ones(scores.shape[0]))
        if isinstance(density_raw, dict):
            density_raw = density_raw.get("scores", np.ones(scores.shape[0]))
        density = np.asarray(density_raw, dtype=float)
        if len(density) != scores.shape[0]:
            density = np.ones(scores.shape[0])
        density = density / density.sum()  # Now density is a fraction (0.0 to 1.0)

        # 3. Weighted Importance (Global average impact per user)
        # Multiply scores by the normalized density
        weighted_imp = np.sum(np.abs(scores) * density[:, np.newaxis], axis=0)

        row = {"feature": feat_name}
        for i, phase in enumerate(phase_names):
            row[f"{phase}_weighted_imp"] = float(weighted_imp[i])

        # 4. Summary Sorting Column
        row["global_weighted_avg"] = float(np.mean(weighted_imp))

        rows.append(row)

    # Sort by the normalized global importance
    return pd.DataFrame(rows).sort_values("global_weighted_avg", ascending=False)

def _parse_bin_midpoints(bin_labels: list) -> np.ndarray:
    """Convert EBM bin-edge strings to numeric midpoints.

    EBM label formats: "a to b" → midpoint, "> a" or "<= a" → boundary value.
    Unparseable labels become NaN.
    """
    x_vals = []
    for label in bin_labels:
        label = str(label)
        try:
            if " to " in label:
                lo, hi = label.split(" to ")
                x_vals.append((float(lo) + float(hi)) / 2)
            elif label.startswith(">"):
                x_vals.append(float(label[1:].strip()))
            elif label.startswith("<="):
                x_vals.append(float(label[2:].strip()))
            else:
                x_vals.append(float(label))
        except ValueError:
            x_vals.append(np.nan)
    return np.array(x_vals, dtype=np.float64)


def extract_shape_function(
    ebm: ExplainableBoostingClassifier,
    feature_name: str,
    feature_names: list[str],
    label_encoder: LabelEncoder,
) -> dict[str, dict]:
    """Extract the shape function for one feature across all phases.

    Returns per-phase dicts with:
        x            : bin midpoints (z-score relative to user mean)
        y            : log-odds contribution per bin
        density      : normalized fraction of users per bin
        weighted_sum : density-weighted mean |log-odds| (overall signal strength)
    """
    if feature_name not in feature_names:
        raise ValueError(f"'{feature_name}' not found. Available: {feature_names[:5]} …")

    data = ebm.explain_global().data(feature_names.index(feature_name))

    scores = np.asarray(data["scores"])
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    x = _parse_bin_midpoints(data["names"])

    # EBM sometimes emits one extra bin label — trim to the shorter length.
    n = min(len(x), scores.shape[0])
    x, scores = x[:n], scores[:n]

    density = _extract_density(data, n)

    result = {}
    for cls_idx, phase in enumerate(label_encoder.classes_):
        if cls_idx >= scores.shape[1]:
            continue
        y = scores[:, cls_idx]
        result[phase] = {
            "x":           x,
            "y":           y,
            "density":     density,
            "weighted_sum": float(np.sum(np.abs(y) * density)),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def plot_shape_function(
    shape_data: dict[str, dict],
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot the EBM shape function (log-odds vs feature value) for all 4 phases.

    Each subplot title shows the density-weighted mean |log-odds|.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    axes = axes.flatten()
    clean_name = _clean_feat_name(feature_name)

    for ax, (phase, d) in zip(axes, shape_data.items()):
        color = PHASE_COLORS.get(phase, "gray")
        x, y = d["x"], d["y"]

        valid = ~np.isnan(x)
        x_v, y_v = x[valid], y[valid]

        ax.step(x_v, y_v, where="mid", color=color, linewidth=2)
        ax.fill_between(x_v, y_v, 0, step="mid", alpha=0.25, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.6, linestyle=":")

        ax.set_title(
            f"{phase}  |  weighted={d['weighted_sum']:.3f}",
            fontsize=8, fontweight="bold", color=color,
        )
        ax.set_xlabel("Feature value (z-score vs user mean)", fontsize=9)
        ax.set_ylabel("Log-odds contribution", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"EBM Shape Function — {clean_name}\n"
        "Positive log-odds → pushes toward phase  |  x=0 → user at personal mean",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape plot → {output_path.name}")

def plot_global_importance_heatmap(
    df_imp: pd.DataFrame,
    label_encoder: LabelEncoder,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Heatmap of mean-absolute log-odds importance: features × phases."""
    top = df_imp.head(top_n)
    phase_names = list(label_encoder.classes_)
    phase_cols = [f"{p}_weighted_imp" for p in phase_names]

    matrix = top[phase_cols].values  # (top_n, n_classes)
    feature_labels = [_clean_feat_name(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=8)
    ax.set_title(
        f"EBM Global Feature Importance (top {top_n})\n"
        "Mean |log-odds| per phase — higher = more discriminative",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Mean |log-odds|")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance heatmap → {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument(
        "--no-anchors", action="store_true",
        help="Load the no-anchors phase-labeled file instead of the default (with-anchors).",
    )
    p.add_argument(
        "--phase-file", default=None,
        help="Path or filename (in interim dir) of a specific phase-labeled CSV to use "
             "(e.g. timeline_phase_labeled_fixed29_20260517T131445.csv).",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of StratifiedGroupKFold splits.",
    )
    p.add_argument(
        "--feature", default=None,
        help="Name of a specific feature to plot the shape function for "
             "(e.g. 'valence_dict_average_zscore'). If omitted, top --top-n are plotted.",
    )
    p.add_argument(
        "--top-n", type=int, default=15,
        help="Number of top-importance features to plot shape functions for.",
    )
    p.add_argument(
        "--skip-cv", action="store_true",
        help="Skip cross-validation and go straight to the final fit + interpretation. "
             "Useful when you only want to explore shape functions quickly.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = ROOT / "reports" / "ebm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    logging.info("\n[1/4] Loading data…")
    df, zscore_cols, le = load_phase_labeled_dataset(cfg, no_anchors=args.no_anchors, phase_file=args.phase_file)

    # Build numpy arrays — these are the X / y / users the model trains on.
    # le is already fitted on PHASE_ORDER by load_phase_labeled_dataset.
    # Alphabetical mapping: Follicular=0, Luteal=1, Menstrual=2, Ovulation=3
    y      = le.transform(df["phase"])
    X      = df[zscore_cols].values.astype(np.float64)
    groups = df["author"].values
    feature_names = zscore_cols  # list of strings, length == X.shape[1]

    logging.info(
        f"\n  X shape : {X.shape}"
        f"\n  Classes : {list(le.classes_)}"
        f"\n  Users   : {len(np.unique(groups)):,}"
    )

    # ── [2] Cross-validation ─────────────────────────────────────────────────
    if not args.skip_cv:
        logging.info(f"\n[2/4] GroupKFold CV ({args.n_folds} folds)…")
        y_pred, y_proba = run_group_kfold_cv(X, y, groups, le, n_splits=args.n_folds, seed=seed)

        oof_report = log_oof_metrics(y, y_proba, le)

        rpt_path = output_dir / f"ebm_oof_report_{timestamp}.txt"
        rpt_path.write_text(
            f"EBM GroupKFold ({args.n_folds} folds)\n"
            f"interactions=0 (pure additive), max_bins=256\n\n"
            + oof_report
        )
        logging.info(f"  OOF report → {rpt_path.name}")
    else:
        logging.info("\n[2/4] CV skipped (--skip-cv).")

    # ── [3] Final fit on full data ────────────────────────────────────────────
    logging.info("\n[3/4] Fitting final EBM on full dataset…")
    ebm = _make_ebm(seed)
    ebm.fit(X, y)
    logging.info("  Final EBM fitted.")

    # ── [4] Interpretation ────────────────────────────────────────────────────
    logging.info("\n[4/4] Extracting interpretation…")

    # --- Global importance table --------------------------------------------
    df_imp = extract_global_importance(ebm, feature_names, le)

    imp_path = output_dir / f"ebm_global_importance_{timestamp}.csv"
    df_imp.to_csv(imp_path, index=False)
    logging.info(f"  Global importance → {imp_path.name}")
    logging.info(
        f"\n  Top 10 features by weighted importance:\n"
        + df_imp[["feature", "global_weighted_avg"]]
          .head(10)
          .to_string(index=False)
    )

    # --- Global importance heatmap ------------------------------------------
    heatmap_path = output_dir / f"ebm_importance_heatmap_{timestamp}.png"
    plot_global_importance_heatmap(df_imp, le, heatmap_path, top_n=min(20, len(df_imp)))

    # --- Shape function plots ------------------------------------------------
    # Decide which features to plot
    if args.feature:
        features_to_plot = [args.feature]
    else:
        features_to_plot = df_imp["feature"].head(args.top_n).tolist()

    logging.info(f"\n  Plotting shape functions for {len(features_to_plot)} feature(s)…")
    for feat in features_to_plot:
        try:
            shape_data = extract_shape_function(ebm, feat, feature_names, le)
        except ValueError as e:
            logging.warning(f"  Skipping '{feat}': {e}")
            continue

        safe_name = feat.replace("/", "_").replace(" ", "_")
        plot_path = output_dir / f"ebm_shape_{safe_name}_{timestamp}.png"
        plot_shape_function(shape_data, feat, plot_path)

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
