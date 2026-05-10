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
      which matters because Follicular has more days than Ovulation.

    Returns:
        y_pred  : (n_samples,) integer hard predictions (OOF)
        y_proba : (n_samples, n_classes) probability matrix (OOF)
    """
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_classes = len(label_encoder.classes_)
    y_pred  = np.empty_like(y)
    y_proba = np.zeros((len(y), n_classes), dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        logging.info(f"  Fold {fold}/{n_splits} — "
                     f"train={len(train_idx):,} test={len(test_idx):,}")

        # EBM is fitted fresh on each fold (no warm start needed).
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

def extract_global_importance(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Extract the global feature importance table from a fitted EBM.

    For a multiclass EBM, ebm.explain_global() returns one importance score per
    feature *per class*.  The score is the mean absolute log-odds contribution
    of that feature across all training samples for that class.

    Higher score → the feature's shape function makes larger swings for that
    class, i.e. the feature is more *discriminative* for that class.

    Returns a DataFrame with columns:
        feature, Follicular, Luteal, Menstrual, Ovulation, mean_importance
    """
    global_exp = ebm.explain_global()

    # global_exp.data(i) returns a dict for feature i with keys:
    #   "names"      : list of bin-edge labels (strings)
    #   "scores"     : per-class log-odds array — shape (n_bins, n_classes)
    #                  or (n_bins,) for binary
    #   "type"       : "univariate"
    #
    # The *importance* score shown in the dashboard is simply the weighted mean
    # of |scores| across bins (weighted by bin density).  We replicate that here
    # so we can sort and export it.

    rows = []
    n_classes = len(label_encoder.classes_)

    # We recompute importance from raw scores rather than using ebm.term_importances_
    # because we need per-class (per-phase) breakdowns, not just global importance.
    for feat_idx, feat_name in enumerate(feature_names):
        data = global_exp.data(feat_idx)
        scores = np.asarray(data["scores"])  # (n_bins, n_classes) for multiclass

        if scores.ndim == 1:
            # Binary or single-class edge case — wrap to 2D
            scores = scores[:, np.newaxis]

        # Mean absolute log-odds per class = overall discriminative power
        # The last bin is usually an "out-of-range" catch-all; include it.
        per_class_importance = np.abs(scores).mean(axis=0)  # shape: (n_classes,)

        row = {"feature": feat_name}
        for cls_idx, cls_name in enumerate(label_encoder.classes_):
            row[cls_name] = float(per_class_importance[cls_idx])
        row["mean_importance"] = float(np.mean(list(row[c] for c in label_encoder.classes_)))
        rows.append(row)

    df_imp = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    return df_imp

def extract_shape_function(
    ebm: ExplainableBoostingClassifier,
    feature_name: str,
    feature_names: list[str],
    label_encoder: LabelEncoder,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract the bin edges and log-odds scores for one feature.

    This is the core of EBM interpretability.  For each class, you get:
        x  = bin midpoints (feature value, in z-score units relative to user mean)
        y  = log-odds contribution at that value

    A positive y at x=1.5 for "Menstrual" means: a user who scores 1.5 SD above
    their personal mean on this feature has that log-odds added to their Menstrual
    score, pushing them toward being classified as Menstrual.

    Returns:
        {class_name: (x_midpoints, log_odds_scores)}

    Note: ebm.explain_global() returns a custom InterpretML object, not numpy arrays.
    We extract bin edges and scores manually so callers get standard arrays for
    saving, plotting, or further analysis.
    """
    if feature_name not in feature_names:
        raise ValueError(
            f"'{feature_name}' not found. Available: {feature_names[:5]} …"
        )
    feat_idx = feature_names.index(feature_name)
    global_exp = ebm.explain_global()
    data = global_exp.data(feat_idx)

    # "names" are bin-edge strings like "-1.23 to 0.45".
    # We parse the left edge of each bin to get a numeric x-axis.
    bin_labels = data["names"]
    scores     = np.asarray(data["scores"])  # (n_bins, n_classes)

    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    # Parse left edge of each bin label as a float x value.
    # Bin labels produced by EBM look like "-1.23 to 0.45" or "> 2.1".
    x_vals = []
    for label in bin_labels:
        label = str(label)
        try:
            # "a to b" format — take midpoint of the bin
            if " to " in label:
                parts = label.split(" to ")
                x_vals.append((float(parts[0]) + float(parts[1])) / 2)
            elif label.startswith(">"):
                x_vals.append(float(label.replace(">", "").strip()))
            elif label.startswith("<="):
                x_vals.append(float(label.replace("<=", "").strip()))
            else:
                x_vals.append(float(label))
        except ValueError:
            x_vals.append(np.nan)

    x_arr = np.array(x_vals, dtype=np.float64)

    # EBM sometimes returns one more bin label than score rows (the final
    # "catch-all" bin has a label but no separate score entry).  Trim to the
    # shorter length so x and y always align.
    n = min(len(x_arr), scores.shape[0])
    x_arr = x_arr[:n]
    scores = scores[:n]

    result = {}
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        if cls_idx < scores.shape[1]:
            result[cls_name] = (x_arr, scores[:, cls_idx])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def plot_shape_function(
    shape_data: dict[str, tuple[np.ndarray, np.ndarray]],
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot the EBM shape function (log-odds vs feature value) for all 4 phases.

    The x-axis is the z-score of the feature relative to the user's personal mean.
      x = 0   → user is at their own mean (no deviation)
      x = 1.5 → user is 1.5 SD above their mean for this feature
      x = -1  → user is 1 SD below their mean

    The y-axis is the log-odds contribution.
      y > 0 → pushes prediction toward this phase
      y < 0 → pushes prediction away from this phase
      y = 0 → feature is uninformative for this phase at this value
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    clean_name = _clean_feat_name(feature_name)

    for ax, (phase, (x, y)) in zip(axes, shape_data.items()):
        color = PHASE_COLORS.get(phase, "gray")

        # Remove NaN x entries before plotting
        valid = ~np.isnan(x)
        x_v, y_v = x[valid], y[valid]

        ax.step(x_v, y_v, where="mid", color=color, linewidth=2)
        ax.fill_between(x_v, y_v, 0, step="mid", alpha=0.25, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.6, linestyle=":")

        # Mark the most extreme positive/negative bin
        if len(y_v) > 0:
            peak_idx = np.argmax(np.abs(y_v))
            ax.scatter(
                x_v[peak_idx], y_v[peak_idx],
                color=color, s=60, zorder=5, edgecolors="black", linewidths=0.8,
            )

        ax.set_title(f"{phase}", fontsize=12, fontweight="bold", color=color)
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
    phase_cols = list(label_encoder.classes_)

    matrix = top[phase_cols].values  # (top_n, n_classes)
    feature_labels = [_clean_feat_name(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(phase_cols)))
    ax.set_xticklabels(phase_cols, fontsize=11, fontweight="bold")
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
    df, zscore_cols, le = load_phase_labeled_dataset(cfg, no_anchors=args.no_anchors)

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
        f"\n  Top 10 features by mean importance:\n"
        + df_imp[["feature", "mean_importance"] + list(le.classes_)]
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
