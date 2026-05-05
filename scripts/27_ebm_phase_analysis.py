"""
Step 27 — Explainable Boosting Machine (EBM) Phase Analysis
=============================================================
Goal: Feature Discovery — identify which linguistic features are characteristic
of each menstrual cycle phase, and in which *direction* they move.

Why EBM over XGBoost for this goal?
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
  [1] Load daily-aggregated timeline (step 06) and build ML arrays via the same
      logic as script 12 (z-scores, phase labels, phase-profile aggregation).
  [2] GroupKFold (5 folds, grouped by author) OOF evaluation.
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
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import (
    assign_phases_to_timeline,
    compute_user_phase_definitions,
    normalize_features_per_user_zscore,
)
from src.config import load_config
from src.io import find_latest_file, find_periodicity_results, save_with_timestamp

warnings.filterwarnings("ignore", category=UserWarning)

# Class order must be consistent with LabelEncoder throughout the script.
# LabelEncoder sorts alphabetically, so this is the canonical mapping:
#   0 = Follicular, 1 = Luteal, 2 = Menstrual, 3 = Ovulation
PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_COLORS = {
    "Follicular": "#4C9BE8",   # blue
    "Luteal":     "#E8884C",   # orange
    "Menstrual":  "#E84C4C",   # red
    "Ovulation":  "#4CE89B",   # green
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (same logic as script 12 — kept here so this script is
# self-contained; replace the body of load_ml_arrays() with your own loader
# if you already have a pre-built DataFrame)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ml_arrays(cfg: dict, anchors: str = "with") -> tuple[pd.DataFrame, list[str]]:
    """Load the step-06 daily-aggregated timeline and return a phase-labelled
    DataFrame together with the list of z-score feature column names.

    ── PLACEHOLDER ────────────────────────────────────────────────────────────
    If you already have a DataFrame ready (e.g. loaded from a CSV), replace
    the body of this function with:

        df = pd.read_csv("path/to/your/labeled_data.csv")
        zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
        return df, zscore_cols

    The returned df must have columns:
        - "author"         : user ID (str)
        - "phase"          : one of PHASE_ORDER strings
        - *_zscore columns : float, per-user z-scored linguistic features
    ───────────────────────────────────────────────────────────────────────────
    """
    interim_dir = Path(cfg["paths"]["interim"])

    # ── Load step-06 output ──────────────────────────────────────────────────
    pattern_map = {
        "with":    "timeline_daily_aggregated_with_anchors_*.csv",
        "without": "timeline_daily_aggregated_no_anchors_*.csv",
        "any":     "timeline_daily_aggregated_*.csv",
    }
    path = find_latest_file(interim_dir, pattern_map.get(anchors, pattern_map["with"]))
    if path is None:
        path = find_latest_file(interim_dir, "timeline_daily_aggregated_*.csv")
    if path is None:
        raise FileNotFoundError(
            "No timeline_daily_aggregated_*.csv found. Run script 06 first."
        )
    logging.info(f"Loading timeline: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days, {df['author'].nunique():,} users")

    # ── Compute per-user z-scores from the raw _mean columns ─────────────────
    # Must happen BEFORE filtering to detected-period users so each user's
    # mean/std is computed over their complete history.
    mean_cols = [
        c for c in df.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = normalize_features_per_user_zscore(df, mean_cols, user_col="author")
    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    logging.info(f"  {len(zscore_cols)} z-score features computed")

    # ── Build user → cycle-length map from step-07 consensus ─────────────────
    results_path = find_periodicity_results(interim_dir)
    if results_path is None:
        raise FileNotFoundError(
            "No periodicity / consensus CSV found. Run script 07/08 first."
        )
    pdf = pd.read_csv(results_path, encoding="utf-8-sig", low_memory=False)
    if "consensus_period" in pdf.columns:
        user_period_map = dict(
            zip(pdf["user"].astype(str), pdf["consensus_period"].astype(float))
        )
    elif "period" in pdf.columns:
        user_period_map = dict(
            pdf.groupby("user")["period"].median().astype(float)
        )
    else:
        raise ValueError("Periodicity file missing 'consensus_period' or 'period' column.")
    logging.info(f"  {len(user_period_map):,} users with detected cycles")

    # ── Filter to detected-period users and assign phase labels ──────────────
    df = df[df["author"].astype(str).isin(user_period_map)].copy()
    df["author"] = df["author"].astype(str)
    user_phase_df = compute_user_phase_definitions(
        {u: p for u, p in user_period_map.items() if u in set(df["author"])}
    )
    df = assign_phases_to_timeline(
        timeline_df=df,
        user_phase_df=user_phase_df,
        user_col="author",
        time_col="offset_from_cd1",
    )
    df = df[df["phase"].notna() & df["phase"].isin(PHASE_ORDER)].copy()
    logging.info(f"  After phase labelling: {len(df):,} user-days")
    logging.info(f"  Phase counts:\n{df['phase'].value_counts().to_string()}")

    # ── Aggregate to one feature vector per (user, phase) ────────────────────
    # Each sample = a user's *mean* profile across all days in one phase.
    # This removes within-phase day-to-day noise and matches the bar-chart view.
    profiles = (
        df.groupby(["author", "phase"])[zscore_cols]
        .mean()
        .reset_index()
    )
    profiles[zscore_cols] = profiles[zscore_cols].fillna(0)
    logging.info(
        f"  Phase profiles: {len(profiles):,} rows "
        f"({profiles['author'].nunique():,} users × up to 4 phases)"
    )
    return profiles, zscore_cols


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_encoder: LabelEncoder,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold OOF evaluation of an EBM.

    Why GroupKFold?
      All phase-profiles from the same user (up to 4 rows) share the same writing
      style, baseline vocabulary, and cycle trajectory.  They are NOT independent
      samples.  If a user's Follicular row appears in train and their Luteal row
      in test, the model has already "seen" that person — this inflates AUC.
      GroupKFold ensures every row for a given author stays in the same fold.

    Returns:
        y_pred  : (n_samples,) integer hard predictions (OOF)
        y_proba : (n_samples, n_classes) probability matrix (OOF)
    """
    gkf = GroupKFold(n_splits=n_splits)
    n_classes = len(label_encoder.classes_)
    y_pred  = np.empty_like(y)
    y_proba = np.zeros((len(y), n_classes), dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        logging.info(f"  Fold {fold}/{n_splits} — "
                     f"train={len(train_idx):,} test={len(test_idx):,}")

        # EBM is fitted fresh on each fold (no warm start needed).
        # n_jobs=-1 parallelises the boosting rounds across features.
        ebm_fold = ExplainableBoostingClassifier(
            max_bins=256,           # resolution of shape functions
            max_interaction_bins=64,
            interactions=0,         # pure additive model — easier to interpret
            learning_rate=0.01,
            max_rounds=5000,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        ebm_fold.fit(X[train_idx], y[train_idx])

        y_pred[test_idx]  = ebm_fold.predict(X[test_idx])
        y_proba[test_idx] = ebm_fold.predict_proba(X[test_idx])

    return y_pred, y_proba


def log_oof_metrics(
    y: np.ndarray,
    y_proba: np.ndarray,
    label_encoder: LabelEncoder,
) -> str:
    """Compute and log macro AUC + per-class AUC.  Returns a formatted string."""
    lines = []
    macro_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    lines.append(f"OOF Macro AUC (OvR): {macro_auc:.4f}  [random = 0.500]")
    lines.append("")
    lines.append(f"{'Phase':<14}  {'AUC':>6}  {'Positives':>10}")
    lines.append("-" * 36)
    for i, cls in enumerate(label_encoder.classes_):
        y_bin = (y == i).astype(int)
        cls_auc = roc_auc_score(y_bin, y_proba[:, i])
        lines.append(f"{cls:<14}  {cls_auc:.4f}  {y_bin.sum():>10,}")

    report = "\n".join(lines)
    logging.info("\n" + report)
    return report


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
            if cls_idx < len(per_class_importance):
                row[cls_name] = float(per_class_importance[cls_idx])
            else:
                row[cls_name] = 0.0
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

    clean_name = feature_name.replace("_zscore", "").replace("_", " ")

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
    feature_labels = [
        f.replace("_zscore", "").replace("_", " ") for f in top["feature"]
    ]

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
        "--anchors", choices=["with", "without", "any"], default="with",
        help="Which timeline variant to use (default: with_anchors).",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of GroupKFold splits.",
    )
    p.add_argument(
        "--feature", default=None,
        help="Name of a specific feature to plot the shape function for "
             "(e.g. 'valence_dict_average_zscore'). If omitted, top --top-n are plotted.",
    )
    p.add_argument(
        "--top-n", type=int, default=10,
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

    output_dir = ROOT / "reports" / "ebm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    logging.info("\n[1/4] Loading and labelling data…")
    df, zscore_cols = load_ml_arrays(cfg, anchors=args.anchors)

    # Build numpy arrays — these are the X / y / users the model trains on.
    le = LabelEncoder()
    le.fit(PHASE_ORDER)          # alphabetical: Follicular=0, Luteal=1, Menstrual=2, Ovulation=3
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
        y_pred, y_proba = run_group_kfold_cv(X, y, groups, le, n_splits=args.n_folds)

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
    ebm = ExplainableBoostingClassifier(
        max_bins=256,
        max_interaction_bins=64,
        interactions=0,       # pure additive — shape functions are per-feature only
        learning_rate=0.01,
        max_rounds=5000,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
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

        # ── HOW TO ACCESS THE RAW DATA ──────────────────────────────────────
        # shape_data is a dict: {phase_name: (x_midpoints, log_odds_scores)}
        #
        # Example — print the Menstrual shape for this feature:
        #   x, y_logodds = shape_data["Menstrual"]
        #   for xi, yi in zip(x, y_logodds):
        #       print(f"  z={xi:+.3f}  →  log-odds {yi:+.4f}")
        #
        # To convert log-odds to an odds multiplier:
        #   odds_multiplier = np.exp(yi)
        #   # e.g. 0.8 log-odds → exp(0.8) = 2.2× more likely to be Menstrual
        #
        # To find the threshold where the feature starts pushing toward a phase:
        #   positive_bins = x[y_logodds > 0]
        #   if len(positive_bins): print(f"  Pushes toward Menstrual when z > {positive_bins[0]:.2f}")
        # ────────────────────────────────────────────────────────────────────

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
