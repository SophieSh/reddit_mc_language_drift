"""
Step 12 — ML Phase Prediction Pipeline
========================================
Predicts menstrual cycle phase (Menstrual / Follicular / Ovulation / Luteal) from
daily-aggregated linguistic features, using two complementary models:

  Model 1 — DecisionTreeClassifier (max_depth 3-4)
      Interpretable clinical baseline. Full tree saved as a high-resolution PNG.

  Model 2 — XGBoostClassifier (RandomForest fallback)
      Maximum-accuracy ensemble + SHAP beeswarm plots.

Why daily aggregates, not individual posts?
  The scientific question is: "Does a user's language on a given cycle DAY differ
  by menstrual phase?"  The day is the unit of analysis — not a single post.

  Multiple posts on the same day are correlated replicates of the same underlying
  daily state.  They all share the same phase label anyway (offset_from_cd1 is
  identical), so using them as separate ML samples inflates N without adding
  independent information.  Daily aggregation (step 06) gives one clean,
  de-noised observation per user-day.

  Step 06 already computes per-user z-score normalized features
  ({feature}_zscore columns) via normalize_features_per_user_zscore().
  Step 12 reuses those directly — no re-normalization needed.

Users included
  Only users with a statistically significant detected period from step 07
  (consensus_periods_*.csv / periodicity_results_*.csv).  Users without a
  reliable detected cycle length get unreliable phase labels and are excluded.

Data leakage prevention
  GroupKFold splits are keyed on author (user_id).  Multiple days from the
  same user are correlated — their posts, writing style, and cycle trajectory
  are shared.  A user's days must stay entirely in either train or test.

Pipeline
  [1] Load daily aggregated timeline (step 06 output)
  [2] Build user → cycle_length map (step 07, detected users only)
  [3] Filter to detected-period users, assign phase per user-day
  [4] Select {feature}_zscore columns (already computed in step 06)
  [5] Save labelled dataset → data/interim/ml_labeled_days_*.csv
  [6] Model 1 — Decision Tree (interpretable baseline)
  [7] Model 2 — XGBoost/RF ensemble + confusion matrix
  [8] SHAP explainability

Output (reports/ml/)
  decision_tree_depth{D}_*.png
  classification_report_tree_*.txt
  confusion_matrix_ensemble_*.png
  shap_summary_global_*.png
  shap_summary_{phase}_*.png

Usage
  python scripts/12_ml_phase_prediction.py --config configs/base.yaml
  python scripts/12_ml_phase_prediction.py --config configs/base.yaml --tree-depth 3
  python scripts/12_ml_phase_prediction.py --config configs/base.yaml --skip-shap
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
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import (
    assign_phases_to_timeline,
    compute_user_phase_definitions,
    normalize_features_per_user_zscore,
)
from src.config import load_config
from src.io import find_latest_file, find_periodicity_results, save_with_timestamp

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not installed — falling back to RandomForestClassifier.")

warnings.filterwarnings("ignore", category=UserWarning)

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_daily_aggregated(interim_dir: Path) -> pd.DataFrame:
    """Load the daily-aggregated timeline produced by step 06.

    Row unit: ONE (user, cycle_day) pair.
    Columns include {feature}_mean and {feature}_zscore (per-user z-scores).

    Why step 06 and not step 05?
    Step 05 has individual posts; step 06 averages all posts per user-day.
    The phase is a day-level construct (offset_from_cd1 maps to exactly one
    phase per user-day), so the daily aggregate is the correct ML sample unit.
    """
    path = find_latest_file(interim_dir, "timeline_daily_aggregated_*.csv")
    if path is None:
        raise FileNotFoundError(
            "No timeline_daily_aggregated_*.csv found in data/interim/. "
            "Run scripts/06_aggregate_and_normalize.py first."
        )
    logging.info(f"Loading daily aggregated timeline: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users")
    return df


def build_user_period_map(interim_dir: Path) -> dict[str, float]:
    """Build {user → cycle_length_days} strictly from step-07 periodicity results.

    Only users with a DETECTED (statistically significant) period are included.
    Users without a detected period are excluded — their phase labels would be
    unreliable guesses based on an assumed cycle length.

    Source preference:
      1. consensus_periods_*.csv  with 'consensus_period' column
      2. feature_periodicity_*.csv / periodicity_results_*.csv  (median per user)
    """
    results_path = find_periodicity_results(interim_dir)
    if results_path is None:
        return {}

    logging.info(f"Loading periodicity results: {results_path.name}")
    pdf = pd.read_csv(results_path, encoding="utf-8-sig", low_memory=False)

    # Case 1: consensus file with a single period per user
    if "consensus_period" in pdf.columns and "user" in pdf.columns:
        valid = pdf[pdf["consensus_period"].notna()].copy()
        user_period_map = dict(
            zip(valid["user"].astype(str), valid["consensus_period"].astype(float))
        )
        logging.info(f"  Detected periods for {len(user_period_map):,} users (consensus)")
        return user_period_map

    # Case 2: long-format results — take median period per user across features
    if "user" in pdf.columns and "period" in pdf.columns:
        medians = pdf.groupby("user")["period"].median().reset_index()
        user_period_map = dict(
            zip(medians["user"].astype(str), medians["period"].astype(float))
        )
        logging.info(f"  Detected periods for {len(user_period_map):,} users (median)")
        return user_period_map

    logging.error("Periodicity file missing expected columns (user / consensus_period / period).")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PHASE LABELLING
# ═══════════════════════════════════════════════════════════════════════════════

def label_days_with_phase(df: pd.DataFrame, user_period_map: dict[str, float]) -> pd.DataFrame:
    """Filter to detected-period users and label each user-day with its phase.

    Uses the project's existing adaptive-phase logic so that boundaries are
    biologically consistent with the rest of the pipeline (follicular length
    adapts to each user's detected cycle length).

    Args:
        df: Daily-aggregated DataFrame with 'author' and 'offset_from_cd1'.
        user_period_map: {user → cycle_length_days} for detected-period users.
    """
    if not user_period_map:
        raise RuntimeError(
            "user_period_map is empty. Run scripts/07_run_fft_analysis.py first."
        )

    # Keep only users with a statistically detected cycle
    df = df[df["author"].astype(str).isin(user_period_map)].copy()
    df["author"] = df["author"].astype(str)
    logging.info(
        f"  After filtering to detected-period users: "
        f"{len(df):,} user-days, {df['author'].nunique():,} users"
    )

    # Pre-compute per-user adaptive phase boundaries (4 rows per user)
    user_phase_df = compute_user_phase_definitions(user_period_map)

    # Vectorised phase assignment using offset_from_cd1
    df = assign_phases_to_timeline(
        timeline_df=df,
        user_phase_df=user_phase_df,
        user_col="author",
        time_col="offset_from_cd1",
    )

    before = len(df)
    df = df[df["phase"].notna() & df["phase"].isin(PHASE_ORDER)].copy()
    if before - len(df):
        logging.warning(f"  Dropped {before - len(df):,} user-days with unassignable phase")

    logging.info(f"  Phase distribution (user-day level):\n{df['phase'].value_counts().to_string()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_select_zscore_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-user z-scores from the _mean columns and return the zscore column names.

    Step 06 produces {feature}_mean columns (daily averages per user-day) but does
    not write z-score columns.  We compute them here using
    normalize_features_per_user_zscore() from src/analysis.py, which normalises
    each feature relative to that user's own mean and std across all their days.

    Per-user z-scores remove inter-individual baseline differences so the model
    learns how a user's language deviates from their personal mean across phases,
    not who the person is.

    No PCA or dimensionality reduction is applied so that original feature names
    are preserved in tree decision rules and SHAP plots.
    """
    mean_cols = [
        c for c in df.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not mean_cols:
        raise RuntimeError(
            "No _mean columns found. Check that the input is a step-06 daily-aggregated file."
        )
    logging.info(f"  Computing per-user z-scores for {len(mean_cols)} _mean columns…")
    df = normalize_features_per_user_zscore(df, mean_cols, user_col="author")

    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    logging.info(f"  {len(zscore_cols)} _zscore feature columns ready")
    return df, zscore_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ML ARRAYS AND GROUP-AWARE CV
# ═══════════════════════════════════════════════════════════════════════════════

def make_arrays(df: pd.DataFrame, feature_cols: list[str]):
    """Extract X, y (int-encoded), groups, and the LabelEncoder."""
    # Fill NaN with 0 — z-score of 0 means "at the user's personal mean", safe imputation
    X = df[feature_cols].fillna(0).values.astype(np.float32)

    le = LabelEncoder()
    le.fit(PHASE_ORDER)   # fixed order for stable class indices across runs
    y = le.transform(df["phase"])

    groups = df["author"].values
    return X, y, groups, le


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights for class balancing.

    Phases differ greatly in length (Luteal ~14 days vs Ovulation ~3 days),
    so the dataset is structurally imbalanced.  RandomForest accepts
    class_weight="balanced" directly; XGBoost requires explicit sample_weight
    passed to fit().  This function produces per-sample weights that give each
    class equal total weight regardless of how many days it contains.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def oof_predictions(model, X, y, groups, n_folds: int) -> np.ndarray:
    """Out-of-fold predictions with GroupKFold on author.

    All user-days from the same user stay entirely in either train or test.
    This is mandatory because days from the same user share the same writing
    style, cycle trajectory, and vocabulary — they are not independent.

    We implement the fold loop manually (rather than cross_val_predict) so we
    can pass sample_weight directly to fit() on each fold — sklearn 1.6+ removed
    the fit_params argument from cross_val_predict in favour of metadata routing.
    """
    gkf = GroupKFold(n_splits=n_folds)
    sample_weights = compute_sample_weights(y)
    y_pred = np.empty_like(y)

    for train_idx, test_idx in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])

    return y_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL 1 — DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_tree(X, y, groups, feature_names, label_encoder, n_folds, max_depth, output_dir, timestamp):
    logging.info(f"\n{'='*60}")
    logging.info(f"MODEL 1 — DecisionTree (max_depth={max_depth})")
    logging.info(f"{'='*60}")

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
    )
    y_pred = oof_predictions(tree, X, y, groups, n_folds)

    report = classification_report(y, y_pred, target_names=label_encoder.classes_, digits=3)
    logging.info(f"\nOOF Report:\n{report}")

    rpt_path = output_dir / f"classification_report_tree_depth{max_depth}_{timestamp}.txt"
    rpt_path.write_text(
        f"DecisionTree (max_depth={max_depth}) | GroupKFold OOF ({n_folds} folds)\n\n{report}"
    )

    # Fit on full data to visualise the complete clinical rule set
    tree.fit(X, y)

    fig_w = max(20, 2 ** max_depth * 3)
    fig_h = max(10, max_depth * 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(
        f"Decision Tree (max_depth={max_depth}) — Menstrual Phase Classifier\n"
        f"GroupKFold OOF Macro-F1 = {_macro_f1(y, y_pred):.3f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / f"decision_tree_depth{max_depth}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Tree plot → {plot_path.name}")
    return tree


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL 2 — ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_ensemble(n_classes: int):
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            objective="multi:softmax",
            num_class=n_classes,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp):
    model_name = "XGBoostClassifier" if XGBOOST_AVAILABLE else "RandomForestClassifier"
    logging.info(f"\n{'='*60}")
    logging.info(f"MODEL 2 — {model_name}")
    logging.info(f"{'='*60}")

    n_classes = len(label_encoder.classes_)
    model = build_ensemble(n_classes)
    y_pred = oof_predictions(model, X, y, groups, n_folds)

    report = classification_report(y, y_pred, target_names=label_encoder.classes_, digits=3)
    logging.info(f"\nOOF Report:\n{report}")

    rpt_path = output_dir / f"classification_report_ensemble_{timestamp}.txt"
    rpt_path.write_text(f"{model_name} | GroupKFold OOF ({n_folds} folds)\n\n{report}")

    # Confusion matrix — shows which phases the model confuses most
    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title(
        f"{model_name} — Confusion Matrix\n"
        f"GroupKFold OOF ({n_folds} folds) | Macro-F1 = {_macro_f1(y, y_pred):.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    cm_path = output_dir / f"confusion_matrix_ensemble_{timestamp}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Confusion matrix → {cm_path.name}")

    # Refit on full dataset for SHAP (with same sample weights for consistency)
    logging.info("  Fitting on full dataset for SHAP…")
    model.fit(X, y, sample_weight=compute_sample_weights(y))
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(model, X, feature_names, label_encoder, output_dir, timestamp, max_display=20):
    logging.info("\n  Computing SHAP values (TreeExplainer)…")

    n_shap = min(len(X), 5_000)
    if n_shap < len(X):
        idx = np.random.default_rng(42).choice(len(X), size=n_shap, replace=False)
        X_shap = X[idx]
    else:
        X_shap = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Normalise to (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
    elif np.asarray(shap_values).ndim == 3:
        shap_array = np.asarray(shap_values)
    else:
        shap_array = np.asarray(shap_values)[:, :, np.newaxis]

    n_classes = shap_array.shape[2]

    # Global bar chart: stacked mean |SHAP| per phase.
    # Must use plot_type="bar" with a list of per-class SHAP arrays.
    # Using plot_type="dot" + np.abs() here would destroy directionality
    # (all dots collapse to the right of zero) — that is a known SHAP misuse.
    shap_list = [shap_array[:, :, i] for i in range(n_classes)]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_list,
        X_shap,
        feature_names=feature_names,
        class_names=list(label_encoder.classes_),
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.gca().set_title(
        "SHAP Global Feature Importance (stacked by phase)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_summary_global_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("  Global SHAP bar chart saved")

    # Per-phase beeswarm — shows directional effect for each phase
    for cls_idx in range(n_classes):
        phase_name = label_encoder.classes_[cls_idx]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_array[:, :, cls_idx],
            X_shap,
            feature_names=feature_names,
            plot_type="dot", show=False, max_display=max_display,
        )
        plt.gca().set_title(
            f"SHAP Feature Impact — {phase_name} Phase\n"
            f"(positive SHAP → higher P({phase_name}))",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_summary_{phase_name.lower()}_{timestamp}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        logging.info(f"  SHAP [{phase_name}] saved")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--tree-depth", type=int, default=4, choices=[3, 4, 5])
    p.add_argument("--n-folds", type=int, default=5,
                   help="GroupKFold splits — each fold holds out all days of some users.")
    p.add_argument("--shap-max-display", type=int, default=20)
    p.add_argument("--skip-shap", action="store_true")
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
    interim_dir = Path(cfg["paths"]["interim"])

    output_dir = ROOT / "reports" / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load daily aggregated timeline (step 06) ───────────────────────────
    logging.info("\n[1/7] Loading daily aggregated timeline (step 06)…")
    df = load_daily_aggregated(interim_dir)

    # ── [2] Build user → period map (step 07, detected users only) ─────────────
    logging.info("\n[2/7] Loading detected cycle lengths (step 07)…")
    user_period_map = build_user_period_map(interim_dir)
    if not user_period_map:
        logging.error("No periodicity results found. Run scripts/07_run_fft_analysis.py first.")
        sys.exit(1)

    # ── [3] Compute per-user z-scores on the FULL timeline BEFORE any filtering ─
    # Must happen before label_days_with_phase() so that each user's mean/std is
    # computed over their complete set of days, not a phase-filtered subset.
    logging.info("\n[3/7] Computing per-user z-score features from _mean columns…")
    df, zscore_cols = compute_and_select_zscore_features(df)

    # ── [4] Filter to detected users and assign phase per user-day ─────────────
    logging.info("\n[4/7] Filtering to detected-period users and assigning phases…")
    df = label_days_with_phase(df, user_period_map)

    # ── [5] Save labelled daily dataset ────────────────────────────────────────
    logging.info("\n[5/7] Saving labelled user-day dataset…")
    save_cols = ["author", "offset_from_cd1", "phase"] + zscore_cols
    for extra in ["subreddit", "ts_date"]:
        if extra in df.columns:
            save_cols.insert(3, extra)
    saved_path = save_with_timestamp(
        df[[c for c in save_cols if c in df.columns]],
        interim_dir,
        "ml_labeled_days",
    )
    logging.info(f"  Saved → {saved_path}")

    # ── Build ML arrays ─────────────────────────────────────────────────────────
    X, y, groups, label_encoder = make_arrays(df, zscore_cols)
    n_unique_users = len(np.unique(groups))
    median_days = int(np.median(np.unique(groups, return_counts=True)[1]))
    logging.info(
        f"\n  X shape: {X.shape} | classes: {list(label_encoder.classes_)} | "
        f"users: {n_unique_users} | median days/user: {median_days}"
    )

    if n_unique_users < args.n_folds:
        logging.warning(f"Only {n_unique_users} users — reducing n_folds to {n_unique_users}")
        args.n_folds = n_unique_users

    # ── [6] Model 1 — Decision Tree ────────────────────────────────────────────
    logging.info("\n[6/7] Decision Tree baseline…")
    train_and_evaluate_tree(
        X, y, groups, zscore_cols, label_encoder,
        args.n_folds, args.tree_depth, output_dir, timestamp,
    )

    # ── [7] Model 2 — Ensemble + SHAP ──────────────────────────────────────────
    logging.info("\n[7/7] Ensemble model…")
    ensemble = train_and_evaluate_ensemble(
        X, y, groups, zscore_cols, label_encoder,
        args.n_folds, output_dir, timestamp,
    )

    if not args.skip_shap:
        run_shap_analysis(
            ensemble, X, zscore_cols, label_encoder,
            output_dir, timestamp, args.shap_max_display,
        )
    else:
        logging.info("  SHAP skipped.")

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
