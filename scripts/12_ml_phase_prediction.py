"""
Step 12 — ML Phase Prediction Pipeline
========================================
Predicts menstrual cycle phase (Menstrual / Follicular / Ovulation / Luteal) from
daily-aggregated linguistic features using XGBoostClassifier.

Why daily aggregates, not individual posts?
  The scientific question is: "Does a user's language on a given cycle DAY differ
  by menstrual phase?"  The day is the unit of analysis — not a single post.

  Multiple posts on the same day are correlated replicates of the same underlying
  daily state.  They all share the same phase label anyway (offset_from_cd1 is
  identical), so using them as separate ML samples inflates N without adding
  independent information.  Daily aggregation (step 06) gives one clean,
  de-noised observation per user-day.

  Step 06 produces {feature}_mean columns (daily averages per user-day).
  Step 12 computes per-user z-score normalized features from those _mean columns
  via normalize_features_per_user_zscore(), which normalises each feature relative
  to that user's own mean and std across all their days.

Users included
  Only users with a statistically significant detected period from step 08
  (consensus_periods_*.csv).  Phase labeling is done once in step 08b and saved
  as timeline_phase_labeled_*.csv.  Run scripts/08b_label_phases.py first.

Data leakage prevention
  GroupKFold splits are keyed on author (user_id).  Multiple days from the
  same user are correlated — their posts, writing style, and cycle trajectory
  are shared.  A user's days must stay entirely in either train or test.

Pipeline
  [1] Load phase-labeled timeline (step 08b output)
  [2] Compute per-user z-scores from _mean columns
  [3] Optionally aggregate to per-user phase profiles (--aggregate-by-phase)
  [4] Save labelled dataset → data/interim/ml_labeled_days_*.csv
  [5] XGBoost ensemble + confusion matrix + SHAP explainability
  [6] Optional OvR ensemble with per-phase AUC

Output (reports/ml/)
  confusion_matrix_ensemble_*.png
  classification_report_ensemble_*.txt
  shap_summary_global_*.png
  shap_summary_{phase}_*.png
  ovr_phase_auc_*.png (with --ovr)

Usage
  python scripts/12_ml_phase_prediction.py --config configs/base.yaml
  python scripts/12_ml_phase_prediction.py --config configs/base.yaml --skip-shap

Recommended mode (best AUC):
  python scripts/12_ml_phase_prediction.py --aggregate-by-phase
  → Macro AUC ~0.75 (OvR ensemble): Menstrual=0.77, Ovulation=0.79, Luteal=0.84

Confirmed null results (do not re-investigate):
  Luteal split (EarlyLuteal/LateLuteal): both AUC near chance (~0.56).
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import (
    aggregate_to_phase_profiles,
    normalize_features_per_user_zscore,
)
from src.config import load_config
from src.io import find_latest_file, save_with_timestamp

warnings.filterwarnings("ignore", category=UserWarning)

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_phase_labeled(interim_dir: Path, pattern: str) -> pd.DataFrame:
    """Load the phase-labeled daily timeline produced by step 08b.

    Row unit: ONE (user, cycle_day) pair, already filtered to detected-period
    users and annotated with 'phase' (Menstrual/Follicular/Ovulation/Luteal).

    Prerequisite: run scripts/08b_label_phases.py before this script.
    """
    path = find_latest_file(interim_dir, pattern)
    if path is None:
        raise FileNotFoundError(
            f"No {pattern} found in data/interim/. "
            "Run scripts/08b_label_phases.py first."
        )
    logging.info(f"Loading phase-labeled timeline: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users")
    if "phase" not in df.columns:
        raise ValueError("Loaded file is missing 'phase' column. Re-run 08b_label_phases.py.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_zscore_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-user z-scores from the _mean columns and return the zscore column names.

    Step 06 produces {feature}_mean columns (daily averages per user-day) but does
    not write z-score columns.  We compute them here via
    normalize_features_per_user_zscore(), which normalises each feature relative
    to that user's own mean and std across all their days.

    Per-user z-scores remove inter-individual baseline differences so the model
    learns how a user's language deviates from their personal mean across phases,
    not who the person is.

    No dimensionality reduction is applied so that original feature names are
    preserved in SHAP plots.
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
# 3. ML ARRAYS AND GROUP-AWARE CV
# ═══════════════════════════════════════════════════════════════════════════════

def make_arrays(df: pd.DataFrame, feature_cols: list[str], binary_phase: str = ""):
    """Convert labeled DataFrame to (X, y_int, groups, LabelEncoder) for sklearn.

    groups = author ids, passed to GroupKFold so that all days from one user
    land entirely on the same side of every train/test split — preventing
    within-user correlation from leaking across folds.

    Args:
        binary_phase: If non-empty (e.g. "Menstrual"), collapse labels to
            1 = target phase, 0 = all other phases.  The label encoder will
            have classes ["Other", binary_phase] so class 1 is always the target.
    """
    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if binary_phase:
        binary_labels = df["phase"].apply(lambda p: binary_phase if p == binary_phase else "Other")
        le = LabelEncoder()
        le.fit(["Other", binary_phase])   # 0=Other, 1=target
        y = le.transform(binary_labels)
        logging.info(
            f"  Binary mode: '{binary_phase}' (1) vs rest (0) — "
            f"positives: {y.sum():,} / {len(y):,} ({100*y.mean():.1f}%)"
        )
    else:
        le = LabelEncoder()
        le.fit(PHASE_ORDER)
        y = le.transform(df["phase"])

    groups = df["author"].values
    return X, y, groups, le

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights for class balancing.

    Phases are structurally imbalanced: Luteal lasts ~14 days per cycle,
    Ovulation only ~3 days.  Unweighted training would bias the model toward
    the Luteal class.  These weights give each class equal total gradient
    influence during XGBoost training, regardless of how many days it contains.
    (XGBoost requires explicit sample_weight at fit() time; it cannot use
    class_weight="balanced" like sklearn estimators.)
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)

def oof_predictions(model, X, y, groups, n_folds: int) -> np.ndarray:
    """Out-of-fold predictions with GroupKFold on author.

    All user-days from the same user stay entirely in either train or test.
    This is mandatory because days from the same user share the same writing
    style, cycle trajectory, and vocabulary — they are not independent samples.

    The fold loop is implemented manually (rather than cross_val_predict) so
    sample_weight can be passed to fit() on each fold.  sklearn 1.6+ removed
    the fit_params argument from cross_val_predict in favour of metadata routing,
    which XGBoost does not support.
    """
    gkf = GroupKFold(n_splits=n_folds)
    sample_weights = compute_sample_weights(y)
    y_pred = np.empty_like(y)
    n_classes = len(np.unique(y))
    y_proba = np.zeros((len(y), n_classes))

    for train_idx, test_idx in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])
        if hasattr(m, "predict_proba"):
            y_proba[test_idx] = m.predict_proba(X[test_idx])

    return y_pred, y_proba

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL — XGBOOST ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_ensemble(n_classes: int) -> XGBClassifier:
    """Build a multiclass XGBoost classifier.

    XGBoost outperforms RandomForest on this task (Macro AUC ~0.75 vs lower RF
    baseline) and is the only supported model.  Install xgboost if missing.
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def train_and_evaluate_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp):
    logging.info(f"\n{'='*60}")
    logging.info("MODEL — XGBoostClassifier")
    logging.info(f"{'='*60}")

    n_classes = len(label_encoder.classes_)
    model = build_ensemble(n_classes)
    y_pred, y_proba = oof_predictions(model, X, y, groups, n_folds)

    report = classification_report(y, y_pred, target_names=label_encoder.classes_, digits=3)
    logging.info(f"\nOOF Report:\n{report}")
    _log_auc(y, y_proba, label_encoder)

    rpt_path = output_dir / f"classification_report_ensemble_{timestamp}.txt"
    rpt_path.write_text(f"XGBoostClassifier | GroupKFold OOF ({n_folds} folds)\n\n{report}")

    # Confusion matrix — shows which phases the model confuses most
    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title(
        f"XGBoostClassifier — Confusion Matrix\n"
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
# 6b. OVR ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def load_consistency_map(
    reports_dir: Path,
    threshold: float,
    feature_names: list[str],
    phases: list[str],
    min_features: int = 3,
) -> dict[str, list[int]]:
    """Load phase_peak_summary CSV (script 10 output) and return
    {phase → [col_indices_in_feature_names]} for features where
    pct_consistent >= threshold in that phase.

    feature_names are zscore column names like 'readability_zscore'.
    CSV has base names like 'readability'. Mapping: strip '_zscore' suffix.

    Falls back to all features for a phase if fewer than min_features pass.
    """
    candidates = sorted((reports_dir / "phase_distributions").glob("phase_peak_summary_*.csv"))
    if not candidates:
        logging.warning("  No phase_peak_summary_*.csv found — consistency filter disabled.")
        return {}

    csv_path = candidates[-1]
    logging.info(f"  Loading consistency map from: {csv_path.name}")
    summary = pd.read_csv(csv_path)

    # Build lookup: base_name → row
    summary_map = {row["feature"]: row for _, row in summary.iterrows()}

    # Strip _zscore suffix from feature_names to get base names
    def base_name(fn: str) -> str:
        return fn.replace("_zscore", "").replace("_mean", "")

    consistency_map = {}
    for phase in phases:
        col = f"{phase}_pct_consistent"
        if col not in summary.columns:
            logging.warning(f"  {phase}: column '{col}' not in summary — skipping")
            continue

        indices = []
        for idx, fn in enumerate(feature_names):
            bn = base_name(fn)
            if bn in summary_map:
                pct = summary_map[bn].get(col, float("nan"))
                if pd.notna(pct) and pct >= threshold:
                    indices.append(idx)

        if len(indices) < min_features:
            logging.warning(
                f"  {phase}: only {len(indices)} features >= {threshold:.0f}% "
                f"(need {min_features}) — using all {len(feature_names)} features"
            )
            indices = list(range(len(feature_names)))

        consistency_map[phase] = indices
        feat_labels = [feature_names[i] for i in indices[:5]]
        logging.info(
            f"  {phase}: {len(indices)} consistent features >= {threshold:.0f}%  "
            f"(top: {', '.join(feat_labels)}{'…' if len(indices) > 5 else ''})"
        )

    return consistency_map


def train_ovr_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp,
                       consistency_map: dict | None = None):
    """Train one binary XGBoost per phase (OvR) with shared GroupKFold splits.

    All classifiers use the exact same fold splits so the OOF probability
    vectors are aligned and can be stacked into a proper (n_samples, n_phases)
    matrix for multiclass AUC computation.

    Returns the stacked normalised OOF probability matrix.
    """
    logging.info(f"\n{'='*60}")
    logging.info("MODEL 2b — XGBoostClassifier (OvR)")
    logging.info(f"{'='*60}")

    phases = label_encoder.classes_
    n_phases = len(phases)

    # Precompute shared splits once — GroupKFold is deterministic on groups
    gkf = GroupKFold(n_splits=n_folds)
    fold_splits = list(gkf.split(X, y, groups))

    ovr_probas = np.zeros((len(y), n_phases))
    phase_aucs = {}

    for i, phase in enumerate(phases):
        y_binary = (y == i).astype(int)
        pos_n, neg_n = y_binary.sum(), (y_binary == 0).sum()
        scale_w = neg_n / max(pos_n, 1)

        # Phase-specific feature subset (consistency filter)
        if consistency_map and phase in consistency_map:
            feat_idx = consistency_map[phase]
            X_phase = X[:, feat_idx]
            n_feats_used = len(feat_idx)
        else:
            X_phase = X
            n_feats_used = X.shape[1]

        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_w,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        y_proba_phase = np.zeros(len(y_binary))
        for train_idx, test_idx in fold_splits:
            m = clone(model)
            m.fit(X_phase[train_idx], y_binary[train_idx])
            y_proba_phase[test_idx] = m.predict_proba(X_phase[test_idx])[:, 1]

        auc = roc_auc_score(y_binary, y_proba_phase)
        phase_aucs[phase] = auc
        logging.info(f"  {phase:<12}: AUC {auc:.3f}  (pos={pos_n}, neg={neg_n}, feats={n_feats_used})")
        ovr_probas[:, i] = y_proba_phase

    # Normalise rows so probabilities sum to 1 (enables multiclass AUC)
    row_sums = ovr_probas.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    ovr_probas_norm = ovr_probas / row_sums

    macro_auc = roc_auc_score(y, ovr_probas_norm, multi_class="ovr", average="macro")
    logging.info(f"\n  OvR ensemble macro AUC: {macro_auc:.3f}  [random=0.500]")

    # Save text report
    lines = [f"XGBoostClassifier (OvR) | GroupKFold ({n_folds} folds)\n"]
    for ph, auc in phase_aucs.items():
        lines.append(f"  {ph:<12}: AUC {auc:.3f}")
    lines.append(f"\n  Ensemble macro AUC: {macro_auc:.3f}")
    rpt_path = output_dir / f"classification_report_ovr_{timestamp}.txt"
    rpt_path.write_text("\n".join(lines))
    logging.info(f"  Report → {rpt_path.name}")

    # Bar chart: OvR per-phase AUC
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(phase_aucs.keys()), list(phase_aucs.values()), color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="k", linestyle="--", lw=1, label="Chance (0.5)")
    ax.axhline(macro_auc, color="tomato", linestyle="-", lw=1.5, label=f"Ensemble macro ({macro_auc:.3f})")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"OvR Per-Phase AUC — XGBoostClassifier\nGroupKFold ({n_folds} folds)")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / f"ovr_phase_auc_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  OvR AUC plot → {plot_path.name}")

    return ovr_probas_norm


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _log_auc(y_true, y_proba, label_encoder) -> None:
    """Log ROC-AUC. For binary: standard AUC. For multiclass: OvR macro AUC.
    AUC = 0.5 is random regardless of class balance — unlike macro F1."""
    n_classes = len(label_encoder.classes_)
    if y_proba.sum() == 0:
        logging.info("  AUC: N/A (model has no predict_proba)")
        return
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            logging.info(f"  AUC-ROC (binary): {auc:.3f}  [random=0.500]")
        else:
            auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            logging.info(f"  AUC-ROC OvR macro: {auc_ovr:.3f}  [random=0.500]")
            for i, cls in enumerate(label_encoder.classes_):
                y_bin = (y_true == i).astype(int)
                cls_auc = roc_auc_score(y_bin, y_proba[:, i])
                logging.info(f"    {cls:<12}: AUC {cls_auc:.3f}")
    except Exception as e:
        logging.warning(f"  AUC computation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--n-folds", type=int, default=5,
                   help="GroupKFold splits — each fold holds out all days of some users.")
    p.add_argument("--shap-max-display", type=int, default=20)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument(
        "--binary-phase",
        choices=["", "Menstrual", "Follicular", "Ovulation", "Luteal"],
        default="",
        help="If set, train a binary classifier: target phase vs all others. "
             "Recommended: 'Menstrual' or 'Ovulation'.",
    )
    p.add_argument(
        "--aggregate-by-phase", action="store_true",
        help="Collapse day-level rows to one averaged feature vector per (user, phase) "
             "before training. Reduces noise — each sample is a user's mean profile for "
             "a phase, not a single day.",
    )
    p.add_argument(
        "--ovr", action="store_true",
        help="Train one binary XGBoost per phase (One-vs-Rest) with shared GroupKFold "
             "splits, then report per-phase AUC and ensemble macro AUC.",
    )
    p.add_argument(
        "--consistency-filter", type=float, default=0.0, metavar="THRESHOLD",
        help="If > 0, filter features per phase in the OvR classifier to only those "
             "where pct_consistent >= THRESHOLD (0-100) from script 10's summary CSV. "
             "Requires reports/phase_distributions/phase_peak_summary_*.csv to exist.",
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
    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg = cfg["paths"]["files"]

    output_dir = ROOT / "reports" / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load phase-labeled timeline (step 08b output) ──────────────────────
    logging.info("\n[1/5] Loading phase-labeled timeline (step 08b)…")
    df = load_phase_labeled(interim_dir, files_cfg["phase_labeled"] + "_*.csv")

    # ── [2] Compute per-user z-scores from _mean columns ───────────────────────
    # Must happen before any aggregation so each user's mean/std is computed
    # over their complete set of labeled days.
    logging.info("\n[2/5] Computing per-user z-score features from _mean columns…")
    df, zscore_cols = compute_zscore_features(df)

    # ── [3] Optionally aggregate to per-user phase profiles ────────────────────
    if args.aggregate_by_phase:
        logging.info("\n[3/5] Aggregating to per-user phase profiles…")
        df = aggregate_to_phase_profiles(df, zscore_cols)

    # ── [4] Save labelled dataset ───────────────────────────────────────────────
    logging.info("\n[4/5] Saving labelled user-day dataset…")
    save_cols = (
        ["author", "phase"]
        + [c for c in ["offset_from_cd1", "subreddit", "ts_date"] if c in df.columns]
        + zscore_cols
    )
    saved_path = save_with_timestamp(
        df[[c for c in save_cols if c in df.columns]],
        interim_dir,
        files_cfg["ml_labeled_days"],
    )
    logging.info(f"  Saved → {saved_path}")

    # ── Build ML arrays ─────────────────────────────────────────────────────────
    X, y, groups, label_encoder = make_arrays(df, zscore_cols, binary_phase=args.binary_phase)
    n_unique_users = len(np.unique(groups))
    median_days = int(np.median(np.unique(groups, return_counts=True)[1]))
    logging.info(
        f"\n  X shape: {X.shape} | classes: {list(label_encoder.classes_)} | "
        f"users: {n_unique_users} | median days/user: {median_days}"
    )

    if n_unique_users < args.n_folds:
        logging.warning(f"Only {n_unique_users} users — reducing n_folds to {n_unique_users}")
        args.n_folds = n_unique_users

    # ── [5] XGBoost Ensemble + SHAP ────────────────────────────────────────────
    logging.info("\n[5/5] XGBoost ensemble…")
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

    # ── [5b] Optional OvR ensemble ──────────────────────────────────────────────
    if args.ovr:
        if args.binary_phase:
            logging.warning("--ovr is ignored when --binary-phase is set (already binary).")
        else:
            logging.info("\n[5b] OvR ensemble — baseline (all features)…")
            train_ovr_ensemble(
                X, y, groups, zscore_cols, label_encoder,
                args.n_folds, output_dir, timestamp,
            )

            if args.consistency_filter > 0:
                logging.info(
                    f"\n[5c] OvR ensemble — consistency filter "
                    f"(pct_consistent >= {args.consistency_filter:.0f}%)…"
                )
                consistency_map = load_consistency_map(
                    reports_dir=ROOT / "reports",
                    threshold=args.consistency_filter,
                    feature_names=zscore_cols,
                    phases=list(label_encoder.classes_),
                )
                ts_filtered = timestamp + "_filtered"
                train_ovr_ensemble(
                    X, y, groups, zscore_cols, label_encoder,
                    args.n_folds, output_dir, ts_filtered,
                    consistency_map=consistency_map,
                )

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
