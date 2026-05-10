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
  StratifiedGroupKFold splits are keyed on author (user_id).  Multiple days from the
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
import random
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import normalize_features_per_user_zscore
from src.config import load_config
from src.constants import CLASSIC_Z_HIGH, CLASSIC_Z_LOW, PHASE_ORDER
from src.io import find_latest_file, save_with_timestamp
from src.ml_data import load_phase_labeled_dataset
from src.ml_eval import log_oof_metrics, run_phase_statistical_gauntlet

warnings.filterwarnings("ignore", category=UserWarning)


def _clean_feat_name(name: str) -> str:
    """Strip z-score suffix and replace underscores with spaces for display.

    Args:
        name: Raw feature column name (e.g. "valence_dict_average_zscore").

    Returns:
        Human-readable label (e.g. "valence dict average").
    """
    return name.replace("_zscore", "").replace("_", " ")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_zscore_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-user z-scores from the _mean columns and return the zscore column names.

    Step 06 outputs {feature}_mean columns (daily mean per user-day). The "normalize"
    in step 06's name means collapsing multiple posts per day into one row — not
    z-scoring. Z-scores are computed here per user via normalize_features_per_user_zscore(),
    which normalises each feature relative to that user's own mean and std across
    all their days.

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
# 2. ML ARRAYS AND GROUP-AWARE CV
# ═══════════════════════════════════════════════════════════════════════════════
def make_arrays(df: pd.DataFrame, feature_cols: list[str], binary_phase: str = ""):
    """Convert labeled DataFrame to (X, y_int, groups, LabelEncoder) for sklearn.

    groups = author ids, passed to StratifiedGroupKFold so that all days from one user
    land entirely on the same side of every train/test split — preventing
    within-user correlation from leaking across folds.

    Args:
        binary_phase: If non-empty (e.g. "Menstrual"), collapse labels to
            1 = target phase, 0 = all other phases.  The label encoder will
            have classes ["Other", binary_phase] so class 1 is always the target.
    """
    # Do not impute NaNs — XGBoost handles missing values natively by learning
    # the optimal split direction for each missing entry. Filling with 0 after
    # z-score normalisation would set missing data to the user's mean, which
    # introduces bias when a feature is missing because too little text was written.
    X = df[feature_cols].values.astype(np.float32)

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

    XGBoost multiclass has no class_weight='balanced' — sample_weight must be
    passed explicitly to fit() for each fold.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)

def oof_predictions(model, X, y, groups, n_folds: int, seed: int = 42) -> np.ndarray:
    """Out-of-fold predictions with StratifiedGroupKFold on author.

    All user-days from the same user stay entirely in either train or test.
    This is mandatory because days from the same user share the same writing
    style, cycle trajectory, and vocabulary — they are not independent samples.

    The fold loop is implemented manually (rather than cross_val_predict) so
    sample_weight can be passed to fit() on each fold.  sklearn 1.6+ removed
    the fit_params argument from cross_val_predict in favour of metadata routing,
    which XGBoost does not support.
    """
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
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
# 3. MODEL — XGBOOST ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp, seed: int = 42):
    logging.info(f"\n{'='*60}")
    logging.info("MODEL — XGBoostClassifier")
    logging.info(f"{'='*60}")

    n_classes = len(label_encoder.classes_)
    # objective="multi:softprob" outputs a probability distribution over all classes,
    # required for predict_proba and for SHAP TreeExplainer to work correctly.
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    y_pred, y_proba = oof_predictions(model, X, y, groups, n_folds, seed=seed)

    report = classification_report(y, y_pred, target_names=label_encoder.classes_, digits=3)
    logging.info(f"\nOOF Report:\n{report}")
    log_oof_metrics(y, y_proba, label_encoder)

    rpt_path = output_dir / f"classification_report_ensemble_{timestamp}.txt"
    rpt_path.write_text(f"XGBoostClassifier | StratifiedGroupKFold OOF ({n_folds} folds)\n\n{report}")

    # Confusion matrix — shows which phases the model confuses most
    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title(
        f"XGBoostClassifier — Confusion Matrix\n"
        f"StratifiedGroupKFold OOF ({n_folds} folds) | Macro-F1 = {_macro_f1(y, y_pred):.3f}",
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
# SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def run_shap_analysis(model, X, y, feature_names, label_encoder, output_dir, timestamp, max_display=20, seed: int = 42):
    logging.info("\n  Computing SHAP values (TreeExplainer)…")

    n_shap = min(len(X), 5_000)
    if n_shap < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n_shap, replace=False)
        X_shap = X[idx]
        y_shap = y[idx]
    else:
        X_shap = X
        y_shap = y

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Normalise to (n_samples, n_features, n_classes).
    # SHAP <0.40 returns a list of n_classes 2D arrays; SHAP >=0.40 returns a
    # single 3D array. Both are handled here for compatibility.
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

    # Save SHAP values as a table: importance and direction per feature per class.
    #
    # Direction is derived from the correlation between feature value and SHAP value
    # across all samples. Positive correlation = HIGH feature value → model assigns
    # higher probability to this phase. This is the only reliable directional metric:
    # - mean SHAP over all samples is dominated by the majority (non-phase) rows
    # - mean SHAP over phase-only samples conflates feature value with SHAP sign
    #   (e.g. low valence → positive SHAP for Menstrual, so Menstrual rows all have
    #   positive SHAP for valence even though the direction is LOW)
    rows = []
    clean_names = [n.replace("_zscore", "") for n in feature_names]
    feat_vals = X_shap.astype(float)  # (n_samples, n_features)
    for cls_idx in range(n_classes):
        phase_name = label_encoder.classes_[cls_idx]
        sv = shap_array[:, :, cls_idx]           # (n_samples, n_features), all samples
        mean_abs = np.abs(sv).mean(axis=0)
        for fidx, clean_name in enumerate(clean_names):
            fv = feat_vals[:, fidx]
            sv_f = sv[:, fidx]
            valid = ~np.isnan(fv)
            corr = float(np.corrcoef(fv[valid], sv_f[valid])[0, 1]) if valid.sum() > 2 else 0.0
            rows.append({
                "phase":         phase_name,
                "feature":       clean_name,
                "mean_abs_shap": round(mean_abs[fidx], 6),
                "shap_feat_corr": round(corr, 4),
                "direction":     "HIGH" if corr > 0 else "LOW",
            })
    shap_df = pd.DataFrame(rows)
    shap_csv = output_dir / f"shap_values_{timestamp}.csv"
    shap_df.to_csv(shap_csv, index=False)
    logging.info(f"  SHAP table → {shap_csv.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# OVR ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

# run_phase_statistical_gauntlet is imported from src.ml_eval


def train_ovr_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp, empirical_tail_pct=10, seed: int = 42):
    """Train one binary XGBoost per phase (One-vs-Rest) with shared StratifiedGroupKFold splits.

    One-vs-Rest means 4 separate binary classifiers: Menstrual vs rest,
    Follicular vs rest, Ovulation vs rest, Luteal vs rest. Each gets its own
    scale_pos_weight to handle class imbalance. All 4 share the same StratifiedGroupKFold
    splits so their OOF probability vectors are aligned and can be stacked into
    an (n_samples, 4) matrix for multiclass AUC.

    OvR is run in addition to the multiclass classifier because it yields
    per-phase AUC scores, which are more interpretable than macro-F1 for
    structurally imbalanced phases (Ovulation ~3 days vs Luteal ~14 days).

    Returns the stacked normalised OOF probability matrix.
    """
    logging.info(f"\n{'='*60}")
    logging.info("MODEL 2b — XGBoostClassifier (OvR)")
    logging.info(f"{'='*60}")

    phases = label_encoder.classes_
    n_phases = len(phases)

    # Precompute shared splits once — StratifiedGroupKFold preserves phase distribution per fold
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = list(gkf.split(X, y, groups))

    ovr_probas = np.zeros((len(y), n_phases))
    phase_aucs = {}
    gauntlet_results = []

    for i, phase in enumerate(phases):
        y_binary = (y == i).astype(int)
        pos_n, neg_n = y_binary.sum(), (y_binary == 0).sum()
        scale_w = neg_n / max(pos_n, 1)

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
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )

        # Accumulate OOF feature importances across folds to avoid selection bias.
        # Using a full-data model for feature selection (then testing on the same
        # data) is a double-dip: the model has seen all labels during selection.
        # OOF importances are computed on held-out data and are therefore unbiased.
        oof_importances = np.zeros(X_phase.shape[1])
        y_proba_phase = np.zeros(len(y_binary))
        for train_idx, test_idx in fold_splits:
            m = clone(model)
            m.fit(X_phase[train_idx], y_binary[train_idx])
            y_proba_phase[test_idx] = m.predict_proba(X_phase[test_idx])[:, 1]
            oof_importances += m.feature_importances_

        # Average OOF importances across folds, then select top 15
        oof_importances /= len(fold_splits)
        top_indices = np.argsort(oof_importances)[::-1][:15].tolist()

        auc = roc_auc_score(y_binary, y_proba_phase)
        phase_aucs[phase] = auc
        logging.info(f"  {phase:<12}: AUC {auc:.3f}  (pos={pos_n}, neg={neg_n}, feats={n_feats_used})")
        ovr_probas[:, i] = y_proba_phase

        gauntlet_df = run_phase_statistical_gauntlet(X_phase, y_binary, list(feature_names), top_indices, empirical_tail_pct)
        gauntlet_df.insert(0, "phase", phase)
        _print_gauntlet_table(phase, gauntlet_df)
        gauntlet_results.append(gauntlet_df)

    # Normalise rows so probabilities sum to 1 (enables multiclass AUC)
    row_sums = ovr_probas.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    ovr_probas_norm = ovr_probas / row_sums

    macro_auc = roc_auc_score(y, ovr_probas_norm, multi_class="ovr", average="macro")
    logging.info(f"\n  OvR ensemble macro AUC: {macro_auc:.3f}  [random=0.500]")

    # Save text report
    lines = [f"XGBoostClassifier (OvR) | StratifiedGroupKFold ({n_folds} folds)\n"]
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
    ax.set_title(f"OvR Per-Phase AUC — XGBoostClassifier\nStratifiedGroupKFold ({n_folds} folds)")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / f"ovr_phase_auc_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  OvR AUC plot → {plot_path.name}")

    if gauntlet_results:
        gauntlet_all = pd.concat(gauntlet_results, ignore_index=True)
        gauntlet_path = output_dir / f"ovr_statistical_gauntlet_{timestamp}.csv"
        gauntlet_all.to_csv(gauntlet_path, index=False)
        logging.info(f"  Statistical gauntlet → {gauntlet_path.name}")

        md_path = output_dir / f"expert_review_phases_{timestamp}.md"
        md_path.write_text(_build_gauntlet_markdown(gauntlet_all, phase_aucs, macro_auc))
        logging.info(f"  Markdown report      → {md_path.name}")

    return ovr_probas_norm


# ═══════════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def permute_labels_within_user(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Shuffle phase labels within each user independently.

    Preserves the number of days per user and the StratifiedGroupKFold structure.
    Destroys the phase→language relationship while keeping per-user
    feature distributions intact.  If AUC stays high after permutation
    the model is not learning genuine phase signals.
    """
    y_perm = y.copy()
    for user in np.unique(groups):
        mask = groups == user
        y_perm[mask] = rng.permutation(y[mask])
    return y_perm


def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_encoder,
    n_folds: int,
    n_permutations: int,
    output_dir: Path,
    timestamp: str,
    seed: int = 42,
) -> None:
    """Run a within-user permutation test.

    For each of n_permutations iterations: shuffle phase labels within each
    user, train the same XGBoost (OvR macro AUC via multiclass softprob),
    collect AUC.  The empirical p-value is the fraction of permuted AUCs >=
    the real AUC (lower = more confident the signal is real).
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"PERMUTATION TEST — {n_permutations} within-user shuffles")
    logging.info(f"{'='*60}")

    n_classes = len(label_encoder.classes_)
    model_template = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    # Real AUC (use actual labels)
    _, y_proba_real = oof_predictions(model_template, X, y, groups, n_folds, seed=seed)
    real_auc = roc_auc_score(y, y_proba_real, multi_class="ovr", average="macro")
    logging.info(f"  Real macro AUC: {real_auc:.4f}")

    rng = np.random.default_rng(seed)
    perm_aucs = []
    for i in range(n_permutations):
        y_perm = permute_labels_within_user(y, groups, rng)
        _, y_proba_perm = oof_predictions(model_template, X, y_perm, groups, n_folds)
        auc_perm = roc_auc_score(y_perm, y_proba_perm, multi_class="ovr", average="macro")
        perm_aucs.append(auc_perm)
        logging.info(f"  Permutation {i+1:>3}/{n_permutations}: AUC {auc_perm:.4f}")

    perm_aucs = np.array(perm_aucs)
    p_value = (perm_aucs >= real_auc).mean()
    logging.info(f"\n  Permuted AUC: mean={perm_aucs.mean():.4f}  std={perm_aucs.std():.4f}")
    logging.info(f"  Real AUC:     {real_auc:.4f}")
    logging.info(f"  Empirical p-value (fraction perm >= real): {p_value:.4f}")
    if p_value < 0.05:
        logging.info("  → Signal is REAL (p < 0.05). Phase labels carry genuine linguistic information.")
    else:
        logging.info("  → Signal NOT significant (p >= 0.05). Possible circularity / noise.")

    # Save text report
    lines = [
        f"Permutation test | StratifiedGroupKFold ({n_folds} folds) | {n_permutations} permutations",
        f"",
        f"Real macro AUC:      {real_auc:.4f}",
        f"Permuted mean AUC:   {perm_aucs.mean():.4f}",
        f"Permuted std AUC:    {perm_aucs.std():.4f}",
        f"Permuted AUCs:       {', '.join(f'{a:.4f}' for a in perm_aucs)}",
        f"Empirical p-value:   {p_value:.4f}",
    ]
    rpt_path = output_dir / f"permutation_test_{timestamp}.txt"
    rpt_path.write_text("\n".join(lines))
    logging.info(f"  Report → {rpt_path.name}")

    # Plot: permuted AUC distribution with real AUC marked
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_aucs, bins=max(5, n_permutations // 2), color="steelblue", alpha=0.7,
            edgecolor="white", label="Permuted")
    ax.axvline(real_auc, color="tomato", lw=2, linestyle="-", label=f"Real AUC ({real_auc:.3f})")
    ax.axvline(perm_aucs.mean(), color="navy", lw=1.5, linestyle="--",
               label=f"Perm mean ({perm_aucs.mean():.3f})")
    ax.set_xlabel("Macro AUC (OvR)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Permutation Test — Within-User Label Shuffle\n"
        f"p={p_value:.3f}  ({n_permutations} permutations, StratifiedGroupKFold {n_folds} folds)",
        fontsize=11,
    )
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / f"permutation_test_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Plot → {plot_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gauntlet_markdown(
    gauntlet_all: pd.DataFrame,
    phase_aucs: dict,
    macro_auc: float,
) -> str:
    """Render the statistical gauntlet results as a clinical Markdown report."""

    def _fmt_p(p: float) -> str:
        return "< 0.001\\*" if p < 0.001 else (f"{p:.3f}\\*" if p < 0.05 else f"{p:.3f}")

    def _fmt_global(row) -> str:
        p = row["global_p_fdr"]
        if p >= 0.05:
            return "—"
        med  = row["global_median_shift"]
        mean = row["global_mean_shift"]
        arrow = "↑" if med > 0 else "↓"
        return f"{_fmt_p(p)} {arrow} Med {abs(med):.2f} / Mean {abs(mean):.2f} SD"

    def _fmt_tail(p, phase_pct, rest_pct, rr, sev_med, sev_mean) -> str:
        if p >= 0.05:
            return "—"
        med_s  = f"{sev_med:+.2f}"  if sev_med  == sev_med  else "—"   # NaN check
        mean_s = f"{sev_mean:+.2f}" if sev_mean == sev_mean else "—"
        return (
            f"{_fmt_p(p)} \\| {phase_pct:.1f}% vs {rest_pct:.1f}%"
            f" ({rr:.1f}x) [Med: {med_s} / Mean: {mean_s} SD]"
        )

    def _phenotype(row) -> str:
        g  = row["global_p_fdr"]    < 0.05
        hi = row["high_tail_p_fdr"] < 0.05
        lo = row["low_tail_p_fdr"]  < 0.05
        if hi and lo: return "Bimodal / mixed response"
        if hi:        return "High-end subgroup enrichment"
        if lo:        return "Low-end subgroup enrichment"
        if g:         return "Global median shift (no tail)"
        return "No significant shift"

    # AUC summary table
    lines = [
        "# Clinical Phase Discovery Report", "",
        "## Model Performance (OvR AUC)", "",
        "| Phase | AUC-ROC | vs Chance |",
        "|---|---|---|",
    ]
    for phase, auc in phase_aucs.items():
        lines.append(f"| {phase} | {auc:.3f} | {auc - 0.5:+.3f} |")
    lines += [
        f"| **Ensemble macro** | **{macro_auc:.3f}** | **{macro_auc - 0.5:+.3f}** |",
        "",
    ]

    for phase in gauntlet_all["phase"].unique():
        df_ph = gauntlet_all[gauntlet_all["phase"] == phase].copy()

        sig_mask = (
            (df_ph["global_p_fdr"]           < 0.05) |
            (df_ph["high_tail_p_fdr"]        < 0.05) |
            (df_ph["low_tail_p_fdr"]         < 0.05) |
            (df_ph["classic_high_tail_p_fdr"] < 0.05) |
            (df_ph["classic_low_tail_p_fdr"]  < 0.05)
        )
        df_ph = df_ph[sig_mask].copy()
        if df_ph.empty:
            continue

        df_ph["_phenotype"] = df_ph.apply(_phenotype, axis=1)
        df_ph = df_ph.sort_values("_phenotype").reset_index(drop=True)

        auc_str = f"{phase_aucs[phase]:.3f}" if phase in phase_aucs else "N/A"
        lines += [
            f"## Phase: {phase} (AUC {auc_str})", "",
            "Top predictive features with at least one significant FDR-corrected p-value. "
            "Sorted by Clinical Phenotype. "
            "Empirical tails use data-driven 10th/90th percentiles; "
            f"Classic tails use fixed z ≥ {CLASSIC_Z_HIGH} / z ≤ {CLASSIC_Z_LOW} (theoretical 10% tails).", "",
            "| Feature | Global Shift (Med/Mean) | Empirical High | Empirical Low"
            " | Classic High | Classic Low | Clinical Phenotype |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in df_ph.iterrows():
            feat = _clean_feat_name(row["feature"])
            lines.append(
                f"| {feat}"
                f" | {_fmt_global(row)}"
                f" | {_fmt_tail(row['high_tail_p_fdr'], row['high_phase_pct'], row['high_rest_pct'], row['high_rr'], row['high_severity_median'], row['high_severity_mean'])}"
                f" | {_fmt_tail(row['low_tail_p_fdr'], row['low_phase_pct'], row['low_rest_pct'], row['low_rr'], row['low_severity_median'], row['low_severity_mean'])}"
                f" | {_fmt_tail(row['classic_high_tail_p_fdr'], row['classic_high_phase_pct'], row['classic_high_rest_pct'], row['classic_high_rr'], row['classic_high_severity_median'], row['classic_high_severity_mean'])}"
                f" | {_fmt_tail(row['classic_low_tail_p_fdr'], row['classic_low_phase_pct'], row['classic_low_rest_pct'], row['classic_low_rr'], row['classic_low_severity_median'], row['classic_low_severity_mean'])}"
                f" | {row['_phenotype']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _print_gauntlet_table(phase: str, df: pd.DataFrame) -> None:
    def _fmt(p: float) -> str:
        return f"{p:.4f}*" if p < 0.05 else f"{p:.4f} "

    logging.info(f"\n  {'─'*96}")
    logging.info(f"  Gauntlet: {phase}")
    logging.info(f"  {'─'*96}")
    logging.info(
        f"  {'Feature':<36} {'Global p':>10} {'EmpHi p':>10} {'EmpLo p':>10}"
        f" {'ClsHi p':>10} {'ClsLo p':>10} {'Med Δ':>8} {'Mean Δ':>8}"
    )
    for _, row in df.iterrows():
        logging.info(
            f"  {row['feature']:<36} {_fmt(row['global_p_fdr']):>10}"
            f" {_fmt(row['high_tail_p_fdr']):>10} {_fmt(row['low_tail_p_fdr']):>10}"
            f" {_fmt(row['classic_high_tail_p_fdr']):>10} {_fmt(row['classic_low_tail_p_fdr']):>10}"
            f" {row['global_median_shift']:>+8.3f} {row['global_mean_shift']:>+8.3f}"
        )


def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# log_oof_metrics is imported from src.ml_eval


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--n-folds", type=int, default=5,
                   help="StratifiedGroupKFold splits — each fold holds out all days of some users.")
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
        help="Train one binary XGBoost per phase (One-vs-Rest) with shared StratifiedGroupKFold "
             "splits, then report per-phase AUC and ensemble macro AUC.",
    )
    p.add_argument("--no-anchors", action="store_true",
                   help="Use no-anchors phase-labeled file (timeline_phase_labeled_no_anchors_*.csv). "
                        "Run scripts/08b_label_phases.py --no-anchors first.")
    p.add_argument(
        "--permute", action="store_true",
        help="Run within-user permutation test to validate that phase labels carry "
             "genuine linguistic signal (not circularity from FFT-detected periodicity).",
    )
    p.add_argument("--n-permutations", type=int, default=100,
                   help="Number of permutations for --permute (default 100; use 500+ for publication).")
    p.add_argument("--empirical-tail-pct", type=int, default=10,
                   help="Percentile for empirical tail thresholds (default 10 → 10th/90th pct). "
                        "Use 20 for 20th/80th pct.")
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

    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg = cfg["paths"]["files"]

    output_dir = ROOT / "reports" / ("ml_no_anchors" if args.no_anchors else "ml")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1+2] Load and z-score phase-labeled timeline ──────────────────────────
    # ── [3] Optionally aggregate to per-user phase profiles ────────────────────
    if args.aggregate_by_phase:
        # load_phase_labeled_dataset handles file location, z-scoring, phase
        # filtering, and aggregation in a single call (same logic as script 27).
        logging.info("\n[1/5] Loading + z-scoring + aggregating via load_phase_labeled_dataset…")
        df, zscore_cols, _le_from_loader = load_phase_labeled_dataset(
            cfg, no_anchors=args.no_anchors
        )
        logging.info("\n[3/5] Phase profiles already aggregated by load_phase_labeled_dataset.")
    else:
        # Day-level path: load raw file, z-score, keep one row per user-day.
        anchor_suffix = "_no_anchors" if args.no_anchors else ""
        phase_labeled_pattern = files_cfg["phase_labeled"] + anchor_suffix + "_*.csv"
        logging.info(
            f"\n[1/5] Loading phase-labeled timeline (step 08b)… [{phase_labeled_pattern}]"
        )
        path = find_latest_file(
            interim_dir,
            phase_labeled_pattern,
            exclude=None if args.no_anchors else "_no_anchors",
        )
        if path is None:
            raise FileNotFoundError(
                f"No {phase_labeled_pattern} found in data/interim/. "
                f"Run scripts/08b_label_phases.py"
                f"{'  --no-anchors' if args.no_anchors else ''} first."
            )
        logging.info(f"  Loading: {path.name}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users")
        if "phase" not in df.columns:
            raise ValueError("Missing 'phase' column. Re-run scripts/08b_label_phases.py.")

        # ── [2] Compute per-user z-scores from _mean columns ───────────────────
        # Must happen before any aggregation so each user's mean/std is computed
        # over their complete set of labeled days.
        logging.info("\n[2/5] Computing per-user z-score features from _mean columns…")
        df, zscore_cols = compute_zscore_features(df)
        logging.info("\n[3/5] Day-level mode — no phase aggregation.")

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
        args.n_folds, output_dir, timestamp, seed=seed,
    )

    if not args.skip_shap:
        run_shap_analysis(
            ensemble, X, y, zscore_cols, label_encoder,
            output_dir, timestamp, args.shap_max_display, seed=seed,
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
                empirical_tail_pct=args.empirical_tail_pct, seed=seed,
            )

    # ── [5c] Optional permutation test ─────────────────────────────────────────
    if args.permute:
        if args.binary_phase:
            logging.warning("--permute with --binary-phase permutes the binary labels within user.")
        run_permutation_test(
            X, y, groups, label_encoder,
            args.n_folds, args.n_permutations, output_dir, timestamp, seed=seed,
        )

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
