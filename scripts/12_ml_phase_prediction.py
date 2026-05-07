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
from scipy import stats
from statsmodels.stats.multitest import multipletests
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

def oof_predictions(model, X, y, groups, n_folds: int) -> np.ndarray:
    """Out-of-fold predictions with StratifiedGroupKFold on author.

    All user-days from the same user stay entirely in either train or test.
    This is mandatory because days from the same user share the same writing
    style, cycle trajectory, and vocabulary — they are not independent samples.

    The fold loop is implemented manually (rather than cross_val_predict) so
    sample_weight can be passed to fit() on each fold.  sklearn 1.6+ removed
    the fit_params argument from cross_val_predict in favour of metadata routing,
    which XGBoost does not support.
    """
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
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
def train_and_evaluate_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp):
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
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    y_pred, y_proba = oof_predictions(model, X, y, groups, n_folds)

    report = classification_report(y, y_pred, target_names=label_encoder.classes_, digits=3)
    logging.info(f"\nOOF Report:\n{report}")
    _log_auc(y, y_proba, label_encoder)

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
def run_shap_analysis(model, X, y, feature_names, label_encoder, output_dir, timestamp, max_display=20):
    logging.info("\n  Computing SHAP values (TreeExplainer)…")

    n_shap = min(len(X), 5_000)
    if n_shap < len(X):
        idx = np.random.default_rng(42).choice(len(X), size=n_shap, replace=False)
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

def run_phase_statistical_gauntlet(
    X: np.ndarray,
    y_binary: np.ndarray,
    feature_names: list[str],
    top_indices: list[int],
) -> pd.DataFrame:
    """Mann-Whitney U + Fisher tail tests (BH-FDR) for OvR top-15 features.

    Returns one row per feature with FDR-corrected p-values and raw tail
    effect sizes (phase % vs rest % in each tail, plus relative risk).
    """
    feat_names = []
    p_global_raw, p_high_raw, p_low_raw = [], [], []
    median_shifts = []
    hi_stats, lo_stats = [], []
    hi_severities, lo_severities = [], []

    phase_mask_global = y_binary == 1

    for idx in top_indices:
        raw_feat = X[:, idx].astype(float)

        # Isolate only valid (non-NaN) rows for this feature before any math.
        # NaNs in X are intentional for XGBoost but must be removed for scipy tests:
        # mannwhitneyu propagates NaN by default, and including NaN users in the
        # denominator artificially shrinks tail percentages (NaN != threshold → False).
        valid_mask = ~np.isnan(raw_feat)
        feat = raw_feat[valid_mask]
        phase_mask = phase_mask_global[valid_mask]

        n_phase_users = phase_mask.sum()
        n_rest_users  = (~phase_mask).sum()

        if n_phase_users == 0 or n_rest_users == 0:
            continue

        median_shifts.append(
            np.median(feat[phase_mask]) - np.median(feat[~phase_mask])
        )

        _, pg = stats.mannwhitneyu(feat[phase_mask], feat[~phase_mask], alternative="two-sided")
        p_global_raw.append(pg)

        high_threshold = np.percentile(feat, 90)
        phase_users_in_high_tail = phase_mask & (feat >= high_threshold)
        rest_users_in_high_tail  = (~phase_mask) & (feat >= high_threshold)
        hi_phase_pct = phase_users_in_high_tail.sum() / n_phase_users * 100
        hi_rest_pct  = rest_users_in_high_tail.sum()  / n_rest_users  * 100
        hi_stats.append((hi_phase_pct, hi_rest_pct, hi_phase_pct / max(hi_rest_pct, 0.001)))
        hi_severities.append(
            np.median(feat[phase_users_in_high_tail]) if phase_users_in_high_tail.sum() > 0 else np.nan
        )
        ct_high = np.array([
            [phase_users_in_high_tail.sum(), (phase_mask & ~(feat >= high_threshold)).sum()],
            [rest_users_in_high_tail.sum(),  ((~phase_mask) & ~(feat >= high_threshold)).sum()],
        ])
        _, ph = stats.fisher_exact(ct_high, alternative="greater")
        p_high_raw.append(ph)

        low_threshold = np.percentile(feat, 10)
        phase_users_in_low_tail = phase_mask & (feat <= low_threshold)
        rest_users_in_low_tail  = (~phase_mask) & (feat <= low_threshold)
        lo_phase_pct = phase_users_in_low_tail.sum() / n_phase_users * 100
        lo_rest_pct  = rest_users_in_low_tail.sum()  / n_rest_users  * 100
        lo_stats.append((lo_phase_pct, lo_rest_pct, lo_phase_pct / max(lo_rest_pct, 0.001)))
        lo_severities.append(
            np.median(feat[phase_users_in_low_tail]) if phase_users_in_low_tail.sum() > 0 else np.nan
        )
        ct_low = np.array([
            [phase_users_in_low_tail.sum(), (phase_mask & ~(feat <= low_threshold)).sum()],
            [rest_users_in_low_tail.sum(),  ((~phase_mask) & ~(feat <= low_threshold)).sum()],
        ])
        _, pl = stats.fisher_exact(ct_low, alternative="greater")
        p_low_raw.append(pl)

        feat_names.append(feature_names[idx])

    _, pg_fdr, _, _ = multipletests(p_global_raw, method="fdr_bh")
    _, ph_fdr, _, _ = multipletests(p_high_raw, method="fdr_bh")
    _, pl_fdr, _, _ = multipletests(p_low_raw, method="fdr_bh")

    return pd.DataFrame({
        "feature":             feat_names,
        "global_p_fdr":        pg_fdr,
        "global_median_shift": median_shifts,
        "high_tail_p_fdr":     ph_fdr,
        "high_phase_pct":      [s[0] for s in hi_stats],
        "high_rest_pct":       [s[1] for s in hi_stats],
        "high_rr":             [s[2] for s in hi_stats],
        "high_severity":       hi_severities,
        "low_tail_p_fdr":      pl_fdr,
        "low_phase_pct":       [s[0] for s in lo_stats],
        "low_rest_pct":        [s[1] for s in lo_stats],
        "low_rr":              [s[2] for s in lo_stats],
        "low_severity":        lo_severities,
    })


def train_ovr_ensemble(X, y, groups, feature_names, label_encoder, n_folds, output_dir, timestamp):
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
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
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

        # Fit full-data model to extract stable feature importances, then run gauntlet
        full_model = clone(model)
        full_model.fit(X_phase, y_binary)
        top_indices = np.argsort(full_model.feature_importances_)[::-1][:15].tolist()
        gauntlet_df = run_phase_statistical_gauntlet(X_phase, y_binary, list(feature_names), top_indices)
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
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Real AUC (use actual labels)
    _, y_proba_real = oof_predictions(model_template, X, y, groups, n_folds)
    real_auc = roc_auc_score(y, y_proba_real, multi_class="ovr", average="macro")
    logging.info(f"  Real macro AUC: {real_auc:.4f}")

    rng = np.random.default_rng(42)
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
        shift = row["global_median_shift"]
        arrow = "↑" if shift > 0 else "↓"
        return f"{_fmt_p(p)} ({arrow} {abs(shift):.2f} SD)"

    def _fmt_extreme(p, phase_pct, rest_pct, rr, severity) -> str:
        if p >= 0.05:
            return "—"
        return f"{_fmt_p(p)} \\| {phase_pct:.1f}% vs {rest_pct:.1f}% ({rr:.1f}x) [Severity: {severity:+.2f} SD]"

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
            (df_ph["global_p_fdr"]    < 0.05) |
            (df_ph["high_tail_p_fdr"] < 0.05) |
            (df_ph["low_tail_p_fdr"]  < 0.05)
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
            "Sorted by Clinical Phenotype.", "",
            "| Feature | Global Shift | Extreme High | Extreme Low | Clinical Phenotype |",
            "|---|---|---|---|---|",
        ]
        for _, row in df_ph.iterrows():
            feat = row["feature"].replace("_zscore", "").replace("_", " ")
            lines.append(
                f"| {feat} "
                f"| {_fmt_global(row)} "
                f"| {_fmt_extreme(row['high_tail_p_fdr'], row['high_phase_pct'], row['high_rest_pct'], row['high_rr'], row['high_severity'])} "
                f"| {_fmt_extreme(row['low_tail_p_fdr'], row['low_phase_pct'], row['low_rest_pct'], row['low_rr'], row['low_severity'])} "
                f"| {row['_phenotype']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _print_gauntlet_table(phase: str, df: pd.DataFrame) -> None:
    def _fmt(p: float) -> str:
        return f"{p:.4f}*" if p < 0.05 else f"{p:.4f} "

    logging.info(f"\n  {'─'*68}")
    logging.info(f"  Gauntlet: {phase}")
    logging.info(f"  {'─'*68}")
    logging.info(f"  {'Feature':<36} {'Global p(FDR)':>14} {'HighTail p(FDR)':>16} {'LowTail p(FDR)':>15}")
    for _, row in df.iterrows():
        logging.info(
            f"  {row['feature']:<36} {_fmt(row['global_p_fdr']):>14} "
            f"{_fmt(row['high_tail_p_fdr']):>16} {_fmt(row['low_tail_p_fdr']):>15}"
        )


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
    p.add_argument("--n-permutations", type=int, default=10,
                   help="Number of permutations for --permute (default 10; use 50+ for publication).")
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

    output_dir = ROOT / "reports" / ("ml_no_anchors" if args.no_anchors else "ml")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load phase-labeled timeline (step 08b output) ──────────────────────
    anchor_suffix = "_no_anchors" if args.no_anchors else ""
    phase_labeled_pattern = files_cfg["phase_labeled"] + anchor_suffix + "_*.csv"
    logging.info(f"\n[1/5] Loading phase-labeled timeline (step 08b)… [{phase_labeled_pattern}]")
    path = find_latest_file(
        interim_dir,
        phase_labeled_pattern,
        exclude=None if args.no_anchors else "_no_anchors",
    )
    if path is None:
        raise FileNotFoundError(
            f"No {phase_labeled_pattern} found in data/interim/. "
            f"Run scripts/08b_label_phases.py{'  --no-anchors' if args.no_anchors else ''} first."
        )
    logging.info(f"  Loading: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days from {df['author'].nunique():,} users")
    if "phase" not in df.columns:
        raise ValueError("Missing 'phase' column. Re-run scripts/08b_label_phases.py.")

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
            ensemble, X, y, zscore_cols, label_encoder,
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

    # ── [5c] Optional permutation test ─────────────────────────────────────────
    if args.permute:
        if args.binary_phase:
            logging.warning("--permute with --binary-phase permutes the binary labels within user.")
        run_permutation_test(
            X, y, groups, label_encoder,
            args.n_folds, args.n_permutations, output_dir, timestamp,
        )

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
