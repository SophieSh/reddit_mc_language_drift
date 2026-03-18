"""Step 14 — User-Level Classification on Raw (Unnormalized) Features
=====================================================================
Same pipeline as step 13, but uses raw per-day language averages instead of
per-user z-scores.

Step 13 (normalized) asks:
  "Do condition users show DIFFERENT CYCLE PATTERNS compared to controls?"
  → removes each user's personal baseline; captures within-person variation

Step 14 (unnormalized) asks:
  "Do condition users write DIFFERENTLY OVERALL — at absolute levels?"
  → keeps each user's personal baseline; captures between-person differences

Method:
  [1] Load phase labels from ml_labeled_days (step 12) — consensus users only
  [2] Load raw _mean features from timeline_daily_aggregated
  [3] Merge on (author, offset_from_cd1) to get phase labels on raw features
  [4] Build user-level feature matrix: mean raw value per user per phase
  [5] Add delta features (Luteal−Follicular, etc.)
  [6] ANOVA feature selection
  [7] Decision Tree (interpretable)
  [8] XGBoost + SHAP

Usage:
  python scripts/14_unnormalized_classification.py --condition pmdd
  python scripts/14_unnormalized_classification.py --condition adhd
  python scripts/14_unnormalized_classification.py --condition depression
"""

import argparse
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not installed — falling back to RandomForestClassifier.")

warnings.filterwarnings("ignore", category=UserWarning)

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]

# ── Same regex patterns as step 13 ────────────────────────────────────────────
CONDITION_REGEX = {
    "pmdd": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
               am\s+dealing\s+with|was\s+told\s+I\s+have)\s+PMDD
        |diagnosed\s+(?:me\s+)?with\s+PMDD
        |(?:my|a)\s+PMDD\s+(?:diagnosis|symptoms?|episodes?|flare)
        |(?:PMDD\s+sufferer|PMDD\s+warrior|living\s+with\s+PMDD|my\s+PMDD)
        |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+PMDD
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    "adhd": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
               was\s+told\s+I\s+have)\s+ADHD
        |diagnosed\s+(?:me\s+)?with\s+ADHD
        |(?:my|a)\s+ADHD\s+(?:diagnosis|symptoms?|brain|medication|meds|treatment)
        |(?:ADHDer|living\s+with\s+ADHD|my\s+ADHD)
        |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+ADHD
        |ADHD\s+and\s+PMDD|PMDD\s+and\s+ADHD
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    "depression": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|struggle\s+with)\s+
               (?:depression|MDD|major\s+depressive\s+disorder|clinical\s+depression)
        |diagnosed\s+(?:me\s+)?with\s+(?:depression|MDD|major\s+depressive\s+disorder)
        |(?:my|a)\s+(?:depression|MDD)\s+(?:diagnosis|symptoms?|episodes?|medication|meds)
        |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+(?:depression|MDD)
        |living\s+with\s+depression
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
}

CONDITION_CONFIG = {
    "pmdd":       {"label": "PMDD"},
    "adhd":       {"label": "ADHD"},
    "depression": {"label": "Depression"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. USER IDENTIFICATION  (same as step 13 — regex only)
# ═══════════════════════════════════════════════════════════════════════════════

def identify_condition_users(interim_dir: Path, condition: str, candidate_users: set) -> set:
    regex = CONDITION_REGEX[condition]
    label = CONDITION_CONFIG[condition]["label"]

    posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_with_anchors_*.csv")
    if posts_path is None:
        posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_*.csv")
    if posts_path is None:
        raise FileNotFoundError("No posts_all_users_preprocessed_*.csv found.")

    logging.info(f"  Loading posts: {posts_path.name}")
    posts = pd.read_csv(posts_path, encoding="utf-8-sig", usecols=["author", "text"], low_memory=False)
    posts = posts[posts["author"].astype(str).isin(candidate_users)].copy()
    posts["author"] = posts["author"].astype(str)

    mask = posts["text"].fillna("").str.contains(regex, regex=True)
    condition_users = set(posts[mask]["author"].unique())
    logging.info(f"  {label} via regex: {len(condition_users)} users")
    return condition_users


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD RAW FEATURES + MERGE PHASE LABELS
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_with_phases(interim_dir: Path, labeled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Load raw _mean features from timeline and merge phase labels from ml_labeled_days.

    Returns (merged_df, raw_feature_cols).
    """
    timeline_path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    if timeline_path is None:
        timeline_path = find_latest_file(interim_dir, "timeline_daily_aggregated_*.csv")
    if timeline_path is None:
        raise FileNotFoundError("No timeline_daily_aggregated_*.csv found.")

    logging.info(f"  Loading timeline: {timeline_path.name}")
    timeline = pd.read_csv(timeline_path, encoding="utf-8-sig", low_memory=False)
    timeline["author"] = timeline["author"].astype(str)

    raw_cols = [c for c in timeline.columns if c.endswith("_mean")]
    logging.info(f"  {len(raw_cols)} raw feature columns found")

    # Keep only consensus users that appear in ml_labeled_days
    consensus_users = set(labeled["author"].unique())
    timeline = timeline[timeline["author"].isin(consensus_users)].copy()

    # Merge phase labels
    phase_map = labeled[["author", "offset_from_cd1", "phase"]].copy()
    merged = timeline.merge(phase_map, on=["author", "offset_from_cd1"], how="inner")
    logging.info(
        f"  After merging phase labels: {len(merged):,} rows | "
        f"{merged['author'].nunique():,} users"
    )
    return merged, raw_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 3. USER-LEVEL FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_user_phase_features(
    labeled_days: pd.DataFrame,
    feature_cols: list[str],
    min_days_per_phase: int = 1,
) -> pd.DataFrame:
    """Pivot to one row per user: mean raw value per feature per phase + deltas."""
    logging.info("  Building user-level feature matrix…")

    user_phase = (
        labeled_days.groupby(["author", "phase"])[feature_cols]
        .mean()
        .reset_index()
    )

    # Quality filter: min days per phase
    day_counts = (
        labeled_days.groupby(["author", "phase"])
        .size()
        .unstack("phase", fill_value=0)
    )
    enough = (day_counts >= min_days_per_phase).all(axis=1)
    valid_users = set(day_counts.index[enough])
    dropped = labeled_days["author"].nunique() - len(valid_users)
    if dropped:
        logging.info(f"  Dropped {dropped} users with < {min_days_per_phase} days in a phase")

    user_phase = user_phase[user_phase["author"].isin(valid_users)].copy()

    wide = user_phase.pivot(index="author", columns="phase", values=feature_cols)
    wide.columns = [f"{feat}__{phase}" for feat, phase in wide.columns]
    wide = wide.reset_index()

    # Delta features
    for feat in feature_cols:
        lut = f"{feat}__Luteal"
        fol = f"{feat}__Follicular"
        men = f"{feat}__Menstrual"
        if lut in wide.columns and fol in wide.columns:
            wide[f"{feat}__delta_LutealMinusFol"] = wide[lut] - wide[fol]
        if men in wide.columns and lut in wide.columns:
            wide[f"{feat}__delta_MenstrualMinusLut"] = wide[men] - wide[lut]
        if men in wide.columns and fol in wide.columns:
            wide[f"{feat}__delta_MenstrualMinusFol"] = wide[men] - wide[fol]

    logging.info(f"  Matrix: {len(wide)} users × {wide.shape[1]-1} features")
    return wide


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_top_features(X, y, feature_names, n_features):
    f_stats, _ = f_classif(np.nan_to_num(X, nan=0.0), y)
    f_stats = np.nan_to_num(f_stats, nan=0.0)
    ranked = np.argsort(f_stats)[::-1]
    logging.info(f"\n  Top {n_features} features by F-statistic:")
    for i in ranked[:n_features]:
        logging.info(f"    {f_stats[i]:8.2f}  {feature_names[i]}")
    return list(ranked[:n_features])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_oof(model, X, y, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = np.empty_like(y)
    y_proba = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in skf.split(X, y):
        m = clone(model)
        sw = compute_sample_weight("balanced", y[train_idx])
        m.fit(X[train_idx], y[train_idx], sample_weight=sw)
        y_pred[test_idx] = m.predict(X[test_idx])
        if hasattr(m, "predict_proba"):
            y_proba[test_idx] = m.predict_proba(X[test_idx])[:, 1]
    return y_pred, y_proba


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def _ts():
    return _TIMESTAMP

def _macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def train_tree(X, y, feature_names, label_names, n_folds, max_depth, output_dir, tag):
    tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=42)
    y_pred, _ = stratified_oof(tree, X, y, n_folds)
    report = classification_report(y, y_pred, target_names=label_names, digits=3)
    logging.info(f"\nDecision Tree OOF:\n{report}")

    rpt_path = output_dir / f"unnorm_{tag}_tree_report_{_ts()}.txt"
    rpt_path.write_text(f"DecisionTree (depth={max_depth}) | StratifiedKFold ({n_folds} folds)\n\n{report}")

    tree.fit(X, y)
    fig_w = max(16, 2 ** max_depth * 2)
    fig_h = max(8, max_depth * 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(tree, feature_names=feature_names, class_names=label_names,
              filled=True, rounded=True, impurity=True, fontsize=8, ax=ax)
    f1 = _macro_f1(y, y_pred)
    ax.set_title(f"Decision Tree (depth={max_depth}) — {tag} [unnormalized]\nOOF Macro-F1 = {f1:.3f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = output_dir / f"unnorm_{tag}_decision_tree_{_ts()}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Tree plot → {plot_path.name}")
    return tree


def train_ensemble(X, y, feature_names, label_names, n_folds, output_dir, tag):
    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=42, n_jobs=-1)
        model_name = "RandomForest"

    y_pred, y_proba = stratified_oof(model, X, y, n_folds)
    report = classification_report(y, y_pred, target_names=label_names, digits=3)
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)

    logging.info(f"\n{model_name} OOF:\n{report}")
    logging.info(f"  AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  "
                 f"(chance PR = {y.mean():.3f})")

    rpt_path = output_dir / f"unnorm_{tag}_ensemble_report_{_ts()}.txt"
    rpt_path.write_text(
        f"{model_name} [UNNORMALIZED] | StratifiedKFold ({n_folds} folds)\n\n{report}\n"
        f"AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  (chance = {y.mean():.3f})\n"
    )

    fpr, tpr, _ = roc_curve(y, y_proba)
    prec, rec, _ = precision_recall_curve(y, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC — {tag} [unnormalized]")
    axes[0].legend()

    axes[1].plot(rec, prec, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y.mean(), color="k", linestyle="--", lw=1,
                    label=f"Chance = {y.mean():.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR — {tag} [unnormalized]")
    axes[1].legend()

    plt.tight_layout()
    roc_path = output_dir / f"unnorm_{tag}_roc_pr_{_ts()}.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  ROC/PR curves → {roc_path.name}")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} [unnorm] — {tag}\nAUC={roc_auc:.3f}", fontsize=10)
    plt.tight_layout()
    cm_path = output_dir / f"unnorm_{tag}_confusion_matrix_{_ts()}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model, roc_auc, pr_auc


def run_shap(model, X, feature_names, label_names, output_dir, tag, max_display=20):
    logging.info("  Computing SHAP values…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X, feature_names=feature_names, show=False,
                      max_display=max_display, plot_type="dot")
    plt.gca().set_title(
        f"SHAP Feature Impact [UNNORMALIZED] — {tag}\n"
        f"(positive → higher P({label_names[1]}))",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    shap_path = output_dir / f"unnorm_{tag}_shap_{_ts()}.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  SHAP → {shap_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

_TIMESTAMP = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--condition", choices=["pmdd", "adhd", "depression"], default="pmdd")
    p.add_argument("--min-days-per-phase", type=int, default=1)
    p.add_argument("--n-features", type=int, default=25)
    p.add_argument("--tree-depth", type=int, default=4, choices=[2, 3, 4, 5])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    global _TIMESTAMP
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = load_config(args.config)
    interim_dir = Path(cfg["paths"]["interim"])
    output_dir = ROOT / "reports" / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_label = CONDITION_CONFIG[args.condition]["label"]
    tag = f"{args.condition}_regex"

    # [1] Load ml_labeled_days for phase labels + consensus user list
    logging.info("\n[1/8] Loading ml_labeled_days (phase labels + consensus users)…")
    labeled_path = find_latest_file(interim_dir, "ml_labeled_days_*.csv")
    if labeled_path is None:
        logging.error("No ml_labeled_days_*.csv found. Run scripts/12_ml_phase_prediction.py first.")
        sys.exit(1)
    logging.info(f"  {labeled_path.name}")
    labeled = pd.read_csv(labeled_path, encoding="utf-8-sig",
                          usecols=["author", "offset_from_cd1", "phase"], low_memory=False)
    labeled["author"] = labeled["author"].astype(str)
    logging.info(f"  {len(labeled):,} user-days | {labeled['author'].nunique():,} consensus users")

    # [2] Load raw features from timeline + merge phase labels
    logging.info("\n[2/8] Loading raw features from timeline…")
    raw_days, raw_cols = load_raw_with_phases(interim_dir, labeled)

    # [3] Identify condition users
    logging.info(f"\n[3/8] Identifying {condition_label} users (regex)…")
    candidate_users = set(raw_days["author"].unique())
    condition_users = identify_condition_users(interim_dir, args.condition, candidate_users)
    control_users = candidate_users - condition_users
    logging.info(f"  {condition_label}: {len(condition_users)} | Control: {len(control_users)}")
    if len(condition_users) < 10:
        logging.error(f"Too few {condition_label} users. Exiting.")
        sys.exit(1)

    # [4] Build user-level feature matrix
    logging.info("\n[4/8] Building per-phase user feature matrix (raw values)…")
    user_features = build_user_phase_features(raw_days, raw_cols, args.min_days_per_phase)

    user_features["label"] = user_features["author"].apply(
        lambda u: condition_label if u in condition_users else "Control"
    )
    user_features = user_features[
        user_features["author"].isin(condition_users | control_users)
    ].copy()

    label_counts = user_features["label"].value_counts()
    logging.info(f"  After quality filter:\n{label_counts.to_string()}")
    if label_counts.min() < 5:
        logging.error("One class has < 5 users. Exiting.")
        sys.exit(1)

    # [5] Build X, y
    logging.info("\n[5/8] Preparing ML arrays…")
    feat_cols = [c for c in user_features.columns if c not in ("author", "label")]
    X_full = user_features[feat_cols].fillna(0).values.astype(np.float32)
    label_map = {"Control": 0, condition_label: 1}
    y = user_features["label"].map(label_map).values
    label_names = ["Control", condition_label]
    logging.info(f"  X shape: {X_full.shape} | prevalence: {y.mean():.3f}")

    # [6] Feature selection
    logging.info(f"\n[6/8] Selecting top {args.n_features} features…")
    top_idx = select_top_features(X_full, y, feat_cols, args.n_features)
    X = X_full[:, top_idx]
    selected_names = [feat_cols[i] for i in top_idx]

    n_folds = min(args.n_folds, label_counts.min())
    if n_folds < args.n_folds:
        logging.warning(f"  Reducing folds to {n_folds}")

    # [7] Decision Tree
    logging.info(f"\n[7/8] Decision Tree (depth={args.tree_depth})…")
    train_tree(X, y, selected_names, label_names, n_folds, args.tree_depth, output_dir, tag)

    # [8] Ensemble + SHAP
    logging.info("\n[8/8] XGBoost ensemble…")
    ensemble, roc_auc, pr_auc = train_ensemble(
        X, y, selected_names, label_names, n_folds, output_dir, tag
    )
    if not args.skip_shap:
        run_shap(ensemble, X, selected_names, label_names, output_dir, tag)
    else:
        logging.info("  SHAP skipped.")

    logging.info(f"\nFINAL: {condition_label} vs Control [UNNORMALIZED]")
    logging.info(f"  AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  "
                 f"(chance = {y.mean():.3f})")
    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
