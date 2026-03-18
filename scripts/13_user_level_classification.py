"""Step 13 — User-Level Condition Classification
================================================
Classifies users as PMDD / ADHD / control based on their per-phase linguistic
fingerprint — a denoised, user-level feature vector computed by averaging each
z-score feature across all of a user's days in each cycle phase.

Why user-level instead of day-level?
  Day-level classification (step 12) failed because the cycle-phase signal
  (~0.1 z-score) is swamped by daily noise (std≈1).  Averaging a user's 10+
  Luteal days reduces noise by √10, recovering the signal.  Each user becomes
  ONE data point: their personal linguistic fingerprint per phase.

User identification — two methods (combinable):
  1. Subreddit: user ever posted in r/PMDD, r/ADHD, etc.
  2. Regex self-report: user's post text contains phrases like
     "I have PMDD", "diagnosed with ADHD", "my PMDD", etc.
     This is stricter — explicit self-declaration of diagnosis.

Feature engineering per user:
  For each (feature, phase) pair → mean z-score across all user-days in that
  phase.  Also adds Luteal−Follicular and Menstrual−Luteal delta features to
  capture the premenstrual transition (biologically most relevant for PMDD).

Pipeline
  [1] Load ml_labeled_days (step 12 output) — consensus users, z-scores, phases
  [2] Load posts — identify condition users via subreddit and/or regex
  [3] Build user-level feature matrix (pivot: mean z-score per user per phase)
  [4] Add delta features (Luteal−Follicular, Menstrual−Luteal)
  [5] ANOVA feature selection (condition vs control, binary)
  [6] Decision tree (interpretable)
  [7] XGBoost + SHAP

Output (reports/ml/)
  user_level_{condition}_classification_report_*.txt
  user_level_{condition}_decision_tree_*.png
  user_level_{condition}_confusion_matrix_*.png
  user_level_{condition}_shap_*.png

Usage
  python scripts/13_user_level_classification.py --condition pmdd
  python scripts/13_user_level_classification.py --condition adhd
  python scripts/13_user_level_classification.py --condition pmdd --id-method regex
  python scripts/13_user_level_classification.py --condition pmdd --id-method both
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
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

# ── Regex patterns for self-reported diagnosis ────────────────────────────────
# Hardcoded per-condition patterns capturing natural Reddit self-report language.
# Each pattern is case-insensitive and tries to match explicit first-person
# statements, diagnoses, or strong self-identifications.

CONDITION_REGEX = {
    "pmdd": re.compile(
        r"""
        # Direct self-report
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
               am\s+dealing\s+with|was\s+told\s+I\s+have)\s+PMDD
        |
        # Diagnosis mention
        (?:my|a)\s+PMDD\s+(?:diagnosis|symptoms?|episodes?|flare)
        |
        # Formal diagnosis
        diagnosed\s+(?:me\s+)?with\s+PMDD
        |
        # Identity / living with
        (?:PMDD\s+sufferer|living\s+with\s+PMDD|PMDD\s+warrior|my\s+PMDD)
        |
        # Treatment mentions (strong signal of having it)
        (?:my\s+PMDD|for\s+my\s+PMDD|treating\s+my\s+PMDD|manage\s+my\s+PMDD)
        |
        # "as someone with PMDD"
        as\s+(?:someone|a\s+(?:person|woman))\s+with\s+PMDD
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    "adhd": re.compile(
        r"""
        # Direct self-report
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
               was\s+told\s+I\s+have)\s+ADHD
        |
        # Diagnosis mention
        (?:my|a)\s+ADHD\s+(?:diagnosis|symptoms?|brain|medication|meds|treatment)
        |
        # Formal diagnosis
        diagnosed\s+(?:me\s+)?with\s+ADHD
        |
        # Identity / living with
        (?:ADHDer|living\s+with\s+ADHD|my\s+ADHD|as\s+someone\s+with\s+ADHD)
        |
        # Treatment mentions
        (?:my\s+ADHD\s+meds?|my\s+ADHD\s+medication|treating\s+my\s+ADHD|
           manage\s+my\s+ADHD|ADHD\s+tax|ADHD\s+paralysis|ADHD\s+brain)
        |
        # "as someone with ADHD"
        as\s+(?:someone|a\s+(?:person|woman))\s+with\s+ADHD
        |
        # Comorbidity with PMDD (common overlap population)
        ADHD\s+and\s+PMDD|PMDD\s+and\s+ADHD
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    "depression": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|struggle\s+with)\s+
               (?:depression|MDD|major\s+depressive\s+disorder|clinical\s+depression)
        |
        diagnosed\s+(?:me\s+)?with\s+(?:depression|MDD|major\s+depressive\s+disorder)
        |
        (?:my|a)\s+(?:depression|MDD)\s+(?:diagnosis|symptoms?|episodes?|medication|meds)
        |
        as\s+(?:someone|a\s+(?:person|woman))\s+with\s+(?:depression|MDD)
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
}

CONDITION_CONFIG = {
    "pmdd": {
        "subreddits": {"PMDD", "PMDDxADHD", "PMDDSharing"},
        "label": "PMDD",
    },
    "adhd": {
        "subreddits": {"ADHD", "adhdwomen", "TwoXADHD", "ADHDWomenAfterDark",
                       "PMDDxADHD", "adhd_anxiety", "adhdmeme", "ADHDers"},
        "label": "ADHD",
    },
    "depression": {
        "subreddits": {"depression", "mentalhealth", "Anxiety",
                       "socialanxiety", "AnxietyDepression"},
        "label": "Depression",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. USER IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def identify_condition_users(
    interim_dir: Path,
    condition: str,
    id_method: str,
    candidate_users: set[str],
) -> set[str]:
    """Return the set of candidate_users who have the condition.

    Args:
        interim_dir: data/interim path.
        condition: one of 'pmdd', 'adhd', 'depression'.
        id_method: 'subreddit', 'regex', or 'both'.
        candidate_users: restrict to this set (consensus users).
    """
    cfg = CONDITION_CONFIG[condition]
    subreddits = cfg["subreddits"]
    regex = CONDITION_REGEX[condition]
    label = cfg["label"]

    posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_with_anchors_*.csv")
    if posts_path is None:
        posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_*.csv")
    if posts_path is None:
        raise FileNotFoundError("No posts_all_users_preprocessed_*.csv found.")

    logging.info(f"Loading posts for user identification: {posts_path.name}")
    usecols = ["author", "subreddit", "text"] if id_method != "subreddit" else ["author", "subreddit"]
    posts = pd.read_csv(posts_path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    posts = posts[posts["author"].astype(str).isin(candidate_users)].copy()
    posts["author"] = posts["author"].astype(str)
    logging.info(f"  Posts from {posts['author'].nunique():,} candidate users loaded")

    sub_users: set[str] = set()
    regex_users: set[str] = set()

    if id_method in ("subreddit", "both"):
        sub_users = set(
            posts[posts["subreddit"].isin(subreddits)]["author"].unique()
        )
        logging.info(f"  {label} via subreddit: {len(sub_users)} users")

    if id_method in ("regex", "both"):
        mask = posts["text"].fillna("").str.contains(regex, regex=True)
        regex_users = set(posts[mask]["author"].unique())
        logging.info(f"  {label} via self-report regex: {len(regex_users)} users")

    if id_method == "subreddit":
        condition_users = sub_users
    elif id_method == "regex":
        condition_users = regex_users
    else:  # both — union
        condition_users = sub_users | regex_users
        logging.info(f"  {label} combined (union): {len(condition_users)} users")

    return condition_users


# ═══════════════════════════════════════════════════════════════════════════════
# 2. USER-LEVEL FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_user_phase_features(
    labeled_days: pd.DataFrame,
    zscore_cols: list[str],
    min_days_per_phase: int = 3,
) -> pd.DataFrame:
    """Pivot day-level data to one row per user with per-phase mean z-scores.

    For each user, computes mean z-score per feature per phase.
    Users without at least min_days_per_phase days in EVERY phase are dropped
    to ensure reliable per-phase estimates.

    Also adds delta features:
      Luteal − Follicular  (premenstrual change)
      Menstrual − Luteal   (menstrual vs premenstrual)
      Menstrual − Follicular (menstrual vs mid-cycle baseline)

    Returns a DataFrame with one row per user.
    """
    logging.info(f"\n  Building user-level feature matrix…")

    # Mean z-score per user per phase
    user_phase = (
        labeled_days.groupby(["author", "phase"])[zscore_cols]
        .mean()
        .reset_index()
    )

    # Count days per user per phase for quality filter
    day_counts = (
        labeled_days.groupby(["author", "phase"])
        .size()
        .unstack("phase", fill_value=0)
    )
    # Keep users with at least min_days_per_phase in EVERY phase
    enough = (day_counts >= min_days_per_phase).all(axis=1)
    valid_users = set(day_counts.index[enough])
    dropped = labeled_days["author"].nunique() - len(valid_users)
    if dropped:
        logging.info(
            f"  Dropped {dropped} users with < {min_days_per_phase} days "
            f"in at least one phase"
        )

    user_phase = user_phase[user_phase["author"].isin(valid_users)].copy()

    # Pivot to wide format: columns = feature_Phase
    wide = user_phase.pivot(index="author", columns="phase", values=zscore_cols)
    wide.columns = [f"{feat}__{phase}" for feat, phase in wide.columns]
    wide = wide.reset_index()

    # Delta features
    for feat in zscore_cols:
        lut = f"{feat}__Luteal"
        fol = f"{feat}__Follicular"
        men = f"{feat}__Menstrual"
        if lut in wide.columns and fol in wide.columns:
            wide[f"{feat}__delta_LutealMinusFol"] = wide[lut] - wide[fol]
        if men in wide.columns and lut in wide.columns:
            wide[f"{feat}__delta_MenstrualMinusLut"] = wide[men] - wide[lut]
        if men in wide.columns and fol in wide.columns:
            wide[f"{feat}__delta_MenstrualMinusFol"] = wide[men] - wide[fol]

    logging.info(
        f"  User-level matrix: {len(wide)} users × {wide.shape[1]-1} features"
    )
    return wide


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features: int,
) -> list[int]:
    """Return indices of top N features by binary F-statistic (condition vs control)."""
    f_stats, _ = f_classif(np.nan_to_num(X, nan=0.0), y)
    f_stats = np.nan_to_num(f_stats, nan=0.0)
    ranked = np.argsort(f_stats)[::-1]

    logging.info(f"\n  Top {n_features} features (condition vs control):")
    for i in ranked[:n_features]:
        logging.info(f"    {f_stats[i]:8.2f}  {feature_names[i]}")

    return list(ranked[:n_features])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION (stratified, user-level)
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_oof(model, X, y, n_folds: int):
    """Stratified k-fold OOF predictions + probabilities.

    Returns (y_pred, y_proba) where y_proba is the probability of the
    positive class (index 1).  Probabilities enable AUC-ROC and PR-AUC
    which are the right metrics for imbalanced binary classification.
    """
    from sklearn.utils.class_weight import compute_sample_weight
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
# 5. MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def train_tree(X, y, feature_names, label_names, n_folds, max_depth, output_dir, tag):
    tree = DecisionTreeClassifier(
        max_depth=max_depth, class_weight="balanced", random_state=42
    )
    y_pred, y_proba = stratified_oof(tree, X, y, n_folds)
    report = classification_report(y, y_pred, target_names=label_names, digits=3)
    logging.info(f"\nDecision Tree OOF:\n{report}")

    rpt_path = output_dir / f"user_level_{tag}_tree_report_{_ts()}.txt"
    rpt_path.write_text(f"DecisionTree (depth={max_depth}) | StratifiedKFold ({n_folds} folds)\n\n{report}")

    tree.fit(X, y)
    fig_w = max(16, 2 ** max_depth * 2)
    fig_h = max(8, max_depth * 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(tree, feature_names=feature_names, class_names=label_names,
              filled=True, rounded=True, impurity=True, fontsize=8, ax=ax)
    f1 = _macro_f1(y, y_pred)
    ax.set_title(
        f"Decision Tree (depth={max_depth}) — {tag}\n"
        f"StratifiedKFold OOF Macro-F1 = {f1:.3f}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = output_dir / f"user_level_{tag}_decision_tree_{_ts()}.png"
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
        model_name = "XGBoostClassifier"
    else:
        model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )
        model_name = "RandomForestClassifier"

    y_pred, y_proba = stratified_oof(model, X, y, n_folds)
    report = classification_report(y, y_pred, target_names=label_names, digits=3)
    logging.info(f"\n{model_name} OOF:\n{report}")

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    logging.info(f"  AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  "
                 f"(chance PR = {y.mean():.3f})")

    rpt_path = output_dir / f"user_level_{tag}_ensemble_report_{_ts()}.txt"
    rpt_path.write_text(
        f"{model_name} | StratifiedKFold ({n_folds} folds)\n\n{report}\n"
        f"AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  "
        f"(chance = {y.mean():.3f})\n"
    )

    # ROC + PR curves side by side
    fpr, tpr, _ = roc_curve(y, y_proba)
    prec, rec, _ = precision_recall_curve(y, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {tag}")
    axes[0].legend()

    axes[1].plot(rec, prec, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y.mean(), color="k", linestyle="--", lw=1,
                    label=f"Chance = {y.mean():.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {tag}")
    axes[1].legend()

    plt.tight_layout()
    roc_path = output_dir / f"user_level_{tag}_roc_pr_{_ts()}.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  ROC/PR curves → {roc_path.name}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, cmap="Blues")
    f1 = _macro_f1(y, y_pred)
    ax.set_title(f"{model_name} — {tag}\nOOF Macro-F1={f1:.3f}  AUC={roc_auc:.3f}", fontsize=10)
    plt.tight_layout()
    cm_path = output_dir / f"user_level_{tag}_confusion_matrix_{_ts()}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    from sklearn.utils.class_weight import compute_sample_weight
    model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model


def run_shap(model, X, feature_names, label_names, output_dir, tag, max_display=20):
    logging.info("  Computing SHAP values…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary XGBoost shap_values is (n_samples, n_features) — positive class
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
        f"SHAP Feature Impact — {tag}\n"
        f"(positive → higher P({label_names[1]}))",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    shap_path = output_dir / f"user_level_{tag}_shap_{_ts()}.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  SHAP → {shap_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_TIMESTAMP = None

def _ts():
    return _TIMESTAMP

def _macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument(
        "--condition", choices=["pmdd", "adhd", "depression"], default="pmdd",
        help="Which condition to classify against controls.",
    )
    p.add_argument(
        "--id-method", choices=["subreddit", "regex", "both"], default="both",
        help="How to identify condition users: subreddit membership, "
             "regex self-report in post text, or both (union).",
    )
    p.add_argument(
        "--min-days-per-phase", type=int, default=3,
        help="Minimum days a user must have in EVERY phase to be included.",
    )
    p.add_argument(
        "--n-features", type=int, default=25,
        help="Top N features by binary F-statistic for model training.",
    )
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

    tag = f"{args.condition}_{args.id_method}"

    # ── [1] Load ml_labeled_days ────────────────────────────────────────────────
    logging.info("\n[1/7] Loading ml_labeled_days (step 12 output)…")
    labeled_path = find_latest_file(interim_dir, "ml_labeled_days_*.csv")
    if labeled_path is None:
        logging.error("No ml_labeled_days_*.csv found. Run scripts/12_ml_phase_prediction.py first.")
        sys.exit(1)
    logging.info(f"  Loading: {labeled_path.name}")
    labeled = pd.read_csv(labeled_path, encoding="utf-8-sig", low_memory=False)
    labeled["author"] = labeled["author"].astype(str)
    zscore_cols = [c for c in labeled.columns if c.endswith("_zscore")]
    logging.info(
        f"  {len(labeled):,} user-days | {labeled['author'].nunique():,} users | "
        f"{len(zscore_cols)} z-score features"
    )

    # ── [2] Identify condition users ────────────────────────────────────────────
    logging.info(f"\n[2/7] Identifying {args.condition.upper()} users ({args.id_method})…")
    candidate_users = set(labeled["author"].unique())
    condition_users = identify_condition_users(
        interim_dir, args.condition, args.id_method, candidate_users
    )
    control_users = candidate_users - condition_users
    logging.info(
        f"  {args.condition.upper()}: {len(condition_users)} | "
        f"Control: {len(control_users)} | "
        f"Total: {len(condition_users) + len(control_users)}"
    )
    if len(condition_users) < 10:
        logging.error(f"Too few {args.condition.upper()} users ({len(condition_users)}). Exiting.")
        sys.exit(1)

    # ── [3] Build user-level feature matrix ────────────────────────────────────
    logging.info("\n[3/7] Building per-phase user feature matrix…")
    user_features = build_user_phase_features(labeled, zscore_cols, args.min_days_per_phase)

    # Attach labels
    condition_label = CONDITION_CONFIG[args.condition]["label"]
    user_features["label"] = user_features["author"].apply(
        lambda u: condition_label if u in condition_users else "Control"
    )
    # Drop users not in either group (shouldn't happen but be safe)
    user_features = user_features[
        user_features["author"].isin(condition_users | control_users)
    ].copy()

    label_counts = user_features["label"].value_counts()
    logging.info(f"  After quality filter:\n{label_counts.to_string()}")

    if label_counts.min() < 5:
        logging.error("One class has < 5 users after quality filter. Exiting.")
        sys.exit(1)

    # ── [4] Build X, y ─────────────────────────────────────────────────────────
    logging.info("\n[4/7] Preparing ML arrays…")
    feat_cols = [c for c in user_features.columns if c not in ("author", "label")]
    X_full = user_features[feat_cols].fillna(0).values.astype(np.float32)
    # Encode explicitly: 0=Control, 1=Condition (don't rely on alphabetical sort)
    label_map = {"Control": 0, condition_label: 1}
    y = user_features["label"].map(label_map).values
    label_names = ["Control", condition_label]
    logging.info(f"  X shape: {X_full.shape} | classes: {label_names}")

    # ── [5] Feature selection ──────────────────────────────────────────────────
    logging.info(f"\n[5/7] Selecting top {args.n_features} features…")
    top_idx = select_top_features(X_full, y, feat_cols, args.n_features)
    X = X_full[:, top_idx]
    selected_names = [feat_cols[i] for i in top_idx]

    n_folds = min(args.n_folds, label_counts.min())
    if n_folds < args.n_folds:
        logging.warning(f"  Reducing folds to {n_folds} (minority class size)")

    # ── [6] Decision Tree ──────────────────────────────────────────────────────
    logging.info(f"\n[6/7] Decision Tree (depth={args.tree_depth})…")
    tree = train_tree(X, y, selected_names, label_names,
                      n_folds, args.tree_depth, output_dir, tag)

    # ── [7] Ensemble + SHAP ────────────────────────────────────────────────────
    logging.info("\n[7/7] Ensemble model…")
    ensemble = train_ensemble(X, y, selected_names, label_names,
                              n_folds, output_dir, tag)
    if not args.skip_shap:
        run_shap(ensemble, X, selected_names, label_names, output_dir, tag)
    else:
        logging.info("  SHAP skipped.")

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
