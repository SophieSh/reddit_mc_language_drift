"""Step 25 — PMDD User-Level Classifier (absolute features, fixed 29-day cycle)
==============================================================================
Classifies users as PMDD vs Control using per-phase linguistic feature profiles
built from the fixed 29-day cycle assumption applied to all timeline users.

Key design decision — NO per-user z-scoring:
  PMDD users have chronically elevated negative sentiment across ALL phases.
  Z-scoring each user to their own mean removes this between-user baseline
  difference, which is the strongest discriminating signal. Using absolute
  feature values (raw per-user-per-phase means) preserves it.
  AUC with z-scoring: ~0.51 (chance). AUC without: ~0.70.

Feature engineering:
  For each user: mean of each feature across all posts in each of the 5
  Eisenlohr-Moul phases (Perimenstrual, Midfollicular, Periovulatory,
  Early Luteal, Midluteal). Also adds Perimenstrual−Midfollicular delta
  features to capture the premenstrual transition.

Pipeline:
  [1] Load timeline_with_offsets_with_anchors (±30d window, per-post features)
  [2] Identify PMDD users from pmdd_users_*.csv (subreddit-identified)
  [3] Assign 5-phase PMDD labels via fixed 29-day cycle (offset_from_cd1 % 29)
  [4] Build user × (phase × feature) matrix — no z-scoring
  [5] Add delta features (Perimenstrual − Midfollicular)
  [6] ANOVA feature selection → top N features
  [7] XGBoost with scale_pos_weight, StratifiedKFold 5-fold OOF
  [8] SHAP beeswarm + ROC/PR curves + classification report

Output (reports/pmdd/):
  pmdd_user_classifier_report_*.txt
  pmdd_user_classifier_roc_pr_*.png
  pmdd_user_classifier_shap_*.png

Usage:
  python scripts/25_pmdd_user_classifier.py
  python scripts/25_pmdd_user_classifier.py --top-features 50
  python scripts/25_pmdd_user_classifier.py --skip-shap
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
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file
from src.visualization import assign_pmdd_phase, create_pmdd_phases

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

CYCLE_LENGTH = 29.0
PHASE_ORDER = ["Perimenstrual", "Midfollicular", "Periovulatory", "Early Luteal", "Midluteal"]

# Columns that are not linguistic features
NON_FEATURE_COLS = {
    "author", "author_lower", "author_flair_text", "group",
    "created_utc", "subreddit", "offset_from_cd1", "offset_mod",
    "phase", "post_id", "permalink", "title", "selftext",
}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--top-features", type=int, default=30,
                   help="Number of top features to keep after ANOVA selection (default: 30)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_data(cfg, interim_dir: Path, processed_dir: Path):
    timeline_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    logging.info(f"Loading timeline: {timeline_path.name}")
    tl = pd.read_csv(timeline_path, low_memory=False)

    pmdd_path = find_latest_file(processed_dir, "pmdd_users_*.csv")
    logging.info(f"Loading PMDD users: {pmdd_path.name}")
    pmdd_users = pd.read_csv(pmdd_path)
    pmdd_set = set(pmdd_users["user"].str.lower())

    return tl, pmdd_set


def assign_groups_and_phases(tl: pd.DataFrame, pmdd_set: set) -> pd.DataFrame:
    tl = tl.copy()
    tl["author_lower"] = tl["author"].str.lower()
    tl["group"] = tl["author_lower"].apply(lambda x: "PMDD" if x in pmdd_set else "Control")
    tl["offset_mod"] = tl["offset_from_cd1"] % CYCLE_LENGTH

    phases = create_pmdd_phases(CYCLE_LENGTH)
    tl["phase"] = tl["offset_mod"].apply(lambda x: assign_pmdd_phase(x, phases, CYCLE_LENGTH))
    tl = tl[tl["phase"].notna()].copy()
    return tl


def get_feature_cols(tl: pd.DataFrame) -> list[str]:
    feat_cols = [
        c for c in tl.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(tl[c])
        and tl[c].notna().mean() > 0.3
    ]
    return feat_cols


def build_feature_matrix(tl: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Build user × (phase × feature) matrix. No z-scoring — absolute values."""
    up = tl.groupby(["author", "group", "phase"])[feat_cols].mean().reset_index()

    records = []
    for (author, group), udf in up.groupby(["author", "group"]):
        rec = {"author": author, "group": group}
        for ph in PHASE_ORDER:
            phdf = udf[udf["phase"] == ph]
            for f in feat_cols:
                rec[f"{ph}__{f}"] = phdf[f].values[0] if len(phdf) > 0 else np.nan
        records.append(rec)

    wide = pd.DataFrame(records)

    # Drop columns with >50% missing
    phase_feat_cols = [c for c in wide.columns if "__" in c]
    phase_feat_cols = [c for c in phase_feat_cols if wide[c].notna().mean() > 0.5]

    # Add Perimenstrual − Midfollicular delta features
    delta_cols = []
    for f in feat_cols:
        c_peri = f"Perimenstrual__{f}"
        c_mid = f"Midfollicular__{f}"
        if c_peri in phase_feat_cols and c_mid in phase_feat_cols:
            col = f"delta_peri_mid__{f}"
            wide[col] = wide[c_peri] - wide[c_mid]
            delta_cols.append(col)

    all_feat_cols = phase_feat_cols + delta_cols
    all_feat_cols = [c for c in all_feat_cols if wide[c].notna().mean() > 0.5]
    wide = wide.dropna(subset=all_feat_cols)

    return wide, all_feat_cols


def select_top_features(X: np.ndarray, y: np.ndarray, feature_names: list[str], top_n: int):
    F, _ = f_classif(np.nan_to_num(X), y)
    top_idx = np.argsort(-F)[:top_n]
    return X[:, top_idx], [feature_names[i] for i in top_idx]


def stratified_oof(model, X, y, n_folds):
    y_pred = np.zeros(len(y), dtype=int)
    y_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
        y_proba[te] = m.predict_proba(X[te])[:, 1]
    return y_pred, y_proba


def build_model(scale_pos_weight: float):
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )


def save_report(output_dir: Path, tag: str, report_str: str, roc_auc: float,
                pr_auc: float, chance: float, model_name: str, n_folds: int):
    path = output_dir / f"pmdd_user_classifier_report_{tag}.txt"
    path.write_text(
        f"{model_name} | StratifiedKFold ({n_folds} folds)\n\n"
        f"{report_str}\n"
        f"AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  (chance = {chance:.3f})\n"
    )
    logging.info(f"  Report → {path.name}")


def plot_roc_pr(y, y_proba, roc_auc, pr_auc, output_dir, tag):
    fpr, tpr, _ = roc_curve(y, y_proba)
    prec, rec, _ = precision_recall_curve(y, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — PMDD vs Control")
    axes[0].legend()

    axes[1].plot(rec, prec, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y.mean(), color="k", linestyle="--", lw=1,
                    label=f"Chance = {y.mean():.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — PMDD vs Control")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / f"pmdd_user_classifier_roc_pr_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  ROC/PR curves → {path.name}")


def plot_confusion(y, y_pred, roc_auc, output_dir, tag, model_name):
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Control", "PMDD"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} | AUC={roc_auc:.3f}", fontsize=10)
    plt.tight_layout()
    path = output_dir / f"pmdd_user_classifier_confusion_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap(model, X, feature_names, output_dir, tag, max_display=20):
    logging.info("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X, feature_names=feature_names, max_display=max_display,
                      show=False, plot_type="dot")
    plt.title("SHAP — PMDD vs Control (positive = PMDD)")
    plt.tight_layout()
    path = output_dir / f"pmdd_user_classifier_shap_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  SHAP → {path.name}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    interim_dir = ROOT / cfg["paths"]["interim"]
    processed_dir = ROOT / cfg["paths"]["processed"]
    output_dir = ROOT / cfg["paths"]["reports"] / "pmdd"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = _ts()

    # [1] Load
    tl, pmdd_set = load_data(cfg, interim_dir, processed_dir)

    # [2] Assign groups and phases
    tl = assign_groups_and_phases(tl, pmdd_set)
    feat_cols = get_feature_cols(tl)
    logging.info(f"Linguistic features: {len(feat_cols)}")
    logging.info(f"PMDD users: {(tl.group=='PMDD')['author'].nunique() if False else tl[tl.group=='PMDD']['author'].nunique()}, "
                 f"Control: {tl[tl.group=='Control']['author'].nunique()}")

    # [3] Build feature matrix
    logging.info("Building user × phase feature matrix (no z-scoring)...")
    wide, all_feat_cols = build_feature_matrix(tl, feat_cols)

    y = (wide["group"] == "PMDD").astype(int).values
    pmdd_n, ctrl_n = y.sum(), (y == 0).sum()
    logging.info(f"Feature matrix: {len(wide)} users ({pmdd_n} PMDD, {ctrl_n} Control), "
                 f"{len(all_feat_cols)} features")

    # [4] Feature selection
    X_all = wide[all_feat_cols].values
    X, top_names = select_top_features(X_all, y, all_feat_cols, args.top_features)
    logging.info(f"Top {args.top_features} features selected by ANOVA F-statistic")
    logging.info(f"  Top 5: {top_names[:5]}")

    # [5] Train + evaluate
    model = build_model(scale_pos_weight=ctrl_n / max(pmdd_n, 1))
    model_name = type(model).__name__
    logging.info(f"Running {args.n_folds}-fold OOF with {model_name}...")

    y_pred, y_proba = stratified_oof(model, X, y, args.n_folds)
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    report_str = classification_report(y, y_pred, target_names=["Control", "PMDD"], digits=3)

    logging.info(f"\n{report_str}")
    logging.info(f"AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  (chance = {y.mean():.3f})")

    # [6] Save outputs
    save_report(output_dir, tag, report_str, roc_auc, pr_auc, y.mean(), model_name, args.n_folds)
    plot_roc_pr(y, y_proba, roc_auc, pr_auc, output_dir, tag)
    plot_confusion(y, y_pred, roc_auc, output_dir, tag, model_name)

    # [7] SHAP — fit on full data
    if not args.skip_shap:
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        plot_shap(model, X, top_names, output_dir, tag)

    logging.info("Done.")


if __name__ == "__main__":
    main()
