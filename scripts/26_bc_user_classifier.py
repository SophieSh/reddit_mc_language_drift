"""Step 26 — BC User-Level Classifier (combined-pill vs stable natural cycle)
==============================================================================
Classifies users as BC (combined hormonal contraception) vs Control (stable
detected-period natural cycle) using per-phase linguistic feature profiles
built from a fixed 28-day cycle.

Groups:
  BC:      two subsets, both combined-pill only (no progestin-only pills):
    (a) Stable users  — no start/stop event anywhere in their posts
    (b) Started users — recently started BC; only posts AFTER start are used.
        Start day is computed per-user as:
          start_cd1 = post.offset_from_cd1 + post.started_offset
        (started_offset is relative to the post date, not CD1)
  Control: stable detected-period users from consensus_periods_*no_bc*.csv

Why combined-pill only?
  Progestin-only pills (POPs) have a different hormonal mechanism and do not
  suppress ovulation in all users. Mixing them with combined pills would
  conflate two distinct biological interventions.

Why 28-day fixed cycle?
  Combined pill packs follow a 28-day schedule (21 active + 7 placebo).
  Applying the same fixed cycle to natural-cycle controls (whose detected
  periods cluster around 28-29 days) makes phases directly comparable.

Phases (offset_from_cd1 % 28, 0-indexed):
  Menstrual   days  1– 5  (offset_mod  0– 4)
  Follicular  days  6–13  (offset_mod  5–12)
  Ovulation   days 14–16  (offset_mod 13–15)
  Luteal      days 17–28  (offset_mod 16–27)

Missing phases:
  Users with no posts in a phase (especially Ovulation, a 3-day window) get
  NaN for that phase's features. XGBoost handles NaN natively by learning the
  optimal split direction for missing values — no imputation needed.

Key design decision — NO per-user z-scoring:
  Raw (absolute) feature values are used. AUC with z-scoring ≈ AUC without
  for BC classification (~0.60 either way at population level), but z-scoring
  halves the usable sample due to NaN propagation. Raw values are preferred.

Pipeline:
  [1] Load bc_wide_candidates_with_labels (LLM labels)
  [2] Filter combined-pill users; split into stable and started subsets
  [3] Load timeline_with_offsets_with_anchors; filter to BC + control posts
      For started users: keep only posts where offset_from_cd1 >= start_cd1
  [4] Load natural-cycle control users from consensus_periods_*no_bc*.csv
  [5] Assign 4-phase labels via fixed 28-day cycle
  [6] Build user × (phase × feature) matrix; missing phases stay NaN
  [7] XGBoost with scale_pos_weight, StratifiedKFold 5-fold OOF
  [8] SHAP beeswarm + ROC/PR curves + classification report

Output (reports/bc/):
  bc_user_classifier_report_*.txt
  bc_user_classifier_roc_pr_*.png
  bc_user_classifier_shap_*.png
  bc_user_classifier_confusion_*.png

Usage:
  python scripts/26_bc_user_classifier.py
  python scripts/26_bc_user_classifier.py --skip-shap
  python scripts/26_bc_user_classifier.py --stable-only
  python scripts/26_bc_user_classifier.py --n-folds 10
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
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

CYCLE_LENGTH = 28.0
PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_BOUNDS = {
    "Menstrual":  (0,  5),
    "Follicular": (5,  13),
    "Ovulation":  (13, 16),
    "Luteal":     (16, 28),
}

# Progestin-only keywords — users whose only pill matches these are excluded
POP_KEYWORDS = [
    "pop", "norethindrone", "norgestrel", "desogestrel", "slynd", "opill",
    "cerazette", "cerelle", "nora-be", "camila", "errin", "jencycla", "lyza",
]

NON_FEATURE_COLS = {
    "author", "author_lower", "group", "created_utc", "subreddit",
    "author_flair_text", "offset_from_cd1", "offset_mod", "phase",
    "post_id", "permalink", "title", "selftext", "start_cd1",
}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--n-folds", type=int, default=None,
                   help="Override bc_classifier.n_folds from config")
    p.add_argument("--stable-only", action="store_true",
                   help="Use only stable BC users (skip recently-started users)")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def is_combined_pill(pill_name) -> bool | None:
    if pd.isna(pill_name):
        return None
    return not any(k in str(pill_name).lower() for k in POP_KEYWORDS)


def load_bc_users(labels_path: Path) -> tuple[set, dict]:
    """Return (stable_set, started_map).

    stable_set:  lowercase author names — stable combined-pill users.
    started_map: author_lower → start_cd1 offset — recently-started combined-pill users.
                 start_cd1 is the offset_from_cd1 at which they began BC.
    """
    try:
        labels = pd.read_excel(labels_path, engine="openpyxl")
    except Exception:
        labels = pd.read_csv(labels_path, encoding="utf-8-sig")

    labels = labels[labels["is_bc_pill"] == True].copy()
    labels["is_combined"] = labels["pill_name"].apply(is_combined_pill)

    user_flags = labels.groupby("author").agg(
        any_started=("started_recently", "any"),
        any_stopped=("stopped_recently", "any"),
        is_comb=("is_combined", lambda x: x.any() and not (x == False).any()),
    ).reset_index()

    stable_set = set(
        user_flags[
            ~user_flags["any_started"] & ~user_flags["any_stopped"] & user_flags["is_comb"]
        ]["author"].str.lower()
    )

    # For started users: compute start_cd1 = post.offset_from_cd1 + post.started_offset
    # (started_offset is relative to the post date, not to CD1)
    started_posts = labels[
        (labels["started_recently"] == True) & (labels["is_combined"] == True)
    ].copy()
    started_posts["start_cd1"] = started_posts["offset_from_cd1"] + started_posts["started_offset"]
    # Use the post where started_offset is closest to 0 (reported closest to actual start)
    started_posts["abs_off"] = started_posts["started_offset"].abs()
    user_start_cd1 = (
        started_posts.sort_values("abs_off")
        .groupby("author")["start_cd1"]
        .first()
    )
    started_df = user_flags[
        user_flags["any_started"] & ~user_flags["any_stopped"] & user_flags["is_comb"]
    ]
    started_map = {
        a.lower(): user_start_cd1[a]
        for a in started_df["author"]
        if a in user_start_cd1.index
    }

    return stable_set, started_map

def load_timeline(interim_dir: Path, files_cfg: dict) -> pd.DataFrame:
    path = find_latest_file(interim_dir, files_cfg["timeline_with_anchors"] + "_*.csv")
    logging.info(f"Timeline: {path.name}")
    tl = pd.read_csv(path, low_memory=False)
    tl["author_lower"] = tl["author"].str.lower()
    return tl


def load_natural_users(interim_dir: Path, files_cfg: dict) -> set:
    path = find_latest_file(interim_dir, files_cfg["consensus_periods"] + "_*no_bc*.csv")
    if path is None:
        path = find_latest_file(interim_dir, files_cfg["consensus_periods"] + "_*.csv")
    logging.info(f"Natural users: {path.name}")
    df = pd.read_csv(path)
    user_col = "user" if "user" in df.columns else df.columns[0]
    return set(df[user_col].str.lower())


def build_combined_timeline(
    tl: pd.DataFrame,
    stable_set: set,
    started_map: dict,
    nat_set: set,
    stable_only: bool,
) -> pd.DataFrame:
    tl_stable = tl[tl["author_lower"].isin(stable_set)].copy()
    tl_stable["group"] = "BC"

    if stable_only:
        tl_bc = tl_stable
        logging.info("BC group: stable users only (--stable-only)")
    else:
        tl_started = tl[tl["author_lower"].isin(started_map)].copy()
        tl_started["start_cd1"] = tl_started["author_lower"].map(started_map)
        tl_started = tl_started[
            tl_started["offset_from_cd1"] >= tl_started["start_cd1"]
        ].copy()
        tl_started["group"] = "BC"
        tl_bc = pd.concat([tl_stable, tl_started], ignore_index=True)

    tl_ctrl = tl[tl["author_lower"].isin(nat_set)].copy()
    tl_ctrl["group"] = "Control"

    combined = pd.concat([tl_bc, tl_ctrl], ignore_index=True)

    overlap = set(tl_bc["author_lower"]) & set(tl_ctrl["author_lower"])
    if overlap:
        logging.warning(f"  {len(overlap)} users appear in both BC and Control — removing from Control")
        combined = combined[
            ~((combined["group"] == "Control") & (combined["author_lower"].isin(overlap)))
        ]

    return combined


def load_data(cfg: dict, interim_dir: Path, labels_path: Path, stable_only: bool) -> pd.DataFrame:
    files_cfg = cfg["paths"]["files"]
    tl = load_timeline(interim_dir, files_cfg)
    stable_set, started_map = load_bc_users(labels_path)
    nat_set = load_natural_users(interim_dir, files_cfg)
    logging.info(f"BC stable: {len(stable_set)}, started: {len(started_map)}, natural: {len(nat_set)}")
    return build_combined_timeline(tl, stable_set, started_map, nat_set, stable_only)


def assign_phases(tl: pd.DataFrame) -> pd.DataFrame:
    tl = tl.copy()
    tl["offset_mod"] = tl["offset_from_cd1"] % CYCLE_LENGTH
    tl["phase"] = tl["offset_mod"].apply(
        lambda x: next((p for p, (lo, hi) in PHASE_BOUNDS.items() if lo <= x < hi), None)
    )
    return tl[tl["phase"].notna()].copy()


def get_feature_cols(tl: pd.DataFrame) -> list[str]:
    return [
        c for c in tl.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(tl[c])
        and tl[c].notna().mean() > 0.3
    ]


def build_feature_matrix(tl: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """User × (phase × feature) matrix. Missing phases stay NaN; XGBoost handles them natively."""
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
    # Drop only columns that are almost entirely empty — XGBoost handles NaN natively,
    # so missingness itself (e.g. no Ovulation posts) is a usable biological signal
    phase_feat_cols = [c for c in wide.columns if "__" in c and wide[c].notna().mean() > 0.1]
    return wide, phase_feat_cols


def build_naive_matrix(tl: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """User × feature matrix — all posts averaged, no phase breakdown."""
    wide = tl.groupby(["author", "group"])[feat_cols].mean().reset_index()
    usable = [c for c in feat_cols if wide[c].notna().mean() > 0.1]
    return wide, usable


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


def save_report(output_dir, tag, report_str, roc_auc, pr_auc, chance, model_name, n_folds):
    path = output_dir / f"bc_user_classifier_report_{tag}.txt"
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
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — BC vs Natural Cycle"); axes[0].legend()
    axes[1].plot(rec, prec, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y.mean(), color="k", linestyle="--", lw=1, label=f"Chance = {y.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — BC vs Natural Cycle"); axes[1].legend()
    plt.tight_layout()
    path = output_dir / f"bc_user_classifier_roc_pr_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logging.info(f"  ROC/PR → {path.name}")


def plot_confusion(y, y_pred, roc_auc, output_dir, tag, model_name):
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Control", "BC"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} | AUC={roc_auc:.3f}", fontsize=10)
    plt.tight_layout()
    path = output_dir / f"bc_user_classifier_confusion_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_shap(model, X, feature_names, output_dir, tag, max_display=20):
    logging.info("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        sv = shap_values
    # Rename Phase__feature → [Phase] feature for readability in the plot
    display_names = [f"[{n.replace('__', '] ', 1)}" if "__" in n else n for n in feature_names]
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X, feature_names=display_names, max_display=max_display,
                      show=False, plot_type="dot")
    plt.title("SHAP — BC vs Natural Cycle (positive = BC)")
    plt.tight_layout()
    path = output_dir / f"bc_user_classifier_shap_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logging.info(f"  SHAP → {path.name}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    bc_cfg = cfg["bc_classifier"]
    interim_dir  = ROOT / cfg["paths"]["interim"]
    processed_dir = ROOT / cfg["paths"]["processed"]
    output_dir   = ROOT / cfg["paths"]["reports"] / bc_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path  = processed_dir / bc_cfg["labels_file"]
    n_folds = args.n_folds if args.n_folds is not None else bc_cfg["n_folds"]
    tag = _ts()

    # [1-4] Load + filter
    tl = load_data(cfg, interim_dir, labels_path, args.stable_only)
    tl = assign_phases(tl)

    feat_cols = get_feature_cols(tl)
    bc_users   = tl[tl["group"] == "BC"]["author"].nunique()
    ctrl_users = tl[tl["group"] == "Control"]["author"].nunique()
    logging.info(f"Linguistic features: {len(feat_cols)}")
    logging.info(f"BC users: {bc_users}, Control: {ctrl_users}")

    # [5] Build feature matrix
    logging.info("Building user × phase feature matrix...")
    wide, phase_feat_cols = build_feature_matrix(tl, feat_cols)

    y = (wide["group"] == "BC").astype(int).values
    bc_n, ctrl_n = y.sum(), (y == 0).sum()
    X = wide[phase_feat_cols].values
    logging.info(f"Feature matrix: {len(wide)} users ({bc_n} BC, {ctrl_n} Control), "
                 f"{len(phase_feat_cols)} features")

    # [6] Train + evaluate
    model = build_model(scale_pos_weight=ctrl_n / max(bc_n, 1))
    model_name = type(model).__name__
    logging.info(f"Running {n_folds}-fold OOF with {model_name}...")

    y_pred, y_proba = stratified_oof(model, X, y, n_folds)
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc  = average_precision_score(y, y_proba)
    report_str = classification_report(y, y_pred, target_names=["Control", "BC"], digits=3)

    logging.info(f"\n{report_str}")
    logging.info(f"AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}  (chance = {y.mean():.3f})")

    # [7] Save outputs
    save_report(output_dir, tag, report_str, roc_auc, pr_auc, y.mean(), model_name, n_folds)
    plot_roc_pr(y, y_proba, roc_auc, pr_auc, output_dir, tag)
    plot_confusion(y, y_pred, roc_auc, output_dir, tag, model_name)

    # [8] Phase-naive baseline — raw user means, no phase breakdown
    logging.info("Running phase-naive baseline (raw user means, no phase breakdown)...")
    wide_naive, naive_feat_cols = build_naive_matrix(tl, feat_cols)
    # Align users: same order as wide
    wide_naive = wide_naive.set_index("author").reindex(wide["author"]).reset_index()
    X_naive = wide_naive[naive_feat_cols].values
    naive_model = build_model(scale_pos_weight=ctrl_n / max(bc_n, 1))
    _, y_proba_naive = stratified_oof(naive_model, X_naive, y, n_folds)
    roc_auc_naive = roc_auc_score(y, y_proba_naive)
    pr_auc_naive  = average_precision_score(y, y_proba_naive)
    logging.info(f"Phase-naive  AUC-ROC = {roc_auc_naive:.3f}  |  PR-AUC = {pr_auc_naive:.3f}"
                 f"  ({len(naive_feat_cols)} features, no phase breakdown)")
    logging.info(f"Phase-aware  AUC-ROC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}"
                 f"  ({len(phase_feat_cols)} features, 4-phase breakdown)")
    logging.info(f"Delta AUC = {roc_auc - roc_auc_naive:+.3f}  (added by phase breakdown)")

    # [9] SHAP — fit on full data
    if not args.skip_shap:
        model.fit(X, y)  # scale_pos_weight already set at init — no sample_weight here
        plot_shap(model, X, phase_feat_cols, output_dir, tag)

    logging.info("Done.")


if __name__ == "__main__":
    main()
