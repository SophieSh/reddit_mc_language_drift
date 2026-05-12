"""
Step 29 — BC vs Natural Cycle Classifier (EBM + RuleFit)
==========================================================
Binary classification: combined-pill BC users vs natural-cycle users.

Representation
  Same as scripts 27/28: up to 4 rows per user (one per phase), each row is
  the user's mean z-scored linguistic feature vector across all days in that
  phase. GroupKFold (grouped by author) prevents leakage — all phase rows for
  a user stay in the same fold.

  This is richer than a single wide row per user (script 26 approach) because
  the model can learn phase-specific signals: e.g. "in the Ovulation phase,
  BC users look different from natural users" vs "in Menstrual phase they
  look similar."

  Phase is included as a numeric feature so the model can condition on it.

Phase assignment
  Natural users: detected period from consensus_periods_*no_bc*.csv
  BC users:      fixed 28-day cycle (combined pills impose an artificial cycle;
                 natural period detection does not apply)

BC users
  Combined-pill stable only — no start/stop event in their posts, not a
  progestin-only pill. Same filter as script 26.

Models
  1. EBM  (ExplainableBoostingClassifier, binary) — shape functions per feature
  2. RuleFit (RandomForest + L1 LogReg, binary)   — surviving IF-THEN rules

No ANOVA pre-selection — both models receive all 69 z-scored features.

Output (reports/bc_classifier/)
  bc_oof_report_<ts>.txt
  bc_ebm_importance_<ts>.csv
  bc_ebm_shape_<feature>_<ts>.png
  bc_rulefit_rules_<ts>.txt / .csv
  bc_rulefit_top_rules_<ts>.png

Usage
  python scripts/29_bc_vs_natural_classifier.py --config configs/base.yaml
  python scripts/29_bc_vs_natural_classifier.py --config configs/base.yaml --skip-cv
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
import scipy.sparse as sp
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import (
    assign_phases_to_timeline,
    compute_user_phase_definitions,
    normalize_features_per_user_zscore,
)
from src.config import load_config
from src.io import find_latest_file, find_periodicity_results

warnings.filterwarnings("ignore", category=UserWarning)

TREE_LEAF    = _tree.TREE_LEAF
PHASE_ORDER  = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_INT    = {p: i for i, p in enumerate(PHASE_ORDER)}   # for numeric encoding
CYCLE_LENGTH = 28.0

# Fixed phase bounds for BC users (identical proportions to adaptive logic
# but anchored to 28-day cycle)
BC_PHASE_BOUNDS = {
    "Menstrual":   (0,  5),
    "Follicular":  (5, 13),
    "Ovulation":  (13, 16),
    "Luteal":     (16, 28),
}

POP_KEYWORDS = [
    "pop", "norethindrone", "norgestrel", "desogestrel", "slynd", "opill",
    "cerazette", "cerelle", "nora-be", "camila", "errin", "jencycla", "lyza",
]

PHASE_COLORS = {
    "Follicular": "#4C9BE8",
    "Luteal":     "#E8884C",
    "Menstrual":  "#E84C4C",
    "Ovulation":  "#4CE89B",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_bc_users() -> tuple[set[str], dict[str, float]]:
    """Return combined-pill BC users split into stable and started.

    stable_set  : authors who were on BC throughout (no start/stop event)
    started_map : author → start_cd1 offset (posts before this are pre-BC, excluded)

    Both groups: combined pill only, not stopped.
    """
    path = ROOT / "data/processed/bc_wide_candidates_with_labels.csv"
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")

    bc = df[df["is_bc_pill"] == True].copy()
    bc["is_combined"] = bc["pill_name"].apply(
        lambda x: False if pd.isna(x)
        else not any(k in str(x).lower() for k in POP_KEYWORDS)
    )
    user_flags = bc.groupby("author").agg(
        any_started=("started_recently", "any"),
        any_stopped=("stopped_recently", "any"),
        is_comb=("is_combined", lambda x: x.any() and not (x == False).any()),
    ).reset_index()

    stable_set = set(
        user_flags[
            ~user_flags["any_started"] & ~user_flags["any_stopped"] & user_flags["is_comb"]
        ]["author"].str.lower()
    )

    # For started users: keep only posts after their BC start date
    started_posts = bc[(bc["started_recently"] == True) & (bc["is_combined"] == True)].copy()
    started_posts["start_cd1"] = started_posts["offset_from_cd1"] + started_posts["started_offset"]
    started_posts["abs_off"]   = started_posts["started_offset"].abs()
    user_start_cd1 = started_posts.sort_values("abs_off").groupby("author")["start_cd1"].first()

    started_df = user_flags[
        user_flags["any_started"] & ~user_flags["any_stopped"] & user_flags["is_comb"]
    ]
    started_map = {
        a.lower(): user_start_cd1[a]
        for a in started_df["author"]
        if a in user_start_cd1.index
    }

    return stable_set, started_map


def _assign_bc_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Assign phases to BC users using a fixed 28-day cycle."""
    df = df.copy()
    df["offset_mod"] = df["offset_from_cd1"] % CYCLE_LENGTH
    df["phase"] = df["offset_mod"].apply(
        lambda x: next(
            (p for p, (lo, hi) in BC_PHASE_BOUNDS.items() if lo <= x < hi), None
        )
    )
    return df[df["phase"].notna()].copy()


def load_phase_profiles(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load phase profiles for both natural and BC users.

    Natural users: detected period → adaptive phase assignment (same as script 27)
    BC users:      fixed 28-day cycle → fixed phase bounds

    Returns:
        profiles  : DataFrame with columns author, phase, label (0=natural, 1=BC),
                    phase_int, and all zscore feature columns
        zscore_cols: list of feature column names
    """
    interim_dir = Path(cfg["paths"]["interim"])

    # ── Load and z-score the daily aggregated timeline ────────────────────────
    path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    if path is None:
        raise FileNotFoundError("No timeline_daily_aggregated_with_anchors_*.csv found.")
    logging.info(f"Loading timeline: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["author"] = df["author"].astype(str)
    logging.info(f"  {len(df):,} user-days, {df['author'].nunique():,} users")

    mean_cols = [c for c in df.columns
                 if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])]
    df = normalize_features_per_user_zscore(df, mean_cols, user_col="author")
    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    logging.info(f"  {len(zscore_cols)} z-score features")

    # ── Natural cycle users ───────────────────────────────────────────────────
    results_path = find_periodicity_results(interim_dir)
    pdf = pd.read_csv(results_path, encoding="utf-8-sig", low_memory=False)
    user_period_map = dict(zip(pdf["user"].astype(str), pdf["consensus_period"].astype(float)))
    natural_users = set(user_period_map.keys())

    df_nat = df[df["author"].isin(natural_users)].copy()
    user_phase_df = compute_user_phase_definitions(
        {u: p for u, p in user_period_map.items() if u in set(df_nat["author"])}
    )
    df_nat = assign_phases_to_timeline(
        df_nat, user_phase_df, user_col="author", time_col="offset_from_cd1"
    )
    df_nat = df_nat[df_nat["phase"].isin(PHASE_ORDER)].copy()
    df_nat["label"] = 0
    logging.info(
        f"  Natural users: {df_nat['author'].nunique():,}  "
        f"user-days: {len(df_nat):,}"
    )

    # ── BC users (stable + recently-started, combined pill only) ─────────────
    stable_set, started_map = _load_bc_users()

    # Stable: all their posts are on-BC
    df_stable = df[df["author"].str.lower().isin(stable_set)].copy()
    df_stable = _assign_bc_phases(df_stable)

    # Started: keep only posts after their BC start date (pre-BC posts excluded)
    df_started = df[df["author"].str.lower().isin(started_map)].copy()
    df_started["author_lower"] = df_started["author"].str.lower()
    df_started["start_cd1"]    = df_started["author_lower"].map(started_map)
    df_started = df_started[df_started["offset_from_cd1"] >= df_started["start_cd1"]].copy()
    df_started = _assign_bc_phases(df_started)

    df_bc = pd.concat([df_stable, df_started], ignore_index=True)
    df_bc["label"] = 1
    logging.info(
        f"  BC users:      {df_bc['author'].nunique():,}  "
        f"(stable={df_stable['author'].nunique():,}  started={df_started['author'].nunique():,})  "
        f"user-days: {len(df_bc):,}"
    )

    # ── Aggregate to phase profiles ───────────────────────────────────────────
    # One row per (user, phase): mean z-score across all days in that phase.
    # Keep label and phase — both are constant within a (user, phase) group.
    all_df = pd.concat([df_nat[["author","phase","label"] + zscore_cols],
                        df_bc[["author","phase","label"] + zscore_cols]],
                       ignore_index=True)

    profiles = (
        all_df.groupby(["author", "phase", "label"])[zscore_cols]
        .mean()
        .reset_index()
    )
    profiles[zscore_cols] = profiles[zscore_cols].fillna(0)
    # Numeric phase encoding so models can condition on phase
    profiles["phase_int"] = profiles["phase"].map(PHASE_INT).astype(float)

    n_nat = (profiles["label"] == 0).sum()
    n_bc  = (profiles["label"] == 1).sum()
    logging.info(
        f"  Phase profiles: {len(profiles):,} total  "
        f"({n_nat:,} natural rows / {n_bc:,} BC rows)"
    )
    logging.info(f"  Natural users: {profiles[profiles['label']==0]['author'].nunique():,}  "
                 f"BC users: {profiles[profiles['label']==1]['author'].nunique():,}")
    return profiles, zscore_cols


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _oof_auc(y_true, y_proba, label=""):
    auc = roc_auc_score(y_true, y_proba)
    pr  = average_precision_score(y_true, y_proba)
    chance = y_true.mean()
    logging.info(f"  {label}  AUC={auc:.4f}  PR-AUC={pr:.4f}  (chance={chance:.3f})")
    return auc, pr


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — EBM
# ═══════════════════════════════════════════════════════════════════════════════

def run_ebm_cv(X, y, groups, n_folds):
    gkf = GroupKFold(n_splits=n_folds)
    y_proba = np.zeros(len(y))

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        ebm = ExplainableBoostingClassifier(
            max_bins=256, interactions=0, learning_rate=0.01,
            max_rounds=3000, min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
        ebm.fit(X[tr], y[tr])
        y_proba[te] = ebm.predict_proba(X[te])[:, 1]
        logging.info(f"    EBM fold {fold}/{n_folds}  fold-AUC={roc_auc_score(y[te], y_proba[te]):.4f}")

    return y_proba


def fit_final_ebm(X, y):
    ebm = ExplainableBoostingClassifier(
        max_bins=256, interactions=0, learning_rate=0.01,
        max_rounds=3000, min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    ebm.fit(X, y)
    return ebm


def extract_ebm_importance(ebm, feature_names):
    """Extract mean |log-odds| importance per feature for the binary EBM."""
    global_exp = ebm.explain_global()
    rows = []
    for i, feat in enumerate(feature_names):
        data   = global_exp.data(i)
        scores = np.abs(np.asarray(data["scores"]))
        if scores.ndim == 2:
            scores = scores.mean(axis=1)
        rows.append({"feature": feat, "importance": float(scores.mean())})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def plot_ebm_shape(ebm, feature_name, feature_names, output_path):
    """Shape function for one feature: log-odds vs z-score value."""
    idx  = feature_names.index(feature_name)
    data = ebm.explain_global().data(idx)

    scores     = np.asarray(data["scores"])
    bin_labels = data["names"]

    # For binary EBM scores may be 1-D
    if scores.ndim == 2:
        scores = scores[:, 1]

    x_vals = []
    for lbl in bin_labels:
        lbl = str(lbl)
        try:
            if " to " in lbl:
                parts = lbl.split(" to ")
                x_vals.append((float(parts[0]) + float(parts[1])) / 2)
            elif lbl.startswith(">"):
                x_vals.append(float(lbl.replace(">", "").strip()))
            else:
                x_vals.append(float(lbl.replace("<=", "").strip()))
        except ValueError:
            x_vals.append(np.nan)

    x = np.array(x_vals)
    n = min(len(x), len(scores))
    x, scores = x[:n], scores[:n]
    valid = ~np.isnan(x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(x[valid], scores[valid], where="mid", color="#E84C4C", linewidth=2)
    ax.fill_between(x[valid], scores[valid], 0, step="mid", alpha=0.2, color="#E84C4C")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray",  linewidth=0.6, linestyle=":")
    clean = feature_name.replace("_zscore","").replace("_"," ")
    ax.set_title(f"EBM Shape — {clean}\nPositive log-odds → pushes toward BC",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature value (z-score vs user mean)")
    ax.set_ylabel("Log-odds contribution (BC)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  EBM shape → {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — RULEFIT
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_leaf_rules(tree_est, feature_names):
    tree_ = tree_est.tree_
    rules = {}
    def _recurse(node, conds):
        if tree_.children_left[node] == TREE_LEAF:
            rules[node] = " AND\n    ".join(conds) if conds else "(always)"
            return
        fname = feature_names[tree_.feature[node]].replace("_zscore","").replace("_"," ")
        thr   = tree_.threshold[node]
        _recurse(tree_.children_left[node],  conds + [f"{fname} ≤ {thr:+.3f}"])
        _recurse(tree_.children_right[node], conds + [f"{fname} > {thr:+.3f}"])
    _recurse(0, [])
    return rules


def _build_rule_matrix(forest, X_array, feature_names):
    leaf_assignments = forest.apply(X_array)
    n_samples = X_array.shape[0]
    blocks, meta = [], []
    for t_idx, tree_est in enumerate(forest.estimators_):
        leaf_rules    = _extract_leaf_rules(tree_est, feature_names)
        unique_leaves = sorted(leaf_rules.keys())
        leaf_to_col   = {lid: col for col, lid in enumerate(unique_leaves)}
        col_indices   = np.array([leaf_to_col[lid] for lid in leaf_assignments[:, t_idx]])
        block = sp.csr_matrix(
            (np.ones(n_samples, dtype=np.float32),
             (np.arange(n_samples), col_indices)),
            shape=(n_samples, len(unique_leaves)),
        )
        blocks.append(block)
        for lid in unique_leaves:
            col  = leaf_to_col[lid]
            supp = float(block[:, col].sum()) / n_samples
            meta.append({"tree_idx": t_idx, "leaf_node_id": lid,
                         "rule_str": leaf_rules[lid], "support_frac": supp})
    return sp.hstack(blocks, format="csr"), meta


def run_rulefit_cv(X, y, groups, feature_names, n_folds, n_estimators, max_depth, C):
    gkf = GroupKFold(n_splits=n_folds)
    y_proba = np.zeros(len(y))

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_leaf=10, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
        forest.fit(X[tr], y[tr])
        X_rules_tr, _ = _build_rule_matrix(forest, X[tr], feature_names)
        X_rules_te, _ = _build_rule_matrix(forest, X[te], feature_names)
        X_tr_comb = sp.hstack([sp.csr_matrix(X[tr].astype(np.float32)), X_rules_tr])
        X_te_comb = sp.hstack([sp.csr_matrix(X[te].astype(np.float32)), X_rules_te])

        logreg = LogisticRegression(solver="saga", l1_ratio=1.0, C=C,
                                     max_iter=3000, random_state=42)
        logreg.fit(X_tr_comb, y[tr])
        y_proba[te] = logreg.predict_proba(X_te_comb)[:, 1]
        logging.info(f"    RuleFit fold {fold}/{n_folds}  fold-AUC={roc_auc_score(y[te], y_proba[te]):.4f}")

    return y_proba


def fit_final_rulefit(X, y, feature_names, n_estimators, max_depth, C):
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    min_samples_leaf=10, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
    forest.fit(X, y)
    X_rules, rule_meta = _build_rule_matrix(forest, X, feature_names)
    X_comb = sp.hstack([sp.csr_matrix(X.astype(np.float32)), X_rules])
    logreg = LogisticRegression(solver="saga", l1_ratio=1.0, C=C,
                                 max_iter=3000, random_state=42)
    logreg.fit(X_comb, y)
    return logreg, rule_meta


def extract_rulefit_rules(logreg, feature_names, rule_meta, top_n=15):
    """Return surviving rules sorted by |coefficient|, positive = BC signal."""
    coefs = logreg.coef_[0]
    n_orig = len(feature_names)
    rows = []
    for i, c in enumerate(coefs):
        if c == 0.0:
            continue
        if i < n_orig:
            rows.append({"type": "original", "name": feature_names[i].replace("_zscore",""),
                         "coefficient": c, "abs_coef": abs(c), "support": None})
        else:
            m = rule_meta[i - n_orig]
            rows.append({"type": "rule", "name": m["rule_str"],
                         "coefficient": c, "abs_coef": abs(c), "support": m["support_frac"]})
    df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
    return df


def format_rulefit_report(df_rules, top_n=15):
    pos = df_rules[df_rules["coefficient"] > 0].head(top_n)
    neg = df_rules[df_rules["coefficient"] < 0].head(top_n)
    lines = []

    lines.append("\n▲ POSITIVE (push toward BC):")
    lines.append(f"  {'Coef':>7}  {'Support':>7}  Rule")
    lines.append("  " + "-"*55)
    for _, r in pos.iterrows():
        sup = f"{r['support']*100:5.1f}%" if r["support"] else "  n/a "
        rule = r["name"].replace("\n    ", "\n             ")
        lines.append(f"  {r['coefficient']:+7.4f}  {sup}  {rule}")

    lines.append("\n▼ NEGATIVE (push away from BC / toward natural):")
    lines.append(f"  {'Coef':>7}  {'Support':>7}  Rule")
    lines.append("  " + "-"*55)
    for _, r in neg.iterrows():
        sup = f"{r['support']*100:5.1f}%" if r["support"] else "  n/a "
        rule = r["name"].replace("\n    ", "\n             ")
        lines.append(f"  {r['coefficient']:+7.4f}  {sup}  {rule}")

    return "\n".join(lines)


def plot_rulefit_rules(df_rules, output_path, top_n=15):
    top = df_rules.head(top_n).copy()
    labels = []
    for r in top["name"]:
        flat = r.replace("\n    ", " / ")
        labels.append(flat[:70] + "…" if len(flat) > 70 else flat)

    colors = ["#E84C4C" if c > 0 else "#4C9BE8" for c in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(12, top_n * 0.45 + 2))
    ax.barh(range(len(labels)), top["coefficient"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("L1 coefficient  (positive = BC signal, negative = Natural signal)")
    ax.set_title("RuleFit — Top surviving rules: BC vs Natural Cycle\n"
                 "Red = pushes toward BC · Blue = pushes toward Natural",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  RuleFit plot → {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def run_xgb_cv(X, y, groups, n_folds):
    gkf = GroupKFold(n_splits=n_folds)
    y_proba = np.zeros(len(y))
    pos_n, neg_n = y.sum(), (y == 0).sum()
    scale_w = neg_n / max(pos_n, 1)

    if XGBOOST_AVAILABLE:
        base_model = XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_w, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        base_model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        m = clone(base_model)
        m.fit(X[tr], y[tr])
        y_proba[te] = m.predict_proba(X[te])[:, 1]
        logging.info(f"    XGB   fold {fold}/{n_folds}  fold-AUC={roc_auc_score(y[te], y_proba[te]):.4f}")

    return y_proba


# ═══════════════════════════════════════════════════════════════════════════════
# PER-PHASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_per_phase_analysis(
    profiles: pd.DataFrame,
    zscore_cols: list[str],
    n_folds: int,
) -> pd.DataFrame:
    """Run a separate EBM binary classifier (BC vs natural) for each phase.

    Answers the question: "which phase shows the strongest BC signal, and
    which features drive it within that specific phase?"

    Returns a DataFrame with columns: phase, feature, importance, auc
    """
    results = []

    for phase in PHASE_ORDER:
        ph_df = profiles[profiles["phase"] == phase].copy()
        n_bc  = (ph_df["label"] == 1).sum()
        n_nat = (ph_df["label"] == 0).sum()

        if n_bc < 10 or n_nat < 10:
            logging.warning(f"  {phase}: too few samples (BC={n_bc}, Nat={n_nat}) — skipping")
            continue

        X_ph  = ph_df[zscore_cols].values.astype(np.float64)
        y_ph  = ph_df["label"].values.astype(int)
        grp   = ph_df["author"].values

        # OOF AUC — reduce folds if not enough users
        n_users = len(np.unique(grp))
        folds   = min(n_folds, n_users // 2)

        gkf     = GroupKFold(n_splits=folds)
        y_proba = np.zeros(len(y_ph))
        for tr, te in gkf.split(X_ph, y_ph, grp):
            ebm = ExplainableBoostingClassifier(
                max_bins=128, interactions=0, learning_rate=0.02,
                max_rounds=2000, random_state=42, n_jobs=-1,
            )
            ebm.fit(X_ph[tr], y_ph[tr])
            y_proba[te] = ebm.predict_proba(X_ph[te])[:, 1]

        auc = roc_auc_score(y_ph, y_proba)
        logging.info(f"  {phase:<12}: AUC={auc:.4f}  (BC={n_bc}, Nat={n_nat})")

        # Feature importance from final fit on full phase data
        ebm_full = ExplainableBoostingClassifier(
            max_bins=128, interactions=0, learning_rate=0.02,
            max_rounds=2000, random_state=42, n_jobs=-1,
        )
        ebm_full.fit(X_ph, y_ph)
        global_exp = ebm_full.explain_global()

        for i, feat in enumerate(zscore_cols):
            data   = global_exp.data(i)
            scores = np.abs(np.asarray(data["scores"]))
            if scores.ndim == 2:
                scores = scores.mean(axis=1)
            results.append({
                "phase":      phase,
                "feature":    feat,
                "importance": float(scores.mean()),
                "auc":        auc,
            })

    df = pd.DataFrame(results)
    return df


def print_per_phase_summary(df_phase: pd.DataFrame, top_n: int = 8) -> str:
    lines = ["\nPer-phase EBM — top features (BC vs Natural within each phase)"]
    lines.append("=" * 60)
    for phase in PHASE_ORDER:
        ph = df_phase[df_phase["phase"] == phase]
        if ph.empty:
            continue
        auc = ph["auc"].iloc[0]
        top = ph.nlargest(top_n, "importance")
        lines.append(f"\n{phase}  (AUC={auc:.4f}):")
        for _, r in top.iterrows():
            feat = r["feature"].replace("_zscore","")
            lines.append(f"  {r['importance']:.4f}  {feat}")
    return "\n".join(lines)


def plot_per_phase_importance(df_phase: pd.DataFrame, output_path: Path, top_n: int = 8):
    """2×2 heatmap-style bar charts: top features per phase."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, phase in zip(axes, PHASE_ORDER):
        ph = df_phase[df_phase["phase"] == phase]
        if ph.empty:
            ax.set_visible(False)
            continue
        auc  = ph["auc"].iloc[0]
        top  = ph.nlargest(top_n, "importance")
        labels = [f.replace("_zscore","").replace("_"," ")[:45] for f in top["feature"]]
        color  = PHASE_COLORS.get(phase, "steelblue")

        ax.barh(range(len(labels)), top["importance"], color=color, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{phase}  —  AUC={auc:.3f}", fontsize=11,
                     fontweight="bold", color=color)
        ax.set_xlabel("EBM mean |log-odds|", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("BC vs Natural Cycle — Top Features per Phase (EBM)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Per-phase importance → {output_path.name}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",       default="configs/base.yaml")
    p.add_argument("--n-folds",      type=int,   default=5)
    p.add_argument("--n-estimators", type=int,   default=100)
    p.add_argument("--max-depth",    type=int,   default=3)
    p.add_argument("--C",            type=float, default=0.1)
    p.add_argument("--top-n",        type=int,   default=15,
                   help="Top features/rules to show in plots and reports.")
    p.add_argument("--skip-cv",      action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_config(args.config)
    out = ROOT / "reports" / "bc_classifier"
    out.mkdir(parents=True, exist_ok=True)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    logging.info("\n[1/5] Loading phase profiles…")
    profiles, zscore_cols = load_phase_profiles(cfg)

    # Feature matrix includes z-score features + numeric phase
    feature_names = zscore_cols + ["phase_int"]
    X      = profiles[feature_names].values.astype(np.float64)
    y      = profiles["label"].values.astype(int)
    groups = profiles["author"].values

    logging.info(
        f"\n  X shape : {X.shape}  ({len(feature_names)} features incl. phase_int)"
        f"\n  BC rows : {y.sum():,}  Natural rows: {(y==0).sum():,}"
        f"\n  Users   : BC={profiles[profiles['label']==1]['author'].nunique():,}  "
        f"Natural={profiles[profiles['label']==0]['author'].nunique():,}"
    )

    # ── [2] Cross-validation ─────────────────────────────────────────────────
    report_lines = []

    if not args.skip_cv:
        logging.info(f"\n[2/5] GroupKFold CV ({args.n_folds} folds)…")

        logging.info("  — EBM —")
        ebm_proba = run_ebm_cv(X, y, groups, args.n_folds)
        ebm_auc, ebm_pr = _oof_auc(y, ebm_proba, "EBM")

        logging.info("  — RuleFit —")
        rf_proba = run_rulefit_cv(X, y, groups, feature_names,
                                  args.n_folds, args.n_estimators, args.max_depth, args.C)
        rf_auc, rf_pr = _oof_auc(y, rf_proba, "RuleFit")

        logging.info("  — XGBoost —")
        xgb_proba = run_xgb_cv(X, y, groups, args.n_folds)
        xgb_auc, xgb_pr = _oof_auc(y, xgb_proba, "XGBoost")

        report_lines += [
            "BC vs Natural Cycle — OOF Results",
            "=" * 40,
            f"EBM     AUC={ebm_auc:.4f}  PR-AUC={ebm_pr:.4f}",
            f"RuleFit AUC={rf_auc:.4f}  PR-AUC={rf_pr:.4f}",
            f"XGBoost AUC={xgb_auc:.4f}  PR-AUC={xgb_pr:.4f}",
            f"Chance  AUC=0.500  PR-AUC={y.mean():.3f}",
        ]
        rpt_path = out / f"bc_oof_report_{ts}.txt"
        rpt_path.write_text("\n".join(report_lines))
        logging.info(f"  OOF report → {rpt_path.name}")
    else:
        logging.info("\n[2/4] CV skipped.")

    # ── [3] Final fits ────────────────────────────────────────────────────────
    logging.info("\n[3/5] Final fits on full dataset…")

    final_ebm = fit_final_ebm(X, y)
    logging.info("  EBM fitted.")

    final_logreg, rule_meta = fit_final_rulefit(
        X, y, feature_names, args.n_estimators, args.max_depth, args.C
    )
    n_nz = (final_logreg.coef_ != 0).sum()
    logging.info(f"  RuleFit fitted — {n_nz} non-zero coefficients.")

    # ── [4] Interpretation ────────────────────────────────────────────────────
    logging.info("\n[4/5] Extracting interpretation…")

    # EBM importance
    df_imp = extract_ebm_importance(final_ebm, feature_names)
    imp_path = out / f"bc_ebm_importance_{ts}.csv"
    df_imp.to_csv(imp_path, index=False)
    logging.info(f"\n  Top 15 EBM features (BC vs Natural):\n"
                 + df_imp.head(15)[["feature","importance"]].to_string(index=False))

    # EBM shape plots for top features
    for feat in df_imp["feature"].head(10):
        if feat == "phase_int":
            continue
        safe = feat.replace("/","_").replace(" ","_")
        plot_ebm_shape(final_ebm, feat, feature_names,
                       out / f"bc_ebm_shape_{safe}_{ts}.png")

    # RuleFit rules
    df_rules = extract_rulefit_rules(final_logreg, feature_names, rule_meta)
    rules_text = format_rulefit_report(df_rules, top_n=args.top_n)
    logging.info(rules_text)

    (out / f"bc_rulefit_rules_{ts}.txt").write_text(
        f"RuleFit BC vs Natural | C={args.C} | max_depth={args.max_depth}\n" + rules_text
    )
    df_rules.to_csv(out / f"bc_rulefit_rules_{ts}.csv", index=False)
    plot_rulefit_rules(df_rules, out / f"bc_rulefit_top_rules_{ts}.png", top_n=args.top_n)

    # ── [5] Per-phase analysis ────────────────────────────────────────────────
    logging.info("\n[5/5] Per-phase analysis (separate EBM per phase)…")
    df_phase = run_per_phase_analysis(profiles, zscore_cols, args.n_folds)
    phase_summary = print_per_phase_summary(df_phase, top_n=args.top_n)
    logging.info(phase_summary)
    (out / f"bc_per_phase_summary_{ts}.txt").write_text(phase_summary)
    df_phase.to_csv(out / f"bc_per_phase_importance_{ts}.csv", index=False)
    plot_per_phase_importance(df_phase, out / f"bc_per_phase_{ts}.png", top_n=args.top_n)

    logging.info(f"\nAll outputs → {out}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
