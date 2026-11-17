"""
Step 28 — RuleFit-Style Multiclass Rule Extraction
====================================================
Extracts highly interpretable IF-THEN rules that characterise each menstrual
cycle phase, using a two-step pipeline:

  Step A — Interaction Discovery
    A RandomForestClassifier is trained on the z-scored feature matrix.
    Its trees encode non-linear interactions between features.  For each
    tree we walk root→leaf and record the conjunction of split conditions
    that defines every leaf.  All leaves across all trees become binary
    "rule features": a sample gets a 1 for a leaf if that sample fell into
    that leaf, 0 otherwise.

  Step B — Pruning via L1 Logistic Regression
    We concatenate the original 69 z-score features with the ~N_trees × max_leaves
    binary rule features.  An L1-penalised Logistic Regression (OvR, one set of
    weights per class) is trained on this combined matrix.  The L1 penalty drives
    redundant/weak coefficients to exactly zero, keeping only the rules and
    original features that are genuinely useful for each class.

Why this approach for Feature Discovery?
  • Decision tree alone: single tree, greedy splits, high variance.
  • Random forest: accurate but opaque — no single rule set.
  • RuleFit: gets the best of both.  The forest generates a *diverse* pool of
    candidate rules; L1 selects the sparse subset that is jointly most predictive.
    The surviving rules are exactly the multivariate conditions that discriminate
    the phases, with a signed coefficient telling you which direction.

Output (reports/rulefit/)
  rulefit_oof_report_<ts>.txt        — OOF AUC scores per fold and aggregate
  rulefit_rules_<ts>.txt             — Human-readable surviving rules per class
  rulefit_rules_<ts>.csv             — Machine-readable rules table
  rulefit_top_rules_<ts>.png         — Bar chart of top rules by |coefficient|

Usage
  python scripts/28_rulefit_phase_analysis.py --config configs/base.yaml
  python scripts/28_rulefit_phase_analysis.py --config configs/base.yaml --max-depth 4 --C 0.05
  python scripts/28_rulefit_phase_analysis.py --config configs/base.yaml --skip-cv
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree

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

PHASE_ORDER  = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_COLORS = {
    "Follicular": "#4C9BE8",
    "Luteal":     "#E8884C",
    "Menstrual":  "#E84C4C",
    "Ovulation":  "#4CE89B",
}

# sklearn internal constant: marks a node as a leaf in the tree arrays
TREE_LEAF = _tree.TREE_LEAF


# ═══════════════════════════════════════════════════════════════════════════════
# STEP A — RULE EXTRACTION FROM RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_leaf_rules(tree_estimator, feature_names: list[str]) -> dict[int, str]:
    """Walk a single decision tree and build a rule string for every leaf.

    Each leaf represents an IF-THEN rule: the conjunction of all split conditions
    encountered on the path from the root down to that leaf.

    Returns:
        {leaf_node_id: rule_string}
        e.g. {7: "valence dict average ≤ -0.330 AND imaginability coverage > -0.260"}

    The leaf_node_id is the raw node index in the sklearn tree arrays — used to
    map back from forest.apply() outputs to rule strings.
    """
    tree_ = tree_estimator.tree_
    rules: dict[int, str] = {}

    def _recurse(node: int, conditions: list[str]) -> None:
        # Base case: this node is a leaf (no children)
        if tree_.children_left[node] == TREE_LEAF:
            rules[node] = " AND\n    ".join(conditions) if conditions else "(always true)"
            return

        # Internal node: retrieve the split feature and threshold
        feat_idx   = tree_.feature[node]      # index into feature_names
        threshold  = tree_.threshold[node]    # numeric split value (z-score)
        feat_label = (
            feature_names[feat_idx]
            .replace("_zscore", "")
            .replace("_", " ")
        )

        # Left branch  → feature ≤ threshold
        _recurse(
            tree_.children_left[node],
            conditions + [f"{feat_label} ≤ {threshold:+.3f}"],
        )
        # Right branch → feature > threshold
        _recurse(
            tree_.children_right[node],
            conditions + [f"{feat_label} > {threshold:+.3f}"],
        )

    _recurse(0, [])
    return rules


def build_rule_feature_matrix(
    forest: RandomForestClassifier,
    X_array: np.ndarray,
    feature_names: list[str],
) -> tuple[sp.csr_matrix, list[dict]]:
    """Convert a fitted RandomForest into a sparse binary rule-feature matrix.

    For each tree t and each possible leaf l in that tree, we add one binary
    column to the output matrix:
        column value = 1  if sample i fell into leaf l of tree t
        column value = 0  otherwise

    Because each sample falls into exactly one leaf per tree, each block of
    `n_leaves_in_tree_t` columns is a one-hot row.

    Args:
        forest:       A *fitted* RandomForestClassifier.
        X_array:      (n_samples, n_features) numpy array to transform.
        feature_names: List of feature name strings, aligned with X_array columns.

    Returns:
        X_rules  : sparse (n_samples × total_rule_columns) binary matrix.
        rule_meta: list of dicts, one per column, with keys:
                     tree_idx, leaf_node_id, rule_str, support_frac
                   Aligned with X_rules columns.
    """
    # forest.apply(X) → (n_samples, n_estimators)
    # Entry [i, t] is the leaf node ID that sample i landed in for tree t.
    leaf_assignments = forest.apply(X_array)   # (n_samples, n_trees)
    n_samples = X_array.shape[0]

    sparse_blocks: list[sp.csr_matrix] = []
    rule_meta:     list[dict]          = []

    for t_idx, tree_est in enumerate(forest.estimators_):
        # --- Extract all leaf rules for this tree --------------------------
        leaf_rules = _extract_leaf_rules(tree_est, feature_names)
        # unique_leaves is ALL leaves in the tree (from structure traversal),
        # not just those observed in X_array — ensures consistency when
        # transforming new data (test fold) with the same column mapping.
        unique_leaves = sorted(leaf_rules.keys())
        leaf_to_col   = {leaf_id: col for col, leaf_id in enumerate(unique_leaves)}

        # --- Build sparse one-hot block for this tree ----------------------
        # tree_leaf_col[i] = leaf ID that sample i landed in for this tree
        tree_leaf_col = leaf_assignments[:, t_idx]

        row_indices = np.arange(n_samples)
        col_indices = np.array([leaf_to_col[lid] for lid in tree_leaf_col])
        # Every sample activates exactly one leaf per tree → all data values are 1
        data = np.ones(n_samples, dtype=np.float32)

        block = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_samples, len(unique_leaves)),
        )
        sparse_blocks.append(block)

        # --- Record metadata for each rule column --------------------------
        for leaf_id in unique_leaves:
            # Support = fraction of *this batch* that satisfies the rule
            col = leaf_to_col[leaf_id]
            support = float(block[:, col].sum()) / n_samples
            rule_meta.append({
                "tree_idx":     t_idx,
                "leaf_node_id": leaf_id,
                "rule_str":     leaf_rules[leaf_id],
                "support_frac": support,
            })

    X_rules = sp.hstack(sparse_blocks, format="csr")
    logging.debug(
        f"  Rule matrix: {X_rules.shape[1]:,} rule columns "
        f"from {len(forest.estimators_)} trees"
    )
    return X_rules, rule_meta


def combine_features(
    X_cont: np.ndarray,
    X_rules: sp.csr_matrix,
) -> sp.csr_matrix:
    """Stack continuous z-score features (left) and binary rule features (right).

    Returns a sparse matrix so L1 LogReg can handle it efficiently.
    The first `n_original` columns are always the original z-scored features;
    the remaining columns are the binary rule activations.
    """
    # scipy.sparse.hstack requires both inputs to be sparse
    X_cont_sparse = sp.csr_matrix(X_cont.astype(np.float32))
    return sp.hstack([X_cont_sparse, X_rules], format="csr")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP B — L1 LOGISTIC REGRESSION (PRUNING)
# ═══════════════════════════════════════════════════════════════════════════════

def build_forest(n_estimators: int, max_depth: int, random_state: int = 42):
    """Return an untrained RandomForestClassifier with settings tuned for
    interpretability (shallow trees = short rules)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,          # depth 3 → max 3 conditions per rule
        min_samples_leaf=10,          # avoids hyper-specific leaves
        class_weight="balanced",      # handles phase-length imbalance
        random_state=random_state,
        n_jobs=-1,
    )


def build_logreg(C: float):
    """Return an untrained L1-penalised OvR Logistic Regression.

    OvR (one-vs-rest) gives one weight vector per class, making it easy to
    read off which rules are characteristic of each phase independently.

    penalty='l1' + solver='saga' is the standard combination for sparse
    solutions on large feature matrices.

    C is the *inverse* regularisation strength: smaller C → more sparsity
    (fewer surviving rules).  Tune downward if you want cleaner outputs.
    """
    # l1_ratio=1 is the new API for L1 (replaces penalty='l1' deprecated in 1.8)
    return LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=C,
        max_iter=3000,
        random_state=42,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    label_encoder: LabelEncoder,
    n_folds: int,
    n_estimators: int,
    max_depth: int,
    C: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """GroupKFold OOF evaluation of the RuleFit pipeline.

    Each fold independently:
      1. Trains a RandomForest on the training split.
      2. Transforms both train and test into combined (cont + rules) matrices.
      3. Trains L1 LogReg on the training combined matrix.
      4. Predicts probabilities on the test combined matrix.

    Why do we retransform inside each fold?
      The rule feature matrix is derived from the *fitted* forest.  If we
      transformed the full dataset once and then split, the test fold's rule
      features would be computed using a forest that had already seen the test
      data.  That would be leakage.

    Returns:
        y_pred  : (n_samples,) OOF hard predictions
        y_proba : (n_samples, n_classes) OOF probability matrix
        fold_reports : list of per-fold AUC strings
    """
    gkf = GroupKFold(n_splits=n_folds)
    n_classes  = len(label_encoder.classes_)
    y_pred     = np.empty_like(y)
    y_proba    = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_reports: list[str] = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups), start=1
    ):
        logging.info(
            f"  Fold {fold}/{n_folds} — "
            f"train: {len(train_idx):,}  test: {len(test_idx):,}  "
            f"train_users: {len(np.unique(groups[train_idx])):,}  "
            f"test_users:  {len(np.unique(groups[test_idx])):,}"
        )
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # ── Step A: fit forest on training fold, extract rule features ──────
        forest = build_forest(n_estimators, max_depth)
        forest.fit(X_tr, y_tr)

        # Transform training data → (continuous + binary rules)
        X_rules_tr, _ = build_rule_feature_matrix(forest, X_tr, feature_names)
        X_combined_tr  = combine_features(X_tr, X_rules_tr)

        # Transform test data using the *same fitted forest* (no leakage)
        X_rules_te, _ = build_rule_feature_matrix(forest, X_te, feature_names)
        X_combined_te  = combine_features(X_te, X_rules_te)

        # ── Step B: fit L1 LogReg on combined training matrix ───────────────
        logreg = build_logreg(C)
        logreg.fit(X_combined_tr, y_tr)

        # ── Evaluate on test fold ───────────────────────────────────────────
        y_pred[test_idx]  = logreg.predict(X_combined_te)
        y_proba[test_idx] = logreg.predict_proba(X_combined_te)

        # Per-fold AUC
        try:
            fold_macro = roc_auc_score(
                y_te, y_proba[test_idx], multi_class="ovr", average="macro"
            )
        except ValueError:
            fold_macro = float("nan")
        fold_reports.append(f"  Fold {fold}: macro AUC = {fold_macro:.4f}")
        logging.info(f"    Fold {fold} macro AUC: {fold_macro:.4f}")

    return y_pred, y_proba, fold_reports


def log_oof_metrics(
    y: np.ndarray,
    y_proba: np.ndarray,
    label_encoder: LabelEncoder,
    fold_reports: list[str],
) -> str:
    """Compute aggregate OOF AUC and return a formatted report string."""
    macro_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    lines = ["=" * 50]
    lines.append("RuleFit OOF Results")
    lines.append("=" * 50)
    lines += fold_reports
    lines.append("")
    lines.append(f"OOF Macro AUC: {macro_auc:.4f}  [random = 0.500]")
    lines.append("")
    lines.append(f"{'Phase':<14}  {'AUC':>6}  {'Positives':>10}")
    lines.append("-" * 36)
    for i, cls in enumerate(label_encoder.classes_):
        y_bin   = (y == i).astype(int)
        cls_auc = roc_auc_score(y_bin, y_proba[:, i])
        lines.append(f"{cls:<14}  {cls_auc:.4f}  {y_bin.sum():>10,}")
    report = "\n".join(lines)
    logging.info("\n" + report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# RULE EXTRACTION FROM FINAL FIT
# ═══════════════════════════════════════════════════════════════════════════════

def extract_surviving_rules(
    logreg: LogisticRegression,
    feature_names: list[str],
    rule_meta: list[dict],
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Parse the L1 LogReg coefficients and return all non-zero rules.

    logreg.coef_ has shape (n_classes, n_combined_features).
    The first len(feature_names) columns correspond to the original z-score
    features; the remaining columns correspond to rule_meta entries.

    A positive coefficient for class C means: "when this rule fires (= 1),
    the model is more confident the sample belongs to class C."
    A negative coefficient means: "when this rule fires, the model is less
    confident — this rule actively pushes *away* from class C."

    Returns a DataFrame with columns:
        class_name, feature_type (original|rule), name_or_rule,
        coefficient, abs_coef, support_frac
    Sorted by class, then by abs_coef descending.
    """
    n_orig  = len(feature_names)
    rows: list[dict] = []

    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        coefs = logreg.coef_[cls_idx]         # (n_combined_features,)

        for feat_i, coef in enumerate(coefs):
            if coef == 0.0:                    # L1 drove this to zero — skip
                continue

            if feat_i < n_orig:
                # ── Original z-score feature ────────────────────────────────
                rows.append({
                    "class_name":    cls_name,
                    "feature_type":  "original",
                    "name_or_rule":  feature_names[feat_i].replace("_zscore", ""),
                    "coefficient":   float(coef),
                    "abs_coef":      abs(float(coef)),
                    "support_frac":  None,      # continuous features have no support
                    "tree_idx":      None,
                })
            else:
                # ── Binary rule feature ─────────────────────────────────────
                meta = rule_meta[feat_i - n_orig]
                rows.append({
                    "class_name":    cls_name,
                    "feature_type":  "rule",
                    "name_or_rule":  meta["rule_str"],
                    "coefficient":   float(coef),
                    "abs_coef":      abs(float(coef)),
                    "support_frac":  meta["support_frac"],
                    "tree_idx":      meta["tree_idx"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["class_name", "abs_coef"], ascending=[True, False])


def format_rules_report(df_rules: pd.DataFrame, top_n: int = 15) -> str:
    """Format the surviving rules as a human-readable text report.

    For each class:
      Positive coefficient rules (top_n by magnitude) → define the class
      Negative coefficient rules (top_n by magnitude) → contradict the class
    """
    lines = []

    for cls_name in sorted(df_rules["class_name"].unique()):
        cls_df = df_rules[df_rules["class_name"] == cls_name]
        pos = cls_df[cls_df["coefficient"] > 0].head(top_n)
        neg = cls_df[cls_df["coefficient"] < 0].head(top_n)

        lines.append("\n" + "═" * 60)
        lines.append(f"  CLASS: {cls_name.upper()}")
        lines.append("═" * 60)

        # ── Positive rules (push TOWARD this class) ──────────────────────
        if not pos.empty:
            lines.append(f"\n  ▲ POSITIVE rules (push TOWARD {cls_name}):")
            lines.append(f"  {'Coef':>7}  {'Support':>8}  Rule")
            lines.append("  " + "-" * 56)
            for _, row in pos.iterrows():
                support_str = (
                    f"{row['support_frac']*100:5.1f}%"
                    if row["support_frac"] is not None
                    else "  n/a "
                )
                rule_display = row["name_or_rule"].replace("\n    ", "\n             ")
                lines.append(
                    f"  {row['coefficient']:+7.4f}  {support_str}  {rule_display}"
                )

        # ── Negative rules (push AWAY from this class) ───────────────────
        if not neg.empty:
            lines.append(f"\n  ▼ NEGATIVE rules (push AWAY from {cls_name}):")
            lines.append(f"  {'Coef':>7}  {'Support':>8}  Rule")
            lines.append("  " + "-" * 56)
            for _, row in neg.iterrows():
                support_str = (
                    f"{row['support_frac']*100:5.1f}%"
                    if row["support_frac"] is not None
                    else "  n/a "
                )
                rule_display = row["name_or_rule"].replace("\n    ", "\n             ")
                lines.append(
                    f"  {row['coefficient']:+7.4f}  {support_str}  {rule_display}"
                )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_top_rules(
    df_rules: pd.DataFrame,
    label_encoder: LabelEncoder,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """2×2 bar chart showing the top surviving rules per class by |coefficient|."""
    classes = list(label_encoder.classes_)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, cls_name in zip(axes, classes):
        cls_df = df_rules[df_rules["class_name"] == cls_name].head(top_n)
        if cls_df.empty:
            ax.set_visible(False)
            continue

        # Truncate rule strings for display
        labels = []
        for r in cls_df["name_or_rule"]:
            # Replace newlines/indents, truncate to 60 chars
            flat = r.replace("\n    ", " / ")
            labels.append(flat[:65] + "…" if len(flat) > 65 else flat)

        coefs  = cls_df["coefficient"].values
        colors = [
            PHASE_COLORS.get(cls_name, "steelblue") if c > 0 else "#888888"
            for c in coefs
        ]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, coefs, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()   # largest |coef| at top
        ax.set_xlabel("L1 Logistic Regression coefficient", fontsize=9)
        ax.set_title(
            f"{cls_name}  —  top {len(labels)} surviving rules",
            fontsize=11, fontweight="bold",
            color=PHASE_COLORS.get(cls_name, "black"),
        )
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "RuleFit — Surviving Rules per Phase (L1 pruned)\n"
        "Coloured bars push TOWARD the phase · Grey bars push AWAY",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Rule bar chart → {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (same logic as scripts 27 / 12)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ml_arrays(cfg: dict, anchors: str = "with") -> tuple[pd.DataFrame, list[str]]:
    """Load step-06 timeline, compute z-scores, assign phases, aggregate to profiles.

    ── PLACEHOLDER ────────────────────────────────────────────────────────────
    Replace this function body if you already have a pre-built labeled DataFrame:

        df = pd.read_csv("path/to/your/labeled_data.csv")
        zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
        return df, zscore_cols

    Required columns: "author", "phase", and *_zscore feature columns.
    ───────────────────────────────────────────────────────────────────────────
    """
    interim_dir = Path(cfg["paths"]["interim"])

    pattern_map = {
        "with":    "timeline_daily_aggregated_with_anchors_*.csv",
        "without": "timeline_daily_aggregated_no_anchors_*.csv",
        "any":     "timeline_daily_aggregated_*.csv",
    }
    path = find_latest_file(interim_dir, pattern_map.get(anchors, pattern_map["with"]))
    if path is None:
        path = find_latest_file(interim_dir, "timeline_daily_aggregated_*.csv")
    if path is None:
        raise FileNotFoundError("No timeline_daily_aggregated_*.csv found. Run script 06 first.")

    logging.info(f"Loading timeline: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(df):,} user-days, {df['author'].nunique():,} users")

    mean_cols = [
        c for c in df.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = normalize_features_per_user_zscore(df, mean_cols, user_col="author")
    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]

    results_path = find_periodicity_results(interim_dir)
    if results_path is None:
        raise FileNotFoundError("No periodicity/consensus CSV found. Run scripts 07/08 first.")
    pdf = pd.read_csv(results_path, encoding="utf-8-sig", low_memory=False)
    if "consensus_period" in pdf.columns:
        user_period_map = dict(zip(pdf["user"].astype(str), pdf["consensus_period"].astype(float)))
    else:
        user_period_map = dict(pdf.groupby("user")["period"].median().astype(float))
    logging.info(f"  {len(user_period_map):,} users with detected cycles")

    df = df[df["author"].astype(str).isin(user_period_map)].copy()
    df["author"] = df["author"].astype(str)
    user_phase_df = compute_user_phase_definitions(
        {u: p for u, p in user_period_map.items() if u in set(df["author"])}
    )
    df = assign_phases_to_timeline(df, user_phase_df, user_col="author", time_col="offset_from_cd1")
    df = df[df["phase"].notna() & df["phase"].isin(PHASE_ORDER)].copy()

    profiles = df.groupby(["author", "phase"])[zscore_cols].mean().reset_index()
    profiles[zscore_cols] = profiles[zscore_cols].fillna(0)
    logging.info(f"  Phase profiles: {len(profiles):,} rows, {profiles['author'].nunique():,} users")
    return profiles, zscore_cols


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--anchors", choices=["with", "without", "any"], default="with")
    p.add_argument(
        "--n-estimators", type=int, default=100,
        help="Trees in the Random Forest (more trees = more candidate rules).",
    )
    p.add_argument(
        "--max-depth", type=int, default=3,
        help="Max tree depth. depth=3 → max 3 conditions per rule (recommended).",
    )
    p.add_argument(
        "--C", type=float, default=0.1,
        help="L1 LogReg inverse regularisation. Lower = fewer surviving rules. "
             "Try 0.05 for very sparse, 0.5 for more inclusive output.",
    )
    p.add_argument("--n-folds",  type=int, default=5)
    p.add_argument("--top-n",    type=int, default=12,
                   help="Rules per class to show in plots and text report.")
    p.add_argument("--skip-cv",  action="store_true",
                   help="Skip cross-validation; jump straight to final fit.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg        = load_config(args.config)
    output_dir = ROOT / "reports" / "rulefit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    logging.info("\n[1/4] Loading data…")
    df, zscore_cols = load_ml_arrays(cfg, anchors=args.anchors)

    le     = LabelEncoder()
    le.fit(PHASE_ORDER)
    y      = le.transform(df["phase"])
    X      = df[zscore_cols].values.astype(np.float32)
    groups = df["author"].values

    logging.info(
        f"\n  X shape : {X.shape}  ({len(zscore_cols)} features)"
        f"\n  Classes : {list(le.classes_)}"
        f"\n  Users   : {len(np.unique(groups)):,}"
        f"\n  Forest  : {args.n_estimators} trees × max_depth={args.max_depth}"
        f"\n  L1 C    : {args.C}"
    )

    # ── [2] Cross-validation ─────────────────────────────────────────────────
    if not args.skip_cv:
        logging.info(f"\n[2/4] GroupKFold CV ({args.n_folds} folds)…")
        y_pred, y_proba, fold_reports = run_cv(
            X, y, groups, zscore_cols, le,
            n_folds=args.n_folds,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            C=args.C,
        )
        oof_report = log_oof_metrics(y, y_proba, le, fold_reports)
        rpt_path = output_dir / f"rulefit_oof_report_{timestamp}.txt"
        rpt_path.write_text(
            f"RuleFit Pipeline — GroupKFold ({args.n_folds} folds)\n"
            f"Forest: n_estimators={args.n_estimators}, max_depth={args.max_depth}\n"
            f"L1 LogReg: C={args.C}\n\n"
            + oof_report
        )
        logging.info(f"  OOF report → {rpt_path.name}")
    else:
        logging.info("\n[2/4] CV skipped.")

    # ── [3] Final fit on full dataset ─────────────────────────────────────────
    logging.info("\n[3/4] Fitting final pipeline on full dataset…")

    # Step A: forest
    final_forest = build_forest(args.n_estimators, args.max_depth)
    final_forest.fit(X, y)

    # Extract rules from the full dataset (used to build rule_meta for display)
    X_rules_full, rule_meta = build_rule_feature_matrix(final_forest, X, zscore_cols)
    X_combined_full = combine_features(X, X_rules_full)
    logging.info(
        f"  Combined feature matrix: {X_combined_full.shape[1]:,} columns "
        f"({len(zscore_cols)} original + {X_rules_full.shape[1]:,} rules)"
    )

    # Step B: L1 logistic regression
    final_logreg = build_logreg(args.C)
    final_logreg.fit(X_combined_full, y)

    # Report sparsity
    n_total_coefs   = final_logreg.coef_.size
    n_nonzero_coefs = (final_logreg.coef_ != 0).sum()
    logging.info(
        f"  Non-zero coefficients: {n_nonzero_coefs:,} / {n_total_coefs:,} "
        f"({100*n_nonzero_coefs/n_total_coefs:.1f}% survived L1)"
    )

    # ── [4] Extract and display surviving rules ───────────────────────────────
    logging.info("\n[4/4] Extracting surviving rules…")

    df_rules = extract_surviving_rules(final_logreg, zscore_cols, rule_meta, le)

    n_surviving = len(df_rules)
    logging.info(f"  {n_surviving:,} non-zero rule/feature entries across all classes")
    for cls in le.classes_:
        cls_n = (df_rules["class_name"] == cls).sum()
        logging.info(f"    {cls:<14}: {cls_n:,} entries")

    # Text report
    rules_text = format_rules_report(df_rules, top_n=args.top_n)
    logging.info(rules_text)

    txt_path = output_dir / f"rulefit_rules_{timestamp}.txt"
    txt_path.write_text(
        f"RuleFit — Surviving Rules\n"
        f"Forest: {args.n_estimators} trees, max_depth={args.max_depth} | "
        f"L1 C={args.C}\n"
        + rules_text
    )
    logging.info(f"  Rules text → {txt_path.name}")

    # CSV (machine-readable)
    csv_path = output_dir / f"rulefit_rules_{timestamp}.csv"
    df_rules.to_csv(csv_path, index=False)
    logging.info(f"  Rules CSV  → {csv_path.name}")

    # Plot
    plot_path = output_dir / f"rulefit_top_rules_{timestamp}.png"
    plot_top_rules(df_rules, le, plot_path, top_n=args.top_n)

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
