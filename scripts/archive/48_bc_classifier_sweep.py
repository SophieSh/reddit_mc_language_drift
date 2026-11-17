"""BC classifier sweep: compare feature selection methods, feature counts, and EBM hyperparameters.

Runs all combinations of:
  - Feature selection methods: none, mannwhitney, anova_f, mutual_info, pointbiserial
  - Feature counts:            15, 25, 50, 100, full (~276)
  - EBM hyperparameter sets:   default, fast (fewer rounds), more_bins, interactions

Same F requirements: all_four_phases_present, combined pill only.

Output:
  reports/ebm/bc/sweep/sweep_results.csv   — AUC table
  reports/ebm/bc/sweep/sweep_heatmap.png   — visual summary

Usage:
  python scripts/48_bc_classifier_sweep.py \\
    --regular data/interim/non_bc_timeline_20260603T000200.csv \\
    --bc      data/interim/bc_timeline_20260603T000200.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import mannwhitneyu, f_oneway, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Sweep grid ────────────────────────────────────────────────────────────────

SELECTION_METHODS = ["none", "mannwhitney", "anova_f", "mutual_info", "pointbiserial"]
FEATURE_COUNTS    = [15, 25, 50, 100, "full"]

EBM_CONFIGS = {
    "default":      dict(max_bins=256, learning_rate=0.01, max_rounds=5000,
                         min_samples_leaf=2, interactions=0),
    "interactions5": dict(max_bins=256, learning_rate=0.01, max_rounds=5000,
                          min_samples_leaf=2, interactions=5),
    "more_bins":    dict(max_bins=512, learning_rate=0.01, max_rounds=5000,
                         min_samples_leaf=2, interactions=0),
    "fast":         dict(max_bins=128, learning_rate=0.05, max_rounds=2000,
                         min_samples_leaf=4, interactions=0),
}


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(X: np.ndarray, y: np.ndarray, method: str, n: int) -> np.ndarray:
    """Return indices of top-n features by the chosen method."""
    if method == "none" or n >= X.shape[1]:
        return np.arange(X.shape[1])

    # Replace NaN with column median for selection scoring only
    X_fill = X.copy()
    for j in range(X.shape[1]):
        col = X_fill[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med
        X_fill[:, j] = col

    bc_mask  = y == 1
    non_mask = y == 0
    scores   = np.zeros(X.shape[1])

    if method == "mannwhitney":
        for j in range(X.shape[1]):
            a = X_fill[bc_mask, j]; b = X_fill[non_mask, j]
            u, _ = mannwhitneyu(a, b, alternative="two-sided")
            scores[j] = abs(1 - 2 * u / (len(a) * len(b)))  # rank-biserial

    elif method == "anova_f":
        # sklearn f_classif returns (F-stats, p-values)
        from sklearn.feature_selection import f_classif
        f_stats, _ = f_classif(X_fill, y)
        scores = np.nan_to_num(f_stats)

    elif method == "mutual_info":
        scores = mutual_info_classif(X_fill, y, random_state=42)

    elif method == "pointbiserial":
        for j in range(X.shape[1]):
            r, _ = pointbiserialr(y, X_fill[:, j])
            scores[j] = abs(r)

    return np.argsort(scores)[::-1][:n]


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            sel_method: str, n_feat: int | str,
            ebm_cfg: dict, seed: int = 42, n_folds: int = 5) -> float:
    n = X.shape[1] if n_feat == "full" else int(n_feat)
    idx = select_features(X, y, sel_method, n)
    Xs  = X[:, idx]

    ebm = ExplainableBoostingClassifier(
        random_state=seed, n_jobs=-1, **ebm_cfg)

    sgkf    = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_proba = np.zeros(len(y))

    for train_idx, test_idx in sgkf.split(Xs, y, groups):
        X_tr, X_te = Xs[train_idx], Xs[test_idx]
        y_tr, y_te = y[train_idx],  y[test_idx]
        g_tr       = groups[train_idx]
        sw = compute_sample_weight("balanced", y_tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ebm.fit(X_tr, y_tr, sample_weight=sw)
        y_proba[test_idx] = ebm.predict_proba(X_te)[:, 1]

    return roc_auc_score(y, y_proba)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_feature_matrix(reg_path: str, bc_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load, phase-assign, all_four_phases_present filter, return (X, y, groups, feat_names)."""
    df_reg = pd.read_csv(reg_path, encoding="utf-8-sig", low_memory=False)
    df_bc  = pd.read_csv(bc_path,  encoding="utf-8-sig", low_memory=False)

    df_reg["bc"] = 0
    df_bc["bc"]  = 1
    if "pill_type" in df_bc.columns:
        df_bc = df_bc[df_bc["pill_type"].str.lower().str.contains("combined", na=False)]
    df = pd.concat([df_reg, df_bc], ignore_index=True)

    def _phase(offset):
        d = int(offset) % 28
        if d < 4:  return "Menstrual"
        if d < 14: return "Follicular"
        if d < 17: return "Ovulation"
        return "Luteal"

    df["phase"] = df["offset_from_cd1"].apply(_phase)

    non_feat = {"author", "offset_from_cd1", "phase", "bc", "pill_type", "bc_status"}
    num_feats = df.select_dtypes(include=[np.number]).columns
    feat_cols = [c for c in num_feats if c not in non_feat and not c.startswith("Unnamed:")]

    profiles = df.groupby(["author", "phase"])[feat_cols].mean().reset_index()
    wide = profiles.pivot(index="author", columns="phase", values=feat_cols)
    wide.columns = [f"{f}_{p}" for f, p in wide.columns]
    wide = wide.reset_index()

    bc_labels = df[["author", "bc"]].drop_duplicates("author")
    df_users  = wide.merge(bc_labels, on="author")

    # all_four_phases_present filter
    all_phases = {"Menstrual", "Follicular", "Ovulation", "Luteal"}
    user_phases = df.groupby("author")["phase"].apply(set)
    users_all4  = set(user_phases[user_phases.apply(lambda s: all_phases.issubset(s))].index)
    df_users = df_users[df_users["author"].isin(users_all4)]

    bc_src = df[["author","bc"]].drop_duplicates("author")
    df_users = df_users.merge(bc_src.rename(columns={"bc":"bc2"}), on="author", how="left")
    # use bc2 as authoritative
    if "bc2" in df_users.columns:
        df_users["bc"] = df_users["bc2"]
        df_users = df_users.drop(columns=["bc2"])

    feature_names = [c for c in df_users.columns if c not in ("author", "bc")]
    X      = df_users[feature_names].values.astype(np.float64)
    y      = df_users["bc"].values.astype(int)
    groups = df_users["author"].values

    n_bc  = (y == 1).sum()
    n_non = (y == 0).sum()
    log.info("Feature matrix: %d users (BC=%d, non-BC=%d), %d features",
             len(y), n_bc, n_non, X.shape[1])
    return X, y, groups, feature_names


def plot_heatmap(results: pd.DataFrame, output_dir: Path) -> None:
    for ebm_name, grp in results.groupby("ebm_config"):
        pivot = grp.pivot(index="selection_method", columns="n_features", values="auc")
        pivot = pivot.reindex(index=SELECTION_METHODS,
                              columns=[str(c) for c in FEATURE_COUNTS])

        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(pivot.values.astype(float), vmin=0.58, vmax=0.70,
                       cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("N features selected", fontsize=10)
        ax.set_ylabel("Selection method", fontsize=10)
        ax.set_title(f"OOF AUC — EBM config: {ebm_name}\n(baseline F-run: 0.6308 — no selection, 276 features)",
                     fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7.5,
                            color="black" if 0.62 < v < 0.67 else "white")
        plt.colorbar(im, ax=ax, label="AUC")
        fig.tight_layout()
        path = output_dir / f"sweep_heatmap_{ebm_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regular", required=True)
    parser.add_argument("--bc",      required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "ebm" / "bc" / "sweep")
    parser.add_argument("--n-folds",  type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--ebm-configs", nargs="+", default=list(EBM_CONFIGS.keys()),
                        choices=list(EBM_CONFIGS.keys()),
                        help="Which EBM configs to run (default: all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building feature matrix…")
    X, y, groups, feat_names = build_feature_matrix(args.regular, args.bc)

    total = len(SELECTION_METHODS) * len(FEATURE_COUNTS) * len(args.ebm_configs)
    log.info("Running %d combinations…", total)

    rows = []
    done = 0
    for ebm_name, sel_method, n_feat in product(
            args.ebm_configs, SELECTION_METHODS, FEATURE_COUNTS):

        # skip: "none" with counts other than "full" (redundant)
        if sel_method == "none" and n_feat != "full":
            done += 1
            continue

        label = f"{ebm_name} | {sel_method} | n={n_feat}"
        try:
            auc = run_one(X, y, groups, sel_method, n_feat,
                          EBM_CONFIGS[ebm_name], seed=args.seed,
                          n_folds=args.n_folds)
            rows.append({"ebm_config": ebm_name, "selection_method": sel_method,
                         "n_features": str(n_feat), "auc": auc})
            done += 1
            log.info("[%d/%d]  %-60s  AUC=%.4f", done, total, label, auc)
        except Exception as e:
            log.warning("FAILED %s: %s", label, e)
            rows.append({"ebm_config": ebm_name, "selection_method": sel_method,
                         "n_features": str(n_feat), "auc": np.nan})
            done += 1

    results = pd.DataFrame(rows)
    csv_path = args.output_dir / "sweep_results.csv"
    results.to_csv(csv_path, index=False)
    log.info("Saved %s", csv_path)

    plot_heatmap(results, args.output_dir)

    print("\n=== Top 15 configurations by AUC ===")
    print(results.sort_values("auc", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
