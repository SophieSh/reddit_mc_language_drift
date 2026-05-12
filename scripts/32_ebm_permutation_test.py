#!/usr/bin/env python3
"""Step 32 — Within-user label permutation test for EBM phase classification.

For each permutation: shuffle phase labels within each user (keeps user-level
feature distributions intact, breaks phase→feature relationship), refit EBM
with GroupKFold CV, record macro OvR AUC.

Compares observed AUC from script 27 against the null distribution to test
whether the phase signal is real or an artifact of user-level variation.

Usage:
  python scripts/32_ebm_permutation_test.py --config configs/base.yaml
  python scripts/32_ebm_permutation_test.py --n-permutations 100 --n-folds 5
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ml_data import load_phase_labeled_dataset

warnings.filterwarnings("ignore", category=UserWarning)


def _make_ebm(seed: int) -> ExplainableBoostingClassifier:
    return ExplainableBoostingClassifier(
        max_bins=256,
        max_interaction_bins=64,
        interactions=0,
        learning_rate=0.01,
        max_rounds=1000,   # fewer rounds than real run — sufficient for null distribution
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def run_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> float:
    """Run GroupKFold CV and return macro OvR AUC."""
    n_classes = len(np.unique(y))
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba = np.zeros((len(y), n_classes), dtype=np.float64)

    for train_idx, test_idx in gkf.split(X, y, groups):
        ebm = _make_ebm(seed)
        ebm.fit(X[train_idx], y[train_idx])
        y_proba[test_idx] = ebm.predict_proba(X[test_idx])

    return roc_auc_score(y, y_proba, multi_class="ovr", average="macro")


def permute_labels_within_users(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each user, randomly shuffle their phase labels in-place."""
    y_perm = y.copy()
    for user in np.unique(groups):
        mask = groups == user
        idx = np.where(mask)[0]
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--observed-auc", type=float, default=None,
                        help="Observed AUC from script 27 (skips re-running real CV). "
                             "If omitted, real CV is run first.")
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = ROOT / "reports" / "ebm"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── [1] Load data ─────────────────────────────────────────────────────────
    logging.info("[1] Loading phase-labeled dataset…")
    df, zscore_cols, le = load_phase_labeled_dataset(cfg, no_anchors=args.no_anchors)

    y      = le.transform(df["phase"])
    X      = df[zscore_cols].values.astype(np.float64)
    groups = df["author"].values

    logging.info(f"  {X.shape[0]:,} profiles, {X.shape[1]} features, "
                 f"{len(np.unique(groups)):,} users")

    # ── [2] Observed AUC ─────────────────────────────────────────────────────
    if args.observed_auc is not None:
        observed_auc = args.observed_auc
        logging.info(f"[2] Using provided observed AUC: {observed_auc:.4f}")
    else:
        logging.info(f"[2] Running real CV ({args.n_folds} folds) for observed AUC…")
        observed_auc = run_cv_auc(X, y, groups, args.n_folds, seed=args.seed)
        logging.info(f"  Observed macro AUC: {observed_auc:.4f}")

    # ── [3] Permutation null distribution ────────────────────────────────────
    logging.info(f"[3] Running {args.n_permutations} within-user permutations…")
    null_aucs: list[float] = []

    for i in range(args.n_permutations):
        y_perm = permute_labels_within_users(y, groups, rng)

        # Skip permutations where some CV fold ends up with only one class
        # (can happen with very few users per phase after shuffling).
        try:
            auc = run_cv_auc(X, y_perm, groups, args.n_folds, seed=args.seed + i + 1)
            null_aucs.append(auc)
        except ValueError as e:
            logging.warning(f"  Permutation {i+1} skipped: {e}")
            continue

        if (i + 1) % 10 == 0 or i == 0:
            logging.info(f"  [{i+1}/{args.n_permutations}]  "
                         f"null AUC = {auc:.4f}  "
                         f"(running mean = {np.mean(null_aucs):.4f})")

    # ── [4] Results ───────────────────────────────────────────────────────────
    null_arr = np.array(null_aucs)
    # Monte Carlo p-value: fraction of null AUCs >= observed (add 1 to numerator
    # and denominator per Phipson & Smyth 2010 for unbiased estimate).
    p_value = (np.sum(null_arr >= observed_auc) + 1) / (len(null_arr) + 1)

    logging.info("\n" + "=" * 55)
    logging.info("PERMUTATION TEST RESULTS")
    logging.info("=" * 55)
    logging.info(f"  Observed macro AUC : {observed_auc:.4f}")
    logging.info(f"  Null mean ± std     : {null_arr.mean():.4f} ± {null_arr.std():.4f}")
    logging.info(f"  Null min / max      : {null_arr.min():.4f} / {null_arr.max():.4f}")
    logging.info(f"  n permutations      : {len(null_arr)}")
    logging.info(f"  p-value (MC)        : {p_value:.4f}")
    logging.info("=" * 55)

    # Save results
    results = {
        "observed_auc": [observed_auc],
        "null_mean": [null_arr.mean()],
        "null_std": [null_arr.std()],
        "null_min": [null_arr.min()],
        "null_max": [null_arr.max()],
        "n_permutations": [len(null_arr)],
        "p_value_mc": [p_value],
    }
    out_path = output_dir / f"ebm_permutation_test_{timestamp}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    logging.info(f"\n  Summary → {out_path.name}")

    # Save full null distribution
    null_path = output_dir / f"ebm_permutation_null_dist_{timestamp}.csv"
    pd.DataFrame({"null_auc": null_arr}).to_csv(null_path, index=False)
    logging.info(f"  Null dist → {null_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
