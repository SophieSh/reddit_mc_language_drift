"""Shared utilities for EBM scripts (38, 35, 42).

All EBM construction, cross-validation, and interpretation logic lives here so
each script only contains its own plotting and CLI.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight


# ── FACTORY ───────────────────────────────────────────────────────────────────

def make_ebm(seed: int, interactions: int = 0,
             learning_rate: float = 0.01, max_rounds: int = 5000,
             max_bins: int = 256, min_samples_leaf: int = 2) -> ExplainableBoostingClassifier:
    """Return unfitted EBM with configurable hyperparameters."""
    return ExplainableBoostingClassifier(
        max_bins=max_bins,
        max_interaction_bins=64,
        interactions=interactions,
        learning_rate=learning_rate,
        max_rounds=max_rounds,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


# ── BIN GEOMETRY ──────────────────────────────────────────────────────────────

def bin_midpoints(names: list) -> np.ndarray:
    """N+1 float bin edges → N midpoints.  Replaces _parse_bin_midpoints everywhere."""
    edges = np.asarray(names, dtype=np.float64)   # hard-fail if not numeric
    assert len(edges) >= 2, f"bin_midpoints: expected ≥2 edges, got {len(edges)}"
    return (edges[:-1] + edges[1:]) / 2


def bin_density(data: dict, n: int) -> np.ndarray:
    """Normalised per-bin density array of length n, aligned by midpoint x.

    Hard-fails if density is missing or malformed.
    """
    density_raw = data["density"]                    # KeyError if absent — intentional
    assert isinstance(density_raw, dict), \
        f"bin_density: expected density dict, got {type(density_raw)}"

    dens_counts = np.asarray(density_raw["scores"], dtype=float)
    dens_edges  = np.asarray(density_raw["names"],  dtype=float)

    assert len(dens_counts) > 0 and len(dens_edges) >= 2, \
        "bin_density: empty density histogram"
    if dens_counts.sum() == 0:
        # Feature has no variation in this dataset — return uniform density
        logging.warning(f"bin_density: all-zero density histogram, returning uniform distribution")
        return np.ones(n, dtype=float) / n

    dens_frac = dens_counts / dens_counts.sum()

    # Fast path: density resolution already matches score bins.
    if len(dens_counts) == n:
        return dens_frac

    # General path: map score-bin midpoints → density bins via shared x-axis.
    score_edges = np.asarray(data["names"], dtype=np.float64)
    assert len(score_edges) >= n + 1, \
        f"bin_density: score edges ({len(score_edges)}) too short for {n} bins"

    score_midpoints = (score_edges[:n] + score_edges[1:n + 1]) / 2
    bin_indices = np.clip(
        np.searchsorted(dens_edges[1:], score_midpoints, side="left"),
        0, len(dens_frac) - 1,
    )
    out = dens_frac[bin_indices]
    return out / out.sum()


# ── INTERPRETATION ─────────────────────────────────────────────────────────────

def global_importance(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """Density-weighted mean |log-odds| per feature per class.

    Columns: feature, {class}_imp for each class, global_avg.
    """
    global_exp = ebm.explain_global()
    rows = []

    for fi, feat in enumerate(feature_names):
        data   = global_exp.data(fi)
        scores = np.asarray(data["scores"])
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]

        n        = scores.shape[0]
        density  = bin_density(data, n)

        weighted_imp = np.sum(np.abs(scores) * density[:, np.newaxis], axis=0)
        row = {"feature": feat}
        for i, cls in enumerate(class_names):
            row[f"{cls}_imp"] = float(weighted_imp[i])
        row["global_avg"] = float(np.mean(weighted_imp))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("global_avg", ascending=False).reset_index(drop=True)


def signed_importance(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
) -> np.ndarray:
    """Density-weighted mean log-odds per feature (signed, binary EBM only).

    Positive = toward class 1.
    """
    global_exp = ebm.explain_global()
    result = np.empty(len(feature_names), dtype=np.float64)

    for fi in range(len(feature_names)):
        data   = global_exp.data(fi)
        scores = np.asarray(data["scores"], dtype=np.float64)
        if scores.ndim == 2:
            scores = scores[:, 0]
        n       = len(scores)
        density = bin_density(data, n)
        result[fi] = float(np.sum(scores * density))   # signed: + = toward class 1

    return result


def shape_functions(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
    class_names: list[str],
) -> dict[str, dict]:
    """For each feature: x (midpoints), y (log-odds per class), density, weighted_sum.

    Returns dict keyed by feature name.

    Binary EBM: scores is 1-D (log-odds toward class 1).  class_names[0]=class0,
    class_names[1]=class1.  class1 y = scores; class0 y = -scores.

    Multiclass EBM: scores is 2-D (n_bins, n_classes).  Each column is one class.
    """
    global_exp = ebm.explain_global()
    out = {}

    for fi, feat in enumerate(feature_names):
        data   = global_exp.data(fi)
        scores = np.asarray(data["scores"])

        x       = bin_midpoints(data["names"])
        n       = min(len(x), scores.shape[0])
        x       = x[:n]
        scores  = scores[:n]
        density = bin_density(data, n)

        per_class = {}
        if scores.ndim == 1:
            # Binary EBM: single score axis = log-odds toward class 1.
            assert len(class_names) == 2, \
                f"shape_functions: 1-D scores but {len(class_names)} class_names"
            per_class[class_names[1]] = {
                "x": x, "y": scores, "density": density,
                "weighted_sum": float(np.sum(np.abs(scores) * density)),
            }
            per_class[class_names[0]] = {
                "x": x, "y": -scores, "density": density,
                "weighted_sum": float(np.sum(np.abs(scores) * density)),
            }
        else:
            for ci, cls in enumerate(class_names):
                if ci >= scores.shape[1]:
                    continue
                y = scores[:, ci]
                per_class[cls] = {
                    "x": x, "y": y, "density": density,
                    "weighted_sum": float(np.sum(np.abs(y) * density)),
                }
        out[feat] = per_class

    return out


# ── CROSS-VALIDATION ───────────────────────────────────────────────────────────

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    seed: int,
    n_splits: int = 5,
    learning_rate: float = 0.01,
    max_rounds: int = 5000,
    max_bins: int = 256,
    min_samples_leaf: int = 2,
    interactions: int = 0,
    top_n_per_fold: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, list[float]]
    | tuple[np.ndarray, np.ndarray, dict, list[float]]
):
    """StratifiedGroupKFold OOF for any EBM (binary or multiclass).

    Returns (y_pred, y_proba, fold_aucs). y_proba is always 2D: (n_samples, n_classes).
    Uses balanced sample weights internally.  fold_aucs is a list of per-fold AUC values
    (float or nan when a fold has only one class).

    If top_n_per_fold is set, also returns fold_counts as the third element: a dict
    mapping feature index → number of folds it appeared in the top-N by |signed
    importance|.  Features present in ALL folds have count == n_splits.
    The return signature in that case is (y_pred, y_proba, fold_counts, fold_aucs).
    """
    gkf        = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred     = np.empty_like(y)
    y_proba    = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_counts: dict[int, int] = {}
    fold_aucs: list[float] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        logging.info(f"  Fold {fold}/{n_splits}  train={len(tr):,}  test={len(te):,}")
        ebm = make_ebm(seed, interactions=interactions,
                       learning_rate=learning_rate, max_rounds=max_rounds,
                       max_bins=max_bins, min_samples_leaf=min_samples_leaf)
        ebm.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        y_pred[te]  = ebm.predict(X[te])
        proba       = ebm.predict_proba(X[te])
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        y_proba[te] = proba

        n_classes_fold = len(np.unique(y[te]))
        if n_classes_fold >= 2:
            if proba.shape[1] == 2:
                auc = roc_auc_score(y[te], proba[:, 1])
            else:
                auc = roc_auc_score(y[te], proba, multi_class="ovr", average="macro")
            fold_aucs.append(float(auc))
            logging.info(f"    → AUC={auc:.4f}")
        else:
            fold_aucs.append(float("nan"))
            logging.info(f"    → single class in test fold, AUC skipped")

        if top_n_per_fold is not None:
            dummy_names = list(range(X.shape[1]))
            sim = signed_importance(ebm, dummy_names)
            top_idx = np.argsort(np.abs(sim))[-top_n_per_fold:]
            for i in top_idx:
                fold_counts[int(i)] = fold_counts.get(int(i), 0) + 1

    if top_n_per_fold is not None:
        return y_pred, y_proba, fold_counts, fold_aucs
    return y_pred, y_proba, fold_aucs


# ── DISPLAY ────────────────────────────────────────────────────────────────────

def clean_name(name: str) -> str:
    """Strip _zscore suffix and replace underscores with spaces."""
    return name.replace("_zscore", "").replace("_", " ")
