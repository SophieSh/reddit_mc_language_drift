"""Shared ML evaluation utilities for the menstrual-cycle NLP pipeline.

Functions here are used by both script 12 (XGBoost) and script 27 (EBM) and
must not contain model-specific logic.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from src.constants import CLASSIC_Z_HIGH, CLASSIC_Z_LOW


def _compute_tail_stats(
    mask_in_tail: np.ndarray,
    phase_mask: np.ndarray,
    n_phase: int,
    n_rest: int,
    feat: np.ndarray,
    alternative: str = "greater",
) -> tuple[tuple[float, float, float], float, float, float]:
    """Compute tail enrichment statistics for one feature × one tail direction.

    Args:
        mask_in_tail: Boolean array (same length as feat) marking samples in the tail.
        phase_mask:   Boolean array marking samples belonging to the target phase.
        n_phase:      Total number of phase samples (denominator for phase_pct).
        n_rest:       Total number of non-phase samples (denominator for rest_pct).
        feat:         Feature values (NaN-free, same length as phase_mask).
        alternative:  Alternative hypothesis for Fisher's exact test ("greater" or "less").

    Returns:
        ((phase_pct, rest_pct, risk_ratio), severity_median, severity_mean, p_value)
    """
    phase_in = phase_mask & mask_in_tail
    rest_in  = (~phase_mask) & mask_in_tail
    phase_pct = phase_in.sum() / n_phase * 100
    rest_pct  = rest_in.sum()  / n_rest  * 100
    rr = phase_pct / max(rest_pct, 0.001)
    sev_med  = float(np.median(feat[phase_in])) if phase_in.sum() > 0 else float("nan")
    sev_mean = float(np.mean(feat[phase_in]))   if phase_in.sum() > 0 else float("nan")
    ct = np.array([
        [phase_in.sum(), (phase_mask & ~mask_in_tail).sum()],
        [rest_in.sum(),  ((~phase_mask) & ~mask_in_tail).sum()],
    ])
    _, p = stats.fisher_exact(ct, alternative=alternative)
    return (phase_pct, rest_pct, rr), sev_med, sev_mean, p


def run_phase_statistical_gauntlet(
    X: np.ndarray,
    y_binary: np.ndarray,
    feature_names: list[str],
    top_indices: list[int],
    empirical_tail_pct: int = 10,
) -> pd.DataFrame:
    """Mann-Whitney U + Fisher tail tests (BH-FDR) for OvR top features.

    Non-parametric (empirical) tests use data-driven 10th/90th percentile thresholds.
    Parametric (classic) tests use fixed ±1.28 SD thresholds (theoretical 10% tails
    of a standard normal), which are comparable across features and runs.

    Both severity metrics (median and mean z-score of phase users past the threshold)
    are returned for each tail so downstream analyses can distinguish typical vs mean
    effects in that subgroup.

    All p-values are BH-FDR corrected within their test family (global, empirical high,
    empirical low, classic high, classic low — five separate corrections).

    Args:
        X:                  Feature matrix (n_samples, n_features); NaN is allowed.
        y_binary:           Binary label vector — 1 = target phase, 0 = rest.
        feature_names:      List of feature name strings, length == X.shape[1].
        top_indices:        Indices into feature_names / X columns to test.
        empirical_tail_pct: Percentile for empirical tail thresholds (default 10).

    Returns:
        DataFrame with one row per feature and columns for every test statistic.
    """
    feat_names = []
    p_global_raw = []
    p_high_raw, p_low_raw = [], []
    p_ch_raw, p_cl_raw = [], []

    median_shifts, mean_shifts = [], []
    hi_stats, lo_stats = [], []
    hi_sev_med, hi_sev_mean = [], []
    lo_sev_med, lo_sev_mean = [], []
    ch_stats, cl_stats = [], []
    ch_sev_med, ch_sev_mean = [], []
    cl_sev_med, cl_sev_mean = [], []

    phase_mask_global = y_binary == 1

    for idx in top_indices:
        raw_feat = X[:, idx].astype(float)

        # Isolate only valid (non-NaN) rows for this feature before any math.
        # NaNs in X are intentional for XGBoost but must be removed for scipy tests:
        # mannwhitneyu propagates NaN by default, and including NaN users in the
        # denominator artificially shrinks tail percentages.
        valid_mask = ~np.isnan(raw_feat)
        feat = raw_feat[valid_mask]
        phase_mask = phase_mask_global[valid_mask]

        n_phase = phase_mask.sum()
        n_rest  = (~phase_mask).sum()

        if n_phase == 0 or n_rest == 0:
            continue

        # ── Global shift (non-parametric median + parametric mean) ───────────
        median_shifts.append(np.median(feat[phase_mask]) - np.median(feat[~phase_mask]))
        mean_shifts.append(np.mean(feat[phase_mask])   - np.mean(feat[~phase_mask]))

        _, pg = stats.mannwhitneyu(feat[phase_mask], feat[~phase_mask], alternative="two-sided")
        p_global_raw.append(pg)

        # ── Empirical tails: data-driven 10th / 90th percentile ─────────────
        hi_mask = feat >= np.percentile(feat, 100 - empirical_tail_pct)
        (hi_pct, hi_rpct, hi_rr), h_smed, h_smean, ph = _compute_tail_stats(
            hi_mask, phase_mask, n_phase, n_rest, feat
        )
        hi_stats.append((hi_pct, hi_rpct, hi_rr))
        hi_sev_med.append(h_smed)
        hi_sev_mean.append(h_smean)
        p_high_raw.append(ph)

        lo_mask = feat <= np.percentile(feat, empirical_tail_pct)
        (lo_pct, lo_rpct, lo_rr), l_smed, l_smean, pl = _compute_tail_stats(
            lo_mask, phase_mask, n_phase, n_rest, feat
        )
        lo_stats.append((lo_pct, lo_rpct, lo_rr))
        lo_sev_med.append(l_smed)
        lo_sev_mean.append(l_smean)
        p_low_raw.append(pl)

        # ── Classic z-score tails: fixed ±1.28 thresholds ───────────────────
        ch_mask = feat >= CLASSIC_Z_HIGH
        (ch_pct, ch_rpct, ch_rr), ch_smed, ch_smean, pch = _compute_tail_stats(
            ch_mask, phase_mask, n_phase, n_rest, feat
        )
        ch_stats.append((ch_pct, ch_rpct, ch_rr))
        ch_sev_med.append(ch_smed)
        ch_sev_mean.append(ch_smean)
        p_ch_raw.append(pch)

        cl_mask = feat <= CLASSIC_Z_LOW
        (cl_pct, cl_rpct, cl_rr), cl_smed, cl_smean, pcl = _compute_tail_stats(
            cl_mask, phase_mask, n_phase, n_rest, feat
        )
        cl_stats.append((cl_pct, cl_rpct, cl_rr))
        cl_sev_med.append(cl_smed)
        cl_sev_mean.append(cl_smean)
        p_cl_raw.append(pcl)

        feat_names.append(feature_names[idx])

    _, pg_fdr,  _, _ = multipletests(p_global_raw, method="fdr_bh")
    _, ph_fdr,  _, _ = multipletests(p_high_raw,   method="fdr_bh")
    _, pl_fdr,  _, _ = multipletests(p_low_raw,    method="fdr_bh")
    _, pch_fdr, _, _ = multipletests(p_ch_raw,     method="fdr_bh")
    _, pcl_fdr, _, _ = multipletests(p_cl_raw,     method="fdr_bh")

    return pd.DataFrame({
        "feature":                      feat_names,
        # global
        "global_p_fdr":                 pg_fdr,
        "global_median_shift":          median_shifts,
        "global_mean_shift":            mean_shifts,
        # empirical high tail
        "high_tail_p_fdr":              ph_fdr,
        "high_phase_pct":               [s[0] for s in hi_stats],
        "high_rest_pct":                [s[1] for s in hi_stats],
        "high_rr":                      [s[2] for s in hi_stats],
        "high_severity_median":         hi_sev_med,
        "high_severity_mean":           hi_sev_mean,
        # empirical low tail
        "low_tail_p_fdr":               pl_fdr,
        "low_phase_pct":                [s[0] for s in lo_stats],
        "low_rest_pct":                 [s[1] for s in lo_stats],
        "low_rr":                       [s[2] for s in lo_stats],
        "low_severity_median":          lo_sev_med,
        "low_severity_mean":            lo_sev_mean,
        # classic high tail (z >= +1.28)
        "classic_high_tail_p_fdr":      pch_fdr,
        "classic_high_phase_pct":       [s[0] for s in ch_stats],
        "classic_high_rest_pct":        [s[1] for s in ch_stats],
        "classic_high_rr":              [s[2] for s in ch_stats],
        "classic_high_severity_median": ch_sev_med,
        "classic_high_severity_mean":   ch_sev_mean,
        # classic low tail (z <= -1.28)
        "classic_low_tail_p_fdr":       pcl_fdr,
        "classic_low_phase_pct":        [s[0] for s in cl_stats],
        "classic_low_rest_pct":         [s[1] for s in cl_stats],
        "classic_low_rr":               [s[2] for s in cl_stats],
        "classic_low_severity_median":  cl_sev_med,
        "classic_low_severity_mean":    cl_sev_mean,
    })


def log_oof_metrics(
    y: np.ndarray,
    y_proba: np.ndarray,
    label_encoder,
) -> str:
    """Compute and log macro AUC + per-class AUC.

    Args:
        y:             Integer label array (OOF ground truth).
        y_proba:       Probability matrix of shape (n_samples, n_classes).
        label_encoder: Fitted sklearn LabelEncoder whose .classes_ attribute
                       maps integer indices to class names.

    Returns:
        Formatted multi-line string with macro AUC and per-class breakdown.
    """
    lines = []
    macro_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    lines.append(f"OOF Macro AUC (OvR): {macro_auc:.4f}  [random = 0.500]")
    lines.append("")
    lines.append(f"{'Phase':<14}  {'AUC':>6}  {'Positives':>10}")
    lines.append("-" * 36)
    for i, cls in enumerate(label_encoder.classes_):
        y_bin = (y == i).astype(int)
        cls_auc = roc_auc_score(y_bin, y_proba[:, i])
        lines.append(f"{cls:<14}  {cls_auc:.4f}  {y_bin.sum():>10,}")

    report = "\n".join(lines)
    logging.info("\n" + report)
    return report
