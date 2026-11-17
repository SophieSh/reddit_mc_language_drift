#!/usr/bin/env python3
"""Does a CYCLE-SHAPE layer improve the EBM birth-control classifier?

Motivated by the ebm_idea finding: trajectory shape lost to level order-statistics
in ACUITY problems, but MIGHT win where "variability itself is the biomarker". BC
detection is exactly that case: the hypothesis is that hormonal contraception
FLATTENS the cycle, so the discriminating signal is the AMPLITUDE of cyclic
variation, not the level. Two women with identical mean valence differ only in
whether valence oscillates across the cycle.

Nested test (mirrors ebm_idea C-vs-B'):
  B   phase-means only (the current 42_ebm_bc_classifier baseline)
  B'  phase-means + per-phase posting counts  (STRONG baseline: knows sparsity)
  C   B' + cycle-shape: 28-day harmonic amplitude (+ noise-corrected) + fit R^2

Guardrail (the ebm_idea lesson): a per-user amplitude from sparse posts is inflated
by sampling noise, so we (a) noise-correct the amplitude and (b) give B' the posting
counts, so any "shape gain" is measured over a baseline that already knows how sparse
each user is. Posting volume is matched between groups (checked separately), so this
is about sensitivity, not bias.

Usage: 49_ebm_bc_cycle_shape.py [--bc FILE --nonbc FILE] [--period 28]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ebm_utils import run_cv

BC_DEFAULT = "data/interim/bc_timeline_20260603T000200.csv"          # raw _mean cols -> ~0.645
NONBC_DEFAULT = "data/interim/non_bc_timeline_20260603T000200.csv"
PHASES = ["Menstrual", "Follicular", "Ovulation", "Luteal"]


def restrict_all4(df):
    """all_four_phases_present: keep only users with posts in every phase (their config)."""
    up = df.groupby("author")["phase"].apply(set)
    keep = set(up[up.apply(lambda s: set(PHASES).issubset(s))].index)
    return df[df["author"].isin(keep)]


def fselect(M, y, n=50):
    """Top-n features by |rank-biserial| (Mann-Whitney), mirroring the pipeline's pre-select."""
    if n is None or n >= M.shape[1]:
        return M
    from scipy.stats import mannwhitneyu
    X = M.values
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        c = X[:, j]; ok = ~np.isnan(c)
        a, b = c[ok & (y == 1)], c[ok & (y == 0)]
        if len(a) < 5 or len(b) < 5:
            continue
        U, _ = mannwhitneyu(a, b)
        scores[j] = abs(2 * U / (len(a) * len(b)) - 1)
    top = np.sort(np.argsort(scores)[::-1][:n])
    return M.iloc[:, top]


def calculate_phase(offset, period=28, m=4, f=7, o=3):
    if pd.isna(offset):
        return np.nan
    day = int(offset) % period
    if day < m:         return "Menstrual"
    if day < m + f:     return "Follicular"
    if day < m + f + o: return "Ovulation"
    if day < period:    return "Luteal"
    return np.nan


def phase_means(df, feats):
    """B: mean per (author, phase), pivoted wide -> <feat>_<phase>."""
    p = df.groupby(["author", "phase"], dropna=True)[feats].mean()
    w = p.unstack("phase")
    w.columns = [f"{c}_{ph}" for c, ph in w.columns]
    return w


def phase_counts(df):
    """per-user posting-day counts per phase + total (the sparsity descriptors for B')."""
    c = df.groupby(["author", "phase"]).size().unstack("phase").reindex(columns=PHASES)
    c.columns = [f"n_{ph}" for ph in PHASES]
    c["n_total"] = c.sum(axis=1)
    return c


def cycle_shape(df, feats, period=28, min_obs=6):
    """C-layer: per (author, feature) 28-day harmonic fit.
    y ~ c + a*cos(2*pi*x/period) + b*sin(2*pi*x/period)  ->  amplitude sqrt(a^2+b^2).
    Returns per-user wide frame: <feat>_amp, <feat>_ampcorr (noise-corrected), <feat>_R2.
    Vectorized: one lstsq per user solves all features at once.
    """
    rows = {}
    for author, sub in df.groupby("author"):
        x = sub["offset_from_cd1"].to_numpy(float)
        Y = sub[feats].to_numpy(float)                       # (n, F)
        n = len(x)
        if n < min_obs:
            continue
        X = np.column_stack([np.ones(n), np.cos(2*np.pi*x/period), np.sin(2*np.pi*x/period)])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)          # (3, F)
        resid = Y - X @ beta                                  # (n, F)
        ss_res = (resid**2).sum(0)                            # (F,)
        ss_tot = ((Y - Y.mean(0))**2).sum(0)
        a, b = beta[1], beta[2]
        amp2 = a**2 + b**2
        # noise correction: subtract sampling variance of (a,b).
        # var(coef) = sigma^2 * diag((X'X)^-1); sigma^2 = ss_res/(n-3)
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
            cos_var, sin_var = xtx_inv[1, 1], xtx_inv[2, 2]
        except np.linalg.LinAlgError:
            cos_var = sin_var = np.nan
        dof = max(n - 3, 1)
        sigma2 = ss_res / dof                                 # (F,)
        amp2_corr = np.maximum(0.0, amp2 - sigma2 * (cos_var + sin_var))
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / ss_tot
        row = {}
        for j, fn in enumerate(feats):
            row[f"{fn}_amp"] = np.sqrt(amp2[j])
            row[f"{fn}_ampcorr"] = np.sqrt(amp2_corr[j])
            row[f"{fn}_R2"] = r2[j]
        rows[author] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def delong_test(y, p1, p2):
    """Fast DeLong test for two correlated ROC AUCs (same labels). Returns (auc1, auc2, p)."""
    y = np.asarray(y); order = np.argsort(-y)  # positives first not required; use groups
    pos = y == 1; neg = ~pos
    def _auc_cov(p):
        m, n = pos.sum(), neg.sum()
        px, py = p[pos], p[neg]
        # midrank structural components (Sun & Xu)
        def midrank(z):
            J = np.argsort(z); Z = z[J]; N = len(z); T = np.zeros(N)
            i = 0
            while i < N:
                j = i
                while j < N and Z[j] == Z[i]:
                    j += 1
                T[i:j] = 0.5*(i+j-1)+1
                i = j
            T2 = np.empty(N); T2[J] = T
            return T2
        tx = midrank(px); ty = midrank(py); tz = midrank(p)
        auc = (tz[pos].sum() - m*(m+1)/2) / (m*n)
        v01 = (tz[pos] - tx) / n
        v10 = 1.0 - (tz[neg] - ty) / m
        return auc, v01, v10, m, n
    a1, v01_1, v10_1, m, n = _auc_cov(p1)
    a2, v01_2, v10_2, _, _ = _auc_cov(p2)
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    S = s01/m + s10/n
    var = S[0, 0] + S[1, 1] - 2*S[0, 1]
    if var <= 0:
        return a1, a2, 1.0
    z = (a1 - a2) / np.sqrt(var)
    p = 2*(1 - stats.norm.cdf(abs(z)))
    return a1, a2, p


def cv_auc(X, y, groups, seed=0):
    Xv = X.values.astype(np.float64)
    _, proba, _ = run_cv(Xv, y, groups, n_classes=2, seed=seed, n_splits=5)
    return proba[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default=BC_DEFAULT)
    ap.add_argument("--nonbc", default=NONBC_DEFAULT)
    ap.add_argument("--period", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=91, help="restrict offset_from_cd1 to +/- N")
    ap.add_argument("--fselect", type=int, default=50, help="top-N feature pre-select (None to disable)")
    ap.add_argument("--all4", action="store_true", default=True)
    args = ap.parse_args()

    dbc = pd.read_csv(ROOT/args.bc, encoding="utf-8-sig", low_memory=False); dbc["bc"] = 1
    dnb = pd.read_csv(ROOT/args.nonbc, encoding="utf-8-sig", low_memory=False); dnb["bc"] = 0
    non_feat = {"author", "offset_from_cd1", "phase", "bc", "pill_type", "bc_status"}
    feats = sorted((set(dbc.columns) & set(dnb.columns)) - non_feat)
    feats = [f for f in feats if not f.startswith("Unnamed:")
             and np.issubdtype(dbc[f].dtype, np.number)]
    df = pd.concat([dbc, dnb], ignore_index=True)
    if args.window:
        df = df[df["offset_from_cd1"].between(-args.window, args.window)]
    df["phase"] = df["offset_from_cd1"].apply(lambda o: calculate_phase(o, args.period))
    df = df.dropna(subset=["phase"])
    if args.all4:
        df = restrict_all4(df)
    print(f"features={len(feats)} | after window/all4: BC={df[df.bc==1].author.nunique()} "
          f"nonBC={df[df.bc==0].author.nunique()} | fselect={args.fselect}")

    labels = df.groupby("author")["bc"].first()
    B  = phase_means(df, feats)
    Nc = phase_counts(df)
    Sh = cycle_shape(df, feats, args.period)
    idx = B.index
    y = labels.loc[idx].astype(int).values
    groups = np.asarray(idx)

    matrices = {
        "B  (phase-means)":            B,
        "B' (phase-means + counts)":   B.join(Nc),
        "C  (B' + cycle-shape)":       B.join(Nc).join(Sh.reindex(idx)),
    }
    proba = {}
    print(f"\n{'model':<28}{'AUROC':>8}{'AUPRC':>8}{'n_in':>7}{'n_sel':>7}")
    for name, M in matrices.items():
        M = M.reindex(idx)
        n_in = M.shape[1]
        Msel = fselect(M, y, args.fselect)
        p = cv_auc(Msel, y, groups, args.seed)
        proba[name] = p
        print(f"{name:<28}{roc_auc_score(y,p):>8.4f}{average_precision_score(y,p):>8.4f}"
              f"{n_in:>7}{Msel.shape[1]:>7}")

    a_bp, a_c, pval = delong_test(y, proba["B' (phase-means + counts)"], proba["C  (B' + cycle-shape)"])
    print(f"\nSHAPE GAIN (C vs B'):  ΔAUROC = {a_c-a_bp:+.4f}   DeLong p = {pval:.3f}")
    base = float(np.mean(y))
    print(f"base rate={base:.3f}")

    out = ROOT/"reports"/"ebm"/"bc"/"cycle_shape_probs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"author": idx, "y": y, **proba}).to_csv(out, index=False)
    print(f"saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
