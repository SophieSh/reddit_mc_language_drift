"""SVD pair-merge (original-pipeline style) for every family pair, gap10/span>=120.

For each pair of family PC1 composites: per user, take the [2 x freqs] spectrum
matrix, row-center + row-z-score, SVD, unified = |V[:,0]|, conf = s0^2/sum(s^2).
Dominant period from the mean unified spectrum over (a) all users and
(b) top-50% cycling-confidence users (pair-specific confidence, as the original).
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXGAP, MINSPAN, AW, MAXOFF = 10, 120, 200, 90
MINPERIOD, MAXPERIOD, BLO, BHI = 21, 40, 26, 30

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMS = sorted(fam[fc].unique())
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in FAMS}


def filter_chain(daily):
    within = daily[daily.offset_from_cd1.between(-AW, AW)]; elig = []
    for user, grp in within.groupby("author"):
        days = sorted(grp.offset_from_cd1.unique())
        if not days: continue
        before = [d for d in days if d <= 0]; after = [d for d in days if d >= 0]
        start = before[-1] if before else after[0]; si = days.index(start)
        right = start
        for i in range(si + 1, len(days)):
            if days[i] - days[i - 1] <= MAXGAP: right = days[i]
            else: break
        left = start
        for i in range(si - 1, -1, -1):
            if days[i + 1] - days[i] <= MAXGAP: left = days[i]
            else: break
        if not (left < 0 and right > 0): continue
        if (right - left) >= MINSPAN: elig.append(user)
    return set(elig)


print("Loading...")
tot = pd.read_csv(TOTAL, low_memory=False)
elig = filter_chain(tot[["author", "offset_from_cd1"]])
df = tot[tot.author.isin(elig) & (tot.offset_from_cd1.abs() <= MAXOFF)].copy()
print(f"{df.author.nunique()} users, {len(df)} rows\n")

allfeats = [c for fl in FAMFEATS.values() for c in fl if c in df.columns]
g = df.groupby("author")
for c in allfeats:
    mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
    df[c] = (df[c] - mu) / sd
for fname, feats in FAMFEATS.items():
    feats = [c for c in feats if c in df.columns]
    C = df[feats].corr().values
    _, V = np.linalg.eigh(np.nan_to_num(C))
    df[f"FAM::{fname}"] = df[feats].fillna(0.0).values @ V[:, -1]

num_freq = 5 * (MAXPERIOD - MINPERIOD)
f_grid = np.linspace(1 / MAXPERIOD, 1 / MINPERIOD, num_freq)
w_grid = 2 * np.pi * f_grid; periods = 1 / f_grid

users = sorted(df.author.unique()); uidx = {u: i for i, u in enumerate(users)}; nU = len(users)


def spectra_for(col):
    S = np.full((nU, num_freq), np.nan)
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean()
        x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        E = np.exp(-2j * np.pi * np.outer(t, w_grid))
        S[uidx[u]] = np.abs(x @ E) / len(x)
    return S


cube = {f: spectra_for(f"FAM::{f}") for f in FAMS}   # per-family spectra


def svd_merge(fi, fj):
    uni = np.full((nU, num_freq), np.nan); conf = np.full(nU, np.nan)
    Si, Sj = cube[fi], cube[fj]
    for u in range(nU):
        if np.isnan(Si[u]).any() or np.isnan(Sj[u]).any(): continue
        M = np.vstack([Si[u], Sj[u]])
        M = M - M.mean(axis=1, keepdims=True)
        rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        uni[u] = np.abs(Vt[0]); conf[u] = s[0] ** 2 / (s ** 2).sum()
    valid = ~np.isnan(conf)
    thr = np.nanmedian(conf[valid]); top = valid & (conf > thr)
    pk_all = periods[int(np.argmax(np.nanmean(uni[valid], axis=0)))]
    pk_top = periods[int(np.argmax(np.nanmean(uni[top], axis=0)))]
    return pk_all, pk_top, valid.sum(), top.sum()


print(f"{'pair':30s} {'peak_ALL':>9s} {'peak_TOP50':>11s}   band?")
print("-" * 60)
rows = []
for fi, fj in itertools.combinations(FAMS, 2):
    pa, pt, nv, nt = svd_merge(fi, fj)
    ba = "ALL✓" if BLO <= pa <= BHI else ""
    bt = "TOP✓" if BLO <= pt <= BHI else ""
    print(f"{fi+' + '+fj:30s} {pa:9.2f} {pt:11.2f}   {ba} {bt}")
    rows.append((f"{fi} + {fj}", pa, pt, nv, nt))
pd.DataFrame(rows, columns=["pair", "peak_all", "peak_top50", "n_valid", "n_top50"]).to_csv(
    ROOT / "outputs" / "freq_gap10_pairs.csv", index=False)
print(f"\nSaved: outputs/freq_gap10_pairs.csv")
