"""Frequency (NUFFT) analysis on the dense gap10/span>=120 cohort.

Reports the dominant period for every family (PC1 composite) and every feature,
computed TWO ways:
  (A) mean spectrum over ALL users
  (B) mean spectrum over the TOP-50% cycling-confidence users
      (confidence = first-singular-value ratio of each user's family x freq SVD)
so we can see what the top-50% selection step actually buys.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXGAP, MINSPAN, AW, MAXOFF = 10, 120, 200, 90
MINP, MINPERIOD, MAXPERIOD = 21, 21, 40

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


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
elig = filter_chain(tot[["author", "offset_from_cd1"]])
df = tot[tot.author.isin(elig) & (tot.offset_from_cd1.abs() <= MAXOFF)].copy()
print(f"gap10/span>=120 cohort: {df.author.nunique()} users, {len(df)} rows (|offset|<={MAXOFF})\n")

allfeats = [c for fl in FAMFEATS.values() for c in fl if c in df.columns]
# per-user z-score every feature
g = df.groupby("author")
for c in allfeats:
    mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
    df[c] = (df[c] - mu) / sd

# family PC1 composites (correlation-eig, like the pipeline)
def pc1(cols):
    C = df[cols].corr().values
    w, V = np.linalg.eigh(np.nan_to_num(C, nan=0.0))
    z = df[cols].fillna(0.0).values
    return z @ V[:, -1]
for fname, feats in FAMFEATS.items():
    feats = [c for c in feats if c in df.columns]
    df[f"FAM::{fname}"] = pc1(feats)

# frequency grid
num_freq = 5 * (MAXPERIOD - MINPERIOD)
f_grid = np.linspace(1 / MAXPERIOD, 1 / MINPERIOD, num_freq)
w_grid = 2 * np.pi * f_grid
periods = 1 / f_grid

users = sorted(df.author.unique())
uidx = {u: i for i, u in enumerate(users)}
nU = len(users)


def spectra_for(col):
    """per-user NUFFT magnitude spectrum (detrend, /N) -> [nU x num_freq], NaN if <4 pts."""
    S = np.full((nU, num_freq), np.nan)
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean()
        x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))  # linear detrend
        E = np.exp(-2j * np.pi * np.outer(t, w_grid))
        S[uidx[u]] = np.abs(x @ E) / len(x)
    return S


# family spectra cube + per-user cycling confidence (SVD across families)
fam_cols = [f"FAM::{f}" for f in FAMS]
fam_cube = np.stack([spectra_for(c) for c in fam_cols], axis=0)   # [nFam x nU x nf]
conf = np.full(nU, np.nan)
for u in range(nU):
    M = fam_cube[:, u, :]
    if np.isnan(M).any() or (M != 0).any(axis=1).sum() < 2: continue
    M = M - M.mean(axis=1, keepdims=True)
    rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
    s = np.linalg.svd(M, compute_uv=False)
    conf[u] = s[0] ** 2 / (s ** 2).sum()
valid = ~np.isnan(conf)
thr = np.nanmedian(conf[valid])
top50 = valid & (conf > thr)
print(f"cycling confidence computed for {valid.sum()} users; top-50% = {top50.sum()}\n")


def peak(S, mask):
    ms = np.nanmean(S[mask], axis=0)
    return periods[int(np.argmax(ms))]


print(f"{'level':7s} {'name':16s} {'peak_ALL':>9s} {'peak_TOP50':>11s}")
print("-" * 48)
rows = []
for f in FAMS:
    S = fam_cube[FAMS.index(f)]
    pa, pt = peak(S, valid), peak(S, top50)
    print(f"{'family':7s} {f:16s} {pa:9.2f} {pt:11.2f}")
    rows.append(("family", f, pa, pt))
print()
for c in allfeats:
    S = spectra_for(c)
    have = ~np.isnan(S).all(axis=1)
    pa = peak(S, valid & have); pt = peak(S, top50 & have)
    print(f"{'feature':7s} {c.replace('_mean',''):16.16s} {pa:9.2f} {pt:11.2f}")
    rows.append(("feature", c.replace("_mean", ""), pa, pt))

out = ROOT / "outputs" / "freq_gap10_120.csv"
pd.DataFrame(rows, columns=["level", "name", "peak_all_users", "peak_top50"]).to_csv(out, index=False)
print(f"\nSaved: {out}")
