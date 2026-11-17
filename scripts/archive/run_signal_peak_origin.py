"""Family-level SVD peak on the origin periodicity cohort (minposts20/span100).

Matches methods 2.3 (option b): per-user z-score -> family PCA (PC1) -> per-user
NUFFT per family -> per-user SVD across families -> importance -> top-2 re-SVD ->
average unified spectra of top-50% cycling users -> argmax = dominant period.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/interim/eligible_users_timeline_minposts20_span100_no_anchors_20260517T134609.csv"
FAM  = ROOT / "data/raw/Attribute_families_cycles.csv"
MAXOFF, MINP, MAXP = 90, 21, 40

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMS = sorted(fam[fc].unique())
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in FAMS}

num_freq = 5 * (MAXP - MINP)
f_grid = np.linspace(1 / MAXP, 1 / MINP, num_freq); w = 2 * np.pi * f_grid; periods = 1 / f_grid

df = pd.read_csv(DATA, low_memory=False, encoding="utf-8-sig")
df = df[df.offset_from_cd1.abs() <= MAXOFF].copy()
print(f"origin cohort: {df.author.nunique()} users, {len(df)} rows (|offset|<={MAXOFF})")

allfeats = [c for fl in FAMFEATS.values() for c in fl if c in df.columns]
g = df.groupby("author")
for c in allfeats:
    mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
    df[c] = (df[c] - mu) / sd
for fname, feats in FAMFEATS.items():
    feats = [c for c in feats if c in df.columns]
    C = df[feats].corr().values; _, V = np.linalg.eigh(np.nan_to_num(C))
    df[f"FAM::{fname}"] = df[feats].fillna(0.0).values @ V[:, -1]

users = sorted(df.author.unique()); uidx = {u: i for i, u in enumerate(users)}; nU = len(users)

def spectra(col):
    S = np.full((nU, num_freq), np.nan)
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        S[uidx[u]] = np.abs(x @ np.exp(-2j * np.pi * np.outer(t, w))) / len(x)
    return S

cube = np.stack([spectra(f"FAM::{f}") for f in FAMS], axis=0)
imp = np.full((nU, len(FAMS)), np.nan); unified = np.full((nU, num_freq), np.nan); conf = np.full(nU, np.nan)
for u in range(nU):
    M = cube[:, u, :]
    if np.isnan(M).any() or (M != 0).any(axis=1).sum() < 2: continue
    M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    imp[u] = np.abs(U[:, 0]); unified[u] = np.abs(Vt[0]); conf[u] = S[0] ** 2 / (S ** 2).sum()
valid = ~np.isnan(conf)
mean_imp = np.nanmean(imp[valid] / np.nanmax(imp[valid], axis=1, keepdims=True), axis=0)
order = np.argsort(mean_imp); top = order[-2:]
print("importance (low->high):", ", ".join(f"{FAMS[i]}={mean_imp[i]:.3f}" for i in order))

def resvd(idx):
    ns = np.full((nU, num_freq), np.nan); cf = np.full(nU, np.nan)
    for u in range(nU):
        M = cube[idx][:, u, :].copy()
        if np.isnan(M).any(): continue
        M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        ns[u] = np.abs(Vt[0]); cf[u] = S[0] ** 2 / (S ** 2).sum()
    v = ~np.isnan(cf); m = v & (cf > np.nanmedian(cf[v]))
    ms = np.nanmean(ns[m], axis=0)
    return periods[int(np.argmax(ms))], int(m.sum())

thr = np.nanmedian(conf[valid]); mask = valid & (conf > thr)
pk_all5 = periods[int(np.argmax(np.nanmean(unified[mask], axis=0)))]
pk_top2, ntop = resvd(top)
print(f"\nall-5 unified peak (top-50% n={mask.sum()}): {pk_all5:.2f} d")
print(f"top-2 [{', '.join(FAMS[i] for i in top)}] peak (top-50% n={ntop}): {pk_top2:.2f} d")
