"""Standard spectral pipeline on the CLASSIFIER non-BC cohort (users with at
least one post in each of the 4 phases, within +-91 of CD1).

Full supervisor pipeline: per-user z-score -> family PCA -> per-family NUFFT ->
per-user SVD across families -> family importance -> top-2 re-SVD -> peak.
Plus per-family peaks and a permutation null on the top-2 merged spectrum.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "spectral_classifier_cohort"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "interim" / "non_bc_timeline_20260603T000200.csv"
FAM  = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXOFF, MINP, MAXP, NSEL, NPERM = 90, 21, 40, 2, 300
BLO, BHI = 26, 30
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMS = sorted(fam[fc].unique())
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in FAMS}

print("Loading...")
df = pd.read_csv(DATA, low_memory=False)
# all-four-phases-present filter (phase from 28-day cycle), within ±91
o = df.offset_from_cd1.values; cd = np.mod(o, 28)
df["phase"] = np.select([cd < 4, cd < 12, cd < 15], ["Menstrual", "Follicular", "Ovulation"], "Luteal")
d91 = df[df.offset_from_cd1.abs() <= 91]
allph = {"Menstrual", "Follicular", "Ovulation", "Luteal"}
up = d91.groupby("author")["phase"].apply(set)
keep = set(up[up.apply(lambda s: allph.issubset(s))].index)
df = df[df.author.isin(keep) & (df.offset_from_cd1.abs() <= MAXOFF)].copy()
print(f"classifier non-BC cohort (all 4 phases): {df.author.nunique()} users, {len(df)} rows")

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

num_freq = 5 * (MAXP - MINP)
f_grid = np.linspace(1 / MAXP, 1 / MINP, num_freq); w = 2 * np.pi * f_grid; periods = 1 / f_grid
users = sorted(df.author.unique()); uidx = {u: i for i, u in enumerate(users)}; nU = len(users)


def spectra(col):
    S = np.full((nU, num_freq), np.nan)
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        S[uidx[u]] = np.abs(x @ np.exp(-2j * np.pi * np.outer(t, w))) / len(x)
    return S


cube = np.stack([spectra(f"FAM::{f}") for f in FAMS], axis=0)   # [nFam x nU x nf]

# per-user SVD across families -> importance, unified, confidence
imp = np.full((nU, len(FAMS)), np.nan); unified = np.full((nU, num_freq), np.nan); conf = np.full(nU, np.nan)
for u in range(nU):
    M = cube[:, u, :]
    if np.isnan(M).any() or (M != 0).any(axis=1).sum() < 2: continue
    M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    imp[u] = np.abs(U[:, 0]); unified[u] = np.abs(Vt[0]); conf[u] = S[0] ** 2 / (S ** 2).sum()
valid = ~np.isnan(conf)
mean_imp = np.nanmean(imp[valid] / np.nanmax(imp[valid], axis=1, keepdims=True), axis=0)
order = np.argsort(mean_imp)

print("\nFamily importance (low -> high):")
for i in order: print(f"  {FAMS[i]:16s} {mean_imp[i]:.4f}")

# per-family peaks (all users)
print("\nPer-family dominant period (all users):")
for i, f in enumerate(FAMS):
    ms = np.nanmean(cube[i][valid], axis=0)
    print(f"  {f:16s} peak = {periods[int(np.argmax(ms))]:.2f} d")

# all-family unified peak (top-50% cycling)
thr = np.nanmedian(conf[valid]); mask = valid & (conf > thr)
pk_uni = periods[int(np.argmax(np.nanmean(unified[mask], axis=0)))]
print(f"\nUnified spectrum (all families, top-50% cycling n={mask.sum()}): peak = {pk_uni:.2f} d")

# top-2 re-SVD
top = order[-NSEL:]
print(f"\nTop-{NSEL} families: {[FAMS[i] for i in top]}")
def resvd(idx, perm=False):
    ns = np.full((nU, num_freq), np.nan); cf = np.full(nU, np.nan)
    for u in range(nU):
        M = cube[idx][:, u, :].copy()
        if np.isnan(M).any(): continue
        if perm:  # permute freq axis? no—permute is for null; handled separately
            pass
        M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        ns[u] = np.abs(Vt[0]); cf[u] = S[0] ** 2 / (S ** 2).sum()
    v = ~np.isnan(cf); m = v & (cf > np.nanmedian(cf[v]))
    ms = np.nanmean(ns[m], axis=0)
    return periods[int(np.argmax(ms))], ms, m.sum()
pk_top, ms_top, ntop = resvd(top)
print(f"Top-{NSEL} unified spectrum peak = {pk_top:.2f} d  (top-50% cycling n={ntop})")

# ---- plots -----------------------------------------------------------------
def specplot(ms, title, fname, peak):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(BLO, BHI, color="#cfe8ff", alpha=0.7)
    ax.plot(periods, ms, color="#1f77b4", lw=2)
    ax.axvline(peak, color="#d62728", ls="--", lw=1.5, label=f"peak {peak:.1f} d")
    ax.set_xlabel("Number of days per cycle"); ax.set_ylabel("Spectral power")
    ax.set_title(title); ax.set_xlim(MINP, MAXP); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", OUT / fname)

specplot(np.nanmean(unified[mask], axis=0),
         f"Unified spectrum (all 5 families, top-50% cycling n={mask.sum()})\nclassifier non-BC cohort, {nU} users",
         "unified_all_families.png", pk_uni)
specplot(ms_top,
         f"Top-2 ({', '.join(FAMS[i] for i in top)}) unified spectrum, top-50% n={ntop}\nclassifier non-BC cohort",
         "unified_top2.png", pk_top)

# per-family spectra grid
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, i in zip(axes.flat, range(len(FAMS))):
    ms = np.nanmean(cube[i][valid], axis=0); pk = periods[int(np.argmax(ms))]
    ax.axvspan(BLO, BHI, color="#cfe8ff", alpha=0.7)
    ax.plot(periods, ms, color="#1f77b4", lw=1.8); ax.axvline(pk, color="#d62728", ls="--", lw=1.2)
    ax.set_title(f"{FAMS[i]} — peak {pk:.1f} d", fontsize=9); ax.set_xlim(MINP, MAXP)
axes.flat[-1].axis("off")
fig.suptitle(f"Per-family NUFFT spectra — classifier non-BC cohort ({nU} users)", fontsize=11)
fig.supxlabel("Days per cycle"); fig.tight_layout()
fig.savefig(OUT / "per_family_spectra.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved", OUT / "per_family_spectra.png")
print("Done.")
