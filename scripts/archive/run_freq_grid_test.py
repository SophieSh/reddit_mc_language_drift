"""Does grid density change the conclusions? Per-family dominant period at grids
of 10 / 20 / 40 / 95 points (period 21-40 d), on the span100 and gap10/120
cohorts. ~4 independent frequencies live in this band, so 10-20 points already
exceed the data's resolution; finer grids should not change the band-level story.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPAN100 = ROOT / "data" / "interim" / "eligible_users_timeline_minposts20_span100_no_anchors_20260517T134609.csv"
TOTAL   = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM     = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXOFF, MINP, MAXP = 90, 21, 40
GRIDS = [10, 20, 40, 95]
MAXGAP, MINSPAN, AW = 10, 120, 200

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
        if left < 0 and right > 0 and (right - left) >= MINSPAN: elig.append(user)
    return set(elig)


def prep(df):
    df = df[df.offset_from_cd1.abs() <= MAXOFF].copy()
    feats = [c for fl in FAMFEATS.values() for c in fl if c in df.columns]
    g = df.groupby("author")
    for c in feats:
        mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
        df[c] = (df[c] - mu) / sd
    for fname, fl in FAMFEATS.items():
        fl = [c for c in fl if c in df.columns]
        C = df[fl].corr().values
        _, V = np.linalg.eigh(np.nan_to_num(C))
        df[f"FAM::{fname}"] = df[fl].fillna(0.0).values @ V[:, -1]
    return df


def fam_peak(df, col, nfreq):
    f_grid = np.linspace(1 / MAXP, 1 / MINP, nfreq); w = 2 * np.pi * f_grid; per = 1 / f_grid
    specs = []
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        specs.append(np.abs(x @ np.exp(-2j * np.pi * np.outer(t, w))) / len(x))
    ms = np.mean(specs, axis=0)
    return per[int(np.argmax(ms))]


print("Loading cohorts...")
sp = prep(pd.read_csv(SPAN100, low_memory=False))
tot = pd.read_csv(TOTAL, low_memory=False)
g10 = prep(tot[tot.author.isin(filter_chain(tot[["author", "offset_from_cd1"]]))])
cohorts = {f"span100 (n={sp.author.nunique()})": sp, f"gap10/120 (n={g10.author.nunique()})": g10}

# Rayleigh note
T = 180.0
print(f"\nBand 21-40 d, T~{T:.0f}d -> ~{(1/MINP-1/MAXP)*T:.1f} independent frequencies; "
      f"period resolution at 28d ~ {28**2/T:.1f} d\n")

for label, df in cohorts.items():
    print(f"=== {label} ===")
    print(f"{'family':16s} " + "".join(f"{'n='+str(n):>9s}" for n in GRIDS))
    for f in FAMS:
        peaks = [fam_peak(df, f"FAM::{f}", n) for n in GRIDS]
        print(f"{f:16s} " + "".join(f"{p:9.1f}" for p in peaks))
    print()
