"""Per-family permutation null test on the gap10/span>=120 cohort.
Statistic = band-peak (26-30 d) / overall spectrum level. Null = permute each
user's signal against its fixed time set (destroys time-locking, keeps sampling
+ red-noise structure). Tells us whether the clean ~27 d Affective peak (and the
others) beats chance at n=174.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXGAP, MINSPAN, AW, MAXOFF = 10, 120, 200, 90
MINP, MAXP, NPERM = 21, 40, 500
rng = np.random.default_rng(20260621)

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


print("Loading...")
tot = pd.read_csv(TOTAL, low_memory=False)
df = tot[tot.author.isin(filter_chain(tot[["author", "offset_from_cd1"]])) & (tot.offset_from_cd1.abs() <= MAXOFF)].copy()
print(f"{df.author.nunique()} users, {len(df)} rows")

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


def stat_band(ms, lo, hi):
    m = (periods >= lo) & (periods <= hi)
    return ms[m].max() / ms.mean()


print(f"\nNPERM={NPERM}\n")
print(f"{'family':16s} {'peak':>6s} {'band':>7s} {'real_st':>8s} {'null_st':>8s} {'p(26-30)':>9s} {'p(25-31)':>9s}")
print("-" * 70)
for fname in FAMS:
    col = f"FAM::{fname}"
    real_sum = np.zeros(num_freq); null_sum = np.zeros((NPERM, num_freq)); nC = 0
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        E = np.exp(-2j * np.pi * np.outer(t, w))
        real_sum += np.abs(x @ E) / len(x)
        XP = np.array([x[rng.permutation(len(x))] for _ in range(NPERM)])
        null_sum += np.abs(XP @ E) / len(x)
        nC += 1
    ms = real_sum / nC; null_ms = null_sum / nC
    pk = periods[int(np.argmax(ms))]
    for (lo, hi), key in [((26, 30), "p1"), ((25, 31), "p2")]:
        rs = stat_band(ms, lo, hi); ns = np.array([stat_band(null_ms[p], lo, hi) for p in range(NPERM)])
        if key == "p1": rs1, ns1, p1 = rs, ns.mean(), (1 + (ns >= rs).sum()) / (1 + NPERM)
        else: p2 = (1 + (ns >= rs).sum()) / (1 + NPERM)
    inb = "yes" if 26 <= pk <= 30 else ""
    print(f"{fname:16s} {pk:6.1f} {inb:>7s} {rs1:8.4f} {ns1:8.4f} {p1:9.3f} {p2:9.3f}")
print("\nDone.")
