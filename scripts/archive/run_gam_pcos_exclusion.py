"""Effect of removing diagnosed-PCOS users from the gap7 GAM cohort, per family.
Compares repeat-correlation + permutation p for each family PC1, full cohort vs
cohort with diagnosed-PCOS users removed.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)

TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
AW, MAXGAP, MINSPAN, WINDOW, NPERM = 28, 7, 28, 28, 200
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


def pc1(df, feats):
    feats = [c for c in feats if c in df.columns]; X = df[feats].copy()
    for c in feats:
        lo, hi = X[c].quantile([0.01, 0.99]); X[c] = X[c].clip(lo, hi)
    gg = df.groupby("author")
    for c in feats:
        mu = gg[c].transform("mean"); sd = gg[c].transform("std").replace(0, np.nan); X[c] = (X[c] - mu) / sd
    return PCA(1).fit_transform(X.fillna(0.0).values)[:, 0]


def rc(off, y):
    sub = pd.DataFrame({"x": off, "y": y}).dropna()
    gam = gam47.fit_gam(sub[["x"]].values, sub["y"].values, n_splines=10)
    XX = gam.generate_X_grid(term=0, n=300); pr = gam.predict(XX); x = XX[:, 0]
    p, c = pr[x < 0], pr[x >= 0]; n = min(len(p), len(c))
    return np.corrcoef(p[:n], c[:n])[0, 1]


def gam_family(df, feats):
    df = df.copy(); df["comp"] = pc1(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    d = df[["author", "offset_from_cd1", "comp"]].dropna()
    off, y, auth = d.offset_from_cd1.values, d.comp.values, d.author.values
    real = rc(off, y)
    order = np.argsort(auth, kind="stable"); a_s, o_s, y_s = auth[order], off[order], y[order]
    bnd = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1], True])
    null = np.empty(NPERM)
    for k in range(NPERM):
        op = o_s.copy()
        for b in range(len(bnd) - 1):
            i, j = bnd[b], bnd[b + 1]; op[i:j] = rng.permutation(op[i:j])
        null[k] = rc(op, y_s)
    return real, (1 + np.sum(null >= real)) / (1 + NPERM), d.author.nunique()


print("Loading...")
tot = pd.read_csv(TOTAL, low_memory=False)
cohort = filter_chain(tot[["author", "offset_from_cd1"]])
diag = set(pd.read_csv(ROOT / "outputs" / "pcos_diagnosed_users.csv").author)
removed = cohort & diag
clean = cohort - diag
print(f"gap7 cohort: {len(cohort)} | diagnosed-PCOS in it: {len(removed)} | clean cohort: {len(clean)}\n")

win = tot[tot.offset_from_cd1.between(-WINDOW, WINDOW)]
full_df = win[win.author.isin(cohort)]
clean_df = win[win.author.isin(clean)]

print(f"{'family':16s} {'FULL repeat (p)':>20s} {'CLEAN repeat (p)':>20s}")
print("-" * 60)
for f in FAMS:
    feats = FAMFEATS[f]
    r0, p0, n0 = gam_family(full_df, feats)
    r1, p1, n1 = gam_family(clean_df, feats)
    print(f"{f:16s}   {r0:+.3f} (p={p0:.3f})      {r1:+.3f} (p={p1:.3f})")
print(f"\n(FULL n={len(cohort)}, CLEAN n={len(clean)})")
print("Done.")
