"""Create a BC gap7 cohort from the total no-anchor timeline and run the
Affective/Lexical PC1 GAM on BC vs non-BC (identical gap7 pipeline).

gap7 = connected posting chain crossing the anchor, max_gap=7, within +/-28,
span>=28 (reproduces canonical 711 all / 591 non-BC / 120 BC). If the signal is
hormonal-menstrual it should attenuate in BC (cycle flattened by contraception).
"""
from __future__ import annotations
import importlib.util, sys, datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)

TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
BCSRC = ROOT / "data" / "interim" / "bc_timeline_20260603T000200.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
AW, MAX_GAP, MIN_SPAN, MIN_POST, WINDOW, NPERM = 28, 7, 28, 3, 28, 200
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in ["Affective", "Lexical access"]}


def filter_chain(daily):
    within = daily[daily.offset_from_cd1.between(-AW, AW)]
    elig = []
    for user, grp in within.groupby("author"):
        days = sorted(grp.offset_from_cd1.unique())
        if not days: continue
        before = [d for d in days if d <= 0]; after = [d for d in days if d >= 0]
        start = before[-1] if before else after[0]; si = days.index(start)
        right = start
        for i in range(si + 1, len(days)):
            if days[i] - days[i - 1] <= MAX_GAP: right = days[i]
            else: break
        left = start
        for i in range(si - 1, -1, -1):
            if days[i + 1] - days[i] <= MAX_GAP: left = days[i]
            else: break
        if not (left < 0 and right > 0): continue
        span = right - left; n = sum(1 for d in days if left <= d <= right)
        if span >= MIN_SPAN and n >= MIN_POST: elig.append(user)
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


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
bc_users = set(pd.read_csv(BCSRC, usecols=["author"]).author.unique())
elig = filter_chain(tot[["author", "offset_from_cd1"]])
bc_g7, nonbc_g7 = elig & bc_users, elig - bc_users
print(f"gap7 eligible: all={len(elig)}  non-BC={len(nonbc_g7)}  BC={len(bc_g7)}")

# save BC gap7 dataset (+/-28 window, anchors already excluded)
ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
bc_df = tot[tot.author.isin(bc_g7) & tot.offset_from_cd1.between(-WINDOW, WINDOW)].copy()
outp = ROOT / "data" / "interim" / f"eligible_users_two_cycles_span28_gap7_no_anchors_bc_{ts}.csv"
bc_df.to_csv(outp, index=False); print(f"Saved BC gap7 dataset: {outp.name} ({bc_df.author.nunique()} users, {len(bc_df)} rows)")

print(f"\n{'family':16s} {'cohort':8s} {'users':>6s} {'repeat':>8s} {'perm_p':>7s}")
print("-" * 52)
for fname, feats in FAMFEATS.items():
    for label, users in [("non-BC", nonbc_g7), ("BC", bc_g7)]:
        df = tot[tot.author.isin(users) & tot.offset_from_cd1.between(-WINDOW, WINDOW)].copy()
        df["comp"] = pc1(df, feats)
        df["comp"] -= df.groupby("author")["comp"].transform("mean")
        d = df[["author", "offset_from_cd1", "comp"]].dropna()
        off, y, auth = d.offset_from_cd1.values, d.comp.values, d.author.values
        real = rc(off, y)
        order = np.argsort(auth, kind="stable"); a_s, off_s, y_s = auth[order], off[order], y[order]
        bnd = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1], True])
        null = np.empty(NPERM)
        for k in range(NPERM):
            op = off_s.copy()
            for b in range(len(bnd) - 1):
                i, j = bnd[b], bnd[b + 1]; op[i:j] = rng.permutation(op[i:j])
            null[k] = rc(op, y_s)
        pval = (1 + np.sum(null >= real)) / (1 + NPERM)
        print(f"{fname:16s} {label:8s} {d.author.nunique():6d} {real:+8.3f} {pval:7.3f}")
print("\nDone.")
