"""Power-matched control: subsample non-BC to 120 users (= BC size), 200 times,
and get the repeat-correlation distribution. If BC sits at the low tail, the
attenuation is real; if BC sits near the middle, it was just small-n noise.
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
BCSRC = ROOT / "data" / "interim" / "bc_timeline_20260603T000200.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
AW, MAX_GAP, MIN_SPAN, MIN_POST, WINDOW, NSUB, NBC = 28, 7, 28, 3, 28, 200, 120
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in ["Affective", "Lexical access"]}


def filter_chain(daily):
    within = daily[daily.offset_from_cd1.between(-AW, AW)]; elig = []
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
        if (right - left) >= MIN_SPAN and sum(1 for d in days if left <= d <= right) >= MIN_POST:
            elig.append(user)
    return set(elig)


def pc1(df, feats):
    feats = [c for c in feats if c in df.columns]; X = df[feats].copy()
    for c in feats:
        lo, hi = X[c].quantile([0.01, 0.99]); X[c] = X[c].clip(lo, hi)
    gg = df.groupby("author")
    for c in feats:
        mu = gg[c].transform("mean"); sd = gg[c].transform("std").replace(0, np.nan); X[c] = (X[c] - mu) / sd
    return PCA(1).fit_transform(X.fillna(0.0).values)[:, 0]


def repeat(df, feats):
    df = df.copy(); df["comp"] = pc1(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    s = df[["offset_from_cd1", "comp"]].dropna()
    gam = gam47.fit_gam(s[["offset_from_cd1"]].values, s["comp"].values, n_splines=10)
    XX = gam.generate_X_grid(term=0, n=300); pr = gam.predict(XX); x = XX[:, 0]
    p, c = pr[x < 0], pr[x >= 0]; n = min(len(p), len(c))
    return np.corrcoef(p[:n], c[:n])[0, 1]


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
bc_users = set(pd.read_csv(BCSRC, usecols=["author"]).author.unique())
elig = filter_chain(tot[["author", "offset_from_cd1"]])
nonbc = np.array(sorted(elig - bc_users)); bc = sorted(elig & bc_users)
win = tot[tot.offset_from_cd1.between(-WINDOW, WINDOW)]
print(f"non-BC={len(nonbc)}  BC={len(bc)}\n")

for fname, feats in FAMFEATS.items():
    bc_val = repeat(win[win.author.isin(bc)], feats)
    full = repeat(win[win.author.isin(nonbc)], feats)
    dist = np.empty(NSUB)
    for k in range(NSUB):
        sub = rng.choice(nonbc, NBC, replace=False)
        dist[k] = repeat(win[win.author.isin(sub)], feats)
    pct = (dist < bc_val).mean() * 100
    print(f"=== {fname} ===")
    print(f"  non-BC full (n=591)      repeat = {full:+.3f}")
    print(f"  non-BC @120 subsamples   mean={dist.mean():+.3f}  median={np.median(dist):+.3f}  "
          f"5th={np.percentile(dist,5):+.3f}  95th={np.percentile(dist,95):+.3f}")
    print(f"  BC (n=120)               repeat = {bc_val:+.3f}")
    print(f"  -> BC is at the {pct:.0f}th percentile of non-BC@120 "
          f"({(dist>=bc_val).mean()*100:.0f}% of subsamples >= BC)\n")
print("Done.")
