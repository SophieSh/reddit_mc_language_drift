"""(1) Random-removal control: is the PCOS-removal effect PCOS-specific or just
    the effect of removing ~61 arbitrary users? Remove 61 random cohort users B
    times, get each family's repeat distribution, mark where PCOS-removal falls.
(2) Also remove diagnosed + suspected PCOS and re-check per family.
"""
from __future__ import annotations
import importlib.util, sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)

TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
LAB   = ROOT / "data" / "raw" / "pcos_users_aggregated_with_labels.xlsx"
AW, MAXGAP, MINSPAN, WINDOW, NDRAW, NPERM = 28, 7, 28, 28, 200, 200
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMS = sorted(fam[fc].unique())
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in FAMS}


def parse(s):
    if not isinstance(s, str): return {}
    s = re.sub(r"^```(json)?|```$", "", s.strip())
    try: return json.loads(s)
    except:
        m = re.search(r"\{.*\}", s, re.S)
        try: return json.loads(m.group(0)) if m else {}
        except: return {}


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


def repeat_for(win, users, feats):
    df = win[win.author.isin(users)].copy(); df["comp"] = pc1(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    d = df[["offset_from_cd1", "comp"]].dropna()
    return rc(d.offset_from_cd1.values, d.comp.values)


def perm_p(win, users, feats, real):
    df = win[win.author.isin(users)].copy(); df["comp"] = pc1(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    d = df[["author", "offset_from_cd1", "comp"]].dropna()
    off, y, a = d.offset_from_cd1.values, d.comp.values, d.author.values
    o = np.argsort(a, kind="stable"); a_s, o_s, y_s = a[o], off[o], y[o]
    bnd = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1], True])
    null = np.empty(NPERM)
    for k in range(NPERM):
        op = o_s.copy()
        for b in range(len(bnd) - 1):
            i, j = bnd[b], bnd[b + 1]; op[i:j] = rng.permutation(op[i:j])
        null[k] = rc(op, y_s)
    return (1 + np.sum(null >= real)) / (1 + NPERM)


print("Loading...")
lab = pd.read_excel(LAB); p = lab.response.apply(parse)
lab["status"] = p.apply(lambda d: d.get("pcos_status"))
diag = set(lab[lab.status == "diagnosed"].author)
susp = set(lab[lab.status == "suspected"].author)

tot = pd.read_csv(TOTAL, low_memory=False)
cohort = filter_chain(tot[["author", "offset_from_cd1"]])
win = tot[tot.offset_from_cd1.between(-WINDOW, WINDOW)]
users = np.array(sorted(cohort))
k_diag = len(cohort & diag)
k_both = len(cohort & (diag | susp))
print(f"gap7 cohort={len(cohort)} | diagnosed-in-cohort={k_diag} | diagnosed+suspected-in-cohort={k_both}\n")

clean_diag = cohort - diag
clean_both = cohort - diag - susp

print("=== (1) Random-removal control (remove", k_diag, "random users,", NDRAW, "draws) ===")
print(f"{'family':16s} {'FULL':>7s} {'PCOS-rm':>8s} {'rand mean[5,95]':>22s} {'pctile of PCOS-rm':>18s}")
for f in FAMS:
    feats = FAMFEATS[f]
    r_full = repeat_for(win, cohort, feats)
    r_pcos = repeat_for(win, clean_diag, feats)
    dist = np.array([repeat_for(win, cohort - set(rng.choice(users, k_diag, replace=False)), feats)
                     for _ in range(NDRAW)])
    pct = (dist < r_pcos).mean() * 100
    print(f"{f:16s} {r_full:+7.3f} {r_pcos:+8.3f}   {dist.mean():+.3f}[{np.percentile(dist,5):+.2f},{np.percentile(dist,95):+.2f}]   {pct:5.0f}th")

print("\n=== (2) Remove diagnosed + suspected PCOS ===")
print(f"{'family':16s} {'FULL rep (p)':>18s} {'diag-removed (p)':>20s} {'diag+susp-removed (p)':>24s}")
for f in FAMS:
    feats = FAMFEATS[f]
    r0 = repeat_for(win, cohort, feats);      p0 = perm_p(win, cohort, feats, r0)
    r1 = repeat_for(win, clean_diag, feats);  p1 = perm_p(win, clean_diag, feats, r1)
    r2 = repeat_for(win, clean_both, feats);  p2 = perm_p(win, clean_both, feats, r2)
    print(f"{f:16s}   {r0:+.3f} ({p0:.3f})      {r1:+.3f} ({p1:.3f})        {r2:+.3f} ({p2:.3f})")
print(f"\n(FULL={len(cohort)}, diag-removed={len(clean_diag)}, diag+susp-removed={len(clean_both)})")
print("Done.")
