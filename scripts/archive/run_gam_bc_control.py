"""BC vs non-BC control: Affective-PC1 and Lexical-PC1 GAM on each cohort.

If the affective/lexical CD1-locked signal is genuinely hormonal-menstrual, it
should ATTENUATE in BC users (hormonal contraception flattens the natural cycle).
Same pipeline for both cohorts: winsorize -> per-user z-score -> family PC1 ->
per-user mean-center -> GAM vs offset_from_cd1 (+/-28, anchors excluded) ->
half-to-half repeat correlation + permutation p (shuffle cycle-day within user).
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)

FAM = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
INTERIM = ROOT / "data" / "interim"
COHORTS = {
    "non-BC": INTERIM / "non_bc_timeline_eligible_minposts18_span100_20260603T000200.csv",
    "BC":     INTERIM / "bc_timeline_eligible_minposts18_span100_20260603T000200.csv",
}
WINDOW = 28; NPERM = 200
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]]
            for f in ["Affective", "Lexical access"]}


def pc1(df, feats):
    feats = [c for c in feats if c in df.columns]
    X = df[feats].copy()
    for c in feats:                                   # winsorize 1/99 pct (ratio outliers)
        lo, hi = X[c].quantile([0.01, 0.99])
        X[c] = X[c].clip(lo, hi)
    gg = df.groupby("author")                          # per-user z-score
    for c in feats:
        mu = gg[c].transform("mean"); sd = gg[c].transform("std").replace(0, np.nan)
        X[c] = (X[c] - mu) / sd
    return pd.Series(PCA(n_components=1).fit_transform(X.fillna(0.0).values)[:, 0], index=df.index)


print(f"{'family':16s} {'cohort':8s} {'users':>6s} {'rows':>7s} {'repeat':>8s} {'perm_p':>7s}")
print("-" * 60)
rows = []
for fname, feats in FAMFEATS.items():
    for label, path in COHORTS.items():
        df = pd.read_csv(path)
        df = df[df["offset_from_cd1"].between(-WINDOW, WINDOW) & (df["offset_from_cd1"] != 0)].copy()
        df["comp"] = pc1(df, feats).values
        df["comp"] -= df.groupby("author")["comp"].transform("mean")
        d = df[["author", "offset_from_cd1", "comp"]].dropna()
        off = d["offset_from_cd1"].values; y = d["comp"].values; auth = d["author"].values

        def rc(o, v):
            sub = pd.DataFrame({"x": o, "y": v}).dropna()
            gam = gam47.fit_gam(sub[["x"]].values, sub["y"].values, n_splines=10)
            XX = gam.generate_X_grid(term=0, n=300); pr = gam.predict(XX); x = XX[:, 0]
            pp, cc = pr[x < 0], pr[x >= 0]; n = min(len(pp), len(cc))
            return np.corrcoef(pp[:n], cc[:n])[0, 1]
        real = rc(off, y)

        order = np.argsort(auth, kind="stable")
        a_s, off_s, y_s = auth[order], off[order], y[order]
        bnd = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1], True])
        null = np.empty(NPERM)
        for k in range(NPERM):
            op = off_s.copy()
            for b in range(len(bnd) - 1):
                i, j = bnd[b], bnd[b + 1]; op[i:j] = rng.permutation(op[i:j])
            null[k] = rc(op, y_s)
        pval = (1 + np.sum(null >= real)) / (1 + NPERM)
        print(f"{fname:16s} {label:8s} {d['author'].nunique():6d} {len(d):7d} {real:+8.3f} {pval:7.3f}")
        rows.append({"family": fname, "cohort": label, "users": d["author"].nunique(),
                     "repeat": real, "perm_p": pval})

pd.DataFrame(rows).to_csv(ROOT / "outputs" / "gam_bc_vs_nonbc.csv", index=False)
print("\nSaved: outputs/gam_bc_vs_nonbc.csv")
