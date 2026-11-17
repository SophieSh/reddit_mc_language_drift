"""GAM per individual Affective feature on the gap7 dataset.

For each Affective feature: per-user mean-center, fit the script-47 GAM vs
offset_from_cd1, and report the half-to-half repeat correlation, the curve
amplitude, and a permutation p (shuffle cycle-day within user, refit, recompute
repeat corr). Resolves whether the Affective CD1-locked signal lives in the
sentiment features or the dictionary-coverage features.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)

GAP7 = ROOT / "data" / "interim" / "eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125.csv"
FAM  = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
OUT  = ROOT / "reports" / "gam_affective_features_gap7"; OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 28; NPERM = 200
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
feats = [a + "_mean" for a in fam.loc[fam[fc] == "Affective", ac]]

df = pd.read_csv(GAP7)
feats = [c for c in feats if c in df.columns]
df = df[df["offset_from_cd1"].between(-WINDOW, WINDOW)].copy()


def repeat_corr_of(off, y):
    sub = pd.DataFrame({"x": off, "y": y}).dropna()
    if sub["y"].std() < 1e-12 or len(sub) < 50:
        return np.nan, np.nan
    gam = gam47.fit_gam(sub[["x"]].values, sub["y"].values, n_splines=10)
    XX = gam.generate_X_grid(term=0, n=300); pr = gam.predict(XX); x = XX[:, 0]
    prev, curr = pr[x < 0], pr[x >= 0]; n = min(len(prev), len(curr))
    rc = np.corrcoef(prev[:n], curr[:n])[0, 1]
    return rc, float(np.std(pr)), gam


print(f"{'feature':28s} {'repeat':>7s} {'amp':>7s} {'perm_p':>7s}")
print("-" * 56)
rows = []
for c in feats:
    d = df[["author", "offset_from_cd1", c]].copy()
    d[c] -= d.groupby("author")[c].transform("mean")     # per-user center
    off = d["offset_from_cd1"].values; y = d[c].values
    res = repeat_corr_of(off, y)
    if not isinstance(res, tuple) or len(res) < 3 or np.isnan(res[0]):
        print(f"{c.replace('_mean',''):28s}     skipped"); continue
    rc, amp, gam = res
    gam47.plot_curve(gam, c, OUT, period=28)

    # permutation: shuffle offset within user, refit, repeat corr
    null = np.empty(NPERM)
    auth = d["author"].values
    order = np.argsort(auth, kind="stable")
    a_sorted = auth[order]; off_sorted = off[order]; y_sorted = y[order]
    bounds = np.flatnonzero(np.r_[True, a_sorted[1:] != a_sorted[:-1], True])
    for p in range(NPERM):
        off_perm = off_sorted.copy()
        for b in range(len(bounds) - 1):
            i, j = bounds[b], bounds[b + 1]
            off_perm[i:j] = rng.permutation(off_perm[i:j])
        sub = pd.DataFrame({"x": off_perm, "y": y_sorted}).dropna()
        gpn = gam47.fit_gam(sub[["x"]].values, sub["y"].values, n_splines=10)
        XX = gpn.generate_X_grid(term=0, n=300); pr = gpn.predict(XX); xx = XX[:, 0]
        pv, cu = pr[xx < 0], pr[xx >= 0]; nn = min(len(pv), len(cu))
        null[p] = np.corrcoef(pv[:nn], cu[:nn])[0, 1]
    perm_p = (1 + np.sum(null >= rc)) / (1 + NPERM)
    print(f"{c.replace('_mean',''):28s} {rc:+7.3f} {amp:7.4f} {perm_p:7.3f}")
    rows.append({"feature": c.replace("_mean", ""), "repeat_corr": rc,
                 "amplitude": amp, "perm_p": perm_p})

pd.DataFrame(rows).to_csv(OUT / "affective_feature_gam.csv", index=False)
print(f"\nSaved table + curves to {OUT}")
