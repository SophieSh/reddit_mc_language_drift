"""Lexical GAM: compare plain mean vs signed (oriented) mean vs PC1.

All three composites are built from the per-user z-scored Lexical features and
run through the SAME GAM + half-to-half repeat-correlation code, so the numbers
are directly comparable (resolves whether PC1's reported +0.79 is real or a
different metric).
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

GAP7 = ROOT / "data" / "interim" / "eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125.csv"
FAM  = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
OUT  = ROOT / "reports" / "gam_lexical_composites_gap7"; OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 28

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
lex = [a + "_mean" for a in fam.loc[fam[fc] == "Lexical access", ac]]

df = pd.read_csv(GAP7)
lex = [c for c in lex if c in df.columns]
df = df[df["offset_from_cd1"].between(-WINDOW, WINDOW)].copy()

# per-user z-score each feature
g = df.groupby("author")
for c in lex:
    mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
    df[c] = (df[c] - mu) / sd

plain = df[lex].mean(axis=1, skipna=True)
# signed mean: flip each feature to positively correlate with the plain mean
signs = {c: (1.0 if df[c].corr(plain) >= 0 else -1.0) for c in lex}
signed = pd.concat([df[c] * signs[c] for c in lex], axis=1).mean(axis=1, skipna=True)
# PC1 on mean-imputed z-scored features
X = df[lex].fillna(0.0).values
pc1 = pd.Series(PCA(n_components=1).fit_transform(X)[:, 0], index=df.index)

def repeat_corr(gam):
    XX = gam.generate_X_grid(term=0, n=300); pr = gam.predict(XX); x = XX[:, 0]
    prev, curr = pr[x < 0], pr[x >= 0]; n = min(len(prev), len(curr))
    return np.corrcoef(prev[:n], curr[:n])[0, 1], float(np.std(pr))

print(f"{'composite':14s} {'repeat_corr':>12s} {'amplitude':>10s}")
for name, comp in [("plain_mean", plain), ("signed_mean", signed), ("pc1", pc1)]:
    d = df[["offset_from_cd1"]].copy(); d["y"] = comp.values
    d["y"] -= d.groupby(df["author"])["y"].transform("mean")
    sub = d.dropna()
    gam = gam47.fit_gam(sub[["offset_from_cd1"]].values, sub["y"].values, n_splines=10)
    rc, amp = repeat_corr(gam)
    gam47.plot_curve(gam, f"lexical_{name}_mean", OUT, period=28)
    print(f"{name:14s} {rc:+12.3f} {amp:10.4f}")
print(f"\nCurves saved to {OUT}")
