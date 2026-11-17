"""GAM cycle curve for the Lexical-access AVERAGE composite (not PC1).

Builds the composite the same way the Octave merge did: per-user z-score each
Lexical feature, then average across features. Then per-user mean-center and fit
the same GAM (vs offset_from_cd1) used in scripts/47, on the gap7 dataset.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# reuse the plotting / GAM fitting from script 47 (module name starts with a digit)
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gam47)

GAP7 = ROOT / "data" / "interim" / "eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125.csv"
FAM  = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
OUT  = ROOT / "reports" / "gam_lexical_avg_gap7"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 28

# --- lexical feature list ---------------------------------------------------
fam = pd.read_csv(FAM)
fam.columns = [c.strip().lower() for c in fam.columns]
attr_col = "attribute" if "attribute" in fam.columns else fam.columns[0]
famcol = "family" if "family" in fam.columns else fam.columns[1]
lex = [a + "_mean" for a in fam.loc[fam[famcol] == "Lexical access", attr_col]]

df = pd.read_csv(GAP7)
lex = [c for c in lex if c in df.columns]
print(f"Lexical features used ({len(lex)}): {[c.replace('_mean','') for c in lex]}")

df = df[df["offset_from_cd1"].between(-WINDOW, WINDOW)].copy()

# --- per-user z-score each feature, then average ---------------------------
g = df.groupby("author")
for c in lex:
    mu = g[c].transform("mean")
    sd = g[c].transform("std").replace(0, np.nan)
    df[c] = (df[c] - mu) / sd
df["lexical_average_mean"] = df[lex].mean(axis=1, skipna=True)

# --- per-user mean-center the composite (as script 47 does) ----------------
df["lexical_average_mean"] -= df.groupby("author")["lexical_average_mean"].transform("mean")

sub = df[["offset_from_cd1", "lexical_average_mean"]].dropna()
sub = sub.rename(columns={"offset_from_cd1": "cycle_day_norm"})
print(f"Rows: {len(sub)}  users: {df['author'].nunique()}")

gam = gam47.fit_gam(sub[["cycle_day_norm"]].values, sub["lexical_average_mean"].values, n_splines=10)
gam47.plot_curve(gam, "lexical_average_mean", OUT, period=28)

# --- half-to-half repeat correlation: does [-28,0] match [0,28]? -----------
XX = gam.generate_X_grid(term=0, n=300)
preds = gam.predict(XX)
x = XX[:, 0]
prev = preds[x < 0]
curr = preds[x >= 0]
n = min(len(prev), len(curr))
# align previous-cycle day d-28 with current-cycle day d
rep = np.corrcoef(prev[-n:], curr[:n])[0, 1]
print(f"Half-to-half repeat correlation: {rep:+.3f}")
print(f"Saved curve to {OUT}/lexical_average.png")
