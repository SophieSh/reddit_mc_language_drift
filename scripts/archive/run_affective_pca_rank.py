"""Affective family PCA: rank features by PC1 loading, then backward-eliminate
the weakest one at a time and watch how PC1 variance% changes.

Matches the pipeline's family-PCA convention: per-user z-score each feature,
build the (pairwise) correlation matrix, eig, PC1 var% = lambda_max / sum(lambda).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "eligible_users_timeline_minposts20_span100_no_anchors_20260517T134609.csv"
FAM  = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
TARGET = "Affective"

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
feats = [a + "_mean" for a in fam.loc[fam[fc] == TARGET, ac]]

df = pd.read_csv(DATA)
feats = [c for c in feats if c in df.columns]

# per-user z-score each feature
g = df.groupby("author")
for c in feats:
    mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
    df[c] = (df[c] - mu) / sd


def pca_var(cols):
    """PC1 variance% and loadings from the pairwise correlation matrix."""
    C = df[cols].corr().values            # pairwise Pearson, NaN-safe
    w, V = np.linalg.eigh(C)              # ascending eigenvalues
    var_pct = w[-1] / w.sum() * 100.0
    loadings = np.abs(V[:, -1])           # |PC1 loading| per feature
    return var_pct, dict(zip(cols, loadings))


print(f"{TARGET} family: {len(feats)} features\n")
cols = list(feats)
print(f"{'k':>2}  {'PC1 var%':>9}  {'dropped (weakest)':28}  loadings (kept, sorted)")
print("-" * 100)
while len(cols) >= 2:
    var_pct, load = pca_var(cols)
    ranked = sorted(load.items(), key=lambda kv: kv[1], reverse=True)
    weakest = ranked[-1][0]
    load_str = ", ".join(f"{k.replace('_mean','')}={v:.2f}" for k, v in ranked)
    drop_show = weakest.replace("_mean", "") if len(cols) > 2 else "-"
    print(f"{len(cols):>2}  {var_pct:8.1f}%  {drop_show:28}  {load_str}")
    cols.remove(weakest)
