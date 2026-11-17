"""Mean NUFFT spectrum plots for the gap10/span>=120 cohort, ALL users (no
top-50% selection). One figure for the 5 family composites, one small-multiples
figure for all 69 features. Cycle band 26-30 d shaded, peak marked.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
OUT   = ROOT / "reports" / "freq_gap10_120"; OUT.mkdir(parents=True, exist_ok=True)
MAXGAP, MINSPAN, AW, MAXOFF = 10, 120, 200, 90
MINPERIOD, MAXPERIOD, BLO, BHI = 21, 40, 26, 30

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
        if not (left < 0 and right > 0): continue
        if (right - left) >= MINSPAN: elig.append(user)
    return set(elig)


print("Loading...")
tot = pd.read_csv(TOTAL, low_memory=False)
elig = filter_chain(tot[["author", "offset_from_cd1"]])
df = tot[tot.author.isin(elig) & (tot.offset_from_cd1.abs() <= MAXOFF)].copy()
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

num_freq = 5 * (MAXPERIOD - MINPERIOD)
f_grid = np.linspace(1 / MAXPERIOD, 1 / MINPERIOD, num_freq)
w_grid = 2 * np.pi * f_grid; periods = 1 / f_grid


def mean_spectrum(col):
    specs = []
    for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
        t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
        if len(x) < 4: continue
        x = x - x.mean()
        x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        E = np.exp(-2j * np.pi * np.outer(t, w_grid))
        specs.append(np.abs(x @ E) / len(x))
    if not specs:
        return np.full(num_freq, np.nan)
    return np.mean(specs, axis=0)


def panel(ax, ms, title):
    ax.axvspan(BLO, BHI, color="#cfe8ff", alpha=0.7, zorder=0)
    if not np.all(np.isfinite(ms)):
        ax.set_title(f"{title}\n(no data)", fontsize=8)
        ax.set_xlim(MINPERIOD, MAXPERIOD); return
    ax.plot(periods, ms, color="#1f77b4", lw=1.8)
    pk = periods[int(np.argmax(ms))]
    ax.axvline(pk, color="#d62728", ls="--", lw=1.2)
    ax.set_title(f"{title}\npeak {pk:.1f} d", fontsize=8)
    ax.set_xlim(MINPERIOD, MAXPERIOD); ax.tick_params(labelsize=6)


# ---- families ----
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, f in zip(axes.flat, FAMS):
    panel(ax, mean_spectrum(f"FAM::{f}"), f)
axes.flat[-1].axis("off")
fig.suptitle(f"NUFFT family spectra — gap10/span≥120 cohort, ALL {df.author.nunique()} users", fontsize=11)
fig.supxlabel("Days per cycle", fontsize=9); fig.supylabel("Mean spectral magnitude", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "family_spectra_all.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved family_spectra_all.png")

# ---- all features (small multiples, grouped by family) ----
ordered = [(f, c) for f in FAMS for c in FAMFEATS[f] if c in df.columns]
n = len(ordered); ncol = 7; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 1.6))
for ax, (f, c) in zip(axes.flat, ordered):
    panel(ax, mean_spectrum(c), c.replace("_mean", "")[:18])
for ax in axes.flat[n:]:
    ax.axis("off")
fig.suptitle(f"NUFFT feature spectra — gap10/span≥120, ALL users (band 26–30 d shaded)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "feature_spectra_all.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("Saved feature_spectra_all.png")
print(f"Outputs in {OUT}")
