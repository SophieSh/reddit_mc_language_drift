"""Overlaid GAM curves: non-BC vs BC (gap7 cohorts), Affective PC1 and Lexical PC1.
One panel per family, both cohorts on the same axes, so the BC attenuation is
visible at a glance.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)
from src.constants import PHASE_COLORS, PHASE_ORDER

TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
BCSRC = ROOT / "data" / "interim" / "bc_timeline_20260603T000200.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
OUT   = ROOT / "reports" / "gam_bc_gap7"; OUT.mkdir(parents=True, exist_ok=True)
AW, MAX_GAP, MIN_SPAN, MIN_POST, WINDOW = 28, 7, 28, 3, 28

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


def pc1_oriented(df, feats):
    feats = [c for c in feats if c in df.columns]; X = df[feats].copy()
    for c in feats:
        lo, hi = X[c].quantile([0.01, 0.99]); X[c] = X[c].clip(lo, hi)
    gg = df.groupby("author")
    for c in feats:
        mu = gg[c].transform("mean"); sd = gg[c].transform("std").replace(0, np.nan); X[c] = (X[c] - mu) / sd
    Xf = X.fillna(0.0).values
    comp = PCA(1).fit_transform(Xf)[:, 0]
    # deterministic orientation: positively correlated with the feature mean
    ref = np.nanmean(X.values, axis=1)
    if np.corrcoef(comp, np.nan_to_num(ref))[0, 1] < 0:
        comp = -comp
    return comp


def fit_grid(df, feats):
    df = df.copy()
    df["comp"] = pc1_oriented(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    s = df[["offset_from_cd1", "comp"]].dropna()
    gam = gam47.fit_gam(s[["offset_from_cd1"]].values, s["comp"].values, n_splines=10)
    XX = gam.generate_X_grid(term=0, n=300)
    return XX[:, 0], gam.predict(XX), gam.confidence_intervals(XX, width=0.95)


def phase_bands(ax, period=28, m=4, f=8, o=3):
    luteal = period - m - f - o
    layout = [("Menstrual", 0, m), ("Follicular", m, f), ("Ovulation", m + f, o), ("Luteal", m + f + o, luteal)]
    for shift in (-period, 0):
        for phase, start, length in layout:
            ax.axvspan(start + shift, start + shift + length, alpha=0.08, color=PHASE_COLORS[phase])
    ax.axvline(0, color="black", lw=1.0, ls="-", alpha=0.4)


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
bc_users = set(pd.read_csv(BCSRC, usecols=["author"]).author.unique())
elig = filter_chain(tot[["author", "offset_from_cd1"]])
cohorts = {"non-BC": elig - bc_users, "BC": elig & bc_users}
print(f"non-BC={len(cohorts['non-BC'])}  BC={len(cohorts['BC'])}")

for fname, feats in FAMFEATS.items():
    fig, ax = plt.subplots(figsize=(12, 4))
    phase_bands(ax)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    styles = {"non-BC": dict(color="C0", lw=2.5), "BC": dict(color="C3", lw=2.5, ls="--")}
    for label, users in cohorts.items():
        df = tot[tot.author.isin(users) & tot.offset_from_cd1.between(-WINDOW, WINDOW)]
        x, pr, ci = fit_grid(df, feats)
        ax.plot(x, pr, label=f"{label} (n={len(users)})", **styles[label])
        ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=0.12,
                        color=styles[label]["color"])
    ax.set_xlabel("Days from CD1 (anchor post)"); ax.set_ylabel("PC1 (deviation from user mean)")
    ax.set_title(f"GAM cycle curve: {fname} PC1 — non-BC vs BC (gap7)")
    handles = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.3, label=p) for p in PHASE_ORDER]
    handles += [plt.Line2D([0], [0], **styles["non-BC"], label="non-BC"),
                plt.Line2D([0], [0], **styles["BC"], label="BC")]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fn = OUT / f"gam_bc_vs_nonbc_{fname.replace(' ', '_')}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {fn}")
print("Done.")
