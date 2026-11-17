"""Background-removed (prewhitened) view of the merged spectra.

Motivation: global argmax over the 21-40d grid is dominated by red-noise edges
(long-period rise + short-period jitter), so it hides a genuine LOCAL bump near
28 d. Here we fit a smooth background (low-order polynomial in frequency),
subtract it, and ask whether the 26-30 d band is a real local peak:
  - in-band residual peak (period + height)
  - RANK of that peak among all equal-width (5 d) sliding windows across the
    spectrum -> "is 28 the tallest local feature, or just one of many?"
Runs the same 3 non-BC cohorts and the same 3 levels (per-family, top-2, all-5),
for ALL users and TOP-50% cycling users. No permutations; fast.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
BCSRC = ROOT / "data" / "interim" / "bc_timeline_20260603T000200.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXOFF, MINP, MAXP, BLO, BHI = 90, 21, 40, 26, 30
BG_DEG = 3                       # polynomial degree for the smooth background
CONFIGS = [(14, 90), (14, 120), (10, 90)]

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMS = sorted(fam[fc].unique())
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in FAMS}

num_freq = 5 * (MAXP - MINP)
f_grid = np.linspace(1 / MAXP, 1 / MINP, num_freq); w = 2 * np.pi * f_grid; periods = 1 / f_grid
bandm = (periods >= BLO) & (periods <= BHI)
BAND_W = BHI - BLO               # 5-day window width for the sliding null


def filter_chain(daily, max_gap, min_span, aw=200):
    within = daily[daily.offset_from_cd1.between(-aw, aw)]; elig = []
    for user, grp in within.groupby("author"):
        d = sorted(grp.offset_from_cd1.unique())
        if not d: continue
        bef = [x for x in d if x <= 0]; aft = [x for x in d if x >= 0]
        st = bef[-1] if bef else aft[0]; si = d.index(st); r = st
        for i in range(si + 1, len(d)):
            if d[i] - d[i - 1] <= max_gap: r = d[i]
            else: break
        l = st
        for i in range(si - 1, -1, -1):
            if d[i + 1] - d[i] <= max_gap: l = d[i]
            else: break
        if l < 0 and r > 0 and (r - l) >= min_span: elig.append(user)
    return set(elig)


def background(ms):
    """Smooth red-noise background via low-order polynomial in frequency."""
    coef = np.polyfit(f_grid, ms, BG_DEG)
    return np.polyval(coef, f_grid)


def prominence(ms):
    """Return (residual, band_peak_period, rank_frac). rank_frac = fraction of
    all 5-day sliding windows whose residual max >= the 26-30 window's max
    (small = 28 stands out; ~0.5 = unremarkable)."""
    r = ms - background(ms)
    band_peak = r[bandm].max()
    pk_period = periods[bandm][int(np.argmax(r[bandm]))]
    # slide a BAND_W-wide window across the whole period range
    maxes = []
    lo = MINP
    while lo + BAND_W <= MAXP + 1e-9:
        m = (periods >= lo) & (periods <= lo + BAND_W)
        if m.any(): maxes.append(r[m].max())
        lo += 0.5
    maxes = np.array(maxes)
    rank_frac = (maxes >= band_peak).mean()
    return r, pk_period, band_peak, rank_frac


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
bc = set(pd.read_csv(BCSRC, usecols=["author"]).author.unique())

all_rows = []
for max_gap, min_span in CONFIGS:
    name = f"gap{max_gap}_span{min_span}_nonbc"
    out = ROOT / "reports" / f"spectral_{name}"; out.mkdir(parents=True, exist_ok=True)
    elig = filter_chain(tot[["author", "offset_from_cd1"]], max_gap, min_span) - bc
    df = tot[tot.author.isin(elig) & (tot.offset_from_cd1.abs() <= MAXOFF)].copy()
    nU = df.author.nunique()
    print(f"\n=== {name}: {nU} non-BC users, {len(df)} rows ===")

    allfeats = [c for fl in FAMFEATS.values() for c in fl if c in df.columns]
    g = df.groupby("author")
    for c in allfeats:
        mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0, np.nan)
        df[c] = (df[c] - mu) / sd
    for fname, feats in FAMFEATS.items():
        feats = [c for c in feats if c in df.columns]
        C = df[feats].corr().values; _, V = np.linalg.eigh(np.nan_to_num(C))
        df[f"FAM::{fname}"] = df[feats].fillna(0.0).values @ V[:, -1]

    users = sorted(df.author.unique()); uidx = {u: i for i, u in enumerate(users)}; nUu = len(users)

    def spectra(col):
        S = np.full((nUu, num_freq), np.nan)
        for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
            t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
            if len(x) < 4: continue
            x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
            S[uidx[u]] = np.abs(x @ np.exp(-2j * np.pi * np.outer(t, w))) / len(x)
        return S

    cube = np.stack([spectra(f"FAM::{f}") for f in FAMS], axis=0)
    imp = np.full((nUu, len(FAMS)), np.nan); unified = np.full((nUu, num_freq), np.nan); conf = np.full(nUu, np.nan)
    for u in range(nUu):
        M = cube[:, u, :]
        if np.isnan(M).any() or (M != 0).any(axis=1).sum() < 2: continue
        M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        imp[u] = np.abs(U[:, 0]); unified[u] = np.abs(Vt[0]); conf[u] = S[0] ** 2 / (S ** 2).sum()
    valid = ~np.isnan(conf)
    mean_imp = np.nanmean(imp[valid] / np.nanmax(imp[valid], axis=1, keepdims=True), axis=0)
    top = np.argsort(mean_imp)[-2:]; top_names = ", ".join(FAMS[i] for i in top)
    thr = np.nanmedian(conf[valid]); topmask = valid & (conf > thr)

    def resvd_unified(idx, mask):
        ns = np.full((nUu, num_freq), np.nan); cf = np.full(nUu, np.nan)
        for u in range(nUu):
            M = cube[idx][:, u, :].copy()
            if np.isnan(M).any(): continue
            M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
            _, S, Vt = np.linalg.svd(M, full_matrices=False)
            ns[u] = np.abs(Vt[0]); cf[u] = S[0] ** 2 / (S ** 2).sum()
        v = ~np.isnan(cf)
        sel = v if mask == "all" else (v & (cf > np.nanmedian(cf[v])))
        return np.nanmean(ns[sel], axis=0), int(sel.sum())

    # assemble the levels x selections we want to score/plot
    panels = {}   # label -> (mean_spectrum, n)
    for i, f in enumerate(FAMS):
        panels[(f, "all")]   = (np.nanmean(cube[i][valid], axis=0), int(valid.sum()))
        panels[(f, "top50")] = (np.nanmean(cube[i][topmask], axis=0), int(topmask.sum()))
    panels[("all_5", "all")]   = (np.nanmean(unified[valid], axis=0), int(valid.sum()))
    panels[("all_5", "top50")] = (np.nanmean(unified[topmask], axis=0), int(topmask.sum()))
    panels[(f"top2:{top_names}", "all")]   = resvd_unified(top, "all")
    panels[(f"top2:{top_names}", "top50")] = resvd_unified(top, "top50")

    for (level, sel), (ms, n) in panels.items():
        raw_pk = periods[int(np.argmax(ms))]
        _, band_pk, height, rank = prominence(ms)
        all_rows.append({"cohort": name, "level": level, "selection": sel, "n": n,
                         "raw_argmax": round(raw_pk, 2), "in_band_peak": round(band_pk, 2),
                         "prom_height": round(float(height), 5), "rank_frac": round(float(rank), 3),
                         "band_is_top": rank <= 1.0 / max(1, round((MAXP - MINP - BAND_W) / 0.5))})
        print(f"  {level:22.22s} {sel:5s} n={n:4d}  raw={raw_pk:5.1f}  in-band={band_pk:5.1f}  rank={rank:.2f}")

    # figure: background-removed (residual) spectra for the two merges, both selections
    figm, axm = plt.subplots(2, 2, figsize=(12, 8))
    merge_panels = [(f"top2:{top_names}", "all"), (f"top2:{top_names}", "top50"),
                    ("all_5", "all"), ("all_5", "top50")]
    for ax, key in zip(axm.flat, merge_panels):
        ms, n = panels[key]; r, band_pk, height, rank = prominence(ms)
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvspan(BLO, BHI, color="#cfe8ff", alpha=0.7)
        ax.plot(periods, r, color="#1f77b4", lw=2)
        ax.axvline(band_pk, color="#d62728", ls="--", lw=1.4,
                   label=f"in-band peak {band_pk:.1f} d\nrank {rank:.0%} of windows")
        ax.set_title(f"{key[0]} — {key[1]} (n={n})", fontsize=9.5)
        ax.set_xlim(MINP, MAXP); ax.set_xlabel("Days per cycle"); ax.set_ylabel("residual power"); ax.legend(fontsize=8)
    figm.suptitle(f"Background-removed (prewhitened) merged spectra — {name}", fontsize=11)
    figm.tight_layout(); figm.savefig(out / "merged_spectra_prewhitened.png", dpi=150, bbox_inches="tight"); plt.close(figm)
    print(f"  saved {out}/merged_spectra_prewhitened.png")

summ = pd.DataFrame(all_rows)
summ.to_csv(ROOT / "outputs" / "spectral_prominence_summary.csv", index=False)
print("\nSaved outputs/spectral_prominence_summary.csv")
print(summ.to_string(index=False))
print("\nDone.")
