"""Per-family spectral analysis + permutation null on assumption-free cohorts
(connected chain crossing CD1, gap/span on RAW offset — no phase assumption).
NON-BC users only. Runs several (gap, span) configs, saves plots + summary per
cohort into reports/spectral_<name>/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
BCSRC = ROOT / "data" / "interim" / "bc_timeline_20260603T000200.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
MAXOFF, MINP, MAXP, NPERM, BLO, BHI = 90, 21, 40, 500, 26, 30
CONFIGS = [(14, 90), (14, 120), (10, 90)]     # (max_gap, min_span)
rng = np.random.default_rng(20260621)

fam = pd.read_csv(FAM); fam.columns = [c.strip().lower() for c in fam.columns]
ac = "attribute" if "attribute" in fam.columns else fam.columns[0]
fc = "family" if "family" in fam.columns else fam.columns[1]
FAMS = sorted(fam[fc].unique())
FAMFEATS = {f: [a + "_mean" for a in fam.loc[fam[fc] == f, ac]] for f in FAMS}

num_freq = 5 * (MAXP - MINP)
f_grid = np.linspace(1 / MAXP, 1 / MINP, num_freq); w = 2 * np.pi * f_grid; periods = 1 / f_grid
bandm = (periods >= BLO) & (periods <= BHI)


def filter_chain(daily, max_gap, min_span, aw=200):
    within = daily[daily.offset_from_cd1.between(-aw, aw)]; elig = []
    for user, grp in within.groupby("author"):
        d = sorted(grp.offset_from_cd1.unique())
        if not d: continue
        bef = [x for x in d if x <= 0]; aft = [x for x in d if x >= 0]
        st = bef[-1] if bef else aft[0]; si = d.index(st)
        r = st
        for i in range(si + 1, len(d)):
            if d[i] - d[i - 1] <= max_gap: r = d[i]
            else: break
        l = st
        for i in range(si - 1, -1, -1):
            if d[i + 1] - d[i] <= max_gap: l = d[i]
            else: break
        if l < 0 and r > 0 and (r - l) >= min_span: elig.append(user)
    return set(elig)


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
bc = set(pd.read_csv(BCSRC, usecols=["author"]).author.unique())


def run_cohort(max_gap, min_span):
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

    rows = []; spectra_store = {}
    for fname in FAMS:
        col = f"FAM::{fname}"
        real_sum = np.zeros(num_freq); null_sum = np.zeros((NPERM, num_freq)); nC = 0
        for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
            t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
            if len(x) < 4: continue
            x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
            E = np.exp(-2j * np.pi * np.outer(t, w))
            real_sum += np.abs(x @ E) / len(x)
            XP = np.array([x[rng.permutation(len(x))] for _ in range(NPERM)])
            null_sum += np.abs(XP @ E) / len(x); nC += 1
        ms = real_sum / nC; null_ms = null_sum / nC
        spectra_store[fname] = (ms, null_ms.mean(0))
        pk = periods[int(np.argmax(ms))]
        def stat(s): return s[bandm].max() / s.mean()
        rs = stat(ms); ns = np.array([stat(null_ms[p]) for p in range(NPERM)])
        pval = (1 + (ns >= rs).sum()) / (1 + NPERM)
        rows.append({"family": fname, "peak_days": round(pk, 2), "in_band": BLO <= pk <= BHI,
                     "real_stat": round(rs, 4), "null_stat": round(ns.mean(), 4), "p_value": round(pval, 4)})
        print(f"  {fname:16s} peak={pk:5.1f}  p={pval:.3f}")
    summ = pd.DataFrame(rows); summ.to_csv(out / "family_null_summary.csv", index=False)

    # ---- SVD merges: top-2 and all-5 -------------------------------------
    # per-user family x freq spectra cube, then supervisor pipeline:
    # per-user SVD across families -> importance -> top-2 re-SVD; all-5 unified.
    users = sorted(df.author.unique()); uidx = {u: i for i, u in enumerate(users)}; nUu = len(users)

    def spectra(col):
        S = np.full((nUu, num_freq), np.nan)
        for u, grp in df[["author", "offset_from_cd1", col]].dropna().groupby("author"):
            t = grp.offset_from_cd1.values.astype(float); x = grp[col].values.astype(float)
            if len(x) < 4: continue
            x = x - x.mean(); x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
            S[uidx[u]] = np.abs(x @ np.exp(-2j * np.pi * np.outer(t, w))) / len(x)
        return S

    cube = np.stack([spectra(f"FAM::{f}") for f in FAMS], axis=0)   # [nFam x nU x nf]
    imp = np.full((nUu, len(FAMS)), np.nan); unified = np.full((nUu, num_freq), np.nan); conf = np.full(nUu, np.nan)
    for u in range(nUu):
        M = cube[:, u, :]
        if np.isnan(M).any() or (M != 0).any(axis=1).sum() < 2: continue
        M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        imp[u] = np.abs(U[:, 0]); unified[u] = np.abs(Vt[0]); conf[u] = S[0] ** 2 / (S ** 2).sum()
    valid = ~np.isnan(conf)
    mean_imp = np.nanmean(imp[valid] / np.nanmax(imp[valid], axis=1, keepdims=True), axis=0)
    order = np.argsort(mean_imp)
    top = order[-2:]

    def resvd(idx):
        """Return unified spectrum for a family subset, both over ALL valid users
        and over the TOP-50% cycling-confidence users."""
        ns = np.full((nUu, num_freq), np.nan); cf = np.full(nUu, np.nan)
        for u in range(nUu):
            M = cube[idx][:, u, :].copy()
            if np.isnan(M).any(): continue
            M = M - M.mean(axis=1, keepdims=True); rs = M.std(axis=1, ddof=1); rs[rs == 0] = 1; M = M / rs[:, None]
            _, S, Vt = np.linalg.svd(M, full_matrices=False)
            ns[u] = np.abs(Vt[0]); cf[u] = S[0] ** 2 / (S ** 2).sum()
        v = ~np.isnan(cf); m = v & (cf > np.nanmedian(cf[v]))
        ms_v = np.nanmean(ns[v], axis=0); ms_m = np.nanmean(ns[m], axis=0)
        return (periods[int(np.argmax(ms_v))], ms_v, int(v.sum()),
                periods[int(np.argmax(ms_m))], ms_m, int(m.sum()))

    thr = np.nanmedian(conf[valid]); mask = valid & (conf > thr)
    ms_all_v = np.nanmean(unified[valid], axis=0); pk_all_v = periods[int(np.argmax(ms_all_v))]
    ms_all_m = np.nanmean(unified[mask], axis=0); pk_all_m = periods[int(np.argmax(ms_all_m))]
    pk_top_v, ms_top_v, ntop_v, pk_top_m, ms_top_m, ntop_m = resvd(top)
    top_names = ", ".join(FAMS[i] for i in top)
    merge_rows = [
        {"level": "all_5", "families": ", ".join(FAMS),
         "peak_all_users": round(pk_all_v, 2), "n_all": int(valid.sum()),
         "peak_top50": round(pk_all_m, 2), "n_top50": int(mask.sum())},
        {"level": "top_2", "families": top_names,
         "peak_all_users": round(pk_top_v, 2), "n_all": ntop_v,
         "peak_top50": round(pk_top_m, 2), "n_top50": ntop_m},
    ]
    pd.DataFrame(merge_rows).to_csv(out / "merge_summary.csv", index=False)
    print(f"  importance (low->high): {', '.join(f'{FAMS[i]}={mean_imp[i]:.3f}' for i in order)}")
    print(f"  all-5  peak: all-users={pk_all_v:.1f} (n={valid.sum()})  top50={pk_all_m:.1f} (n={mask.sum()})")
    print(f"  top-2 [{top_names}] peak: all-users={pk_top_v:.1f} (n={ntop_v})  top50={pk_top_m:.1f} (n={ntop_m})")

    # merged-spectra plot: rows = {all-5, top-2}, cols = {all users, top-50%}
    figm, axm = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (ms_all_v, pk_all_v, f"All 5 families — ALL users (n={valid.sum()})"),
        (ms_all_m, pk_all_m, f"All 5 families — top-50% (n={mask.sum()})"),
        (ms_top_v, pk_top_v, f"Top-2: {top_names} — ALL users (n={ntop_v})"),
        (ms_top_m, pk_top_m, f"Top-2: {top_names} — top-50% (n={ntop_m})")]
    for ax, (ms_, pk_, ttl) in zip(axm.flat, panels):
        ax.axvspan(BLO, BHI, color="#cfe8ff", alpha=0.7)
        ax.plot(periods, ms_, color="#1f77b4", lw=2)
        ax.axvline(pk_, color="#d62728", ls="--", lw=1.4, label=f"peak {pk_:.1f} d")
        ax.set_title(ttl, fontsize=9.5); ax.set_xlim(MINP, MAXP); ax.legend(fontsize=8)
        ax.set_xlabel("Days per cycle")
    figm.suptitle(f"SVD-merged unified spectra — {name} ({nU} users): all users vs top-50%", fontsize=11)
    figm.tight_layout(); figm.savefig(out / "merged_spectra.png", dpi=150, bbox_inches="tight"); plt.close(figm)
    print(f"  saved {out}/merged_spectra.png  +  merge_summary.csv")

    # per-family plot grid (real vs null-mean)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, fname in zip(axes.flat, FAMS):
        ms, nm = spectra_store[fname]; pk = periods[int(np.argmax(ms))]
        r = summ[summ.family == fname].iloc[0]
        ax.axvspan(BLO, BHI, color="#cfe8ff", alpha=0.7)
        ax.plot(periods, ms, color="#1f77b4", lw=1.8, label="real")
        ax.plot(periods, nm, color="k", lw=0.9, ls="--", label="null mean")
        ax.axvline(pk, color="#d62728", ls="--", lw=1.1)
        ax.set_title(f"{fname} — peak {pk:.1f} d, p={r.p_value}", fontsize=9); ax.set_xlim(MINP, MAXP)
    axes.flat[0].legend(fontsize=7); axes.flat[-1].axis("off")
    fig.suptitle(f"Per-family NUFFT spectra + null — {name} ({nU} users)", fontsize=11)
    fig.supxlabel("Days per cycle"); fig.tight_layout()
    fig.savefig(out / "per_family_spectra.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}/per_family_spectra.png  +  family_null_summary.csv")


for mg, ms in CONFIGS:
    run_cohort(mg, ms)
print("\nDone.")
