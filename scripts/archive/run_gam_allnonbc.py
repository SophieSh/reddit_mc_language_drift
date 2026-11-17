"""(1) GAM on ALL non-BC users (no gap7 filter) vs the gap7 cohort, and
(2) user-equal weighting (each user contributes equally, not by post count),
to see whether more/sparser users help and whether prolific posters drive it.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from pygam import LinearGAM, s

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gam47", ROOT / "scripts" / "47_gam_cycle_curve.py")
gam47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gam47)

TOTAL = ROOT / "data" / "interim" / "timeline_phase_labeled_fixed28_raw_no_anchors_20260604T174637.csv"
BCSRC = ROOT / "data" / "interim" / "bc_timeline_20260603T000200.csv"
FAM   = ROOT / "data" / "raw" / "Attribute_families_cycles.csv"
AW, MAX_GAP, MIN_SPAN, MIN_POST, WINDOW, NPERM = 28, 7, 28, 3, 28, 100
rng = np.random.default_rng(20260621)

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


def pc1(df, feats):
    feats = [c for c in feats if c in df.columns]; X = df[feats].copy()
    for c in feats:
        lo, hi = X[c].quantile([0.01, 0.99]); X[c] = X[c].clip(lo, hi)
    gg = df.groupby("author")
    for c in feats:
        mu = gg[c].transform("mean"); sd = gg[c].transform("std").replace(0, np.nan); X[c] = (X[c] - mu) / sd
    return PCA(1).fit_transform(X.fillna(0.0).values)[:, 0]


def repeat_from_pred(pr, x):
    p, c = pr[x < 0], pr[x >= 0]; n = min(len(p), len(c))
    return np.corrcoef(p[:n], c[:n])[0, 1]


def fit_repeat(df, feats, user_equal=False):
    df = df.copy(); df["comp"] = pc1(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    s_ = df[["offset_from_cd1", "comp", "author"]].dropna()
    w = None
    if user_equal:
        nrows = s_.groupby("author")["comp"].transform("size")
        w = (1.0 / nrows).values                          # each user sums to weight 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam = LinearGAM(s(0, n_splines=10, spline_order=3)).fit(
            s_[["offset_from_cd1"]].values, s_["comp"].values, weights=w)
    XX = gam.generate_X_grid(term=0, n=300)
    return repeat_from_pred(gam.predict(XX), XX[:, 0]), s_["author"].nunique(), len(s_)


def perm_p(df, feats, real):
    df = df.copy(); df["comp"] = pc1(df, feats)
    df["comp"] -= df.groupby("author")["comp"].transform("mean")
    s_ = df[["offset_from_cd1", "comp", "author"]].dropna()
    off, y, auth = s_.offset_from_cd1.values, s_.comp.values, s_.author.values
    order = np.argsort(auth, kind="stable"); a_s, off_s, y_s = auth[order], off[order], y[order]
    bnd = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1], True])
    null = np.empty(NPERM)
    for k in range(NPERM):
        op = off_s.copy()
        for b in range(len(bnd) - 1):
            i, j = bnd[b], bnd[b + 1]; op[i:j] = rng.permutation(op[i:j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = LinearGAM(s(0, n_splines=10, spline_order=3)).fit(op.reshape(-1, 1), y_s)
        XX = g.generate_X_grid(term=0, n=300); null[k] = repeat_from_pred(g.predict(XX), XX[:, 0])
    return (1 + np.sum(null >= real)) / (1 + NPERM)


print("Loading total timeline...")
tot = pd.read_csv(TOTAL, low_memory=False)
bc_users = set(pd.read_csv(BCSRC, usecols=["author"]).author.unique())
win = tot[tot.offset_from_cd1.between(-WINDOW, WINDOW)].copy()
g7 = filter_chain(tot[["author", "offset_from_cd1"]]) - bc_users
allnonbc = set(win.author.unique()) - bc_users

print(f"gap7 non-BC = {len(g7)}   all non-BC (±28) = {len(allnonbc)}\n")
print(f"{'family':16s} {'cohort':18s} {'users':>6s} {'rows':>7s} {'repeat':>8s} {'perm_p':>7s}")
print("-" * 70)
for fname, feats in FAMFEATS.items():
    for label, users, ue in [("gap7 (row-wt)", g7, False),
                             ("gap7 (USER-equal)", g7, True),
                             ("ALL non-BC", allnonbc, False)]:
        df = win[win.author.isin(users)]
        r, nu, nr = fit_repeat(df, feats, user_equal=ue)
        pv = perm_p(df, feats, r) if not ue else np.nan
        pvs = f"{pv:7.3f}" if not np.isnan(pv) else "   -  "
        print(f"{fname:16s} {label:18s} {nu:6d} {nr:7d} {r:+8.3f} {pvs}")
print("\nDone.")
