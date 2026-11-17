"""Replot the EBM (BC classifier) shape functions, GROUPED BY LINGUISTIC FAMILY.

Re-fits the committed symmetric ±91 baseline (AUC 0.645) exactly:
  BC   = data/interim/bc_timeline_20260603T000200.csv
  nonBC= data/interim/non_bc_timeline_20260603T000200.csv
  window ±91, phase-means, all-four-phases-present, top-50 Mann-Whitney pre-select,
  additive EBM (interactions=0), balanced class weights.

Only the original run's PNG was saved (no model), so we re-fit to recover the shapes.

Two things the replot fixes vs plot_shape_grid in script 42:
  1. Directionality kept as up = BC (log-odds toward BC up; red = BC, blue = non-BC),
     consistent across every panel.
  2. Panels are laid out as ONE ROW PER FAMILY (Affective / Lexical access / Syntactic /
     General / Pragmatic), showing that family's top-K features by EBM strength, so the
     linguistic families read separately instead of as a mixed top-N grid.

Usage: python scripts/52_ebm_shapes_by_family.py [--per-family 4] [--fselect 50]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ebm_utils import make_ebm, shape_functions, signed_importance, clean_name

BC_FILE = "data/interim/bc_timeline_20260603T000200.csv"
NONBC_FILE = "data/interim/non_bc_timeline_20260603T000200.csv"
FAMILY_XLSX = "data/raw/Attribute_families_cycles.xlsx"
PHASES = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
FAMILY_ORDER = ["Affective", "Lexical access", "Syntactic", "General", "Pragmatic"]
CLASS_NAMES = ["non-BC", "BC"]   # binary EBM: shape "BC" y = log-odds toward BC (up = BC)


def calculate_phase(offset, period=28, m=4, f=7, o=3):
    if pd.isna(offset):
        return np.nan
    day = int(offset) % period
    if day < m:         return "Menstrual"
    if day < m + f:     return "Follicular"
    if day < m + f + o: return "Ovulation"
    if day < period:    return "Luteal"
    return np.nan


def mannwhitney_top(X, y, names, n):
    """Top-n features by |rank-biserial| (Mann-Whitney), matching the pipeline pre-select."""
    from scipy.stats import mannwhitneyu
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        c = X[:, j]; ok = ~np.isnan(c)
        a, b = c[ok & (y == 1)], c[ok & (y == 0)]
        if len(a) < 5 or len(b) < 5:
            continue
        U, _ = mannwhitneyu(a, b)
        scores[j] = abs(2 * U / (len(a) * len(b)) - 1)
    keep = np.sort(np.argsort(scores)[::-1][:n])
    return X[:, keep], [names[i] for i in keep]


def feature_family(feat, attr2fam):
    """`<attribute>_mean_<Phase>` -> family via the attribute base name."""
    base = feat
    for ph in PHASES:
        if base.endswith("_" + ph):
            base = base[: -(len(ph) + 1)]
            break
    base = re.sub(r"_mean$", "", base)
    return attr2fam.get(base, "Other")


def feature_phase(feat):
    for ph in PHASES:
        if feat.endswith("_" + ph):
            return ph
    return ""


def _mask_sentinels(df, feats):
    """Drop feature-generation garbage before aggregation:
    - uint16 divide-by-zero sentinel (65535) that leaks into any feature -> NaN if |x|>=60000
    - concrete_abstract_ratio blows up on zero abstract words (sentinel averaged in);
      legit ratios are <~3 (p95=3.0), so values >100 are contamination -> NaN.
    """
    df = df.copy()
    present = [f for f in feats if f in df.columns]
    df[present] = df[present].mask(df[present].abs() >= 60000)
    rc = "concrete_abstract_ratio_mean"
    if rc in df.columns:
        df[rc] = df[rc].mask(df[rc] > 100)
    return df


def _prep_group(df, window, min_per_phase):
    """Window a group, phase-label it, keep authors with >= min_per_phase posting days
    (rows are daily-aggregated, so one row = one posting day) in EVERY phase."""
    df = df[df["offset_from_cd1"].between(-window, window)].copy()
    df["phase"] = df["offset_from_cd1"].apply(calculate_phase)
    df = df.dropna(subset=["phase"])
    cnt = df.groupby(["author", "phase"]).size().unstack("phase").reindex(columns=PHASES)
    keep = set(cnt[(cnt >= min_per_phase).all(axis=1)].index)
    return df[df["author"].isin(keep)]


def build_matrix(bc_window=91, nonbc_window=91, bc_min_per_phase=1, nonbc_min_per_phase=1,
                 max_zero_frac=None, bc_file=None, nonbc_file=None):
    """Per-user phase-means matrix. Windows/inclusion can differ by group; the
    control_w28_min2days design = BC(win91,min1), non-BC(win28,min2)."""
    dbc = pd.read_csv(ROOT / (bc_file or BC_FILE), encoding="utf-8-sig", low_memory=False); dbc["bc"] = 1
    dnb = pd.read_csv(ROOT / (nonbc_file or NONBC_FILE), encoding="utf-8-sig", low_memory=False); dnb["bc"] = 0
    non_feat = {"author", "offset_from_cd1", "phase", "bc", "pill_type", "bc_status"}
    feats = sorted((set(dbc.columns) & set(dnb.columns)) - non_feat)
    feats = [f for f in feats if not f.startswith("Unnamed:") and np.issubdtype(dbc[f].dtype, np.number)]
    dbc = _mask_sentinels(dbc, feats)
    dnb = _mask_sentinels(dnb, feats)
    if max_zero_frac is not None:
        pooled = pd.concat([dbc[feats], dnb[feats]], ignore_index=True)
        zfrac = (pooled == 0).mean()
        drop = zfrac[zfrac > max_zero_frac].index.tolist()
        feats = [f for f in feats if f not in drop]
        print(f"sparsity filter (>{max_zero_frac:.0%} zeros): dropped {len(drop)} degenerate "
              f"features -> {len(feats)} kept" + (f"  e.g. {drop[:4]}" if drop else ""))
    dbc = _prep_group(dbc, bc_window, bc_min_per_phase)
    dnb = _prep_group(dnb, nonbc_window, nonbc_min_per_phase)
    df = pd.concat([dbc, dnb], ignore_index=True)
    prof = df.groupby(["author", "phase"], dropna=True)[feats].mean()
    wide = prof.unstack("phase")
    wide.columns = [f"{c}_{ph}" for c, ph in wide.columns]
    labels = df.groupby("author")["bc"].first()
    y = labels.loc[wide.index].astype(int).values
    groups = np.asarray(wide.index)
    return wide, y, groups


def draw_shape(ax, d, title):
    # white background: the only colored fill is the data (red=BC above 0, blue=non-BC below)
    x, ylo = d["x"], d["y"]
    valid = ~np.isnan(x)
    xv, yv = x[valid], ylo[valid]
    # up = BC: fill above 0 red (BC), below 0 blue (non-BC)
    ax.fill_between(xv, yv, 0, where=yv >= 0, step="mid", color="#d9534f", alpha=0.35, linewidth=0)
    ax.fill_between(xv, yv, 0, where=yv < 0, step="mid", color="#4c8fd0", alpha=0.35, linewidth=0)
    ax.step(xv, yv, where="mid", color="black", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.margins(x=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-frac", type=float, default=None,
                    help="keep the top fraction by |signed importance| (e.g. 0.25 = top quartile)")
    ap.add_argument("--min-importance", type=float, default=None,
                    help="keep features with |signed importance| >= this; default = mean importance")
    ap.add_argument("--fselect", type=int, default=50, help="top-N Mann-Whitney pre-select (matches baseline)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bc-window", type=int, default=91)
    ap.add_argument("--nonbc-window", type=int, default=91)
    ap.add_argument("--bc-min-per-phase", type=int, default=1)
    ap.add_argument("--nonbc-min-per-phase", type=int, default=1)
    ap.add_argument("--max-zero-frac", type=float, default=None,
                    help="drop features that are 0 for more than this fraction of users (e.g. 0.9)")
    ap.add_argument("--cv", action="store_true", help="also report 5-fold GroupKFold AUC")
    ap.add_argument("--bc-file", default=None, help="override BC timeline path (rel to repo root)")
    ap.add_argument("--nonbc-file", default=None, help="override non-BC timeline path")
    ap.add_argument("--out", default="reports/ebm/bc/shapes_by_family_baseline_w91.png")
    args = ap.parse_args()

    attr2fam = dict(pd.read_excel(ROOT / FAMILY_XLSX).values)

    wide, y, groups = build_matrix(args.bc_window, args.nonbc_window,
                                   args.bc_min_per_phase, args.nonbc_min_per_phase,
                                   max_zero_frac=args.max_zero_frac,
                                   bc_file=args.bc_file, nonbc_file=args.nonbc_file)
    names = list(wide.columns)
    X = wide.values.astype(np.float64)
    print(f"design: BC(win{args.bc_window},min{args.bc_min_per_phase}) "
          f"nonBC(win{args.nonbc_window},min{args.nonbc_min_per_phase})")
    print(f"users={len(y)}  BC={int(y.sum())}  non-BC={int((y==0).sum())}  features={X.shape[1]}")

    Xsel, sel_names = mannwhitney_top(X, y, names, args.fselect)
    print(f"pre-selected {len(sel_names)} features")

    if args.cv:
        from sklearn.metrics import roc_auc_score
        from src.ebm_utils import run_cv
        _, proba, _ = run_cv(Xsel, y, groups, n_classes=2, seed=args.seed, n_splits=5)
        print(f"CV AUROC (5-fold GroupKFold): {roc_auc_score(y, proba[:,1]):.4f}")

    ebm = make_ebm(args.seed)
    from sklearn.utils.class_weight import compute_sample_weight
    ebm.fit(Xsel, y, sample_weight=compute_sample_weight("balanced", y))

    shapes = shape_functions(ebm, sel_names, CLASS_NAMES)   # per feat -> {"BC":{x,y,...}, "non-BC":...}
    # Rank features the SAME way the original grid / paper importance chart does:
    # signed importance (net density-weighted log-odds toward BC), not the direction-
    # agnostic weighted_sum. Positive sim -> pushes toward BC. This drops features whose
    # shape swings a lot but cancels out (e.g. hedging modal verbs · Follicular).
    sim = signed_importance(ebm, sel_names)
    rows = []
    for i, f in enumerate(sel_names):
        rows.append({"feat": f, "family": feature_family(f, attr2fam), "phase": feature_phase(f),
                     "signed_imp": float(sim[i]), "abs_imp": abs(float(sim[i]))})
    fdf = pd.DataFrame(rows)

    # Keep every feature whose |signed importance| clears a THRESHOLD, then group the
    # survivors into family rows — a principled cutoff, not a fixed count, so weak
    # families (General, Lexical access) simply contribute fewer / no panels.
    #   --top-frac F      -> keep the top F fraction (threshold = (1-F) quantile)
    #   --min-importance  -> explicit cutoff
    #   default           -> mean importance ("above-average features")
    if args.top_frac is not None:
        thr = float(fdf["abs_imp"].quantile(1 - args.top_frac))
        rule = f"top {args.top_frac:.0%}"
    elif args.min_importance is not None:
        thr = args.min_importance; rule = "user-set"
    else:
        thr = float(fdf["abs_imp"].mean()); rule = "mean"
    top = fdf[fdf["abs_imp"] >= thr].sort_values("abs_imp", ascending=False)
    print(f"importance threshold |imp| >= {thr:.4f}  ->  {len(top)} features kept ({rule})")
    per_fam = {fam: top[top.family == fam].sort_values("abs_imp", ascending=False)
               for fam in FAMILY_ORDER if (top.family == fam).any()}
    families = list(per_fam)                       # only families that made the cut
    ncols = max(len(v) for v in per_fam.values())  # ragged: rows left-aligned, blanks hidden
    nrows = len(families)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), squeeze=False)
    for r, fam in enumerate(families):
        sub = per_fam[fam]
        axes[r, 0].annotate(fam, xy=(-0.42, 0.5), xycoords="axes fraction", ha="center",
                            va="center", rotation=90, fontsize=13, fontweight="bold")
        for c in range(ncols):
            ax = axes[r, c]
            if c < len(sub):
                row = sub.iloc[c]
                d = shapes[row["feat"]]["BC"]
                base = re.sub(r"_mean_(%s)$" % "|".join(PHASES), "", row["feat"])
                title = f"{clean_name(base)}\n{row['phase']}  (imp {row['signed_imp']:+.3f})"
                draw_shape(ax, d, title)
                if c == 0:
                    ax.set_ylabel("log-odds", fontsize=8)
            else:
                ax.set_visible(False)
    # shared reading key (up = BC) as one figure-level note, not a per-panel title
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#d9534f", alpha=0.4, label="above 0 → BC"),
        plt.Rectangle((0, 0), 1, 1, color="#4c8fd0", alpha=0.4, label="below 0 → non-BC"),
    ], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.supxlabel("feature mean value in phase", fontsize=10, y=0.06)
    fig.subplots_adjust(left=0.09, bottom=0.09, hspace=0.55, wspace=0.28, top=0.965)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
