"""Trait-level PCOS classifier: NO phases. Each user = the average of every
feature across ALL their posts. Diagnosed-PCOS vs non-PCOS (users who never
mentioned PCOS), among dense posters (>=20 posts). EBM, 5-fold stratified CV.
"""
from __future__ import annotations
import sys, warnings, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from interpret.glassbox import ExplainableBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.ebm_utils import shape_functions, clean_name


def plot_shape_grid(shapes, output_path, target_class, base_class, top_n=10):
    feats = list(shapes.keys())[:top_n]; n_feat = len(feats); ncols = 2
    nrows = (n_feat + ncols - 1) // ncols
    c_t, c_b = "#d62728", "#1f77b4"
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]; d = shapes[feat][target_class]; x, y = d["x"], d["y"]
        valid = ~np.isnan(x); xv, yv = x[valid], y[valid]
        for j in range(len(xv) - 1):
            ax.axvspan(xv[j], xv[j + 1], alpha=0.12, linewidth=0,
                       color=c_t if yv[j] >= 0 else c_b)
        ax.step(xv, yv, where="mid", color="black", linewidth=1.8)
        ax.fill_between(xv, yv, 0, step="mid", where=yv >= 0, alpha=0.35, color=c_t, label=target_class)
        ax.fill_between(xv, yv, 0, step="mid", where=yv < 0, alpha=0.35, color=c_b, label=base_class)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"{clean_name(feat)}\nstrength={d['weighted_sum']:.3f}", fontsize=8, fontweight="bold")
        ax.set_xlabel("mean value (all posts)", fontsize=8); ax.set_ylabel("Log-odds", fontsize=8)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    for j in range(n_feat, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"EBM Shape Functions — {base_class} vs {target_class}\n"
                 f"Above 0 → {target_class}  |  Below 0 → {base_class}", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)


MOON = [ROOT / "data" / "raw" / "moon1_all_post_2015_2025_filtered_features.csv",
        ROOT / "data" / "raw" / "moon2_all_post_2015_2025_filtered_features.csv"]
META = {"author", "author_flair_text", "created_utc", "permalink", "selftext",
        "subreddit", "title"}
MINPOSTS, FSEL = 20, 50

diag = set(pd.read_csv(ROOT / "outputs" / "pcos_diagnosed_users.csv").author)
mentioners = set(pd.read_csv(ROOT / "outputs" / "pcos_users_aggregated.csv", usecols=["author"]).author)

# feature columns
cols0 = list(pd.read_csv(MOON[0], nrows=0).columns)
feats = [c for c in cols0 if c not in META]

PCOS_PAT = re.compile(r"\bpcos\b|polycystic ovar", re.I)
print("Aggregating per-user averages across all posts (DROPPING PCOS-mentioning posts)...")
ssum = None; scnt = None; nposts = None
for f in MOON:
    for ch in pd.read_csv(f, usecols=["author", "title", "selftext"] + feats, chunksize=200000, low_memory=False):
        txt = ch["title"].fillna("") + " " + ch["selftext"].fillna("")
        ch = ch[~txt.str.contains(PCOS_PAT)]                       # exclude the PCOS posts themselves
        ch[feats] = ch[feats].apply(pd.to_numeric, errors="coerce")
        g = ch.groupby("author")
        s = g[feats].sum(min_count=1); c = g[feats].count(); n = g.size().rename("n")
        ssum = s if ssum is None else ssum.add(s, fill_value=0)
        scnt = c if scnt is None else scnt.add(c, fill_value=0)
        nposts = n if nposts is None else nposts.add(n, fill_value=0)
avg = ssum / scnt.replace(0, np.nan)
avg["n_posts"] = nposts
avg = avg[avg["n_posts"] >= MINPOSTS].copy()
print(f"users with >= {MINPOSTS} posts: {len(avg)}")

# labels: PCOS=diagnosed; controls = never mentioned PCOS; drop the ambiguous middle
avg["y"] = np.where(avg.index.isin(diag), 1,
                    np.where(avg.index.isin(mentioners), -1, 0))
data = avg[avg.y >= 0].copy()
y = data.y.values
X = data[feats].fillna(data[feats].median()).values
print(f"PCOS={int((y==1).sum())}  non-PCOS={int((y==0).sum())}  base rate={y.mean():.3f}\n")

# feature pre-selection (anova) — informational; done on full data like script 42
F, _ = f_classif(X, y); top = np.argsort(np.nan_to_num(F))[::-1][:FSEL]
Xs = X[:, top]

sel_feats = [feats[i] for i in top]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tr, te in skf.split(Xs, y):
    w = compute_sample_weight("balanced", y[tr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ExplainableBoostingClassifier(interactions=0, random_state=42)
        m.fit(Xs[tr], y[tr], sample_weight=w)
    oof[te] = m.predict_proba(Xs[te])[:, 1]
auc = roc_auc_score(y, oof); ap = average_precision_score(y, oof)
print(f"OOF AUC = {auc:.4f} | PR-AUC(AP) = {ap:.4f} (base rate {y.mean():.3f}, lift {ap/y.mean():.2f}x)\n")

# global feature importances (fit one EBM on all data) + direction (PCOS vs non-PCOS mean)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ebm = ExplainableBoostingClassifier(interactions=0, random_state=42)
    ebm.fit(Xs, y, sample_weight=compute_sample_weight("balanced", y))
imp = dict(zip(ebm.term_names_, ebm.term_importances()))
imp = {sel_feats[int(k.replace("feature_", "").split()[0])] if k.startswith("feature_") else k: v
       for k, v in imp.items()} if any(t.startswith("feature_") for t in ebm.term_names_) else \
      dict(zip(sel_feats, ebm.term_importances()))
zstd = data[sel_feats].std().replace(0, np.nan)
dirn = (data.loc[y == 1, sel_feats].mean() - data.loc[y == 0, sel_feats].mean()) / zstd
rank = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:15]
print("Top-15 features driving PCOS classification (importance | dir: + = higher in PCOS, in SDs):")
for name, v in rank:
    d = dirn.get(name, np.nan)
    print(f"  {name:42s} imp={v:.3f}  dir={d:+.2f}")
pd.DataFrame(rank, columns=["feature", "importance"]).to_csv(ROOT / "outputs" / "pcos_trait_ebm_importance_droppcos.csv", index=False)
print("\nsaved outputs/pcos_trait_ebm_importance_droppcos.csv")

# shape-function grid (as in the original script)
ebm_named = ExplainableBoostingClassifier(interactions=0, random_state=42, feature_names=sel_feats)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ebm_named.fit(Xs, y, sample_weight=compute_sample_weight("balanced", y))
shapes_all = shape_functions(ebm_named, sel_feats, ["non-PCOS", "PCOS"])
top_names = [name for name, _ in rank[:10]]
shapes = {f: shapes_all[f] for f in top_names if f in shapes_all}
plot_shape_grid(shapes, ROOT / "outputs" / "pcos_trait_shapes_droppcos.png", "PCOS", "non-PCOS", top_n=10)
print("saved outputs/pcos_trait_shapes_droppcos.png")
print("Done.")
