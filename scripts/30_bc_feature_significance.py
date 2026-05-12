"""Step 30 — BC Feature Significance and Effect Sizes
======================================================
For each phase × feature combination from the BC classifier, computes:
  - Mann-Whitney U test (BC vs Control user means)
  - FDR correction (Benjamini-Hochberg) across all tests
  - Cohen's d effect size
  - SHAP rank from the full-data XGBoost model

A feature is a solid finding when high SHAP rank, significant q-value, and
meaningful Cohen's d all converge. This script surfaces where they diverge:
features the model uses but that have tiny absolute differences (e.g. arousal
Luteal: |d| < 0.1) vs features with large effects that also survive FDR.

Usage:
  python scripts/30_bc_feature_significance.py
  python scripts/30_bc_feature_significance.py --top 40
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("xgboost is required")

# ── Constants (mirror script 26) ────────────────────────────────────────────
CYCLE_LENGTH = 28.0
PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_BOUNDS = {
    "Menstrual":  (0,  5),
    "Follicular": (5,  13),
    "Ovulation":  (13, 16),
    "Luteal":     (16, 28),
}
POP_KEYWORDS = [
    "pop", "norethindrone", "norgestrel", "desogestrel", "slynd", "opill",
    "cerazette", "cerelle", "nora-be", "camila", "errin", "jencycla", "lyza",
]
NON_FEATURE_COLS = {
    "author", "author_lower", "group", "created_utc", "subreddit",
    "author_flair_text", "offset_from_cd1", "offset_mod", "phase",
    "post_id", "permalink", "title", "selftext", "start_cd1",
}
PHASE_COLORS = {
    "Menstrual": "#e74c3c",
    "Follicular": "#2ecc71",
    "Ovulation": "#f39c12",
    "Luteal": "#9b59b6",
}


def _ts():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def is_combined_pill(pill_name):
    if pd.isna(pill_name):
        return None
    return not any(k in str(pill_name).lower() for k in POP_KEYWORDS)


# ── Data loading (mirrors script 26) ────────────────────────────────────────

def load_bc_users(labels_path):
    try:
        labels = pd.read_excel(labels_path, engine="openpyxl")
    except Exception:
        labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels = labels[labels["is_bc_pill"] == True].copy()
    labels["is_combined"] = labels["pill_name"].apply(is_combined_pill)
    user_flags = labels.groupby("author").agg(
        any_started=("started_recently", "any"),
        any_stopped=("stopped_recently", "any"),
        is_comb=("is_combined", lambda x: x.any() and not (x == False).any()),
    ).reset_index()
    stable_set = set(
        user_flags[~user_flags["any_started"] & ~user_flags["any_stopped"] & user_flags["is_comb"]]
        ["author"].str.lower()
    )
    started_posts = labels[(labels["started_recently"] == True) & (labels["is_combined"] == True)].copy()
    started_posts["start_cd1"] = started_posts["offset_from_cd1"] + started_posts["started_offset"]
    started_posts["abs_off"] = started_posts["started_offset"].abs()
    user_start_cd1 = started_posts.sort_values("abs_off").groupby("author")["start_cd1"].first()
    started_df = user_flags[user_flags["any_started"] & ~user_flags["any_stopped"] & user_flags["is_comb"]]
    started_map = {a.lower(): user_start_cd1[a] for a in started_df["author"] if a in user_start_cd1.index}
    return stable_set, started_map


def load_data(cfg, interim_dir, labels_path):
    files_cfg = cfg["paths"]["files"]

    path = find_latest_file(interim_dir, files_cfg["timeline_with_anchors"] + "_*.csv")
    logging.info(f"Timeline: {path.name}")
    tl = pd.read_csv(path, low_memory=False)
    tl["author_lower"] = tl["author"].str.lower()

    stable_set, started_map = load_bc_users(labels_path)

    nat_path = find_latest_file(interim_dir, files_cfg["consensus_periods"] + "_*no_bc*.csv")
    if nat_path is None:
        nat_path = find_latest_file(interim_dir, files_cfg["consensus_periods"] + "_*.csv")
    logging.info(f"Natural users: {nat_path.name}")
    nat_df = pd.read_csv(nat_path)
    user_col = "user" if "user" in nat_df.columns else nat_df.columns[0]
    nat_set = set(nat_df[user_col].str.lower())

    logging.info(f"BC stable: {len(stable_set)}, started: {len(started_map)}, natural: {len(nat_set)}")

    tl_stable = tl[tl["author_lower"].isin(stable_set)].copy(); tl_stable["group"] = "BC"
    tl_started = tl[tl["author_lower"].isin(started_map)].copy()
    tl_started["start_cd1"] = tl_started["author_lower"].map(started_map)
    tl_started = tl_started[tl_started["offset_from_cd1"] >= tl_started["start_cd1"]].copy()
    tl_started["group"] = "BC"
    tl_bc = pd.concat([tl_stable, tl_started], ignore_index=True)
    tl_ctrl = tl[tl["author_lower"].isin(nat_set)].copy(); tl_ctrl["group"] = "Control"
    combined = pd.concat([tl_bc, tl_ctrl], ignore_index=True)
    overlap = set(tl_bc["author_lower"]) & set(tl_ctrl["author_lower"])
    if overlap:
        combined = combined[~((combined["group"] == "Control") & (combined["author_lower"].isin(overlap)))]

    combined["offset_mod"] = combined["offset_from_cd1"] % CYCLE_LENGTH
    combined["phase"] = combined["offset_mod"].apply(
        lambda x: next((p for p, (lo, hi) in PHASE_BOUNDS.items() if lo <= x < hi), None)
    )
    return combined[combined["phase"].notna()].copy()


def get_feature_cols(tl):
    return [
        c for c in tl.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(tl[c])
        and tl[c].notna().mean() > 0.3
    ]


def zscore_per_user(tl: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Normalize each feature to (x − user_mean) / user_std across all of a user's posts."""
    tl = tl.copy()
    stats = tl.groupby("author_lower")[feat_cols].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    tl = tl.join(stats, on="author_lower")
    for f in feat_cols:
        std = tl[f"{f}_std"].replace(0, np.nan)
        tl[f] = (tl[f] - tl[f"{f}_mean"]) / std
    return tl.drop(columns=[c for c in tl.columns if c.endswith("_mean") or c.endswith("_std")])


def build_feature_matrix(tl, feat_cols):
    up = tl.groupby(["author", "group", "phase"])[feat_cols].mean().reset_index()
    records = []
    for (author, group), udf in up.groupby(["author", "group"]):
        rec = {"author": author, "group": group}
        for ph in PHASE_ORDER:
            phdf = udf[udf["phase"] == ph]
            for f in feat_cols:
                rec[f"{ph}__{f}"] = phdf[f].values[0] if len(phdf) > 0 else np.nan
        records.append(rec)
    wide = pd.DataFrame(records)
    phase_feat_cols = [c for c in wide.columns if "__" in c and wide[c].notna().mean() > 0.1]
    return wide, phase_feat_cols


# ── Statistics ───────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Pooled Cohen's d (positive = a > b)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * np.std(a, ddof=1) ** 2 + (n_b - 1) * np.std(b, ddof=1) ** 2) / (n_a + n_b - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def compute_statistics(wide, phase_feat_cols, y):
    bc_mask = y == 1
    ctrl_mask = y == 0
    X = wide[phase_feat_cols].values
    rows = []
    for i, col in enumerate(phase_feat_cols):
        phase, feat = col.split("__", 1)
        bc_vals = X[bc_mask, i]
        ctrl_vals = X[ctrl_mask, i]
        bc_mean = np.nanmean(bc_vals)
        ctrl_mean = np.nanmean(ctrl_vals)
        d = cohens_d(bc_vals, ctrl_vals)
        bc_clean = bc_vals[~np.isnan(bc_vals)]
        ctrl_clean = ctrl_vals[~np.isnan(ctrl_vals)]
        if len(bc_clean) >= 5 and len(ctrl_clean) >= 5:
            _, p = mannwhitneyu(bc_clean, ctrl_clean, alternative="two-sided")
        else:
            p = np.nan
        rows.append({
            "phase": phase,
            "feature": feat,
            "bc_mean": bc_mean,
            "ctrl_mean": ctrl_mean,
            "cohen_d": d,
            "mwu_p": p,
            "direction": "HIGH" if bc_mean > ctrl_mean else "LOW",
        })
    df = pd.DataFrame(rows)
    valid = df["mwu_p"].notna()
    _, q_vals, _, _ = multipletests(df.loc[valid, "mwu_p"], method="fdr_bh")
    df.loc[valid, "fdr_q"] = q_vals
    df["abs_d"] = df["cohen_d"].abs()
    return df


def get_shap_ranks(model, X, phase_feat_cols):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    ranks = pd.Series(mean_abs, index=phase_feat_cols).rank(ascending=False).astype(int)
    return ranks


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_effect_sizes(df, output_dir, tag, top_n=30):
    """Horizontal bar chart of top features by |Cohen's d|, colored by phase."""
    df_plot = df.nlargest(top_n, "abs_d").sort_values("abs_d")
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))

    colors = [PHASE_COLORS.get(ph, "#999") for ph in df_plot["phase"]]
    bars = ax.barh(range(len(df_plot)), df_plot["cohen_d"], color=colors, alpha=0.8, height=0.7)

    # Significance markers
    for i, (_, row) in enumerate(df_plot.iterrows()):
        if pd.notna(row.get("fdr_q")) and row["fdr_q"] < 0.05:
            ax.text(row["cohen_d"] + (0.005 if row["cohen_d"] >= 0 else -0.005),
                    i, "*", va="center", ha="left" if row["cohen_d"] >= 0 else "right",
                    fontsize=10, color="black")

    labels = [f"[{r['phase'][:3]}] {r['feature']}" for _, r in df_plot.iterrows()]
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d  (positive = BC higher)")
    ax.set_title(f"Top {top_n} features by effect size — BC vs Natural Cycle\n* = FDR q < 0.05")

    legend_patches = [mpatches.Patch(color=c, label=ph) for ph, c in PHASE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    path = output_dir / f"bc_effect_sizes_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Effect size plot → {path.name}")


def plot_shap_vs_effect(df, output_dir, tag):
    """Scatter: SHAP rank vs |Cohen's d|, colored by phase, labeled for notable features."""
    df_valid = df[df["shap_rank"].notna() & df["abs_d"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for ph, color in PHASE_COLORS.items():
        sub = df_valid[df_valid["phase"] == ph]
        ax.scatter(sub["shap_rank"], sub["abs_d"], c=color, alpha=0.6, s=30, label=ph)

    # Label top-right quadrant: high SHAP AND high effect
    notable = df_valid[(df_valid["shap_rank"] <= 20) & (df_valid["abs_d"] >= 0.2)]
    for _, row in notable.iterrows():
        ax.annotate(f"{row['phase'][:3]}_{row['feature'][:20]}",
                    (row["shap_rank"], row["abs_d"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    ax.invert_xaxis()  # rank 1 on the right = most important
    ax.set_xlabel("SHAP rank (1 = most important to model, lower = less important)")
    ax.set_ylabel("|Cohen's d|")
    ax.set_title("SHAP rank vs Effect size — features in upper-right are most reliable")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, label="|d|=0.2 threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / f"bc_shap_vs_effect_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  SHAP vs effect scatter → {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--top", type=int, default=30, help="Top N features in effect size plot")
    p.add_argument("--zscore", action="store_true", default=True,
                   help="Apply per-user z-score normalization before analysis (default: True)")
    p.add_argument("--no-zscore", dest="zscore", action="store_false",
                   help="Use raw feature values instead of z-scored")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    bc_cfg = cfg["bc_classifier"]
    interim_dir = ROOT / cfg["paths"]["interim"]
    processed_dir = ROOT / cfg["paths"]["processed"]
    output_dir = ROOT / cfg["paths"]["reports"] / bc_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = processed_dir / bc_cfg["labels_file"]
    tag = _ts()

    # Load + build feature matrix
    tl = load_data(cfg, interim_dir, labels_path)
    feat_cols = get_feature_cols(tl)
    if args.zscore:
        logging.info("Applying per-user z-score normalization...")
        tl = zscore_per_user(tl, feat_cols)
    wide, phase_feat_cols = build_feature_matrix(tl, feat_cols)

    y = (wide["group"] == "BC").astype(int).values
    bc_n, ctrl_n = y.sum(), (y == 0).sum()
    X = wide[phase_feat_cols].values
    logging.info(f"Users: {len(wide)} ({bc_n} BC, {ctrl_n} Control), {len(phase_feat_cols)} features")

    # Fit model for SHAP
    logging.info("Fitting XGBoost for SHAP rankings...")
    model = XGBClassifier(
        objective="binary:logistic", n_estimators=300, max_depth=3,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ctrl_n / max(bc_n, 1),
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X, y)

    shap_ranks = get_shap_ranks(model, X, phase_feat_cols)

    # Statistical tests
    logging.info("Computing Mann-Whitney U tests + Cohen's d...")
    df = compute_statistics(wide, phase_feat_cols, y)
    df["col"] = df["phase"] + "__" + df["feature"]
    df["shap_rank"] = df["col"].map(shap_ranks)
    df = df.sort_values("abs_d", ascending=False).reset_index(drop=True)

    # Save full CSV
    csv_path = output_dir / f"bc_feature_significance_{tag}.csv"
    df.drop(columns="col").to_csv(csv_path, index=False)
    logging.info(f"  Full table → {csv_path.name}")

    # Filtered text report: q < 0.05 AND |d| >= 0.2
    sig = df[(df["fdr_q"] < 0.05) & (df["abs_d"] >= 0.2)].copy()
    mode = "z-scored (within-user deviations)" if args.zscore else "raw values"
    txt_lines = [
        f"BC vs Natural Cycle — Significant features (FDR q < 0.05, |Cohen's d| ≥ 0.2)  [{mode}]",
        f"Total features tested: {len(df)}  |  Surviving filter: {len(sig)}",
        "",
        f"{'Phase':<12} {'Feature':<50} {'Dir':>4}  {'d':>7}  {'q':>8}  {'SHAP rank':>9}",
        "-" * 95,
    ]
    for _, row in sig.iterrows():
        shap_str = f"{int(row['shap_rank'])}" if pd.notna(row["shap_rank"]) else "—"
        txt_lines.append(
            f"{row['phase']:<12} {row['feature']:<50} {row['direction']:>4}  "
            f"{row['cohen_d']:>+7.3f}  {row['fdr_q']:>8.4f}  {shap_str:>9}"
        )
    txt_lines += [
        "",
        "Effect size guide: small |d|≥0.2, medium |d|≥0.5, large |d|≥0.8",
        "",
        "=== By phase ===",
    ]
    for ph in PHASE_ORDER:
        ph_sig = sig[sig["phase"] == ph]
        txt_lines.append(f"\n{ph} ({len(ph_sig)} significant features):")
        for _, row in ph_sig.iterrows():
            shap_str = f"(SHAP #{int(row['shap_rank'])})" if pd.notna(row["shap_rank"]) else ""
            txt_lines.append(
                f"  {row['direction']:>4}  {row['feature']:<50}  d={row['cohen_d']:+.3f}  q={row['fdr_q']:.4f}  {shap_str}"
            )

    txt_path = output_dir / f"bc_feature_significance_{tag}.txt"
    txt_path.write_text("\n".join(txt_lines))
    logging.info(f"  Filtered report → {txt_path.name}")

    # Summary to console
    print("\n=== Significant features (FDR q<0.05, |d|≥0.2) ===")
    print(f"{'Phase':<12} {'Feature':<50} {'Dir':>4}  {'d':>7}  {'q':>8}  {'SHAP':>5}")
    print("-" * 90)
    for _, row in sig.iterrows():
        shap_str = f"#{int(row['shap_rank'])}" if pd.notna(row["shap_rank"]) else "—"
        print(f"{row['phase']:<12} {row['feature']:<50} {row['direction']:>4}  "
              f"{row['cohen_d']:>+7.3f}  {row['fdr_q']:>8.4f}  {shap_str:>5}")

    # How many high-SHAP features are NOT significant?
    top_shap = df[df["shap_rank"] <= 20].copy()
    not_sig = top_shap[~((top_shap["fdr_q"] < 0.05) & (top_shap["abs_d"] >= 0.2))]
    if len(not_sig):
        print(f"\n=== Top-20 SHAP features that did NOT survive (q<0.05 AND |d|≥0.2) ===")
        for _, row in not_sig.sort_values("shap_rank").iterrows():
            q_str = f"q={row['fdr_q']:.3f}" if pd.notna(row.get("fdr_q")) else "q=n/a"
            print(f"  SHAP #{int(row['shap_rank']):2d}  {row['phase']:<12} {row['feature']:<50}  d={row['cohen_d']:+.3f}  {q_str}")

    # Plots
    plot_effect_sizes(df, output_dir, tag, top_n=args.top)
    plot_shap_vs_effect(df, output_dir, tag)

    logging.info("Done.")


if __name__ == "__main__":
    main()
