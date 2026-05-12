#!/usr/bin/env python3
"""
Script 27: Hybrid Discovery-Validation Pipeline
BC vs Natural Cycle — OOF SHAP Discovery + Mann-Whitney U + Benjamini-Hochberg FDR

Architecture:
  [1] Load & build matrix — day-level timeline → per-user z-scored features
                             → phase means → user × (Phase × feature) wide matrix
  [2] ML Discovery         — 5-fold StratifiedKFold XGBoost, SHAP on OOF test folds only
  [3] Stat Gauntlet        — Mann-Whitney U (user-level Phase__feature means) → BH-FDR
                             → Cohen's d for survivors
  [4] Outputs              — terminal report + CSV table + SHAP beeswarm PNG

Data:
  BC cohort   : combined-pill stable users + recently-started (posts after start date)
                from processed/bc_wide_candidates_with_labels.csv
  Control     : natural-cycle users from interim/consensus_periods_*no_bc*.csv
  Features    : {feature}_mean columns from daily-aggregated timeline,
                z-scored per user then averaged per (user, phase)
  Phases      : fixed 28-day cycle (same as script 26)

Why per-user z-scoring + phase means is valid for Mann-Whitney:
  Global per-user z-score mean is ≈ 0 across ALL days, but the mean within a
  specific phase is NOT 0 — it captures how much that phase deviates from the
  user's own baseline.  BC users have suppressed hormonal variation, so their
  phase-level deviations will differ systematically from natural-cycle users.

Usage:
  python scripts/27_discovery_pipeline.py
  python scripts/27_discovery_pipeline.py --stable-only
  python scripts/27_discovery_pipeline.py --top-n 20 --fdr-alpha 0.10
  python scripts/27_discovery_pipeline.py --n-folds 10 --skip-shap-plot -v
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file

# ── Phase constants (identical to script 26) ─────────────────────────────────
CYCLE_LENGTH = 28.0
PHASE_ORDER  = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_BOUNDS = {
    "Menstrual":  (0,  5),
    "Follicular": (5,  13),
    "Ovulation":  (13, 16),
    "Luteal":     (16, 28),
}

# ── Progestin-only pill keywords ─────────────────────────────────────────────
_POP_KEYWORDS: frozenset[str] = frozenset({
    "pop", "norethindrone", "norgestrel", "desogestrel", "slynd", "opill",
    "cerazette", "cerelle", "nora-be", "camila", "errin", "jencycla", "lyza",
})

# ── Metadata columns to exclude from features ────────────────────────────────
_META_COLS: frozenset[str] = frozenset({
    "author", "author_lower", "group", "label",
    "created_utc", "subreddit", "author_flair_text",
    "offset_from_cd1", "offset_mod", "phase",
    "post_id", "permalink", "title", "selftext",
    "start_cd1", "cycle_length", "pattern_source", "anchor_date",
})

RANDOM_STATE = 42


def _ts() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# ── Helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pooled Cohen's d: positive means group a > group b.
    Standard two-sample formula with (n-1)-weighted pooled variance.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    pooled_var = (
        (n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)
    ) / (n_a + n_b - 2)
    sd = np.sqrt(pooled_var)
    return (np.mean(a) - np.mean(b)) / sd if sd > 0.0 else 0.0


def _is_combined_pill(pill_name) -> bool | None:
    """
    True = confirmed combined pill.  False = confirmed POP.  None = unknown.
    Unknown pill names are not disqualifying — treated as possibly combined.
    """
    if pd.isna(pill_name):
        return None
    return not any(k in str(pill_name).lower() for k in _POP_KEYWORDS)


def _assign_phases(tl: pd.DataFrame) -> pd.DataFrame:
    """Assign 4-phase labels from offset_from_cd1 using a fixed 28-day cycle."""
    tl = tl.copy()
    tl["offset_mod"] = tl["offset_from_cd1"] % CYCLE_LENGTH
    tl["phase"] = tl["offset_mod"].apply(
        lambda x: next((p for p, (lo, hi) in PHASE_BOUNDS.items() if lo <= x < hi), None)
    )
    return tl[tl["phase"].notna()].copy()


# ── Cohort loading ────────────────────────────────────────────────────────────

def _load_bc_users(
    labels_path: Path,
    include_started: bool,
) -> tuple[set[str], dict[str, float]]:
    """
    Returns
    -------
    stable_set  : lowercase author names, stable combined-pill users
    started_map : author_lower → start_cd1 offset (float), recently-started users
    """
    try:
        df = pd.read_excel(labels_path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(labels_path, encoding="utf-8-sig")

    df = df[df["is_bc_pill"] == True].copy()
    df["_combined"] = df["pill_name"].apply(_is_combined_pill)

    by_user = df.groupby("author").agg(
        any_started=("started_recently", "any"),
        any_stopped=("stopped_recently", "any"),
        # True if any row confirms combined AND no row confirms POP (None = unknown, ok)
        is_comb=("_combined", lambda x: x.any() and not (x == False).any()),
    ).reset_index()

    stable_set: set[str] = set(
        by_user[
            ~by_user["any_started"] & ~by_user["any_stopped"] & by_user["is_comb"]
        ]["author"].str.lower()
    )

    started_map: dict[str, float] = {}
    if include_started:
        started_posts = df[
            (df["started_recently"] == True) & (df["_combined"] != False)
        ].copy()
        started_posts["start_cd1"] = (
            started_posts["offset_from_cd1"] + started_posts["started_offset"]
        )
        started_posts["abs_off"] = started_posts["started_offset"].abs()
        user_start_cd1 = (
            started_posts.sort_values("abs_off")
            .groupby("author")["start_cd1"]
            .first()
        )
        started_df = by_user[
            by_user["any_started"] & ~by_user["any_stopped"] & by_user["is_comb"]
        ]
        started_map = {
            a.lower(): float(user_start_cd1[a])
            for a in started_df["author"]
            if a in user_start_cd1.index
        }

    return stable_set, started_map


def _load_natural_users(interim_dir: Path, files_cfg: dict) -> set[str]:
    path = find_latest_file(interim_dir, files_cfg["consensus_periods"] + "_*no_bc*.csv")
    if path is None:
        path = find_latest_file(interim_dir, files_cfg["consensus_periods"] + "_*.csv")
    if path is None:
        raise FileNotFoundError("No consensus_periods file found in interim directory.")
    df = pd.read_csv(path)
    user_col = next(
        (c for c in ("author", "user") if c in df.columns),
        df.columns[0],
    )
    return set(df[user_col].str.lower())


# ── Dataset builder ───────────────────────────────────────────────────────────

def load_dataset(
    cfg: dict,
    include_started: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a user-level Phase × feature wide matrix with per-user z-scored features.

    Pipeline:
      1. Load day-level daily-aggregated timeline (with anchors).
      2. Apply per-user z-score to each raw {feature}_mean column across ALL days.
         This removes individual linguistic baselines while preserving phase-level
         deviations: a user's luteal z-score mean ≠ 0 even though their global
         mean is ≈ 0, making between-group tests valid.
      3. Assign 4-phase labels via fixed 28-day cycle (same as script 26).
      4. Compute per-(user, phase) means of z-scored features.
      5. Pivot to one row per user: Phase__feature columns.

    Returns
    -------
    wide      : DataFrame [author_lower, label, Phase__feature × N]
    feat_cols : Phase__feature column names with ≥10% coverage
    """
    interim_dir   = ROOT / cfg["paths"]["interim"]
    processed_dir = ROOT / cfg["paths"]["processed"]
    files_cfg     = cfg["paths"]["files"]
    bc_cfg        = cfg["bc_classifier"]

    # ── Cohort membership ────────────────────────────────────────────────────
    labels_path = processed_dir / bc_cfg["labels_file"]
    stable_set, started_map = _load_bc_users(labels_path, include_started)
    nat_users = _load_natural_users(interim_dir, files_cfg)

    bc_all = stable_set | set(started_map.keys())
    overlap = bc_all & nat_users
    if overlap:
        logging.warning(f"  {len(overlap)} users in both cohorts — removing from Control")
        nat_users -= overlap

    logging.info(f"  BC stable   : {len(stable_set):,}")
    logging.info(f"  BC started  : {len(started_map):,}")
    logging.info(f"  Natural     : {len(nat_users):,}")

    # ── Timeline ─────────────────────────────────────────────────────────────
    # BC users live in the with-anchors timeline; phase-labeled files are
    # cohort-specific and would silently drop most BC users.
    tl_path = find_latest_file(
        interim_dir, files_cfg["daily_aggregated_with_anchors"] + "_*.csv"
    )
    if tl_path is None:
        tl_path = find_latest_file(
            interim_dir, files_cfg["timeline_with_anchors"] + "_*.csv"
        )
    if tl_path is None:
        raise FileNotFoundError("No daily-aggregated or anchored timeline found.")

    logging.info(f"  Timeline    : {Path(tl_path).name}")
    tl = pd.read_csv(tl_path, low_memory=False)
    tl["author_lower"] = tl["author"].str.lower()

    # ── Feature detection ────────────────────────────────────────────────────
    raw_cols = [c for c in tl.columns if c.endswith("_mean")]
    if len(raw_cols) < 5:
        raw_cols = [
            c for c in tl.columns
            if c not in _META_COLS
            and pd.api.types.is_numeric_dtype(tl[c])
            and tl[c].notna().mean() > 0.10
        ]
    logging.info(f"  Raw feature cols : {len(raw_cols)}")

    # ── Filter to cohort rows ─────────────────────────────────────────────────
    tl_stable  = tl[tl["author_lower"].isin(stable_set)].copy()

    tl_started = tl[tl["author_lower"].isin(started_map)].copy()
    if not tl_started.empty:
        tl_started["_start_cd1"] = tl_started["author_lower"].map(started_map)
        tl_started = tl_started[
            tl_started["offset_from_cd1"] >= tl_started["_start_cd1"]
        ].copy()

    tl_nat = tl[tl["author_lower"].isin(nat_users)].copy()
    tl = pd.concat([tl_stable, tl_started, tl_nat], ignore_index=True)
    tl["label"] = tl["author_lower"].map(lambda a: 1 if a in bc_all else 0)

    # ── Per-user z-score across all days ─────────────────────────────────────
    eps = 1e-10
    for col in raw_cols:
        grp   = tl.groupby("author_lower")[col]
        mu    = grp.transform("mean")
        sigma = grp.transform("std").fillna(eps).clip(lower=eps)
        tl[col] = (tl[col] - mu) / sigma

    # ── Phase assignment ──────────────────────────────────────────────────────
    tl = _assign_phases(tl)

    # ── Per-(user, phase) means of z-scored features ──────────────────────────
    up = (
        tl.groupby(["author_lower", "label", "phase"])[raw_cols]
        .mean()
        .reset_index()
    )

    # ── Wide pivot: one row per user ──────────────────────────────────────────
    label_by_user = tl.groupby("author_lower")["label"].first().to_dict()
    records = []
    for author, udf in up.groupby("author_lower"):
        rec = {"author_lower": author, "label": label_by_user[author]}
        for ph in PHASE_ORDER:
            ph_df = udf[udf["phase"] == ph]
            for f in raw_cols:
                key = f"{ph}__{f}"
                rec[key] = float(ph_df[f].values[0]) if len(ph_df) > 0 else np.nan
        records.append(rec)

    wide = pd.DataFrame(records)

    feat_cols = [
        c for c in wide.columns
        if "__" in c and wide[c].notna().mean() > 0.10
    ]

    return wide, feat_cols


# ── ML Discovery Engine ───────────────────────────────────────────────────────

def run_oof_discovery(
    wide: pd.DataFrame,
    feat_cols: list[str],
    n_folds: int,
) -> tuple[pd.Series, float, np.ndarray]:
    """
    5-fold StratifiedKFold XGBoost with SHAP computed on OOF test folds only.
    Data is user-level (one row per user), so no GroupKFold is needed.

    Returns
    -------
    mean_abs_shap : Series[feature → mean |OOF SHAP|], sorted descending
    mean_auc      : mean OOF ROC-AUC
    oof_shap      : signed OOF SHAP matrix (n_users × n_features)
    """
    X = wide[feat_cols].to_numpy(dtype=float)
    y = wide["label"].to_numpy()

    bc_n   = int((y == 1).sum())
    ctrl_n = int((y == 0).sum())
    spw    = ctrl_n / bc_n
    logging.info(f"  Users — BC: {bc_n:,}   Control: {ctrl_n:,}")
    logging.info(f"  scale_pos_weight = {ctrl_n}/{bc_n} = {spw:.3f}")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_shap: np.ndarray = np.zeros_like(X, dtype=float)
    fold_aucs: list[float] = []

    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        fold_aucs.append(auc)

        # SHAP strictly on the test fold — never on training data
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_te)
        if isinstance(sv, list):   # older shap API: [class0_vals, class1_vals]
            sv = sv[1]
        oof_shap[te_idx] = sv

        bc_te   = int(y_te.sum())
        ctrl_te = int((y_te == 0).sum())
        logging.info(
            f"    Fold {fold_i}/{n_folds}  AUC={auc:.4f}  "
            f"[BC={bc_te}, Control={ctrl_te}]"
        )

    mean_abs_shap = pd.Series(
        np.abs(oof_shap).mean(axis=0),
        index=feat_cols,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    return mean_abs_shap, float(np.mean(fold_aucs)), oof_shap


# ── Statistical Gauntlet ──────────────────────────────────────────────────────

def run_statistical_gauntlet(
    wide: pd.DataFrame,
    top_features: list[str],
    fdr_alpha: float,
) -> pd.DataFrame:
    """
    Two independent tests per feature, each with its own BH-FDR correction:

    Global (Mann-Whitney U):
      Two-sided test on the per-user Phase__feature value distributions.
      Cohen's d computed only for features whose global p survives BH-FDR.

    Subgroup tail (Fisher's Exact):
      Threshold = 90th percentile of the feature across ALL users combined.
      Contingency: [[BC_above, BC_below], [Ctrl_above, Ctrl_below]].
      Detects extreme responders whose signal is washed out by global median tests.
    """
    bc_mask   = wide["label"] == 1
    ctrl_mask = wide["label"] == 0

    raw_pvals_mw: list[float] = []
    raw_pvals_fe: list[float] = []

    # Single pass: collect values, Mann-Whitney p, Fisher p, and counts
    _cache: list[dict] = []
    for feat in top_features:
        bc_vals   = wide.loc[bc_mask,   feat].dropna().to_numpy()
        ctrl_vals = wide.loc[ctrl_mask, feat].dropna().to_numpy()

        # Global Mann-Whitney
        _, p_mw = mannwhitneyu(bc_vals, ctrl_vals, alternative="two-sided")
        raw_pvals_mw.append(float(p_mw))

        # Subgroup tail: 90th percentile across ALL users (BC + Control combined)
        threshold  = float(np.nanpercentile(wide[feat].values, 90))
        bc_above   = int((bc_vals   > threshold).sum())
        bc_below   = int(len(bc_vals)   - bc_above)
        ctrl_above = int((ctrl_vals > threshold).sum())
        ctrl_below = int(len(ctrl_vals) - ctrl_above)

        _, p_fe = fisher_exact(
            [[bc_above, bc_below], [ctrl_above, ctrl_below]],
            alternative="two-sided",
        )
        raw_pvals_fe.append(float(p_fe))

        _cache.append({
            "bc_vals": bc_vals, "ctrl_vals": ctrl_vals,
            "threshold": threshold,
            "bc_above": bc_above, "ctrl_above": ctrl_above,
        })

    # Independent BH-FDR corrections on each p-value family
    reject_mw, adj_mw, _, _ = multipletests(raw_pvals_mw, alpha=fdr_alpha, method="fdr_bh")
    reject_fe, adj_fe, _, _ = multipletests(raw_pvals_fe, alpha=fdr_alpha, method="fdr_bh")

    rows = []
    for i, feat in enumerate(top_features):
        c = _cache[i]
        d = cohens_d(c["bc_vals"], c["ctrl_vals"]) if reject_mw[i] else np.nan
        phase, raw = feat.split("__", 1) if "__" in feat else ("", feat)
        clean = f"[{phase}] {raw.replace('_mean', '')}" if phase else raw
        rows.append({
            "feature":                feat,
            "feature_clean":          clean,
            "raw_p_mw":               raw_pvals_mw[i],
            "adj_p_bh_mw":            float(adj_mw[i]),
            "fdr_significant_mw":     bool(reject_mw[i]),
            "cohens_d":               d,
            "p90_threshold":          c["threshold"],
            "bc_above_p90":           c["bc_above"],
            "ctrl_above_p90":         c["ctrl_above"],
            "raw_p_fisher":           raw_pvals_fe[i],
            "adj_p_bh_fisher":        float(adj_fe[i]),
            "fdr_significant_fisher": bool(reject_fe[i]),
        })

    return pd.DataFrame(rows)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(
    mean_abs_shap: pd.Series,
    stats_df: pd.DataFrame,
    mean_auc: float,
    n_folds: int,
    fdr_alpha: float,
) -> None:
    def _stars(adj_p: float, sig: bool) -> str:
        if not sig:       return "   "
        if adj_p < 0.001: return "***"
        if adj_p < 0.01:  return "** "
        return "*  "

    W = 108
    bar  = "═" * W
    thin = "─" * W

    print(f"\n{bar}")
    print(f"  HYBRID DISCOVERY-VALIDATION REPORT  —  BC vs Natural Cycle")
    print(f"  OOF CV AUC (XGBoost, {n_folds}-fold StratifiedKFold):  {mean_auc:.4f}")
    print(bar)
    print(
        f"  {'Feature':<44} {'|SHAP|':>7}  "
        f"{'Global p(BH)':>13}     {'Cohen d':>8}  "
        f"{'Subgrp p(BH)':>13}"
    )
    print(thin)

    for _, row in stats_df.iterrows():
        imp   = mean_abs_shap[row["feature"]]
        d_str = f"{row['cohens_d']:+.3f}" if pd.notna(row["cohens_d"]) else "   —  "
        mw_s  = _stars(row["adj_p_bh_mw"],    row["fdr_significant_mw"])
        fe_s  = _stars(row["adj_p_bh_fisher"], row["fdr_significant_fisher"])
        print(
            f"  {row['feature_clean']:<44} "
            f"{imp:>7.4f}  "
            f"{row['adj_p_bh_mw']:>13.3e} {mw_s}  "
            f"{d_str:>8}  "
            f"{row['adj_p_bh_fisher']:>13.3e} {fe_s}"
        )

    print(thin)
    n_mw     = int(stats_df["fdr_significant_mw"].sum())
    n_fe     = int(stats_df["fdr_significant_fisher"].sum())
    n_either = int((stats_df["fdr_significant_mw"] | stats_df["fdr_significant_fisher"]).sum())
    print(f"  Global   (Mann-Whitney BH α={fdr_alpha}):  {n_mw}/{len(stats_df)} significant")
    print(f"  Subgroup (Fisher p90   BH α={fdr_alpha}):  {n_fe}/{len(stats_df)} significant"
          f"  [90th pctile threshold, BC+Control combined]")
    print(f"  Either test significant: {n_either}/{len(stats_df)}")
    print(f"  * p<0.05  ** p<0.01  *** p<0.001  (all FDR-adjusted)")
    print(f"{bar}\n")


def save_shap_beeswarm(
    wide: pd.DataFrame,
    top_features: list[str],
    feat_cols: list[str],
    oof_shap: np.ndarray,
    out_dir: Path,
    tag: str,
    top_n: int,
) -> Path:
    """
    SHAP beeswarm using the *signed* OOF SHAP values.
    Positive SHAP → feature pushes prediction toward BC (label=1).
    Dot colour = feature value (high = red, low = blue).
    """
    top_idx  = [feat_cols.index(f) for f in top_features]
    shap_top = oof_shap[:, top_idx]
    X_top    = wide[top_features].to_numpy(dtype=float)

    # Format: "[Phase] feature_name" — matches bc_classifier.md convention
    display_names = []
    for f in top_features:
        if "__" in f:
            phase, raw = f.split("__", 1)
            display_names.append(f"[{phase}] {raw.replace('_mean', '')}")
        else:
            display_names.append(f)

    fig, ax = plt.subplots(figsize=(11, 9))
    shap.summary_plot(
        shap_top,
        X_top,
        feature_names=display_names,
        max_display=top_n,
        plot_type="dot",
        show=False,
    )
    plt.title(
        f"OOF SHAP — BC vs Natural Cycle  "
        f"(Top {top_n} Phase × features, positive = BC)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / f"discovery_shap_beeswarm_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--n-folds", type=int, default=None,
                   help="CV folds (default: bc_classifier.n_folds from config)")
    p.add_argument("--top-n", type=int, default=15,
                   help="Top N global features passed to statistical validation (default: 15)")
    p.add_argument("--phase-top-n", type=int, default=5,
                   help="Top K features guaranteed from EACH phase (default: 5). "
                        "Merged with global top-N so no phase is crowded out.")
    p.add_argument("--fdr-alpha", type=float, default=0.05,
                   help="FDR significance threshold (default: 0.05)")
    p.add_argument("--stable-only", action="store_true",
                   help="Use only stable BC users (exclude recently-started)")
    p.add_argument("--skip-shap-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg     = load_config(ROOT / args.config)
    bc_cfg  = cfg["bc_classifier"]
    out_dir = ROOT / cfg["paths"]["reports"] / bc_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_folds         = args.n_folds if args.n_folds is not None else int(bc_cfg["n_folds"])
    top_n           = args.top_n
    phase_top_n     = args.phase_top_n
    fdr_alpha       = args.fdr_alpha
    include_started = not args.stable_only
    tag             = _ts()

    # ── [1] Build matrix ─────────────────────────────────────────────────────
    logging.info("[1/4] Loading + building phase × feature matrix …")
    wide, feat_cols = load_dataset(cfg, include_started=include_started)

    n_bc  = int((wide["label"] == 1).sum())
    n_nat = int((wide["label"] == 0).sum())
    logging.info(f"  Users: {len(wide):,}  (BC={n_bc}, Natural={n_nat})")
    logging.info(f"  Phase × feature columns: {len(feat_cols)}")

    # ── [2] ML Discovery ─────────────────────────────────────────────────────
    logging.info(f"[2/4] ML Discovery — {n_folds}-fold OOF SHAP …")
    mean_abs_shap, mean_auc, oof_shap = run_oof_discovery(wide, feat_cols, n_folds)

    # ── Per-phase SHAP breakdown ──────────────────────────────────────────────
    logging.info(f"\n  SHAP importance — top {phase_top_n} per phase:")
    phase_top_features: list[str] = []
    for ph in PHASE_ORDER:
        ph_series = mean_abs_shap[
            mean_abs_shap.index.str.startswith(ph + "__")
        ]
        logging.info(f"    {'─'*4} {ph} {'─'*4}")
        for rank, (feat, imp) in enumerate(ph_series.head(phase_top_n).items(), 1):
            raw = feat.split("__", 1)[1].replace("_mean", "")
            global_rank = int(mean_abs_shap.index.get_loc(feat)) + 1
            logging.info(f"      {rank}. {raw:<42} {imp:.4f}  (global #{global_rank})")
        phase_top_features.extend(ph_series.head(phase_top_n).index.tolist())

    # ── Combined feature list: global top-N ∪ per-phase top-K (order-preserving) ──
    seen: set[str] = set()
    top_features: list[str] = []
    for f in list(mean_abs_shap.head(top_n).index) + phase_top_features:
        if f not in seen:
            seen.add(f)
            top_features.append(f)

    n_added = len(top_features) - top_n
    logging.info(
        f"\n  Gauntlet candidates: {top_n} global + {n_added} phase-guaranteed "
        f"= {len(top_features)} total (deduplicated)"
    )

    # ── [3] Statistical Gauntlet ─────────────────────────────────────────────
    logging.info("[3/4] Statistical Gauntlet — Mann-Whitney + Fisher Tail → BH-FDR …")
    stats_df = run_statistical_gauntlet(wide, top_features, fdr_alpha)

    # ── [4] Outputs ──────────────────────────────────────────────────────────
    logging.info("[4/4] Outputs …")

    print_report(mean_abs_shap, stats_df, mean_auc, n_folds, fdr_alpha)

    stats_df["shap_importance"] = [mean_abs_shap[f] for f in stats_df["feature"]]
    col_order = [
        "feature_clean", "feature", "shap_importance",
        "raw_p_mw", "adj_p_bh_mw", "fdr_significant_mw", "cohens_d",
        "p90_threshold", "bc_above_p90", "ctrl_above_p90",
        "raw_p_fisher", "adj_p_bh_fisher", "fdr_significant_fisher",
    ]
    out_csv = out_dir / f"discovery_validation_report_{tag}.csv"
    stats_df[col_order].to_csv(out_csv, index=False)
    logging.info(f"  Stats table  → {out_csv.name}")

    if not args.skip_shap_plot:
        out_png = save_shap_beeswarm(
            wide, top_features, feat_cols, oof_shap, out_dir, tag, top_n
        )
        logging.info(f"  SHAP beeswarm → {out_png.name}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
