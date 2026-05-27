"""
Step 35 — Binary EBM Classifiers for Consecutive Phase Pairs
=============================================================
Trains one EBM per consecutive phase transition:
  Menstrual → Follicular
  Follicular → Ovulation
  Ovulation  → Luteal
  Luteal     → Menstrual

Why binary pairs instead of 4-class?
  The 4-class EBM must separate all phases simultaneously.  Binary classifiers
  let the model focus on the specific linguistic shift that occurs at each
  transition, which is more interpretable and clinically meaningful.

Design choices (same rules as script 27):
  • StratifiedGroupKFold (5 folds, grouped by author) — a user's profiles
    appear in EITHER train OR test, never both.
  • sample_weight="balanced" — passed to ebm.fit() so minority phases are not
    drowned out (Ovulation is ~40% the size of Luteal at profile level).
  • Pure additive EBM (interactions=0) — every feature has an interpretable
    shape function.

Output (reports/ebm/pairs/)
  ebm_pair_<A>_vs_<B>_importance_<ts>.png   — top-N importance bar chart
  ebm_pair_<A>_vs_<B>_shapes_<ts>.png       — log-odds shape grid (top N features)
  ebm_pairs_oof_summary_<ts>.txt            — OOF metrics for all 4 pairs

Usage
  python scripts/35_ebm_binary_phase_pairs.py --config configs/base.yaml \\
      --phase-file data/interim/timeline_phase_labeled_fixed29_20260517T131445.csv
  python scripts/35_ebm_binary_phase_pairs.py --config configs/base.yaml \\
      --phase-file data/interim/timeline_phase_labeled_fixed29_20260517T131445.csv \\
      --top-n 10
"""

import argparse
import logging
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import PHASE_COLORS, PHASE_ORDER
from src.ml_data import load_phase_labeled_dataset
from src.analysis import aggregate_to_phase_profiles
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# All pairwise combinations (C(4,2) = 6)
PAIRS = [
    ("Menstrual",  "Follicular"),
    ("Menstrual",  "Ovulation"),
    ("Menstrual",  "Luteal"),
    ("Follicular", "Ovulation"),
    ("Follicular", "Luteal"),
    ("Ovulation",  "Luteal"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# EBM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ebm(seed: int) -> ExplainableBoostingClassifier:
    return ExplainableBoostingClassifier(
        max_bins=256,
        max_interaction_bins=64,
        interactions=0,
        learning_rate=0.01,
        max_rounds=5000,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SHAPE FUNCTION EXTRACTION  (same as scripts 27 / 33)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_bin_midpoints(bin_labels: list) -> np.ndarray:
    x_vals = []
    for label in bin_labels:
        label = str(label)
        try:
            if " to " in label:
                lo, hi = label.split(" to ")
                x_vals.append((float(lo) + float(hi)) / 2)
            elif label.startswith(">"):
                x_vals.append(float(label[1:].strip()))
            elif label.startswith("<="):
                x_vals.append(float(label[2:].strip()))
            else:
                x_vals.append(float(label))
        except ValueError:
            x_vals.append(np.nan)
    return np.array(x_vals, dtype=np.float64)


def _extract_density(data: dict, n_bins: int) -> np.ndarray:
    density_raw = data.get("density")
    if not isinstance(density_raw, dict):
        return np.ones(n_bins) / n_bins
    dens_counts = np.asarray(density_raw.get("scores", []), dtype=float)
    dens_edges  = np.asarray(density_raw.get("names",  []), dtype=float)
    if len(dens_counts) == 0 or len(dens_edges) < 2:
        return np.ones(n_bins) / n_bins
    total = dens_counts.sum()
    if total == 0:
        return np.ones(n_bins) / n_bins
    dens_frac = dens_counts / total
    if len(dens_counts) == n_bins:
        return dens_frac
    score_edges = np.asarray(data.get("names", []), dtype=float)
    if len(score_edges) < n_bins + 1:
        return np.ones(n_bins) / n_bins
    score_midpoints = (score_edges[:n_bins] + score_edges[1 : n_bins + 1]) / 2
    bin_indices = np.clip(
        np.searchsorted(dens_edges[1:], score_midpoints, side="left"),
        0, len(dens_frac) - 1,
    )
    density_per_bin = dens_frac[bin_indices]
    s = density_per_bin.sum()
    return density_per_bin / s if s > 0 else np.ones(n_bins) / n_bins


def extract_binary_shapes(
    ebm: ExplainableBoostingClassifier,
    feature_names: list[str],
    top_n: int,
    top_features_idx: np.ndarray,
) -> dict[int, dict]:
    """Return shape data for top-N features of a binary EBM.

    Returns dict keyed by feature index:
        x            : bin midpoints (z-score)
        y            : log-odds toward class 1 (the second phase)
        density      : normalised fraction of users per bin
        weighted_sum : density-weighted mean |log-odds|
    """
    global_exp = ebm.explain_global()
    result = {}
    for fi in top_features_idx[:top_n]:
        data   = global_exp.data(fi)
        scores = np.asarray(data["scores"])
        if scores.ndim == 2:          # some EBM versions return (n_bins, 1)
            scores = scores[:, 0]
        x = _parse_bin_midpoints(data["names"])
        n = min(len(x), len(scores))
        x, scores = x[:n], scores[:n]
        density = _extract_density(data, n)
        result[fi] = {
            "x":            x,
            "y":            scores,
            "density":      density,
            "weighted_sum": float(np.sum(np.abs(scores) * density)),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_binary_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> dict:
    """StratifiedGroupKFold OOF evaluation for a binary pair.

    Users are grouped so no user appears in both train and test.
    sample_weight is applied on each fold's training set.
    """
    gkf     = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred  = np.empty_like(y)
    y_proba = np.zeros(len(y), dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        logging.info(f"    Fold {fold}/{n_splits}  train={len(train_idx):,}  test={len(test_idx):,}")
        ebm = _make_ebm(seed)
        ebm.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        y_pred[test_idx]  = ebm.predict(X[test_idx])
        y_proba[test_idx] = ebm.predict_proba(X[test_idx])[:, 1]

    return {
        "y_true":  y,
        "y_pred":  y_pred,
        "y_proba": y_proba,
        "auc":             roc_auc_score(y, y_proba),
        "balanced_acc":    balanced_accuracy_score(y, y_pred),
        "accuracy":        accuracy_score(y, y_pred),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(name: str) -> str:
    return name.replace("_zscore", "").replace("_", " ")


def plot_importance(
    feature_importances: np.ndarray,
    feature_names: list[str],
    phase_a: str,
    phase_b: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    sorted_idx = np.argsort(np.abs(feature_importances))
    top_idx    = sorted_idx[-top_n:]

    vals   = [feature_importances[i] for i in top_idx]
    labels = [_clean(feature_names[i]) for i in top_idx]
    # positive = toward phase_b, negative = toward phase_a
    color_b = PHASE_COLORS.get(phase_b, "steelblue")
    color_a = PHASE_COLORS.get(phase_a, "tomato")
    colors  = [color_b if v >= 0 else color_a for v in vals]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(top_idx)), vals, color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Feature Importance (mean |log-odds|)")
    ax.set_title(
        f"EBM Feature Importances — {phase_a} vs {phase_b} (top {top_n})\n"
        f"Positive (right) → {phase_b}  |  Negative (left) → {phase_a}",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    patch_b = mpatches.Patch(color=color_b, label=phase_b)
    patch_a = mpatches.Patch(color=color_a, label=phase_a)
    ax.legend(handles=[patch_b, patch_a], fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance plot → {output_path.name}")


def plot_shape_grid(
    shape_data: dict[int, dict],
    feature_names: list[str],
    phase_a: str,
    phase_b: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Grid of binary log-odds shape plots.

    Positive log-odds (above y=0) → pushes toward phase_b.
    Negative log-odds (below y=0) → pushes toward phase_a.
    Background shading shows which phase dominates at each interval.
    """
    feats  = list(shape_data.keys())[:top_n]
    n_feat = len(feats)
    ncols  = 2
    nrows  = (n_feat + ncols - 1) // ncols

    color_a = PHASE_COLORS.get(phase_a, "tomato")
    color_b = PHASE_COLORS.get(phase_b, "steelblue")

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()

    for i, fi in enumerate(feats):
        ax   = axes[i]
        d    = shape_data[fi]
        x, y = d["x"], d["y"]
        valid = ~np.isnan(x)
        x_v, y_v = x[valid], y[valid]

        # Shade background by dominant phase
        for j in range(len(x_v) - 1):
            color = color_b if y_v[j] >= 0 else color_a
            ax.axvspan(x_v[j], x_v[j + 1], alpha=0.12, color=color, linewidth=0)

        ax.step(x_v, y_v, where="mid", color="black", linewidth=1.8)
        ax.fill_between(x_v, y_v, 0, step="mid",
                        where=y_v >= 0, alpha=0.35, color=color_b, label=phase_b)
        ax.fill_between(x_v, y_v, 0, step="mid",
                        where=y_v < 0,  alpha=0.35, color=color_a, label=phase_a)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.5, linestyle=":")
        ax.set_title(
            f"{_clean(feature_names[fi])}\n"
            f"strength={d['weighted_sum']:.3f}",
            fontsize=8, fontweight="bold",
        )
        ax.set_xlabel("z-score vs user mean", fontsize=8)
        ax.set_ylabel("Log-odds", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"EBM Shape Functions — {phase_a}  →  {phase_b}  (top {n_feat} features)\n"
        f"Above 0 → {phase_b}  |  Below 0 → {phase_a}  |"
        f"  Background: dominant phase per interval",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape grid → {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING WITH ELIGIBLE-USER Z-SCORES
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_with_eligible_zscores(
    cfg: dict,
    phase_file: str,
    eligible_file: str,
) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    """Load phase-labeled data, z-scored from each user's full eligible timeline.

    Steps:
      1. Load the eligible-users timeline (no phase labels, full day range).
      2. Compute per-user mean & std for every _mean feature from that full timeline.
      3. Load the phase-labeled file; filter to the eligible users.
      4. Apply the per-user stats from step 2 to produce _zscore columns.
      5. Aggregate to one mean profile per (user, phase).

    Using the full timeline for z-scoring gives a more stable per-user baseline
    than z-scoring from phase-labeled days only.
    """
    import numpy as np

    EPSILON = 1e-8

    # ── Step 1: eligible-user full timeline ───────────────────────────────────
    logging.info(f"[Z-score baseline] Loading eligible-users timeline: {Path(eligible_file).name}")
    df_elig = pd.read_csv(eligible_file, encoding="utf-8-sig", low_memory=False)
    eligible_users = set(df_elig["author"].unique())
    logging.info(f"  Eligible users: {len(eligible_users):,}  rows: {len(df_elig):,}")

    mean_cols = [
        c for c in df_elig.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df_elig[c])
    ]
    logging.info(f"  Features to z-score: {len(mean_cols)}")

    # ── Step 2: per-user mean & std from full timeline ────────────────────────
    user_stats = {}   # author -> {col: (mean, std)}
    for col in mean_cols:
        col_num = pd.to_numeric(df_elig[col], errors="coerce")
        grp = col_num.groupby(df_elig["author"])
        means = grp.mean()
        stds  = grp.std()
        for author in eligible_users:
            if author not in user_stats:
                user_stats[author] = {}
            user_stats[author][col] = (
                float(means.get(author, np.nan)),
                float(stds.get(author, np.nan)),
            )

    # ── Step 3: phase-labeled file, filter to eligible users ──────────────────
    logging.info(f"[Phase labels] Loading: {Path(phase_file).name}")
    df_phase = pd.read_csv(phase_file, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  Rows before filter: {len(df_phase):,}  Users: {df_phase['author'].nunique():,}")
    df_phase = df_phase[df_phase["author"].isin(eligible_users)].copy()
    logging.info(f"  Rows after filter : {len(df_phase):,}  Users: {df_phase['author'].nunique():,}")

    if "phase" not in df_phase.columns:
        raise ValueError("Missing 'phase' column in phase-labeled file.")

    # ── Step 4: apply per-user z-scores ──────────────────────────────────────
    zscore_cols = []
    for col in mean_cols:
        if col not in df_phase.columns:
            continue
        zscore_col = col.replace("_mean", "_zscore")
        col_vals = pd.to_numeric(df_phase[col], errors="coerce").to_numpy(dtype=np.float64)
        authors  = df_phase["author"].to_numpy()

        u_mean = np.array([user_stats[a][col][0] for a in authors], dtype=np.float64)
        u_std  = np.array([user_stats[a][col][1] for a in authors], dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (col_vals - u_mean) / (u_std + EPSILON)
            z = np.where(np.isfinite(z), z, np.nan)

        df_phase[zscore_col] = z
        zscore_cols.append(zscore_col)

    logging.info(f"  Z-score columns produced: {len(zscore_cols)}")

    # ── Step 5: restrict to canonical phases and aggregate ────────────────────
    df_phase = df_phase[df_phase["phase"].isin(PHASE_ORDER)].copy()
    logging.info(f"  Phase counts:\n{df_phase['phase'].value_counts().to_string()}")

    df_profiles = aggregate_to_phase_profiles(df_phase, zscore_cols)
    logging.info(
        f"  Profiles: {len(df_profiles):,} rows  "
        f"({df_profiles['author'].nunique():,} users × up to 4 phases)"
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(PHASE_ORDER)

    return df_profiles, zscore_cols, label_encoder


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument(
        "--phase-file", default=None,
        help="Path to phase-labeled CSV.",
    )
    p.add_argument(
        "--eligible-file", default=None,
        help=(
            "Path to eligible-users timeline CSV (no phase labels, full day range). "
            "When provided, per-user z-scores are computed from this full timeline "
            "before being applied to the phase-labeled rows. "
            "If omitted, z-scores are computed within the phase-labeled file only."
        ),
    )
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--skip-cv", action="store_true",
                   help="Skip cross-validation, go straight to final fit + plots.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg  = load_config(args.config)
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = ROOT / "reports" / "ebm" / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load full phase profiles ──────────────────────────────────────────
    logging.info("[1] Loading data…")
    if args.eligible_file:
        if args.phase_file is None:
            raise ValueError("--phase-file is required when --eligible-file is provided.")
        df, zscore_cols, le = load_dataset_with_eligible_zscores(
            cfg, args.phase_file, args.eligible_file
        )
    else:
        df, zscore_cols, le = load_phase_labeled_dataset(cfg, phase_file=args.phase_file)
    feature_names = zscore_cols
    logging.info(
        f"  Profiles: {len(df):,}  Users: {df['author'].nunique():,}"
        f"  Features: {len(feature_names)}"
    )

    oof_summary_lines = [
        f"Binary EBM Phase-Pair OOF Report  ({timestamp})",
        f"StratifiedGroupKFold n_folds={args.n_folds}  |  sample_weight=balanced",
        "=" * 65,
    ]

    # ── [2] One classifier per pair ───────────────────────────────────────────
    t_global = time.time()
    for phase_a, phase_b in PAIRS:
        pair_label = f"{phase_a}_vs_{phase_b}"
        logging.info(f"\n{'='*60}")
        logging.info(f"  Pair: {phase_a}  →  {phase_b}")
        logging.info(f"{'='*60}")

        # Filter to the two phases only
        mask = df["phase"].isin([phase_a, phase_b])
        df_pair = df[mask].copy()

        # Encode: phase_a=0, phase_b=1
        df_pair["y"] = (df_pair["phase"] == phase_b).astype(int)

        X      = df_pair[feature_names].values.astype(np.float64)
        y      = df_pair["y"].values
        groups = df_pair["author"].values

        n_a = (y == 0).sum()
        n_b = (y == 1).sum()
        logging.info(f"  {phase_a}: {n_a:,}  {phase_b}: {n_b:,}  total: {len(y):,}")

        # Balanced sample weights
        weights = compute_sample_weight("balanced", y)

        # ── Cross-validation ─────────────────────────────────────────────────
        if not args.skip_cv:
            logging.info(f"  GroupKFold CV ({args.n_folds} folds)…")
            oof = run_binary_cv(X, y, groups, weights, seed, n_splits=args.n_folds)

            logging.info(
                f"  OOF AUC={oof['auc']:.4f}  "
                f"Balanced-Acc={oof['balanced_acc']:.4f}  "
                f"Acc={oof['accuracy']:.4f}"
            )
            oof_summary_lines += [
                f"\n{phase_a} vs {phase_b}",
                f"  N: {phase_a}={n_a}  {phase_b}={n_b}",
                f"  OOF AUC            : {oof['auc']:.4f}",
                f"  OOF Balanced-Acc   : {oof['balanced_acc']:.4f}",
                f"  OOF Accuracy       : {oof['accuracy']:.4f}",
            ]
        else:
            logging.info("  CV skipped (--skip-cv).")

        # ── Final fit on full pair data ───────────────────────────────────────
        logging.info("  Fitting final EBM…")
        t0  = time.time()
        ebm = _make_ebm(seed)
        ebm.fit(X, y, sample_weight=weights)
        logging.info(f"  Fitted in {time.time() - t0:.1f}s")

        # ── Feature importance ────────────────────────────────────────────────
        global_exp          = ebm.explain_global()
        raw_scores          = global_exp.data()["scores"]
        feature_importances = np.array(raw_scores, dtype=float)

        top_features_idx = np.argsort(np.abs(feature_importances))[-args.top_n:][::-1]

        print(f"\n  Top {args.top_n} features — {phase_a} vs {phase_b}:")
        for rank, fi in enumerate(top_features_idx, 1):
            sign = "→ " + phase_b if feature_importances[fi] >= 0 else "→ " + phase_a
            print(f"    {rank:2d}. {feature_names[fi]:<55s}  {feature_importances[fi]:>+.4f}  {sign}")

        # ── Extract shapes ────────────────────────────────────────────────────
        shape_data = extract_binary_shapes(ebm, feature_names, args.top_n, top_features_idx)

        # ── Plots ─────────────────────────────────────────────────────────────
        imp_path = output_dir / f"ebm_pair_{pair_label}_importance_{timestamp}.png"
        plot_importance(feature_importances, feature_names, phase_a, phase_b,
                        imp_path, top_n=args.top_n)

        shape_path = output_dir / f"ebm_pair_{pair_label}_shapes_{timestamp}.png"
        plot_shape_grid(shape_data, feature_names, phase_a, phase_b,
                        shape_path, top_n=args.top_n)

    # ── OOF summary file ──────────────────────────────────────────────────────
    if not args.skip_cv:
        summary_path = output_dir / f"ebm_pairs_oof_summary_{timestamp}.txt"
        summary_path.write_text("\n".join(oof_summary_lines))
        logging.info(f"\n  OOF summary → {summary_path.name}")

    logging.info(f"\nAll done. Total time: {time.time() - t_global:.1f}s")
    logging.info(f"Outputs → {output_dir}/")


if __name__ == "__main__":
    main()
