"""
Step 42 — EBM Binary BC Classifier
=====================================
Binary classifier: BC users (label=1) vs non-BC users (label=0).

Input
  Two timelines CSVs with columns:
    author          — user ID
    offset_from_cd1 — cycle day (can be positive or negative)
    <features>      — per-user linguistic features

Design
  • Automatically calculates menstrual phase from offset_from_cd1 for a 28-day cycle:
      M: 0 to 3
      F: 4 to 10
      O: 11 to 13
      L: 14 to 27
  • Unites both files, adding a 'bc' label column (1 = BC, 0 = non-BC).
  • Aggregates day-level rows to one mean profile per (author, phase); columns
    become <feature>_<phase>_mean (e.g. negative_sentiment_menstrual_mean).
    Users with no posts in a phase get NaN for those columns — EBM handles NaN.
  • StratifiedGroupKFold (5 folds, grouped by author) — no user leakage
  • sample_weight="balanced" — corrects class imbalance
  • Pure additive EBM (interactions=0)

Usage
  python scripts/42_ebm_bc_classifier.py \
    --regular-users-file data/interim/regular_timeline.csv \
    --bc-users-file data/interim/bc_timeline.csv \
    --output-run-name run_v1
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ebm_utils import (
    clean_name,
    make_ebm,
    run_cv,
    shape_functions,
    signed_importance,
)

warnings.filterwarnings("ignore", category=UserWarning)

CLASS_NAMES = ["non-BC", "BC"]


# ── PHASES CALCULATION ────────────────────────────────────────────────────────

def calculate_phase(offset: float, period: int = 28,
                    m: int = 4, f: int = 7, o: int = 3) -> str:
    """Maps offset_from_cd1 to phase using a configurable cycle model."""
    if pd.isna(offset):
        return np.nan
    day = int(offset) % period
    if day < m:               return "Menstrual"
    if day < m + f:           return "Follicular"
    if day < m + f + o:       return "Ovulation"
    if day < period:          return "Luteal"
    return np.nan


# ── FEATURE SELECTION ─────────────────────────────────────────────────────────

def select_features_mannwhitney(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n: int,
) -> tuple[np.ndarray, list[str]]:
    """Return (X_subset, names_subset): top-n features by Mann-Whitney rank-biserial."""
    from scipy.stats import mannwhitneyu

    bc_mask, non_mask = y == 1, y == 0

    X_fill = X.copy()
    for j in range(X.shape[1]):
        med = np.nanmedian(X_fill[:, j])
        X_fill[np.isnan(X_fill[:, j]), j] = med

    scores = np.array([
        abs(1 - 2 * mannwhitneyu(X_fill[bc_mask, j], X_fill[non_mask, j],
                                  alternative="two-sided").statistic
            / (bc_mask.sum() * non_mask.sum()))
        for j in range(X.shape[1])
    ])

    top_idx = np.argsort(scores)[::-1][:n]
    selected_names = [feature_names[i] for i in top_idx]
    logging.info(
        f"  Feature pre-selection (mannwhitney): {len(top_idx)} / {len(scores)} features kept.\n"
        f"  Top 10: {selected_names[:10]}"
    )
    return X[:, top_idx], selected_names


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_importance(
    signed_imp: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Signed horizontal bar chart.  Positive (right) = toward BC (class 1)."""
    sorted_idx = np.argsort(np.abs(signed_imp))
    top_idx    = sorted_idx[-top_n:]

    vals   = [signed_imp[i] for i in top_idx]
    labels = [clean_name(feature_names[i]) for i in top_idx]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(top_idx)), vals, color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Density-weighted mean log-odds (signed)")
    ax.set_title(
        f"EBM Feature Importances — non-BC vs BC (top {top_n})\n"
        "Positive (right) → BC  |  Negative (left) → non-BC",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(handles=[
        mpatches.Patch(color="#d62728", label=CLASS_NAMES[1]),
        mpatches.Patch(color="#1f77b4", label=CLASS_NAMES[0]),
    ], fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Importance plot → {output_path.name}")


def plot_shape_grid(
    shapes: dict[str, dict],
    output_path: Path,
    target_class: str,
    base_class: str,
    top_n: int = 10,
) -> None:
    """Binary shape grid: above 0 pushes toward BC, below 0 toward non-BC."""
    feats  = list(shapes.keys())[:top_n]
    n_feat = len(feats)
    ncols  = 2
    nrows  = (n_feat + ncols - 1) // ncols
    color_bc     = "#d62728"
    color_non_bc = "#1f77b4"

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(feats):
        ax    = axes[i]
        d     = shapes[feat][target_class]
        x, y  = d["x"], d["y"]
        valid = ~np.isnan(x)
        xv, yv = x[valid], y[valid]

        for j in range(len(xv) - 1):
            color = color_bc if yv[j] >= 0 else color_non_bc
            ax.axvspan(xv[j], xv[j + 1], alpha=0.12, color=color, linewidth=0)

        ax.step(xv, yv, where="mid", color="black", linewidth=1.8)
        ax.fill_between(xv, yv, 0, step="mid", where=yv >= 0,
                        alpha=0.35, color=color_bc, label=target_class)
        ax.fill_between(xv, yv, 0, step="mid", where=yv < 0,
                        alpha=0.35, color=color_non_bc, label=base_class)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"{clean_name(feat)}\nstrength={d['weighted_sum']:.3f}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("mean value in phase", fontsize=8)
        ax.set_ylabel("Log-odds", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"EBM Shape Functions — {base_class} vs {target_class}\n"
        f"Above 0 → {target_class}  |  Below 0 → {base_class}  |  x=0 → user at personal mean",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Shape grid → {output_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regular-users-file", required=True,
                   help="CSV timeline with regular users data")
    p.add_argument("--bc-users-file", required=True,
                   help="CSV timeline with oral contraceptive users data")
    p.add_argument("--output-run-name", required=True,
                   help="Name of the subdirectory inside reports/ebm/bc/ to save results")
    p.add_argument("--pill-type", default="all", choices=["all", "combined", "mini"],
                   help="Filter BC users by pill type (default: all)")
    p.add_argument("--feature-strategy", default="phase_pivot",
                   choices=["phase_pivot", "all_four_phases_present"],
                   help=(
                       "How to build the user feature matrix:\n"
                       "  phase_pivot             — mean per (user, phase), pivot wide; NaN where phase missing (default)\n"
                       "  all_four_phases_present — phase_pivot but drop users missing any phase\n"
                   ))
    p.add_argument("--top-n",   type=int, default=15)
    p.add_argument("--feature-select-n", type=int, default=None,
                   help="Pre-select top N features before CV (Mann-Whitney U test).")
    p.add_argument("--learning-rate",     type=float, default=0.01)
    p.add_argument("--max-rounds",        type=int,   default=5000)
    p.add_argument("--max-bins",          type=int,   default=256)
    p.add_argument("--min-samples-leaf",  type=int,   default=2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--stable-refit-min-folds", type=int, default=None,
                   help="After initial CV, refit using only features that appear in top-15 "
                        "in at least this many folds (e.g. 4). Reports second AUC separately.")
    p.add_argument("--window-days", type=int, default=None,
                   help="Restrict both files to offset_from_cd1 ∈ [-N, N] before feature building.")
    p.add_argument("--permutation-test", action="store_true",
                   help="Run permutation test after CV to estimate empirical p-value.")
    p.add_argument("--n-permutations", type=int, default=100,
                   help="Number of label permutations (default: 100).")
    p.add_argument("--perm-n-folds", type=int, default=3,
                   help="Folds to use per permutation for speed (default: 3).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = ROOT / "reports" / "ebm" / "bc" / args.output_run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = args.seed
    np.random.seed(seed)

    # ── [1] Load & Merge ─────────────────────────────────────────────────────
    logging.info("[1] Loading regular and BC timeline files…")

    for path in (args.regular_users_file, args.bc_users_file):
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    df_reg = pd.read_csv(args.regular_users_file, encoding="utf-8-sig", low_memory=False)
    df_bc  = pd.read_csv(args.bc_users_file,      encoding="utf-8-sig", low_memory=False)

    required = {"author", "offset_from_cd1"}
    for label, df in (("regular", df_reg), ("bc", df_bc)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} file is missing required columns: {sorted(missing)}")

    non_feat = {"author", "offset_from_cd1", "phase", "bc", "pill_type", "bc_status"}
    reg_feats = {c for c in df_reg.columns if c not in non_feat and not c.startswith("Unnamed:")}
    bc_feats  = {c for c in df_bc.columns  if c not in non_feat and not c.startswith("Unnamed:")}
    only_reg = reg_feats - bc_feats
    only_bc  = bc_feats  - reg_feats
    if only_reg or only_bc:
        raise ValueError(
            f"Feature columns don't match between the two files.\n"
            f"  Only in regular file ({len(only_reg)}): {sorted(only_reg)[:5]}{'...' if len(only_reg)>5 else ''}\n"
            f"  Only in BC file ({len(only_bc)}): {sorted(only_bc)[:5]}{'...' if len(only_bc)>5 else ''}"
        )

    if args.pill_type != "all":
        if "pill_type" not in df_bc.columns:
            raise ValueError(f"--pill-type '{args.pill_type}' requested but 'pill_type' column not found in BC file.")
        before = df_bc["author"].nunique()
        df_bc = df_bc[df_bc["pill_type"] == args.pill_type]
        after = df_bc["author"].nunique()
        logging.info(f"  Filtered BC to pill_type='{args.pill_type}': {before} → {after} users")
        if df_bc.empty:
            raise ValueError(f"No BC users remain after filtering to pill_type='{args.pill_type}'.")

    # Remove ALL authors in the BC file from the regular file regardless of pill-type filter,
    # so no LLM-verified BC user can appear in the non-BC group.
    all_bc_authors = set(pd.read_csv(args.bc_users_file, encoding="utf-8-sig",
                                     usecols=["author"], low_memory=False)["author"].unique())
    overlap = set(df_reg["author"].unique()) & all_bc_authors
    if overlap:
        logging.info(f"  Removing {len(overlap):,} LLM-verified BC users from regular file.")
        df_reg = df_reg[~df_reg["author"].isin(all_bc_authors)]

    if args.window_days is not None:
        w = args.window_days
        before_reg = df_reg["author"].nunique()
        before_bc  = df_bc["author"].nunique()
        df_reg = df_reg[df_reg["offset_from_cd1"].between(-w, w)]
        df_bc  = df_bc[df_bc["offset_from_cd1"].between(-w, w)]
        logging.info(f"  Window ±{w} days: regular {before_reg:,} → {df_reg['author'].nunique():,} users  |  BC {before_bc:,} → {df_bc['author'].nunique():,} users")
        if df_reg.empty or df_bc.empty:
            raise ValueError(f"No users remain after applying --window-days {w}.")

    df_reg["bc"] = 0
    df_bc["bc"]  = 1

    df = pd.concat([df_reg, df_bc], ignore_index=True)
    logging.info(f"  Merged size: {len(df):,} rows, {df['author'].nunique():,} unique users")

    if "offset_from_cd1" not in df.columns:
        raise ValueError("Required column 'offset_from_cd1' not found in input datasets.")

    logging.info("  Calculating menstrual cycle phases...")
    df["phase"] = df["offset_from_cd1"].apply(calculate_phase)

    invalid_phases = df["phase"].isna().sum()
    if invalid_phases > 0:
        logging.warning(f"  {invalid_phases:,} timeline records couldn't be mapped to a phase.")

    combined_csv_path = out_dir / f"combined_labeled_phased_timeline_{ts}.csv"
    df.to_csv(combined_csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"  Saved combined timeline data → {combined_csv_path.name}")

    non_feature_cols = {"author", "offset_from_cd1", "phase", "bc"}
    possible_features = [c for c in df.columns if c not in non_feature_cols and not c.startswith("Unnamed:")]
    numeric_features = df[possible_features].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric linguistic features found in the input datasets.")

    # ── [2] Build user feature matrix ────────────────────────────────────────
    strategy = args.feature_strategy
    logging.info(f"[2] Building feature matrix (strategy: {strategy})…")

    bc_labels = df.groupby("author")["bc"].first().astype(int).reset_index()

    profiles = df.groupby(["author", "phase"], dropna=True)[numeric_features].mean().reset_index()
    profiles_wide = profiles.pivot(index="author", columns="phase", values=numeric_features)
    profiles_wide.columns = [f"{feat}_{phase}" for feat, phase in profiles_wide.columns]
    wide = profiles_wide.reset_index()
    feature_names = [c for c in wide.columns if c != "author"]

    if strategy == "all_four_phases_present":
        all_phases = {"Menstrual", "Follicular", "Ovulation", "Luteal"}
        user_phases = df.groupby("author")["phase"].apply(set)
        users_all4 = set(user_phases[user_phases.apply(lambda s: all_phases.issubset(s))].index)
        wide = wide[wide["author"].isin(users_all4)]
        merged_tmp = wide.merge(bc_labels, on="author")
        n_bc_kept  = (merged_tmp["bc"] == 1).sum()
        n_non_kept = (merged_tmp["bc"] == 0).sum()
        logging.info(f"  all_four_phases_present: kept {n_bc_kept} BC and {n_non_kept} non-BC users with all 4 phases")

    df_users = wide.merge(bc_labels, on="author", how="left")
    df_users = df_users[df_users["bc"].notna()].copy()
    df_users["bc"] = df_users["bc"].astype(int)

    feature_names = [c for c in df_users.columns if c not in ("author", "bc")]
    X      = df_users[feature_names].values.astype(np.float64)
    y      = df_users["bc"].values
    groups = df_users["author"].values

    if args.feature_select_n and args.feature_select_n < len(feature_names):
        X, feature_names = select_features_mannwhitney(X, y, feature_names, args.feature_select_n)

    n_bc, n_non = (y == 1).sum(), (y == 0).sum()
    nan_frac_bc  = np.isnan(X[y==1]).mean()
    nan_frac_non = np.isnan(X[y==0]).mean()
    logging.info(
        f"  Total users={len(df_users):,} | BC={n_bc:,} | non-BC={n_non:,}\n"
        f"  Features={X.shape[1]}  |  NaN fraction: BC={nan_frac_bc:.1%}  non-BC={nan_frac_non:.1%}"
    )

    # ── [3] Cross-validation ──────────────────────────────────────────────────
    fold_counts: dict[int, int] = {}
    if not args.skip_cv:
        logging.info(f"\n[3] GroupKFold CV ({args.n_folds} folds)…")

        track_folds = args.stable_refit_min_folds is not None
        cv_kwargs = dict(
            seed=seed, n_splits=args.n_folds,
            learning_rate=args.learning_rate,
            max_rounds=args.max_rounds,
            max_bins=args.max_bins,
            min_samples_leaf=args.min_samples_leaf,
        )
        if track_folds:
            cv_kwargs["top_n_per_fold"] = 15

        cv_result = run_cv(X, y, groups, n_classes=2, **cv_kwargs)
        if track_folds:
            _, y_proba, fold_counts, fold_aucs = cv_result
        else:
            _, y_proba, fold_aucs = cv_result

        auc = roc_auc_score(y, y_proba[:, 1])
        logging.info(f"  OOF AUC={auc:.4f}")

        n_folds_all = args.n_folds
        all_stable = {i for i, cnt in fold_counts.items() if cnt == n_folds_all}
        print(f"\n=== Features in top-15 in ALL {n_folds_all} folds "
              f"({len(all_stable)} stable features) ===")
        if all_stable:
            for i in sorted(all_stable, key=lambda i: fold_counts[i], reverse=True):
                print(f"  {feature_names[i]}")
        else:
            print("  None — no feature appeared in top-15 across all folds.")

        if fold_counts:
            print(f"\n=== Fold appearance counts (top-15, all features) ===")
            print(f"  {'Feature':<60} folds")
            for i, cnt in sorted(fold_counts.items(), key=lambda x: -x[1]):
                marker = " ★" if cnt == n_folds_all else ""
                print(f"  {feature_names[i]:<60} {cnt}/{n_folds_all}{marker}")

        stable_lines = "\n".join(f"  {feature_names[i]}" for i in all_stable)
        valid_fold_aucs = [a for a in fold_aucs if not np.isnan(a)]
        fold_auc_lines = "\n".join(
            f"  Fold {i + 1}: {a:.4f}" if not np.isnan(a) else f"  Fold {i + 1}: N/A"
            for i, a in enumerate(fold_aucs)
        )
        fold_auc_summary = (
            f"  Mean:   {np.mean(valid_fold_aucs):.4f}  Std: {np.std(valid_fold_aucs):.4f}"
            if valid_fold_aucs else "  Mean:   N/A"
        )
        oof_text = (
            f"EBM BC Classifier Run — OOF Report ({ts})\n"
            f"Directory Run Name    : {args.output_run_name}\n"
            f"Feature strategy     : {strategy}\n"
            f"StratifiedGroupKFold  n_folds={args.n_folds}  "
            f"interactions=0  max_bins=256  sample_weight=balanced\n"
            f"{'='*60}\n"
            f"AUC (ROC)    : {auc:.4f}\n"
            f"N(BC)        : {n_bc}\n"
            f"N(non-BC)    : {n_non}\n"
            f"\nPer-fold AUC:\n"
            f"{fold_auc_lines}\n"
            f"{fold_auc_summary}\n"
            f"\nStable features (top-15 in all {args.n_folds} folds):\n"
            + stable_lines
        )
        (out_dir / f"ebm_bc_oof_report_{ts}.txt").write_text(oof_text)
        logging.info("  OOF report saved.")
    else:
        logging.info("\n[3] CV skipped (--skip-cv).")

    # ── [3b] Identify stable features for display filter ─────────────────────
    stable_display_names: list[str] = []
    if args.stable_refit_min_folds is not None and not args.skip_cv and fold_counts:
        min_f = args.stable_refit_min_folds
        stable_idx_sorted = sorted(
            [i for i, cnt in fold_counts.items() if cnt >= min_f],
            key=lambda i: fold_counts[i], reverse=True
        )
        if stable_idx_sorted:
            stable_display_names = [feature_names[i] for i in stable_idx_sorted]
            logging.info(
                f"\n[3b] Stable features for display: {len(stable_display_names)} "
                f"(top-15 in ≥{min_f}/{args.n_folds} folds):"
            )
            for fn in stable_display_names:
                logging.info(f"  {fn}")
        else:
            logging.warning(
                f"  No features appeared in top-15 in ≥{min_f}/{args.n_folds} folds."
            )

    # ── [3c] Permutation test ─────────────────────────────────────────────────
    if args.permutation_test and not args.skip_cv:
        logging.info(f"\n[3c] Permutation test ({args.n_permutations} permutations, "
                     f"{args.perm_n_folds} folds each)…")
        rng = np.random.default_rng(seed)
        perm_aucs = []
        for i in range(args.n_permutations):
            y_perm = rng.permutation(y)
            _, proba_perm, _ = run_cv(X, y_perm, groups, n_classes=2,
                                     seed=seed, n_splits=args.perm_n_folds)
            perm_aucs.append(roc_auc_score(y_perm, proba_perm[:, 1]))
            if (i + 1) % 10 == 0:
                logging.info(f"  {i+1}/{args.n_permutations}  mean_perm_AUC={np.mean(perm_aucs):.4f}")

        perm_aucs = np.array(perm_aucs)
        p_value = (perm_aucs >= auc).mean()
        logging.info(f"  Real AUC={auc:.4f}  |  Permuted mean={perm_aucs.mean():.4f} "
                     f"± {perm_aucs.std():.4f}  |  p={p_value:.4f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(perm_aucs, bins=30, color="#1f77b4", alpha=0.7, label="Permuted AUCs")
        ax.axvline(auc, color="#d62728", linewidth=2, label=f"Real AUC={auc:.4f}")
        ax.axvline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5, label="Chance (0.5)")
        ax.set_xlabel("AUC")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation Test (n={args.n_permutations})  |  p={p_value:.4f}", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        perm_plot_path = out_dir / f"ebm_bc_permutation_{ts}.png"
        fig.savefig(perm_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"  Permutation plot → {perm_plot_path.name}")

        perm_text = (f"\nPermutation test ({args.n_permutations} permutations, "
                     f"{args.perm_n_folds} folds)\n"
                     f"  Permuted AUC: mean={perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}\n"
                     f"  p-value      : {p_value:.4f}\n")
        report_path = out_dir / f"ebm_bc_oof_report_{ts}.txt"
        if report_path.exists():
            with open(report_path, "a") as f:
                f.write(perm_text)

    # ── [4] Final fit ─────────────────────────────────────────────────────────
    logging.info("\n[4] Final fit on full balanced user data…")
    ebm = make_ebm(seed, interactions=0,
                   learning_rate=args.learning_rate,
                   max_rounds=args.max_rounds,
                   max_bins=args.max_bins,
                   min_samples_leaf=args.min_samples_leaf)
    ebm.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    logging.info("  Fitted.")

    # ── [5] Signed importance ─────────────────────────────────────────────────
    logging.info("\n[5] Computing signed importance features…")
    sim = signed_importance(ebm, feature_names)

    if stable_display_names:
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        display_idx = sorted(
            [name_to_idx[n] for n in stable_display_names if n in name_to_idx],
            key=lambda i: abs(sim[i]), reverse=True,
        )
        label = f"stable (≥{args.stable_refit_min_folds}/{args.n_folds} folds)"
        imp_sim   = sim[np.array(display_idx)]
        imp_names = [feature_names[i] for i in display_idx]
        imp_top_n = len(display_idx)
    else:
        display_idx = list(np.argsort(np.abs(sim))[-args.top_n:][::-1])
        label = f"top {args.top_n}"
        imp_sim, imp_names, imp_top_n = sim, feature_names, args.top_n

    print(f"\n{label} features — non-BC vs BC:")
    for rank, fi in enumerate(display_idx, 1):
        direction = f"→ {CLASS_NAMES[1]}" if sim[fi] >= 0 else f"→ {CLASS_NAMES[0]}"
        print(f"  {rank:2d}. {feature_names[fi]:<60s}  {sim[fi]:>+.4f}  {direction}")

    plot_importance(imp_sim, imp_names,
                    out_dir / f"ebm_bc_importance_{ts}.png",
                    top_n=imp_top_n)

    # ── [6] Shape grid ────────────────────────────────────────────────────────
    logging.info("\n[6] Rendering shape functions grid…")
    top_feat_names = [feature_names[fi] for fi in display_idx]
    shapes_all = shape_functions(ebm, feature_names, CLASS_NAMES)
    shapes = {feat: shapes_all[feat] for feat in top_feat_names}

    plot_shape_grid(shapes,
                    out_dir / f"ebm_bc_shapes_{ts}.png",
                    target_class=CLASS_NAMES[1],
                    base_class=CLASS_NAMES[0],
                    top_n=len(top_feat_names))

    logging.info(f"\nAll outputs successfully saved to folder → {out_dir}/")


if __name__ == "__main__":
    main()
