"""
Step 31 — EBM Split-Half Stability Analysis
============================================
Validates stability of EBM phase analysis findings by randomly splitting users
into two equal halves and running the full EBM pipeline on each independently.

A finding is considered stable when:
  - The same features appear in the top-15 of both halves (high top-15 overlap)
  - Importance rankings correlate (high Spearman ρ across all features)
  - Median shifts have the same direction in both halves
  - Tail enrichments are consistent (significant in both or neither)

Reuses the EBM hyperparameters, gauntlet, and importance logic from script 27.

Usage
  python scripts/31_ebm_split_half_stability.py --config configs/base.yaml
  python scripts/31_ebm_split_half_stability.py --config configs/base.yaml --seed 123
  python scripts/31_ebm_split_half_stability.py --config configs/base.yaml --n-seeds 10

Output (reports/ebm_stability/)
  ebm_stability_importance_<ts>.csv    — importance per feature × phase × half (canonical seed)
  ebm_stability_gauntlet_<ts>.csv      — gauntlet comparison table (union of top-N, canonical seed)
  ebm_stability_comparison_<ts>.md     — human-readable stability report with multi-seed stats
"""

import argparse
import importlib.util
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy import stats as scipy_stats
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Import reusable functions from script 27 ──────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "ebm_phase_analysis", ROOT / "scripts" / "27_ebm_phase_analysis.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_group_kfold_cv        = _mod.run_group_kfold_cv
extract_global_importance = _mod.extract_global_importance

from src.config import load_config
from src.constants import CLASSIC_Z_HIGH, CLASSIC_Z_LOW, PHASE_ORDER
from src.ml_data import load_phase_labeled_dataset
from src.ml_eval import run_phase_statistical_gauntlet

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# BALANCE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_balance_table(
    profiles_h1: pd.DataFrame,
    profiles_h2: pd.DataFrame,
    phases: list[str],
) -> str:
    """Compute and return a balance diagnostic table for the two halves.

    Checks per-phase profile counts, mean profiles per user, and mean
    consensus period if a cycle-length column is present. Returns a
    plain-text table suitable for logging and markdown embedding.
    """
    cycle_col: Optional[str] = None
    for candidate in ("consensus_period", "cycle_length", "period_length"):
        if candidate in profiles_h1.columns:
            cycle_col = candidate
            break

    rows = []
    for label, df in [("H1", profiles_h1), ("H2", profiles_h2)]:
        n_users = df["author"].nunique()
        n_profiles = len(df)
        mean_per_user = n_profiles / n_users if n_users > 0 else float("nan")
        phase_counts = {ph: int((df["phase"] == ph).sum()) for ph in phases}

        if cycle_col is not None:
            mean_cycle = df.groupby("author")[cycle_col].mean().mean()
            cycle_str = f"{mean_cycle:.1f}"
        else:
            cycle_str = "n/a"

        row = {
            "Half": label,
            "Users": n_users,
            "Profiles": n_profiles,
            "Mean profiles/user": f"{mean_per_user:.1f}",
            "Mean cycle (days)": cycle_str,
        }
        for ph in phases:
            row[f"{ph} profiles"] = phase_counts.get(ph, 0)
        rows.append(row)

    df_bal = pd.DataFrame(rows)
    col_order = ["Half", "Users", "Profiles", "Mean profiles/user", "Mean cycle (days)"] + [
        f"{ph} profiles" for ph in phases
    ]
    df_bal = df_bal[col_order]

    # Build fixed-width plain-text table
    col_widths = [max(len(c), max(len(str(df_bal.iloc[r][c])) for r in range(len(df_bal))))
                  for c in df_bal.columns]
    sep = "  ".join("-" * w for w in col_widths)
    header = "  ".join(c.ljust(w) for c, w in zip(df_bal.columns, col_widths))
    table_lines = [header, sep]
    for _, row in df_bal.iterrows():
        table_lines.append("  ".join(str(row[c]).ljust(w) for c, w in zip(df_bal.columns, col_widths)))

    note = "" if cycle_col is not None else "\n(cycle length column not found; omitted from table)"
    return "\n".join(table_lines) + note


# ═══════════════════════════════════════════════════════════════════════════════
# HALF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def fit_half(
    profiles: pd.DataFrame,
    zscore_cols: list[str],
    le: LabelEncoder,
    n_folds: int,
    label: str,
    top_n: int = 15,
) -> dict:
    """Fit EBM + run gauntlet on a subset of users (one split half).

    Returns a dict with:
      df_imp     : feature importance DataFrame (all features × phases)
      gauntlet   : gauntlet DataFrame (top-N features × phases)
      phase_aucs : {phase: OOF AUC}
      macro_auc  : OOF macro AUC
      n_users    : number of users in this half
    """
    y      = le.transform(profiles["phase"])
    X      = profiles[zscore_cols].values.astype(np.float64)
    groups = profiles["author"].values

    logging.info(f"\n  [{label}] users={len(np.unique(groups)):,}  profiles={len(profiles):,}")

    # ── CV for AUC ────────────────────────────────────────────────────────────
    logging.info(f"  [{label}] GroupKFold CV ({n_folds} folds)…")
    _y_pred, y_proba = run_group_kfold_cv(X, y, groups, le, n_splits=n_folds)
    macro_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    phase_aucs = {}
    for i, cls in enumerate(le.classes_):
        y_bin = (y == i).astype(int)
        phase_aucs[cls] = roc_auc_score(y_bin, y_proba[:, i])
    logging.info(f"  [{label}] Macro AUC: {macro_auc:.4f}  "
                 + "  ".join(f"{cls}={auc:.3f}" for cls, auc in phase_aucs.items()))

    # ── Final EBM fit ─────────────────────────────────────────────────────────
    logging.info(f"  [{label}] Fitting final EBM…")
    ebm = ExplainableBoostingClassifier(
        max_bins=256, max_interaction_bins=64, interactions=0,
        learning_rate=0.01, max_rounds=5000, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    ebm.fit(X, y)

    # ── Importance + gauntlet ─────────────────────────────────────────────────
    df_imp = extract_global_importance(ebm, zscore_cols, le)

    gauntlet_results = []
    for i, phase in enumerate(le.classes_):
        y_binary = (y == i).astype(int)
        top_feats = (
            df_imp.sort_values(phase, ascending=False)["feature"].head(top_n).tolist()
        )
        top_indices = [zscore_cols.index(f) for f in top_feats if f in zscore_cols]
        g_df = run_phase_statistical_gauntlet(X, y_binary, list(zscore_cols), top_indices)
        g_df.insert(0, "phase", phase)
        gauntlet_results.append(g_df)

    return {
        "df_imp":    df_imp,
        "gauntlet":  pd.concat(gauntlet_results, ignore_index=True),
        "phase_aucs": phase_aucs,
        "macro_auc": macro_auc,
        "n_users":   int(len(np.unique(groups))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_halves(
    r1: dict,
    r2: dict,
    le: LabelEncoder,
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build importance-ranking and gauntlet comparison tables.

    Computes per-phase:
      - Spearman rho across all features
      - Top-N overlap count
      - Hypergeometric p-value: P(overlap >= observed | random selection)
      - Direction agreement fraction for features in both top-N sets

    Returns:
      imp_df      : feature × phase importance comparison (all features)
      gauntlet_df : gauntlet comparison for the union of top-N features per phase,
                    with a 'replicated' column (in both halves AND direction concordant)
    """
    imp_rows      = []
    gauntlet_rows = []

    # Total feature pool N comes from the actual feature list in the importance table.
    n_total_features = len(r1["df_imp"])

    for phase in le.classes_:
        imp1 = r1["df_imp"].set_index("feature")[phase]
        imp2 = r2["df_imp"].set_index("feature")[phase]

        rank1 = imp1.rank(ascending=False, method="min").astype(int)
        rank2 = imp2.rank(ascending=False, method="min").astype(int)

        rho, rho_p = scipy_stats.spearmanr(imp1.values, imp2.values)

        top_h1 = set(imp1.nlargest(top_n).index)
        top_h2 = set(imp2.nlargest(top_n).index)
        overlap_count = len(top_h1 & top_h2)

        # ── Hypergeometric p-value: P(X >= overlap_count) ────────────────────
        # N = total feature pool, K = top_n selected in H1, n = top_n in H2
        N = n_total_features
        K = top_n
        n = top_n
        k = overlap_count
        p_hypergeom = float(hypergeom.sf(k - 1, N, K, n))

        # ── Direction agreement for in-both features (for summary stat) ───────
        g1 = r1["gauntlet"][r1["gauntlet"]["phase"] == phase].set_index("feature")
        g2 = r2["gauntlet"][r2["gauntlet"]["phase"] == phase].set_index("feature")

        in_both = top_h1 & top_h2
        dir_agrees = []
        for feat in in_both:
            ms1 = float(g1.loc[feat, "global_median_shift"]) if feat in g1.index else float("nan")
            ms2 = float(g2.loc[feat, "global_median_shift"]) if feat in g2.index else float("nan")
            if _both_valid(ms1, ms2):
                dir_agrees.append(bool(np.sign(ms1) == np.sign(ms2)))
        direction_agreement_frac = float(np.mean(dir_agrees)) if dir_agrees else float("nan")

        # ── Importance table (all features) ───────────────────────────────────
        for feat in imp1.index:
            imp_rows.append({
                "phase":                    phase,
                "feature":                  feat,
                "imp_h1":                   float(imp1[feat]),
                "rank_h1":                  int(rank1[feat]),
                "imp_h2":                   float(imp2[feat]),
                "rank_h2":                  int(rank2[feat]),
                "rank_diff":                int(rank1[feat]) - int(rank2[feat]),
                "in_top_h1":                feat in top_h1,
                "in_top_h2":                feat in top_h2,
                "in_both":                  feat in top_h1 and feat in top_h2,
                "spearman_rho":             round(rho, 4),
                "spearman_p":               round(rho_p, 4),
                "top_overlap":              overlap_count,
                "p_hypergeom":              round(p_hypergeom, 6),
                "direction_agreement_frac": round(direction_agreement_frac, 4)
                                            if not np.isnan(direction_agreement_frac) else float("nan"),
            })

        # ── Gauntlet comparison (union of top-N) ─────────────────────────────
        union_feats = top_h1 | top_h2

        for feat in union_feats:
            def _get(g, col):
                return float(g.loc[feat, col]) if feat in g.index else float("nan")

            ms1  = _get(g1, "global_median_shift")
            ms2  = _get(g2, "global_median_shift")
            mn1  = _get(g1, "global_mean_shift")
            mn2  = _get(g2, "global_mean_shift")
            gp1  = _get(g1, "global_p_fdr")
            gp2  = _get(g2, "global_p_fdr")
            hi1  = _get(g1, "high_tail_p_fdr")
            hi2  = _get(g2, "high_tail_p_fdr")
            lo1  = _get(g1, "low_tail_p_fdr")
            lo2  = _get(g2, "low_tail_p_fdr")

            dir_agree   = bool(np.sign(ms1) == np.sign(ms2)) if _both_valid(ms1, ms2) else False
            sig_agree   = bool((gp1 < 0.05) == (gp2 < 0.05)) if _both_valid(gp1, gp2) else False
            hi_agree    = bool((hi1 < 0.05) == (hi2 < 0.05)) if _both_valid(hi1, hi2) else False
            lo_agree    = bool((lo1 < 0.05) == (lo2 < 0.05)) if _both_valid(lo1, lo2) else False

            # Change 4: replicated = in both top-N AND direction concordant
            in_both_feat = feat in top_h1 and feat in top_h2
            replicated   = bool(in_both_feat and dir_agree)

            gauntlet_rows.append({
                "phase":          phase,
                "feature":        feat,
                "in_top_h1":      feat in top_h1,
                "in_top_h2":      feat in top_h2,
                "in_both":        in_both_feat,
                "replicated":     replicated,
                "rank_h1":        int(rank1.get(feat, -1)),
                "rank_h2":        int(rank2.get(feat, -1)),
                "imp_h1":         float(imp1.get(feat, float("nan"))),
                "imp_h2":         float(imp2.get(feat, float("nan"))),
                "median_shift_h1": ms1,
                "median_shift_h2": ms2,
                "mean_shift_h1":   mn1,
                "mean_shift_h2":   mn2,
                "global_p_h1":     gp1,
                "global_p_h2":     gp2,
                "high_tail_p_h1":  hi1,
                "high_tail_p_h2":  hi2,
                "low_tail_p_h1":   lo1,
                "low_tail_p_h2":   lo2,
                "direction_agree": dir_agree,
                "sig_agree":       sig_agree,
                "high_tail_agree": hi_agree,
                "low_tail_agree":  lo_agree,
            })

    return pd.DataFrame(imp_rows), pd.DataFrame(gauntlet_rows)


def _both_valid(a, b) -> bool:
    return a == a and b == b  # NaN check


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def build_markdown(
    imp_df: pd.DataFrame,
    gauntlet_df: pd.DataFrame,
    r1: dict,
    r2: dict,
    le: LabelEncoder,
    seed: int,
    top_n: int,
    balance_table: str = "",
    seed_stats: Optional[dict] = None,
) -> str:
    """Build the full markdown stability report.

    Parameters
    ----------
    imp_df, gauntlet_df : canonical-seed comparison tables.
    r1, r2              : fit_half results for the canonical seed.
    le                  : fitted LabelEncoder.
    seed                : canonical seed used for feature lists.
    top_n               : top-N threshold.
    balance_table       : pre-formatted balance diagnostic text (plain).
    seed_stats          : dict keyed by phase with keys
                          {mean_rho, sd_rho, mean_overlap, sd_overlap,
                           mean_p_hypergeom, sd_p_hypergeom,
                           mean_dir_agree, sd_dir_agree}
                          populated when --n-seeds > 1. None for single-seed runs.
    """
    lines = ["# EBM Split-Half Stability Report", ""]

    n_seeds_str = (
        f"Seeds: 0–{len(list(seed_stats.values())[0].get('raw_rho', [None])) - 1}"
        if seed_stats is not None else f"Seed: {seed}"
    )
    lines += [f"{n_seeds_str}  |  Canonical seed: {seed}  |  Top-N: {top_n}", ""]

    # ── [A] Summary section ───────────────────────────────────────────────────
    lines += ["## Summary", ""]

    # Determine stable vs exploratory using mean stats (or single-seed stats)
    stable_phases = []
    exploratory_phases = []
    phase_summary_stats: dict[str, dict] = {}

    for phase in le.classes_:
        phase_imp = imp_df[imp_df["phase"] == phase].iloc[0]
        if seed_stats is not None and phase in seed_stats:
            mean_rho   = seed_stats[phase]["mean_rho"]
            mean_p     = seed_stats[phase]["mean_p_hypergeom"]
        else:
            mean_rho = float(phase_imp["spearman_rho"])
            mean_p   = float(phase_imp["p_hypergeom"])

        phase_summary_stats[phase] = {"mean_rho": mean_rho, "mean_p": mean_p}

        if mean_p < 0.05 and mean_rho > 0.45:
            stable_phases.append(phase)
        else:
            exploratory_phases.append(phase)

    if stable_phases:
        lines.append(f"**Stable phases** (hypergeometric p < 0.05 AND Spearman rho > 0.45): "
                     f"{', '.join(stable_phases)}")
        for phase in stable_phases:
            n_rep = int(gauntlet_df[
                (gauntlet_df["phase"] == phase) & gauntlet_df["replicated"]
            ].shape[0])
            lines.append(f"  - {phase}: {n_rep} replicated feature(s) in top-{top_n}")
    else:
        lines.append("No phases meet the stability threshold (p < 0.05 AND rho > 0.45).")

    if exploratory_phases:
        lines.append(f"\n**Exploratory phases** (below stability threshold): "
                     f"{', '.join(exploratory_phases)}")
        for phase in exploratory_phases:
            ps = phase_summary_stats[phase]
            if phase == "Follicular" or ps["mean_p"] >= 0.05:
                lines.append(
                    f"  - {phase}: Feature overlap not significantly above chance expectation "
                    f"(hypergeometric p = {ps['mean_p']:.4f}). "
                    f"Findings for this phase are treated as exploratory."
                )

    lines.append("")

    # ── [B] Multi-seed or single-seed statistics table ────────────────────────
    if seed_stats is not None:
        # Change 5: Spearman rho leads
        lines += [
            "## Stability Statistics (mean ± SD across seeds)", "",
            "| Phase | Spearman rho | Overlap / N | Hypergeom p | Dir-agree frac |",
            "|---|---|---|---|---|",
        ]
        for phase in le.classes_:
            st = seed_stats[phase]
            lines.append(
                f"| {phase}"
                f" | {st['mean_rho']:.3f} ± {st['sd_rho']:.3f}"
                f" | {st['mean_overlap']:.1f} ± {st['sd_overlap']:.1f} / {top_n}"
                f" | {st['mean_p_hypergeom']:.4f} ± {st['sd_p_hypergeom']:.4f}"
                f" | {st['mean_dir_agree']:.2f} ± {st['sd_dir_agree']:.2f} |"
            )
        lines.append("")

    # ── [C] Canonical-seed AUC + overlap table ────────────────────────────────
    # Change 5: Spearman rho leads in this table too
    lines += [
        f"## Canonical Run (seed={seed})", "",
        "| Phase | Spearman rho | Top-N overlap | Hypergeom p | AUC H1 | AUC H2 |",
        "|---|---|---|---|---|---|",
    ]

    for phase in le.classes_:
        phase_imp = imp_df[imp_df["phase"] == phase].iloc[0]
        rho     = phase_imp["spearman_rho"]
        overlap = phase_imp["top_overlap"]
        p_hg    = phase_imp["p_hypergeom"]
        auc1    = r1["phase_aucs"].get(phase, float("nan"))
        auc2    = r2["phase_aucs"].get(phase, float("nan"))
        lines.append(
            f"| {phase} | {rho:.3f} | {overlap}/{top_n} | {p_hg:.4f} | {auc1:.3f} | {auc2:.3f} |"
        )

    lines += [
        f"| **Macro** | | | | **{r1['macro_auc']:.3f}** | **{r2['macro_auc']:.3f}** |",
        "",
        f"H1 users: {r1['n_users']:,}  |  H2 users: {r2['n_users']:,}", "",
    ]

    # ── [D] Half balance check ─────────────────────────────────────────────────
    if balance_table:
        lines += ["## Half Balance Check", "", "```", balance_table, "```", ""]

    # ── [E] Per-phase detailed gauntlet tables ────────────────────────────────
    for phase in le.classes_:
        g_phase = gauntlet_df[gauntlet_df["phase"] == phase].copy()
        g_phase = g_phase.sort_values(
            ["in_both", "rank_h1"], ascending=[False, True]
        ).reset_index(drop=True)

        lines += [
            f"## Phase: {phase}", "",
            "Features from the union of both halves' top-15, ranked by H1 importance. "
            "✓ = same direction / same significance in both halves. "
            "**Replicated** = in both top-N and direction concordant.", "",
            "| Feature | Spearman rho | Rank H1 | Rank H2 | Imp H1 | Imp H2 "
            "| Med Δ H1 | Med Δ H2 | Dir? | Sig? | Hi-tail? | Lo-tail? | In Both? | Replicated |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        # Grab phase-level spearman rho once (same for all rows in phase)
        phase_rho = float(imp_df[imp_df["phase"] == phase].iloc[0]["spearman_rho"])

        for _, row in g_phase.iterrows():
            feat = row["feature"].replace("_zscore", "").replace("_", " ")
            r1_s = str(row["rank_h1"]) if row["rank_h1"] != -1 else "—"
            r2_s = str(row["rank_h2"]) if row["rank_h2"] != -1 else "—"

            def _ms(v):
                return f"{v:+.3f}" if v == v else "—"

            def _tick(v):
                return "✓" if v else "✗"

            rep_cell = "**✓**" if row["replicated"] else "✗"

            lines.append(
                f"| {feat}"
                f" | {phase_rho:.3f}"
                f" | {r1_s} | {r2_s}"
                f" | {row['imp_h1']:.4f} | {row['imp_h2']:.4f}"
                f" | {_ms(row['median_shift_h1'])} | {_ms(row['median_shift_h2'])}"
                f" | {_tick(row['direction_agree'])}"
                f" | {_tick(row['sig_agree'])}"
                f" | {_tick(row['high_tail_agree'])}"
                f" | {_tick(row['low_tail_agree'])}"
                f" | {'✓' if row['in_both'] else '✗'}"
                f" | {rep_cell} |"
            )
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--seed", type=int, default=42,
                   help="Canonical random seed used for the reported feature lists.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--top-n", type=int, default=15,
                   help="Top-N features per phase to include in stability comparison.")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Number of random seeds to run. Seeds 0..n_seeds-1 are used. "
                        "The canonical seed (--seed) is always included. "
                        "Use --n-seeds 1 for a fast single-seed run.")
    p.add_argument("--no-anchors", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _run_one_seed(
    seed: int,
    profiles: pd.DataFrame,
    zscore_cols: list[str],
    le: LabelEncoder,
    n_folds: int,
    top_n: int,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, str]:
    """Run the full split-half pipeline for a single seed.

    Returns (r1, r2, imp_df, gauntlet_df, balance_table_str).
    """
    all_users = profiles["author"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_users)
    mid = len(shuffled) // 2
    users_h1 = set(shuffled[:mid])
    users_h2 = set(shuffled[mid:])

    profiles_h1 = profiles[profiles["author"].isin(users_h1)].copy()
    profiles_h2 = profiles[profiles["author"].isin(users_h2)].copy()

    balance_table = compute_balance_table(profiles_h1, profiles_h2, list(le.classes_))

    r1 = fit_half(profiles_h1, zscore_cols, le, n_folds, f"H1(seed={seed})", top_n=top_n)
    r2 = fit_half(profiles_h2, zscore_cols, le, n_folds, f"H2(seed={seed})", top_n=top_n)
    imp_df, gauntlet_df = compare_halves(r1, r2, le, top_n=top_n)

    return r1, r2, imp_df, gauntlet_df, balance_table


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_config(args.config)

    output_dir = ROOT / "reports" / "ebm_stability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load full dataset ─────────────────────────────────────────────────
    logging.info("\n[1] Loading phase-labeled data…")
    profiles, zscore_cols, le = load_phase_labeled_dataset(cfg, no_anchors=args.no_anchors)

    all_users = profiles["author"].unique()
    logging.info(f"  Total users: {len(all_users):,}  |  profiles: {len(profiles):,}")

    # ── [2] Multi-seed loop ───────────────────────────────────────────────────
    # Always include the canonical seed. Seeds are 0..n_seeds-1 if n_seeds > 1,
    # else just [args.seed].
    if args.n_seeds == 1:
        seed_list = [args.seed]
    else:
        seed_list = list(range(args.n_seeds))
        if args.seed not in seed_list:
            seed_list.append(args.seed)

    logging.info(f"\n[2] Running split-half analysis over {len(seed_list)} seed(s): {seed_list}")

    # Accumulate per-phase per-seed stats
    per_seed_phase_stats: dict[str, list[dict]] = {ph: [] for ph in le.classes_}

    # Canonical seed artifacts (for feature lists and per-seed balance table)
    canonical_r1 = canonical_r2 = canonical_imp = canonical_gauntlet = canonical_balance = None

    for i, seed in enumerate(seed_list):
        logging.info(f"\n  --- Seed {seed} ({i+1}/{len(seed_list)}) ---")
        r1, r2, imp_df, gauntlet_df, balance_table = _run_one_seed(
            seed, profiles, zscore_cols, le, args.n_folds, args.top_n
        )

        if seed == args.seed:
            canonical_r1      = r1
            canonical_r2      = r2
            canonical_imp     = imp_df
            canonical_gauntlet = gauntlet_df
            canonical_balance  = balance_table
            logging.info(f"  [canonical seed={seed}] Balance table:")
            for line in balance_table.split("\n"):
                logging.info(f"    {line}")

        for phase in le.classes_:
            phase_row = imp_df[imp_df["phase"] == phase].iloc[0]
            per_seed_phase_stats[phase].append({
                "rho":       float(phase_row["spearman_rho"]),
                "overlap":   float(phase_row["top_overlap"]),
                "p_hg":      float(phase_row["p_hypergeom"]),
                "dir_agree": float(phase_row["direction_agreement_frac"])
                             if not np.isnan(phase_row["direction_agreement_frac"]) else float("nan"),
            })

        # Log per-seed summary
        for phase in le.classes_:
            phase_row = imp_df[imp_df["phase"] == phase].iloc[0]
            logging.info(
                f"  {phase:<14}: Spearman rho={phase_row['spearman_rho']:.3f}  "
                f"top-{args.top_n} overlap={phase_row['top_overlap']:.0f}/{args.top_n}  "
                f"hypergeom p={phase_row['p_hypergeom']:.4f}  "
                f"AUC H1={r1['phase_aucs'][phase]:.3f}  AUC H2={r2['phase_aucs'][phase]:.3f}"
            )

    # ── [3] Aggregate multi-seed statistics ───────────────────────────────────
    seed_stats: Optional[dict] = None
    if len(seed_list) > 1:
        seed_stats = {}
        logging.info("\n[3] Multi-seed aggregate statistics:")
        for phase in le.classes_:
            records = per_seed_phase_stats[phase]
            rhos       = [r["rho"]       for r in records]
            overlaps   = [r["overlap"]   for r in records]
            p_hgs      = [r["p_hg"]      for r in records]
            dir_agrees = [r["dir_agree"] for r in records if not np.isnan(r["dir_agree"])]

            seed_stats[phase] = {
                "mean_rho":          float(np.mean(rhos)),
                "sd_rho":            float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
                "mean_overlap":      float(np.mean(overlaps)),
                "sd_overlap":        float(np.std(overlaps, ddof=1)) if len(overlaps) > 1 else 0.0,
                "mean_p_hypergeom":  float(np.mean(p_hgs)),
                "sd_p_hypergeom":    float(np.std(p_hgs, ddof=1)) if len(p_hgs) > 1 else 0.0,
                "mean_dir_agree":    float(np.mean(dir_agrees)) if dir_agrees else float("nan"),
                "sd_dir_agree":      float(np.std(dir_agrees, ddof=1))
                                     if len(dir_agrees) > 1 else 0.0,
                "raw_rho":           rhos,  # used for n_seeds display in markdown
            }
            logging.info(
                f"  {phase:<14}: rho={seed_stats[phase]['mean_rho']:.3f}"
                f"±{seed_stats[phase]['sd_rho']:.3f}  "
                f"overlap={seed_stats[phase]['mean_overlap']:.1f}"
                f"±{seed_stats[phase]['sd_overlap']:.1f}/{args.top_n}  "
                f"hypergeom p={seed_stats[phase]['mean_p_hypergeom']:.4f}"
                f"±{seed_stats[phase]['sd_p_hypergeom']:.4f}"
            )

        # Direction and significance agreement rates for canonical run (in-both features)
        logging.info("\n  Canonical-run direction/significance agreement (in-both features):")
        for phase in le.classes_:
            g_ph = canonical_gauntlet[
                (canonical_gauntlet["phase"] == phase) & canonical_gauntlet["in_both"]
            ]
            if len(g_ph) == 0:
                continue
            dir_rate = g_ph["direction_agree"].mean()
            sig_rate = g_ph["sig_agree"].mean()
            logging.info(
                f"  {phase:<14}: direction agree={dir_rate:.0%}  sig agree={sig_rate:.0%}"
                f"  (n={len(g_ph)} features in both top-{args.top_n})"
            )
    else:
        # Single-seed: log direction/significance agreement
        for phase in le.classes_:
            g_ph = canonical_gauntlet[
                (canonical_gauntlet["phase"] == phase) & canonical_gauntlet["in_both"]
            ]
            if len(g_ph) == 0:
                continue
            dir_rate = g_ph["direction_agree"].mean()
            sig_rate = g_ph["sig_agree"].mean()
            logging.info(
                f"  {phase:<14}: direction agree={dir_rate:.0%}  sig agree={sig_rate:.0%}"
                f"  (n={len(g_ph)} features in both top-{args.top_n})"
            )

    # ── [4] Save canonical-seed artifacts ─────────────────────────────────────
    logging.info("\n[4] Saving outputs…")

    imp_path = output_dir / f"ebm_stability_importance_{timestamp}.csv"
    canonical_imp.to_csv(imp_path, index=False)
    logging.info(f"  Importance table → {imp_path.name}")

    g_path = output_dir / f"ebm_stability_gauntlet_{timestamp}.csv"
    canonical_gauntlet.to_csv(g_path, index=False)
    logging.info(f"  Gauntlet table   → {g_path.name}")

    md = build_markdown(
        canonical_imp,
        canonical_gauntlet,
        canonical_r1,
        canonical_r2,
        le,
        args.seed,
        args.top_n,
        balance_table=canonical_balance or "",
        seed_stats=seed_stats,
    )
    md_path = output_dir / f"ebm_stability_comparison_{timestamp}.md"
    md_path.write_text(md)
    logging.info(f"  Markdown report  → {md_path.name}")

    logging.info(f"\nAll outputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
