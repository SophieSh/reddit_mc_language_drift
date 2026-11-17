"""Method 1: Linear Mixed-Effects Model for phase effects.

For each feature:
  feature_value ~ phase + (1 | user)

The random intercept absorbs each user's stable linguistic baseline.
The fixed effect of phase is estimated purely from within-person variation
across days — no phase-length bias, no centering artifacts.

Reference level: Luteal (most days, most stable estimates).
Contrasts: Menstrual, Follicular, Ovulation vs Luteal.

Multiple testing: Benjamini-Hochberg FDR across 69 features × 3 contrasts.

Outputs:
  reports/lme_phase_effects/forest_plot.png   — coefficients per feature × contrast
  reports/lme_phase_effects/lme_results.csv   — full results table
  stdout: significant features after FDR correction

Usage:
  python scripts/46_lme_phase_effects.py \\
    --phase-labeled data/interim/timeline_phase_labeled_fixed28_raw_*.csv \\
    --eligible      data/interim/eligible_users_timeline_minposts20_span100_minphase3_*.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import PHASE_COLORS, PHASE_ORDER
from src.io import find_latest_file

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CONTRASTS = ["Menstrual", "Follicular", "Ovulation"]  # vs Luteal reference
CONTRAST_COLORS = {c: PHASE_COLORS[c] for c in CONTRASTS}


def _resolve_features(columns: list[str]) -> tuple[list[str], str]:
    non_feature = {"author", "offset_from_cd1", "phase"}
    for suffix in ("_mean", "_zscore"):
        selected = sorted(c for c in columns if c.endswith(suffix) and c not in non_feature)
        if selected:
            return selected, suffix
    return [], ""


def _clean(col: str) -> str:
    return col.replace("_zscore", "").replace("_mean", "").replace("_", " ")


def _assign_phase(offset: float, period: int, boundaries: dict) -> str | None:
    day = int(offset) % period
    for phase, (start, end) in boundaries.items():
        if start <= day <= end:
            return phase
    return None


def fit_lme(df: pd.DataFrame, feature: str) -> dict | None:
    """Fit feature ~ phase + (1|author) with Luteal as reference."""
    sub = df[["author", "phase", feature]].dropna().copy()
    if sub[feature].std() < 1e-10:
        return None

    # Rename feature column to 'y' so the formula parser never sees special chars
    sub = sub.rename(columns={feature: "y"})
    formula = "y ~ C(phase, Treatment(reference='Luteal'))"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mixedlm(formula, sub, groups=sub["author"])
            result = model.fit(method="lbfgs", disp=False)

        row = {"feature": feature}
        for contrast in CONTRASTS:
            key = f"C(phase, Treatment(reference='Luteal'))[T.{contrast}]"
            if key in result.params:
                row[f"coef_{contrast}"]  = result.params[key]
                row[f"se_{contrast}"]    = result.bse[key]
                row[f"pval_{contrast}"]  = result.pvalues[key]
            else:
                row[f"coef_{contrast}"]  = np.nan
                row[f"se_{contrast}"]    = np.nan
                row[f"pval_{contrast}"]  = np.nan
        row["n_obs"]   = int(result.nobs)
        row["n_users"] = sub["author"].nunique()
        return row

    except Exception as e:
        log.warning("LME failed for %s: %s", feature, e)
        return None


def fit_lme_interaction(df: pd.DataFrame, feature: str) -> dict | None:
    """Fit feature ~ phase * bc_status + (1|author).

    The phase:bc_status interaction coefficients answer:
    'Does the phase effect differ between BC and non-BC users?'
    Positive = BC users show MORE of that phase shift vs Luteal.
    Negative = BC users show LESS (i.e. the phase effect is attenuated).
    """
    sub = df[["author", "phase", "bc_status", feature]].dropna().copy()
    if sub[feature].std() < 1e-10:
        return None

    sub = sub.rename(columns={feature: "y"})
    formula = "y ~ C(phase, Treatment(reference='Luteal')) * bc_status"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mixedlm(formula, sub, groups=sub["author"])
            result = model.fit(method="lbfgs", disp=False)

        row = {"feature": feature}
        for contrast in CONTRASTS:
            # Main phase effect (non-BC users)
            main_key = f"C(phase, Treatment(reference='Luteal'))[T.{contrast}]"
            # Interaction: how BC users differ from non-BC in this phase contrast
            ix_key   = f"C(phase, Treatment(reference='Luteal'))[T.{contrast}]:bc_status"
            for prefix, key in [("main", main_key), ("ix", ix_key)]:
                if key in result.params:
                    row[f"{prefix}_coef_{contrast}"] = result.params[key]
                    row[f"{prefix}_se_{contrast}"]   = result.bse[key]
                    row[f"{prefix}_pval_{contrast}"] = result.pvalues[key]
                else:
                    row[f"{prefix}_coef_{contrast}"] = np.nan
                    row[f"{prefix}_se_{contrast}"]   = np.nan
                    row[f"{prefix}_pval_{contrast}"] = np.nan
        row["n_obs"]    = int(result.nobs)
        row["n_users"]  = sub["author"].nunique()
        row["n_bc"]     = int((sub["bc_status"] == 1).groupby(sub["author"]).first().sum())
        row["n_non_bc"] = int((sub["bc_status"] == 0).groupby(sub["author"]).first().sum())
        return row

    except Exception as e:
        log.warning("LME interaction failed for %s: %s", feature, e)
        return None


def plot_forest(results: pd.DataFrame, output_dir: Path) -> None:
    """Forest plot: one row per feature, one panel per contrast."""
    # Sort by absolute Menstrual coefficient
    results = results.copy()
    results["sort_key"] = results["coef_Menstrual"].abs().fillna(0)
    results = results.sort_values("sort_key", ascending=True)
    results["label"] = results["feature"].map(_clean)

    n = len(results)
    fig, axes = plt.subplots(1, len(CONTRASTS),
                             figsize=(len(CONTRASTS) * 6, max(8, n * 0.28)),
                             sharey=True)

    for ax, contrast in zip(axes, CONTRASTS):
        color = CONTRAST_COLORS[contrast]
        coefs = results[f"coef_{contrast}"].values
        ses   = results[f"se_{contrast}"].values
        sig   = results[f"qval_{contrast}"].values < 0.05

        y = np.arange(n)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        for i, (c, se, s) in enumerate(zip(coefs, ses, sig)):
            if np.isnan(c):
                continue
            col = color if s else "lightgray"
            ax.errorbar(c, i, xerr=1.96 * se, fmt="o", color=col,
                        markersize=4 if s else 2.5,
                        linewidth=1.2 if s else 0.7, capsize=2)

        ax.set_title(f"{contrast} vs Luteal\n(filled = FDR < 5%)",
                     fontsize=10, color=color)
        ax.set_xlabel("Coefficient (raw units)", fontsize=8)
        ax.tick_params(axis="y", labelsize=6.5)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels(results["label"].tolist(), fontsize=6.5)

    fig.suptitle(
        "LME fixed effects: feature ~ phase + (1|user),  reference = Luteal\n"
        "Error bars = 95% CI.  Grey = not significant after BH-FDR correction.",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "forest_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_interaction_forest(results: pd.DataFrame, output_dir: Path) -> None:
    """Forest plot for phase × bc_status interaction coefficients."""
    results = results.copy()
    results["sort_key"] = results["ix_coef_Ovulation"].abs().fillna(0)
    results = results.sort_values("sort_key", ascending=True)
    results["label"] = results["feature"].map(_clean)

    n = len(results)
    fig, axes = plt.subplots(1, len(CONTRASTS),
                             figsize=(len(CONTRASTS) * 6, max(8, n * 0.28)),
                             sharey=True)

    for ax, contrast in zip(axes, CONTRASTS):
        color = CONTRAST_COLORS[contrast]
        coefs = results[f"ix_coef_{contrast}"].values
        ses   = results[f"ix_se_{contrast}"].values
        sig   = results[f"ix_qval_{contrast}"].values < 0.05

        y = np.arange(n)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        for i, (c, se, s) in enumerate(zip(coefs, ses, sig)):
            if np.isnan(c):
                continue
            col = color if s else "lightgray"
            ax.errorbar(c, i, xerr=1.96 * se, fmt="o", color=col,
                        markersize=4 if s else 2.5,
                        linewidth=1.2 if s else 0.7, capsize=2)

        ax.set_title(f"BC interaction: {contrast} vs Luteal\n(filled = FDR < 5%)",
                     fontsize=10, color=color)
        ax.set_xlabel("Interaction coef (BC − non-BC phase shift)", fontsize=8)
        ax.tick_params(axis="y", labelsize=6.5)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels(results["label"].tolist(), fontsize=6.5)

    fig.suptitle(
        "LME: feature ~ phase × bc_status + (1|user),  reference = Luteal\n"
        "Positive = BC users show MORE of this phase shift  |  "
        "Negative = BC users show LESS (attenuated)\n"
        "Error bars = 95% CI.  Grey = not significant after BH-FDR.",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "interaction_forest_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    interim = ROOT / "data" / "interim"
    parser.add_argument("--phase-labeled", type=Path, default=None)
    parser.add_argument("--eligible", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "lme_phase_effects")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--menstrual-days", type=int, default=4)
    parser.add_argument("--follicular-days", type=int, default=11)
    parser.add_argument("--ovulation-days", type=int, default=3)
    parser.add_argument("--exclude-anchor", action="store_true",
                        help="Drop offset_from_cd1==0 (anchor posts) to avoid contamination")
    parser.add_argument("--bc-timeline", type=Path, default=None,
                        help="BC users timeline CSV. When provided, runs phase × bc_status interaction model.")
    parser.add_argument("--bc-eligible", type=Path, default=None,
                        help="Eligible BC users CSV (optional filter). Defaults to all BC users in --bc-timeline.")
    args = parser.parse_args()

    if args.phase_labeled is None:
        candidates = sorted(interim.glob("timeline_phase_labeled_fixed28_*.csv"))
        pool = [f for f in candidates if "_raw_" in f.name]
        args.phase_labeled = max(pool or candidates, key=lambda p: p.stat().st_mtime)
    if args.eligible is None:
        el = [f for f in sorted(interim.glob(
              "eligible_users_timeline_minposts20_span100_minphase3_*.csv"))
              if not any(t in f.name for t in ("_no_anchors_", "_bc_"))]
        if el:
            after = [f for f in el if f.stat().st_mtime >= args.phase_labeled.stat().st_mtime]
            args.eligible = min(after, key=lambda f: f.stat().st_mtime) if after else el[-1]
        else:
            args.eligible = find_latest_file(
                interim, "eligible_users_timeline_minposts20_span100_minphase3_*.csv")
    return args


def main() -> None:
    args = _parse_args()

    eligible_authors = set(pd.read_csv(args.eligible, usecols=["author"])["author"].unique())
    log.info("Eligible authors: %d", len(eligible_authors))

    df = pd.read_csv(args.phase_labeled)
    df = df[df["author"].isin(eligible_authors)].copy()

    m, f, o = args.menstrual_days, args.follicular_days, args.ovulation_days
    l = args.period - m - f - o
    boundaries = {
        "Menstrual":  (0,         m - 1),
        "Follicular": (m,         m + f - 1),
        "Ovulation":  (m + f,     m + f + o - 1),
        "Luteal":     (m + f + o, args.period - 1),
    }
    log.info("Phase model: period=%d M=%d F=%d O=%d L=%d", args.period, m, f, o, l)
    df["phase"] = df["offset_from_cd1"].apply(
        lambda x: _assign_phase(x, args.period, boundaries))
    df = df[df["phase"].isin(PHASE_ORDER)].copy()

    if args.window:
        df = df[df["offset_from_cd1"].between(-args.window, args.window)].copy()

    if args.exclude_anchor:
        before = len(df)
        df = df[df["offset_from_cd1"] != 0].copy()
        log.info("Excluded anchor posts (offset=0): %d → %d rows", before, len(df))

    log.info("Phase counts:\n%s", df["phase"].value_counts().reindex(PHASE_ORDER).to_string())

    features, suffix = _resolve_features(list(df.columns))
    # ── BC interaction mode ───────────────────────────────────────────────────
    if args.bc_timeline is not None:
        log.info("BC timeline provided — running phase × bc_status interaction model.")
        df_bc = pd.read_csv(args.bc_timeline)
        if args.bc_eligible is not None:
            bc_authors = set(pd.read_csv(args.bc_eligible, usecols=["author"])["author"].unique())
            df_bc = df_bc[df_bc["author"].isin(bc_authors)]
        df_bc["phase"] = df_bc["offset_from_cd1"].apply(
            lambda x: _assign_phase(x, args.period, boundaries))
        if args.window:
            df_bc = df_bc[df_bc["offset_from_cd1"].between(-args.window, args.window)]
        if args.exclude_anchor:
            df_bc = df_bc[df_bc["offset_from_cd1"] != 0]
        df_bc = df_bc[df_bc["phase"].isin(PHASE_ORDER)].copy()

        # Normalise feature suffixes: strip _zscore / _mean → base name, then rename
        # both dataframes to base names so they can be matched regardless of suffix.
        def _strip_suffix(col: str) -> str:
            for s in ("_zscore", "_mean"):
                if col.endswith(s):
                    return col[: -len(s)]
            return col

        non_meta = {"author", "offset_from_cd1", "phase", "bc_status", "pill_type"}
        nat_feat_cols = [c for c in features]          # already resolved with suffix
        bc_feat_cols  = [c for c in df_bc.columns
                         if c not in non_meta and not c.startswith("Unnamed:")
                         and pd.api.types.is_numeric_dtype(df_bc[c])]

        nat_base = {_strip_suffix(c): c for c in nat_feat_cols}
        bc_base  = {_strip_suffix(c): c for c in bc_feat_cols}
        shared_base = sorted(set(nat_base) & set(bc_base))

        if not shared_base:
            log.error("No shared feature columns between natural-cycle and BC timelines.")
            return

        log.info("Shared features (by base name): %d", len(shared_base))

        # Rename both to a common name (use the natural-cycle suffix as canonical)
        rename_bc = {bc_base[b]: nat_base[b] for b in shared_base}
        df_bc = df_bc.rename(columns=rename_bc)
        shared_features = [nat_base[b] for b in shared_base]
        log.info("Shared features for interaction model: %d", len(shared_features))

        df["bc_status"]    = 0
        df_bc["bc_status"] = 1
        df_pooled = pd.concat([df[["author", "phase", "bc_status"] + shared_features],
                                df_bc[["author", "phase", "bc_status"] + shared_features]],
                               ignore_index=True)
        log.info("Pooled: %d rows, %d non-BC users, %d BC users",
                 len(df_pooled),
                 df[df["bc_status"] == 0]["author"].nunique(),
                 df_bc["author"].nunique())

        log.info("Fitting LME interaction for %d features…", len(shared_features))
        ix_rows = []
        for i, feat in enumerate(shared_features):
            if (i + 1) % 10 == 0:
                log.info("  %d / %d", i + 1, len(shared_features))
            row = fit_lme_interaction(df_pooled, feat)
            if row:
                ix_rows.append(row)

        ix_results = pd.DataFrame(ix_rows)

        # BH-FDR on interaction p-values only
        all_pvals, all_keys = [], []
        for contrast in CONTRASTS:
            col  = f"ix_pval_{contrast}"
            mask = ix_results[col].notna()
            all_pvals.extend(ix_results.loc[mask, col].tolist())
            all_keys.extend([(idx, contrast) for idx in ix_results.index[mask]])
        _, qvals, _, _ = multipletests(all_pvals, method="fdr_bh")
        for (idx, contrast), q in zip(all_keys, qvals):
            ix_results.loc[idx, f"ix_qval_{contrast}"] = q
        for contrast in CONTRASTS:
            if f"ix_qval_{contrast}" not in ix_results.columns:
                ix_results[f"ix_qval_{contrast}"] = np.nan

        args.output_dir.mkdir(parents=True, exist_ok=True)
        ix_csv = args.output_dir / "lme_interaction_results.csv"
        ix_results.to_csv(ix_csv, index=False)
        log.info("Saved %s", ix_csv)

        plot_interaction_forest(ix_results, args.output_dir)

        print("\n=== Significant INTERACTION effects (BC × phase) after BH-FDR < 5% ===")
        print("Positive coef = BC users show MORE of this phase shift vs Luteal")
        print("Negative coef = BC users show LESS (attenuated phase effect)")
        print(f"{'Feature':<45} {'Contrast':<12} {'Coef':>8} {'SE':>7} {'q-val':>8}")
        print("-" * 85)
        found = False
        for contrast in CONTRASTS:
            sig = ix_results[ix_results[f"ix_qval_{contrast}"] < 0.05].copy()
            sig = sig.sort_values(f"ix_coef_{contrast}", key=abs, ascending=False)
            for _, r in sig.iterrows():
                print(f"{_clean(r['feature']):<45} {contrast:<12} "
                      f"{r[f'ix_coef_{contrast}']:>8.4f} {r[f'ix_se_{contrast}']:>7.4f} "
                      f"{r[f'ix_qval_{contrast}']:>8.4g}")
                found = True
        if not found:
            print("  None.")
        return

    # ── Standard (non-BC) LME ─────────────────────────────────────────────────
    log.info("Fitting LME for %d features…", len(features))

    rows = []
    for i, feat in enumerate(features):
        if (i + 1) % 10 == 0:
            log.info("  %d / %d", i + 1, len(features))
        row = fit_lme(df, feat)
        if row:
            rows.append(row)

    results = pd.DataFrame(rows)

    # BH-FDR correction across all features × contrasts jointly
    all_pvals = []
    all_keys  = []
    for contrast in CONTRASTS:
        col = f"pval_{contrast}"
        mask = results[col].notna()
        all_pvals.extend(results.loc[mask, col].tolist())
        all_keys.extend([(idx, contrast) for idx in results.index[mask]])

    _, qvals, _, _ = multipletests(all_pvals, method="fdr_bh")
    for (idx, contrast), q in zip(all_keys, qvals):
        results.loc[idx, f"qval_{contrast}"] = q
    for contrast in CONTRASTS:
        if f"qval_{contrast}" not in results.columns:
            results[f"qval_{contrast}"] = np.nan

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "lme_results.csv"
    results.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    plot_forest(results, args.output_dir)

    # Print significant findings
    print("\n=== Significant features after BH-FDR < 5% ===")
    print(f"{'Feature':<45} {'Contrast':<12} {'Coef':>8} {'SE':>7} {'q-val':>8}")
    print("-" * 85)
    found = False
    for contrast in CONTRASTS:
        sig = results[results[f"qval_{contrast}"] < 0.05].copy()
        sig = sig.sort_values(f"coef_{contrast}", key=abs, ascending=False)
        for _, row in sig.iterrows():
            print(f"{_clean(row['feature']):<45} {contrast:<12} "
                  f"{row[f'coef_{contrast}']:>8.4f} {row[f'se_{contrast}']:>7.4f} "
                  f"{row[f'qval_{contrast}']:>8.4g}")
            found = True
    if not found:
        print("  None.")


if __name__ == "__main__":
    main()
