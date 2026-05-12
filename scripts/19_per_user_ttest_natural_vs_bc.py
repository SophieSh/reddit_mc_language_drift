#!/usr/bin/env python3
"""Step 19: Per-user t-test — ovulation vs other phases.

For each user, independently tests whether each linguistic feature is
significantly different during ovulation compared to all other phases.
Reports the *fraction of users* showing a significant effect (p < 0.05).

Compares two groups:
  Group A — Natural cycle users (910, min23 consensus minus BC)
             Phase assigned by each user's detected cycle period.
  Group B — Stable BC pill users on confirmed-suppressing pills (203)
             Phase assigned by fixed 28-day cycle.

Key idea: if the hormonal signal is real, more natural cycle users should
show a significant ovulation effect than BC users. If BC attenuates but
does not eliminate the signal, the fraction should be lower but > 5%.

Input:
  data/interim/consensus_periods_min23_no_bc.csv
  data/interim/bc_stable_suppressing_only.csv
  data/interim/timeline_daily_aggregated_with_anchors_*.csv

Output:
  reports/per_user_ttest_natural_vs_bc_{timestamp}.png
  reports/per_user_ttest_natural_vs_bc_{timestamp}.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.io import find_latest_file
from src.analysis import create_adaptive_phases, assign_phase_to_day

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]

ORIGINAL_FEATURES = [
    "negative_sentiment_mean", "positive_sentiment_mean", "num_words_mean",
    "avg_word_length_mean", "num_sentences_mean", "unique_word_fraction_mean",
    "readability_mean", "spelling_errors_frac_mean",
    "syntactic_complexity_subordination_index_mean",
    "cohesion_analysis_lexical_overlap_mean",
]


def assign_phases_for_group(
    timeline: pd.DataFrame,
    user_periods: dict[str, float],
    features: list[str],
    fixed_period: float | None = None,
) -> pd.DataFrame:
    """Assign phase labels to user-days.

    Args:
        timeline: user-days DataFrame with 'author' and 'offset_from_cd1'
        user_periods: {user -> detected period in days}
        features: feature columns to keep
        fixed_period: if set, all users get this period (BC group)

    Returns:
        DataFrame with 'author', 'phase', and feature columns (z-scored per user).
    """
    keep_cols = ["author", "offset_from_cd1"] + [f for f in features if f in timeline.columns]
    df = timeline[keep_cols].copy()

    # Daily aggregation (mean per user per day)
    df = df.groupby(["author", "offset_from_cd1"]).mean().reset_index()

    # Per-user z-score normalisation
    for feat in features:
        if feat not in df.columns:
            continue
        mu = df.groupby("author")[feat].transform("mean")
        sigma = df.groupby("author")[feat].transform("std").replace(0, np.nan)
        df[feat] = (df[feat] - mu) / sigma

    # Phase assignment
    records = []
    for user, udf in df.groupby("author"):
        period = fixed_period if fixed_period is not None else user_periods.get(str(user))
        if period is None:
            continue
        phases = create_adaptive_phases(float(period))
        for _, row in udf.iterrows():
            phase = assign_phase_to_day(row["offset_from_cd1"], phases)
            if phase is None:
                continue
            rec = {"author": user, "phase": phase}
            for feat in features:
                if feat in row.index:
                    rec[feat] = row[feat]
            records.append(rec)

    return pd.DataFrame(records)


def per_user_ttest(
    phased: pd.DataFrame,
    features: list[str],
    target_phase: str = "Ovulation",
    min_days_target: int = 2,
    min_days_other: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """For each user, t-test: target_phase days vs all other days.

    Returns DataFrame with columns:
        feature, n_users_tested, n_significant, frac_significant,
        median_effect (Cohen's d), mean_pvalue
    """
    other_phases = [p for p in PHASE_ORDER if p != target_phase]
    records = []

    for feat in features:
        if feat not in phased.columns:
            continue

        n_tested = 0
        n_sig = 0
        effects = []
        pvals = []

        for user, udf in phased.groupby("author"):
            target_vals = udf.loc[udf["phase"] == target_phase, feat].dropna()
            other_vals  = udf.loc[udf["phase"].isin(other_phases), feat].dropna()

            if len(target_vals) < min_days_target or len(other_vals) < min_days_other:
                continue

            n_tested += 1
            t_stat, pval = stats.ttest_ind(target_vals, other_vals, equal_var=False)
            pvals.append(pval)

            # Cohen's d
            pooled_std = np.sqrt(
                (target_vals.std() ** 2 + other_vals.std() ** 2) / 2
            )
            d = (target_vals.mean() - other_vals.mean()) / (pooled_std + 1e-10)
            effects.append(d)

            if pval < alpha:
                n_sig += 1

        if n_tested == 0:
            continue

        records.append({
            "feature": feat,
            "n_users_tested": n_tested,
            "n_significant": n_sig,
            "frac_significant": n_sig / n_tested,
            "median_cohens_d": float(np.median(effects)) if effects else np.nan,
            "mean_pvalue": float(np.mean(pvals)) if pvals else np.nan,
        })

    return pd.DataFrame(records).sort_values("frac_significant", ascending=False)


def plot_ttest_comparison(
    nat_df: pd.DataFrame,
    bc_df: pd.DataFrame,
    target_phase: str,
    alpha: float,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Horizontal bar chart: fraction significant, natural vs BC."""
    merged = nat_df[["feature", "frac_significant", "n_users_tested"]].merge(
        bc_df[["feature", "frac_significant", "n_users_tested"]],
        on="feature",
        suffixes=("_nat", "_bc"),
    )
    # Sort by natural cycle fraction descending, take top_n
    merged = merged.sort_values("frac_significant_nat", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    y = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, merged["frac_significant_nat"], bar_h,
            color="#1f77b4", alpha=0.85, label="Natural cycle (910 users)")
    ax.barh(y - bar_h / 2, merged["frac_significant_bc"],  bar_h,
            color="#ff7f0e", alpha=0.85, label=f"Stable BC suppressing (203 users)")

    # Chance line
    ax.axvline(alpha, color="red", linestyle="--", linewidth=1.0,
               label=f"Chance = {alpha:.0%}")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f.replace("_mean", "").replace("_", " ").title() for f in merged["feature"]],
        fontsize=8,
    )
    ax.set_xlabel("Fraction of users with significant effect (p < 0.05)", fontsize=10)
    ax.set_title(
        f"Per-user t-test: {target_phase} vs other phases\n"
        f"Top {top_n} features by natural-cycle fraction",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(merged["frac_significant_nat"].max(),
                        merged["frac_significant_bc"].max()) * 1.15)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    config_path: str = "configs/base.yaml",
    natural_consensus: str = "consensus_periods_min23_no_bc.csv",
    bc_stable_file: str = "data/interim/bc_stable_suppressing_only.csv",
    target_phase: str = "Ovulation",
    alpha: float = 0.05,
    original_features_only: bool = False,
    top_n: int = 30,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 19: Per-user t-test — Natural Cycle vs Stable BC")
    print("=" * 60)
    print()

    # ── [1] Load natural cycle users ────────────────────────────────────────
    nat_path = interim_dir / natural_consensus
    print(f"[1] Natural cycle: {nat_path.name}")
    nat_df = pd.read_csv(nat_path)
    nat_periods = dict(zip(nat_df["user"].astype(str), nat_df["consensus_period"]))
    print(f"    {len(nat_periods):,} users, median period {nat_df['consensus_period'].median():.1f} d")

    # ── [2] Load BC users ────────────────────────────────────────────────────
    bc_path = Path(bc_stable_file)
    print(f"\n[2] Stable BC (suppressing): {bc_path.name}")
    bc_df = pd.read_csv(bc_path)
    bc_users = set(bc_df["author"].astype(str))
    print(f"    {len(bc_users):,} users, fixed period: 28 days")

    # ── [3] Load timeline ────────────────────────────────────────────────────
    print("\n[3] Loading daily aggregated timeline...")
    agg_path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    timeline = pd.read_csv(agg_path, encoding="utf-8-sig", low_memory=False)
    print(f"    {len(timeline):,} user-days, {timeline['author'].nunique():,} users")

    # ── [4] Feature columns ──────────────────────────────────────────────────
    meta = {"author", "offset_from_cd1"}
    all_feat = [
        c for c in timeline.columns
        if c not in meta and c.endswith("_mean")
        and pd.api.types.is_numeric_dtype(timeline[c])
    ]
    features = [f for f in ORIGINAL_FEATURES if f in all_feat] if original_features_only else all_feat
    print(f"\n[4] Features: {len(features)}")

    # ── [5] Assign phases ────────────────────────────────────────────────────
    print(f"\n[5] Assigning phases — natural cycle group ({len(nat_periods)} users)...")
    nat_timeline = timeline[timeline["author"].astype(str).isin(nat_periods)].copy()
    nat_phased = assign_phases_for_group(nat_timeline, nat_periods, features)
    print(f"    {nat_phased['author'].nunique():,} users, {len(nat_phased):,} user-days with phase")
    for ph in PHASE_ORDER:
        n = (nat_phased["phase"] == ph).sum()
        nu = nat_phased.loc[nat_phased["phase"] == ph, "author"].nunique()
        print(f"      {ph}: {n} days, {nu} users")

    print(f"\n    Assigning phases — BC group ({len(bc_users)} users)...")
    bc_timeline = timeline[timeline["author"].astype(str).isin(bc_users)].copy()
    bc_phased = assign_phases_for_group(bc_timeline, {}, features, fixed_period=28.0)
    print(f"    {bc_phased['author'].nunique():,} users, {len(bc_phased):,} user-days with phase")
    for ph in PHASE_ORDER:
        n = (bc_phased["phase"] == ph).sum()
        nu = bc_phased.loc[bc_phased["phase"] == ph, "author"].nunique()
        print(f"      {ph}: {n} days, {nu} users")

    # ── [6] Per-user t-test ──────────────────────────────────────────────────
    print(f"\n[6] Running per-user t-tests (target: {target_phase})...")
    nat_results = per_user_ttest(nat_phased, features, target_phase=target_phase, alpha=alpha)
    bc_results  = per_user_ttest(bc_phased,  features, target_phase=target_phase, alpha=alpha)
    print(f"    Natural: {len(nat_results)} features tested")
    print(f"    BC:      {len(bc_results)} features tested")

    # ── [7] Summary table ────────────────────────────────────────────────────
    summary = nat_results[["feature", "frac_significant", "n_users_tested", "median_cohens_d"]].merge(
        bc_results[["feature", "frac_significant", "n_users_tested", "median_cohens_d"]],
        on="feature", suffixes=("_nat", "_bc"),
    )
    summary["attenuation"] = summary["frac_significant_nat"] - summary["frac_significant_bc"]
    summary = summary.sort_values("frac_significant_nat", ascending=False)

    print(f"\n    Top 20 features by natural-cycle ovulation fraction:")
    print(f"    {'Feature':<50} {'Nat%':>6} {'BC%':>6} {'Δ':>6}")
    print(f"    {'-'*50} {'------':>6} {'------':>6} {'------':>6}")
    for _, row in summary.head(20).iterrows():
        label = row["feature"].replace("_mean", "").replace("_", " ")[:48]
        print(f"    {label:<50} {row['frac_significant_nat']:>5.1%} "
              f"{row['frac_significant_bc']:>5.1%} "
              f"{row['attenuation']:>+5.1%}")

    # Save CSV
    csv_path = reports_dir / f"per_user_ttest_natural_vs_bc_{TIMESTAMP}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n    Saved CSV → {csv_path.name}")

    # ── [8] Plot ─────────────────────────────────────────────────────────────
    out_path = reports_dir / f"per_user_ttest_natural_vs_bc_{TIMESTAMP}.png"
    print(f"\n[8] Plotting → {out_path.name}...")
    plot_ttest_comparison(nat_results, bc_results, target_phase, alpha, out_path, top_n=top_n)
    print(f"    Saved → {out_path.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-user t-test: ovulation vs other phases, natural cycle vs BC"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--natural-consensus", default="consensus_periods_min23_no_bc.csv")
    ap.add_argument("--bc-stable-file", default="data/interim/bc_stable_suppressing_only.csv")
    ap.add_argument("--target-phase", default="Ovulation",
                    choices=["Ovulation", "Menstrual", "Follicular", "Luteal"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--original-features-only", action="store_true")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        natural_consensus=args.natural_consensus,
        bc_stable_file=args.bc_stable_file,
        target_phase=args.target_phase,
        alpha=args.alpha,
        original_features_only=args.original_features_only,
        top_n=args.top_n,
    ))
