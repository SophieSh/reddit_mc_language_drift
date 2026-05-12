#!/usr/bin/env python3
"""Step 20: Per-user t-test before vs after starting BC pill.

For users who recently started a BC pill, runs a per-user t-test
(target phase vs other phases) separately in the BEFORE and AFTER windows.
Reports the fraction of users showing a significant effect in each window.

If the pill suppresses the hormonal signal, the fraction should drop
from BEFORE to AFTER — especially at Ovulation.

Phase assignment uses day_rel % 28 (day_rel = offset_from_cd1 - pill_start_offset),
so phase is relative to pill start in both windows.

Input:
  data/processed/bc_wide_candidates_with_labels.csv  (LLM labels)
  data/interim/timeline_daily_aggregated_with_anchors_*.csv

Output:
  reports/per_user_ttest_before_after_bc_{timestamp}.png
  reports/per_user_ttest_before_after_bc_{timestamp}.csv
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

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]

PHASE_BOUNDS_28 = {
    "Menstrual":  (0,  3),
    "Follicular": (4,  10),
    "Ovulation":  (11, 13),
    "Luteal":     (14, 27),
}

ORIGINAL_FEATURES = [
    "negative_sentiment_mean", "positive_sentiment_mean", "num_words_mean",
    "avg_word_length_mean", "num_sentences_mean", "unique_word_fraction_mean",
    "readability_mean", "spelling_errors_frac_mean",
    "syntactic_complexity_subordination_index_mean",
    "cohesion_analysis_lexical_overlap_mean",
]


def assign_phase_28(day_in_cycle: int) -> str | None:
    for phase, (lo, hi) in PHASE_BOUNDS_28.items():
        if lo <= day_in_cycle <= hi:
            return phase
    return None


def per_user_ttest(
    phased: pd.DataFrame,
    features: list[str],
    target_phase: str,
    min_days_target: int = 2,
    min_days_other: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-user t-test: target_phase vs all other phases.

    Returns DataFrame: feature, n_users_tested, n_significant,
                       frac_significant, median_cohens_d, mean_pvalue.
    """
    other_phases = [p for p in PHASE_ORDER if p != target_phase]
    records = []

    for feat in features:
        if feat not in phased.columns:
            continue
        n_tested = n_sig = 0
        effects, pvals = [], []

        for _, udf in phased.groupby("author"):
            tgt  = udf.loc[udf["phase"] == target_phase, feat].dropna()
            rest = udf.loc[udf["phase"].isin(other_phases), feat].dropna()
            if len(tgt) < min_days_target or len(rest) < min_days_other:
                continue
            n_tested += 1
            _, pval = stats.ttest_ind(tgt, rest, equal_var=False)
            pvals.append(pval)
            pooled = np.sqrt((tgt.std() ** 2 + rest.std() ** 2) / 2)
            effects.append((tgt.mean() - rest.mean()) / (pooled + 1e-10))
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


def plot_before_after_ttest(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    target_phase: str,
    alpha: float,
    n_before: int,
    n_after: int,
    output_path: Path,
    top_n: int = 30,
    mode: str = "started",
) -> None:
    merged = before_df[["feature", "frac_significant"]].merge(
        after_df[["feature", "frac_significant"]],
        on="feature", suffixes=("_before", "_after"),
    )
    merged["drop"] = merged["frac_significant_before"] - merged["frac_significant_after"]
    merged = merged.sort_values("frac_significant_before", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, merged["frac_significant_before"], bar_h,
            color="#1f77b4", alpha=0.85,
            label=f"Before pill start (n={n_before} users)")
    ax.barh(y - bar_h / 2, merged["frac_significant_after"], bar_h,
            color="#ff7f0e", alpha=0.85,
            label=f"After pill start (n={n_after} users)")

    ax.axvline(alpha, color="red", linestyle="--", linewidth=1.0,
               label=f"Chance = {alpha:.0%}")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f.replace("_mean", "").replace("_", " ").title() for f in merged["feature"]],
        fontsize=8,
    )
    ax.set_xlabel("Fraction of users with significant effect (p < 0.05)", fontsize=10)
    action = "starting" if mode == "started" else "stopping"
    ax.set_title(
        f"Per-user t-test: {target_phase} vs other phases\n"
        f"Before vs After {action} BC pill  |  Top {top_n} features by before-fraction",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    config_path: str = "configs/base.yaml",
    labels_file: str = "data/processed/bc_wide_candidates_with_labels.csv",
    target_phase: str = "Ovulation",
    alpha: float = 0.05,
    original_features_only: bool = False,
    top_n: int = 30,
    mode: str = "started",  # "started" or "stopped"
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    action = "Starting" if mode == "started" else "Stopping"
    print("=" * 60)
    print(f"Step 20: Per-user t-test Before vs After {action} BC Pill")
    print("=" * 60)
    print()

    # ── [1] Load LLM labels ──────────────────────────────────────────────────
    print("[1] Loading LLM labels...")
    try:
        labels = pd.read_excel(labels_file, engine="openpyxl")
    except Exception:
        labels = pd.read_csv(labels_file, encoding="utf-8-sig")

    if mode == "started":
        flag_col, offset_col = "started_recently", "started_offset"
    else:
        flag_col, offset_col = "stopped_recently", "stopped_offset"

    filtered = labels[(labels["is_bc_pill"] == True) & (labels[flag_col] == True)].copy()
    # event_offset is relative to the post day, not to CD1.
    # Actual event day in CD1 terms = offset_from_cd1 + event_offset
    # Only use posts within the ±90 day timeline window to estimate the event offset.
    filtered["actual_event_cd1"] = filtered["offset_from_cd1"] + filtered[offset_col]
    filtered_in_window = filtered[filtered["offset_from_cd1"].between(-90, 90)]
    user_start = (
        filtered_in_window.groupby("author")["actual_event_cd1"]
        .median().reset_index()
        .rename(columns={"actual_event_cd1": "pill_start_offset"})
    )
    user_start["pill_start_offset"] = user_start["pill_start_offset"].round().astype(int)
    print(f"    {len(user_start):,} users who recently {mode} BC pill")
    print(f"    event offset range: {user_start['pill_start_offset'].min()} "
          f"to {user_start['pill_start_offset'].max()}")

    # ── [2] Load timeline ────────────────────────────────────────────────────
    print("\n[2] Loading daily aggregated timeline...")
    agg_path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    timeline = pd.read_csv(agg_path, encoding="utf-8-sig", low_memory=False)
    timeline = timeline.merge(user_start, on="author", how="inner")
    n_users = timeline["author"].nunique()
    print(f"    {n_users:,} started-BC users found in timeline")
    if n_users == 0:
        print("    No users found.")
        return 1

    # ── [3] Feature columns ──────────────────────────────────────────────────
    meta_cols = {"author", "offset_from_cd1", "pill_start_offset"}
    all_feat = [
        c for c in timeline.columns
        if c not in meta_cols and c.endswith("_mean")
        and pd.api.types.is_numeric_dtype(timeline[c])
    ]
    features = [f for f in ORIGINAL_FEATURES if f in all_feat] if original_features_only else all_feat
    print(f"\n[3] Features: {len(features)}")

    # ── [4] day_rel, z-score normalisation, phase assignment ────────────────
    print("\n[4] Computing day_rel and normalising...")
    timeline["day_rel"] = (
        timeline["offset_from_cd1"] - timeline["pill_start_offset"]
    ).round().astype(int)

    # Daily aggregation first
    keep = ["author", "day_rel"] + [f for f in features if f in timeline.columns]
    daily = timeline[keep].groupby(["author", "day_rel"]).mean().reset_index()

    # Per-user z-score over full timeline
    for feat in features:
        if feat not in daily.columns:
            continue
        mu    = daily.groupby("author")[feat].transform("mean")
        sigma = daily.groupby("author")[feat].transform("std").replace(0, np.nan)
        daily[feat] = (daily[feat] - mu) / sigma

    # Phase from day_rel % 28
    daily["day_in_cycle"] = daily["day_rel"] % 28
    daily["phase"] = daily["day_in_cycle"].apply(assign_phase_28)
    daily = daily[daily["phase"].notna()].copy()

    # ── [5] Split before / after, keep users with data in both ──────────────
    print("\n[5] Splitting before / after pill start...")
    before_all = daily[daily["day_rel"] < 0].copy()
    after_all  = daily[daily["day_rel"] >= 0].copy()
    users_both = set(before_all["author"]) & set(after_all["author"])
    print(f"    Users with data before: {before_all['author'].nunique():,}")
    print(f"    Users with data after:  {after_all['author'].nunique():,}")
    print(f"    Users with data in BOTH: {len(users_both):,}  ← using these only")

    before = before_all[before_all["author"].isin(users_both)].copy()
    after  = after_all[after_all["author"].isin(users_both)].copy()

    print(f"\n    Phase distribution BEFORE ({target_phase} target):")
    for ph in PHASE_ORDER:
        n = (before["phase"] == ph).sum()
        nu = before.loc[before["phase"] == ph, "author"].nunique()
        print(f"      {ph}: {n} days, {nu} users")
    print(f"\n    Phase distribution AFTER ({target_phase} target):")
    for ph in PHASE_ORDER:
        n = (after["phase"] == ph).sum()
        nu = after.loc[after["phase"] == ph, "author"].nunique()
        print(f"      {ph}: {n} days, {nu} users")

    n_before_users = before["author"].nunique()
    n_after_users  = after["author"].nunique()

    # ── [6] Per-user t-test in each window ───────────────────────────────────
    print(f"\n[6] Running per-user t-tests (target: {target_phase})...")
    before_results = per_user_ttest(before, features, target_phase=target_phase, alpha=alpha)
    after_results  = per_user_ttest(after,  features, target_phase=target_phase, alpha=alpha)
    print(f"    Before: {len(before_results)} features tested")
    print(f"    After:  {len(after_results)} features tested")

    # ── [7] Summary ──────────────────────────────────────────────────────────
    summary = before_results[["feature", "frac_significant", "n_users_tested", "median_cohens_d"]].merge(
        after_results[["feature", "frac_significant", "n_users_tested", "median_cohens_d"]],
        on="feature", suffixes=("_before", "_after"),
    )
    summary["drop"] = summary["frac_significant_before"] - summary["frac_significant_after"]
    summary = summary.sort_values("frac_significant_before", ascending=False)

    print(f"\n    Top 20 features by before-fraction (target: {target_phase}):")
    print(f"    {'Feature':<50} {'Before':>7} {'After':>7} {'Drop':>7}")
    print(f"    {'-'*50} {'-------':>7} {'-------':>7} {'-------':>7}")
    for _, row in summary.head(20).iterrows():
        label = row["feature"].replace("_mean", "").replace("_", " ")[:48]
        print(f"    {label:<50} {row['frac_significant_before']:>6.1%} "
              f"{row['frac_significant_after']:>6.1%} "
              f"{row['drop']:>+6.1%}")

    csv_path = reports_dir / f"per_user_ttest_before_after_bc_{mode}_{TIMESTAMP}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n    Saved CSV → {csv_path.name}")

    # ── [8] Plot ─────────────────────────────────────────────────────────────
    out_path = reports_dir / f"per_user_ttest_before_after_bc_{mode}_{TIMESTAMP}.png"
    print(f"\n[8] Plotting → {out_path.name}...")
    plot_before_after_ttest(
        before_results, after_results, target_phase, alpha,
        n_before_users, n_after_users, out_path, top_n=top_n,
        mode=mode,
    )
    print(f"    Saved → {out_path.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-user t-test before vs after starting BC pill"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--labels-file",
                    default="data/processed/bc_wide_candidates_with_labels.csv")
    ap.add_argument("--target-phase", default="Ovulation",
                    choices=["Ovulation", "Menstrual", "Follicular", "Luteal"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--original-features-only", action="store_true")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--mode", default="started", choices=["started", "stopped"],
                    help="'started': before/after starting pill. 'stopped': before/after stopping pill.")
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        labels_file=args.labels_file,
        target_phase=args.target_phase,
        alpha=args.alpha,
        original_features_only=args.original_features_only,
        top_n=args.top_n,
        mode=args.mode,
    ))
