"""Script 21 — Volume Confound Analysis for Suicidality
=========================================================

Tests whether the suicidality phase pattern in Script 15 is driven by
a general posting-volume increase during menstrual phase rather than
a genuine increase in suicidal expression.

Three metrics computed per user × phase, then compared as excess
(phase rate − mean of other phases):

  1. rate_per_day   : n_suicidal_posts / phase_length_days       [current method]
  2. fraction       : n_suicidal_posts / n_posts_in_phase         [volume-controlled]
  3. volume         : n_posts / phase_length_days                 [pure volume check]

If the menstrual elevation disappears under metric 2 → volume artifact.
If metrics 1 and 2 agree → the signal is real regardless of posting frequency.
If metric 3 is elevated in menstrual but metric 2 is not → confirmed confound.

Also outputs a correlation plot: per-user per-phase (volume vs suicidal rate)
to show how tightly the two track each other.

Usage:
  python scripts/21_volume_confound_analysis.py
  python scripts/21_volume_confound_analysis.py --condition pmdd --phase-system pmdd
  python scripts/21_volume_confound_analysis.py --scores-file data/interim/suicidality_scores_ourafla_mental-health-bert-finetuned_checkpoint.csv
"""

import argparse
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PMDD_PHASE_ORDER = ["Perimenstrual", "Midfollicular", "Periovulatory", "Early Luteal", "Midluteal"]

PMDD_REGEX = re.compile(
    r"""
    I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
           suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
           am\s+dealing\s+with|was\s+told\s+I\s+have)\s+PMDD
    |diagnosed\s+(?:me\s+)?with\s+PMDD
    |(?:my|a)\s+PMDD\s+(?:diagnosis|symptoms?|episodes?|flare)
    |(?:PMDD\s+sufferer|PMDD\s+warrior|living\s+with\s+PMDD|my\s+PMDD)
    |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+PMDD
    """, re.IGNORECASE | re.VERBOSE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path, encoding="utf-8-sig")
    df["author"] = df["author"].astype(str)
    df["offset_from_cd1"] = pd.to_numeric(df["offset_from_cd1"], errors="coerce")
    df["p_suicidal"] = pd.to_numeric(df["p_suicidal"], errors="coerce")
    df = df.dropna(subset=["author", "offset_from_cd1", "p_suicidal"])
    logging.info(f"  Scores loaded: {len(df):,} posts | {df['author'].nunique():,} users")
    return df


def load_user_periods(interim_dir: Path) -> dict:
    path = find_latest_file(interim_dir, "consensus_periods_min23features_*.csv")
    if path is None:
        path = find_latest_file(interim_dir, "consensus_periods_*.csv")
    if path is None:
        raise FileNotFoundError("No consensus_periods_*.csv found in interim dir.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    col = "user" if "user" in df.columns else "author"
    result = dict(zip(df[col].astype(str), df["consensus_period"].astype(float)))
    logging.info(f"  User periods: {len(result):,} users | range {min(result.values()):.0f}–{max(result.values()):.0f} days")
    return result


def identify_pmdd_users(interim_dir: Path, candidate_users: set) -> set:
    posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_with_anchors_*.csv")
    if posts_path is None:
        posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_*.csv")
    if posts_path is None:
        logging.warning("  No preprocessed posts file found — PMDD users identified from scores only.")
        return set()
    posts = pd.read_csv(posts_path, encoding="utf-8-sig", usecols=["author", "text"], low_memory=False)
    posts = posts[posts["author"].astype(str).isin(candidate_users)]
    mask = posts["text"].fillna("").str.contains(PMDD_REGEX, regex=True)
    pmdd = set(posts[mask]["author"].astype(str).unique())
    logging.info(f"  PMDD users (regex): {len(pmdd)}")
    return pmdd


# ─────────────────────────────────────────────────────────────────────────────
# Phase assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_phases(scores: pd.DataFrame, user_periods: dict, phase_system: str) -> pd.DataFrame:
    from src.visualization import (
        create_adaptive_phases,
        create_pmdd_phases,
        assign_pmdd_phase,
    )

    default_period = 28.0
    scores = scores.copy()
    scores["period"] = scores["author"].map(user_periods).fillna(default_period)
    scores["offset_mod"] = scores["offset_from_cd1"] % scores["period"]

    unique_periods = scores["period"].unique()

    if phase_system == "pmdd":
        phase_cache = {p: create_pmdd_phases(p) for p in unique_periods}

        def _assign(offset_mod, period):
            return assign_pmdd_phase(offset_mod, phase_cache[period], period)
    else:
        phase_cache = {p: create_adaptive_phases(p) for p in unique_periods}

        def _assign(offset_mod, period):
            for name, (start, end) in phase_cache[period].items():
                if start <= offset_mod <= end:
                    return name
            return None

    scores["phase"] = scores.apply(lambda r: _assign(r["offset_mod"], r["period"]), axis=1)
    scores = scores[scores["phase"].notna()].copy()

    # Cache phase lengths
    def get_phase_length(phase_name, period):
        phases = phase_cache.get(period, create_adaptive_phases(period))
        if phase_name not in phases:
            return np.nan
        start, end = phases[phase_name]
        return abs(start) + end + 1 if start < 0 else end - start + 1

    return scores, phase_cache, get_phase_length


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(scores: pd.DataFrame, user_periods: dict, get_phase_length) -> pd.DataFrame:
    """Compute all three metrics per user × phase."""
    scores = scores.copy()
    scores["is_suicidal"] = (scores["p_suicidal"] > 0.5).astype(int)

    agg = (
        scores.groupby(["author", "group", "phase"])
        .agg(
            n_suicidal=("is_suicidal", "sum"),
            n_posts=("is_suicidal", "count"),
        )
        .reset_index()
    )
    default_period = 28.0
    agg["period"] = agg["author"].map(user_periods).fillna(default_period)
    agg["phase_length"] = agg.apply(
        lambda r: get_phase_length(r["phase"], r["period"]), axis=1
    )

    # Metric 1: current approach
    agg["rate_per_day"] = agg["n_suicidal"] / agg["phase_length"]

    # Metric 2: proportion (volume-controlled)
    agg["fraction"] = np.where(agg["n_posts"] > 0, agg["n_suicidal"] / agg["n_posts"], np.nan)

    # Metric 3: posting volume
    agg["volume"] = agg["n_posts"] / agg["phase_length"]

    return agg


def compute_excess(phase_df: pd.DataFrame, metric: str, phase_order: list) -> pd.DataFrame:
    """For each user × target phase: excess = phase_value − mean(other phases)."""
    rows = []
    for target_phase in phase_order:
        target = phase_df[phase_df["phase"] == target_phase][
            ["author", "group", "phase", metric]
        ].copy()
        other_mean = (
            phase_df[phase_df["phase"] != target_phase]
            .groupby("author")[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "other_mean"})
        )
        merged = target.merge(other_mean, on="author", how="inner").dropna(subset=[metric])
        merged["excess"] = merged[metric] - merged["other_mean"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def group_summary(excess_df: pd.DataFrame, condition_label: str, phase_order: list) -> pd.DataFrame:
    gp = (
        excess_df.groupby(["group", "phase"])["excess"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    gp.columns = ["group", "phase", "mean", "sem", "n"]
    gp["phase"] = pd.Categorical(gp["phase"], categories=phase_order, ordered=True)
    return gp.sort_values("phase")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_stats(excess_df: pd.DataFrame, condition_label: str, metric_name: str, phase_order: list) -> list[str]:
    lines = [
        f"\n{'=' * 60}",
        f"METRIC: {metric_name}",
        "=" * 60,
        "Within-group: is excess significantly different from 0? (one-sample t-test)",
    ]
    for group in [condition_label, "Control"]:
        lines.append(f"\n  {group}:")
        for phase in phase_order:
            vals = excess_df[
                (excess_df["group"] == group) & (excess_df["phase"] == phase)
            ]["excess"].dropna()
            if len(vals) >= 5:
                t, p = stats.ttest_1samp(vals, 0.0)
                lines.append(
                    f"    {phase:14s}  mean={vals.mean():+.5f}  t={t:.3f}  p={p:.4f}  n={len(vals)}"
                )

    lines.append(f"\nBetween-group ({condition_label} vs Control, Mann-Whitney U):")
    for phase in phase_order:
        cond = excess_df[
            (excess_df["group"] == condition_label) & (excess_df["phase"] == phase)
        ]["excess"].dropna()
        ctrl = excess_df[
            (excess_df["group"] == "Control") & (excess_df["phase"] == phase)
        ]["excess"].dropna()
        if len(cond) >= 5 and len(ctrl) >= 5:
            _, p = stats.mannwhitneyu(cond, ctrl, alternative="two-sided")
            diff = cond.mean() - ctrl.mean()
            lines.append(
                f"  {phase:14s}  {condition_label}={cond.mean():+.5f}  "
                f"Control={ctrl.mean():+.5f}  diff={diff:+.5f}  p={p:.4f}"
            )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "rate_per_day": "Suicidal posts / phase-day\n(current method)",
    "fraction":     "Suicidal posts / total posts\n(volume-controlled)",
    "volume":       "Total posts / phase-day\n(posting volume check)",
}
METRIC_COLORS = {
    "rate_per_day": ("#d62728", "#1f77b4"),
    "fraction":     ("#e6550d", "#3182bd"),
    "volume":       ("#756bb1", "#74c476"),
}


def plot_side_by_side(summaries: dict, condition_label: str, phase_order: list, output_dir: Path, tag: str, ts: str):
    """3-panel bar chart: one panel per metric."""
    metrics = ["rate_per_day", "fraction", "volume"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        f"Volume Confound Analysis — {condition_label} vs Control\n"
        "Excess = per-phase metric − mean(other phases); bars = ±SEM",
        fontsize=11,
    )

    x = np.arange(len(phase_order))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        gp = summaries[metric]
        cond_color, ctrl_color = METRIC_COLORS[metric]

        for i, (group, color) in enumerate([(condition_label, cond_color), ("Control", ctrl_color)]):
            sub = gp[gp["group"] == group].set_index("phase")
            means = [sub.loc[p, "mean"] if p in sub.index else np.nan for p in phase_order]
            sems  = [sub.loc[p, "sem"]  if p in sub.index else np.nan for p in phase_order]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, label=group, color=color, alpha=0.82,
                   yerr=sems, capsize=4, error_kw={"linewidth": 1})

        ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order, rotation=20, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_ylabel("Excess (phase − mean other phases)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = output_dir / f"{tag}_three_metrics_{ts}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  3-panel plot → {path.name}")


def plot_volume_vs_suicidality(phase_df: pd.DataFrame, condition_label: str, phase_order: list, output_dir: Path, tag: str, ts: str):
    """Scatter: per-user per-phase posting volume vs suicidal rate.

    Shows whether users who post more in a phase also produce more suicidal posts,
    which would indicate the volume confound is real.
    """
    fig, axes = plt.subplots(1, len(phase_order), figsize=(4 * len(phase_order), 4), sharey=False)
    if len(phase_order) == 1:
        axes = [axes]

    for ax, phase in zip(axes, phase_order):
        sub = phase_df[phase_df["phase"] == phase].dropna(subset=["volume", "rate_per_day"])
        if len(sub) < 5:
            ax.set_title(phase)
            continue

        for group, color, marker in [
            (condition_label, "#d62728", "o"),
            ("Control", "#1f77b4", "x"),
        ]:
            g = sub[sub["group"] == group]
            ax.scatter(g["volume"], g["rate_per_day"], c=color, marker=marker,
                       alpha=0.35, s=18, label=group)

        # Correlation across all users in this phase
        r, p = stats.spearmanr(sub["volume"], sub["rate_per_day"])
        ax.set_title(f"{phase}\nSpearman r={r:.2f}, p={p:.3f}", fontsize=9)
        ax.set_xlabel("Volume (posts/day)")
        ax.set_ylabel("Rate (suicidal/day)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=7, label=condition_label),
        plt.Line2D([0], [0], marker="x", color="#1f77b4", markersize=7, label="Control"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(
        "Volume vs Suicidality Rate per User per Phase\n"
        "High correlation → volume confound; Low correlation → signal is independent of volume",
        fontsize=10,
    )
    plt.tight_layout()
    path = output_dir / f"{tag}_volume_vs_suicidality_scatter_{ts}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Scatter plot → {path.name}")


def plot_phase_profiles_per_metric(summaries: dict, condition_label: str, phase_order: list, output_dir: Path, tag: str, ts: str):
    """Line plot across phases for each metric — easier to read convergence/divergence."""
    metrics = ["rate_per_day", "fraction", "volume"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    fig.suptitle(
        f"Phase profiles — {condition_label} vs Control",
        fontsize=11,
    )
    x = np.arange(len(phase_order))

    for ax, metric in zip(axes, metrics):
        gp = summaries[metric]
        cond_color, ctrl_color = METRIC_COLORS[metric]

        for group, color in [(condition_label, cond_color), ("Control", ctrl_color)]:
            sub = gp[gp["group"] == group].set_index("phase")
            means = [sub.loc[p, "mean"] if p in sub.index else np.nan for p in phase_order]
            sems  = [sub.loc[p, "sem"]  if p in sub.index else np.nan for p in phase_order]
            means_arr = np.array(means, dtype=float)
            sems_arr  = np.array(sems,  dtype=float)
            ax.plot(x, means_arr, marker="o", color=color, lw=2, label=group)
            ax.fill_between(x, means_arr - sems_arr, means_arr + sems_arr,
                            color=color, alpha=0.15)

        ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order, rotation=20, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_ylabel("Excess")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = output_dir / f"{tag}_phase_profiles_{ts}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Phase profiles plot → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument(
        "--scores-file", default=None,
        help="Path to scored posts CSV. Default: auto-detect latest checkpoint in interim dir.",
    )
    p.add_argument(
        "--condition", choices=["pmdd", "subreddit"], default="pmdd",
        help="How to identify the condition group. "
             "'pmdd' = regex self-report (default); "
             "'subreddit' = users who posted to PMDD subreddits.",
    )
    p.add_argument(
        "--phase-system", choices=["adaptive", "pmdd"], default="adaptive",
        help="Phase system to use (default: adaptive = M/F/O/L).",
    )
    p.add_argument(
        "--min-posts", type=int, default=3,
        help="Minimum posts per user per phase to include in analysis (default: 3). "
             "Filters out very sparse data that makes fraction estimates unreliable.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    interim_dir = ROOT / cfg["paths"]["interim"]
    analysis_dir = ROOT / cfg["paths"].get("analysis_dir", cfg["paths"]["interim"])
    reports_dir = ROOT / cfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"volume_confound_{args.condition}"

    logging.info("=== Script 21: Volume Confound Analysis ===\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info("Loading scored posts…")
    if args.scores_file:
        scores_path = Path(args.scores_file)
    else:
        # ±3-month window checkpoint (w3m suffix) — more data per user, more stable
        # per-phase estimates across multiple cycles.
        # ourafla/mental-health-bert-finetuned (4-class BERT, P(LABEL_3=Suicidal))
        scores_path = find_latest_file(
            interim_dir, "suicidality_scores_ourafla*w3m*checkpoint.csv"
        )
        if scores_path is None:
            scores_path = find_latest_file(
                interim_dir, "suicidality_scores_ourafla*checkpoint.csv"
            )
        if scores_path is None:
            scores_path = find_latest_file(interim_dir, "suicidality_scores_*checkpoint.csv")
        if scores_path is None:
            raise FileNotFoundError(
                "No suicidality scores checkpoint found. "
                "Run script 15 with --mode score first, or pass --scores-file.\n"
                "Expected: suicidality_scores_ourafla_mental-health-bert-finetuned_checkpoint.csv"
            )
    logging.info(f"  Scores file : {scores_path.name}")
    logging.info(f"  Model       : ourafla/mental-health-bert-finetuned (4-class BERT, P(LABEL_3=Suicidal))")
    scores = load_scores(scores_path)

    logging.info("Loading consensus periods…")
    user_periods = load_user_periods(interim_dir)

    # All consensus users — used for PMDD regex search across full post history.
    all_consensus_users = set(user_periods.keys())

    # Keep only consensus users in the scored posts
    scores = scores[scores["author"].isin(all_consensus_users)].copy()
    logging.info(f"  After filtering to consensus users: {len(scores):,} posts | {scores['author'].nunique():,} users")

    # Report effective window from the scores file itself
    actual_window = scores["offset_from_cd1"].abs().max()
    logging.info(f"  Scores window: ±{actual_window:.0f} days from CD1 (as scored)")

    # ── Identify condition group ───────────────────────────────────────────────
    # PMDD self-disclosure is searched across the user's FULL posting history,
    # not just the ±3-month suicidality window. A user who mentioned PMDD once
    # two years ago is still a PMDD user.
    logging.info(f"\nIdentifying condition group ({args.condition})…")
    logging.info("  Searching FULL post history of all consensus users for PMDD self-disclosure…")

    if args.condition == "pmdd":
        condition_users = identify_pmdd_users(analysis_dir, all_consensus_users)
        if not condition_users:
            condition_users = identify_pmdd_users(interim_dir, all_consensus_users)
        condition_label = "PMDD"
    else:  # subreddit
        pmdd_subs = set(cfg.get("subreddits", {}).get("pmdd", ["PMDD", "PMDDxADHD", "PMDDSharing"]))
        condition_users = set(
            scores[scores["subreddit"].isin(pmdd_subs)]["author"].unique()
        )
        condition_label = "PMDD_subreddit"
        logging.info(f"  PMDD subreddit users: {len(condition_users)}")

    scores["group"] = scores["author"].apply(
        lambda u: condition_label if u in condition_users else "Control"
    )
    logging.info(
        f"  {condition_label}: {scores[scores['group']==condition_label]['author'].nunique()} users | "
        f"Control: {scores[scores['group']=='Control']['author'].nunique()} users"
    )

    # ── Assign phases ─────────────────────────────────────────────────────────
    logging.info(f"\nAssigning phases ({args.phase_system})…")
    scores, phase_cache, get_phase_length = assign_phases(scores, user_periods, args.phase_system)
    phase_order = PMDD_PHASE_ORDER if args.phase_system == "pmdd" else PHASE_ORDER
    logging.info(f"  Posts after phase assignment: {len(scores):,}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    logging.info("\nComputing per-user × phase metrics…")
    phase_df = compute_metrics(scores, user_periods, get_phase_length)

    # Apply min-posts filter to fraction (too few posts → fraction is 0 or 1, unreliable)
    phase_df_filtered = phase_df[phase_df["n_posts"] >= args.min_posts].copy()
    n_dropped = len(phase_df) - len(phase_df_filtered)
    logging.info(
        f"  Rows after min_posts≥{args.min_posts} filter: {len(phase_df_filtered):,} "
        f"(dropped {n_dropped} sparse user-phase pairs)"
    )

    # ── Compute excess and summaries ──────────────────────────────────────────
    logging.info("\nComputing per-user excess (phase − mean other phases)…")
    metrics = ["rate_per_day", "fraction", "volume"]
    summaries = {}
    excess_dfs = {}

    for metric in metrics:
        # Use unfiltered df for rate_per_day and volume; filtered for fraction
        df_to_use = phase_df_filtered if metric == "fraction" else phase_df
        excess = compute_excess(df_to_use, metric, phase_order)
        excess_dfs[metric] = excess
        summaries[metric] = group_summary(excess, condition_label, phase_order)

    # ── Stats report ──────────────────────────────────────────────────────────
    all_lines = [
        f"VOLUME CONFOUND ANALYSIS — {condition_label} vs Control",
        f"Scores file: {scores_path.name}",
        f"Phase system: {args.phase_system}",
        f"Min posts per user-phase (for fraction): {args.min_posts}",
        "",
        "INTERPRETATION GUIDE:",
        "  rate_per_day + fraction agree → signal is real, not a volume artifact",
        "  rate_per_day elevated but fraction flat → volume artifact (posting more = more suicidal posts, but same proportion)",
        "  volume elevated in same phases as rate_per_day → supports artifact explanation",
        "",
    ]

    for metric in metrics:
        lines = run_stats(excess_dfs[metric], condition_label, metric, phase_order)
        all_lines.extend(lines)

    # Correlation: volume vs suicidal rate per user per phase
    all_lines.append("\n" + "=" * 60)
    all_lines.append("VOLUME ↔ SUICIDAL RATE CORRELATION (Spearman, per phase)")
    all_lines.append("High r → confound likely; Low r → rate is independent of volume")
    all_lines.append("")
    for phase in phase_order:
        sub = phase_df[phase_df["phase"] == phase].dropna(subset=["volume", "rate_per_day"])
        if len(sub) >= 10:
            r, p = stats.spearmanr(sub["volume"], sub["rate_per_day"])
            all_lines.append(f"  {phase:14s}  r={r:.3f}  p={p:.4f}  n={len(sub)}")

    report_text = "\n".join(all_lines)
    logging.info("\n" + report_text)

    rpt_path = reports_dir / f"{tag}_report_{ts}.txt"
    rpt_path.write_text(report_text)
    logging.info(f"\n  Report → {rpt_path.name}")

    # ── Save per-user data ────────────────────────────────────────────────────
    phase_df.to_csv(reports_dir / f"{tag}_user_phase_data_{ts}.csv", index=False, encoding="utf-8-sig")
    logging.info(f"  User-phase data → {tag}_user_phase_data_{ts}.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logging.info("\nGenerating plots…")
    plot_side_by_side(summaries, condition_label, phase_order, reports_dir, tag, ts)
    plot_phase_profiles_per_metric(summaries, condition_label, phase_order, reports_dir, tag, ts)
    plot_volume_vs_suicidality(phase_df, condition_label, phase_order, reports_dir, tag, ts)

    logging.info(f"\nDone. All outputs in {reports_dir}/")


if __name__ == "__main__":
    main()
