"""Step 15 — Suicidality Scores Across the Menstrual Cycle
==========================================================
Two-phase design:

  PHASE 1 — SCORING (slow, run once overnight)
    Score all posts within a time window for chosen users using the
    sentinet/suicidality HuggingFace model. Saves raw scores to CSV.
    No phase logic here — just posts + scores.

  PHASE 2 — ANALYSIS (fast, re-run freely)
    Load saved scores, apply per-user cycle phase assignment (using modulo
    of each user's consensus period), identify condition users, run stats,
    generate plots.

Usage:
  # Score all posts (±3 months window, 1042 consensus users):
  python scripts/15_suicidality_cycle_analysis.py --mode score --window-months 3

  # Re-run analysis on already-scored data:
  python scripts/15_suicidality_cycle_analysis.py --mode analyze

  # Both in sequence:
  python scripts/15_suicidality_cycle_analysis.py --mode both

Model: https://huggingface.co/sentinet/suicidality
  Binary ELECTRA: LABEL_0 (non-suicidal) vs LABEL_1 (suicidal)
  We save P(LABEL_1) as a continuous suicidality score per post.

Checkpoint:
  Scoring saves progress every --checkpoint-every posts.
  If interrupted, re-run the same command — already-scored posts are skipped.
"""

import argparse
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file

warnings.filterwarnings("ignore")

PHASE_BOUNDARIES = {
    "Menstrual":  (-2,  4),
    "Follicular": ( 5, 12),
    "Ovulation":  (12, 16),
    "Luteal":     (16, 28),
}
PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]

CONDITION_REGEX = {
    "pmdd": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
               am\s+dealing\s+with|was\s+told\s+I\s+have)\s+PMDD
        |diagnosed\s+(?:me\s+)?with\s+PMDD
        |(?:my|a)\s+PMDD\s+(?:diagnosis|symptoms?|episodes?|flare)
        |(?:PMDD\s+sufferer|PMDD\s+warrior|living\s+with\s+PMDD|my\s+PMDD)
        |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+PMDD
        """, re.IGNORECASE | re.VERBOSE,
    ),
    "adhd": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|deal\s+with|struggle\s+with|
               was\s+told\s+I\s+have)\s+ADHD
        |diagnosed\s+(?:me\s+)?with\s+ADHD
        |(?:my|a)\s+ADHD\s+(?:diagnosis|symptoms?|brain|medication|meds|treatment)
        |(?:ADHDer|living\s+with\s+ADHD|my\s+ADHD)
        |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+ADHD
        |ADHD\s+and\s+PMDD|PMDD\s+and\s+ADHD
        """, re.IGNORECASE | re.VERBOSE,
    ),
    "depression": re.compile(
        r"""
        I\s+(?:have|had|was\s+diagnosed\s+with|got\s+diagnosed\s+with|
               suffer\s+from|live\s+with|struggle\s+with)\s+
               (?:depression|MDD|major\s+depressive\s+disorder|clinical\s+depression)
        |diagnosed\s+(?:me\s+)?with\s+(?:depression|MDD|major\s+depressive\s+disorder)
        |(?:my|a)\s+(?:depression|MDD)\s+(?:diagnosis|symptoms?|episodes?|medication|meds)
        |as\s+(?:someone|a\s+(?:person|woman))\s+with\s+(?:depression|MDD)
        |living\s+with\s+depression
        """, re.IGNORECASE | re.VERBOSE,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def load_consensus_users(interim_dir: Path) -> tuple[set, dict]:
    """Returns (set of user IDs, dict of user -> consensus_period_days)."""
    consensus_path = find_latest_file(interim_dir, "consensus_periods_min23features_*.csv")
    if consensus_path is None:
        consensus_path = find_latest_file(interim_dir, "consensus_periods_*.csv")
    if consensus_path is None:
        raise FileNotFoundError("No consensus_periods_*.csv found.")
    logging.info(f"  Consensus file: {consensus_path.name}")
    df = pd.read_csv(consensus_path, encoding="utf-8-sig")
    user_col = "user" if "user" in df.columns else "author"
    df[user_col] = df[user_col].astype(str)
    user_periods = dict(zip(df[user_col], df["consensus_period"].astype(float)))
    logging.info(
        f"  {len(user_periods):,} users | "
        f"periods {min(user_periods.values()):.0f}–{max(user_periods.values()):.0f} days"
    )
    return set(user_periods.keys()), user_periods


def load_posts_window(
    interim_dir: Path,
    users: set,
    window_months: float,
) -> pd.DataFrame:
    """Load all posts within ±window_months of CD1 for the given users.

    No phase logic here — just raw posts with their offset_from_cd1.
    Saves: author, offset_from_cd1, subreddit, combined_text.
    """
    timeline_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    if timeline_path is None:
        timeline_path = find_latest_file(interim_dir, "timeline_with_offsets_*.csv")
    if timeline_path is None:
        raise FileNotFoundError("No timeline_with_offsets_*.csv found.")

    max_days = int(window_months * 30.44)  # average days per month
    logging.info(f"  Timeline: {timeline_path.name}")
    logging.info(f"  Window: ±{window_months} months = ±{max_days} days")

    cols = ["author", "offset_from_cd1", "selftext", "title", "subreddit"]
    df = pd.read_csv(timeline_path, encoding="utf-8-sig", usecols=cols, low_memory=False)
    df["author"] = df["author"].astype(str)
    df = df[df["author"].isin(users)].copy()
    df["offset_from_cd1"] = pd.to_numeric(df["offset_from_cd1"], errors="coerce")
    df = df[df["offset_from_cd1"].between(-max_days, max_days)].copy()
    df = df.reset_index(drop=True)

    # Combined text for scoring
    df["combined_text"] = (
        df["title"].fillna("").astype(str) + " " +
        df["selftext"].fillna("").astype(str)
    ).str.strip()
    df = df[df["combined_text"].str.len() > 10].copy()
    df = df.reset_index(drop=True)

    logging.info(f"  Posts loaded: {len(df):,} | Users: {df['author'].nunique():,}")
    return df[["author", "offset_from_cd1", "subreddit", "combined_text"]]


def score_posts(
    posts: pd.DataFrame,
    clf,
    batch_size: int,
    checkpoint_path: Path,
    checkpoint_every: int,
) -> pd.DataFrame:
    """Score posts with suicidality model. Checkpoints progress.

    Checkpoint CSV columns: post_index, author, offset_from_cd1, subreddit, p_suicidal
    Saves author/offset alongside scores so the checkpoint is self-contained
    (no need to reload the original posts file for analysis).

    If checkpoint exists, skips already-scored posts and appends new results.
    """
    scored_ids = set()
    if checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path, encoding="utf-8-sig")
        scored_ids = set(existing["post_index"])
        logging.info(f"  Checkpoint: {len(scored_ids):,} posts already scored — resuming.")

    to_score = posts[~posts.index.isin(scored_ids)].copy()
    if len(to_score) == 0:
        logging.info("  All posts already scored.")
        return pd.read_csv(checkpoint_path, encoding="utf-8-sig")

    logging.info(f"  Scoring {len(to_score):,} posts (batch_size={batch_size})…")

    results = []
    texts = to_score["combined_text"].tolist()
    indices = to_score.index.tolist()
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num in range(n_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_idx = indices[start:end]

        try:
            preds = clf(batch_texts, batch_size=batch_size)
        except Exception as e:
            logging.warning(f"  Batch {batch_num} failed: {e} — skipping")
            continue

        for post_idx, pred in zip(batch_idx, preds):
            p_suicidal = pred["score"] if pred["label"] == "LABEL_1" else 1.0 - pred["score"]
            results.append({
                "post_index":    post_idx,
                "author":        posts.at[post_idx, "author"],
                "offset_from_cd1": posts.at[post_idx, "offset_from_cd1"],
                "subreddit":     posts.at[post_idx, "subreddit"],
                "p_suicidal":    p_suicidal,
            })

        # Checkpoint every N posts
        if (batch_num + 1) % max(1, checkpoint_every // batch_size) == 0 \
                or (batch_num + 1) == n_batches:
            batch_df = pd.DataFrame(results)
            mode = "a" if checkpoint_path.exists() else "w"
            header = not checkpoint_path.exists()
            batch_df.to_csv(checkpoint_path, mode=mode, header=header,
                            index=False, encoding="utf-8-sig")
            results = []
            pct = 100 * end / len(texts)
            n_done = len(scored_ids) + end
            logging.info(
                f"  [{n_done:,} total scored] {end:,}/{len(texts):,} this run "
                f"({pct:.1f}%) — checkpoint saved"
            )

    if results:  # flush any remainder
        pd.DataFrame(results).to_csv(
            checkpoint_path, mode="a", header=False, index=False, encoding="utf-8-sig"
        )

    scores = pd.read_csv(checkpoint_path, encoding="utf-8-sig")
    logging.info(f"  Done. {len(scores):,} posts scored total.")
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def identify_condition_users(
    interim_dir: Path,
    condition: str,
    candidate_users: set,
) -> set:
    """Identify users with a condition via regex self-report in post text."""
    regex = CONDITION_REGEX[condition]
    posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_with_anchors_*.csv")
    if posts_path is None:
        posts_path = find_latest_file(interim_dir, "posts_all_users_preprocessed_*.csv")
    if posts_path is None:
        raise FileNotFoundError("No posts_all_users_preprocessed_*.csv found.")

    posts = pd.read_csv(posts_path, encoding="utf-8-sig", usecols=["author", "text"],
                        low_memory=False)
    posts = posts[posts["author"].astype(str).isin(candidate_users)].copy()
    posts["author"] = posts["author"].astype(str)
    mask = posts["text"].fillna("").str.contains(regex, regex=True)
    condition_users = set(posts[mask]["author"].unique())
    logging.info(f"  {condition.upper()} users (regex): {len(condition_users)}")
    return condition_users


def assign_phase_modulo(offset: float, period: float) -> str | None:
    """Map offset to cycle phase using per-user period (modulo arithmetic).

    offset=35, period=28 → 35%28=7 → Follicular.
    offset=-5, period=28 → -5%28=23 → Luteal (premenstrual, correct).
    """
    offset_mod = offset % period
    for phase, (lo, hi) in PHASE_BOUNDARIES.items():
        if lo <= offset_mod < hi:
            return phase
    return None  # falls in gap between defined phases


def run_analysis(
    scores: pd.DataFrame,
    user_periods: dict,
    condition_users: set,
    condition_label: str,
    output_dir: Path,
    timestamp: str,
):
    """Apply phase labels, compare condition vs control, plot results.

    scores DataFrame must have: author, offset_from_cd1, subreddit, p_suicidal
    """
    scores = scores.copy()
    scores["author"] = scores["author"].astype(str)

    # Condition vs Control label
    scores["group"] = scores["author"].apply(
        lambda u: condition_label if u in condition_users else "Control"
    )

    # ── Step 1: assign phase using adaptive per-user boundaries ───────────
    # Uses create_adaptive_phases(user_period) — same logic as existing
    # analysis scripts. Follicular stretches to fill the cycle; Menstrual=4d,
    # Ovulation=3d, Luteal=14d are fixed. Phase length = theoretical duration.
    from src.visualization import create_adaptive_phases

    default_period = 28.0
    scores["period"] = scores["author"].map(user_periods).fillna(default_period)
    scores["offset_mod"] = scores["offset_from_cd1"] % scores["period"]

    def assign_adaptive_phase(offset_mod: float, period: float) -> str | None:
        phases = create_adaptive_phases(period)
        for phase_name, (start, end) in phases.items():
            if start <= offset_mod <= end:
                return phase_name
        return None

    scores["phase"] = scores.apply(
        lambda r: assign_adaptive_phase(r["offset_mod"], r["period"]), axis=1
    )
    scores = scores[scores["phase"].notna()].copy()

    n_condition = scores[scores["group"] == condition_label]["author"].nunique()
    n_control = scores[scores["group"] == "Control"]["author"].nunique()
    logging.info(
        f"\n  Posts with phase: {len(scores):,} | "
        f"{condition_label} users: {n_condition} | Control users: {n_control}"
    )
    if n_condition < 5:
        logging.warning(f"  Only {n_condition} {condition_label} users — stats will be unreliable.")

    # ── Step 2: count suicidal posts per phase / theoretical phase length ────
    # A post is "suicidal" if p_suicidal > 0.5 (model's decision threshold).
    # Denominator is theoretical phase length (not observed days) so that
    # longer phases don't artificially inflate the rate.
    scores["is_suicidal"] = (scores["p_suicidal"] > 0.5).astype(int)

    def get_phase_length(phase_name: str, period: float) -> float:
        phases = create_adaptive_phases(period)
        if phase_name not in phases:
            return np.nan
        start, end = phases[phase_name]
        return end - start + 1

    phase_counts = (
        scores.groupby(["author", "group", "phase"])["is_suicidal"]
        .agg(n_suicidal="sum", n_posts="count")
        .reset_index()
    )
    phase_counts["period"] = phase_counts["author"].map(user_periods).fillna(default_period)
    phase_counts["phase_length"] = phase_counts.apply(
        lambda r: get_phase_length(r["phase"], r["period"]), axis=1
    )
    # daily_rate = suicidal posts per day in this phase for this user
    phase_counts["daily_rate"] = phase_counts["n_suicidal"] / phase_counts["phase_length"]

    # ── Step 3: z-score normalization within user ──────────────────────────
    # For each user, compute mean and std of daily_rate across all their phases.
    # z = (rate - user_mean) / user_std
    # 0 = personal baseline; positive = above baseline for that phase.
    user_stats = (
        phase_counts.groupby("author")["daily_rate"]
        .agg(user_mean="mean", user_std="std")
        .reset_index()
    )
    user_phase_raw = phase_counts.merge(user_stats, on="author")
    # Drop users with std=0 (same rate in all phases — no within-user variation)
    user_phase_raw = user_phase_raw[user_phase_raw["user_std"] > 0].copy()
    user_phase_raw["normalized_rate"] = (
        (user_phase_raw["daily_rate"] - user_phase_raw["user_mean"])
        / user_phase_raw["user_std"]
    )
    user_phase = user_phase_raw.copy()

    n_dropped = scores["author"].nunique() - user_phase["author"].nunique()
    logging.info(
        f"  Users after z-score normalization: {user_phase['author'].nunique():,} "
        f"(dropped {n_dropped} with no within-user rate variation)"
    )

    # ── Group summary ──────────────────────────────────────────────────────
    group_phase = (
        user_phase.groupby(["group", "phase"])["normalized_rate"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    group_phase.columns = ["group", "phase", "mean", "sem", "n"]
    group_phase["phase"] = pd.Categorical(group_phase["phase"],
                                           categories=PHASE_ORDER, ordered=True)
    group_phase = group_phase.sort_values("phase")

    logging.info("\n  Group × Phase z-scored daily suicidality rate (user-level means):")
    logging.info("  (0 = user's personal baseline; positive = above baseline)")
    logging.info(group_phase.to_string(index=False))

    # ── Statistical tests ──────────────────────────────────────────────────
    tag = f"{condition_label.lower()}_suicidality"
    report_lines = [
        f"SUICIDALITY ANALYSIS — {condition_label} vs Control",
        "=" * 60,
        "Metric: z-scored daily suicidal post rate (within-user normalization)",
        "  = (count posts with p_suicidal>0.5 in phase / theoretical phase length - user_mean) / user_std",
        "  0 = user's personal baseline; positive = above baseline",
        "",
    ]

    for group in [condition_label, "Control"]:
        report_lines.append(f"{group}:")
        for phase in PHASE_ORDER:
            row = group_phase[
                (group_phase["group"] == group) & (group_phase["phase"] == phase)
            ]
            if len(row):
                r = row.iloc[0]
                report_lines.append(
                    f"  {phase:12s}  mean={r['mean']:.4f}  "
                    f"sem={r['sem']:.4f}  n_users={int(r['n'])}"
                )
        report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append(
        "Within-group phase tests (ttest_1samp vs 0): "
        "is the z-score significantly different from personal baseline?"
    )
    for group in [condition_label, "Control"]:
        report_lines.append(f"\n  {group}:")
        for phase in PHASE_ORDER:
            rates = user_phase[
                (user_phase["group"] == group) & (user_phase["phase"] == phase)
            ]["normalized_rate"].dropna()
            if len(rates) >= 5:
                t, p = stats.ttest_1samp(rates, 0.0)
                report_lines.append(
                    f"    {phase:12s}  mean={rates.mean():+.4f}  "
                    f"t={t:.3f}  p={p:.4f}  n={len(rates)}"
                )

    report_lines.append("")
    report_lines.append(
        f"Between-group tests ({condition_label} vs Control within each phase, Mann-Whitney):"
    )
    for phase in PHASE_ORDER:
        cond = user_phase[
            (user_phase["group"] == condition_label) & (user_phase["phase"] == phase)
        ]["normalized_rate"].dropna()
        ctrl = user_phase[
            (user_phase["group"] == "Control") & (user_phase["phase"] == phase)
        ]["normalized_rate"].dropna()
        if len(cond) >= 5 and len(ctrl) >= 5:
            _, p = stats.mannwhitneyu(cond, ctrl, alternative="two-sided")
            diff = cond.mean() - ctrl.mean()
            report_lines.append(
                f"  {phase:12s}  {condition_label} mean={cond.mean():+.4f}  "
                f"Control mean={ctrl.mean():+.4f}  diff={diff:+.4f}  p={p:.4f}"
            )

    report_text = "\n".join(report_lines)
    logging.info("\n" + report_text)

    rpt_path = output_dir / f"{tag}_report_{timestamp}.txt"
    rpt_path.write_text(report_text)
    logging.info(f"\n  Report → {rpt_path.name}")

    # Save user-phase normalized rates for further analysis
    out_csv = output_dir / f"{tag}_user_phase_rates_{timestamp}.csv"
    user_phase[["author", "group", "phase", "n_suicidal", "n_posts", "daily_rate", "user_mean", "user_std", "normalized_rate"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    logging.info(f"  User-phase rates → {out_csv.name}")

    # Plots
    _plot_phase_bars(group_phase, condition_label, output_dir, tag, timestamp)
    _plot_trajectory(scores, user_phase, user_periods, condition_label, output_dir, tag, timestamp)


def _plot_phase_bars(group_phase, condition_label, output_dir, tag, timestamp):
    """Bar chart: mean suicidality per phase, condition vs control."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(PHASE_ORDER))
    width = 0.35
    colors = {condition_label: "#d62728", "Control": "#1f77b4"}

    for i, group in enumerate([condition_label, "Control"]):
        sub = group_phase[group_phase["group"] == group].set_index("phase")
        means = [sub.loc[p, "mean"] if p in sub.index else np.nan for p in PHASE_ORDER]
        sems  = [sub.loc[p, "sem"]  if p in sub.index else np.nan for p in PHASE_ORDER]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=group,
               color=colors[group], alpha=0.8, yerr=sems, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_ORDER)
    ax.set_xlabel("Cycle Phase")
    ax.set_ylabel("Z-scored daily suicidality rate (0 = personal baseline)")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Suicidality Language by Cycle Phase\n"
        f"{condition_label} vs Control (±SEM, z-scored within-user)"
    )
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"{tag}_phase_barplot_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Barplot → {path.name}")


def _plot_trajectory(scores, user_phase, user_periods, condition_label, output_dir, tag, timestamp):
    """Line plot: z-scored daily suicidality by cycle day, condition vs control.

    Each post is normalized by the user's personal mean p_suicidal so the
    trajectory reflects within-user deviation from baseline across the cycle.
    """
    scores = scores.copy()
    scores["period"] = scores["author"].map(user_periods).fillna(28.0)
    scores["offset_mod"] = scores["offset_from_cd1"] % scores["period"]
    scores["cycle_day"] = scores["offset_mod"].apply(np.floor).astype(int)

    # Keep only users who passed z-score normalization (scores already has group column)
    valid_authors = set(user_phase["author"].unique())
    scores = scores[scores["author"].isin(valid_authors)].copy()

    # Per-user mean suicidal-post indicator (for within-user normalization)
    user_stats_post = (
        scores.groupby("author")["is_suicidal"]
        .agg(user_mean_psui="mean", user_std_psui="std")
        .reset_index()
    )
    scores = scores.merge(user_stats_post, on="author")
    # Z-score each post's suicidal indicator relative to user's baseline
    scores["deviation"] = np.where(
        scores["user_std_psui"] > 0,
        (scores["is_suicidal"] - scores["user_mean_psui"]) / scores["user_std_psui"],
        0.0,
    )

    # Average deviation per group × cycle_day
    traj = (
        scores.groupby(["group", "cycle_day"])["deviation"]
        .mean()
        .reset_index()
        .rename(columns={"cycle_day": "offset_bin"})
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {condition_label: "#d62728", "Control": "#1f77b4"}

    for group in [condition_label, "Control"]:
        sub = traj[traj["group"] == group].sort_values("offset_bin")
        ax.plot(sub["offset_bin"], sub["deviation"],
                label=group, color=colors[group], lw=2, alpha=0.85)

    # Phase shading using fixed PHASE_BOUNDARIES for reference
    phase_colors = {
        "Menstrual": "#fee0d2", "Follicular": "#e2f4e2",
        "Ovulation": "#fff7bb", "Luteal": "#e0ecff",
    }
    for phase, (lo, hi) in PHASE_BOUNDARIES.items():
        ax.axvspan(lo, hi, alpha=0.18, color=phase_colors[phase])
        ax.text((lo + hi) / 2, 0.02, phase, ha="center", fontsize=8,
                style="italic", transform=ax.get_xaxis_transform())

    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=1, linestyle="--", label="CD1")
    ax.set_xlabel("Cycle day (modulo period)")
    ax.set_ylabel("Z-score deviation from personal baseline")
    ax.set_title(
        f"Suicidality Trajectory Across Cycle\n"
        f"{condition_label} vs Control (z-scored within-user, 1-day bins)"
    )
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"{tag}_trajectory_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Trajectory → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument(
        "--mode", choices=["score", "analyze", "both"], default="both",
        help="'score' = run HuggingFace model only; "
             "'analyze' = load existing scores and run analysis; "
             "'both' = score then analyze (default).",
    )
    p.add_argument(
        "--window-months", type=float, default=3.0,
        help="Time window around CD1 in months (default: 3).",
    )
    p.add_argument(
        "--condition", choices=["pmdd", "adhd", "depression"], default="pmdd",
        help="Condition to compare against controls in analysis (default: pmdd).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_config(args.config)
    interim_dir = Path(cfg["paths"]["interim"])
    output_dir = ROOT / "reports" / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = interim_dir / "suicidality_scores_checkpoint.csv"

    # ── Load consensus users + periods (always needed) ─────────────────────
    logging.info("\n[1] Loading consensus users…")
    consensus_users, user_periods = load_consensus_users(interim_dir)

    # ══ PHASE 1: SCORING ══════════════════════════════════════════════════
    if args.mode in ("score", "both"):
        logging.info(f"\n[SCORING] Window: ±{args.window_months} months")

        logging.info("\n[2] Loading posts…")
        posts = load_posts_window(interim_dir, consensus_users, args.window_months)

        logging.info("\n[3] Loading suicidality model…")
        from transformers import pipeline as hf_pipeline
        clf = hf_pipeline(
            "text-classification",
            model="sentinet/suicidality",
            device=0 if args.device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )
        logging.info("  Model loaded.")

        logging.info(f"\n[4] Scoring {len(posts):,} posts…")
        scores = score_posts(
            posts, clf, args.batch_size, checkpoint_path, args.checkpoint_every
        )

    # ══ PHASE 2: ANALYSIS ═════════════════════════════════════════════════
    if args.mode in ("analyze", "both"):
        logging.info("\n[ANALYSIS]")

        if args.mode == "analyze":
            if not checkpoint_path.exists():
                logging.error(
                    f"No scores found at {checkpoint_path}. "
                    "Run with --mode score first."
                )
                sys.exit(1)
            scores = pd.read_csv(checkpoint_path, encoding="utf-8-sig")
            logging.info(f"  Loaded {len(scores):,} scored posts from checkpoint.")

        logging.info(f"\n[5] Identifying {args.condition.upper()} users…")
        candidate_users = set(scores["author"].astype(str).unique())
        condition_users = identify_condition_users(
            interim_dir, args.condition, candidate_users
        )
        condition_label = args.condition.upper() if args.condition != "depression" \
            else "Depression"

        logging.info("\n[6] Running analysis…")
        run_analysis(
            scores, user_periods, condition_users, condition_label,
            output_dir, timestamp
        )

    logging.info(f"\nOutputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
