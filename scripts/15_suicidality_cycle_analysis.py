"""Step 15 — Suicidality Scores Across the Menstrual Cycle
==========================================================
Two-phase design:

  PHASE 1 — SCORING (slow, run once overnight)
    Score all posts within a time window for chosen users using a
    HuggingFace mental health classifier. Saves raw scores to CSV.
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

Model: https://huggingface.co/ourafla/mental-health-bert-finetuned
  4-class BERT: LABEL_0=Anxiety, LABEL_1=Depression, LABEL_2=Normal, LABEL_3=Suicidal
  We save P(LABEL_3) as a continuous suicidality score per post.
  Unlike the previous binary model, this does NOT mislabel IVF/fertility language as suicidal.

  Legacy model (still selectable via --model):
    sentinet/suicidality — binary ELECTRA, LABEL_1=suicidal

Checkpoint:
  Scoring saves progress every --checkpoint-every posts.
  If interrupted, re-run the same command — already-scored posts are skipped.
  Each model uses its own checkpoint file to avoid mixing scores.
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

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
SPLIT_LUTEAL_PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Early Luteal", "Late Luteal"]
PMDD_PHASE_ORDER = ["Perimenstrual", "Midfollicular", "Periovulatory", "Early Luteal", "Midluteal"]

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


def load_all_timeline_users(interim_dir: Path, fixed_period: float) -> tuple[set, dict]:
    """Returns (set of user IDs, dict user -> fixed_period) for ALL timeline users."""
    timeline_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    if timeline_path is None:
        timeline_path = find_latest_file(interim_dir, "timeline_with_offsets_*.csv")
    if timeline_path is None:
        raise FileNotFoundError("No timeline_with_offsets_*.csv found.")
    df = pd.read_csv(timeline_path, usecols=["author"], low_memory=False)
    users = set(df["author"].astype(str).unique())
    user_periods = {u: fixed_period for u in users}
    logging.info(f"  All timeline users: {len(users):,} | fixed period: {fixed_period} days")
    return users, user_periods


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
            # Multi-class model returns a list of dicts per post (top_k=None)
            if isinstance(pred, list):
                suicidal = next((d["score"] for d in pred if d["label"] == "LABEL_3"), 0.0)
            else:
                # Legacy binary model (sentinet/suicidality): LABEL_1 = suicidal
                suicidal = pred["score"] if pred["label"] == "LABEL_1" else 1.0 - pred["score"]
            p_suicidal = suicidal
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


def run_analysis(
    scores: pd.DataFrame,
    user_periods: dict,
    condition_users: set,
    condition_label: str,
    output_dir: Path,
    timestamp: str,
    phase_system: str = "adaptive",
    split_luteal: bool = False,
    score_method: str = "binary",
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

    # ── Step 1: assign phase using per-user boundaries ────────────────────
    from src.visualization import (
        create_adaptive_phases,
        create_pmdd_phases,
        assign_pmdd_phase,
    )

    default_period = 28.0
    scores["period"] = scores["author"].map(user_periods).fillna(default_period)
    scores["offset_mod"] = scores["offset_from_cd1"] % scores["period"]

    # Build phase lookup cache once per unique period value
    if phase_system == "pmdd":
        phase_cache: dict[float, dict] = {
            p: create_pmdd_phases(p) for p in scores["period"].unique()
        }
        active_phase_order = PMDD_PHASE_ORDER

        def _assign_phase(offset_mod: float, period: float) -> str | None:
            return assign_pmdd_phase(offset_mod, phase_cache[period], period)
    else:
        phase_cache = {
            p: create_adaptive_phases(p, split_luteal=split_luteal)
            for p in scores["period"].unique()
        }
        active_phase_order = SPLIT_LUTEAL_PHASE_ORDER if split_luteal else PHASE_ORDER

        def _assign_phase(offset_mod: float, period: float) -> str | None:
            for phase_name, (start, end) in phase_cache[period].items():
                if start <= offset_mod <= end:
                    return phase_name
            return None

    scores["phase"] = scores.apply(
        lambda r: _assign_phase(r["offset_mod"], r["period"]), axis=1
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

    # ── Step 2: compute per-user per-phase rate ───────────────────────────
    # Two methods selectable via score_method:
    #   "binary"     — is_suicidal = p_suicidal > 0.5; rate = n_suicidal / phase_length_days
    #                  More sensitive to rare but high-confidence events.
    #   "continuous" — rate = mean(p_suicidal) across all posts in phase
    #                  Uses full probability distribution; better when posts are dense.
    scores["is_suicidal"] = (scores["p_suicidal"] > 0.5).astype(int)

    def get_phase_length(phase_name: str, period: float) -> float:
        phases = phase_cache.get(period, create_adaptive_phases(period))
        if phase_name not in phases:
            return np.nan
        start, end = phases[phase_name]
        # Perimenstrual uses a negative-start sentinel: e.g. (-3, 2) → 3+2+1=6 days
        return abs(start) + end + 1 if start < 0 else end - start + 1

    if score_method == "continuous":
        # mean p_suicidal per user per phase (no threshold)
        phase_counts = (
            scores.groupby(["author", "group", "phase"])
            .agg(mean_p=("p_suicidal", "mean"), n_suicidal=("is_suicidal", "sum"), n_posts=("is_suicidal", "count"))
            .reset_index()
        )
        phase_counts["daily_rate"] = phase_counts["mean_p"]
    else:
        # binary: n_suicidal / theoretical phase length
        phase_counts = (
            scores.groupby(["author", "group", "phase"])["is_suicidal"]
            .agg(n_suicidal="sum", n_posts="count")
            .reset_index()
        )
        phase_counts["period"] = phase_counts["author"].map(user_periods).fillna(default_period)
        phase_counts["phase_length"] = phase_counts.apply(
            lambda r: get_phase_length(r["phase"], r["period"]), axis=1
        )
        phase_counts["daily_rate"] = phase_counts["n_suicidal"] / phase_counts["phase_length"]

    logging.info(f"  Score method: {score_method}")

    # ── Step 3: per-phase excess relative to all other phases ─────────────
    # excess_i = rate_i − mean(rate for all other phases)
    # Users must have data in the target phase AND ≥1 other phase.
    excess_rows = []
    for target_phase in active_phase_order:
        target = phase_counts[phase_counts["phase"] == target_phase][
            ["author", "group", "phase", "n_suicidal", "n_posts", "daily_rate"]
        ].copy()
        other_mean = (
            phase_counts[phase_counts["phase"] != target_phase]
            .groupby("author")["daily_rate"]
            .mean()
            .reset_index()
            .rename(columns={"daily_rate": "other_mean_rate"})
        )
        merged = target.merge(other_mean, on="author", how="inner")
        merged["excess"] = merged["daily_rate"] - merged["other_mean_rate"]
        excess_rows.append(merged)

    user_phase = pd.concat(excess_rows, ignore_index=True)

    logging.info(
        f"  Users in excess analysis: {user_phase['author'].nunique():,} "
        f"(require data in target phase + ≥1 other phase)"
    )

    # ── Group summary ──────────────────────────────────────────────────────
    group_phase = (
        user_phase.groupby(["group", "phase"])["excess"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    group_phase.columns = ["group", "phase", "mean", "sem", "n"]
    group_phase["phase"] = pd.Categorical(group_phase["phase"],
                                           categories=active_phase_order, ordered=True)
    group_phase = group_phase.sort_values("phase")

    logging.info("\n  Group × Phase excess suicidality rate (phase − mean other phases):")
    logging.info("  (0 = same as other phases; positive = elevated relative to rest of cycle)")
    logging.info(group_phase.to_string(index=False))

    # ── Statistical tests ──────────────────────────────────────────────────
    tag = f"{condition_label.lower()}_suicidality"
    report_lines = [
        f"SUICIDALITY ANALYSIS — {condition_label} vs Control",
        "=" * 60,
        "Metric: per-phase excess suicidality rate",
        "  = daily_rate_in_phase − mean(daily_rate_in_all_other_phases)",
        "  daily_rate = suicidal_posts_in_phase / theoretical_phase_length_days",
        "  0 = same rate as rest of cycle; positive = elevated relative to other phases",
        "",
    ]

    for group in [condition_label, "Control"]:
        report_lines.append(f"{group}:")
        for phase in active_phase_order:
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
        "is the excess significantly different from zero?"
    )
    for group in [condition_label, "Control"]:
        report_lines.append(f"\n  {group}:")
        for phase in active_phase_order:
            vals = user_phase[
                (user_phase["group"] == group) & (user_phase["phase"] == phase)
            ]["excess"].dropna()
            if len(vals) >= 5:
                t, p = stats.ttest_1samp(vals, 0.0)
                report_lines.append(
                    f"    {phase:12s}  mean={vals.mean():+.4f}  "
                    f"t={t:.3f}  p={p:.4f}  n={len(vals)}"
                )

    report_lines.append("")
    report_lines.append(
        f"Between-group tests ({condition_label} vs Control within each phase, Mann-Whitney):"
    )
    for phase in active_phase_order:
        cond = user_phase[
            (user_phase["group"] == condition_label) & (user_phase["phase"] == phase)
        ]["excess"].dropna()
        ctrl = user_phase[
            (user_phase["group"] == "Control") & (user_phase["phase"] == phase)
        ]["excess"].dropna()
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

    # Save user-phase excess rates for further analysis
    out_csv = output_dir / f"{tag}_user_phase_rates_{timestamp}.csv"
    user_phase[["author", "group", "phase", "n_suicidal", "n_posts", "daily_rate", "other_mean_rate", "excess"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    logging.info(f"  User-phase rates → {out_csv.name}")

    # Plots
    _plot_phase_bars(group_phase, condition_label, output_dir, tag, timestamp, active_phase_order)
    if phase_system == "pmdd":
        ref_period = min(phase_cache.keys(), key=lambda p: abs(p - 28.0))
        ref_phases = phase_cache[ref_period]
    else:
        ref_phases = phase_cache.get(28.0, create_adaptive_phases(28.0))
    _plot_trajectory(scores, user_phase, user_periods, phase_cache, ref_phases, condition_label, output_dir, tag, timestamp, phase_system=phase_system)


def _plot_phase_bars(group_phase, condition_label, output_dir, tag, timestamp, phase_order=None):
    """Bar chart: mean suicidality per phase, condition vs control."""
    if phase_order is None:
        phase_order = PHASE_ORDER
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(phase_order))
    width = 0.35
    colors = {condition_label: "#d62728", "Control": "#1f77b4"}

    for i, group in enumerate([condition_label, "Control"]):
        sub = group_phase[group_phase["group"] == group].set_index("phase")
        means = [sub.loc[p, "mean"] if p in sub.index else np.nan for p in phase_order]
        sems  = [sub.loc[p, "sem"]  if p in sub.index else np.nan for p in phase_order]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=group,
               color=colors[group], alpha=0.8, yerr=sems, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_order)
    ax.set_xlabel("Cycle Phase")
    ax.set_ylabel("Excess suicidality rate vs other phases (0 = no elevation)")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Suicidality Language by Cycle Phase\n"
        f"{condition_label} vs Control (±SEM, excess over other phases)"
    )
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"{tag}_phase_barplot_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Barplot → {path.name}")


def _plot_trajectory(scores, user_phase, user_periods, phase_cache, ref_phases, condition_label, output_dir, tag, timestamp, phase_system="adaptive"):
    """Line plot: z-scored daily suicidality by cycle day, condition vs control.

    Each post is normalized by the user's personal mean p_suicidal so the
    trajectory reflects within-user deviation from baseline across the cycle.
    """
    scores = scores.copy()
    scores["period"] = scores["author"].map(user_periods).fillna(28.0)
    scores["offset_mod"] = scores["offset_from_cd1"] % scores["period"]
    scores["cycle_day"] = scores["offset_mod"].apply(np.floor).astype(int)

    # Keep only users present in the excess analysis
    valid_authors = set(user_phase["author"].unique())
    scores = scores[scores["author"].isin(valid_authors)].copy()

    # Per-user mean suicidal-post indicator; deviation = mean-centred (no std division)
    # Consistent with the excess approach: avoids noisy std estimates from few data points.
    user_stats_post = (
        scores.groupby("author")["is_suicidal"]
        .mean()
        .reset_index()
        .rename(columns={"is_suicidal": "user_mean_psui"})
    )
    scores = scores.merge(user_stats_post, on="author")
    scores["deviation"] = scores["is_suicidal"] - scores["user_mean_psui"]

    # Average per user per cycle_day first, then across users per group.
    # This prevents prolific users from dominating the trajectory.
    user_day_avg = (
        scores.groupby(["author", "group", "cycle_day"])["deviation"]
        .mean()
        .reset_index()
    )
    traj = (
        user_day_avg.groupby(["group", "cycle_day"])["deviation"]
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

    # Phase shading — colours depend on phase system
    adaptive_phase_colors = {
        "Menstrual": "#fee0d2", "Follicular": "#e2f4e2",
        "Ovulation": "#fff7bb", "Luteal": "#e0ecff",
    }
    pmdd_phase_colors = {
        "Perimenstrual":  "#fee0d2",
        "Midfollicular":  "#e2f4e2",
        "Periovulatory":  "#fff7bb",
        "Early Luteal":   "#d4eaff",
        "Midluteal":      "#e0ecff",
    }
    phase_colors = pmdd_phase_colors if phase_system == "pmdd" else adaptive_phase_colors

    # Use the cycle length from any entry in phase_cache for wrap maths
    ref_cl = int(round(list(phase_cache.keys())[0]))

    for phase, (lo, hi) in ref_phases.items():
        color = phase_colors.get(phase, "#eeeeee")
        if lo < 0:  # wrap-around (Perimenstrual)
            # Tail: e.g. days 25–27 (end of previous cycle display)
            ax.axvspan(ref_cl + lo, ref_cl, alpha=0.18, color=color)
            # Head: days 0–2
            ax.axvspan(0, hi, alpha=0.18, color=color)
            ax.text(hi / 2, 0.02, phase, ha="center", fontsize=8,
                    style="italic", transform=ax.get_xaxis_transform())
        else:
            ax.axvspan(lo, hi, alpha=0.18, color=color)
            ax.text((lo + hi) / 2, 0.02, phase, ha="center", fontsize=8,
                    style="italic", transform=ax.get_xaxis_transform())

    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=1, linestyle="--", label="CD1")
    ax.set_xlabel("Cycle day (modulo period)")
    ax.set_ylabel("Mean-centred suicidality (deviation from personal baseline)")
    ax.set_title(
        f"Suicidality Trajectory Across Cycle\n"
        f"{condition_label} vs Control (mean-centred within-user, 1-day bins)"
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
        help="Time window around CD1 in months for scoring (default: 3).",
    )
    p.add_argument(
        "--analysis-window-days", type=float, default=None,
        help="Restrict analysis to posts within ±N days of CD1 (default: use all scored posts).",
    )
    p.add_argument(
        "--condition", choices=["pmdd", "adhd", "depression"], default="pmdd",
        help="Condition to compare against controls in analysis (default: pmdd).",
    )
    p.add_argument(
        "--phase-system", choices=["adaptive", "pmdd"], default="adaptive",
        help="'adaptive' = 4-phase biological system; "
             "'pmdd' = 5-phase Eisenlohr-Moul 2023 system (default: adaptive).",
    )
    p.add_argument(
        "--user-source", choices=["consensus", "all"], default="consensus",
        help="'consensus' = only users with detected cycle lengths (default); "
             "'all' = all timeline users, condition identified before scoring.",
    )
    p.add_argument(
        "--fixed-period", type=float, default=29.0,
        help="Fixed cycle length (days) applied to every user when --user-source all (default: 29).",
    )
    p.add_argument(
        "--split-luteal", action="store_true", default=False,
        help="Split the Luteal phase into Early Luteal (7d) and Late Luteal (7d).",
    )
    p.add_argument(
        "--subreddit-group", default=None,
        help="Restrict analysis to a subreddit group defined in configs/base.yaml "
             "(e.g. 'suicidality_calibrated'). If not set, use all posts.",
    )
    p.add_argument(
        "--model", default="ourafla/mental-health-bert-finetuned",
        help="HuggingFace model for suicidality scoring. "
             "Default: ourafla/mental-health-bert-finetuned (4-class, LABEL_3=Suicidal). "
             "Legacy: sentinet/suicidality (binary, LABEL_1=suicidal).",
    )
    p.add_argument(
        "--score-method", choices=["binary", "continuous"], default="binary",
        help="'binary' = count posts with p_suicidal>0.5 / phase_length (default); "
             "'continuous' = mean(p_suicidal) across all posts in phase.",
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

    # Checkpoint path encodes model + user-source + window to avoid mixing datasets
    model_slug = args.model.replace("/", "_")
    window_tag = f"w{args.window_months:.0f}m"
    if args.user_source == "all":
        checkpoint_path = interim_dir / f"suicidality_scores_{model_slug}_fp{int(args.fixed_period)}_{window_tag}_checkpoint.csv"
    else:
        checkpoint_path = interim_dir / f"suicidality_scores_{model_slug}_{window_tag}_checkpoint.csv"

    # ── Load users + periods ───────────────────────────────────────────────
    logging.info("\n[1] Loading users…")
    if args.user_source == "all":
        all_users, user_periods = load_all_timeline_users(interim_dir, args.fixed_period)
    else:
        all_users, user_periods = load_consensus_users(interim_dir)

    # ══ PHASE 1: SCORING ══════════════════════════════════════════════════
    if args.mode in ("score", "both"):
        logging.info(f"\n[SCORING] Window: ±{args.window_months} months")

        if args.user_source == "all":
            # Identify condition users first so we score only their posts
            logging.info(f"\n[2] Identifying {args.condition.upper()} users from all timeline users…")
            users_to_score = identify_condition_users(interim_dir, args.condition, all_users)
            logging.info(f"  Scoring only {len(users_to_score):,} {args.condition.upper()} users.")
            logging.info("\n[3] Loading posts…")
        else:
            users_to_score = all_users
            logging.info("\n[2] Loading posts…")

        posts = load_posts_window(interim_dir, users_to_score, args.window_months)

        step_model = 4 if args.user_source == "all" else 3
        logging.info(f"\n[{step_model}] Loading suicidality model…")
        from transformers import pipeline as hf_pipeline
        # Multi-class model (ourafla/mental-health-bert-finetuned) needs top_k=None
        # so that all 4 label scores are returned per post.
        # Legacy binary model (sentinet/suicidality) uses default top_k=1.
        is_multiclass = args.model != "sentinet/suicidality"
        clf = hf_pipeline(
            "text-classification",
            model=args.model,
            device=0 if args.device == "cuda" else -1,
            truncation=True,
            max_length=512,
            top_k=None if is_multiclass else 1,
        )
        logging.info(f"  Model loaded: {args.model}")

        step_score = 5 if args.user_source == "all" else 4
        logging.info(f"\n[{step_score}] Scoring {len(posts):,} posts…")
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

        if args.analysis_window_days is not None:
            before = len(scores)
            scores = scores[scores["offset_from_cd1"].abs() <= args.analysis_window_days].copy()
            logging.info(
                f"  Analysis window: ±{args.analysis_window_days:.0f} days — "
                f"{len(scores):,} posts (dropped {before - len(scores):,})"
            )

        if args.subreddit_group is not None:
            allowed = cfg.get("subreddits", {}).get(args.subreddit_group)
            if not allowed:
                logging.error(f"Subreddit group '{args.subreddit_group}' not found in config.")
                sys.exit(1)
            before = len(scores)
            scores = scores[scores["subreddit"].isin(allowed)].copy()
            logging.info(
                f"  Subreddit filter '{args.subreddit_group}': "
                f"{len(scores):,} posts from {scores['author'].nunique():,} users "
                f"(dropped {before - len(scores):,} posts)"
            )

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
            output_dir, timestamp, phase_system=args.phase_system,
            split_luteal=args.split_luteal, score_method=args.score_method,
        )

    logging.info(f"\nOutputs → {output_dir}/")
    logging.info("Done.")


if __name__ == "__main__":
    main()
