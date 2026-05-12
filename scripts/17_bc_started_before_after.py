#!/usr/bin/env python3
"""Step 17: Before vs After starting BC pill — phase feature comparison.

For users who recently started a BC pill (LLM label: started_recently=True),
compares linguistic features by menstrual phase in two windows:

  BEFORE window:  [started_offset - 28, started_offset - 1]  — natural cycle
  AFTER  window:  [started_offset,       started_offset + 27] — on-pill

Phase assignment in both windows uses (offset_from_cd1 - started_offset) % 28,
so day 0 = first pill day.  Phase boundaries for 28-day cycle:
  Menstrual  0-3   (4 days)
  Follicular 4-10  (7 days)
  Ovulation  11-13 (3 days)
  Luteal     14-27 (14 days)

Input:
  data/processed/bc_wide_candidates_with_labels.csv  (LLM labels)
  data/interim/timeline_daily_aggregated_with_anchors_*.csv

Output:
  reports/bc_started_before_after_{timestamp}.png
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.io import find_latest_file

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

PHASE_ORDER  = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_COLORS = {
    "Menstrual":  "#d62728",
    "Follicular": "#ff7f0e",
    "Ovulation":  "#2ca02c",
    "Luteal":     "#1f77b4",
}
PHASE_WIDTHS = {"Menstrual": 1.5, "Follicular": 2.0, "Ovulation": 1.0, "Luteal": 4.5}

# 28-day cycle phase boundaries (0-indexed, relative to pill start)
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
    """Map 0-27 day position to phase name."""
    for phase, (lo, hi) in PHASE_BOUNDS_28.items():
        if lo <= day_in_cycle <= hi:
            return phase
    return None


def aggregate_phase_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    user_col: str = "author",
) -> pd.DataFrame:
    """Per-user phase mean → cross-user mean ± SEM.

    Returns DataFrame with columns: feature, phase, mean, sem, n_users.
    """
    records = []
    # Step 1: per-user mean per phase
    user_phase = (
        df.groupby([user_col, "phase"])[feature_cols]
        .mean()
        .reset_index()
    )
    # Step 2: cross-user stats per phase
    for feature in feature_cols:
        for phase in PHASE_ORDER:
            vals = user_phase.loc[user_phase["phase"] == phase, feature].dropna()
            if len(vals) < 3:
                continue
            records.append({
                "feature": feature,
                "phase": phase,
                "mean": vals.mean(),
                "sem": vals.sem(),
                "n_users": len(vals),
            })
    return pd.DataFrame(records)


def plot_before_after(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    features: list[str],
    n_before: int,
    n_after: int,
    output_path: Path,
    n_cols: int = 5,
) -> None:
    """Side-by-side before/after bar chart for each feature."""
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 4.2 * n_rows),
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        f"Feature Values by Phase — Before vs After Starting BC Pill\n"
        f"Before window: 28 days prior to pill start  |  "
        f"After window: 28 days post pill start  |  "
        f"n_before={n_before}, n_after={n_after} users",
        fontsize=10, fontweight="bold", y=1.002,
    )

    # Build x positions: paired bars per phase
    bar_w = 0.6
    gap_within = 0.1   # gap between before/after bar
    gap_phase  = 0.5   # gap between phase groups
    x_centers, phase_labels_x = [], []
    cur = 0.0
    for phase in PHASE_ORDER:
        # Two bars: before (left), after (right)
        x_before = cur
        x_after  = cur + bar_w + gap_within
        x_centers.append((phase, x_before, x_after))
        phase_labels_x.append((cur + bar_w / 2 + gap_within / 2, phase))
        cur += 2 * bar_w + gap_within + gap_phase

    for idx, feature in enumerate(features):
        ax = axes[idx // n_cols, idx % n_cols]

        b_data = before_df[before_df["feature"] == feature].set_index("phase")
        a_data = after_df[after_df["feature"] == feature].set_index("phase")

        for phase, xb, xa in x_centers:
            color = PHASE_COLORS[phase]
            # Before bar — hatched
            if phase in b_data.index:
                row = b_data.loc[phase]
                ax.bar(xb, row["mean"], width=bar_w, color=color, alpha=0.45,
                       hatch="///", edgecolor="black", linewidth=0.8,
                       yerr=row["sem"], capsize=3)
                ax.text(xb, row["mean"], f"n={int(row['n_users'])}",
                        ha="center", va="bottom", fontsize=5)
            # After bar — solid
            if phase in a_data.index:
                row = a_data.loc[phase]
                ax.bar(xa, row["mean"], width=bar_w, color=color, alpha=0.85,
                       edgecolor="black", linewidth=0.8,
                       yerr=row["sem"], capsize=3)
                ax.text(xa, row["mean"], f"n={int(row['n_users'])}",
                        ha="center", va="bottom", fontsize=5)

        ax.set_xticks([x for x, _ in phase_labels_x])
        ax.set_xticklabels([p[:3] for _, p in phase_labels_x], fontsize=7)
        ax.set_ylabel("z-score", fontsize=6)
        fname = feature.replace("_mean", "").replace("_", " ").title()
        ax.set_title(fname, fontsize=7, fontweight="bold", pad=3)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="grey", alpha=0.45, hatch="///", edgecolor="black", label="Before pill"),
        Patch(facecolor="grey", alpha=0.85, edgecolor="black", label="After pill"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    # Hide unused subplots
    for idx in range(len(features), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_path: str = "configs/base.yaml",
    labels_file: str = "data/processed/bc_wide_candidates_with_labels.csv",
    window: int = 28,
    original_features_only: bool = False,
    n_cols: int = 5,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 17: BC Started — Before vs After Phase Comparison")
    print("=" * 60)
    print()

    # ── [1] Load LLM labels ──────────────────────────────────────────────────
    print("[1] Loading LLM labels...")
    try:
        labels = pd.read_excel(labels_file, engine="openpyxl")
    except Exception:
        labels = pd.read_csv(labels_file, encoding="utf-8-sig")

    started = labels[(labels["is_bc_pill"] == True) & (labels["started_recently"] == True)].copy()
    print(f"    Started users: {started['author'].nunique():,} ({len(started):,} posts)")

    # started_offset is relative to the post day, not to CD1.
    # Actual start day in CD1 terms = offset_from_cd1 + started_offset
    # Only use posts within the ±90 day timeline window to estimate the event offset.
    started["actual_start_cd1"] = started["offset_from_cd1"] + started["started_offset"]
    started_in_window = started[started["offset_from_cd1"].between(-90, 90)]
    user_start = (
        started_in_window.groupby("author")["actual_start_cd1"]
        .median()
        .reset_index()
        .rename(columns={"actual_start_cd1": "pill_start_offset"})
    )
    user_start["pill_start_offset"] = user_start["pill_start_offset"].round().astype(int)
    print(f"    pill start offset range: {user_start['pill_start_offset'].min()} "
          f"to {user_start['pill_start_offset'].max()}")

    # ── [2] Load daily aggregated timeline ──────────────────────────────────
    print("\n[2] Loading daily aggregated timeline...")
    agg_path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    timeline = pd.read_csv(agg_path, encoding="utf-8-sig", low_memory=False)
    print(f"    {len(timeline):,} user-days, {timeline['author'].nunique():,} users")

    # Filter to started users present in timeline
    timeline = timeline.merge(user_start, on="author", how="inner")
    n_users = timeline["author"].nunique()
    print(f"    Started users found in timeline: {n_users:,}")
    if n_users == 0:
        print("    No users found. Check author IDs.")
        return 1

    # ── [3] Identify feature columns ────────────────────────────────────────
    meta_cols = {"author", "offset_from_cd1", "pill_start_offset"}
    all_feat = [
        c for c in timeline.columns
        if c not in meta_cols and c.endswith("_mean")
        and pd.api.types.is_numeric_dtype(timeline[c])
    ]
    if original_features_only:
        features = [f for f in ORIGINAL_FEATURES if f in all_feat]
    else:
        features = all_feat
    print(f"\n[3] Feature columns: {len(features)}")

    # ── [4] Compute phase relative to pill start ─────────────────────────────
    print("\n[4] Assigning phases relative to pill start date...")
    # day_rel = 0 on first pill day, negative = before, positive = after
    timeline["day_rel"] = (
        timeline["offset_from_cd1"] - timeline["pill_start_offset"]
    ).round().astype(int)

    # Per-user z-score normalisation (over full timeline before splitting)
    print("    Per-user z-score normalisation...")
    for feat in features:
        user_stats = timeline.groupby("author")[feat].agg(["mean", "std"])
        user_stats.columns = ["_mu", "_sigma"]
        timeline = timeline.join(user_stats, on="author")
        timeline[feat] = (timeline[feat] - timeline["_mu"]) / timeline["_sigma"].replace(0, np.nan)
        timeline.drop(columns=["_mu", "_sigma"], inplace=True)

    # Assign phase using day_rel % 28  (with modulo handling for negatives)
    timeline["day_in_cycle"] = timeline["day_rel"] % 28
    timeline["phase"] = timeline["day_in_cycle"].apply(assign_phase_28)
    timeline = timeline[timeline["phase"].notna()].copy()

    # ── [5] Split into before / after using full available timeline ─────────
    # Use ALL data before started_offset as before, ALL data from started_offset
    # onward as after — no artificial window cutoff.  The ±90-day timeline
    # provides the natural boundary on both sides.
    print(f"\n[5] Splitting on pill start date (using full ±90-day timeline)...")
    before_all = timeline[timeline["day_rel"] < 0].copy()
    after_all  = timeline[timeline["day_rel"] >= 0].copy()

    # Keep only users with data in BOTH windows
    users_with_both = set(before_all["author"]) & set(after_all["author"])
    print(f"    Users with data in before window: {before_all['author'].nunique():,}")
    print(f"    Users with data in after  window: {after_all['author'].nunique():,}")
    print(f"    Users with data in BOTH windows:  {len(users_with_both):,}  ← using these only")

    before = before_all[before_all["author"].isin(users_with_both)].copy()
    after  = after_all[after_all["author"].isin(users_with_both)].copy()

    n_before_users = before["author"].nunique()
    n_after_users  = after["author"].nunique()
    print(f"    Before window: {len(before):,} user-days from {n_before_users:,} users")
    print(f"    After  window: {len(after):,} user-days from {n_after_users:,} users")

    print("\n    Phase distribution BEFORE:")
    for ph in PHASE_ORDER:
        n = (before["phase"] == ph).sum()
        nu = before.loc[before["phase"] == ph, "author"].nunique()
        print(f"      {ph}: {n} user-days, {nu} users")
    print("\n    Phase distribution AFTER:")
    for ph in PHASE_ORDER:
        n = (after["phase"] == ph).sum()
        nu = after.loc[after["phase"] == ph, "author"].nunique()
        print(f"      {ph}: {n} user-days, {nu} users")

    # ── [6] Aggregate by phase ───────────────────────────────────────────────
    print("\n[6] Aggregating features by phase...")
    before_agg = aggregate_phase_features(before, features)
    after_agg  = aggregate_phase_features(after,  features)

    features_present = sorted(
        set(before_agg["feature"]) & set(after_agg["feature"])
    )
    print(f"    {len(features_present)} features with data in both windows")

    # ── [7] Plot ─────────────────────────────────────────────────────────────
    out_path = reports_dir / f"bc_started_before_after_{TIMESTAMP}.png"
    print(f"\n[7] Plotting → {out_path.name}...")
    plot_before_after(
        before_agg, after_agg,
        features=features_present,
        n_before=n_before_users,
        n_after=n_after_users,
        output_path=out_path,
        n_cols=n_cols,
    )
    print(f"    Saved → {out_path.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Before vs after starting BC pill — phase feature comparison"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--labels-file",
        default="data/processed/bc_wide_candidates_with_labels.csv",
    )
    ap.add_argument(
        "--window", type=int, default=28,
        help="Days before and after pill start to include (default: 28).",
    )
    ap.add_argument("--original-features-only", action="store_true")
    ap.add_argument("--n-cols", type=int, default=5)
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        labels_file=args.labels_file,
        window=args.window,
        original_features_only=args.original_features_only,
        n_cols=args.n_cols,
    ))
