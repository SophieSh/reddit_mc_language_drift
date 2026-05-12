#!/usr/bin/env python3
"""Step 18: Natural cycle vs stable BC pill users — side-by-side phase comparison.

Directly compares linguistic features by menstrual phase between:
  Group A — Clean natural cycle users (667 agreed, BC-verified removed)
             Phase assignment uses each user's FFT-detected cycle period.
  Group B — Stable BC pill users (658 LLM-confirmed, no recent start/stop)
             Phase assignment uses a fixed 28-day pill-pack cycle.

The expected result: Group A shows ovulation peaks (positive sentiment,
valence, syntactic complexity, pronouns); Group B is flat at ovulation
because the pill suppresses the LH surge and follicular estrogen peak.

Input:
  data/interim/consensus_periods_overlap_agreed667_no_bc.csv
  data/interim/bc_stable_users_*.csv
  data/interim/timeline_daily_aggregated_with_anchors_*.csv

Output:
  reports/natural_vs_bc_phase_{timestamp}.png
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
from src.visualization import aggregate_features_by_phase

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

PHASE_ORDER  = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_COLORS = {
    "Menstrual":  "#d62728",
    "Follicular": "#ff7f0e",
    "Ovulation":  "#2ca02c",
    "Luteal":     "#1f77b4",
}
PHASE_WIDTHS = {"Menstrual": 1.5, "Follicular": 2.0, "Ovulation": 1.0, "Luteal": 4.5}

ORIGINAL_FEATURES = [
    "negative_sentiment_mean", "positive_sentiment_mean", "num_words_mean",
    "avg_word_length_mean", "num_sentences_mean", "unique_word_fraction_mean",
    "readability_mean", "spelling_errors_frac_mean",
    "syntactic_complexity_subordination_index_mean",
    "cohesion_analysis_lexical_overlap_mean",
]


def draw_comparison_bar(
    ax,
    nat_data: pd.DataFrame,
    bc_data: pd.DataFrame,
    feature: str,
    title: str,
) -> None:
    """Two bars per phase: natural cycle (hatched) vs BC stable (solid)."""
    bar_w = 0.55
    gap_within = 0.08
    gap_phase  = 0.4
    cur = 0.0
    phase_tick_x = []

    for phase in PHASE_ORDER:
        color = PHASE_COLORS[phase]
        xn = cur
        xb = cur + bar_w + gap_within

        nat_row = nat_data[(nat_data["feature"] == feature) & (nat_data["phase"] == phase)]
        bc_row  = bc_data[(bc_data["feature"]  == feature) & (bc_data["phase"]  == phase)]

        if not nat_row.empty:
            r = nat_row.iloc[0]
            ax.bar(xn, r["mean"], width=bar_w, color=color, alpha=0.5,
                   hatch="///", edgecolor="black", linewidth=0.8,
                   yerr=r["sem"], capsize=3, error_kw={"linewidth": 0.8})
            ax.text(xn, r["mean"] + (0.002 if r["mean"] >= 0 else -0.004),
                    f"n={int(r['n_users'])}", ha="center", va="bottom", fontsize=4.5)

        if not bc_row.empty:
            r = bc_row.iloc[0]
            ax.bar(xb, r["mean"], width=bar_w, color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.8,
                   yerr=r["sem"], capsize=3, error_kw={"linewidth": 0.8})
            ax.text(xb, r["mean"] + (0.002 if r["mean"] >= 0 else -0.004),
                    f"n={int(r['n_users'])}", ha="center", va="bottom", fontsize=4.5)

        phase_tick_x.append((cur + bar_w + gap_within / 2, phase[:3]))
        cur += 2 * bar_w + gap_within + gap_phase

    ax.set_xticks([x for x, _ in phase_tick_x])
    ax.set_xticklabels([l for _, l in phase_tick_x], fontsize=7)
    ax.set_ylabel("z-score", fontsize=6)
    ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def main(
    config_path: str = "configs/base.yaml",
    natural_consensus: str = "consensus_periods_overlap_agreed667_no_bc.csv",
    bc_stable_file: str | None = None,
    original_features_only: bool = False,
    n_cols: int = 5,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 18: Natural Cycle vs Stable BC — Phase Comparison")
    print("=" * 60)
    print()

    # ── [1] Load natural cycle consensus ────────────────────────────────────
    nat_path = interim_dir / natural_consensus
    print(f"[1] Natural cycle users: {nat_path.name}")
    nat_consensus = pd.read_csv(nat_path)
    nat_results = nat_consensus[["user", "consensus_period"]].rename(
        columns={"consensus_period": "period"}
    )
    nat_results["method"] = "fft_interpolation"
    nat_users = set(nat_results["user"].astype(str))
    print(f"    {len(nat_users):,} users  |  "
          f"median period: {nat_results['period'].median():.1f} days")

    # ── [2] Load stable BC users ─────────────────────────────────────────────
    if bc_stable_file:
        bc_path = Path(bc_stable_file)
    else:
        bc_path = find_latest_file(interim_dir, "bc_stable_users_*.csv")
    print(f"\n[2] Stable BC users: {bc_path.name}")
    bc_stable = pd.read_csv(bc_path)
    bc_results = pd.DataFrame({
        "user": bc_stable["author"].astype(str),
        "period": 28.0,
        "method": "fixed_28day",
    })
    bc_users = set(bc_results["user"])
    print(f"    {len(bc_users):,} users  |  fixed period: 28 days")

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
    if original_features_only:
        features = [f for f in ORIGINAL_FEATURES if f in all_feat]
    else:
        features = all_feat
    print(f"\n[4] Features: {len(features)}")

    # ── [5] Aggregate by phase for each group ────────────────────────────────
    print("\n[5] Aggregating natural cycle users by phase...")
    nat_timeline = timeline[timeline["author"].astype(str).isin(nat_users)]
    print(f"    Found {nat_timeline['author'].nunique():,} natural cycle users in timeline")
    nat_phase_df = aggregate_features_by_phase(
        timeline_df=nat_timeline,
        results_df=nat_results.rename(columns={"user": "user"}),
        features=features,
        time_col="offset_from_cd1",
        user_col="author",
        method="fft_interpolation",
        normalize=True,
        average_per_user=True,
    )

    print("\n    Aggregating stable BC users by phase...")
    bc_timeline = timeline[timeline["author"].astype(str).isin(bc_users)]
    print(f"    Found {bc_timeline['author'].nunique():,} stable BC users in timeline")
    bc_phase_df = aggregate_features_by_phase(
        timeline_df=bc_timeline,
        results_df=bc_results,
        features=features,
        time_col="offset_from_cd1",
        user_col="author",
        method="fixed_28day",
        normalize=True,
        average_per_user=True,
    )

    features_present = sorted(
        set(nat_phase_df["feature"]) & set(bc_phase_df["feature"])
    )
    print(f"\n    {len(features_present)} features with data in both groups")

    # ── [6] Plot ─────────────────────────────────────────────────────────────
    n_rows = (len(features_present) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        f"Phase Features: Natural Cycle (667, hatched) vs Stable BC Pill (658, solid)\n"
        f"Natural cycle: FFT-detected period  |  BC stable: fixed 28-day pill cycle",
        fontsize=10, fontweight="bold", y=1.002,
    )

    for idx, feature in enumerate(features_present):
        ax = axes[idx // n_cols, idx % n_cols]
        label = feature.replace("_mean", "").replace("_", " ").title()
        draw_comparison_bar(ax, nat_phase_df, bc_phase_df, feature, label)

    for idx in range(len(features_present), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="grey", alpha=0.5, hatch="///", edgecolor="black",
              label="Natural cycle (667 users, FFT period)"),
        Patch(facecolor="grey", alpha=0.85, edgecolor="black",
              label="Stable BC pill (658 users, 28-day fixed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = reports_dir / f"natural_vs_bc_phase_{TIMESTAMP}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[6] Saved → {out_path.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Natural cycle vs stable BC users — side-by-side phase comparison"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--natural-consensus",
        default="consensus_periods_overlap_agreed667_no_bc.csv",
    )
    ap.add_argument("--bc-stable-file", default=None)
    ap.add_argument("--original-features-only", action="store_true")
    ap.add_argument("--n-cols", type=int, default=5)
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        natural_consensus=args.natural_consensus,
        bc_stable_file=args.bc_stable_file,
        original_features_only=args.original_features_only,
        n_cols=args.n_cols,
    ))
