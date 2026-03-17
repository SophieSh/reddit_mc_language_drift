#!/usr/bin/env python3
"""Step 11b: Before/After phase analysis around a BC event (STARTED or STOPPED).

Uses BC event posts as anchors: for each user, the day they posted about starting/stopping
BC pills becomes day 0. Posts are split into Before [-window, -1] and After [1, +window].

For each window, features are aggregated by menstrual phase (Menstrual / Follicular /
Ovulation / Luteal) using a fixed 29-day cycle and offset_from_cd1 — exactly like the
general population plot in 09_visualize_phase_features.py.

The output plot has two columns per feature: left = Before, right = After.

Input:
  data/interim/bc_users_event_*.csv
  data/interim/timeline_with_offsets_with_anchors_*.csv

Output:
  reports/bc_started_phase_before_after_[ts].png
  reports/bc_stopped_phase_before_after_[ts].png
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
from src.utils import identify_feature_columns
from src.visualization import (
    aggregate_features_by_phase,
    plot_phase_analysis,
)

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

PHASE_ORDER  = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
PHASE_COLORS = {
    "Menstrual":  "#d62728",
    "Follicular": "#ff7f0e",
    "Ovulation":  "#2ca02c",
    "Luteal":     "#1f77b4",
}
PHASE_WIDTHS = {"Menstrual": 1.5, "Follicular": 2.0, "Ovulation": 1.0, "Luteal": 4.5}


# ---------------------------------------------------------------------------

def select_bc_event_day(bc_df: pd.DataFrame) -> pd.DataFrame:
    """One canonical BC event day per user.
    STARTED → earliest post; STOPPED → latest post.
    Returns: author, regex_label, bc_event_day.
    """
    rows = []
    for label, grp in bc_df.groupby("regex_label"):
        agg = (
            grp.groupby("author")["offset_from_cd1"].min().reset_index()
            if label == "STARTED"
            else grp.groupby("author")["offset_from_cd1"].max().reset_index()
        )
        agg = agg.rename(columns={"offset_from_cd1": "bc_event_day"})
        agg["regex_label"] = label
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def plot_phase_two_columns(
    phase_before: pd.DataFrame,
    phase_after: pd.DataFrame,
    features: list[str],
    output_path: Path,
    group: str,
    window: int,
    error_bar_type: str = "sem",
) -> None:
    """For each feature: two side-by-side subplots (Before | After), each with 4 phase bars."""

    n_features = len(features)
    n_feat_cols = 4                          # features per row, per window
    n_rows = (n_features + n_feat_cols - 1) // n_feat_cols
    n_cols  = n_feat_cols * 2                # *2 for Before | After

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"BC {group}: Feature Values by Phase — Before (left) vs After (right)  |  ±{window} days",
        fontsize=13, fontweight="bold", y=0.998,
    )

    # Column headers
    for c in range(n_feat_cols):
        axes[0, c].set_title("BEFORE", fontsize=10, color="gray",
                              fontweight="bold", pad=12)
    for c in range(n_feat_cols, n_cols):
        axes[0, c].set_title("AFTER", fontsize=10, color="#1f77b4" if group == "STARTED" else "#ff7f0e",
                              fontweight="bold", pad=12)

    def _draw_phase_bars(ax, feat_data, feature_display):
        if feat_data.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(feature_display, fontsize=8, fontweight="bold")
            return

        means, errors, n_list, colors_list = [], [], [], []
        for phase in PHASE_ORDER:
            row = feat_data[feat_data["phase"] == phase]
            if not row.empty:
                means.append(row.iloc[0]["mean"])
                errors.append(row.iloc[0][error_bar_type])
                n_list.append(int(row.iloc[0]["n_users"]))
            else:
                means.append(0.0); errors.append(0.0); n_list.append(0)
            colors_list.append(PHASE_COLORS[phase])

        widths = [PHASE_WIDTHS[p] for p in PHASE_ORDER]
        x_positions, cur = [], 0
        for w in widths:
            x_positions.append(cur + w / 2)
            cur += w + 0.2

        bars = ax.bar(x_positions, means, width=widths, yerr=errors, capsize=4,
                      color=colors_list, alpha=0.75, edgecolor="black", linewidth=1.2)
        for bar, n in zip(bars, n_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={n}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(PHASE_ORDER, rotation=40, ha="right", fontsize=7)
        ax.set_xlim(-0.5, cur + 0.5)
        ax.set_ylabel("z-score", fontsize=7)
        ax.set_title(feature_display, fontsize=8, fontweight="bold")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    for feat_idx, feature in enumerate(features):
        row_i = feat_idx // n_feat_cols
        col_i = feat_idx %  n_feat_cols

        display = feature.replace("_mean", "").replace("_", " ").title()

        before_data = phase_before[phase_before["feature"] == feature]
        after_data  = phase_after [phase_after ["feature"] == feature]

        ax_before = axes[row_i, col_i]
        ax_after  = axes[row_i, col_i + n_feat_cols]

        _draw_phase_bars(ax_before, before_data, display)
        _draw_phase_bars(ax_after,  after_data,  display)

    # Hide unused axes
    for feat_idx in range(n_features, n_rows * n_feat_cols):
        row_i = feat_idx // n_feat_cols
        col_i = feat_idx %  n_feat_cols
        axes[row_i, col_i].axis("off")
        axes[row_i, col_i + n_feat_cols].axis("off")

    # Vertical separator line between Before and After halves
    fig.add_artist(plt.Line2D(
        [0.5, 0.5], [0.01, 0.99],
        transform=fig.transFigure,
        color="black", linewidth=1.5, linestyle="--", alpha=0.4,
    ))

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {output_path}")


# ---------------------------------------------------------------------------

def main(
    config_path: str = "configs/base.yaml",
    window: int = 90,
    cycle_length: float = 28.0,
    event_file: str | None = None,
    timeline_file: str | None = None,
    error_bars: str = "sem",
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 11b: BC Before/After Phase Analysis")
    print("=" * 60)
    print(f"  Window: ±{window} days  |  Cycle: {cycle_length} days (fixed)")
    print()

    # ---- Load BC event users -------------------------------------------------
    ev_path = Path(event_file) if event_file else find_latest_file(interim_dir, "bc_users_event_*.csv")
    print(f"[1] BC event file: {ev_path.name}")
    bc_df = pd.read_csv(ev_path)
    print(f"    {len(bc_df):,} event posts — {bc_df['regex_label'].value_counts().to_dict()}")

    event_days = select_bc_event_day(bc_df)
    print(f"    Canonical events: {(event_days['regex_label']=='STARTED').sum()} STARTED, "
          f"{(event_days['regex_label']=='STOPPED').sum()} STOPPED")

    # ---- Load timeline -------------------------------------------------------
    tl_path = Path(timeline_file) if timeline_file else find_latest_file(
        interim_dir, "timeline_with_offsets_with_anchors_*.csv"
    )
    print(f"\n[2] Timeline: {tl_path.name}")
    timeline = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)

    event_users = set(event_days["author"])
    timeline_ev = timeline[timeline["author"].isin(event_users)].copy()
    print(f"    {len(timeline_ev):,} posts from {timeline_ev['author'].nunique():,} BC event users")

    # ---- Feature columns -----------------------------------------------------
    features = identify_feature_columns(timeline_ev, cfg)
    print(f"\n[3] Features: {len(features)}")

    # ---- Merge event day + compute bc_relative_offset ------------------------
    timeline_ev = timeline_ev.merge(
        event_days[["author", "bc_event_day", "regex_label"]],
        on="author", how="inner",
    )
    timeline_ev["bc_relative_offset"] = (
        timeline_ev["offset_from_cd1"] - timeline_ev["bc_event_day"]
    )

    # ---- For each group, aggregate by phase in Before and After windows ------
    # aggregate_features_by_phase needs a dummy results_df when fixed_cycle_length is set
    dummy_results = pd.DataFrame()

    for group in ["STARTED", "STOPPED"]:
        grp_df = timeline_ev[timeline_ev["regex_label"] == group].copy()
        if grp_df.empty:
            print(f"\n  No {group} users — skipping.")
            continue

        print(f"\n[4] {group} group ({grp_df['author'].nunique()} users)")

        before_df = grp_df[grp_df["bc_relative_offset"].between(-window, -1)].copy()
        after_df  = grp_df[grp_df["bc_relative_offset"].between(1, window)].copy()
        print(f"    Before: {len(before_df):,} posts from {before_df['author'].nunique()} users")
        print(f"    After:  {len(after_df):,} posts from {after_df['author'].nunique()} users")

        print(f"  Aggregating BEFORE by phase...")
        phase_before = aggregate_features_by_phase(
            timeline_df=before_df,
            results_df=dummy_results,
            features=features,
            time_col="offset_from_cd1",
            user_col="author",
            fixed_cycle_length=cycle_length,
            normalize=True,
            average_per_user=True,
        )

        print(f"  Aggregating AFTER by phase...")
        phase_after = aggregate_features_by_phase(
            timeline_df=after_df,
            results_df=dummy_results,
            features=features,
            time_col="offset_from_cd1",
            user_col="author",
            fixed_cycle_length=cycle_length,
            normalize=True,
            average_per_user=True,
        )

        if phase_before.empty and phase_after.empty:
            print(f"  No phase data for {group} — skipping plot.")
            continue

        feature_order = sorted(
            set(phase_before["feature"].unique()) | set(phase_after["feature"].unique())
        )

        out_plot = reports_dir / f"bc_{group.lower()}_phase_before_after_{TIMESTAMP}.png"
        print(f"  Plotting {len(feature_order)} features...")
        plot_phase_two_columns(
            phase_before=phase_before,
            phase_after=phase_after,
            features=feature_order,
            output_path=out_plot,
            group=group,
            window=window,
            error_bar_type=error_bars,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BC before/after phase analysis")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--window", type=int, default=90,
                    help="Days before/after BC event (default: 90)")
    ap.add_argument("--cycle-length", type=float, default=28.0,
                    help="Fixed cycle length for phase assignment (default: 29)")
    ap.add_argument("--event-file", default=None)
    ap.add_argument("--timeline-file", default=None)
    ap.add_argument("--error-bars", choices=["sem", "std"], default="sem")
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        window=args.window,
        cycle_length=args.cycle_length,
        event_file=args.event_file,
        timeline_file=args.timeline_file,
        error_bars=args.error_bars,
    ))
