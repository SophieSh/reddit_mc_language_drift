#!/usr/bin/env python3
"""Step 11c: On-pills vs Off-pills phase analysis combining all BC user groups.

Pools data across STARTED, STOPPED, and STABLE_USING users:
  - "On pills":  STARTED (after event), STOPPED (before event), STABLE_USING (all)
  - "Off pills": STARTED (before event), STOPPED (after event)

Only users with posts covering >= min_phases distinct phases in a given pool
are included (reduces noise from single-phase users).

Input:
  data/interim/bc_users_event_*.csv
  data/interim/bc_users_stable_*.csv
  data/interim/timeline_with_offsets_with_anchors_*.csv

Output:
  reports/bc_on_off_pills_phase_[ts].png
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
    assign_phase_to_day,
    create_adaptive_phases,
)

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
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


def filter_min_phases(
    df: pd.DataFrame,
    time_col: str,
    user_col: str,
    cycle_length: float,
    min_phases: int = 2,
) -> pd.DataFrame:
    """Keep only users who have posts covering >= min_phases distinct phases."""
    phase_def = create_adaptive_phases(cycle_length)

    def _phase(day):
        if pd.isna(day):
            return None
        return assign_phase_to_day(day, phase_def)

    tmp = df.copy()
    tmp["_phase"] = tmp[time_col].apply(_phase)
    n_phases = tmp.groupby(user_col)["_phase"].nunique()
    valid = n_phases[n_phases >= min_phases].index
    dropped = df[user_col].nunique() - len(valid)
    print(f"    >= {min_phases} phases filter: kept {len(valid)} / "
          f"{df[user_col].nunique()} users (dropped {dropped})")
    return df[df[user_col].isin(valid)].copy()


def plot_on_off(
    phase_off: pd.DataFrame,
    phase_on: pd.DataFrame,
    features: list[str],
    output_path: Path,
    window: int,
    error_bar_type: str = "sem",
) -> None:
    """Two-column plot: Off Pills (left) | On Pills (right), 4-phase bars per feature."""
    n_features = len(features)
    n_feat_cols = 4
    n_rows = (n_features + n_feat_cols - 1) // n_feat_cols
    n_cols = n_feat_cols * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Feature Values by Phase — Off Pills (left) vs On Pills (right)  |  ±{window}-day window",
        fontsize=13, fontweight="bold", y=0.998,
    )

    for c in range(n_feat_cols):
        axes[0, c].set_title("OFF PILLS", fontsize=10, color="gray",
                              fontweight="bold", pad=12)
    for c in range(n_feat_cols, n_cols):
        axes[0, c].set_title("ON PILLS", fontsize=10, color="#9467bd",
                              fontweight="bold", pad=12)

    def _draw(ax, feat_data, display):
        if feat_data.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(display, fontsize=8, fontweight="bold")
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
        ax.set_title(display, fontsize=8, fontweight="bold")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    for feat_idx, feature in enumerate(features):
        row_i = feat_idx // n_feat_cols
        col_i = feat_idx % n_feat_cols
        display = feature.replace("_zscore", "").replace("_mean", "").replace("_", " ").title()
        off_data = phase_off[phase_off["feature"] == feature]
        on_data  = phase_on [phase_on ["feature"] == feature]
        _draw(axes[row_i, col_i], off_data, display)
        _draw(axes[row_i, col_i + n_feat_cols], on_data, display)

    for feat_idx in range(n_features, n_rows * n_feat_cols):
        row_i = feat_idx // n_feat_cols
        col_i = feat_idx % n_feat_cols
        axes[row_i, col_i].axis("off")
        axes[row_i, col_i + n_feat_cols].axis("off")

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
    min_phases: int = 2,
    event_file: str | None = None,
    stable_file: str | None = None,
    timeline_file: str | None = None,
    error_bars: str = "sem",
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 11c: On-Pills vs Off-Pills Phase Analysis")
    print("=" * 60)
    print(f"  Window: ±{window} days  |  Cycle: {cycle_length} days  |  Min phases: {min_phases}")
    print()

    # ---- Load BC event users -------------------------------------------------
    ev_path = Path(event_file) if event_file else find_latest_file(interim_dir, "bc_users_event_*.csv")
    print(f"[1] BC event file: {ev_path.name}")
    bc_df = pd.read_csv(ev_path)
    print(f"    {len(bc_df):,} event posts — {bc_df['regex_label'].value_counts().to_dict()}")

    event_days = select_bc_event_day(bc_df)
    n_started = (event_days["regex_label"] == "STARTED").sum()
    n_stopped = (event_days["regex_label"] == "STOPPED").sum()
    print(f"    Canonical events: {n_started} STARTED, {n_stopped} STOPPED")

    # ---- Load BC stable users ------------------------------------------------
    st_path = Path(stable_file) if stable_file else find_latest_file(interim_dir, "bc_users_stable_*.csv")
    print(f"\n[2] BC stable file: {st_path.name}")
    stable_df = pd.read_csv(st_path)
    stable_users = set(stable_df["author"].unique())
    print(f"    {len(stable_users)} STABLE_USING users")

    # ---- Load timeline -------------------------------------------------------
    tl_path = Path(timeline_file) if timeline_file else find_latest_file(
        interim_dir, "timeline_with_offsets_with_anchors_*.csv"
    )
    print(f"\n[3] Timeline: {tl_path.name}")
    timeline = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)

    all_bc_users = set(event_days["author"]) | stable_users
    timeline_bc = timeline[timeline["author"].isin(all_bc_users)].copy()
    print(f"    {len(timeline_bc):,} posts from {timeline_bc['author'].nunique():,} BC users")

    # ---- Feature columns -----------------------------------------------------
    features = identify_feature_columns(timeline_bc, cfg)
    print(f"\n[4] Features: {len(features)}")

    # ---- Attach event day and compute bc_relative_offset ---------------------
    timeline_ev = timeline_bc[timeline_bc["author"].isin(set(event_days["author"]))].copy()
    timeline_ev = timeline_ev.merge(
        event_days[["author", "bc_event_day", "regex_label"]],
        on="author", how="inner",
    )
    timeline_ev["bc_relative_offset"] = (
        timeline_ev["offset_from_cd1"] - timeline_ev["bc_event_day"]
    )

    started = timeline_ev[timeline_ev["regex_label"] == "STARTED"]
    stopped = timeline_ev[timeline_ev["regex_label"] == "STOPPED"]
    stable  = timeline_bc[timeline_bc["author"].isin(stable_users)].copy()

    # ---- Build On/Off pools --------------------------------------------------
    print(f"\n[5] Building On/Off pills pools (window: ±{window} days)...")

    on_parts = []
    if not started.empty:
        on_parts.append(started[started["bc_relative_offset"].between(1, window)])
    if not stopped.empty:
        on_parts.append(stopped[stopped["bc_relative_offset"].between(-window, -1)])
    if not stable.empty:
        on_parts.append(stable)
    on_df = pd.concat(on_parts, ignore_index=True) if on_parts else pd.DataFrame()

    off_parts = []
    if not started.empty:
        off_parts.append(started[started["bc_relative_offset"].between(-window, -1)])
    if not stopped.empty:
        off_parts.append(stopped[stopped["bc_relative_offset"].between(1, window)])
    off_df = pd.concat(off_parts, ignore_index=True) if off_parts else pd.DataFrame()

    print(f"    ON  pills: {len(on_df):,} posts from {on_df['author'].nunique():,} users "
          f"(STARTED-after={len(started[started['bc_relative_offset'].between(1, window)]) if not started.empty else 0}, "
          f"STOPPED-before={len(stopped[stopped['bc_relative_offset'].between(-window, -1)]) if not stopped.empty else 0}, "
          f"STABLE={len(stable):,})")
    print(f"    OFF pills: {len(off_df):,} posts from {off_df['author'].nunique():,} users "
          f"(STARTED-before={len(started[started['bc_relative_offset'].between(-window, -1)]) if not started.empty else 0}, "
          f"STOPPED-after={len(stopped[stopped['bc_relative_offset'].between(1, window)]) if not stopped.empty else 0})")

    # ---- Min-phases filter ---------------------------------------------------
    print(f"\n[6] Filtering to users with >= {min_phases} phases covered...")
    print("  ON pills:")
    on_df  = filter_min_phases(on_df,  "offset_from_cd1", "author", cycle_length, min_phases)
    print("  OFF pills:")
    off_df = filter_min_phases(off_df, "offset_from_cd1", "author", cycle_length, min_phases)

    # ---- Aggregate by phase --------------------------------------------------
    dummy = pd.DataFrame()

    print(f"\n[7] Aggregating OFF pills by phase...")
    phase_off = aggregate_features_by_phase(
        timeline_df=off_df,
        results_df=dummy,
        features=features,
        time_col="offset_from_cd1",
        user_col="author",
        fixed_cycle_length=cycle_length,
        normalize=True,
        average_per_user=True,
    )

    print(f"\n[8] Aggregating ON pills by phase...")
    phase_on = aggregate_features_by_phase(
        timeline_df=on_df,
        results_df=dummy,
        features=features,
        time_col="offset_from_cd1",
        user_col="author",
        fixed_cycle_length=cycle_length,
        normalize=True,
        average_per_user=True,
    )

    if phase_off.empty and phase_on.empty:
        print("No phase data in either pool — nothing to plot.")
        return 1

    feature_order = sorted(
        set(phase_off["feature"].unique()) | set(phase_on["feature"].unique())
    )

    out_plot = reports_dir / f"bc_on_off_pills_phase_{TIMESTAMP}.png"
    print(f"\n[9] Plotting {len(feature_order)} features...")
    plot_on_off(
        phase_off=phase_off,
        phase_on=phase_on,
        features=feature_order,
        output_path=out_plot,
        window=window,
        error_bar_type=error_bars,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="On-pills vs off-pills phase analysis")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--window", type=int, default=90,
                    help="Days before/after BC event (default: 90)")
    ap.add_argument("--cycle-length", type=float, default=28.0)
    ap.add_argument("--min-phases", type=int, default=2,
                    help="Min distinct phases per user per pool (default: 2)")
    ap.add_argument("--event-file", default=None)
    ap.add_argument("--stable-file", default=None)
    ap.add_argument("--timeline-file", default=None)
    ap.add_argument("--error-bars", choices=["sem", "std"], default="sem")
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        window=args.window,
        cycle_length=args.cycle_length,
        min_phases=args.min_phases,
        event_file=args.event_file,
        stable_file=args.stable_file,
        timeline_file=args.timeline_file,
        error_bars=args.error_bars,
    ))
