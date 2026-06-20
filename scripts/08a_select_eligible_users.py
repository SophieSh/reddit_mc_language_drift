#!/usr/bin/env python3
"""Step 08a: Filter phase-labeled timeline to eligible users.

Keeps only users who meet minimum activity criteria and have sufficient
days in every phase, then saves the filtered timeline.

Criteria:
  - at least min_post_days distinct posting days
  - span (max_offset - min_offset) >= min_span_days
  - at least min_days_per_phase days in each of the 4 phases

Input:
  data/interim/timeline_phase_labeled_*.csv  (step 07 output)

Output:
  data/interim/eligible_users_timeline_minposts{N}_span{S}_minphase{M}_*.csv
  (same columns as step 07 output, rows restricted to eligible users)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.constants import PHASE_ORDER
from src.io import find_latest_file, save_with_timestamp


def _filter_eligible(
    daily_agg: pd.DataFrame,
    min_post_days: int,
    min_span_days: int,
    min_days_per_phase: int,
    skip_phase_filter: bool = False,
) -> pd.DataFrame:
    rows = []
    for user, grp in daily_agg.groupby("author"):
        n = len(grp)
        span = grp["offset_from_cd1"].max() - grp["offset_from_cd1"].min()
        if n < min_post_days or span < min_span_days:
            continue
        if not skip_phase_filter:
            phase_counts = grp["phase"].value_counts()
            if any(phase_counts.get(p, 0) < min_days_per_phase for p in PHASE_ORDER):
                continue
        rows.append({"user": user, "n_post_days": n, "span_days": span})
    return pd.DataFrame(rows)


def _filter_chain(
    daily_agg: pd.DataFrame,
    anchor_window: int,
    max_gap: int,
    min_span_days: int,
    min_post_days: int,
) -> pd.DataFrame:
    """Connected-chain eligibility: users whose posting chain crosses the anchor.

    For each user, considers only days within [-anchor_window, +anchor_window].
    Starting from the day closest to 0, extends left and right as long as
    consecutive posting days are ≤ max_gap apart. The resulting chain must:
      - cross the anchor (left < 0 AND right > 0)
      - span at least min_span_days
      - contain at least min_post_days distinct posting days
    """
    within = daily_agg[daily_agg["offset_from_cd1"].between(-anchor_window, anchor_window)]
    rows = []
    for user, grp in within.groupby("author"):
        days = sorted(grp["offset_from_cd1"].unique())
        if not days:
            continue

        # Find starting point: day closest to 0 (prefer <=0)
        before = [d for d in days if d <= 0]
        after  = [d for d in days if d >= 0]
        start  = before[-1] if before else after[0]
        si     = days.index(start)

        # Extend right from start
        right = start
        for i in range(si + 1, len(days)):
            if days[i] - days[i - 1] <= max_gap:
                right = days[i]
            else:
                break

        # Extend left from start
        left = start
        for i in range(si - 1, -1, -1):
            if days[i + 1] - days[i] <= max_gap:
                left = days[i]
            else:
                break

        if not (left < 0 and right > 0):
            continue

        span   = right - left
        n_days = sum(1 for d in days if left <= d <= right)

        if span >= min_span_days and n_days >= min_post_days:
            rows.append({"user": user, "n_post_days": n_days, "span_days": span})

    return pd.DataFrame(rows)


def _run_variant(
    label: str,
    timeline_file: Path,
    min_post_days: int,
    min_span_days: int,
    min_days_per_phase: int,
    interim_dir: Path,
    out_stem: str,
    skip_phase_filter: bool = False,
    max_gap: int | None = None,
    anchor_window: int = 28,
) -> int:
    print(f"\n[{label}] Loading: {timeline_file.name}")
    daily_agg = pd.read_csv(timeline_file, encoding="utf-8-sig", low_memory=False)

    required = {"author", "offset_from_cd1"} if (skip_phase_filter or max_gap is not None) \
               else {"author", "offset_from_cd1", "phase"}
    missing = required - set(daily_agg.columns)
    if missing:
        raise ValueError(f"Input file '{timeline_file.name}' is missing required columns: {sorted(missing)}")

    total_users = daily_agg["author"].nunique()
    print(f"  {len(daily_agg):,} (user, day) rows, {total_users:,} users")

    if max_gap is not None:
        eligible_df = _filter_chain(daily_agg, anchor_window, max_gap, min_span_days, min_post_days)
    else:
        eligible_df = _filter_eligible(daily_agg, min_post_days, min_span_days, min_days_per_phase,
                                       skip_phase_filter=skip_phase_filter)
    eligible_users = set(eligible_df["user"])
    print(f"  Eligible: {len(eligible_users):,} / {total_users:,} users "
          f"({100 * len(eligible_users) / max(total_users, 1):.1f}%)")

    if len(eligible_users) == 0:
        print("  No eligible users found.")
        return 1

    filtered = daily_agg[daily_agg["author"].isin(eligible_users)].copy()
    print(f"  Filtered table: {len(filtered):,} rows")

    output_file = save_with_timestamp(filtered, interim_dir, out_stem)
    print(f"  Saved: {output_file.name}")
    return 0


def main(
    config_path: str = "configs/base.yaml",
    min_post_days: int = 20,
    min_span_days: int = 100,
    min_days_per_phase: int = 3,
    no_anchors_only: bool = False,
    with_anchors_only: bool = False,
    skip_phase_filter: bool = False,
    timeline_file: str | None = None,
    out_tag: str | None = None,
    max_gap: int | None = None,
    anchor_window: int = 28,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    files_cfg = cfg["paths"]["files"]

    print("=" * 60)
    print("Step 08a: Select Eligible Users")
    print("=" * 60)
    print(f"  Min post days  : {min_post_days}")
    print(f"  Min span       : {min_span_days} days")
    if max_gap is not None:
        print(f"  Mode           : connected chain (max_gap={max_gap}, anchor_window=±{anchor_window})")
    elif skip_phase_filter:
        print(f"  Min days/phase : (skipped)")
    else:
        print(f"  Min days/phase : {min_days_per_phase}")

    exit_code = 0
    tag_suffix = f"_{out_tag}" if out_tag else ""

    if max_gap is not None:
        phase_tag = f"gap{max_gap}"
        out_stem_base = f"eligible_users_two_cycles_minposts{min_post_days}_span{min_span_days}_{phase_tag}{tag_suffix}"
    else:
        phase_tag = "nophase" if skip_phase_filter else f"minphase{min_days_per_phase}"
        out_stem_base = f"eligible_users_timeline_minposts{min_post_days}_span{min_span_days}_{phase_tag}{tag_suffix}"

    # ── Direct file mode: skip auto-discovery entirely ────────────────────────
    if timeline_file:
        tl_path = Path(timeline_file)
        if not tl_path.exists():
            tl_path = interim_dir / timeline_file
        if not tl_path.exists():
            print(f"  Timeline file not found: {timeline_file}")
            return 1
        return _run_variant(
            label=tl_path.stem,
            timeline_file=tl_path,
            min_post_days=min_post_days,
            min_span_days=min_span_days,
            min_days_per_phase=min_days_per_phase,
            interim_dir=interim_dir,
            out_stem=out_stem_base,
            skip_phase_filter=skip_phase_filter,
            max_gap=max_gap,
            anchor_window=anchor_window,
        )

    # ── Auto-discovery mode ───────────────────────────────────────────────────
    run_with = not no_anchors_only
    run_without = not with_anchors_only

    if run_with:
        pattern = files_cfg["phase_labeled"] + "_*.csv"
        tl_path = find_latest_file(interim_dir, pattern, exclude="_no_anchors")
        if not tl_path:
            print("\n  [with anchors] No phase-labeled timeline file found — skipping.")
            exit_code = 1
        else:
            rc = _run_variant(
                label="with anchors",
                timeline_file=tl_path,
                min_post_days=min_post_days,
                min_span_days=min_span_days,
                min_days_per_phase=min_days_per_phase,
                interim_dir=interim_dir,
                out_stem=out_stem_base,
                skip_phase_filter=skip_phase_filter,
                max_gap=max_gap,
                anchor_window=anchor_window,
            )
            exit_code = exit_code or rc

    if run_without:
        pattern_no = files_cfg["phase_labeled"] + "_no_anchors_*.csv"
        timeline_file_no = find_latest_file(interim_dir, pattern_no)
        if not timeline_file_no:
            print("\n  [no anchors] No phase-labeled timeline file found — skipping.")
            exit_code = 1
        else:
            rc = _run_variant(
                label="no anchors",
                timeline_file=timeline_file_no,
                min_post_days=min_post_days,
                min_span_days=min_span_days,
                min_days_per_phase=min_days_per_phase,
                interim_dir=interim_dir,
                out_stem=out_stem_base + "_no_anchors",
                skip_phase_filter=skip_phase_filter,
                max_gap=max_gap,
                anchor_window=anchor_window,
            )
            exit_code = exit_code or rc

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 08a: Select eligible users")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--min-post-days", type=int, default=20,
                        help="Min posting days within window (default: 20)")
    parser.add_argument("--min-span-days", type=int, default=100,
                        help="Min span in days within window (default: 100)")
    parser.add_argument("--min-days-per-phase", type=int, default=3,
                        help="Min days per phase required per user (default: 3)")
    parser.add_argument("--no-anchors-only", action="store_true",
                        help="Only run the no-anchors variant")
    parser.add_argument("--with-anchors-only", action="store_true",
                        help="Only run the with-anchors variant")
    parser.add_argument("--skip-phase-filter", action="store_true",
                        help="Skip the min-days-per-phase requirement (keeps users missing some phases)")
    parser.add_argument("--timeline-file", default=None,
                        help="Use a specific phase-labeled CSV instead of auto-discovery.")
    parser.add_argument("--out-tag", default=None,
                        help="Label appended to output filename (e.g. 'bc' → …_bc_*.csv).")
    parser.add_argument("--max-gap", type=int, default=None,
                        help="Enable connected-chain mode: max allowed gap (days) between "
                             "consecutive posting days. Chain must cross the anchor (day 0). "
                             "When set, --min-days-per-phase and --skip-phase-filter are ignored.")
    parser.add_argument("--anchor-window", type=int, default=28,
                        help="Window (±days) used for connected-chain eligibility check (default: 28). "
                             "Only applies when --max-gap is set.")

    args = parser.parse_args()
    exit(main(
        config_path=args.config,
        min_post_days=args.min_post_days,
        min_span_days=args.min_span_days,
        min_days_per_phase=args.min_days_per_phase,
        no_anchors_only=args.no_anchors_only,
        with_anchors_only=args.with_anchors_only,
        skip_phase_filter=args.skip_phase_filter,
        timeline_file=args.timeline_file,
        out_tag=args.out_tag,
        max_gap=args.max_gap,
        anchor_window=args.anchor_window,
    ))
