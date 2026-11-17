#!/usr/bin/env python3
"""Step 7: Run periodicity detection and calculate SNR.

Supports four method choices:
  fft_zeropad   -- zero-padded FFT evaluated at all 15 integer periods [21..35]
                   (backward-compatible default; fixes the bin-landing bias of the
                   old fft_interpolation approach)
  ls_rednoise   -- Lomb-Scargle with AR(1) red-noise normalisation
  acf_gaussian  -- ACF with Bartlett CI and Gaussian gap-fill
  all           -- run all three methods, emit one CSV with a 'method' column

Input:
  data/interim/timeline_daily_aggregated_with_anchors_*.csv
  (columns: author, offset_from_cd1, {feature}_mean for each feature)

Output:
  data/interim/periodicity_results_{method}_snr{threshold}[_no_anchors]_{timestamp}.csv
  columns: user, feature, method, period, snr, passed, n_points, span_days
  method-specific extras:
    fft_zeropad  : power_at_peak
    ls_rednoise  : rho, normalized_power_at_peak
    acf_gaussian : acf_at_peak, eligible
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import aggregate_by_day, normalize_values, validate_timeline_data
from src.config import load_config, DEFAULT_PERIOD_MIN, DEFAULT_PERIOD_MAX
from src.io import find_latest_file, save_with_timestamp
from src.periodicity_fft_zeropad import detect_period_fft_zeropad
from src.periodicity_ls_rednoise import detect_period_ls_rednoise
from src.periodicity_acf_gaussian import detect_period_acf_gaussian, check_user_eligibility


_NEW_METHODS = {"fft_zeropad", "ls_rednoise", "acf_gaussian"}


def _run_one(
    user: str,
    feature_col: str,
    feature_name: str,
    user_df: pd.DataFrame,
    method: str,
    period_min: int,
    period_max: int,
    snr_threshold: float,
) -> list[dict]:
    """Aggregate + normalise a single (user, feature) slice and dispatch to detector(s).

    Returns a list of result dicts (one per method run).
    """
    daily = aggregate_by_day(user_df, offset_col="offset_from_cd1", value_col=feature_col)
    if len(daily) < 10:
        return []

    offsets = daily["offset_from_cd1"].values
    values = daily["feature_mean"].values

    is_valid, _ = validate_timeline_data(offsets, values, min_span_days=period_min)
    if not is_valid:
        return []

    values_norm = normalize_values(values, method="zscore")
    n_points = len(offsets)
    span_days = float(offsets.max() - offsets.min())

    methods_to_run = list(_NEW_METHODS) if method == "all" else [method]

    rows = []
    for m in methods_to_run:
        if m == "fft_zeropad":
            res = detect_period_fft_zeropad(offsets, values_norm, period_min, period_max,
                                            snr_threshold=snr_threshold)
            rows.append({
                "user": user, "feature": feature_name, "method": m,
                "period": res["period"], "snr": res["snr"], "passed": res["passed"],
                "n_points": n_points, "span_days": span_days,
                "power_at_peak": res.get("power_at_peak"),
            })
        elif m == "ls_rednoise":
            res = detect_period_ls_rednoise(offsets, values_norm, period_min, period_max,
                                            snr_threshold=snr_threshold)
            rows.append({
                "user": user, "feature": feature_name, "method": m,
                "period": res["period"], "snr": res["snr"], "passed": res["passed"],
                "n_points": n_points, "span_days": span_days,
                "rho": res.get("rho"),
                "normalized_power_at_peak": res.get("normalized_power_at_peak"),
            })
        elif m == "acf_gaussian":
            res = detect_period_acf_gaussian(offsets, values_norm, period_min, period_max,
                                             snr_threshold=snr_threshold)
            rows.append({
                "user": user, "feature": feature_name, "method": m,
                "period": res["period"], "snr": res["snr"], "passed": res["passed"],
                "n_points": n_points, "span_days": span_days,
                "acf_at_peak": res.get("acf_at_peak"),
                "eligible": res.get("eligible"),
            })
    return rows


def main(
    config_path: str = "configs/base.yaml",
    method: str = "fft_zeropad",
    snr_threshold: float | None = None,
    use_checkpoint: bool = True,
    force_recompute: bool = False,
    no_anchors: bool = False,
    aggregated_file: str | None = None,
    period_min_override: int | None = None,
    period_max_override: int | None = None,
    consensus_min_agreement: int = 10,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    analysis_cfg = cfg.get("analysis", {})

    period_min = period_min_override if period_min_override is not None else int(analysis_cfg.get("period_min", DEFAULT_PERIOD_MIN))
    period_max = period_max_override if period_max_override is not None else int(analysis_cfg.get("period_max", DEFAULT_PERIOD_MAX))
    if snr_threshold is None:
        snr_threshold = float(analysis_cfg.get("snr_threshold", 3.0))

    anchor_suffix = "_no_anchors" if no_anchors else ""
    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 7: Run Periodicity Detection")
    print("=" * 60)
    print(f"  Method: {method}")
    print(f"  Period range: {period_min}-{period_max} days")
    print(f"  SNR threshold: {snr_threshold}")
    print(f"  Anchor posts: {'excluded' if no_anchors else 'included'}")
    print()

    files_cfg = cfg["paths"]["files"]
    pr_prefix = files_cfg["periodicity_results"]
    checkpoint_pattern = f"{pr_prefix}_{method}_snr{snr_threshold}{anchor_suffix}_*.csv"
    checkpoint = find_latest_file(
        interim_dir, checkpoint_pattern,
        exclude="_consensus" if not no_anchors else "_no_anchors",
    )

    if use_checkpoint and not force_recompute and checkpoint:
        print(f"  Found checkpoint: {checkpoint.name}")
        print("  Loading checkpoint for consensus step...")
        results_df = pd.read_csv(checkpoint, low_memory=False)
        passed_df = results_df[results_df["passed"]]
        # Jump straight to consensus
        print(f"  {len(passed_df):,} passed rows, {passed_df['user'].nunique():,} users")
    else:
        results_df = None  # will be built below

    if results_df is None:
        # Load daily aggregated timeline
        timeline_pattern = (
            files_cfg["daily_aggregated_no_anchors"] + "_*.csv" if no_anchors
            else files_cfg["daily_aggregated_with_anchors"] + "_*.csv"
        )
        print(f"[Step 7.1] Loading daily aggregated timeline ({('no anchors' if no_anchors else 'with anchors')})...")
        if aggregated_file:
            timeline_file = Path(aggregated_file)
            if not timeline_file.exists():
                timeline_file = interim_dir / aggregated_file
        else:
            timeline_file = find_latest_file(interim_dir, timeline_pattern)

        if not timeline_file:
            raise FileNotFoundError(
                f"No daily aggregated timeline ({timeline_pattern}) found in {interim_dir}. "
                "Please run scripts/06_aggregate_and_normalize.py first."
            )

        daily_agg = pd.read_csv(timeline_file, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(daily_agg):,} (user, day) rows from {timeline_file.name}")
        print(f"  Users: {daily_agg['author'].nunique():,}")

        # Identify feature columns (_mean suffix from aggregation)
        print("\n[Step 7.2] Identifying feature columns...")
        metadata_cols = {"author", "offset_from_cd1", "text", "ts_utc"}
        mean_cols = [c for c in daily_agg.columns if c.endswith("_mean") and c not in metadata_cols]

        if not mean_cols:
            raise ValueError("No feature columns found (expected _mean columns from aggregation step)")

        feature_base_names = [c.replace("_mean", "") for c in mean_cols]
        print(f"  Found {len(mean_cols)} features to analyze")

        # ACF eligibility report (cheap pre-scan, useful to know before a long run)
        if method in ("acf_gaussian", "all"):
            print("\n[Step 7.2b] ACF eligibility pre-scan (span>=70, max_gap<=3)...")
            n_eligible = 0
            for user, user_df in daily_agg.groupby("author"):
                offsets = user_df["offset_from_cd1"].values
                ok, _ = check_user_eligibility(offsets, max_gap_days=3, min_span_days=70)
                if ok:
                    n_eligible += 1
            n_users_total = daily_agg["author"].nunique()
            print(f"  Eligible users: {n_eligible:,} / {n_users_total:,} ({100*n_eligible/max(n_users_total,1):.1f}%)")

        # Main loop
        print(f"\n[Step 7.3] Running '{method}' analysis...")
        print("  This may take several minutes...")

        results: list[dict] = []
        user_groups = list(daily_agg.groupby("author"))
        total = len(mean_cols) * len(user_groups)
        processed = 0

        # Pre-filter: keep only users with >=15 posting days and span>=70d within ±90d of anchor
        anchor_window = 90
        min_post_days = 15
        min_span_user = 70
        eligible_users = set()
        for user, user_df in user_groups:
            w = user_df[
                (user_df["offset_from_cd1"] >= -anchor_window) &
                (user_df["offset_from_cd1"] <= anchor_window)
            ]
            if len(w) < min_post_days:
                continue
            if w["offset_from_cd1"].max() - w["offset_from_cd1"].min() < min_span_user:
                continue
            eligible_users.add(user)
        print(f"  Eligible users (>={min_post_days} posting days, span>={min_span_user}d within ±{anchor_window}d): {len(eligible_users):,} / {len(user_groups):,}")

        for feature_col, feature_name in zip(mean_cols, feature_base_names):
            for user, user_df in user_groups:
                processed += 1
                if user not in eligible_users:
                    continue
                if processed % 100_000 == 0:
                    pct = 100.0 * processed / total
                    print(f"    Progress: {processed:,}/{total:,} ({pct:.1f}%)")

                rows = _run_one(
                    user=user,
                    feature_col=feature_col,
                    feature_name=feature_name,
                    user_df=user_df,
                    method=method,
                    period_min=period_min,
                    period_max=period_max,
                    snr_threshold=snr_threshold,
                )
                results.extend(rows)

        print(f"    Progress: {processed:,}/{total:,} (100.0%)")

        if not results:
            print(f"\n  No results found (all below SNR threshold {snr_threshold})")
            return 1

        results_df = pd.DataFrame(results)

        n_passed = results_df["passed"].sum()
        print(f"\n  Analysis complete: {len(results_df):,} total rows, {n_passed:,} passed SNR threshold")
        print(f"  Users with any detection: {results_df['user'].nunique():,}")
        print(f"  Features with any detection: {results_df['feature'].nunique()}")

        # Save
        print("\n[Step 7.4] Saving results...")
        output_file = save_with_timestamp(
            results_df,
            interim_dir,
            f"{pr_prefix}_{method}_snr{snr_threshold}{anchor_suffix}",
        )
        print(f"  Saved: {output_file.name}")

    # Summary over passed rows
    passed_df = results_df[results_df["passed"]]
    if len(passed_df) > 0:
        print(f"\nSummary (passed rows only):")
        for m_name, grp in passed_df.groupby("method"):
            print(f"  [{m_name}]  mean period={grp['period'].mean():.1f}  "
                  f"median={grp['period'].median():.1f}  "
                  f"mean SNR={grp['snr'].mean():.2f}  N={len(grp):,}")

    # Consensus: per-user mode of detected periods that landed in the biological window [21,35].
    # Features that went to boundary sinks (T=14 or T=42) are noise — exclude them.
    bio_min, bio_max = 22, 35
    bio_df = passed_df[(passed_df["period"] >= bio_min) & (passed_df["period"] <= bio_max)]

    print(f"\n[Step 7.5] Computing consensus periods (bio window [{bio_min},{bio_max}], min_agreement={consensus_min_agreement})...")
    print(f"  Features landing in bio window: {len(bio_df):,} / {len(passed_df):,} "
          f"({100*len(bio_df)/max(len(passed_df),1):.1f}%)")

    consensus_rows = []
    for user, grp in bio_df.groupby("user"):
        n_in_window = len(grp)
        if n_in_window < consensus_min_agreement:
            continue
        periods = grp["period"].values
        # Fuzzy agreement: candidates are [bio_min+1, bio_max-1] so every candidate
        # has a full symmetric ±1 window within [bio_min, bio_max].
        best_period, best_count = None, 0
        for candidate in range(bio_min + 1, bio_max):
            count = int(((periods >= candidate - 1) & (periods <= candidate + 1)).sum())
            if count > best_count:
                best_count = count
                best_period = candidate
        if best_count < consensus_min_agreement:
            continue
        consensus_rows.append({
            "user": user,
            "consensus_period": best_period,
            "n_features_in_window": n_in_window,
            "fuzzy_agreement_count": best_count,
            "mean_snr": float(grp["snr"].mean()),
        })

    consensus_df = pd.DataFrame(consensus_rows)
    if len(consensus_df) == 0:
        print("  No users reached consensus.")
        return 0

    print(f"  Consensus users in [{bio_min},{bio_max}]: {len(consensus_df):,}  "
          f"median period={int(consensus_df['consensus_period'].median())}d")
    print(f"\n  Distribution:")
    print(consensus_df["consensus_period"].value_counts().sort_index().to_string())

    consensus_file = save_with_timestamp(
        consensus_df,
        interim_dir,
        f"{pr_prefix}_{method}_snr{snr_threshold}{anchor_suffix}_consensus{consensus_min_agreement}",
    )
    print(f"\n  Saved consensus: {consensus_file.name}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 7: Run periodicity detection and calculate SNR"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config YAML file")
    parser.add_argument(
        "--method",
        type=str,
        default="fft_zeropad",
        choices=["fft_zeropad", "ls_rednoise", "acf_gaussian", "all"],
        help=(
            "Periodicity detection method. "
            "'fft_zeropad' (default): zero-padded FFT evaluated at integer periods [21..35]. "
            "'ls_rednoise': Lomb-Scargle with AR(1) red-noise normalisation. "
            "'acf_gaussian': ACF with Bartlett CI and Gaussian gap-fill. "
            "'all': run all three and emit a single CSV with a 'method' column."
        ),
    )
    parser.add_argument("--snr-threshold", type=float, default=None,
                        help="SNR threshold (default: from config)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable checkpoint loading")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute even if checkpoint exists")
    parser.add_argument("--no-anchors", action="store_true",
                        help="Use timeline without anchor posts")
    parser.add_argument("--aggregated-file", type=str, default=None,
                        help="Path or filename (in interim dir) of a specific aggregated file")
    parser.add_argument("--period-min", type=int, default=None,
                        help="Override period_min from config (e.g. 14 for wide search)")
    parser.add_argument("--period-max", type=int, default=None,
                        help="Override period_max from config (e.g. 45 for wide search)")
    parser.add_argument("--consensus-min-agreement", type=int, default=10,
                        help="Min features that must detect AND agree on the same period (default: 10)")

    args = parser.parse_args()
    exit(main(
        config_path=args.config,
        method=args.method,
        snr_threshold=args.snr_threshold,
        use_checkpoint=not args.no_checkpoint,
        force_recompute=args.force_recompute,
        no_anchors=args.no_anchors,
        aggregated_file=args.aggregated_file,
        period_min_override=args.period_min,
        period_max_override=args.period_max,
        consensus_min_agreement=args.consensus_min_agreement,
    ))
