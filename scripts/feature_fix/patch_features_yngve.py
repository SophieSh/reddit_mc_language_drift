"""Inject the supervisor's parser-corrected features into the pipeline.

Overwrites ALL 69 per-post feature columns on the pipeline's own step-05 timelines
(join on `permalink`), masks the still-broken `concrete_abstract_ratio`, then re-aggregates
with the pipeline's exact logic (aggregate_all_features_by_day + per-author z-score, matching
scripts/06). Post set / offsets / cohort are preserved because we patch VALUES onto the
pipeline's existing posts; his surplus posts (not in the timeline) are ignored by the join.

Outputs new timestamped daily-aggregated files:
  timeline_daily_aggregated_with_anchors_{raw,zscore}_<ts>.csv
  timeline_daily_aggregated_no_anchors_{raw,zscore}_<ts>.csv
which then feed 12_bc_nonbc_split.py and 08a_select_eligible_users.py via --timeline-file.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis import aggregate_all_features_by_day
from src.utils import identify_feature_columns
from src.config import load_config

PARQUET = Path.home() / "Downloads" / "yngve_posts_for_recompute_pm365.18aug2026.parquet"
STEP05 = {
    "with_anchors": ROOT / "data/interim/timeline_with_offsets_with_anchors_20260608T200331.csv",
    "no_anchors":   ROOT / "data/interim/timeline_with_offsets_no_anchors_20260608T200308.csv",
}
INDEX_COLS = ["author", "offset_from_cd1"]


def mask_concrete(df):
    """concrete_abstract_ratio still divide-by-zero in his output: inf/>=60000/>100 -> NaN."""
    c = "concrete_abstract_ratio"
    if c in df.columns:
        bad = (~np.isfinite(df[c])) | (df[c].abs() >= 60000) | (df[c] > 100)
        n = int(bad.sum())
        df.loc[bad, c] = np.nan
        print(f"    masked concrete_abstract_ratio: {n} posts -> NaN")
    return df


def main():
    cfg = load_config("configs/base.yaml")
    ts = time.strftime("%Y%m%dT%H%M%S")
    interim = ROOT / "data/interim"

    print(f"Loading corrected parquet: {PARQUET.name}")
    new = pd.read_parquet(PARQUET)
    new = mask_concrete(new)

    for tag, path in STEP05.items():
        print(f"\n=== {tag}: {path.name} ===")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        feats = identify_feature_columns(df, cfg)
        assert "permalink" in df.columns, f"{path.name} has no permalink"
        # features present in BOTH the timeline and his file (should be all 69)
        patch_cols = [c for c in feats if c in new.columns]
        print(f"    timeline posts={len(df):,}  features to overwrite={len(patch_cols)}/{len(feats)}")

        # coverage: how many timeline permalinks are absent from his file?
        his = new.drop_duplicates("permalink").set_index("permalink")
        miss = (~df["permalink"].isin(his.index)).sum()
        print(f"    coverage: {len(df)-miss:,}/{len(df):,} posts covered; {miss} uncovered -> NaN")

        # overwrite: reindex his values to the timeline's permalink order (uncovered -> NaN)
        mapped = his.reindex(df["permalink"].values)[patch_cols].to_numpy()
        df[patch_cols] = mapped

        # aggregate exactly like scripts/06
        daily = aggregate_all_features_by_day(df, feats, "author", "offset_from_cd1")
        mean_cols = [c for c in daily.columns if c not in INDEX_COLS and c.endswith("_mean")]
        z_cols = [c.replace("_mean", "_zscore") for c in mean_cols]
        daily[z_cols] = daily.groupby("author")[mean_cols].transform(lambda x: (x - x.mean()) / x.std())

        raw_p = interim / f"timeline_daily_aggregated_{tag}_raw_{ts}.csv"
        z_p   = interim / f"timeline_daily_aggregated_{tag}_zscore_{ts}.csv"
        daily[INDEX_COLS + mean_cols].to_csv(raw_p, index=False)
        daily[INDEX_COLS + z_cols].to_csv(z_p, index=False)
        print(f"    wrote {raw_p.name}  ({len(daily):,} rows, {daily.author.nunique():,} users)")
        print(f"    wrote {z_p.name}")

    print(f"\nDONE. Timestamp={ts}")
    print("Next: 12_bc_nonbc_split.py --timeline-file <with_anchors_raw>  |  "
          "08a --timeline-file <no_anchors_raw>")


if __name__ == "__main__":
    main()
