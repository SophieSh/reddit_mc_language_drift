"""Column-swap the corrected daily feature values into the analysis-input files,
preserving each file's EXACT cohort/rows (only <feature>_mean values change).

Overwrites the 69 `<feat>_mean` columns in each target by joining on (author, offset_from_cd1)
to the corrected daily-aggregated files produced by patch_features_yngve.py. Cells with no match
in the new daily -> NaN (per decision: "NaN for consistency"). All other columns (pill_type,
bc_status, phase) and every row are kept identical.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TS_NEW = "20260906T135827"   # patch_features_yngve.py output timestamp
KEY = ["author", "offset_from_cd1"]

# target -> which new daily provides corrected _mean values
TARGETS = {
    "data/interim/bc_timeline_20260603T000200.csv":
        f"data/interim/timeline_daily_aggregated_with_anchors_raw_{TS_NEW}.csv",
    "data/interim/non_bc_timeline_20260603T000200.csv":
        f"data/interim/timeline_daily_aggregated_with_anchors_raw_{TS_NEW}.csv",
    "data/interim/eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125.csv":
        f"data/interim/timeline_daily_aggregated_no_anchors_raw_{TS_NEW}.csv",
}


def main():
    ts = time.strftime("%Y%m%dT%H%M%S")
    for tgt, daily in TARGETS.items():
        tgt_p, daily_p = ROOT / tgt, ROOT / daily
        t = pd.read_csv(tgt_p, encoding="utf-8-sig", low_memory=False)
        d = pd.read_csv(daily_p, encoding="utf-8-sig", low_memory=False)
        mean_cols = [c for c in t.columns if c.endswith("_mean") and c in d.columns]
        # index new daily by key, reindex to target rows -> NaN where no match
        d_idx = d.set_index(KEY)[mean_cols]
        keyed = list(map(tuple, t[KEY].to_numpy()))
        new_vals = d_idx.reindex(keyed)
        matched = new_vals[mean_cols[0]].notna().to_numpy().sum()

        # sanity: lexicon feature should be ~unchanged; parse feature should change
        before = t[["positive_sentiment_mean", "syntactic_complexity_yngve_score_mean"]].to_numpy(copy=True)
        t[mean_cols] = new_vals.to_numpy()
        ps_chg = (~np.isclose(before[:, 0], t["positive_sentiment_mean"],
                              rtol=1e-4, atol=1e-6, equal_nan=True)).mean() * 100
        yn_chg = (~np.isclose(before[:, 1], t["syntactic_complexity_yngve_score_mean"],
                              rtol=1e-4, atol=1e-6, equal_nan=True)).mean() * 100

        out = tgt_p.with_name(tgt_p.stem + f"_yngvefix_{ts}.csv")
        t.to_csv(out, index=False)
        print(f"{tgt_p.name}")
        print(f"   rows={len(t):,} authors={t.author.nunique():,}  cols swapped={len(mean_cols)}  "
              f"cells matched={matched:,}/{len(t):,} ({matched/len(t)*100:.1f}%)")
        print(f"   positive_sentiment changed {ps_chg:.1f}% (want ~0)   yngve changed {yn_chg:.1f}% (want ~100)")
        print(f"   -> {out.name}\n")


if __name__ == "__main__":
    main()
