#!/usr/bin/env python3
"""How many TRUE birth-control-pill users ever posted in r/PCOS?

TRUE bc users  = is_bc_pill == True in bc_wide_candidates_with_labels (xlsx).
"ever posted"  = >=1 post with subreddit == "PCOS" anywhere in their full history
                 (moon1 + moon2). Also reports any PCOS-related subreddit.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BC_FILE    = ROOT / "data/processed/bc_wide_candidates_with_labels.csv"   # actually xlsx
POST_FILES = [ROOT / "data/raw/moon1_all_post_2015_2025_filtered_features.csv",
              ROOT / "data/raw/moon2_all_post_2015_2025_filtered_features.csv"]

bc = pd.read_excel(BC_FILE)
bc_true = set(bc.loc[bc["is_bc_pill"] == True, "author"].dropna())
print(f"TRUE bc-pill users (is_bc_pill == True): {len(bc_true):,}")

posted_pcos = set()        # subreddit == "PCOS"
posted_pcos_any = set()    # any subreddit containing "pcos"
for post_file in POST_FILES:
    for chunk in pd.read_csv(post_file, usecols=["author", "subreddit"],
                             encoding="utf-8-sig", chunksize=300_000):
        chunk = chunk[chunk["author"].isin(bc_true)]
        sub = chunk["subreddit"].astype(str)
        posted_pcos     |= set(chunk.loc[sub.str.lower() == "pcos", "author"])
        posted_pcos_any |= set(chunk.loc[sub.str.lower().str.contains("pcos"), "author"])

n = len(bc_true)
print(f"\n... posted in r/PCOS (exact):        {len(posted_pcos):,}  ({100*len(posted_pcos)/n:.1f}%)")
print(f"... posted in ANY PCOS* subreddit:   {len(posted_pcos_any):,}  ({100*len(posted_pcos_any)/n:.1f}%)")

out = ROOT / "data/processed" / "bc_true_users_pcos_flag.csv"
pd.DataFrame({"author": sorted(bc_true)}).assign(
    posted_rPCOS=lambda d: d["author"].isin(posted_pcos),
    posted_any_PCOS=lambda d: d["author"].isin(posted_pcos_any),
).to_csv(out, index=False)
print(f"\nWrote {out}")
