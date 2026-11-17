#!/usr/bin/env python3
"""Build timeline files for a PCOS-vs-non-BC EBM run (feed to script 42).

Positive class : users who posted >=1 time in r/PCOS (excluding BC-pill users,
                 so the contrast stays "non-BC PCOS" vs "non-BC baseline").
Negative class : non-BC, non-PCOS users.
Both are sliced from the SAME master daily timeline, so feature columns match.

Outputs (data/interim/):
  pcos_timeline_<ts>.csv
  non_bc_non_pcos_timeline_<ts>.csv
"""
import glob
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BC_LABELS = ROOT / "data/processed/bc_wide_candidates_with_labels.csv"        # xlsx
POST_FILES = [ROOT / "data/raw/moon1_all_post_2015_2025_filtered_features.csv",
              ROOT / "data/raw/moon2_all_post_2015_2025_filtered_features.csv"]
MASTER = sorted(glob.glob(str(ROOT / "data/interim/timeline_daily_aggregated_with_anchors_zscore_*.csv")))[-1]

# ── PCOS posters (exact subreddit r/PCOS) ──────────────────────────────────────
pcos_users = set()
for post_file in POST_FILES:
    for chunk in pd.read_csv(post_file, usecols=["author", "subreddit"],
                             encoding="utf-8-sig", chunksize=300_000):
        pcos_users |= set(chunk.loc[chunk["subreddit"].astype(str).str.lower() == "pcos", "author"])
print(f"Users who posted in r/PCOS (all posts): {len(pcos_users):,}")

# ── BC-pill users (exclude from both classes) ──────────────────────────────────
bc = pd.read_excel(BC_LABELS)
bc_users = set(bc.loc[bc["is_bc_pill"] == True, "author"].dropna())
print(f"BC-pill users (is_bc_pill=True): {len(bc_users):,}")

# ── Master daily timeline ──────────────────────────────────────────────────────
print(f"Master timeline: {Path(MASTER).name}")
tl = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
master_users = set(tl["author"])
print(f"  {len(tl):,} rows, {len(master_users):,} users")
print(f"  PCOS users present in master: {len(pcos_users & master_users):,}")

# ── Split ──────────────────────────────────────────────────────────────────────
pos_users = (pcos_users & master_users) - bc_users          # PCOS, not BC
neg_users = master_users - pcos_users - bc_users            # non-PCOS, non-BC

pcos_tl   = tl[tl["author"].isin(pos_users)].copy()
nonbc_tl  = tl[tl["author"].isin(neg_users)].copy()
print(f"\nPositive (PCOS, non-BC): {pcos_tl['author'].nunique():,} users, {len(pcos_tl):,} rows")
print(f"Negative (non-BC, non-PCOS): {nonbc_tl['author'].nunique():,} users, {len(nonbc_tl):,} rows")

ts = time.strftime("%Y%m%dT%H%M%S")
pcos_path  = ROOT / "data/interim" / f"pcos_timeline_{ts}.csv"
nonbc_path = ROOT / "data/interim" / f"non_bc_non_pcos_timeline_{ts}.csv"
pcos_tl.to_csv(pcos_path, index=False, encoding="utf-8-sig")
nonbc_tl.to_csv(nonbc_path, index=False, encoding="utf-8-sig")
print(f"\nWrote {pcos_path.name}\nWrote {nonbc_path.name}")
print(f"\nNext:\n  python scripts/42_ebm_bc_classifier.py \\\n"
      f"    --bc-users-file {pcos_path} \\\n"
      f"    --regular-users-file {nonbc_path} \\\n"
      f"    --output-run-name pcos_vs_nonbc")
