#!/usr/bin/env python3
"""
Step 12 — BC / Non-BC Timeline Split
======================================
Splits the daily timeline into two files:
  bc_timeline_<ts>.csv     — BC users only, restricted to their pill-relevant window
  non_bc_timeline_<ts>.csv — everyone else, with ALL LLM-confirmed BC pill users removed

BC categories (all require known pill name, stopped users excluded entirely):
  combined stable  — no start/stop events, combined hormonal pill
                     window: full timeline (already ±3 months around CD1)
  combined started — recently started, combined hormonal pill
                     window: offset_from_cd1 >= start_cd1
  mini stable      — same rules, progestin-only pill
  mini started     — same rules, progestin-only pill

pill_type is combined when pill name is NOT in the POP keyword list.
pill_type is mini    when pill name IS     in the POP keyword list.
Users where no post has a known pill name are excluded.
Users with mixed combined+mini mentions are excluded (ambiguous).

Note: stopped and mixed users are excluded from bc_timeline but also excluded
from non_bc_timeline (via all is_bc_pill=True authors), so they do not pollute
either group.

Input:
  data/processed/bc_wide_candidates_with_labels.csv  — LLM-labelled Excel
  data/interim/timeline_daily_aggregated_with_anchors_raw_*.csv  — step 06

Output:
  data/interim/bc_timeline_<ts>.csv
    Columns: author, offset_from_cd1, *_mean, pill_type, bc_status
  data/interim/non_bc_timeline_<ts>.csv
    Columns: author, offset_from_cd1, *_mean  (all is_bc_pill=True users removed)

Next step:
  python scripts/42_ebm_bc_classifier.py \\
      --bc-users-file data/interim/bc_timeline_<ts>.csv \\
      --regular-users-file data/interim/non_bc_timeline_<ts>.csv \\
      --output-run-name <run_name>

Usage:
  python scripts/12_bc_nonbc_split.py
  python scripts/12_bc_nonbc_split.py --timeline-file data/interim/timeline_daily_aggregated_with_anchors_raw_20260603T000200.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp

# ── PILL CLASSIFICATION ────────────────────────────────────────────────────────

POP_KEYWORDS = [
    "pop", "norethindrone", "norgestrel", "desogestrel", "slynd", "opill",
    "cerazette", "cerelle", "nora-be", "nora be", "camila", "errin",
    "jencycla", "lyza", "slinda",
]


def _is_combined(pill_name) -> bool | None:
    """True = combined hormonal pill, False = progestin-only (mini), None = unknown."""
    if pd.isna(pill_name):
        return None
    return not any(k in str(pill_name).lower() for k in POP_KEYWORDS)


# ── BC USER EXTRACTION ─────────────────────────────────────────────────────────

def load_bc_categories_all_confirmed(labels_path: Path) -> dict[str, dict[str, float]]:
    """Return all is_bc_pill=True users with no additional filtering.

    Started users: window from start_cd1.
    Everyone else (stable + stopped + unknown pill): full timeline.
    pill_type set to combined/mini/unknown; bc_status stable/started.
    """
    try:
        labels = pd.read_excel(labels_path, engine="openpyxl")
    except Exception:
        labels = pd.read_csv(labels_path, encoding="latin-1")

    labels = labels[labels["is_bc_pill"] == True].copy()
    labels["_is_combined"] = labels["pill_name"].apply(_is_combined)

    def _pill_type(x):
        vals = x.dropna()
        if vals.empty:       return "unknown"
        if vals.all():       return "combined"
        if (~vals).all():    return "mini"
        return "mixed"

    user_info = labels.groupby("author").agg(
        any_started=("started_recently", "any"),
        pill_type  =("_is_combined", _pill_type),
    ).reset_index()

    # start_cd1 for started users
    sp = labels[(labels["started_recently"] == True) & labels["started_offset"].notna()].copy()
    sp["start_cd1"] = sp["offset_from_cd1"] + sp["started_offset"]
    sp["_abs"] = sp["started_offset"].abs()
    start_map = sp.sort_values("_abs").groupby("author")["start_cd1"].first().to_dict()

    # Anchors for non-started users (full timeline — anchor value unused)
    anchors = labels.groupby("author")["offset_from_cd1"].median().to_dict()

    categories: dict[str, dict[str, float]] = {
        "stable_combined": {}, "started_combined": {},
        "stable_mini":     {}, "started_mini":     {},
        "stable_unknown":  {}, "started_unknown":  {},
        "stable_mixed":    {}, "started_mixed":    {},
    }
    for _, row in user_info.iterrows():
        author   = row["author"]
        pt       = row["pill_type"]
        status   = "started" if row["any_started"] and author in start_map else "stable"
        anchor   = start_map[author] if status == "started" else anchors.get(author, 0)
        key      = f"{status}_{pt}"
        if key in categories:
            categories[key][author] = anchor

    return categories


def load_bc_categories(labels_path: Path) -> dict[str, dict[str, float]]:
    """Parse LLM labels and return per-category {author: anchor_offset} maps.

    Keys: stable_combined, started_combined, stable_mini, started_mini
    Values: {author: anchor} where anchor is
      - for stable:  median(offset_from_cd1) of their is_bc_pill=True posts
      - for started: start_cd1 = offset_from_cd1 + started_offset
                     (post with smallest |started_offset| used as best estimate)
    """
    try:
        labels = pd.read_excel(labels_path, engine="openpyxl")
    except Exception:
        labels = pd.read_csv(labels_path, encoding="latin-1")

    # Keep only GPT-confirmed BC pill posts
    labels = labels[labels["is_bc_pill"] == True].copy()
    labels["_is_combined"] = labels["pill_name"].apply(_is_combined)

    # User-level flags
    user_flags = labels.groupby("author").agg(
        any_started =("started_recently", "any"),
        any_stopped =("stopped_recently", "any"),
        # combined: at least one confirmed combined, no confirmed mini
        has_combined=("_is_combined", lambda x: bool(x.any()) and not bool((x == False).any())),
        # mini: at least one confirmed mini, no confirmed combined
        has_mini    =("_is_combined", lambda x: bool((x == False).any()) and not bool(x.any())),
    ).reset_index()

    # Drop stopped users — their pill status during the window is ambiguous
    user_flags = user_flags[~user_flags["any_stopped"]]

    stable_users  = user_flags[~user_flags["any_started"]]
    started_users = user_flags[ user_flags["any_started"]]

    # Anchor for stable users: median BC mention offset
    anchors = (
        labels.groupby("author")["offset_from_cd1"]
        .median()
        .to_dict()
    )

    # start_cd1 for started users: post closest to actual start day
    started_posts = labels[
        (labels["started_recently"] == True) & labels["started_offset"].notna()
    ].copy()
    started_posts["start_cd1"] = (
        started_posts["offset_from_cd1"] + started_posts["started_offset"]
    )
    started_posts["_abs_off"] = started_posts["started_offset"].abs()
    start_cd1_map = (
        started_posts.sort_values("_abs_off")
        .groupby("author")["start_cd1"]
        .first()
        .to_dict()
    )

    def _stable_map(flag_df) -> dict[str, float]:
        return {a: anchors[a] for a in flag_df["author"] if a in anchors}

    def _started_map(flag_df) -> dict[str, float]:
        return {a: start_cd1_map[a] for a in flag_df["author"] if a in start_cd1_map}

    return {
        "stable_combined":  _stable_map (stable_users [stable_users ["has_combined"]]),
        "started_combined": _started_map(started_users[started_users["has_combined"]]),
        "stable_mini":      _stable_map (stable_users [stable_users ["has_mini"]]),
        "started_mini":     _started_map(started_users[started_users["has_mini"]]),
    }


# ── TIMELINE FILTERING ─────────────────────────────────────────────────────────

def build_bc_timeline(
    tl: pd.DataFrame,
    categories: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Return rows from tl restricted to each user's BC-relevant window.

    Stable users: full timeline (already ±3 months around CD1, no further cut).
    Started users: offset_from_cd1 >= start_cd1 (posts from BC start onward).
    """
    chunks = []

    for category, user_anchors in categories.items():
        bc_status, pill_type = category.split("_", 1)

        for author, anchor in user_anchors.items():
            rows = tl[tl["author"] == author]
            if rows.empty:
                continue

            if bc_status == "started":
                rows = rows[rows["offset_from_cd1"] >= anchor].copy()

            if rows.empty:
                continue

            rows = rows.copy()
            rows["pill_type"] = pill_type
            rows["bc_status"] = bc_status
            chunks.append(rows)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(
    config_path: str = "configs/base.yaml",
    labels_file: str | None = None,
    timeline_file: str | None = None,
    all_confirmed: bool = False,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(config_path)
    interim_dir   = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    files_cfg     = cfg["paths"]["files"]

    print("=" * 60)
    print("Step 12: BC / Non-BC Timeline Split")
    print("=" * 60)
    mode_str = "all confirmed (is_bc_pill=True, no further filters)" if all_confirmed else "strict (known pill, not stopped, unambiguous type)"
    print(f"  Mode    : {mode_str}")
    print(f"  Stable  : full timeline (±3 months around CD1)")
    print(f"  Started : offset_from_cd1 >= start_cd1")

    # ── [1] Load LLM labels ───────────────────────────────────────────────────
    lp = Path(labels_file) if labels_file else processed_dir / "bc_wide_candidates_with_labels.csv"
    logging.info(f"\n[1] LLM labels: {lp.name}")

    # Load labels once to get the full set of is_bc_pill=True authors (including
    # stopped and mixed users who won't enter bc_tl but must not pollute non_bc).
    try:
        _lbl = pd.read_excel(lp, engine="openpyxl")
    except Exception:
        _lbl = pd.read_csv(lp, encoding="latin-1")
    all_bc_pill_authors = set(_lbl[_lbl["is_bc_pill"] == True]["author"].unique())
    logging.info(f"  {len(all_bc_pill_authors):,} total is_bc_pill=True authors in labels")

    categories = load_bc_categories_all_confirmed(lp) if all_confirmed else load_bc_categories(lp)

    for cat, m in categories.items():
        logging.info(f"  {cat:<20s}: {len(m):>4} users")
    total = sum(len(m) for m in categories.values())
    logging.info(f"  {'TOTAL':<20s}: {total:>4} users")

    if total == 0:
        logging.error("No BC users found. Check labels file.")
        return 1

    # ── [2] Load raw timeline ─────────────────────────────────────────────────
    if timeline_file:
        tl_path = Path(timeline_file)
    else:
        tl_path = find_latest_file(
            interim_dir,
            files_cfg["daily_aggregated_with_anchors_raw"] + "_*.csv",
            exclude="_no_anchors",
        )
    if tl_path is None:
        logging.error("No raw timeline found. Run script 06 first.")
        return 1

    logging.info(f"\n[2] Timeline: {tl_path.name}")
    tl = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)
    logging.info(f"  {len(tl):,} rows, {tl['author'].nunique():,} users")

    # ── [3] Filter to BC users with per-user windows ──────────────────────────
    logging.info("\n[3] Applying BC windows…")
    bc_tl = build_bc_timeline(tl, categories)

    if bc_tl.empty:
        logging.error(
            "No rows survived windowing. "
            "Author names in the labels file may not match the timeline."
        )
        return 1

    logging.info(f"  {len(bc_tl):,} rows, {bc_tl['author'].nunique():,} users kept")
    for pill in ("combined", "mini"):
        for status in ("stable", "started"):
            sub = bc_tl[(bc_tl["pill_type"] == pill) & (bc_tl["bc_status"] == status)]
            logging.info(
                f"  {pill} {status:<8s}: {sub['author'].nunique():>4} users  "
                f"{len(sub):>6,} rows"
            )

    # ── [4] Save BC timeline ──────────────────────────────────────────────────
    out_path = save_with_timestamp(bc_tl, interim_dir, "bc_timeline")
    logging.info(f"\n[4] Saved BC timeline → {out_path.name}")

    # ── [5] Save non-BC timeline — exclude ALL is_bc_pill=True users ─────────
    # Using all_bc_pill_authors (not just bc_tl authors) ensures stopped and
    # mixed-pill users don't appear in the non-BC group either.
    non_bc_tl = tl[~tl["author"].isin(all_bc_pill_authors)].copy()
    non_bc_path = save_with_timestamp(non_bc_tl, interim_dir, "non_bc_timeline")
    logging.info(f"[5] Saved non-BC timeline → {non_bc_path.name}")
    logging.info(f"  Excluded {len(all_bc_pill_authors):,} BC pill authors → {non_bc_tl['author'].nunique():,} users remain")

    logging.info(
        f"\nNext step:\n"
        f"  python scripts/42_ebm_bc_classifier.py \\\n"
        f"      --bc-users-file {out_path} \\\n"
        f"      --regular-users-file {non_bc_path} \\\n"
        f"      --output-run-name <run_name>"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",        default="configs/base.yaml")
    p.add_argument("--labels-file",   default=None,
                   help="LLM labels file (default: processed/bc_wide_candidates_with_labels.csv)")
    p.add_argument("--timeline-file", default=None,
                   help="Input timeline CSV (default: latest step-06 raw with-anchors file)")
    p.add_argument("--all-confirmed", action="store_true",
                   help="Use all is_bc_pill=True users, skipping pill name / stopped / type filters")
    args = p.parse_args()
    raise SystemExit(main(
        config_path=args.config,
        labels_file=args.labels_file,
        timeline_file=args.timeline_file,
        all_confirmed=args.all_confirmed,
    ))
