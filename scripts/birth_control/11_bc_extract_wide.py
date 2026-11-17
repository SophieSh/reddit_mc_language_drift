#!/usr/bin/env python3
"""Step 11: Wide extraction of birth control pill posts from the timeline.

Casts a broad net to collect every post that possibly mentions birth control
pills by matching brand names and generic pill terms.  Intentionally permissive:
the goal is recall, not precision.  The output is a flat CSV meant to be sent
through an API (e.g. Claude) for nuanced classification in the next step.

Note: the timeline is already windowed to ±3 months around each user's CD1
anchor (set in configs/base.yaml → pipeline.window_months, default 3).

Input:
  data/interim/timeline_with_offsets_with_anchors_*.csv

Output:
  data/interim/bc_wide_candidates_{timestamp}.csv
    Columns:
      author            - user identifier
      offset_from_cd1   - day relative to first CD1
      text              - full post text
      text_snippet      - first 400 chars of text (for quick review)
      subreddit         - subreddit the post was made in
      matched_terms     - comma-separated list of all regex hits
      ts_utc            - post timestamp (if available in timeline)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp


# ---------------------------------------------------------------------------
# Pill mention patterns — brand names + generic terms only
# ---------------------------------------------------------------------------

# Brand names (combined / mini pill, progestin-only; patches and rings excluded
# intentionally since those are non-pill forms).
_BRAND_NAMES = (
    r"Yaz|Yasmin|Yasminelle|Junel|Sprintec|Tri[\-\s]?Sprintec|"
    r"Loestrin|Lo\.?\s*Loestrin|Microgestin|Ortho[\-\s]?Tri[\-\s]?Cyclen|"
    r"Levlen|Alesse|Aviane|Lutera|Portia|Cryselle|Blisovi|"
    r"Seasonique|Seasonale|Lybrel|Mircette|Kariva|Estrostep|"
    r"Nordette|Levora|TriNessa|Elinest|Camrese|Introvale|Quasense|"
    r"Daysee|Amethia|Chateal|Falmina|Larissia|Ocella|Zarah|"
    r"Novynette|Dianette|Cilest|Mercilon|Marvelon|Microgynon|"
    r"Rigevidon|Levest|Gedarel|Femodene|Femodette|Millinette|"
    r"Logynon|Trinovum|Brevicon|Modicon|Nelova|Nortrel|"
    r"Ortho[\-\s]?Novum|Zenchent|Gianvi|Loryna|Vestura|Nikki|"
    r"Sylara|Cyred|Eminique|Aurovela|Larin|Natazia|Qlaira|"
    r"Zoely|Slinda|Slynd|Camila|Errin|Jencycla|Lyza|Nora[\-\s]?BE|"
    r"Aubra|Aygestin|Minovral|Tri[\-\s]?Lo[\-\s]?Sprintec|"
    r"Norethindrone|Norgestrel|Desogestrel|Drospirenone|"
    r"Ethinyl\s+estradiol|Levonorgestrel"
)

# Generic pill terms — informal and clinical names, common abbreviations
_GENERIC_TERMS = (
    r"birth\s+control\s+pills?|bc\s+pills?|the\s+pill|oral\s+contraceptives?|"
    r"contraceptive\s+pills?|combined\s+pills?|combination\s+pills?|"
    r"mini[\-\s]?pills?|minipills?|progestin[\-\s]?only\s+pills?|"
    r"\bBCPs?\b|\bOCPs?\b"
)

_WIDE_REGEX = re.compile(
    rf"\b(?:{_BRAND_NAMES})\b|(?:{_GENERIC_TERMS})",
    flags=re.IGNORECASE,
)


def find_all_matches(text: str) -> list[str]:
    """Return all non-overlapping regex matches in the text (up to 10)."""
    return [m.group(0).strip() for m in list(_WIDE_REGEX.finditer(str(text)))[:10]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/base.yaml",
         timeline_file: str | None = None,
         limit: int | None = None) -> int:

    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 11: Wide BC Pill Post Extraction")
    print("=" * 60)
    print()

    # ---- Load timeline -------------------------------------------------------
    if timeline_file:
        tl_path = Path(timeline_file)
    else:
        tl_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    if not tl_path:
        raise FileNotFoundError(
            f"No timeline file found in {interim_dir}.\n"
            "Run scripts/05_build_timeline.py first."
        )
    print(f"[1] Timeline: {tl_path.name}")
    df = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)
    print(f"    {len(df):,} posts from {df['author'].nunique():,} users")

    for col in ("author", "offset_from_cd1", "text"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing. Available: {df.columns.tolist()}")

    if limit:
        print(f"    Limiting to first {limit:,} rows (--limit)")
        df = df.head(limit)

    # ---- Apply wide regex filter ---------------------------------------------
    print("\n[2] Applying wide BC pill filter...")
    mask = df["text"].apply(lambda t: bool(_WIDE_REGEX.search(str(t))))
    hits = df[mask].copy()
    print(f"    Matched: {len(hits):,} posts from {hits['author'].nunique():,} users"
          f"  ({100 * len(hits) / len(df):.2f}% of timeline)")

    if hits.empty:
        print("    No matches found. Exiting.")
        return 1

    # ---- Collect matched terms -----------------------------------------------
    print("\n[3] Extracting matched terms...")
    hits["matched_terms"] = hits["text"].apply(
        lambda t: ", ".join(find_all_matches(t))
    )

    # ---- Assemble output columns --------------------------------------------
    keep = ["author", "offset_from_cd1"]

    # Add timestamp column if present
    for ts_col in ("ts_utc", "created_utc", "timestamp"):
        if ts_col in hits.columns:
            keep.append(ts_col)
            break

    if "subreddit" in hits.columns:
        keep.append("subreddit")

    keep += ["text", "matched_terms"]
    out = hits[keep].copy()

    # ---- Save ----------------------------------------------------------------
    out_path = save_with_timestamp(out, interim_dir, "bc_wide_candidates")
    print(f"\n[4] Saved → {out_path.name}")
    print(f"    Shape: {out.shape[0]:,} rows × {out.shape[1]} columns")

    # Quick term-frequency summary
    print("\n[5] Top 20 most common matched terms:")
    all_terms: list[str] = []
    for terms_str in out["matched_terms"]:
        all_terms.extend([t.strip().lower() for t in str(terms_str).split(",") if t.strip()])
    term_counts = pd.Series(all_terms).value_counts().head(20)
    for term, cnt in term_counts.items():
        print(f"    {cnt:>5}  {term}")

    print("\nDone. Ready for API classification (next step).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Wide extraction of BC pill posts for API classification"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--timeline-file",
        default=None,
        help="Path to a specific timeline CSV (default: latest in interim dir).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows (for testing).",
    )
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        timeline_file=args.timeline_file,
        limit=args.limit,
    ))
