#!/usr/bin/env python3
"""Step 16: Phase feature visualization for stable BC pill users.

Loads LLM-labelled BC pill posts, normalizes pill names, filters out
emergency/OTC contraception, identifies stable users (no recent start
or stop event), then assigns a fixed 28-day synthetic cycle to each user
and visualizes linguistic features by phase.

Why 28-day fixed cycle for pill users?
  Hormonal contraception suppresses the natural cycle, but most combined
  pill packs follow a 28-day schedule (21 active + 7 placebo).  The
  withdrawal bleed on placebo days acts as a synthetic "CD1" anchor.
  Applying a fixed 28-day cycle lets us ask: do stable pill users show
  ANY language variation that tracks the pill pack schedule?

Input:
  data/processed/bc_wide_candidates_with_labels.csv  (LLM labels)
  data/interim/timeline_daily_aggregated_with_anchors_*.csv  (step 06)

Output:
  reports/bc_stable_phase_features_{timestamp}.png
  data/interim/bc_stable_users_{timestamp}.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.io import find_latest_file
from src.visualization import aggregate_features_by_phase, plot_phase_analysis

TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

# ---------------------------------------------------------------------------
# Pill name normalisation map  (lower-case key → canonical display name)
# Variants and generics are collapsed into their brand family.
# ---------------------------------------------------------------------------
PILL_NAME_MAP: dict[str, str] = {
    # Yaz / drospirenone family
    "yaz": "Yaz",
    "yasmin": "Yasmin",
    "yasminelle": "Yasmin",
    "loryna": "Yaz (generic)",
    "ocella": "Yasmin (generic)",
    "zarah": "Yasmin (generic)",
    "gianvi": "Yaz (generic)",
    "nikki": "Yaz (generic)",
    "vestura": "Yaz (generic)",
    # Loestrin family
    "loestrin": "Loestrin",
    "loestrin fe": "Loestrin Fe",
    "loestrin 24 fe": "Loestrin Fe",
    "loestrin 1/20": "Loestrin",
    "lo loestrin fe": "Lo Loestrin Fe",
    "lo loestrin": "Lo Loestrin Fe",
    "microgestin": "Loestrin (generic)",
    "microgestin fe": "Loestrin Fe (generic)",
    "microgestin fe 1/20": "Loestrin Fe (generic)",
    "larin": "Loestrin (generic)",
    "larin fe": "Loestrin Fe (generic)",
    "junel": "Junel",
    "junel fe": "Junel Fe",
    "junel fe 1/20": "Junel Fe",
    "blisovi": "Junel Fe (generic)",
    "aurovela": "Loestrin (generic)",
    # Sprintec family
    "sprintec": "Sprintec",
    "tri-sprintec": "Tri-Sprintec",
    "tri sprintec": "Tri-Sprintec",
    "mononessa": "Sprintec (generic)",
    "trinessa": "Tri-Sprintec (generic)",
    "tri nessa": "Tri-Sprintec (generic)",
    "elinest": "Sprintec (generic)",
    "zenchent": "Sprintec (generic)",
    "syeda": "Sprintec (generic)",
    # Levonorgestrel / Alesse family
    "alesse": "Alesse",
    "aviane": "Alesse (generic)",
    "lutera": "Alesse (generic)",
    "portia": "Alesse (generic)",
    "cryselle": "Lo/Ovral (generic)",
    "falmina": "Alesse (generic)",
    "larissia": "Alesse (generic)",
    "levlen": "Levlen",
    "nordette": "Levlen",
    "levora": "Levlen (generic)",
    # Seasonique / extended cycle
    "seasonique": "Seasonique",
    "seasonale": "Seasonale",
    "lybrel": "Lybrel",
    "amethia": "Seasonique (generic)",
    "camrese": "Seasonique (generic)",
    "introvale": "Seasonale (generic)",
    "quasense": "Seasonale (generic)",
    "daysee": "Seasonique (generic)",
    # Progestin-only pills
    "slynd": "Slynd (POP)",
    "slinda": "Slynd (POP)",
    "norethindrone": "Norethindrone (POP)",
    "norgestrel": "Norgestrel (POP)",
    "desogestrel": "Desogestrel (POP)",
    "cerazette": "Desogestrel (POP)",
    "cerelle": "Desogestrel (POP)",
    "nora-be": "Norethindrone (POP)",
    "nora be": "Norethindrone (POP)",
    "camila": "Norethindrone (POP)",
    "errin": "Norethindrone (POP)",
    "jencycla": "Norethindrone (POP)",
    "lyza": "Norethindrone (POP)",
    "microgynon": "Microgynon",
    "rigevidon": "Microgynon (generic)",
    "levest": "Microgynon (generic)",
    "gedarel": "Microgynon (generic)",
    "femodene": "Femodene",
    "femodette": "Femodette",
    "millinette": "Femodette (generic)",
    "marvelon": "Marvelon",
    "mercilon": "Mercilon",
    "mircette": "Mircette",
    "kariva": "Mircette (generic)",
    "azurette": "Mircette (generic)",
    "zoely": "Zoely",
    "natazia": "Natazia",
    "qlaira": "Natazia",
    # Other named brands — keep as-is
}

# Emergency / OTC single-use contraception to EXCLUDE
EXCLUDE_PILLS = {
    "plan b", "plan b one-step", "plan b onestep",
    "ella", "ellaone",
    "next choice", "next choice one dose",
    "take action", "take action one dose",
    "my way", "opcicon",
    "afterpill", "after pill",
    "morning after pill", "morning-after pill",
    "emergency contraception", "emergency contraceptive",
    "copper iud", "paragard",
    "levonelle",
}

# Opill is a regular progestin-only pill (recently OTC) — keep but flag
OTC_REGULAR = {"opill"}


def normalize_pill_name(name: str | None) -> str | None:
    """Return canonical pill name or None if excluded."""
    if pd.isna(name) or name is None:
        return None
    key = str(name).strip().lower()
    if key in EXCLUDE_PILLS:
        return "__EXCLUDE__"
    if key in OTC_REGULAR:
        return "Opill (OTC POP)"
    return PILL_NAME_MAP.get(key, str(name).strip().title())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_path: str = "configs/base.yaml",
    labels_file: str = "data/processed/bc_wide_candidates_with_labels.csv",
    original_features_only: bool = False,
) -> int:
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 16: BC Stable Users — Phase Feature Visualization")
    print("=" * 60)
    print()

    # ── [1] Load LLM labels ──────────────────────────────────────────────────
    print("[1] Loading LLM labels...")
    labels_path = Path(labels_file)
    try:
        df = pd.read_excel(labels_path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(labels_path, encoding="utf-8-sig")
    print(f"    {len(df):,} posts, {df['author'].nunique():,} users")

    # ── [2] Filter confirmed BC pill posts ──────────────────────────────────
    df = df[df["is_bc_pill"] == True].copy()
    print(f"\n[2] Confirmed BC pill posts: {len(df):,} from {df['author'].nunique():,} users")

    # ── [3] Normalize pill names & filter emergency contraception ───────────
    print("\n[3] Normalizing pill names...")
    df["pill_name_clean"] = df["pill_name"].apply(normalize_pill_name)
    n_excluded = (df["pill_name_clean"] == "__EXCLUDE__").sum()
    excluded_users = df.loc[df["pill_name_clean"] == "__EXCLUDE__", "author"].unique()
    print(f"    Posts with emergency/OTC contraception excluded: {n_excluded}")

    # Exclude posts (and users whose ONLY posts are emergency contraception)
    df = df[df["pill_name_clean"] != "__EXCLUDE__"].copy()

    # Show normalized name distribution
    name_counts = df["pill_name_clean"].dropna().value_counts()
    print(f"    Named pills remaining — {len(name_counts)} unique canonical names")
    print("    Top 20:")
    for name, cnt in name_counts.head(20).items():
        print(f"      {cnt:>4}  {name}")

    # ── [4] Identify stable users ───────────────────────────────────────────
    print("\n[4] Identifying stable users (no start/stop event in any post)...")
    user_flags = df.groupby("author").agg(
        any_started=("started_recently", "any"),
        any_stopped=("stopped_recently", "any"),
    ).reset_index()
    stable_authors = set(
        user_flags.loc[~user_flags["any_started"] & ~user_flags["any_stopped"], "author"]
    )
    print(f"    Stable users: {len(stable_authors):,}")
    print(f"    Started recently:  {user_flags['any_started'].sum():,}")
    print(f"    Stopped recently:  {user_flags['any_stopped'].sum():,}")

    df_stable = df[df["author"].isin(stable_authors)].copy()

    # Save stable user list
    stable_out = df_stable[["author", "pill_name_clean"]].drop_duplicates("author")
    stable_path = interim_dir / f"bc_stable_users_{TIMESTAMP}.csv"
    stable_out.to_csv(stable_path, index=False)
    print(f"    Saved stable user list → {stable_path.name}")

    # ── [5] Load daily aggregated timeline ──────────────────────────────────
    print("\n[5] Loading daily aggregated timeline...")
    agg_path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    if not agg_path:
        raise FileNotFoundError("No daily aggregated timeline found. Run step 06 first.")
    timeline_df = pd.read_csv(agg_path, encoding="utf-8-sig", low_memory=False)
    print(f"    {len(timeline_df):,} user-days from {timeline_df['author'].nunique():,} users")

    # Filter to stable BC users present in the timeline
    timeline_stable = timeline_df[timeline_df["author"].isin(stable_authors)].copy()
    n_found = timeline_stable["author"].nunique()
    print(f"    Stable BC users found in timeline: {n_found:,}")

    if n_found == 0:
        print("    No stable BC users in timeline. Check that author IDs match.")
        return 1

    # ── [6] Build synthetic 28-day period map ───────────────────────────────
    print("\n[6] Assigning fixed 28-day cycle to all stable users...")
    results_df = pd.DataFrame({
        "user": list(stable_authors & set(timeline_df["author"])),
        "period": 28.0,
        "method": "fixed_28day",
    })
    print(f"    {len(results_df):,} users with period = 28 days")

    # ── [7] Identify feature columns ────────────────────────────────────────
    ORIGINAL_FEATURES = [
        "negative_sentiment_mean", "positive_sentiment_mean", "num_words_mean",
        "avg_word_length_mean", "num_sentences_mean", "unique_word_fraction_mean",
        "readability_mean", "spelling_errors_frac_mean",
        "syntactic_complexity_subordination_index_mean",
        "cohesion_analysis_lexical_overlap_mean",
    ]
    meta_cols = {"author", "offset_from_cd1"}
    all_feature_cols = [
        c for c in timeline_stable.columns
        if c not in meta_cols and c.endswith("_mean")
        and pd.api.types.is_numeric_dtype(timeline_stable[c])
    ]
    if original_features_only:
        features = [f for f in ORIGINAL_FEATURES if f in all_feature_cols]
        print(f"\n[7] Using {len(features)} original features (filtered)")
    else:
        features = all_feature_cols
        print(f"\n[7] Using {len(features)} feature columns")

    # ── [8] Aggregate by phase ───────────────────────────────────────────────
    print("\n[8] Aggregating features by phase (28-day fixed cycle)...")
    phase_df = aggregate_features_by_phase(
        timeline_df=timeline_stable,
        results_df=results_df,
        features=features,
        time_col="offset_from_cd1",
        user_col="author",
        method="fixed_28day",
        normalize=True,
        average_per_user=True,
    )

    if phase_df.empty:
        print("    No phase data produced.")
        return 1

    feature_order = sorted(phase_df["feature"].unique())
    print(f"    {len(feature_order)} features with phase data")

    # ── [9] Plot ─────────────────────────────────────────────────────────────
    out_path = reports_dir / f"bc_stable_phase_features_{TIMESTAMP}.png"
    print(f"\n[9] Plotting {len(feature_order)} features → {out_path.name}...")
    plot_phase_analysis(
        phase_df=phase_df,
        features=feature_order,
        output_path=out_path,
        error_bar_type="sem",
        average_per_user=True,
    )
    print(f"    Saved → {out_path.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Phase feature visualization for stable BC pill users (fixed 28-day cycle)"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--labels-file",
        default="data/processed/bc_wide_candidates_with_labels.csv",
        help="Path to LLM-labelled BC candidates Excel/CSV file.",
    )
    ap.add_argument(
        "--original-features-only", action="store_true",
        help="Restrict to the 10 original features only.",
    )
    args = ap.parse_args()
    exit(main(
        config_path=args.config,
        labels_file=args.labels_file,
        original_features_only=args.original_features_only,
    ))
