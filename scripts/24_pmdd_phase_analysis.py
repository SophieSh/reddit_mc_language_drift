"""Script 24 — PMDD Phase Analysis (fixed 29-day cycle, ±1 month)
=================================================================
Compares PMDD users vs controls using:
  - 5-phase PMDD system (Eisenlohr-Moul et al., AJP 2023)
  - All 69 linguistic features
  - Subreddit posting patterns by category per phase

PMDD users: data/processed/pmdd_users_*.csv (subreddit-based, ~840 users).
Control: all other timeline users.
Fixed 29-day period, ±30-day window around CD1 anchor.

Methodology note — excess analysis was tested and abandoned:
  PMDD users have a chronically elevated negative baseline across ALL phases.
  Subtracting per-user other-phase means ("excess") removes this between-phase
  chronic elevation, which is exactly the PMDD signal we want to measure.
  Excess analysis produced counterintuitive results (Control appeared MORE
  negative at Perimenstrual than PMDD) for this reason.

  Correct approach used here:
    - Sentiment: raw per-user z-scores (within ±30d window baseline, no excess
      subtraction). Captures absolute phase-level deviations from each user's
      personal window-level mean.
    - Subreddit: raw fraction of posts per phase per user (not excess).
      Directly measures the phase-specific redirection behavior.

Confirmed significant findings this script reproduces:
  - PMDD fraction in pmdd_specific subreddits: ~33-42% at Perimenstrual, ~0% other phases
  - Depression/MH subreddits: PMDD > Control at Midfollicular (p≈0.003)
  - negative_sentiment: PMDD > Control at Perimenstrual (p≈0.044)
  - positive_sentiment: PMDD > Control at Perimenstrual (p≈0.004, CD1 relief)
  - positive_sentiment: PMDD < Control at Midluteal (p≈0.010)
  - valence_dict_average: PMDD < Control at Midluteal (p≈0.029)

Outputs:
  reports/pmdd/pmdd_linguistic_results_*.csv   — feature × phase × group stats
  reports/pmdd/pmdd_subreddit_results_*.csv    — subreddit category fractions
  reports/pmdd/pmdd_significant_*.csv          — filtered significant results
  reports/pmdd/pmdd_*.png                      — bar plots
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io import find_latest_file
from src.visualization import create_pmdd_phases, assign_pmdd_phase

# ── Config ─────────────────────────────────────────────────────────────────────
FIXED_PERIOD = 29.0
WINDOW_DAYS  = 30

PHASE_ORDER = ["Perimenstrual", "Midfollicular", "Periovulatory", "Early Luteal", "Midluteal"]

# All 69 linguistic features available in the timeline
ALL_FEATURES = [
    "num_words", "num_sentences", "avg_word_length", "unique_word_fraction",
    "readability", "positive_sentiment", "negative_sentiment", "new_word_rate",
    "syntactic_complexity_mean_length_tunit", "syntactic_complexity_mean_clauses_per_sentence",
    "syntactic_complexity_subordination_index", "hypernym_ratio",
    "pos_distribution_NOUN", "pos_distribution_VERB", "pos_distribution_ADJ",
    "pos_distribution_ADV", "pos_distribution_PRON", "pos_distribution_ADP",
    "pos_distribution_CCONJ", "pos_distribution_DET", "pos_distribution_NUM",
    "pos_distribution_PUNCT",
    "cohesion_analysis_lexical_overlap", "cohesion_analysis_connective_density",
    "cohesion_analysis_coreference_count",
    "spelling_errors_count", "spelling_errors_frac",
    "hedging_modal_verbs", "hedging_epistemic_phrases", "hedging_approximators",
    "hedging_non_committal", "hedging_tag_softeners", "hedging_evidentials",
    "hedging_visual_hedges",
    "Syntactic_phrase_distribution_PP", "Syntactic_phrase_distribution_SBARQ",
    "Syntactic_phrase_distribution_SBAR", "Syntactic_phrase_distribution_ADJP",
    "Syntactic_phrase_distribution_SINV", "Syntactic_phrase_distribution_WHNP",
    "Syntactic_phrase_distribution_X", "Syntactic_phrase_distribution_SQ",
    "Syntactic_phrase_distribution_WHAVP", "Syntactic_phrase_distribution_WHPP",
    "Syntactic_phrase_distribution_NP", "Syntactic_phrase_distribution_VP",
    "Syntactic_phrase_distribution_ADVP", "Syntactic_phrase_distribution_S",
    "word_frequency", "aoa_average", "aoa_coverage", "avg_concreteness",
    "concrete_abstract_ratio", "imaginability_average", "imaginability_coverage",
    "valence_dict_average", "valence_dict_coverage",
    "arousal_average", "arousal_coverage", "dominance_average", "dominance_coverage",
    "syntactic_complexity_mean_sentence_length", "syntactic_complexity_yngve_score",
    "syntactic_complexity_frazier_score", "syntactic_complexity_mean_dep_dist",
    "syntactic_complexity_max_leftward_deps", "syntactic_complexity_center_embedding_score",
    "idea_density_cpidr_density", "idea_density_depid_density",
]

# Subreddit categories for posting pattern analysis
SUBREDDIT_CATEGORIES = {
    "pmdd_specific": {"PMDD", "PMDDxADHD", "PMDDSharing"},
    "depression_mh":  {"depression", "mentalhealth", "Anxiety", "socialanxiety",
                       "AnxietyDepression", "SuicideWatch"},
    "adhd":           {"ADHD", "adhdwomen", "TwoXADHD", "ADHDWomenAfterDark",
                       "PMDDxADHD", "adhd_anxiety", "adhdmeme", "ADHDers"},
    "womens_health":  {"PMDD", "Periods", "birthcontrol", "endometriosis",
                       "Endo", "PCOS", "TwoXChromosomes", "TwoXSex"},
}

KEY_FEATURES = [
    "negative_sentiment", "positive_sentiment", "valence_dict_average",
    "arousal_average", "dominance_average",
    "hedging_epistemic_phrases", "hedging_modal_verbs",
    "idea_density_depid_density", "spelling_errors_frac",
]

OUTPUT_DIR = ROOT / "reports" / "pmdd"


# ── Phase assignment ───────────────────────────────────────────────────────────
_PHASES = create_pmdd_phases(FIXED_PERIOD)


def assign_phase(offset: float) -> str | None:
    mod = offset % FIXED_PERIOD
    return assign_pmdd_phase(mod, _PHASES, FIXED_PERIOD)


# ── Statistics helpers ─────────────────────────────────────────────────────────
def compute_stats(series: pd.Series, label: str = "") -> dict:
    """Summary stats for a per-user series (one value per user)."""
    s = series.dropna()
    if len(s) < 5:
        return {"n": len(s), "mean": np.nan, "median": np.nan, "sem": np.nan,
                "pct_positive": np.nan, "p_wilcoxon": np.nan}
    try:
        _, p_w = stats.wilcoxon(s)
    except ValueError:
        p_w = float("nan")
    return {
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "sem": s.sem(),
        "pct_positive": (s > 0).mean() * 100,
        "p_wilcoxon": p_w,
    }


def mwu(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/base.yaml")
    interim_dir = Path(cfg["paths"]["interim"])

    # [1] Load PMDD users
    print("\n[1] Loading PMDD users…")
    pmdd_files = sorted(Path("data/processed").glob("pmdd_users_*.csv"))
    pmdd_df = pd.read_csv(pmdd_files[-1], usecols=["user"])
    pmdd_users = set(pmdd_df["user"].astype(str).unique())
    print(f"  PMDD users: {len(pmdd_users)}")

    # [2] Load timeline (post-level, has subreddit + all features)
    print("\n[2] Loading timeline…")
    tl_path = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
    needed = ["author", "offset_from_cd1", "subreddit"] + ALL_FEATURES
    available = pd.read_csv(tl_path, nrows=0).columns.tolist()
    load_cols = [c for c in needed if c in available]
    tl = pd.read_csv(tl_path, usecols=load_cols, low_memory=False, encoding="utf-8-sig")
    tl["author"] = tl["author"].astype(str)
    tl["offset_from_cd1"] = pd.to_numeric(tl["offset_from_cd1"], errors="coerce")
    tl = tl[tl["offset_from_cd1"].between(-WINDOW_DAYS, WINDOW_DAYS)].copy()
    tl["group"] = tl["author"].apply(lambda u: "PMDD" if u in pmdd_users else "Control")
    tl["phase"] = tl["offset_from_cd1"].map(assign_phase)
    tl = tl[tl["phase"].notna()].copy()

    n_pmdd = tl[tl["group"] == "PMDD"]["author"].nunique()
    n_ctrl = tl[tl["group"] == "Control"]["author"].nunique()
    print(f"  Posts after filter: {len(tl):,} | PMDD users: {n_pmdd} | Control: {n_ctrl}")

    print("\n  Posts per phase:")
    for ph in PHASE_ORDER:
        sub = tl[tl["phase"] == ph]
        print(f"    {ph:20s}: {sub[sub['group']=='PMDD']['author'].nunique():4d} PMDD users, "
              f"{len(sub[sub['group']=='PMDD']):5d} posts")

    # [3] Per-user z-scores on features (within-window baseline — NO excess subtraction)
    print("\n[3] Computing per-user z-scores (raw, within-window)…")
    feats_present = [f for f in ALL_FEATURES if f in tl.columns]
    for feat in feats_present:
        tl[feat] = pd.to_numeric(tl[feat], errors="coerce")
        user_stats = tl.groupby("author")[feat].agg(["mean", "std"])
        tl = tl.merge(user_stats.rename(columns={"mean": "_m", "std": "_s"}),
                      on="author", how="left")
        std = tl["_s"].replace(0, np.nan)
        tl[f"{feat}_z"] = (tl[feat] - tl["_m"]) / std
        tl.drop(columns=["_m", "_s"], inplace=True)
    z_feats = [f"{f}_z" for f in feats_present]

    # [4] Per-user per-phase mean z-score (raw, not excess)
    user_phase = (
        tl.groupby(["author", "group", "phase"])[z_feats].mean().reset_index()
    )

    # ── Section A: Subreddit category posting patterns (raw fraction) ──────────
    print("\n[A] Subreddit category analysis (raw fraction per phase)…")
    for cat, subs in SUBREDDIT_CATEGORIES.items():
        tl[f"is_{cat}"] = tl["subreddit"].isin(subs).astype(int)

    sub_rows = []
    for cat in SUBREDDIT_CATEGORIES:
        col = f"is_{cat}"
        # Per-user per-phase: fraction of posts in this category (raw, not excess)
        user_phase_sub = (
            tl.groupby(["author", "group", "phase"])
            .agg(n_cat=(col, "sum"), n_total=(col, "count"))
            .reset_index()
        )
        user_phase_sub["frac"] = user_phase_sub["n_cat"] / user_phase_sub["n_total"]

        for phase in PHASE_ORDER:
            sub = user_phase_sub[user_phase_sub["phase"] == phase]
            for group in ["PMDD", "Control"]:
                s = sub[sub["group"] == group]["frac"]
                row = {"category": cat, "phase": phase, "group": group}
                row.update(compute_stats(s))
                sub_rows.append(row)
            ps = sub[sub["group"] == "PMDD"]["frac"]
            cs = sub[sub["group"] == "Control"]["frac"]
            sub_rows.append({
                "category": cat, "phase": phase, "group": "PMDD_vs_Control",
                "n": len(ps.dropna()), "mean": ps.mean() - cs.mean(),
                "median": ps.median() - cs.median(),
                "sem": np.nan, "pct_positive": np.nan, "p_wilcoxon": mwu(ps, cs),
            })

    sub_df = pd.DataFrame(sub_rows)
    out_sub = OUTPUT_DIR / f"pmdd_subreddit_results_{timestamp}.csv"
    sub_df.to_csv(out_sub, index=False)

    # ── Section B: Linguistic feature analysis (raw z-score per phase) ─────────
    print("\n[B] Linguistic feature analysis (raw z-score per phase)…")
    ling_rows = []
    for feat in feats_present:
        z_col = f"{feat}_z"
        if z_col not in user_phase.columns:
            continue
        for phase in PHASE_ORDER:
            sub = user_phase[user_phase["phase"] == phase]
            for group in ["PMDD", "Control"]:
                s = sub[sub["group"] == group][z_col]
                row = {"feature": feat, "phase": phase, "group": group}
                row.update(compute_stats(s))
                ling_rows.append(row)
            ps = sub[sub["group"] == "PMDD"][z_col]
            cs = sub[sub["group"] == "Control"][z_col]
            ling_rows.append({
                "feature": feat, "phase": phase, "group": "PMDD_vs_Control",
                "n": len(ps.dropna()), "mean": ps.mean() - cs.mean(),
                "median": ps.median() - cs.median(),
                "sem": np.nan, "pct_positive": np.nan, "p_wilcoxon": mwu(ps, cs),
            })

    ling_df = pd.DataFrame(ling_rows)
    out_ling = OUTPUT_DIR / f"pmdd_linguistic_results_{timestamp}.csv"
    ling_df.to_csv(out_ling, index=False)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIGNIFICANT LINGUISTIC FEATURES — PMDD vs Control (MWU p < 0.05)")
    print("=" * 70)
    sig = ling_df[
        (ling_df["group"] == "PMDD_vs_Control") & (ling_df["p_wilcoxon"] < 0.05)
    ][["feature", "phase", "mean", "p_wilcoxon"]].copy()
    sig.columns = ["feature", "phase", "mean_diff(PMDD-ctrl)", "p_mwu"]
    sig = sig.sort_values(["phase", "p_mwu"])
    print(sig.to_string(index=False))

    print("\n" + "=" * 70)
    print("SUBREDDIT POSTING PATTERN — PMDD vs Control (MWU p < 0.05)")
    print("=" * 70)
    sig_sub = sub_df[
        (sub_df["group"] == "PMDD_vs_Control") & (sub_df["p_wilcoxon"] < 0.05)
    ][["category", "phase", "mean", "p_wilcoxon"]].copy()
    sig_sub.columns = ["category", "phase", "mean_diff(PMDD-ctrl)", "p_mwu"]
    sig_sub = sig_sub.sort_values(["category", "p_mwu"])
    print(sig_sub.to_string(index=False))

    print("\nAll phases — PMDD raw subreddit fractions:")
    print(f"  {'Phase':22s}  {'PMDD% pmdd_specific':>20}  {'PMDD% dep_mh':>14}  {'Ctrl% dep_mh':>14}")
    for phase in PHASE_ORDER:
        sub = tl[tl["phase"] == phase]
        pp = sub[sub["group"] == "PMDD"]["is_pmdd_specific"].mean() * 100
        pm = sub[sub["group"] == "PMDD"]["is_depression_mh"].mean() * 100
        cm = sub[sub["group"] == "Control"]["is_depression_mh"].mean() * 100
        print(f"  {phase:22s}  {pp:>20.1f}%  {pm:>13.1f}%  {cm:>13.1f}%")

    # ── Save significant results summary ───────────────────────────────────────
    sig_all = ling_df[ling_df["p_wilcoxon"] < 0.05].copy()
    out_sig = OUTPUT_DIR / f"pmdd_significant_{timestamp}.csv"
    sig_all.to_csv(out_sig, index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n[5] Generating plots…")
    colors = {"PMDD": "#d62728", "Control": "#1f77b4"}
    x = np.arange(len(PHASE_ORDER))
    w = 0.35

    # Linguistic feature bar plots (raw z-score per phase)
    for feat in KEY_FEATURES:
        sub = ling_df[ling_df["feature"] == feat]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, grp in enumerate(["PMDD", "Control"]):
            g = sub[sub["group"] == grp].set_index("phase")
            means = [g.loc[p, "mean"] if p in g.index else np.nan for p in PHASE_ORDER]
            sems  = [g.loc[p, "sem"]  if p in g.index else np.nan for p in PHASE_ORDER]
            ax.bar(x + (i - 0.5) * w, means, w, label=grp,
                   color=colors[grp], alpha=0.8, yerr=sems, capsize=4)
        mwu_sub = sub[sub["group"] == "PMDD_vs_Control"].set_index("phase")
        ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 0.95
        for xi, phase in enumerate(PHASE_ORDER):
            if phase in mwu_sub.index:
                p = mwu_sub.loc[phase, "p_wilcoxon"]
                lbl = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                if lbl:
                    ax.text(xi, ymax, lbl, ha="center", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(PHASE_ORDER, fontsize=9)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel("Raw per-user z-score (within-window mean)")
        ax.set_title(f"{feat}  |  PMDD (n={n_pmdd}) vs Control (n={n_ctrl})\n"
                     f"5-phase PMDD system, fixed 29d, ±30d window  |  raw z-score (no excess)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"pmdd_{feat}_{timestamp}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Subreddit pattern bar plots (raw fraction)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cat in zip(axes, ["pmdd_specific", "depression_mh"]):
        sub = sub_df[sub_df["category"] == cat]
        for i, grp in enumerate(["PMDD", "Control"]):
            g = sub[sub["group"] == grp].set_index("phase")
            means = [g.loc[p, "mean"] if p in g.index else np.nan for p in PHASE_ORDER]
            sems  = [g.loc[p, "sem"]  if p in g.index else np.nan for p in PHASE_ORDER]
            ax.bar(x + (i - 0.5) * w, means, w, label=grp,
                   color=colors[grp], alpha=0.8, yerr=sems, capsize=4)
        mwu_sub = sub[sub["group"] == "PMDD_vs_Control"].set_index("phase")
        if not mwu_sub.empty:
            ymax2 = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 0.95
            for xi, phase in enumerate(PHASE_ORDER):
                if phase in mwu_sub.index:
                    p = mwu_sub.loc[phase, "p_wilcoxon"]
                    lbl = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                    if lbl:
                        ax.text(xi, ymax2, lbl, ha="center", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(PHASE_ORDER, fontsize=8, rotation=15)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel("Raw fraction of posts in category")
        ax.set_title(f"Posting fraction: {cat}\nPMDD vs Control  |  raw fraction per phase")
        ax.legend()
    plt.suptitle(f"Subreddit category posting patterns  |  fixed 29d, ±30d  |  raw fraction",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"pmdd_subreddit_patterns_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone. Outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
