#!/usr/bin/env python3
"""Script 23: Natural cycle vs Combined pill — phase comparison.

For every linguistic feature, compare natural cycle users to combined-pill
users (known combined + unknown-pill stable BC users) at the Menstrual and
Ovulation phases. Report features that are significant within either group
or significantly different between groups.

Groups:
  Natural  — consensus non-BC users with anchors (n~904)
  Pill     — combined + unknown-pill stable BC users (long-term or started),
             on-pill posts only, fixed 28-day cycle (n~637)

Method:
  1. Per-user z-score on FULL timeline (before any filtering)
  2. ±91 day window, adaptive phases for natural / fixed 28-day for pill
  3. Per-user mean z-score in each phase
  4. Wilcoxon signed-rank test vs 0 within each group
  5. Mann-Whitney U between groups
  6. Report: n, mean z-score, % in each direction, p-values

Output:
  reports/natural_vs_pill_comparison_<timestamp>.csv
  reports/natural_vs_pill_comparison_<timestamp>.png  (significant features)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.io import find_latest_file
from src.analysis import create_adaptive_phases, assign_phase_to_day

TIMESTAMP   = datetime.now().strftime("%Y%m%dT%H%M%S")
MAX_OFFSET  = 91
PHASES      = ["Menstrual", "Follicular", "Ovulation", "Luteal"]
TARGET_PHASES = ["Menstrual", "Ovulation"]
MIN_USERS   = 20   # minimum per-group users to report a feature×phase

PROGESTIN_KW = {
    "norethindrone", "slynd", "opill", "micronor", "heather", "camila",
    "errin", "cerazette", "cerelle", "zelleta", "desogestrel", "jencycla",
    "sharobel", "tulana", "incassia", "norgeston", "noriday", "microlut",
    "vibin mini", "jolivette", "micronorette", "nora-be", "femulen",
}

EXCLUDE_COLS = {
    "author", "offset_from_cd1",
    "arousal_coverage_mean", "aoa_coverage_mean",
    "dominance_coverage_mean", "valence_dict_coverage_mean",
    "imaginability_coverage_mean",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_pill(name: str) -> str:
    if pd.isna(name) or str(name).strip() == "":
        return "unknown"
    return "progestin_only" if any(k in str(name).lower() for k in PROGESTIN_KW) else "combined"


def build_pill_cutoffs(bc: pd.DataFrame) -> dict[str, float | None]:
    """Return {author -> on-pill cutoff offset_from_cd1} for combined+unknown users.
    None means all posts are on-pill (long-term users).
    Excludes progestin-only and stopped-only users.
    """
    cutoffs: dict[str, float | None] = {}
    bc = bc.copy()
    bc["pill_type"]        = bc["pill_name"].apply(classify_pill)
    bc["started_recently"] = bc["started_recently"].fillna(False)
    bc["long_term_user"]   = bc["long_term_user"].fillna(False)
    bc["stopped_recently"] = bc["stopped_recently"].fillna(False)

    for author, udf in bc[bc["pill_type"].isin(["combined", "unknown"])].groupby("author"):
        # Skip if any row flags this user as progestin-only
        if "progestin_only" in bc[bc["author"] == author]["pill_type"].values:
            continue
        is_lt  = udf["long_term_user"].any()
        is_st  = udf["started_recently"].any()
        is_stp = udf["stopped_recently"].any() and not is_lt and not is_st
        if is_stp or (not is_lt and not is_st):
            continue
        if is_lt:
            cutoffs[author] = None
        else:
            sr = udf[udf["started_recently"]].copy()
            sr["started_offset"]  = pd.to_numeric(sr["started_offset"],  errors="coerce")
            sr["offset_from_cd1"] = pd.to_numeric(sr["offset_from_cd1"], errors="coerce")
            sr = sr.dropna(subset=["started_offset", "offset_from_cd1"])
            if len(sr) == 0:
                continue
            cutoffs[author] = (sr["offset_from_cd1"] + sr["started_offset"]).median()
    return cutoffs


def build_phased(
    tl: pd.DataFrame,
    features: list[str],
    user_periods: dict[str, float] | None = None,
    fixed_period: float | None = None,
    cutoff_map: dict[str, float | None] | None = None,
    window: int = MAX_OFFSET,
) -> pd.DataFrame:
    """Z-score on full timeline, apply cutoff/window, assign phases."""
    feats = [f for f in features if f in tl.columns]
    df = (tl[["author", "offset_from_cd1"] + feats]
          .groupby(["author", "offset_from_cd1"])
          .mean(numeric_only=True)
          .reset_index())

    # Per-user z-score on FULL timeline
    for f in feats:
        mu = df.groupby("author")[f].transform("mean")
        sd = df.groupby("author")[f].transform("std").replace(0, np.nan)
        df[f] = (df[f] - mu) / sd

    # On-pill cutoff (before window, so window is applied on already-filtered data)
    if cutoff_map is not None:
        rows = []
        for author, udf in df.groupby("author"):
            cutoff = cutoff_map.get(author)
            rows.append(udf if cutoff is None else udf[udf["offset_from_cd1"] >= cutoff])
        df = pd.concat(rows)

    # Window
    df = df[df["offset_from_cd1"].abs() <= window].copy()

    # Phase assignment
    records = []
    for user, udf in df.groupby("author"):
        period = fixed_period or (user_periods.get(str(user)) if user_periods else None)
        if period is None:
            continue
        phases = create_adaptive_phases(float(period))
        for _, row in udf.iterrows():
            phase = assign_phase_to_day(row["offset_from_cd1"], phases)
            if phase is None:
                continue
            rec = {"author": user, "phase": phase}
            for f in feats:
                rec[f] = row[f]
            records.append(rec)

    return pd.DataFrame(records)


def per_user_phase_means(phased: pd.DataFrame, feat: str, phase: str) -> pd.Series:
    return pd.Series([
        udf[udf["phase"] == phase][feat].dropna().mean()
        for _, udf in phased.groupby("author")
        if len(udf[udf["phase"] == phase][feat].dropna()) >= 1
    ]).dropna()


def compute_stats(s: pd.Series) -> dict:
    if len(s) < 5:
        return {"n": len(s), "mean": np.nan, "pct_pos": np.nan, "p_wilcox": np.nan}
    try:
        _, pw = stats.wilcoxon(s)
    except ValueError:
        pw = float("nan")
    return {
        "n":        len(s),
        "mean":     s.mean(),
        "pct_pos":  (s > 0).mean() * 100,
        "p_wilcox": pw,
    }


def sig_stars(p: float) -> str:
    if np.isnan(p): return "   "
    return ("***" if p < 0.001 else
            ("** " if p < 0.01  else
             ("*  " if p < 0.05  else
              ("~  " if p < 0.10  else "   "))))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config("configs/base.yaml")
    interim_dir = Path(cfg["paths"]["interim"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Script 23: Natural vs Pill — Phase Comparison (all features)")
    print("=" * 65)

    # ── [1] Natural cycle users ───────────────────────────────────────────
    print("\n[1] Natural cycle users (with anchors, no BC)...")
    nat_df      = pd.read_csv(interim_dir / "consensus_periods_min23_no_bc.csv")
    nat_periods = dict(zip(nat_df["user"].astype(str), nat_df["consensus_period"]))
    print(f"    {len(nat_periods):,} users | median period {nat_df['consensus_period'].median():.1f} d")

    # ── [2] Combined pill users ───────────────────────────────────────────
    print("\n[2] Combined + unknown-pill BC users...")
    bc      = pd.read_excel(Path("data/processed/bc_wide_candidates_with_labels.csv"), engine="openpyxl")
    bc["author"] = bc["author"].astype(str)
    pill_cutoffs = build_pill_cutoffs(bc)
    n_lt  = sum(1 for v in pill_cutoffs.values() if v is None)
    n_st  = sum(1 for v in pill_cutoffs.values() if v is not None)
    print(f"    {len(pill_cutoffs):,} users  (long-term: {n_lt}, post-start cutoff: {n_st})")

    # ── [3] Load timeline ─────────────────────────────────────────────────
    print("\n[3] Loading timeline...")
    tl_path = find_latest_file(interim_dir, "timeline_daily_aggregated_with_anchors_*.csv")
    tl = pd.read_csv(tl_path, encoding="utf-8-sig", low_memory=False)
    tl["author"] = tl["author"].astype(str)
    print(f"    {len(tl):,} user-days | {tl['author'].nunique():,} users")

    features = [c for c in tl.columns
                if c not in EXCLUDE_COLS
                and tl[c].dtype in [np.float64, np.float32, float]]
    print(f"    {len(features)} features")

    # ── [4] Build phased data ─────────────────────────────────────────────
    print("\n[4] Building phased data...")
    nat_tl  = tl[tl["author"].isin(nat_periods)].copy()
    pill_tl = tl[tl["author"].isin(pill_cutoffs)].copy()

    nat_phased  = build_phased(nat_tl,  features, user_periods=nat_periods)
    pill_phased = build_phased(pill_tl, features, fixed_period=28.0, cutoff_map=pill_cutoffs)

    print(f"    Natural: {nat_phased['author'].nunique()} users")
    print(f"    Pill:    {pill_phased['author'].nunique()} users")

    # ── [5] Compute stats for every feature × phase ───────────────────────
    print("\n[5] Computing statistics...")
    rows = []
    for feat in features:
        for phase in TARGET_PHASES:
            nat_s  = per_user_phase_means(nat_phased,  feat, phase)
            pill_s = per_user_phase_means(pill_phased, feat, phase)

            if len(nat_s) < MIN_USERS or len(pill_s) < MIN_USERS:
                continue

            nat_r  = compute_stats(nat_s)
            pill_r = compute_stats(pill_s)
            _, mwu_p = stats.mannwhitneyu(nat_s, pill_s, alternative="two-sided")

            rows.append({
                "feature": feat,
                "phase":   phase,
                # Natural
                "nat_n":       nat_r["n"],
                "nat_mean":    nat_r["mean"],
                "nat_pct_pos": nat_r["pct_pos"],
                "nat_p":       nat_r["p_wilcox"],
                # Pill
                "pill_n":       pill_r["n"],
                "pill_mean":    pill_r["mean"],
                "pill_pct_pos": pill_r["pct_pos"],
                "pill_p":       pill_r["p_wilcox"],
                # Between groups
                "mwu_p":   mwu_p,
                "opposite": nat_r["mean"] * pill_r["mean"] < 0,
            })

    results = pd.DataFrame(rows)
    print(f"    {len(results)} feature×phase combinations computed")

    # ── [6] Report significant results ───────────────────────────────────
    # Significant = nat_p < 0.05 OR pill_p < 0.05 OR mwu_p < 0.05
    sig = results[
        (results["nat_p"]  < 0.05) |
        (results["pill_p"] < 0.05) |
        (results["mwu_p"]  < 0.05)
    ].copy()
    sig = sig.sort_values(["phase", "mwu_p"])

    def short(f):
        return (f.replace("syntactic_complexity_", "")
                 .replace("Syntactic_phrase_distribution_", "SPD_")
                 .replace("idea_density_cpidr_density_mean", "cpidr_density")
                 .replace("idea_density_depid_density_mean", "depid_density")
                 .replace("arousal_average_mean", "arousal_avg")
                 .replace("_mean", "").replace("_", " "))

    print(f"\n{'─'*115}")
    print(f"  {'Feature':<40} {'Phase':<11}  "
          f"{'Nat n':>5} {'Nat mean':>8} {'Nat%':>5} {'Nat p':>7}  "
          f"{'Pill n':>6} {'Pill mean':>9} {'Pill%':>5} {'Pill p':>7}  "
          f"{'MWU p':>7}  {'':>3}")
    print(f"{'─'*115}")

    for phase in TARGET_PHASES:
        phase_rows = sig[sig["phase"] == phase]
        if phase_rows.empty:
            continue
        print(f"\n  ── {phase} ({len(phase_rows)} significant features) ──")
        for _, r in phase_rows.iterrows():
            opp = "←→" if r.opposite else "  "
            mwu_s = sig_stars(r.mwu_p)
            nat_s = sig_stars(r.nat_p)
            pil_s = sig_stars(r.pill_p)
            print(f"  {short(r.feature):<40} {r.phase:<11}  "
                  f"{int(r.nat_n):>5} {r.nat_mean:>+8.3f} {r.nat_pct_pos:>4.0f}% {r.nat_p:>6.4f}{nat_s}  "
                  f"{int(r.pill_n):>6} {r.pill_mean:>+9.3f} {r.pill_pct_pos:>4.0f}% {r.pill_p:>6.4f}{pil_s}  "
                  f"{r.mwu_p:>7.4f}{mwu_s}  {opp}")

    # ── [7] Highlight opposite-direction, both significant ────────────────
    both_opp = results[
        results["opposite"] &
        (results["nat_p"]  < 0.05) &
        (results["pill_p"] < 0.05)
    ].sort_values("mwu_p")

    print(f"\n\n{'═'*90}")
    print(f"  OPPOSITE DIRECTIONS — both groups individually significant (p<0.05)")
    print(f"{'═'*90}")
    print(f"  {'Feature':<40} {'Phase':<11}  "
          f"{'Nat mean':>8} {'Nat p':>7}  "
          f"{'Pill mean':>9} {'Pill p':>7}  {'MWU p':>7}")
    print(f"{'─'*90}")
    for _, r in both_opp.iterrows():
        mwu_s = sig_stars(r.mwu_p)
        print(f"  {short(r.feature):<40} {r.phase:<11}  "
              f"{r.nat_mean:>+8.3f} {r.nat_p:>6.4f}{sig_stars(r.nat_p)}  "
              f"{r.pill_mean:>+9.3f} {r.pill_p:>6.4f}{sig_stars(r.pill_p)}  "
              f"{r.mwu_p:>7.4f}{mwu_s}")

    # ── [8] Save CSV ──────────────────────────────────────────────────────
    csv_path = reports_dir / f"natural_vs_pill_comparison_{TIMESTAMP}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n\n  Full results CSV → {csv_path.name}")

    # ── [9] Plot significant features ────────────────────────────────────
    print("\n[9] Plotting...")
    _plot_significant(sig, short, reports_dir)

    print("\nDone.")


def _plot_significant(sig: pd.DataFrame, short_fn, reports_dir: Path) -> None:
    for phase in TARGET_PHASES:
        phase_df = sig[sig["phase"] == phase].copy()
        if phase_df.empty:
            continue

        phase_df = phase_df.sort_values("mwu_p")
        labels   = [short_fn(f) for f in phase_df["feature"]]
        x        = np.arange(len(labels))
        w        = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5))
        ax.bar(x - w/2, phase_df["nat_mean"],  w, color="#1f77b4", alpha=0.85, label="Natural")
        ax.bar(x + w/2, phase_df["pill_mean"], w, color="#ff7f0e", alpha=0.85, label="Pill (combined+unknown)")

        # Significance markers above bars
        for i, (_, r) in enumerate(phase_df.iterrows()):
            y_max = max(abs(r.nat_mean), abs(r.pill_mean)) + 0.05
            if r.mwu_p < 0.05:
                stars = "***" if r.mwu_p < 0.001 else ("**" if r.mwu_p < 0.01 else "*")
                ax.text(i, y_max + 0.02, stars, ha="center", fontsize=7, color="black")
            if r.opposite and r.nat_p < 0.05 and r.pill_p < 0.05:
                ax.text(i, y_max + 0.08, "←→", ha="center", fontsize=7, color="red")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean z-score (per-user, vs own baseline)")
        ax.set_title(f"{phase} phase — significant features\n"
                     f"Blue=Natural (n~904), Orange=Pill (n~637)  |  ←→ = opposite directions",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        plt.tight_layout()

        out = reports_dir / f"natural_vs_pill_{phase.lower()}_{TIMESTAMP}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → {out.name}")


if __name__ == "__main__":
    main()
