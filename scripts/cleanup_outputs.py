"""
Move old run outputs to a holding folder. This script does not delete files.

Default: dry run. The script writes the move list to data/_to_delete_plan.txt
and prints a summary. With --apply, it moves the files to
data/_to_delete_<date>/<same relative path>. You can move a file back if needed.

Rules
-----
data/raw/, data/outputs/, outputs/, voice_cycle/: not scanned (never touched).
data/interim/, data/processed/:
    keep every file that matches KEEP_PATTERNS (files the paper uses);
    keep the NEWEST file of each name group (name with timestamps removed);
    move all other files.
data/submission/: move all (old March snapshot, not used by the paper).
reports/: keep only KEEP_PATTERNS; move all other entries (top-level files/dirs,
    and inside reports/ebm/bc/).

Usage
-----
    python scripts/cleanup_outputs.py            # dry run
    python scripts/cleanup_outputs.py --apply    # move files
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths relative to ROOT. fnmatch patterns.
KEEP_PATTERNS = [
    # --- step 02/03/04 outputs and annotation samples
    "data/interim/*validation_sample*",
    "data/interim/posts_all_users_preprocessed_*_20260517*",
    # --- lineage of the 20260608 run (GAM cohort + Yngve fix input)
    "data/interim/timeline_with_offsets_*_20260608*",
    "data/interim/timeline_daily_aggregated_*_20260608*",
    "data/interim/timeline_phase_labeled_fixed28*_20260608*",
    "data/interim/eligible_users_two_cycles_span28_gap7_no_anchors_w28_20260608T202125*",
    # --- birth-control classifier inputs (original + yngvefix)
    "data/interim/bc_timeline_20260603T000200.csv",
    "data/interim/bc_timeline_20260603T000200_yngvefix_*",
    "data/interim/non_bc_timeline_20260603T000200.csv",
    "data/interim/non_bc_timeline_20260603T000200_yngvefix_*",
    # --- Appendix E frequency cohort (07c default input)
    "data/interim/eligible_users_timeline_minposts20_span100_no_anchors_20260517T134609.csv",
    # --- initial cohort DB (15,165 users) used by demographics/extract_top_subreddits.py
    "data/submission/processed/users_database_CD_20260329T064620.csv",
    # --- processed: GPT-5 labels (not reproducible without the API run)
    "data/processed/bc_wide_candidates_with_labels.csv",
    "data/processed/user_age_distribution_notitle.png",
    # --- paper figures and the file sent to the supervisor
    "reports/gam_cycle_curves_yngvefix",
    "reports/ebm/bc/*yngvefix*",
    "reports/phase_bars",
    "reports/freq_gap10_120",
    "reports/yngve_posts_for_recompute_pm365.csv.gz",
    # --- not reproducible by any current script (hand-written / manual / removed code)
    "reports/hormone_pivot",
    "reports/ml",
    "reports/ebm/ebm_expert_review_phases*",
    "reports/yngve_recompute_userset*",
    "data/interim/*NOPCOS*",
    "data/interim/*DOWNSAMPLED*",
    "data/interim/prepared_ebm_input.csv",
    "data/interim/*paper_signal*",
]

# --strict: in data/interim and data/processed keep ONLY these (plus KEEP_PATTERNS).
# One checkpoint per pipeline step of the paper run.
STRICT_KEEP = [
    # step 02: anchors + uncertainty flag (newest = Oct 10 run)
    "data/interim/moon_with_uncertainty_20261010T205503.csv",
    "data/interim/moon2_with_uncertainty_20261010T205508.csv",
    # step 03: user database
    "data/processed/users_database_CD_20260704T144759.csv",
    # step 05: timeline with offsets
    "data/interim/timeline_no_anchors_20260608T195836.csv",
    "data/interim/timeline_with_anchors_20260608T195859.csv",
    # step 06: daily aggregated (second 0608 run feeds step 07) + yngvefix re-aggregation
    "data/interim/timeline_daily_aggregated_*_20260608T2005*",
    "data/interim/timeline_daily_aggregated_*_20260608T2006*",
    "data/interim/timeline_daily_aggregated_*_20260906T135827.csv",
    # step 07: phase labels
    "data/interim/timeline_phase_labeled_fixed28_20260608T200705.csv",
    "data/interim/timeline_phase_labeled_fixed28_no_anchors_20260608T200730.csv",
    # demographics (Figure 1)
    "data/processed/age_disclosure_posts_20260704T164512.csv",
    "data/processed/user_age_distribution_20260704T164512.*",
]

TS_RE = re.compile(r"_?\d{8}[T_]\d{6}")


def is_kept(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in KEEP_PATTERNS)


def group_key(name: str) -> str:
    return TS_RE.sub("", name)


def size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def plan(strict: bool = False) -> tuple[list[Path], list[Path]]:
    move, keep = [], []

    # data/interim and data/processed: keep list + newest per group
    for d in ("data/interim", "data/processed"):
        files = [f for f in (ROOT / d).iterdir() if f.is_file() and not f.name.startswith(".")]
        groups: dict[str, list[Path]] = {}
        for f in files:
            groups.setdefault(group_key(f.name), []).append(f)
        for members in groups.values():
            newest = max(members, key=lambda f: f.stat().st_mtime)
            for f in members:
                rel = str(f.relative_to(ROOT))
                if strict:
                    ok = is_kept(rel) or any(fnmatch.fnmatch(rel, p) for p in STRICT_KEEP)
                else:
                    ok = f is newest or is_kept(rel)
                (keep if ok else move).append(f)

    # data/submission: move everything except KEEP_PATTERNS
    sub = ROOT / "data/submission"
    if sub.exists():
        for f in sub.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(ROOT))
                (keep if is_kept(rel) else move).append(f)

    # reports: keep list only (top level, plus inside reports/ebm/bc)
    rep = ROOT / "reports"
    for entry in rep.iterdir():
        if entry.name.startswith("."):
            continue
        rel = str(entry.relative_to(ROOT))
        if rel == "reports/ebm":
            for sub_entry in entry.iterdir():
                srel = str(sub_entry.relative_to(ROOT))
                if srel == "reports/ebm/bc":
                    for e in sub_entry.iterdir():
                        erel = str(e.relative_to(ROOT))
                        (keep if is_kept(erel) else move).append(e)
                else:
                    (keep if is_kept(srel) else move).append(sub_entry)
            continue
        (keep if is_kept(rel) else move).append(entry)

    return move, keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="move the files (default: dry run)")
    ap.add_argument("--strict", action="store_true",
                    help="interim/processed: keep only KEEP_PATTERNS + STRICT_KEEP (no 'newest' rule)")
    args = ap.parse_args()

    move, keep = plan(strict=args.strict)
    move_sizes = {p: size_of(p) for p in move}
    keep_size = sum(size_of(p) for p in keep)

    plan_file = ROOT / "data/_to_delete_plan.txt"
    with plan_file.open("w") as fh:
        fh.write("# MOVE\n")
        for p in sorted(move):
            fh.write(f"{move_sizes[p] / 1e6:10.1f} MB  {p.relative_to(ROOT)}\n")
        fh.write("\n# KEEP\n")
        for p in sorted(keep):
            fh.write(f"{size_of(p) / 1e6:10.1f} MB  {p.relative_to(ROOT)}\n")

    total = sum(move_sizes.values())
    print(f"Move: {len(move)} entries, {total / 1e9:.2f} GB")
    print(f"Keep: {len(keep)} entries, {keep_size / 1e9:.2f} GB (scanned folders only)")
    print(f"Full list: {plan_file.relative_to(ROOT)}")

    if not args.apply:
        print("Dry run. Nothing moved. Add --apply to move.")
        return

    dest_root = ROOT / f"data/_to_delete_{date.today():%Y%m%d}"
    for p in move:
        dest = dest_root / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
    print(f"Moved to {dest_root.relative_to(ROOT)}. Check the paper scripts, then delete that folder.")


if __name__ == "__main__":
    main()
