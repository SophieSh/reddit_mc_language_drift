# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project detects **cyclical language drift across the menstrual cycle** using Reddit data. Users who post anchor phrases (e.g., "I got my period today") are identified, and their linguistic patterns are tracked relative to their cycle day 1 (CD1).

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_md   # required for syntactic/cohesion features
```

> **Note:** spaCy requires Python 3.11 or 3.12 — it does not work with Python 3.14+. The repo contains both a `.venv` (Python 3.14) and `.venv312` (Python 3.12); use `.venv312` when running spaCy-dependent code.

## Common Commands

```bash
# Run the full pipeline (Makefile targets are outdated — scripts have been renumbered)
python scripts/01_ingest.py --config configs/base.yaml
python scripts/02_preprocess_moon1.py --config configs/base.yaml
python scripts/02_preprocess_moon2.py --config configs/base.yaml
python scripts/02_preprocess_moon3.py --config configs/base.yaml
python scripts/03_create_users_database.py --config configs/base.yaml
python scripts/04_filter_and_preprocess_posts.py --config configs/base.yaml
python scripts/05_build_timeline.py --config configs/base.yaml
python scripts/06_aggregate_and_normalize.py --config configs/base.yaml
python scripts/07_run_fft_analysis.py --config configs/base.yaml
python scripts/08_pmdd.py --config configs/base.yaml

# Formatting
ruff check --fix . || true
python -m black . || true

# Tests (none currently exist)
pytest -q
```

## Pipeline Architecture

Each numbered script is a self-contained step that reads from `data/` and writes timestamped outputs back to `data/`:

| Step | Script | Input → Output |
|------|--------|----------------|
| 1 | `01_ingest.py` | Validates raw feature files (moon1/moon2 CSVs) |
| 2 | `02_preprocess_moon{1,2,3}.py` | Raw anchor Excel/CSV → `data/interim/moon*_with_uncertainty_*.csv` |
| 3 | `03_create_users_database.py` | Uncertainty files → `data/processed/users_database_{CD,DPO}_*.csv` |
| 4 | `04_filter_and_preprocess_posts.py` | Users DB + raw posts → `data/interim/posts_all_users_preprocessed_*.csv` |
| 5 | `05_build_timeline.py` | Preprocessed posts + users DB → `data/interim/timeline_with_offsets_*.csv` |
| 6 | `06_aggregate_and_normalize.py` | Timeline → `data/interim/timeline_daily_aggregated_*.csv` |
| 7 | `07_run_fft_analysis.py` | Daily timeline → `data/interim/periodicity_results_*.csv` |
| 8 | `08_pmdd.py` | Periodicity + timeline → PMDD phase analysis outputs |

Scripts use `find_latest_file()` from `src/io.py` to auto-detect their input from previous steps.

## Configuration (`configs/base.yaml`)

All scripts accept `--config` to override the default `configs/base.yaml`. Key sections:

- **`paths`**: directories (`data/raw`, `data/interim`, `data/processed`) and specific file names for moon1/moon2/moon3 anchor and posts files
- **`patterns`**: regex patterns organized by moon source; `pattern_1–6,9` are CD (cycle day) patterns; `pattern_7` is DPO (days past ovulation); `pattern_8` excluded
- **`analysis`**: periodicity detection methods (`lomb_scargle`, `fft_interpolation`, `fft_zeropad`), normalization, SNR/FAP thresholds
- **`subreddits`**: lists for PMDD, depression, suicide, mental_health groupings used in step 8

## `src/` Library

- **`config.py`**: Loads `configs/base.yaml` at import time; exports `MIN_DATA_POINTS`, `EPSILON`, `DEFAULT_PERIOD_MIN/MAX`, `AVG_DAYS_PER_MONTH`
- **`preprocess.py`**: Text cleaning, timestamp parsing, anchor matching, uncertainty flagging, offset calculation (`offset_from_cd1`), NaN filtering
- **`features.py`**: Feature computation: VADER sentiment, TextBlob sentiment, basic linguistic (word count, Flesch-Kincaid), advanced linguistic (MATTR, spelling), syntactic complexity, cohesion
- **`analysis.py`**: Periodicity detection (FFT interpolation, FFT zero-pad, Lomb-Scargle), signal normalization, phase assignment, phase statistics
- **`timeline.py`**: Builds anchor dictionaries, user selection by pattern, timeline loading utilities
- **`attributes.py`**: Low-level NLP helpers (spaCy-based syntactic complexity, MATTR scoring, cohesion scoring, spelling error fraction)
- **`io.py`**: `find_latest_file()` glob helper, `save_with_timestamp()`, JSONL parsing, periodicity results finder
- **`utils.py`**: Feature column identification, user extraction from subreddits, latest file loading
- **`visualization.py`**: Plotting functions for phase analysis outputs

## Key Data Concepts

- **Anchor post**: A post matching a cycle-day regex (e.g., "got my period today"), which establishes the user's CD1 date
- **`offset_from_cd1`**: Days since the user's most recent anchor post; negative = pre-menstrual, 0 = CD1
- **moon1/moon2/moon3**: Three datasets of anchor posts collected from different Reddit search patterns; moon1 and moon2 are CD patterns, moon3 contains DPO and LMP patterns
- **CD vs. DPO users**: `users_database_CD_*.csv` for cycle-day patterns; `users_database_DPO_*.csv` for days-past-ovulation patterns
- **Uncertainty flag**: Posts where the exact cycle day is ambiguous (e.g., "yesterday") get `has_uncertainty=True` and are excluded from the users database

## Makefile Note

The `Makefile` references scripts (`03_make_features.py`, `04_run_drift.py`, `05_report.py`) that no longer exist. The current numbered scripts (`04_filter_and_preprocess_posts.py` etc.) are the active pipeline.
