#!/usr/bin/env bash
# run_pipeline.sh — Full pipeline rerun (steps 01–09)
# Usage: bash run_pipeline.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/base.yaml"

# Activate Python 3.12 venv (required for spaCy)
source .venv312/bin/activate
export PYTHONPATH="$SCRIPT_DIR"

step() {
    echo ""
    echo "============================================================"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')]  STEP $1: $2"
    echo "============================================================"
}

# ---------------------------------------------------------------------------
step "01" "Ingest — validate raw feature files"
python scripts/01_ingest.py --config "$CONFIG"

# ---------------------------------------------------------------------------
step "02a" "Preprocess moon1 anchors"
python scripts/02_preprocess_moon1.py --config "$CONFIG"

step "02b" "Preprocess moon2 anchors"
python scripts/02_preprocess_moon2.py --config "$CONFIG"

# step "02c" "Preprocess moon3 anchors"  # skipped — moon3 not used
# python scripts/02_preprocess_moon3.py --config "$CONFIG"

# ---------------------------------------------------------------------------
step "03" "Create users database (CD + DPO)"
python scripts/03_create_users_database.py --config "$CONFIG"

# ---------------------------------------------------------------------------
step "04" "Filter and preprocess posts"
python scripts/04_filter_and_preprocess_posts.py \
    --config "$CONFIG" \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "05" "Build timeline (window: 3 months)"
python scripts/05_build_timeline.py \
    --config "$CONFIG" \
    --window-months 3 \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "06" "Aggregate and normalize"
python scripts/06_aggregate_and_normalize.py \
    --config "$CONFIG" \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "07" "FFT periodicity analysis (method: fft_interpolation, SNR threshold: 3)"
python scripts/07_run_fft_analysis.py \
    --config "$CONFIG" \
    --method fft_interpolation \
    --snr-threshold 3 \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "08a" "Consensus period assignment (min-features: 23)"
python scripts/08_consensus.py \
    --config "$CONFIG" \
    --min-features 23 \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "08b" "PMDD phase analysis"
python scripts/08_pmdd.py --config "$CONFIG"

# ---------------------------------------------------------------------------
step "09" "Visualize cycle length distributions"
python scripts/09_visualize_distributions.py \
    --config "$CONFIG" \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "09b" "Visualize feature values by menstrual phase"
python scripts/09_visualize_phase_features.py --config "$CONFIG"

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [$(date '+%Y-%m-%d %H:%M:%S')]  PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Expected outputs (check for fresh timestamps):"
echo "  data/processed/users_database_CD_*.csv"
echo "  data/interim/posts_all_users_preprocessed_with_anchors_*.csv"
echo "  data/interim/timeline_with_offsets_with_anchors_*.csv"
echo "  data/interim/timeline_daily_aggregated_with_anchors_*.csv"
echo "  data/interim/periodicity_results_*.csv"
echo "  data/interim/consensus_periods_*.csv"
echo "  data/interim/pmdd_analysis/pmdd_phase_statistics_*.csv"
echo "  reports/cycle_distributions_*.png"
echo "  reports/feature_values_by_phase_*.png"
