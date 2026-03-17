#!/usr/bin/env bash
# run_submission.sh — Submission pipeline run (steps 01–09)
# Outputs go to data/submission/ and uses only the original 10-11 features.
# Usage: bash run_submission.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/submission.yaml"

# Activate Python 3.12 venv (required for spaCy)
source .venv312/bin/activate
export PYTHONPATH="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Pre-create all output directories
# ---------------------------------------------------------------------------
mkdir -p data/submission/interim/pmdd_analysis
mkdir -p data/submission/processed
mkdir -p data/submission/features
mkdir -p data/submission/models
mkdir -p data/submission/reports

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
step "07a" "FFT periodicity analysis — WITH anchor posts"
python scripts/07_run_fft_analysis.py \
    --config "$CONFIG" \
    --method fft_interpolation \
    --snr-threshold 3 \
    --force-recompute \
    --no-checkpoint

step "07b" "FFT periodicity analysis — WITHOUT anchor posts"
python scripts/07_run_fft_analysis.py \
    --config "$CONFIG" \
    --method fft_interpolation \
    --snr-threshold 3 \
    --no-anchors \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "08a-with" "Consensus period assignment — WITH anchor posts (original features, min-features: 5)"
python scripts/08_consensus.py \
    --config "$CONFIG" \
    --min-features 5 \
    --original-features-only \
    --force-recompute \
    --no-checkpoint

step "08a-no" "Consensus period assignment — WITHOUT anchor posts (original features, min-features: 5)"
python scripts/08_consensus.py \
    --config "$CONFIG" \
    --min-features 5 \
    --original-features-only \
    --no-anchors \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "08b" "PMDD phase analysis"
python scripts/08_pmdd.py --config "$CONFIG"

# ---------------------------------------------------------------------------
step "09a-with" "Visualize cycle length distributions — WITH anchor posts"
python scripts/09_visualize_distributions.py \
    --config "$CONFIG" \
    --original-features-only \
    --force-recompute \
    --no-checkpoint

step "09a-no" "Visualize cycle length distributions — WITHOUT anchor posts"
python scripts/09_visualize_distributions.py \
    --config "$CONFIG" \
    --original-features-only \
    --no-anchors \
    --force-recompute \
    --no-checkpoint

# ---------------------------------------------------------------------------
step "09b-with" "Visualize feature values by phase — WITH anchor posts"
python scripts/09_visualize_phase_features.py \
    --config "$CONFIG" \
    --original-features-only

step "09b-no" "Visualize feature values by phase — WITHOUT anchor posts"
python scripts/09_visualize_phase_features.py \
    --config "$CONFIG" \
    --original-features-only \
    --no-anchors

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [$(date '+%Y-%m-%d %H:%M:%S')]  SUBMISSION PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Submission outputs (data/submission/):"
echo "  data/submission/processed/users_database_CD_*.csv"
echo "  data/submission/interim/posts_all_users_preprocessed_with_anchors_*.csv"
echo "  data/submission/interim/timeline_with_offsets_with_anchors_*.csv"
echo "  data/submission/interim/timeline_daily_aggregated_with_anchors_*.csv"
echo "  data/submission/interim/periodicity_results_*.csv"
echo "  data/submission/interim/consensus_periods_min5features_*.csv"
echo "  data/submission/interim/pmdd_analysis/pmdd_phase_statistics_*.csv"
echo "  data/submission/reports/cycle_distributions_*.png"
echo "  data/submission/reports/feature_values_by_phase_*.png"
