"""Shared constants for the menstrual-cycle NLP pipeline.

All values that are duplicated across scripts must be defined here and imported
from here.  Do not redefine them locally in scripts.
"""

# Canonical phase ordering used in ML models, plots, and phase-label filtering.
# LabelEncoder sorts alphabetically, so the integer mapping is:
#   Follicular=0, Luteal=1, Menstrual=2, Ovulation=3
PHASE_ORDER: list[str] = ["Menstrual", "Follicular", "Ovulation", "Luteal"]

# Phase colours — consistent across all visualisations.
PHASE_COLORS: dict[str, str] = {
    "Menstrual":  "#d62728",   # red
    "Follicular": "#2ca02c",   # green
    "Ovulation":  "#ff7f0e",   # orange
    "Luteal":     "#1f77b4",   # blue
}

# Features used in the original biologically-aligned publication analysis.
ORIGINAL_FEATURES: list[str] = [
    "negative_sentiment",
    "positive_sentiment",
    "num_words",
    "avg_word_length",
    "num_sentences",
    "unique_word_fraction",
    "readability",
    "spelling_errors_frac",
    "syntactic_complexity_subordination_index",
    "cohesion_analysis_lexical_overlap",
]

# Classic z-score tail thresholds (theoretical 10th / 90th percentile of a
# standard normal).  Used in the OvR statistical gauntlet to test whether
# phase users are enriched at the tails of each feature distribution.
CLASSIC_Z_HIGH: float =  1.28   # 90th percentile
CLASSIC_Z_LOW:  float = -1.28   # 10th percentile
