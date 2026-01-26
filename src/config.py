from __future__ import annotations
import yaml
import logging
import subprocess
import pathlib
import os

# Analysis constants
MIN_DATA_POINTS = 10
EPSILON = 1e-10
DEFAULT_PERIOD_MIN = 24
DEFAULT_PERIOD_MAX = 35
AVG_DAYS_PER_MONTH = 30.5  # For converting days to months

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

