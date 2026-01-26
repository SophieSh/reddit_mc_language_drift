from __future__ import annotations
import yaml


def load_config(path: str = "configs/base.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        path: Path to config YAML file (default: configs/base.yaml)
    
    Returns:
        Configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Load config once at module level
_cfg = load_config()

# Analysis constants (from analysis section)
MIN_DATA_POINTS = _cfg.get("analysis", {}).get("min_data_points", 10)
EPSILON = float(_cfg.get("analysis", {}).get("epsilon", 1e-10))  # Ensure float type
DEFAULT_PERIOD_MIN = _cfg.get("analysis", {}).get("period_min", 21)
DEFAULT_PERIOD_MAX = _cfg.get("analysis", {}).get("period_max", 35)

# Pipeline constants (from pipeline section)
AVG_DAYS_PER_MONTH = _cfg.get("pipeline", {}).get("avg_days_per_month", 30.5)

