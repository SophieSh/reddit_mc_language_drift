from pathlib import Path
from src.your_project.utils.config import load_config, get_logger

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    log = get_logger("drift")
    models_dir = Path(cfg["paths"]["models"])
    figs_dir = Path(cfg["paths"]["reports"]) / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    log.info("Drift placeholder: run spectral/drift tests, save metrics to %s and plots to %s", models_dir, figs_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
