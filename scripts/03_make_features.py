from pathlib import Path
from src.your_project.utils.config import load_config, get_logger

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    log = get_logger("features")
    features_dir = Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)
    log.info("Features placeholder: compute features from processed data and save to %s", features_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
