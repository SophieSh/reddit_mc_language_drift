from pathlib import Path
from src.your_project.utils.config import load_config, get_logger

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    log = get_logger("ingest")
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("Ingest placeholder: put your raw Reddit dumps into %s", raw_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
