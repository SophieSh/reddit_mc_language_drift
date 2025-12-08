## Project Overview

This project explores **language drift across the menstrual cycle** using Reddit data.
By anchoring posts that mark cycle events (e.g., "I got my period today"),
we can track how linguistic patterns (sentiment, lexical richness, topic usage)
vary cyclically over time.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# run full pipeline
make all          # or: python run_pipeline.py
```

See `docs/drift_pipeline.md` for an overview of the stages.
