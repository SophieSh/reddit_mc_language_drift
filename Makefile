.PHONY: all ingest preprocess features drift report fmt test

all: ingest preprocess features drift report

ingest:
	python scripts/01_ingest.py --config configs/base.yaml
preprocess:
	python scripts/02_preprocess.py --config configs/base.yaml
features:
	python scripts/03_make_features.py --config configs/base.yaml
drift:
	python scripts/04_run_drift.py --config configs/base.yaml
report:
	python scripts/05_report.py --config configs/base.yaml

fmt:
	ruff check --fix . || true
	python -m black . || true

test:
	pytest -q
