PY=python

.PHONY: setup train api web

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements-core.txt

train:
	$(PY) scripts/train_baseline.py

api:
	uvicorn api.main:app --reload --port 8000

web:
	python -m http.server 5173 -d web
