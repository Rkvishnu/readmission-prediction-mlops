PYTHON ?= python3

.PHONY: install test lint train api docker-build compose-up compose-down compose-logs

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src api tests workflows scripts

train:
	$(PYTHON) scripts/run_training.py

api:
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build api

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
