PYTHON ?= python3

.PHONY: install test lint train drift retrain api docker-build compose-up compose-down compose-logs k8s-build k8s-deploy k8s-status k8s-port-forward k8s-delete helm-template helm-deploy

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src api tests workflows scripts

train:
	$(PYTHON) scripts/run_training.py

drift:
	$(PYTHON) scripts/generate_drift_report.py

retrain:
	PREFECT_HOME=.prefect MPLCONFIGDIR=.matplotlib $(PYTHON) scripts/run_retraining_flow.py

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

k8s-build:
	docker build -f deployment/docker/Dockerfile.api -t readmission-api:latest .

k8s-deploy:
	kubectl apply -f deployment/kubernetes

k8s-status:
	kubectl get all -n readmission-mlops

k8s-port-forward:
	kubectl port-forward -n readmission-mlops svc/readmission-api 8000:80

k8s-delete:
	kubectl delete -f deployment/kubernetes

helm-template:
	helm template readmission-api deployment/helm/readmission-api --namespace readmission-mlops

helm-deploy:
	helm upgrade --install readmission-api deployment/helm/readmission-api --namespace readmission-mlops --create-namespace
