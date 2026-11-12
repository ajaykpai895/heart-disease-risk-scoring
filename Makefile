# -----------------------------------------------------------------------------
# Cardiovascular Risk Screening & Clinical Decision Support Makefile
# -----------------------------------------------------------------------------

.PHONY: help setup data train evaluate explain app api test lint metadata mlflow mlflow-ui docker-build docker-up clean all

PYTHON := python
STREAMLIT := streamlit
UVICORN := uvicorn
PYTEST := pytest
RUFF := ruff

help: ## Show this help message
	@echo "Cardiovascular Risk Screening Pipeline Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install pinned dependencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data: ## Download raw Cleveland dataset from UCI and validate schema
	$(PYTHON) -m src.data

train: ## Run full research-grade training, nested CV, calibration, and DCA
	$(PYTHON) -m src.train --target-recall 0.90 --seed 42

evaluate: ## Run fairness, subgroup calibration, and demographic audit
	$(PYTHON) -c "import pandas as pd, joblib, json; from src.fairness import run_fairness_audit; X=pd.read_csv('reports/X_test.csv'); y=pd.read_csv('reports/y_test.csv').iloc[:,0]; lr=joblib.load('models/logreg.joblib'); xgb=joblib.load('models/xgb.joblib'); thr=json.load(open('models/threshold.json')); run_fairness_audit(X, y, {'Logistic Regression': lr.predict_proba(X)[:,1], 'XGBoost': xgb.predict_proba(X)[:,1]}, {'Logistic Regression': thr['threshold'], 'XGBoost': thr['threshold_xgb']})"

explain: ## Generate SHAP summary plots and epidemiological Odds Ratios
	$(PYTHON) -m src.explain

app: ## Launch interactive Streamlit clinical decision support dashboard
	$(STREAMLIT) run app.py

api: ## Launch FastAPI production REST inference microservice
	$(UVICORN) src.api:app --host 0.0.0.0 --port 8000 --reload

test: ## Run complete unit and integration test suite
	$(PYTEST) tests/ -v

lint: ## Execute static analysis and style formatting checks
	$(RUFF) check src/ tests/

metadata: ## Regenerate production model provenance and data hash metadata
	$(PYTHON) -m src.metadata

mlflow: ## Log model runs and metrics to local MLflow SQLite store
	$(PYTHON) -m src.mlflow_utils

mlflow-ui: ## Launch interactive MLflow dashboard on port 5000
	mlflow ui --port 5000

docker-build: ## Build production Docker container image
	docker build -t cardioscreen-ai:latest .

docker-up: ## Launch Streamlit, FastAPI, and MLflow using Docker Compose
	docker compose up -d

clean: ## Clean cached bytecode and temporary runtime artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: lint test metadata ## Run linter, test suite, and metadata generation
