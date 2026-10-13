# MCPGuard — common tasks. On Windows use Git Bash / WSL, or run the python commands directly.
PY ?= python

.PHONY: install data train optimize evaluate explain figures report pipeline fast run test docker docker-run mlflow

install:
	$(PY) -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data.download
	$(PY) -m src.data.build_dataset
	$(PY) -m src.data.preprocess
	$(PY) -m src.features.build

optimize:
	$(PY) -m src.training.optimize

train:
	$(PY) -m src.training.train

explain:
	$(PY) -m src.explainability.generate

evaluate:
	$(PY) -m src.evaluation.evaluate

figures:
	$(PY) -m src.visualization.figures

report:
	$(PY) -m src.reporting.report

pipeline:
	$(PY) -m src.pipeline.run_all

fast:
	$(PY) -m src.pipeline.run_all --skip-download --fast

run:
	$(PY) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 7860

test:
	$(PY) -m pytest -q

docker:
	docker compose build

docker-run:
	docker compose up --build

mlflow:
	mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
