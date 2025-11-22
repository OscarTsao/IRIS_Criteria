.PHONY: help setup train hpo eval clean test lint format

help:
	@echo "DSM-5 NLI Criteria Matching - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  setup    - Install dependencies and setup environment"
	@echo "  train    - Run 5-fold cross-validation training"
	@echo "  hpo      - Run hyperparameter optimization (50 trials)"
	@echo "  eval     - Evaluate fold 0"
	@echo "  clean    - Clean outputs, cache, and logs"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linting checks"
	@echo "  format   - Format code with black"

setup:
	python -m pip install --upgrade pip
	pip install -e '.[dev]'
	@echo "✓ Setup complete!"

train:
	python -m dsm5_nli.cli command=train

hpo:
	python -m dsm5_nli.cli command=hpo n_trials=50

eval:
	python -m dsm5_nli.cli command=eval fold=0

clean:
	rm -rf outputs/
	rm -rf mlruns/
	rm -rf optuna.db
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned all outputs and cache"

test:
	pytest tests/ -v --cov=src/dsm5_nli --cov-report=html

lint:
	ruff check src tests
	@echo "✓ Linting complete"

format:
	black src tests
	ruff check --fix src tests
	@echo "✓ Code formatted"
