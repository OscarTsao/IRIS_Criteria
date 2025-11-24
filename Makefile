# ============================================================================
# DSM-5 NLI Binary Classification - Makefile
# ============================================================================
# Automation for common development and training tasks
# Usage: make <target>
# Example: make setup && make train
# ============================================================================

# Declare all targets as phony (not actual files)
.PHONY: help setup train hpo eval clean test lint format
.PHONY: train_bert train_roberta train_deberta train_modernbert train_mentalbert train_psychbert
.PHONY: hpo_bert hpo_roberta hpo_deberta hpo_modernbert hpo_mentalbert hpo_psychbert

# Default tracking backends (override via env if needed)
MLFLOW_URI ?= sqlite:///mlflow.db
OPTUNA_URI ?= sqlite:///optuna.db
PYTHON ?= python3
N_TRIALS ?= 500
EXTRA_ARGS ?=

# ============================================================================
# HELP - Display available targets and their descriptions
# ============================================================================
help:
	@echo "DSM-5 NLI Criteria Matching - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  setup    - Install dependencies and setup environment"
	@echo "  train    - Run 5-fold cross-validation training"
	@echo "  hpo      - Run hyperparameter optimization (500 trials)"
	@echo "  eval     - Evaluate fold 0"
	@echo "  clean    - Clean outputs, cache, and logs"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linting checks"
	@echo "  format   - Format code with black"

# ============================================================================
# SETUP - Install dependencies in editable mode
# ============================================================================
# Upgrades pip and installs the package with development dependencies
# Run this once after cloning the repository
# Creates: .venv/lib/python3.10/site-packages/Project.egg-link
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev]'  # Editable install with dev dependencies (pytest, ruff, black)
	@echo "✓ Setup complete!"

# ============================================================================
# TRAIN - Run full K-fold cross-validation training
# ============================================================================
# Trains 5 separate models (one per fold) with default hyperparameters
# Logs results to MLflow (mlruns/ directory)
# Runtime: ~30-60 minutes depending on GPU
# Output: mlruns/, outputs/dsm5_criteria_matching/checkpoints/
train:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train $(EXTRA_ARGS)

train_bert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train model=bert_base $(EXTRA_ARGS)

train_roberta:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train model=roberta $(EXTRA_ARGS)

train_deberta:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train model=deberta_v3 $(EXTRA_ARGS)

train_modernbert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train model=modernbert $(EXTRA_ARGS)

train_mentalbert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train model=mentalbert $(EXTRA_ARGS)

train_psychbert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=train model=psychbert $(EXTRA_ARGS)

# ============================================================================
# HPO - Run hyperparameter optimization with Optuna
# ============================================================================
# Searches 500 hyperparameter combinations using Optuna
# Each trial runs full 100-epoch K-fold CV with patience 20
# Results stored in SQLite (Optuna) and MLflow by default
# Override n_trials or storage URIs via env if needed
hpo:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) $(EXTRA_ARGS)

hpo_bert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) model=bert_base $(EXTRA_ARGS)

hpo_roberta:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) model=roberta $(EXTRA_ARGS)

hpo_deberta:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) model=deberta_v3 $(EXTRA_ARGS)

hpo_modernbert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) model=modernbert $(EXTRA_ARGS)

hpo_mentalbert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) model=mentalbert $(EXTRA_ARGS)

hpo_psychbert:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m Project.cli command=hpo n_trials=$(N_TRIALS) model=psychbert $(EXTRA_ARGS)

# ============================================================================
# EVAL - Evaluate a specific fold (not yet implemented)
# ============================================================================
# Loads trained model from fold 0 and runs evaluation
# Displays per-criterion metrics and aggregate performance
eval:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m Project.cli command=eval fold=0 $(EXTRA_ARGS)

# ============================================================================
# CLEAN - Remove all generated files, outputs, and cache
# ============================================================================
# Deletes:
#   - outputs/ - Model checkpoints and training artifacts
#   - mlruns/ - MLflow experiment tracking data
#   - optuna.db - HPO trial history
#   - .pytest_cache/ - pytest cache
#   - __pycache__/ - Python bytecode cache (all directories)
#   - *.pyc - Compiled Python files
# WARNING: This is destructive. Trained models will be lost.
clean:
	rm -rf outputs/
	rm -rf mlruns/
	rm -rf optuna.db
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	# Find and remove all __pycache__ directories recursively (ignore errors)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	# Find and remove all .pyc files recursively
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned all outputs and cache"

# ============================================================================
# TEST - Run pytest test suite with coverage reporting
# ============================================================================
# Runs all tests in tests/ directory
# Generates HTML coverage report in htmlcov/
# Flags:
#   -v: Verbose output (show individual test results)
#   --cov: Measure code coverage for src/Project
#   --cov-report=html: Generate HTML coverage report
# Output: htmlcov/index.html (open in browser to view coverage)
test:
	pytest tests/ -v --cov=src/Project --cov-report=html

# ============================================================================
# LINT - Run code quality checks with ruff
# ============================================================================
# Checks for:
#   - PEP 8 style violations
#   - Common bugs and code smells
#   - Import sorting issues
#   - Unused imports and variables
# Does NOT modify files (use 'make format' to auto-fix)
# Exit code: 0 if clean, 1 if issues found
lint:
	ruff check src tests
	@echo "✓ Linting complete"

# ============================================================================
# FORMAT - Auto-format code with black and fix linting issues
# ============================================================================
# Steps:
#   1. black: Formats all Python files to consistent style (line length 100)
#   2. ruff --fix: Auto-fixes safe linting issues (imports, unused vars, etc.)
# Modifies files in-place
# Always run before committing code
format:
	black src tests                # Format with black (line length from pyproject.toml)
	ruff check --fix src tests     # Auto-fix safe linting issues
	@echo "✓ Code formatted"
