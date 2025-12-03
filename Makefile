# ============================================================================
# DSM-5 NLI Binary Classification - Makefile
# ============================================================================
# Automation for common development and training tasks
# Usage: make <target>
# Example: make setup && make train
# ============================================================================

# Declare all targets as phony (not actual files)
.PHONY: help setup train hpo eval clean test lint format

# Default tracking backends (override via env if needed)
MLFLOW_URI ?= file:mlruns
OPTUNA_URI ?= sqlite:///optuna.db
PYTHON ?= python3
N_TRIALS ?= 100
N_OUTER_FOLDS ?= 5
N_INNER_FOLDS ?= 3
EXTRA_ARGS ?=
# Ensure this repo's src/ is first on PYTHONPATH (avoids picking up other editable installs)
PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH),)
export PYTHONPATH

# ============================================================================
# HELP - Display available targets and their descriptions
# ============================================================================
help:
	@echo "IRIS Criteria Matching - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and setup environment"
	@echo "  train         - Run 5-fold cross-validation training (100 epochs, patience 20)"
	@echo "  hpo           - Run hyperparameter optimization (100 trials, default)"
	@echo "  eval          - Evaluate fold 0 (not yet implemented - use MLflow UI)"
	@echo "  clean         - Clean outputs, cache, and logs"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black"
	@echo ""
	@echo "Environment variables:"
	@echo "  N_TRIALS      - Number of HPO trials (default: 100)"
	@echo "  EXTRA_ARGS    - Additional Hydra arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make train EXTRA_ARGS='training.num_epochs=50'"
	@echo "  make hpo N_TRIALS=200"

# ============================================================================
# SETUP - Install dependencies in editable mode
# ============================================================================
# Upgrades pip and installs the package with development dependencies
# Run this once after cloning the repository
# Creates: .venv/lib/python3.10/site-packages/criteria_bge_hpo.egg-link
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev]'  # Editable install with dev dependencies (pytest, ruff, black)
	@echo "✓ Setup complete!"

# ============================================================================
# TRAIN - Run full K-fold cross-validation training
# ============================================================================
# Trains 5 separate models (one per fold) with default hyperparameters
# Optimizations: BF16 mixed precision, pin_memory, persistent workers
# Logs results to MLflow (mlruns/ directory)
# Runtime: ~5 min per fold on A100 (25 min total for 5-fold CV)
# Output: mlruns/, checkpoints/criteria_matching/fold_*/
train:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m criteria_bge_hpo.cli train training.num_epochs=100 training.early_stopping_patience=20 $(EXTRA_ARGS)

# ============================================================================
# HPO - Run hyperparameter optimization with Optuna
# ============================================================================
# Optimizes IRIS hyperparameters: num_queries, k_retrieved, temperature, etc.
# Defaults to 100 trials with MedianPruner for early stopping
# Optimizations: BF16 mixed precision, fold-1 pruning
# Results stored in optuna.db (SQLite) and MLflow
# Pass EXTRA_ARGS for overrides (e.g., hpo.n_warmup_steps=5)
# Runtime: ~2-4 hours for 100 trials with pruning (A100)
hpo:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli hpo n_trials=$(N_TRIALS) training.num_epochs=100 training.early_stopping_patience=20 $(EXTRA_ARGS)

# ============================================================================
# EVAL - Evaluate a specific fold
# ============================================================================
# NOTE: Evaluation command not yet implemented in CLI
# Workaround: Results are logged to MLflow during training
# View with: mlflow ui
eval:
	@echo "⚠️  Evaluation command not yet implemented"
	@echo "Workaround: View training results with 'mlflow ui'"
	@echo "Or implement evaluation in src/criteria_bge_hpo/cli.py:evaluate()"

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
#   --cov: Measure code coverage for src/criteria_bge_hpo
#   --cov-report=html: Generate HTML coverage report
# Output: htmlcov/index.html (open in browser to view coverage)
test:
	pytest tests/ -v --cov=src/criteria_bge_hpo --cov-report=html

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
