# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IRIS (Interpretable Retrieval-Augmented Classification) implementation for DSM-5 criteria matching. This repository implements the ACL 2025 IRIS paper for binary classification of social media posts against DSM-5 Major Depressive Disorder diagnostic criteria.

**Architecture**: IRIS-only (retrieval-augmented model with learnable query vectors)
**Dataset**: 14,840 samples (1,484 posts × 10 criteria), 90.7% negative / 9.3% positive
**Stack**: PyTorch, Transformers, MLflow, Optuna, Hydra

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

# Train IRIS model (5-fold CV)
python -m criteria_bge_hpo.cli train

# Run HPO
python -m criteria_bge_hpo.cli hpo n_trials=100

# Alternative: use Makefile
make setup
make train
make hpo
```

## Development Commands

### Testing
```bash
# Run all tests (38 tests)
pytest

# Run specific test file
pytest tests/test_iris_model.py -v

# Run integration tests
python tests/test_integration.py

# With coverage report
pytest --cov=src/criteria_bge_hpo --cov-report=html
# Open htmlcov/index.html in browser
```

### Code Quality
```bash
# Lint (no modifications)
ruff check src tests

# Format and auto-fix
black src tests
ruff check --fix src tests

# Alternative: use Makefile
make lint
make format
```

### Clean Artifacts
```bash
make clean  # Removes outputs/, mlruns/, optuna.db, __pycache__/
```

## Configuration System (Hydra)

Main config: `configs/config.yaml`

### Config Structure
```
configs/
├── config.yaml           # Main: experiment, MLflow, paths, device
├── model/
│   └── iris.yaml        # IRIS architecture (queries, retrieval, encoder)
├── training/
│   └── default.yaml     # Optimizer, scheduler, loss, AMP, early stopping
└── hpo/
    └── optuna.yaml      # HPO search space and pruning settings
```

### Command-Line Overrides
```bash
# Override individual parameters
python -m criteria_bge_hpo.cli train \
  model.num_queries=12 \
  model.k_retrieved=16 \
  training.optimizer.lr=1e-3 \
  training.loss.type=weighted_bce \
  training.num_epochs=50

# Use different config file
python -m criteria_bge_hpo.cli train --config-name my_config

# HPO with custom trials
python -m criteria_bge_hpo.cli hpo n_trials=200
```

## Architecture Overview

### Pipeline Flow (cli.py orchestrates)

1. **Data Loading** (`data/preprocessing.py`)
   - Loads `data/groundtruth/criteria_matching_groundtruth.csv` (posts + labels)
   - Loads `data/DSM5/MDD_Criteria.json` (criterion definitions)
   - Returns pandas DataFrame with post, DSM5_symptom, groundtruth columns

2. **K-Fold Splits** (`training/kfold.py`)
   - Creates 5-fold CV with `group_column="post_id"` (prevents leakage)
   - Single post may have 10 samples (one per criterion)
   - Ensures post appears in only one fold

3. **Dataset** (`data/dataset.py`)
   - `CriterionMatchingDataset`: Pairs (post, criterion) for binary classification
   - Returns raw text (IRIS doesn't need tokenization)
   - IRIS model handles embedding internally

4. **IRIS Model** (`models/iris_model.py`)
   - `IRISForCriterionMatching`: Complete IRIS architecture
   - `build_retriever()`: Pre-encodes all posts, builds FAISS index (one-time)
   - Forward: Concatenates post+criterion → retrieves k chunks per query → attention → classify

5. **Training** (`training/trainer.py`)
   - Unified trainer supporting IRIS and generic models
   - Gradient accumulation, mixed precision (bf16/fp16), early stopping
   - Automatic model-type detection (IRIS vs token-based)

6. **Loss Functions** (`training/losses.py`)
   - FocalLoss (γ=2.0, α auto-computed from class distribution)
   - WeightedBCELoss (pos_weight=9.82 for 90/10 imbalance)
   - Standard BCE

7. **Evaluation** (`evaluation/evaluator.py`)
   - Binary metrics: F1, precision, recall, sensitivity, AUC-ROC, AUC-PR
   - Per-criterion analysis: `PerCriterionEvaluator`
   - IRIS interpretability: retrieved chunks per query

8. **MLflow Logging** (`utils/mlflow_setup.py`)
   - Each fold = separate MLflow run
   - Logs hyperparameters, metrics per epoch, final summary

### IRIS Components

**Query Vectors** (`models/query_attention.py`):
- 8 learnable 768-dim vectors (initialized random, L2-normalized)
- Each query specializes to different symptom aspects
- Query penalty loss prevents collapse (λ=0.1, threshold=0.4)

**Retrieval** (`models/retrieval.py`):
- FAISS GPU index for fast k-NN search
- Retrieves top-k=12 chunks per query
- Returns chunk embeddings + indices

**Attention** (`models/query_attention.py`):
- Linear attention: `softmax((Q @ K^T) / T)` where T=0.1
- Aggregates retrieved chunks weighted by similarity
- Output: 8 × 768 query-aggregated representations

**Classification** (`models/classifier_heads.py`):
- 7 head variants: linear, pooler_linear, mlp1, mlp2, mean_pooling, max_pooling, attention_pooling
- Input: Concatenation of query outputs (8×768 → 6144-dim)
- Output: Binary logit

## Key Implementation Details

### IRIS-Only Enforcement
The CLI enforces IRIS-only usage:
```python
# cli.py lines 119-121
if cfg.model.model_type != "iris":
    raise ValueError(f"Only IRIS model_type is supported, got: {cfg.model.model_type}")
```

To add other models, modify:
1. `cli.py`: Remove ValueError checks
2. `hpo/optuna_search.py`: Add search space for new model
3. `models/__init__.py`: Import and export new model class

### Data Leakage Prevention
K-fold uses `group_column="post_id"`:
```python
# Correct: Same post never in train AND val
splits = create_kfold_splits(df, n_folds=5, group_column="post_id")

# Wrong: Would leak (same post in train and val)
splits = create_kfold_splits(df, n_folds=5, group_column=None)  # DON'T DO THIS
```

### Class Imbalance Handling
90.7% negative, 9.3% positive → Focal Loss recommended:
```python
# Auto-computed from training labels
loss_fn = create_loss_function("focal", labels=train_labels)
# Sets alpha=0.093, gamma=2.0

# Alternative: Weighted BCE
loss_fn = create_loss_function("weighted_bce", labels=train_labels)
# Sets pos_weight=9.82
```

### IRIS Retriever Build
**Must call `build_retriever()` before training**:
```python
model = IRISForCriterionMatching(...)

# Build retriever (encode all posts, create FAISS index)
all_posts = df["post"].unique().tolist()
model.build_retriever(all_posts, batch_size=32, use_gpu=True)

# Now ready to train
trainer = Trainer(model, ...)
trainer.train(...)
```

### Mixed Precision Training
```yaml
# configs/training/default.yaml
use_amp: true
amp_dtype: "bfloat16"  # Requires Ampere+ GPU (A100, RTX 3090+)
# Use "float16" for older GPUs (V100, RTX 2080)
```

## HPO Configuration

### Search Space (IRIS)
Defined in `hpo/optuna_search.py`:
```python
ModelSearchSpace.iris_search_space(trial):
    num_queries: [4, 8, 12, 16]
    k_retrieved: [8, 12, 16, 20]
    temperature: [0.05, 0.2]
    query_penalty_lambda: [0.05, 0.2]
    learning_rate: [1e-4, 1e-2] (log scale)
    weight_decay: [1e-5, 1e-3] (log scale)
    batch_size: [8, 16, 32]
    num_epochs: [30, 50, 70, 100]
    loss_type: ["focal", "weighted_bce"]
    focal_gamma: [1.0, 3.0] (if focal loss)
```

### Pruning Strategy
```python
# HPORunner defaults (cli.py line 339-351)
pruning=True
n_warmup_steps=0  # Aggressive fold-1 pruning (80% compute savings)
# Uses MedianPruner: prunes if trial worse than median of completed trials
```

### Running HPO
```bash
# Default: 100 trials, MedianPruner, maximize val_acc
python -m criteria_bge_hpo.cli hpo n_trials=100

# Override metric and direction
python -m criteria_bge_hpo.cli hpo \
  hpo.metric_name=val_f1 \
  hpo.direction=maximize \
  n_trials=200

# Disable pruning (for debugging)
python -m criteria_bge_hpo.cli hpo hpo.pruning=false
```

## MLflow Experiment Tracking

### Structure
```
mlruns/
└── <experiment_id>/
    ├── <run_id_fold0>/
    │   ├── params/  (hyperparameters)
    │   ├── metrics/ (train_loss, val_loss, val_acc per epoch)
    │   └── artifacts/
    ├── <run_id_fold1>/
    └── <run_id_summary>/  (aggregate CV results)
```

### Viewing Results
```bash
mlflow ui
# Open http://localhost:5000
```

### Logging Pattern
Each fold is a separate run:
```python
# cli.py lines 192-255
for fold_idx, (train_idx, val_idx) in enumerate(splits):
    mlflow_logger = MLflowLogger(
        experiment_name="criteria_matching",
        run_name=f"fold{fold_idx}",
        tags={"fold": str(fold_idx), "model_type": "iris"}
    )

    # Log hyperparameters
    mlflow_logger.log_params({"learning_rate": 1e-3, ...})

    # Train (metrics logged automatically by trainer)
    history = trainer.train(...)

    # Log summary metrics
    mlflow_logger.log_metric("best_val_loss", min(history["val_loss"]))
    mlflow_logger.end_run()

# After all folds, log aggregate
with MLflowLogger(..., run_name="summary") as summary_logger:
    summary_logger.log_metric("cv_val_acc_mean", mean_acc)
    summary_logger.log_metric("cv_val_acc_std", std_acc)
```

## Evaluation Metrics

### Binary Classification Metrics
```python
from criteria_bge_hpo.evaluation import BinaryClassificationMetrics

metrics = BinaryClassificationMetrics.compute_all_metrics(
    y_true, y_pred, y_prob
)
# Returns: accuracy, precision, recall, f1, macro_f1,
#          sensitivity, specificity, auc_roc, auc_pr, brier
```

### Per-Criterion Analysis
```python
from criteria_bge_hpo.evaluation import PerCriterionEvaluator

evaluator = PerCriterionEvaluator(criterion_names=["A.1", "A.2", ...])

# Update per batch
for batch in dataloader:
    evaluator.update(
        criterion_ids=batch["criterion"],
        y_pred=predictions,
        y_true=labels,
        y_prob=probabilities
    )

# Get per-criterion metrics
df = evaluator.compute_metrics()
# Returns DataFrame with accuracy, f1, sensitivity per criterion
```

### IRIS Interpretability
```python
from criteria_bge_hpo.evaluation import IRISInterpretabilityAnalyzer

analyzer = IRISInterpretabilityAnalyzer(iris_model)

# Get retrieved chunks for a sample
retrieved = analyzer.get_retrieved_chunks_for_sample(
    post="I feel very sad and hopeless...",
    criterion="Depressed mood most of the day"
)
# Returns: {query_idx: [chunk1, chunk2, ...]} for each of 8 queries
```

## Common Patterns

### Adding New Loss Function
```python
# 1. Implement in training/losses.py
class MyCustomLoss(torch.nn.Module):
    def forward(self, logits, labels):
        ...

# 2. Add to create_loss_function()
def create_loss_function(loss_type, labels, **kwargs):
    if loss_type == "my_custom":
        return MyCustomLoss(**kwargs)
    ...

# 3. Use in config
training.loss.type=my_custom
```

### Adding New Classifier Head
```python
# 1. Implement in models/classifier_heads.py
class MyHead(torch.nn.Module):
    ...

# 2. Register in ClassifierHeadFactory
@staticmethod
def create(...):
    if head_type == "my_head":
        return MyHead(...)
    ...

# 3. Use in config or HPO search space
model.classifier_head=my_head
```

### Debugging Training Issues
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data leakage
from criteria_bge_hpo.training import create_kfold_splits
splits = create_kfold_splits(df, n_folds=5, group_column="post_id")
for fold, (train_idx, val_idx) in enumerate(splits):
    train_posts = set(df.iloc[train_idx]["post_id"].unique())
    val_posts = set(df.iloc[val_idx]["post_id"].unique())
    overlap = train_posts & val_posts
    assert len(overlap) == 0, f"Fold {fold} has post overlap: {overlap}"

# Verify IRIS retriever is built
assert model.retriever is not None, "Call model.build_retriever() first!"

# Check GPU usage
print(f"Model device: {next(model.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## File Locations

**Data**:
- `data/groundtruth/criteria_matching_groundtruth.csv` - Main dataset
- `data/DSM5/MDD_Criteria.json` - Criterion definitions

**Models**:
- Checkpoints: `checkpoints/<experiment_name>/fold_<i>/best.pt`
- MLflow artifacts: `mlruns/<experiment_id>/<run_id>/artifacts/`

**Outputs**:
- Training logs: `outputs/<experiment_name>/`
- HPO database: `optuna.db` (SQLite)

## Performance Expectations

| Model | Trainable Params | Train Time (5-fold) | Expected F1 |
|-------|------------------|---------------------|-------------|
| IRIS (frozen encoder) | ~10k | ~5 min (A100) | 0.70-0.75 |
| IRIS (with HPO) | Varies | ~2-4 hours | 0.75-0.85 |

**Loss function impact**:
- Standard BCE: F1 = 0.60-0.70 (poor on imbalanced data)
- Weighted BCE: F1 = 0.70-0.75
- Focal Loss: F1 = 0.75-0.85 (recommended for 90/10 imbalance)
