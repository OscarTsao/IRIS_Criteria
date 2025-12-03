"""Integration tests for end-to-end workflows.

Tests:
- Complete training pipeline (data → model → training → evaluation)
- Config loading and validation
- MLflow integration
- K-fold cross-validation workflow
- Model checkpointing and resumption
"""

import sys
from pathlib import Path
import tempfile
import shutil

import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_dummy_dataset(num_samples=50):
    """Create a small dummy dataset for testing."""
    data = {
        "post_id": [f"post_{i//10}" for i in range(num_samples)],
        "post": [f"Test post content {i}. This is sample text." for i in range(num_samples)],
        "DSM5_symptom": [f"A.{(i % 9) + 1}" for i in range(num_samples)],
        "groundtruth": [1 if i % 5 == 0 else 0 for i in range(num_samples)],
    }
    return pd.DataFrame(data)


def create_dummy_criteria():
    """Create dummy DSM-5 criteria."""
    return {f"A.{i}": f"Criterion A.{i} description" for i in range(1, 10)}


def test_end_to_end_training_simple_model():
    """Test complete training pipeline with a simple token-based model."""
    from criteria_bge_hpo.data import CriterionMatchingDataset
    from criteria_bge_hpo.training import Trainer, create_loss_function
    from torch.utils.data import DataLoader

    print("\n1. Testing end-to-end training (simple model)...")

    class DummyTokenizer:
        def __call__(
            self,
            text_a,
            text_b=None,
            max_length: int = 128,
            padding: str = "max_length",
            truncation: bool = True,
            return_tensors: str = "pt",
        ):
            if isinstance(text_a, list):
                batch_size = len(text_a)
            else:
                batch_size = 1
            input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim: int = 128):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, 1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            # Treat token positions as features
            features = input_ids.float()
            return self.linear(features).squeeze(-1)

    # Create dummy data
    df = create_dummy_dataset(num_samples=40)
    criteria = create_dummy_criteria()

    # Split data
    train_df = df.iloc[:30]
    val_df = df.iloc[30:]

    # Create tokenizer
    tokenizer = DummyTokenizer()

    # Create datasets
    train_dataset = CriterionMatchingDataset(
        train_df, criteria, tokenizer=tokenizer, max_length=128
    )
    val_dataset = CriterionMatchingDataset(
        val_df, criteria, tokenizer=tokenizer, max_length=128
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create model
    model = SimpleModel(input_dim=128)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create loss
    labels = torch.tensor(train_df["groundtruth"].values)
    loss_fn = create_loss_function("bce", labels)

    # Create temporary checkpoint dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            gradient_accumulation_steps=1,
            use_amp=False,
            early_stopping_patience=3,
            checkpoint_dir=tmpdir,
        )

        # Train for 2 epochs
        print("   Training for 2 epochs...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
        )

        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"   Final val acc: {history['val_acc'][-1]:.4f}")

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(isinstance(x, float) for x in history["train_loss"])
        assert 0 <= history["val_acc"][-1] <= 1

        # Check checkpoint was saved
        checkpoint_files = list(Path(tmpdir).glob("*.pt"))
        assert len(checkpoint_files) > 0, "No checkpoint saved"
        print(f"   Checkpoint saved: {checkpoint_files[0].name}")

    print("   ✓ End-to-end training (simple model) passed")


def test_kfold_integration():
    """Test K-fold cross-validation workflow."""
    from criteria_bge_hpo.data import CriterionMatchingDataset
    from criteria_bge_hpo.training import create_kfold_splits

    print("\n2. Testing K-fold integration...")

    # Create dummy data
    df = create_dummy_dataset(num_samples=50)
    criteria = create_dummy_criteria()

    # Create K-fold splits
    splits = create_kfold_splits(df, n_folds=3, group_column="post_id")

    print(f"   Number of folds: {len(splits)}")
    assert len(splits) == 3

    class DummyTokenizer:
        def __call__(
            self,
            text_a,
            text_b=None,
            max_length: int = 128,
            padding: str = "max_length",
            truncation: bool = True,
            return_tensors: str = "pt",
        ):
            if isinstance(text_a, list):
                batch_size = len(text_a)
            else:
                batch_size = 1
            input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer = DummyTokenizer()

    # Test each fold
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Create datasets
        train_dataset = CriterionMatchingDataset(
            train_df, criteria, tokenizer=tokenizer, max_length=128
        )
        val_dataset = CriterionMatchingDataset(
            val_df, criteria, tokenizer=tokenizer, max_length=128
        )

        print(f"   Fold {fold_idx}: train={len(train_dataset)}, val={len(val_dataset)}")

        assert len(train_dataset) > 0
        assert len(val_dataset) > 0

        # Verify no post_id overlap
        train_posts = set(train_df["post_id"].unique())
        val_posts = set(val_df["post_id"].unique())
        overlap = train_posts & val_posts
        assert len(overlap) == 0, f"Post overlap detected: {overlap}"

    print("   ✓ K-fold integration passed")


def test_mlflow_logging():
    """Test MLflow logging integration."""
    from criteria_bge_hpo.utils import setup_mlflow, MLflowLogger

    print("\n3. Testing MLflow logging...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup MLflow with temporary directory
        tracking_uri = f"file:{tmpdir}/mlruns"
        experiment_id = setup_mlflow(
            experiment_name="test_experiment",
            tracking_uri=tracking_uri,
        )

        print(f"   Experiment ID: {experiment_id}")
        assert experiment_id is not None

        # Create logger
        with MLflowLogger(
            experiment_name="test_experiment",
            run_name="test_run",
            tags={"test": "true"},
            tracking_uri=tracking_uri,
        ) as logger:
            # Log params
            logger.log_params({
                "learning_rate": 1e-4,
                "batch_size": 8,
            })

            # Log metrics
            logger.log_metrics({
                "train_loss": 0.5,
                "val_loss": 0.6,
            }, step=0)

            logger.log_metrics({
                "train_loss": 0.4,
                "val_loss": 0.55,
            }, step=1)

            print("   Params and metrics logged successfully")

        # Verify run was created
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) > 0, "No runs found"
        print(f"   Found {len(runs)} run(s)")

    print("   ✓ MLflow logging passed")


def test_loss_function_integration():
    """Test loss functions with real training."""
    from criteria_bge_hpo.training import create_loss_function, FocalLoss, WeightedBCELoss
    import torch

    print("\n4. Testing loss function integration...")

    # Create dummy labels
    labels = torch.randint(0, 2, (20,)).float()

    # Test each loss type
    loss_types = ["bce", "weighted_bce", "focal"]

    for loss_type in loss_types:
        # Create fresh logits with gradient tracking for each loss type
        logits = torch.randn(20, requires_grad=True)

        loss_fn = create_loss_function(loss_type, labels)
        loss = loss_fn(logits, labels)

        print(f"   {loss_type}: loss={loss.item():.4f}")

        assert torch.isfinite(loss), f"{loss_type} produced non-finite loss"
        assert loss.item() >= 0, f"{loss_type} produced negative loss"

        # Test backward pass
        loss.backward()

        # Verify gradients were computed
        assert logits.grad is not None, f"{loss_type} didn't compute gradients"

    print("   ✓ Loss function integration passed")


def test_config_loading():
    """Test Hydra config loading."""
    from omegaconf import OmegaConf

    print("\n5. Testing config loading...")

    # Test loading main config
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)

        print(f"   Loaded config: {config_path.name}")
        print(f"   Experiment name: {cfg.experiment.name}")
        print(f"   Device: {cfg.device}")

        assert "experiment" in cfg
        assert "mlflow" in cfg
        assert "paths" in cfg

        # Test model config
        model_config_path = Path(__file__).parent.parent / "configs" / "model" / "iris.yaml"
        if model_config_path.exists():
            model_cfg = OmegaConf.load(model_config_path)
            print(f"   Loaded: {model_config_path.name} (type: {model_cfg.model_type})")
            assert "model_type" in model_cfg

        print("   ✓ Config loading passed")
    else:
        print("   ⚠ Config files not found, skipping")


def test_evaluation_integration():
    """Test evaluation metrics integration."""
    from criteria_bge_hpo.evaluation import (
        BinaryClassificationMetrics,
        PerCriterionEvaluator,
        compute_aggregate_metrics,
    )

    print("\n6. Testing evaluation integration...")

    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)

    # Test metrics
    metrics = BinaryClassificationMetrics.compute_all_metrics(y_true, y_pred, y_prob)

    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1: {metrics['f1']:.4f}")
    print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "macro_f1" in metrics

    # Test per-criterion evaluator
    criteria = ["A.1", "A.2", "A.3"]
    evaluator = PerCriterionEvaluator(criteria)

    # Add predictions for each criterion
    for criterion in criteria:
        evaluator.update(
            criterion_ids=[criterion] * 10,
            y_pred=np.random.randint(0, 2, 10),
            y_true=np.random.randint(0, 2, 10),
            y_prob=np.random.rand(10),
        )

    df = evaluator.compute_metrics()
    print(f"   Per-criterion metrics: {len(df)} criteria")
    assert len(df) == 3

    # Test aggregate metrics
    fold_results = [
        [y_true[:20], y_pred[:20], y_prob[:20]],
        [y_true[20:40], y_pred[20:40], y_prob[20:40]],
        [y_true[40:60], y_pred[40:60], y_prob[40:60]],
    ]

    all_predictions = [r[1] for r in fold_results]
    all_labels = [r[0] for r in fold_results]
    all_probabilities = [r[2] for r in fold_results]

    aggregate = compute_aggregate_metrics(
        all_predictions, all_labels, all_probabilities
    )

    print(f"   Aggregate accuracy: {aggregate['accuracy'][0]:.4f} ± {aggregate['accuracy'][1]:.4f}")
    assert "accuracy" in aggregate

    print("   ✓ Evaluation integration passed")


def test_checkpoint_resume():
    """Test model checkpointing and resumption with a simple model."""
    from criteria_bge_hpo.training import Trainer, create_loss_function
    from torch.utils.data import DataLoader, TensorDataset

    print("\n7. Testing checkpoint resume...")

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim: int = 10):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, 1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            features = input_ids.float()
            return self.linear(features).squeeze(-1)

    # Create dummy dataset
    inputs = torch.randn(20, 10)
    labels_tensor = torch.randint(0, 2, (20,)).float()
    dataset = TensorDataset(inputs, labels_tensor)

    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "label": labels}

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = create_loss_function("bce", labels_tensor)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for 1 epoch and save checkpoint
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            checkpoint_dir=tmpdir,
        )

        history = trainer.train(loader, loader, num_epochs=1)
        print(f"   Initial training: val_loss={history['val_loss'][0]:.4f}")

        checkpoint_path = Path(tmpdir) / "best.pt"
        assert checkpoint_path.exists(), "Checkpoint not saved"

        # Load checkpoint into new trainer
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_trainer = Trainer(
            model=new_model,
            optimizer=new_optimizer,
            loss_fn=loss_fn,
            device="cpu",
        )

        new_trainer.load_checkpoint(str(checkpoint_path))
        print("   Checkpoint loaded successfully")

        assert new_trainer.current_epoch >= 0

    print("   ✓ Checkpoint resume passed")


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Integration Tests")
    print("=" * 70)

    try:
        test_end_to_end_training_simple_model()
        test_kfold_integration()
        test_mlflow_logging()
        test_loss_function_integration()
        test_config_loading()
        test_evaluation_integration()
        test_checkpoint_resume()

        print("\n" + "=" * 70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ INTEGRATION TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
