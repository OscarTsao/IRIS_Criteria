"""Tests for training components (losses and trainer)."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_focal_loss():
    """Test Focal Loss implementation."""
    from criteria_bge_hpo.training import FocalLoss

    print("\n1. Testing FocalLoss...")

    # Create loss function
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Test with balanced data
    logits = torch.randn(100)
    labels = torch.randint(0, 2, (100,)).float()

    loss = loss_fn(logits, labels)
    print(f"   Focal loss (balanced): {loss.item():.4f}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"

    # Test with imbalanced data
    labels_imbalanced = torch.cat([torch.zeros(90), torch.ones(10)])
    logits_imbalanced = torch.randn(100)

    loss_imbalanced = loss_fn(logits_imbalanced, labels_imbalanced)
    print(f"   Focal loss (imbalanced): {loss_imbalanced.item():.4f}")

    # Test gamma effect (higher gamma = more focus on hard examples)
    loss_gamma0 = FocalLoss(alpha=0.25, gamma=0.0)(logits, labels)
    loss_gamma2 = FocalLoss(alpha=0.25, gamma=2.0)(logits, labels)

    print(f"   γ=0.0 (CE): {loss_gamma0.item():.4f}")
    print(f"   γ=2.0 (Focal): {loss_gamma2.item():.4f}")

    print("   ✓ FocalLoss passed")


def test_weighted_bce_loss():
    """Test Weighted BCE Loss."""
    from criteria_bge_hpo.training import WeightedBCELoss

    print("\n2. Testing WeightedBCELoss...")

    # Create loss with pos_weight
    loss_fn = WeightedBCELoss(pos_weight=4.0)

    logits = torch.randn(100)
    labels = torch.randint(0, 2, (100,)).float()

    loss = loss_fn(logits, labels)
    print(f"   Weighted BCE loss: {loss.item():.4f}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"

    # Compare with standard BCE
    loss_standard = WeightedBCELoss(pos_weight=None)(logits, labels)
    print(f"   Standard BCE loss: {loss_standard.item():.4f}")

    print("   ✓ WeightedBCELoss passed")


def test_class_weight_computation():
    """Test class weight computation utilities."""
    from criteria_bge_hpo.training import (
        compute_class_weights,
        compute_pos_weight,
        compute_focal_alpha,
    )

    print("\n3. Testing weight computation utilities...")

    # Imbalanced dataset: 90% negative, 10% positive
    labels = torch.cat([torch.zeros(900), torch.ones(100)])

    # Test class weights
    weights = compute_class_weights(labels, method="inverse")
    print(f"   Class weights (inverse): {weights.tolist()}")
    assert weights.shape == (2,)
    assert weights[1] > weights[0], "Positive class should have higher weight"

    # Test pos_weight
    pos_weight = compute_pos_weight(labels)
    print(f"   Pos weight: {pos_weight.item():.2f}")
    assert pos_weight.item() == 9.0, "pos_weight should be 900/100 = 9.0"

    # Test focal alpha
    alpha = compute_focal_alpha(labels, method="balanced")
    print(f"   Focal alpha (balanced): {alpha:.2f}")
    assert abs(alpha - 0.1) < 0.01, "Alpha should be ~0.1 (10% positive)"

    print("   ✓ Weight computation passed")


def test_loss_factory():
    """Test loss function factory."""
    from criteria_bge_hpo.training import create_loss_function

    print("\n4. Testing loss function factory...")

    labels = torch.cat([torch.zeros(900), torch.ones(100)])

    # Test BCE
    loss_bce = create_loss_function("bce")
    print(f"   Created BCE loss: {type(loss_bce).__name__}")

    # Test Weighted BCE with auto-computed weight
    loss_weighted = create_loss_function("weighted_bce", labels)
    print(f"   Created Weighted BCE: {type(loss_weighted).__name__}")

    # Test Focal with auto-computed alpha
    loss_focal = create_loss_function("focal", labels, gamma=2.0)
    print(f"   Created Focal loss: {type(loss_focal).__name__}")

    # Test forward pass
    logits = torch.randn(100)
    test_labels = torch.randint(0, 2, (100,)).float()

    for name, loss_fn in [("BCE", loss_bce), ("Weighted BCE", loss_weighted), ("Focal", loss_focal)]:
        loss = loss_fn(logits, test_labels)
        print(f"   {name}: {loss.item():.4f}")
        assert torch.isfinite(loss)

    print("   ✓ Loss factory passed")


def test_trainer_initialization():
    """Test Trainer initialization."""
    from criteria_bge_hpo.training import Trainer

    print("\n5. Testing Trainer initialization...")

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.linear(input_ids.float()).squeeze(-1)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Test basic initialization
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_type="bce",
        device="cpu",
        gradient_accumulation_steps=2,
        use_amp=False,
    )

    print(f"   Device: {trainer.device}")
    print(f"   Loss function: {type(trainer.loss_fn).__name__}")
    print(f"   Gradient accumulation: {trainer.gradient_accumulation_steps}")
    print(f"   AMP enabled: {trainer.use_amp}")

    assert trainer.device == "cpu"
    assert trainer.gradient_accumulation_steps == 2

    print("   ✓ Trainer initialization passed")


def test_trainer_forward():
    """Test Trainer forward pass with token-based inputs."""
    from criteria_bge_hpo.training import Trainer

    print("\n6. Testing Trainer forward pass...")

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.linear(input_ids.float()).squeeze(-1)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_type="bce",
        device="cpu",
    )

    # Create dummy batch
    batch = {
        "input_ids": torch.randn(4, 10),
        "attention_mask": torch.ones(4, 10),
        "label": torch.randint(0, 2, (4,)).float(),
    }

    # Test forward
    logits = trainer._forward(batch)
    print(f"   Input shape: {batch['input_ids'].shape}")
    print(f"   Logits shape: {logits.shape}")

    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"

    print("   ✓ Trainer forward passed")


def test_trainer_training_step():
    """Test single training step."""
    from criteria_bge_hpo.training import Trainer

    print("\n7. Testing Trainer training step...")

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.linear(input_ids.float()).squeeze(-1)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_type="bce",
        device="cpu",
        gradient_accumulation_steps=1,
    )

    # Create dummy batch
    batch = {
        "input_ids": torch.randn(4, 10),
        "attention_mask": torch.ones(4, 10),
        "label": torch.randint(0, 2, (4,)).float(),
    }

    # Training step
    model.train()
    loss_value, correct, batch_size = trainer._train_step(batch, batch_idx=0)

    print(f"   Loss: {loss_value:.4f}")
    print(f"   Correct: {correct}/{batch_size}")

    assert loss_value >= 0, "Loss should be non-negative"
    assert 0 <= correct <= batch_size, "Correct should be in valid range"

    print("   ✓ Trainer training step passed")


def test_trainer_evaluate():
    """Test evaluation loop."""
    from criteria_bge_hpo.training import Trainer

    print("\n8. Testing Trainer evaluation...")

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.linear(input_ids.float()).squeeze(-1)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_type="bce",
        device="cpu",
    )

    # Create dummy dataset
    dataset = TensorDataset(
        torch.randn(20, 10),  # input_ids
        torch.ones(20, 10),   # attention_mask
        torch.randint(0, 2, (20,)).float(),  # labels
    )

    # Wrap to convert to dict format
    def collate_fn(batch):
        # batch is a list of tuples
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

    # Create data loader with custom collate
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Evaluate
    metrics = trainer.evaluate(loader)

    print(f"   Val loss: {metrics['loss']:.4f}")
    print(f"   Val accuracy: {metrics['accuracy']:.4f}")

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    print("   ✓ Trainer evaluation passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Training Component Tests")
    print("=" * 70)

    try:
        test_focal_loss()
        test_weighted_bce_loss()
        test_class_weight_computation()
        test_loss_factory()
        test_trainer_initialization()
        test_trainer_forward()
        test_trainer_training_step()
        test_trainer_evaluate()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
