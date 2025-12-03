"""Loss functions for imbalanced binary classification.

Implements:
- Focal Loss: Addresses class imbalance by down-weighting easy examples
- Weighted BCE Loss: Binary cross-entropy with positive class weighting
- Class weight utilities: Compute weights from label distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
        https://arxiv.org/abs/1708.02002

    The focal loss focuses training on hard examples by down-weighting
    easy/well-classified examples. The modulating factor (1-p_t)^γ reduces
    the loss contribution from easy examples.

    Args:
        alpha: Weight for positive class (balances class frequencies)
               If None, computed from data
        gamma: Focusing parameter (default: 2.0)
               - γ=0: Equivalent to standard cross-entropy
               - γ>0: More focus on hard examples
               - Typical values: 1.0-3.0
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [batch_size] or [batch_size, 1] - Raw model outputs
            targets: [batch_size] - Binary labels (0 or 1)

        Returns:
            loss: Scalar if reduction != 'none', else [batch_size]
        """
        # Ensure shapes match
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t: probability of true class
        # p_t = p if y=1, else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy term: -log(p_t)
        # Use log_sigmoid for numerical stability
        ce_loss = -(
            targets * F.logsigmoid(logits) + (1 - targets) * F.logsigmoid(-logits)
        )

        # Combine focal weight and cross-entropy
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss.

    Standard BCE with class weighting via pos_weight parameter.
    Equivalent to BCEWithLogitsLoss with pos_weight.

    Formula: -w_pos * y * log(p) - (1-y) * log(1-p)
             where w_pos is the weight for positive class

    Args:
        pos_weight: Weight for positive class (default: None, computed from data)
                    Typical value: neg_count / pos_count
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = (
            torch.tensor([pos_weight]) if pos_weight is not None else None
        )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: [batch_size] or [batch_size, 1] - Raw model outputs
            targets: [batch_size] - Binary labels (0 or 1)

        Returns:
            loss: Scalar if reduction != 'none', else [batch_size]
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Move pos_weight to same device as logits
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=self.reduction
        )


def compute_class_weights(
    labels: torch.Tensor,
    method: str = "inverse",
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: [num_samples] - Binary labels (0 or 1)
        method: Weight computation method
            - 'inverse': 1 / frequency (default)
            - 'effective': Effective number of samples
                          (1 - beta^n) / (1 - beta)
        beta: Parameter for effective number method (default: 0.999)

    Returns:
        weights: [2] - Weights for [negative_class, positive_class]

    Example:
        >>> labels = torch.tensor([0, 0, 0, 0, 1])
        >>> weights = compute_class_weights(labels)
        >>> print(weights)  # [0.625, 2.5] (inverse of frequencies)
    """
    # Count samples per class
    num_samples = len(labels)
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()

    if method == "inverse":
        # Inverse frequency weighting
        weight_neg = num_samples / (2 * num_neg) if num_neg > 0 else 1.0
        weight_pos = num_samples / (2 * num_pos) if num_pos > 0 else 1.0

    elif method == "effective":
        # Effective number of samples
        # Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
        effective_neg = (1 - beta**num_neg) / (1 - beta) if num_neg > 0 else 1.0
        effective_pos = (1 - beta**num_pos) / (1 - beta) if num_pos > 0 else 1.0

        weight_neg = 1.0 / effective_neg
        weight_pos = 1.0 / effective_pos

        # Normalize
        total = weight_neg + weight_pos
        weight_neg /= total
        weight_pos /= total
        weight_neg *= 2  # Scale to sum to 2
        weight_pos *= 2

    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse' or 'effective'")

    return torch.tensor([weight_neg, weight_pos])


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss.

    Formula: pos_weight = num_negative / num_positive

    Args:
        labels: [num_samples] - Binary labels (0 or 1)

    Returns:
        pos_weight: Scalar tensor

    Example:
        >>> labels = torch.tensor([0, 0, 0, 0, 1])
        >>> pos_weight = compute_pos_weight(labels)
        >>> print(pos_weight)  # 4.0 (4 negatives / 1 positive)
    """
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()

    if num_pos == 0:
        return torch.tensor(1.0)

    return torch.tensor(num_neg / num_pos)


def compute_focal_alpha(
    labels: torch.Tensor,
    method: str = "balanced",
) -> float:
    """
    Compute alpha parameter for Focal Loss.

    Args:
        labels: [num_samples] - Binary labels (0 or 1)
        method: Computation method
            - 'balanced': alpha = num_pos / num_samples (default)
            - 'inverse': alpha = num_neg / num_samples

    Returns:
        alpha: Float value for positive class weight

    Example:
        >>> labels = torch.tensor([0, 0, 0, 0, 1])
        >>> alpha = compute_focal_alpha(labels, method='balanced')
        >>> print(alpha)  # 0.2 (1 positive / 5 samples)
    """
    num_samples = len(labels)
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()

    if method == "balanced":
        # Alpha equals positive class frequency
        alpha = num_pos / num_samples if num_samples > 0 else 0.5
    elif method == "inverse":
        # Alpha equals negative class frequency
        alpha = num_neg / num_samples if num_samples > 0 else 0.5
    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced' or 'inverse'")

    return alpha


def create_loss_function(
    loss_type: str,
    labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss functions with auto-computed weights.

    Args:
        loss_type: Type of loss function
            - 'bce': Standard binary cross-entropy
            - 'weighted_bce': BCE with pos_weight
            - 'focal': Focal loss with alpha and gamma
        labels: Training labels for computing weights (optional)
        **kwargs: Additional arguments for loss function
            - For focal: gamma (default: 2.0)
            - For weighted_bce: pos_weight (computed if not provided)
            - For focal: alpha (computed if not provided)

    Returns:
        loss_fn: Configured loss function

    Example:
        >>> labels = torch.tensor([0, 0, 0, 0, 1] * 1000)
        >>> loss_fn = create_loss_function('focal', labels, gamma=2.0)
        >>> logits = torch.randn(100)
        >>> targets = torch.randint(0, 2, (100,))
        >>> loss = loss_fn(logits, targets)
    """
    if loss_type == "bce":
        return WeightedBCELoss(pos_weight=None)

    elif loss_type == "weighted_bce":
        pos_weight = kwargs.get("pos_weight")
        if pos_weight is None and labels is not None:
            pos_weight = compute_pos_weight(labels).item()
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_type == "focal":
        gamma = kwargs.get("gamma", 2.0)
        alpha = kwargs.get("alpha")
        if alpha is None and labels is not None:
            alpha = compute_focal_alpha(labels, method="balanced")
        return FocalLoss(alpha=alpha, gamma=gamma)

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Use 'bce', 'weighted_bce', or 'focal'"
        )
