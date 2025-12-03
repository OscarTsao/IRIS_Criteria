"""Classification head architectures for transformers."""

import torch
import torch.nn as nn


class ClassifierHeadFactory:
    """Factory for creating different classification head architectures."""

    @staticmethod
    def create(
        head_type: str,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        intermediate_size: int = None,
    ) -> nn.Module:
        """
        Create classification head.

        Args:
            head_type: Type of head (linear, pooler_linear, mlp1, mlp2,
                       mean_pooling, max_pooling, attention_pooling)
            hidden_size: Input hidden size
            num_labels: Number of output labels
            dropout: Dropout rate
            intermediate_size: Intermediate size for MLP heads (default: hidden_size)

        Returns:
            Classification head module
        """
        if intermediate_size is None:
            intermediate_size = hidden_size

        heads = {
            "linear": LinearHead(hidden_size, num_labels),
            "pooler_linear": PoolerLinearHead(hidden_size, num_labels, dropout),
            "mlp1": MLP1Head(hidden_size, num_labels, dropout),
            "mlp2": MLP2Head(hidden_size, intermediate_size, num_labels, dropout),
            "mean_pooling": MeanPoolingHead(hidden_size, num_labels, dropout),
            "max_pooling": MaxPoolingHead(hidden_size, num_labels, dropout),
            "attention_pooling": AttentionPoolingHead(hidden_size, num_labels, dropout),
        }

        if head_type not in heads:
            raise ValueError(
                f"Unknown head type: {head_type}. "
                f"Available: {list(heads.keys())}"
            )

        return heads[head_type]


class LinearHead(nn.Module):
    """Simple linear classification head."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, num_labels]
        """
        # Use CLS token if sequence
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, 0]  # CLS token

        return self.classifier(hidden_states)


class PoolerLinearHead(nn.Module):
    """Classification head with dense layer, tanh, and dropout."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, 0]

        pooled = self.dense(hidden_states)
        pooled = torch.tanh(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class MLP1Head(nn.Module):
    """Single-layer MLP with GELU activation."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, 0]

        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)


class MLP2Head(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(intermediate_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, 0]

        x = self.dense1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        return self.dense2(x)


class MeanPoolingHead(nn.Module):
    """Mean pooling over sequence, then linear."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with mean pooling.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, num_labels]
        """
        if hidden_states.dim() == 2:
            # Already pooled
            pooled = hidden_states
        else:
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Expand mask to match hidden_states
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden = hidden_states * mask
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1)
                pooled = sum_hidden / sum_mask.clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class MaxPoolingHead(nn.Module):
    """Max pooling over sequence, then linear."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            pooled = hidden_states
        else:
            # Max pooling with attention mask
            if attention_mask is not None:
                # Set masked positions to large negative value
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden = hidden_states.clone()
                masked_hidden[mask == 0] = -1e9
                pooled = masked_hidden.max(dim=1)[0]
            else:
                pooled = hidden_states.max(dim=1)[0]

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class AttentionPoolingHead(nn.Module):
    """Attention-based pooling over sequence."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            pooled = hidden_states
        else:
            # Compute attention scores
            attention_scores = self.attention(hidden_states).squeeze(-1)
            # attention_scores: [batch_size, seq_len]

            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask == 0, -1e9
                )

            # Softmax normalization
            attention_weights = torch.softmax(attention_scores, dim=1)
            # attention_weights: [batch_size, seq_len]

            # Weighted sum
            pooled = torch.bmm(
                attention_weights.unsqueeze(1), hidden_states
            ).squeeze(1)
            # pooled: [batch_size, hidden_size]

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
