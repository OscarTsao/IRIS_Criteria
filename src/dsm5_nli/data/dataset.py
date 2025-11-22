"""PyTorch Dataset for DSM-5 NLI."""

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DSM5NLIDataset(Dataset):
    """Dataset for DSM-5 NLI binary classification.

    Each sample is a (post, criterion) pair with binary label.
    Input format: [CLS] post [SEP] criterion [SEP]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        verify_format: bool = False,
    ):
        """Initialize dataset.

        Args:
            data: DataFrame with columns: post, criterion, label
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            verify_format: Whether to validate column presence (debug helper)
        """
        if verify_format:
            required_columns = {"post", "criterion", "label"}
            missing = required_columns - set(data.columns)
            if missing:
                raise ValueError(
                    f"DSM5NLIDataset missing required columns: {sorted(missing)}"
                )

        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.data.iloc[idx]

        # Tokenize: [CLS] post [SEP] criterion [SEP]
        encoding = self.tokenizer(
            str(row["post"]),
            str(row["criterion"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


def create_dataloaders(train_dataset, val_dataset, batch_size: int,
                      num_workers: int = 4, pin_memory: bool = True):
    """Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
