"""PyTorch Dataset for DSM-5 criteria matching."""

from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from .chunking import apply_chunking


class CriterionMatchingDataset(Dataset):
    """
    Dataset for binary classification of post-criterion pairs.

    Each sample consists of:
    - post_text: The social media post
    - criterion_text: The DSM-5 criterion description
    - label: Binary label (0=no match, 1=match)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dsm5_criteria: Dict[str, str],
        tokenizer: Optional[object] = None,
        max_length: int = 512,
        use_chunking: bool = False,
        chunk_strategy: str = "auto",
        chunk_params: Optional[Dict] = None,
        return_raw_text: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            dataframe: DataFrame with columns: post_id, post, DSM5_symptom, groundtruth
            dsm5_criteria: Dictionary mapping criterion_id to criterion text
            tokenizer: Optional tokenizer for automatic tokenization
            max_length: Maximum sequence length for tokenization (default: 512)
            use_chunking: Whether to chunk posts into smaller pieces (default: False)
            chunk_strategy: Chunking strategy (default: "auto")
            chunk_params: Parameters for chunking (default: None)
            return_raw_text: Whether to return raw text in samples (default: True)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.dsm5_criteria = dsm5_criteria
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chunking = use_chunking
        self.chunk_strategy = chunk_strategy
        self.chunk_params = chunk_params or {}
        self.return_raw_text = return_raw_text

        # Validate criteria coverage
        unique_criteria = self.dataframe["DSM5_symptom"].unique()
        missing_criteria = set(unique_criteria) - set(self.dsm5_criteria.keys())
        if missing_criteria:
            print(f"Warning: Missing criterion definitions for: {missing_criteria}")

        # Pre-process chunks if chunking is enabled
        if self.use_chunking:
            self._prepare_chunks()

    def _prepare_chunks(self) -> None:
        """Pre-compute chunks for all posts to avoid repeated computation."""
        self.dataframe["post_chunks"] = self.dataframe["post"].apply(
            lambda text: apply_chunking(text, strategy=self.chunk_strategy, **self.chunk_params)
        )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str], torch.Tensor, int]]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample data
        """
        row = self.dataframe.iloc[idx]

        # Get post and criterion text
        post_text = row["post"]
        criterion_id = row["DSM5_symptom"]
        criterion_text = self.dsm5_criteria.get(criterion_id, "")
        label = int(row["groundtruth"])

        # Build sample dictionary
        sample = {"label": label}

        # Add raw text if requested
        if self.return_raw_text:
            sample["post_text"] = post_text
            sample["criterion_text"] = criterion_text

            # Add chunks if using chunking
            if self.use_chunking:
                sample["post_chunks"] = row.get("post_chunks", [post_text])

        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                post_text,
                criterion_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            sample["input_ids"] = encoding["input_ids"].squeeze(0)
            sample["attention_mask"] = encoding["attention_mask"].squeeze(0)

            if "token_type_ids" in encoding:
                sample["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.

        Returns:
            Tensor of shape (2,) with weights for [negative_class, positive_class]
        """
        labels = self.dataframe["groundtruth"].values
        n_samples = len(labels)
        n_classes = 2

        n_negative = (labels == 0).sum()
        n_positive = (labels == 1).sum()

        weight_negative = n_samples / (n_classes * n_negative) if n_negative > 0 else 0.0
        weight_positive = n_samples / (n_classes * n_positive) if n_positive > 0 else 0.0

        return torch.tensor([weight_negative, weight_positive], dtype=torch.float32)

    def get_pos_weight(self) -> torch.Tensor:
        """
        Calculate positive class weight for BCEWithLogitsLoss.

        Returns:
            Scalar tensor with weight for positive class
        """
        labels = self.dataframe["groundtruth"].values
        n_negative = (labels == 0).sum()
        n_positive = (labels == 1).sum()

        if n_positive == 0:
            return torch.tensor(1.0)

        return torch.tensor(n_negative / n_positive, dtype=torch.float32)


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from CriterionMatchingDataset

    Returns:
        Batched dictionary with stacked tensors and lists
    """
    tensor_keys = ["input_ids", "attention_mask", "token_type_ids", "label"]
    list_keys = ["post_text", "criterion_text", "post_chunks"]

    collated = {}

    # Stack tensor keys
    for key in tensor_keys:
        if key in batch[0]:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = torch.tensor(values)

    # Collect list keys
    for key in list_keys:
        if key in batch[0]:
            collated[key] = [sample[key] for sample in batch]

    return collated


def create_dataloaders(
    train_dataset: CriterionMatchingDataset,
    val_dataset: Optional[CriterionMatchingDataset] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Union[torch.utils.data.DataLoader, tuple]:
    """
    Create DataLoader(s) for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size (default: 16)
        num_workers: Number of worker processes (default: 4)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)

    Returns:
        DataLoader if only train_dataset provided, else (train_loader, val_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    if val_dataset is None:
        return train_loader

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
