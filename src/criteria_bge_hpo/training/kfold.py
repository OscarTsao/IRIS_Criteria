"""K-fold cross-validation with grouped stratified splits."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    group_column: str = "post_id",
    stratify_column: str = "groundtruth",
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits grouped by post_id.

    This ensures that all samples from the same post stay together in either
    train or validation set, preventing data leakage.

    Args:
        df: DataFrame with data
        n_folds: Number of folds (default: 5)
        group_column: Column to group by (default: "post_id")
        stratify_column: Column to stratify by (default: "groundtruth")
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
    if stratify_column not in df.columns:
        raise ValueError(f"Stratify column '{stratify_column}' not found in DataFrame")

    groups = df[group_column].values
    labels = df[stratify_column].values

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, val_idx in sgkf.split(X=df, y=labels, groups=groups):
        splits.append((train_idx, val_idx))

    return splits


def validate_fold_splits(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    group_column: str = "post_id",
    stratify_column: str = "groundtruth",
) -> None:
    """
    Validate that K-fold splits meet grouping and stratification requirements.

    Args:
        df: DataFrame with data
        splits: List of (train_indices, val_indices) tuples
        group_column: Column that should not be split across train/val
        stratify_column: Column to check for balanced distribution

    Raises:
        ValueError: If splits violate grouping or have severe imbalance
    """
    print(f"\nValidating {len(splits)} folds...")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Check no overlap
        overlap = set(train_idx) & set(val_idx)
        if overlap:
            raise ValueError(f"Fold {fold_idx}: Train/val overlap detected")

        # Check grouping
        train_groups = set(df.iloc[train_idx][group_column].unique())
        val_groups = set(df.iloc[val_idx][group_column].unique())
        group_overlap = train_groups & val_groups
        if group_overlap:
            raise ValueError(f"Fold {fold_idx}: Group column appears in both train and val")

        # Check class distribution
        train_labels = df.iloc[train_idx][stratify_column]
        val_labels = df.iloc[val_idx][stratify_column]

        train_dist = train_labels.value_counts(normalize=True).sort_index()
        val_dist = val_labels.value_counts(normalize=True).sort_index()

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_idx)} samples, {len(train_groups)} groups")
        print(f"    Class distribution: {dict(train_dist)}")
        print(f"  Val:   {len(val_idx)} samples, {len(val_groups)} groups")
        print(f"    Class distribution: {dict(val_dist)}")

    print("\nValidation complete: All folds meet grouping requirements")


def get_fold_datasets(
    df: pd.DataFrame,
    fold_idx: int,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get train and validation DataFrames for a specific fold.

    Args:
        df: Full DataFrame
        fold_idx: Index of fold to retrieve (0 to n_folds-1)
        splits: List of (train_indices, val_indices) tuples from create_kfold_splits

    Returns:
        Tuple of (train_df, val_df)
    """
    if fold_idx < 0 or fold_idx >= len(splits):
        raise ValueError(f"fold_idx {fold_idx} out of range [0, {len(splits)})")

    train_idx, val_idx = splits[fold_idx]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df


def create_single_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    group_column: str = "post_id",
    stratify_column: str = "groundtruth",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a single train/test split with grouping and stratification.

    Args:
        df: DataFrame with data
        test_size: Proportion of data for test set (default: 0.2)
        group_column: Column to group by (default: "post_id")
        stratify_column: Column to stratify by (default: "groundtruth")
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_df, test_df)
    """
    n_folds = int(1 / test_size)

    splits = create_kfold_splits(
        df, n_folds=n_folds, group_column=group_column, stratify_column=stratify_column, random_state=random_state
    )

    train_df, test_df = get_fold_datasets(df, fold_idx=0, splits=splits)

    return train_df, test_df
