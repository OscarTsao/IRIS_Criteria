"""Data loading and preprocessing utilities for DSM-5 criteria matching."""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_groundtruth_data(csv_path: str) -> pd.DataFrame:
    """
    Load groundtruth data from CSV file.

    Args:
        csv_path: Path to CSV file with columns: post_id, post, DSM5_symptom, groundtruth

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify required columns
    required_columns = {"post_id", "post", "DSM5_symptom", "groundtruth"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert groundtruth to int
    df["groundtruth"] = df["groundtruth"].astype(int)

    # Remove rows with missing data
    initial_len = len(df)
    df = df.dropna(subset=list(required_columns))
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows with missing data")

    return df


def load_dsm5_criteria(json_path: str) -> Dict[str, str]:
    """
    Load DSM-5 criteria definitions from JSON file.

    Args:
        json_path: Path to JSON file with DSM-5 criteria

    Returns:
        Dictionary mapping criterion_id (e.g., "A.1") to criterion text

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON structure is invalid
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "criteria" not in data:
        raise ValueError("JSON must contain 'criteria' field")

    # Extract criteria as dictionary
    criteria_dict = {}
    for criterion in data["criteria"]:
        if "id" not in criterion or "text" not in criterion:
            print(f"Warning: Skipping criterion without id or text: {criterion}")
            continue

        criterion_id = criterion["id"]
        criterion_text = criterion["text"]

        if not criterion_text or criterion_text.strip() == "":
            print(f"Warning: Skipping criterion {criterion_id} with empty text")
            continue

        criteria_dict[criterion_id] = criterion_text

    if not criteria_dict:
        raise ValueError("No valid criteria found in JSON file")

    return criteria_dict


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Args:
        df: DataFrame to split
        test_size: Proportion of data to use for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Whether to stratify by groundtruth label (default: True)

    Returns:
        Tuple of (train_df, test_df)
    """
    if stratify:
        stratify_col = df["groundtruth"]
    else:
        stratify_col = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_class_distribution(df: pd.DataFrame, label_col: str = "groundtruth") -> Dict[int, float]:
    """
    Calculate class distribution in dataset.

    Args:
        df: DataFrame with labels
        label_col: Name of label column (default: "groundtruth")

    Returns:
        Dictionary mapping class label to proportion
    """
    value_counts = df[label_col].value_counts()
    total = len(df)

    distribution = {int(label): count / total for label, count in value_counts.items()}

    return distribution


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics for dataset.

    Args:
        df: DataFrame to summarize
    """
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Unique posts: {df['post_id'].nunique()}")
    print(f"Unique criteria: {df['DSM5_symptom'].nunique()}")

    # Class distribution
    dist = get_class_distribution(df)
    print("\nClass Distribution:")
    for label, proportion in sorted(dist.items()):
        count = int(proportion * len(df))
        print(f"  Class {label}: {count:6d} ({proportion:6.1%})")

    # Post length statistics
    if "post" in df.columns:
        post_lengths = df["post"].str.split().str.len()
        print("\nPost Length (words):")
        print(f"  Mean:   {post_lengths.mean():.1f}")
        print(f"  Median: {post_lengths.median():.1f}")
        print(f"  Min:    {post_lengths.min()}")
        print(f"  Max:    {post_lengths.max()}")

    print("=" * 60)
