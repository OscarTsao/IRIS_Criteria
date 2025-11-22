"""K-fold cross-validation with grouped splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from rich.console import Console
from rich.table import Table

console = Console()


def create_kfold_splits(data: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """Create K-fold splits with grouped stratification.

    Args:
        data: DataFrame with post_id and label columns
        n_splits: Number of folds
        random_state: Random seed

    Yields:
        Tuple of (train_df, val_df) for each fold
    """
    X = np.arange(len(data))
    y = data["label"].values
    groups = data["post_id"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)

        # Verify no post overlap
        train_posts = set(train_df["post_id"].unique())
        val_posts = set(val_df["post_id"].unique())
        assert len(train_posts & val_posts) == 0, f"Fold {fold_idx}: Data leakage detected!"

        yield train_df, val_df


def get_fold_statistics(data: pd.DataFrame, splits):
    """Get statistics for each fold.

    Args:
        data: Full dataset
        splits: K-fold splits iterator

    Returns:
        DataFrame with fold statistics
    """
    stats = []
    for fold_idx, (train_df, val_df) in enumerate(splits):
        stats.append({
            "Fold": fold_idx,
            "Train Size": len(train_df),
            "Val Size": len(val_df),
            "Train Posts": train_df["post_id"].nunique(),
            "Val Posts": val_df["post_id"].nunique(),
            "Train Pos%": f"{(train_df['label']==1).mean()*100:.1f}%",
            "Val Pos%": f"{(val_df['label']==1).mean()*100:.1f}%",
        })
    return pd.DataFrame(stats)


def display_fold_statistics(stats_df: pd.DataFrame):
    """Display fold statistics in rich table."""
    table = Table(title="K-Fold Statistics", show_header=True)

    for col in stats_df.columns:
        table.add_column(col, style="cyan")

    for _, row in stats_df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)
