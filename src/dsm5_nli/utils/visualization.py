"""Visualization utilities for terminal output."""

from rich.console import Console
from rich.table import Table

console = Console()


def print_header(title: str):
    """Print a formatted header."""
    console.print(f"\n[cyan bold]{'=' * 60}[/cyan bold]")
    console.print(f"[cyan bold]{title.center(60)}[/cyan bold]")
    console.print(f"[cyan bold]{'=' * 60}[/cyan bold]\n")


def print_config_summary(config):
    """Print configuration summary."""
    print_header("CONFIGURATION SUMMARY")

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Model", config.model.model_name)
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Epochs", str(config.training.num_epochs))
    table.add_row("K-Folds", str(config.kfold.n_splits))
    table.add_row("BF16", str(config.training.optimization.use_bf16))
    table.add_row("TF32", str(config.training.optimization.use_tf32))

    console.print(table)
    console.print()


def print_fold_summary(fold_results):
    """Print K-fold summary."""
    print_header("K-FOLD CROSS-VALIDATION RESULTS")

    table = Table()
    table.add_column("Fold", style="cyan")
    table.add_column("F1", style="green")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Precision", style="blue")
    table.add_column("Recall", style="magenta")

    for i, result in enumerate(fold_results):
        agg = result["aggregate"]
        table.add_row(
            str(i),
            f"{agg['f1']:.4f}",
            f"{agg['accuracy']:.4f}",
            f"{agg['precision']:.4f}",
            f"{agg['recall']:.4f}",
        )

    # Add mean row
    import numpy as np
    mean_f1 = np.mean([r["aggregate"]["f1"] for r in fold_results])
    mean_acc = np.mean([r["aggregate"]["accuracy"] for r in fold_results])
    mean_prec = np.mean([r["aggregate"]["precision"] for r in fold_results])
    mean_recall = np.mean([r["aggregate"]["recall"] for r in fold_results])

    table.add_row(
        "[bold]Mean[/bold]",
        f"[bold]{mean_f1:.4f}[/bold]",
        f"[bold]{mean_acc:.4f}[/bold]",
        f"[bold]{mean_prec:.4f}[/bold]",
        f"[bold]{mean_recall:.4f}[/bold]",
    )

    console.print(table)
    console.print()

    return {
        "mean_f1": mean_f1,
        "mean_accuracy": mean_acc,
        "mean_precision": mean_prec,
        "mean_recall": mean_recall,
    }
