"""Rich terminal visualization utilities.

Provides:
- Colored console output
- Training progress tables
- Fold results visualization
- Status messages with icons
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich import box
from typing import Dict, List, Optional
import numpy as np

# Global console instance
console = Console()


def print_header(title: str, subtitle: Optional[str] = None):
    """Print a styled header.

    Args:
        title: Main title
        subtitle: Optional subtitle
    """
    if subtitle:
        text = f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]"
    else:
        text = f"[bold cyan]{title}[/bold cyan]"

    console.print(Panel(text, box=box.DOUBLE, border_style="cyan"))


def print_success(message: str):
    """Print success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str):
    """Print warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_error(message: str):
    """Print error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_info(message: str):
    """Print info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_config_table(config: Dict):
    """Print configuration as a table.

    Args:
        config: Configuration dictionary
    """
    table = Table(title="Configuration", box=box.ROUNDED, show_header=True)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in config.items():
        if isinstance(value, dict):
            # Nested dict - flatten with dot notation
            for k, v in value.items():
                table.add_row(f"{key}.{k}", str(v))
        else:
            table.add_row(key, str(value))

    console.print(table)


def print_fold_header(fold_idx: int, n_folds: int, train_size: int, val_size: int):
    """Print fold information header.

    Args:
        fold_idx: Current fold index (0-based)
        n_folds: Total number of folds
        train_size: Training set size
        val_size: Validation set size
    """
    console.print()
    console.rule(f"[bold yellow]Fold {fold_idx + 1}/{n_folds}[/bold yellow]")
    console.print(f"  Train: [cyan]{train_size:,}[/cyan] samples")
    console.print(f"  Val:   [cyan]{val_size:,}[/cyan] samples")
    console.print()


def print_training_summary(history: Dict[str, List[float]]):
    """Print training history summary.

    Args:
        history: Training history with metrics
    """
    num_epochs = len(history["train_loss"])

    table = Table(title="Training History (Last 5 Epochs)", box=box.SIMPLE)
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Train Loss", justify="right", style="red")
    table.add_column("Val Loss", justify="right", style="yellow")
    table.add_column("Val Acc", justify="right", style="green")

    # Show last 5 epochs
    start_idx = max(0, num_epochs - 5)
    for i in range(start_idx, num_epochs):
        table.add_row(
            str(i + 1),
            f"{history['train_loss'][i]:.4f}",
            f"{history['val_loss'][i]:.4f}",
            f"{history['val_acc'][i]:.4f}",
        )

    console.print(table)


def print_fold_results(fold_results: List[Dict]):
    """Print K-fold cross-validation results.

    Args:
        fold_results: List of fold result dictionaries
    """
    table = Table(title="K-Fold Cross-Validation Results", box=box.DOUBLE, show_header=True)
    table.add_column("Fold", justify="center", style="cyan")
    table.add_column("Best Val Loss", justify="right", style="red")
    table.add_column("Best Val Acc", justify="right", style="green")

    for result in fold_results:
        table.add_row(
            str(result["fold"] + 1),
            f"{result['best_val_loss']:.4f}",
            f"{result['best_val_acc']:.4f}",
        )

    # Add summary row
    mean_loss = np.mean([r["best_val_loss"] for r in fold_results])
    std_loss = np.std([r["best_val_loss"] for r in fold_results])
    mean_acc = np.mean([r["best_val_acc"] for r in fold_results])
    std_acc = np.std([r["best_val_acc"] for r in fold_results])

    table.add_section()
    table.add_row(
        "[bold]Mean ± Std[/bold]",
        f"[bold]{mean_loss:.4f} ± {std_loss:.4f}[/bold]",
        f"[bold]{mean_acc:.4f} ± {std_acc:.4f}[/bold]",
    )

    console.print(table)


def print_hpo_summary(study, n_trials: int):
    """Print HPO study summary.

    Args:
        study: Optuna study object
        n_trials: Number of trials run
    """
    console.print()
    console.rule("[bold green]HPO Complete[/bold green]")

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Trials", str(n_trials))
    table.add_row("Best Trial", str(study.best_trial.number))
    table.add_row("Best Value", f"{study.best_value:.4f}")

    console.print(table)

    # Best hyperparameters
    console.print("\n[bold cyan]Best Hyperparameters:[/bold cyan]")
    params_table = Table(box=box.SIMPLE, show_header=False)
    params_table.add_column("Parameter", style="yellow")
    params_table.add_column("Value", style="green")

    for key, value in study.best_params.items():
        if isinstance(value, float):
            params_table.add_row(key, f"{value:.6f}")
        else:
            params_table.add_row(key, str(value))

    console.print(params_table)


def print_model_summary(model, num_params: int):
    """Print model architecture summary.

    Args:
        model: PyTorch model
        num_params: Number of trainable parameters
    """
    table = Table(title="Model Summary", box=box.ROUNDED)
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model Type", model.__class__.__name__)
    table.add_row("Trainable Params", f"{num_params:,}")

    # Add model-specific info
    if hasattr(model, "num_queries"):
        table.add_row("Num Queries", str(model.num_queries))
    if hasattr(model, "k_retrieved"):
        table.add_row("K Retrieved", str(model.k_retrieved))
    if hasattr(model, "temperature"):
        table.add_row("Temperature", f"{model.temperature:.2f}")

    console.print(table)


def create_training_progress() -> Progress:
    """Create a Rich progress bar for training.

    Returns:
        Progress: Rich Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
