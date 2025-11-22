"""Reproducibility utilities."""

import random
import numpy as np
import torch
from rich.console import Console

console = Console()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    console.print(f"[green]✓[/green] Set random seed: {seed}")


def enable_deterministic(tf32: bool = True):
    """Enable deterministic operations."""
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # TF32 for speed
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            console.print("[green]✓[/green] Enabled TF32 operations")

    console.print("[green]✓[/green] Enabled deterministic algorithms")


def get_device():
    """Get device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]✓[/green] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠[/yellow] CUDA not available, using CPU")

    return device


def verify_cuda_setup():
    """Verify CUDA setup."""
    if not torch.cuda.is_available():
        console.print("[yellow]⚠[/yellow] CUDA not available")
        return

    console.print("\n[cyan]═══════════════════════════════════════════════════════════[/cyan]")
    console.print("[cyan bold]                    CUDA INFORMATION                       [/cyan bold]")
    console.print("[cyan]═══════════════════════════════════════════════════════════[/cyan]")
    console.print(f"  Device: {torch.cuda.get_device_name(0)}")
    console.print(f"  CUDA Version: {torch.version.cuda}")
    console.print(f"  Device Capability: {torch.cuda.get_device_capability(0)}")

    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    console.print(f"  Total Memory: {memory_total:.2f} GB")
    console.print(f"  BF16 Supported: {torch.cuda.is_bf16_supported()}")
    console.print("[cyan]═══════════════════════════════════════════════════════════[/cyan]\n")
