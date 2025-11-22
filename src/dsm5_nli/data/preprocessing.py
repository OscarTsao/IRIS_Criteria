"""Data loading and preprocessing for DSM-5 NLI."""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from rich.console import Console

console = Console()


def load_and_preprocess_data(config) -> pd.DataFrame:
    """Load and preprocess all data for DSM-5 NLI training.

    Args:
        config: Hydra configuration object

    Returns:
        DataFrame with columns: post_id, post, criterion_id, criterion, label
    """
    console.print("\n[cyan]═══════════════════════════════════════════════════════════[/cyan]")
    console.print("[cyan bold]               DATA LOADING & PREPROCESSING                 [/cyan bold]")
    console.print("[cyan]═══════════════════════════════════════════════════════════[/cyan]\n")

    # Load posts
    console.print(f"[yellow]Loading posts from:[/yellow] {config.data.posts_csv}")
    posts_df = pd.read_csv(config.data.posts_csv)
    console.print(f"[green]✓[/green] Loaded {len(posts_df)} posts")

    # Load annotations
    console.print(f"[yellow]Loading annotations from:[/yellow] {config.data.annotations_csv}")
    annotations_df = pd.read_csv(config.data.annotations_csv)
    console.print(f"[green]✓[/green] Loaded {len(annotations_df)} annotations")

    # Load DSM-5 criteria
    console.print(f"[yellow]Loading DSM-5 criteria from:[/yellow] {config.data.criteria_json}")
    with open(config.data.criteria_json, 'r') as f:
        criteria_data = json.load(f)

    criteria_dict = {item["id"]: item["text"] for item in criteria_data["criteria"]}
    console.print(f"[green]✓[/green] Loaded {len(criteria_dict)} DSM-5 criteria")

    # Create mapping from symptom names to criterion IDs
    symptom_to_criterion = {
        "DEPRESSED_MOOD": "A.1",
        "ANHEDONIA": "A.2",
        "APPETITE_CHANGE": "A.3",
        "SLEEP_ISSUES": "A.4",
        "PSYCHOMOTOR": "A.5",
        "FATIGUE": "A.6",
        "WORTHLESSNESS": "A.7",
        "COGNITIVE_ISSUES": "A.8",
        "SUICIDAL_THOUGHTS": "A.9",
    }

    # Create NLI pairs
    console.print("\n[yellow]Creating post-criterion pairs...[/yellow]")

    # Merge posts with annotations
    pairs_df = annotations_df.merge(posts_df, on="post_id", how="inner")

    # Map symptom names to criterion IDs, then to criterion text
    pairs_df["criterion_id"] = pairs_df["DSM5_symptom"].map(symptom_to_criterion)
    pairs_df["criterion"] = pairs_df["criterion_id"].map(criteria_dict)

    # Create binary label (1 if present, 0 if absent)
    pairs_df["label"] = (pairs_df["status"] == 1).astype(int)

    # Select and rename columns
    pairs_df = pairs_df[["post_id", "text", "criterion_id", "criterion", "label"]]
    pairs_df = pairs_df.rename(columns={"text": "post"})

    # Remove NaN values
    pairs_df = pairs_df.dropna()

    console.print(f"[green]✓[/green] Created {len(pairs_df)} NLI pairs")
    console.print(f"  • Unique posts: {pairs_df['post_id'].nunique()}")
    console.print(f"  • Unique criteria: {pairs_df['criterion_id'].nunique()}")
    console.print(f"  • Positive samples: {(pairs_df['label'] == 1).sum()}")
    console.print(f"  • Negative samples: {(pairs_df['label'] == 0).sum()}")

    console.print("\n[cyan]═══════════════════════════════════════════════════════════[/cyan]\n")

    return pairs_df
