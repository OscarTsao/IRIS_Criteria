"""
Command-line interface for DSM-5 NLI Binary Classification.

Usage:
    python -m dsm5_nli.cli train training.num_epochs=100 training.early_stopping_patience=20  # Run K-fold training
    python -m dsm5_nli.cli hpo --n-trials 500       # Run HPO
    python -m dsm5_nli.cli eval --fold 0            # Evaluate specific fold
"""

import sys

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
import mlflow
import optuna
from optuna.pruners import MedianPruner
import numpy as np
from rich.console import Console

from dsm5_nli.data.preprocessing import load_and_preprocess_data
from dsm5_nli.data.dataset import DSM5NLIDataset, create_dataloaders
from dsm5_nli.models.bert_classifier import BERTClassifier
from dsm5_nli.training.kfold import (
    create_kfold_splits,
    get_fold_statistics,
    display_fold_statistics,
)
from dsm5_nli.training.trainer import Trainer, create_optimizer_and_scheduler
from dsm5_nli.evaluation.evaluator import (
    Evaluator,
)
from dsm5_nli.utils.reproducibility import (
    set_seed,
    enable_deterministic,
    get_device,
    verify_cuda_setup,
)
from dsm5_nli.utils.mlflow_setup import setup_mlflow, log_config, start_run
from dsm5_nli.utils.visualization import print_header, print_config_summary, print_fold_summary

console = Console()


def run_single_fold(
    config: DictConfig,
    pairs_df,
    train_idx,
    val_idx,
    fold: int,
    tokenizer,
    device,
):
    """
    Train and evaluate a single fold.

    Args:
        config: Hydra configuration
        pairs_df: Full dataset
        train_idx: Training indices
        val_idx: Validation indices
        fold: Fold number
        tokenizer: HuggingFace tokenizer
        device: Device to train on

    Returns:
        Dictionary of validation metrics
    """
    console.print(f"\n[bold cyan]Fold {fold + 1}/{config.kfold.n_splits}[/bold cyan]\n")

    # Create datasets
    train_dataset = DSM5NLIDataset(
        pairs_df.iloc[train_idx],
        tokenizer,
        max_length=config.data.max_length,
        verify_format=(fold == 0),  # Verify format only for first fold
        model_name=config.model.model_name,
    )

    val_dataset = DSM5NLIDataset(
        pairs_df.iloc[val_idx],
        tokenizer,
        max_length=config.data.max_length,
        model_name=config.model.model_name,
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    # Create model
    model = BERTClassifier(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        freeze_bert=config.model.freeze_bert,
    )

    console.print(
        f"Model parameters: {model.get_num_trainable_params():,} trainable / {model.get_num_total_params():,} total"
    )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        train_loader,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        use_fused=config.training.optimization.fused_adamw,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_bf16=config.training.optimization.use_bf16,
        use_compile=config.training.optimization.use_torch_compile,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        mlflow_enabled=True,
        early_stopping_patience=config.training.early_stopping_patience,
    )

    # Train
    trainer.train(num_epochs=config.training.num_epochs, fold=fold)

    # Evaluate trained model on validation set for aggregate metrics
    evaluator = Evaluator(
        model=trainer.model,
        device=device,
        use_bf16=config.training.optimization.use_bf16,
    )
    val_data = pairs_df.iloc[val_idx].reset_index(drop=True)
    eval_results = evaluator.evaluate(val_loader, val_data)

    # Log aggregate metrics to MLflow if enabled
    if mlflow.active_run():
        for metric_name, metric_value in eval_results["aggregate"].items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)

    # Get best metrics
    best_metrics = {
        "val_f1": trainer.best_val_f1,
        "fold": fold,
        "aggregate": eval_results["aggregate"],
        "per_criterion": eval_results["per_criterion"],
    }

    return best_metrics


def run_kfold_training(config: DictConfig):
    """
    Run complete K-fold cross-validation training.

    Args:
        config: Hydra configuration
    """
    print_header("DSM-5 NLI K-Fold Training", f"Experiment: {config.experiment_name}")

    # Set up reproducibility
    set_seed(config.seed)
    enable_deterministic(config.reproducibility.deterministic, config.reproducibility.tf32)

    # Verify CUDA
    verify_cuda_setup()
    device = get_device()

    # Print configuration
    print_config_summary(config)

    # Load data
    pairs_df = load_and_preprocess_data(config)

    # Create K-fold splits (store to reuse for stats and training)
    splits = list(create_kfold_splits(pairs_df, n_splits=config.kfold.n_splits))

    # Display split statistics
    stats_df = get_fold_statistics(pairs_df, splits)
    display_fold_statistics(stats_df)

    # Set up MLflow
    setup_mlflow(config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    console.print(f"[green]✓[/green] Loaded tokenizer: {config.model.model_name}\n")

    # Train each fold
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        with start_run(run_name=f"fold_{fold}", tags={"fold": str(fold)}):
            # Log config
            log_config(config)

            # Train fold
            metrics = run_single_fold(
                config,
                pairs_df,
                train_idx,
                val_idx,
                fold,
                tokenizer,
                device,
            )

            fold_results.append(metrics)

    # Print summary
    console.print("\n")
    mean_metrics = print_fold_summary(fold_results)

    # Log overall results
    with start_run(run_name="overall_results", tags={"type": "summary"}):
        for key, value in mean_metrics.items():
            mlflow.log_metric(key, value)

    console.print("\n[green]✓[/green] K-fold training complete!")
    console.print(
        f"Mean F1: {mean_metrics['mean_f1']:.4f} ± {np.std([r['val_f1'] for r in fold_results]):.4f}\n"
    )


def run_hpo(config: DictConfig, n_trials: int):
    """
    Run Optuna hyperparameter optimization.

    Args:
        config: Hydra configuration
        n_trials: Number of trials to run
    """
    print_header("DSM-5 NLI Hyperparameter Optimization", f"Trials: {n_trials}")

    # Set up reproducibility
    set_seed(config.seed)
    enable_deterministic(config.reproducibility.deterministic, config.reproducibility.tf32)
    device = get_device()

    # Load data
    pairs_df = load_and_preprocess_data(config)

    # Create K-fold splits (will be reused across trials)
    splits = list(create_kfold_splits(pairs_df, n_splits=config.kfold.n_splits))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Set up MLflow
    setup_mlflow(config)

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Samples hyperparameters and evaluates via K-fold CV.
        """
        # Sample hyperparameters (using Optuna 3.x API)
        learning_rate = trial.suggest_float(
            "learning_rate",
            config.hpo.search_space.learning_rate.low,
            config.hpo.search_space.learning_rate.high,
            log=True,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            config.hpo.search_space.batch_size.choices,
        )

        dropout = trial.suggest_float(
            "dropout",
            config.hpo.search_space.dropout.low,
            config.hpo.search_space.dropout.high,
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            config.hpo.search_space.weight_decay.low,
            config.hpo.search_space.weight_decay.high,
            log=True,
        )

        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            config.hpo.search_space.warmup_ratio.low,
            config.hpo.search_space.warmup_ratio.high,
        )

        console.print(f"\n[bold magenta]Trial {trial.number}[/bold magenta]")
        console.print(f"  LR: {learning_rate:.2e}, BS: {batch_size}, Dropout: {dropout:.3f}")

        # Run K-fold CV with sampled hyperparameters
        fold_scores = []

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            # Log trial parameters
            mlflow.log_params(trial.params)

            for fold, (train_idx, val_idx) in enumerate(splits):
                # Create datasets
                train_dataset = DSM5NLIDataset(
                    pairs_df.iloc[train_idx],
                    tokenizer,
                    max_length=config.data.max_length,
                    model_name=config.model.model_name,
                )

                val_dataset = DSM5NLIDataset(
                    pairs_df.iloc[val_idx],
                    tokenizer,
                    max_length=config.data.max_length,
                    model_name=config.model.model_name,
                )

                # Create dataloaders with trial batch size
                train_loader, val_loader = create_dataloaders(
                    train_dataset,
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=config.training.num_workers,
                    pin_memory=config.training.pin_memory,
                )

                # Create model with trial dropout
                model = BERTClassifier(
                    model_name=config.model.model_name,
                    num_labels=config.model.num_labels,
                    dropout=dropout,
                )

                # Create optimizer with trial hyperparameters
                optimizer, scheduler = create_optimizer_and_scheduler(
                    model,
                    train_loader,
                    num_epochs=config.training.num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    warmup_ratio=warmup_ratio,
                    use_fused=config.training.optimization.fused_adamw,
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                )

                # Create trainer
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    use_bf16=config.training.optimization.use_bf16,
                    use_compile=config.training.optimization.use_torch_compile,
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                    max_grad_norm=config.training.max_grad_norm,
                    mlflow_enabled=False,  # Disable per-step logging during HPO
                    early_stopping_patience=config.training.early_stopping_patience,
                )

                # Train for the configured number of epochs (default 100 with patience 20)
                trainer.train(num_epochs=config.training.num_epochs, fold=fold)

                # Get best F1
                fold_f1 = trainer.best_val_f1
                fold_scores.append(fold_f1)

                # Report intermediate value for pruning
                trial.report(fold_f1, fold)

                # Prune if unpromising
                if trial.should_prune():
                    console.print(f"[yellow]⚠[/yellow] Trial {trial.number} pruned at fold {fold}")
                    raise optuna.TrialPruned()

            # Calculate mean F1 across folds
            mean_f1 = np.mean(fold_scores)
            std_f1 = np.std(fold_scores)

            # Log results
            mlflow.log_metric("mean_f1", mean_f1)
            mlflow.log_metric("std_f1", std_f1)

            console.print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

        return mean_f1

    # Create Optuna study
    pruner = MedianPruner(
        n_startup_trials=config.hpo.pruner.n_startup_trials,
        n_warmup_steps=config.hpo.pruner.n_warmup_steps,
        interval_steps=config.hpo.pruner.interval_steps,
    )

    study = optuna.create_study(
        study_name=config.hpo.study_name,
        storage=config.hpo.storage,
        direction=config.hpo.direction,
        pruner=pruner,
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Print results
    console.print("\n[bold green]HPO Complete![/bold green]\n")
    console.print(f"Best trial: {study.best_trial.number}")
    console.print(f"Best F1: {study.best_value:.4f}")
    console.print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        console.print(f"  {key}: {value}")

    # Log best trial to MLflow
    with start_run(run_name="best_trial", tags={"type": "hpo_best"}):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """Main entry point."""
    # Get command from Hydra config (set via command=train, command=hpo, etc.)
    command = config.get("command", None)

    if command is None:
        console.print(
            "[red]Error:[/red] No command specified. Use command=train, command=hpo, or command=eval"
        )
        console.print("\nExamples:")
        console.print(
            "  python -m dsm5_nli.cli command=train training.num_epochs=100 training.early_stopping_patience=20"
        )
        console.print("  python -m dsm5_nli.cli command=hpo n_trials=500")
        console.print("  python -m dsm5_nli.cli command=eval fold=0")
        sys.exit(1)

    if command == "train":
        run_kfold_training(config)
    elif command == "hpo":
        n_trials = config.get("n_trials", 500)
        run_hpo(config, n_trials)
    elif command == "eval":
        console.print("[yellow]⚠[/yellow] Evaluation command not yet implemented")
    else:
        console.print(f"[red]Error:[/red] Unknown command '{command}'. Use train, hpo, or eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
