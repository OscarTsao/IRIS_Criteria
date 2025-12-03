"""Command-line interface for IRIS training.

Usage:
    # Train with default config
    python -m criteria_bge_hpo.cli train

    # Train with specific model
    python -m criteria_bge_hpo.cli train model=iris

    # Override hyperparameters
    python -m criteria_bge_hpo.cli train training.num_epochs=50 training.lr=1e-4

    # Evaluate model
    python -m criteria_bge_hpo.cli eval checkpoint=path/to/checkpoint.pt

    # Run HPO
    python -m criteria_bge_hpo.cli hpo n_trials=100
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    """
    Train IRIS model.

    Args:
        cfg: Hydra configuration
    """
    from criteria_bge_hpo.data import (
        load_groundtruth_data,
        load_dsm5_criteria,
        CriterionMatchingDataset,
    )
    from criteria_bge_hpo.training import create_kfold_splits, Trainer, create_loss_function
    from criteria_bge_hpo.utils import setup_mlflow, MLflowLogger
    from torch.utils.data import DataLoader, Subset

    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Setup MLflow
    setup_mlflow(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    # Load data
    logger.info(f"Loading data from {cfg.paths.groundtruth_csv}")
    df = load_groundtruth_data(cfg.paths.groundtruth_csv)
    criteria = load_dsm5_criteria(cfg.paths.dsm5_json)

    # Create K-fold splits
    logger.info(f"Creating {cfg.data.n_folds}-fold splits")
    splits = create_kfold_splits(
        df,
        n_folds=cfg.data.n_folds,
        group_column=cfg.data.group_column,
        stratify_column=cfg.data.stratify_column,
    )

    # K-fold cross-validation
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*70}")
        logger.info(f"Fold {fold_idx + 1}/{cfg.data.n_folds}")
        logger.info(f"{'='*70}")

        # Create datasets
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = CriterionMatchingDataset(
            train_df, criteria, tokenizer=None, max_length=cfg.data.max_length
        )
        val_dataset = CriterionMatchingDataset(
            val_df, criteria, tokenizer=None, max_length=cfg.data.max_length
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True if cfg.data.num_workers > 0 else False,
            prefetch_factor=2 if cfg.data.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size * 2,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.data.num_workers > 0 else False,
            prefetch_factor=2 if cfg.data.num_workers > 0 else None,
        )

        # Create model (IRIS only)
        if cfg.model.model_type != "iris":
            raise ValueError(f"Only IRIS model_type is supported, got: {cfg.model.model_type}")

        from criteria_bge_hpo.models import IRISForCriterionMatching

        model = IRISForCriterionMatching(
            num_queries=cfg.model.num_queries,
            k_retrieved=cfg.model.k_retrieved,
            embedding_dim=cfg.model.embedding_dim,
            temperature=cfg.model.temperature,
            encoder_name=cfg.model.encoder_name,
            query_penalty_lambda=cfg.model.query_penalty_lambda,
            query_penalty_threshold=cfg.model.query_penalty_threshold,
        )

        # Build retriever
        logger.info("Building IRIS retriever...")
        all_posts = df["post"].unique().tolist()
        model.build_retriever(
            all_posts,
            batch_size=cfg.model.build_retriever_batch_size,
            use_gpu=cfg.model.use_gpu_retrieval,
        )

        # Create optimizer
        if cfg.training.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.optimizer.lr,
                weight_decay=cfg.training.optimizer.weight_decay,
                betas=cfg.training.optimizer.betas,
                eps=cfg.training.optimizer.eps,
            )
        elif cfg.training.optimizer.name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.training.optimizer.lr,
                weight_decay=cfg.training.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.training.optimizer.name}")

        # Create loss function
        labels = torch.tensor(train_df[cfg.data.stratify_column].values)
        loss_fn = create_loss_function(
            cfg.training.loss.type,
            labels=labels,
            gamma=cfg.training.loss.get("focal_gamma"),
            pos_weight=cfg.training.loss.get("pos_weight"),
        )

        # Learning rate scheduler
        scheduler = None
        if cfg.training.scheduler.name:
            total_steps = len(train_loader) * cfg.training.num_epochs // cfg.training.gradient_accumulation_steps

            if cfg.training.scheduler.name == "linear_warmup":
                from torch.optim.lr_scheduler import LinearLR, SequentialLR

                warmup_steps = cfg.training.scheduler.warmup_steps
                if warmup_steps is None:
                    warmup_steps = int(total_steps * cfg.training.scheduler.warmup_ratio)

                # Warmup then decay
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )

        # Create MLflow logger and train (use context manager)
        run_name = f"{cfg.experiment.name}_fold{fold_idx}"

        with MLflowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            run_name=run_name,
            tags={
                **cfg.experiment.tags,
                "fold": str(fold_idx),
                "model_type": cfg.model.model_type,
            },
        ) as mlflow_logger:
            # Log hyperparameters
            mlflow_logger.log_params({
                "model_type": cfg.model.model_type,
                "num_epochs": cfg.training.num_epochs,
                "batch_size": cfg.data.batch_size,
                "learning_rate": cfg.training.optimizer.lr,
                "loss_type": cfg.training.loss.type,
                "fold": fold_idx,
            })

            # Create trainer
            checkpoint_dir = Path(cfg.paths.checkpoint_dir) / f"fold_{fold_idx}"
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                device=cfg.device,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                max_grad_norm=cfg.training.max_grad_norm,
                use_amp=cfg.training.use_amp,
                amp_dtype=cfg.training.amp_dtype,
                early_stopping_patience=cfg.training.early_stopping_patience,
                checkpoint_dir=str(checkpoint_dir),
                save_best_only=cfg.training.save_best_only,
            )

            # Train
            logger.info(f"Training for {cfg.training.num_epochs} epochs...")
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=cfg.training.num_epochs,
            )

            # Log training history
            from criteria_bge_hpo.utils import log_training_history
            log_training_history(history, mlflow_logger)

            # Save best metrics
            best_val_loss = min(history["val_loss"])
            best_val_acc = max(history["val_acc"])

            fold_results.append({
                "fold": fold_idx,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            })

            mlflow_logger.log_metric("best_val_loss", best_val_loss)
            mlflow_logger.log_metric("best_val_acc", best_val_acc)

        logger.info(f"Fold {fold_idx} - Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}")

    # Log aggregate results
    logger.info(f"\n{'='*70}")
    logger.info("K-Fold Cross-Validation Results")
    logger.info(f"{'='*70}")

    mean_loss = np.mean([r["best_val_loss"] for r in fold_results])
    std_loss = np.std([r["best_val_loss"] for r in fold_results])
    mean_acc = np.mean([r["best_val_acc"] for r in fold_results])
    std_acc = np.std([r["best_val_acc"] for r in fold_results])

    logger.info(f"Mean val loss: {mean_loss:.4f} ± {std_loss:.4f}")
    logger.info(f"Mean val acc:  {mean_acc:.4f} ± {std_acc:.4f}")

    # Log to MLflow
    with MLflowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"{cfg.experiment.name}_summary",
        tags={**cfg.experiment.tags, "type": "summary"},
    ) as summary_logger:
        summary_logger.log_params({
            "model_type": cfg.model.model_type,
            "n_folds": cfg.data.n_folds,
        })
        summary_logger.log_metric("cv_val_loss_mean", mean_loss)
        summary_logger.log_metric("cv_val_loss_std", std_loss)
        summary_logger.log_metric("cv_val_acc_mean", mean_acc)
        summary_logger.log_metric("cv_val_acc_std", std_acc)
        summary_logger.log_dict(fold_results, "fold_results.json")

    logger.info("Training complete!")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig):
    """
    Evaluate trained model.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Evaluation not yet implemented")
    # TODO: Implement evaluation


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def hpo(cfg: DictConfig):
    """
    Run hyperparameter optimization.

    Args:
        cfg: Hydra configuration
    """
    from criteria_bge_hpo.hpo import HPORunner
    from criteria_bge_hpo.utils import setup_mlflow, MLflowLogger

    # Print config
    logger.info("Starting Hyperparameter Optimization")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Setup MLflow
    setup_mlflow(
        experiment_name=f"{cfg.mlflow.experiment_name}_hpo",
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    # Get HPO parameters from command line or use defaults
    n_trials = cfg.get("n_trials", 100)
    n_folds = cfg.data.get("n_folds", 5)
    model_type = cfg.model.model_type

    if model_type != "iris":
        raise ValueError(f"Only IRIS model_type is supported for HPO, got: {model_type}")

    logger.info(f"Model type: {model_type}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Number of folds: {n_folds}")

    # Create HPO runner
    runner = HPORunner(
        model_type=model_type,
        n_folds=n_folds,
        n_trials=n_trials,
        n_jobs=1,  # Sequential for now
        study_name=f"{model_type}_hpo_{cfg.experiment.name}",
        storage="sqlite:///optuna.db",
        direction="maximize",  # Maximize F1/accuracy
        metric_name="val_acc",  # or "val_f1"
        pruning=True,
        n_warmup_steps=0,  # Aggressive pruning
    )

    # Run optimization
    study = runner.optimize()

    # Log best config to MLflow
    with MLflowLogger(
        experiment_name=f"{cfg.mlflow.experiment_name}_hpo",
        run_name=f"hpo_best_{model_type}",
        tags={
            "type": "hpo_summary",
            "model_type": model_type,
        },
    ) as mlflow_logger:
        # Log best hyperparameters
        mlflow_logger.log_params(study.best_params)
        mlflow_logger.log_metric("best_value", study.best_value)
        mlflow_logger.log_metric("n_trials", len(study.trials))

        # Save best config
        import json
        best_config_path = f"best_config_{model_type}.json"
        with open(best_config_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
        mlflow_logger.log_artifact(best_config_path)

    logger.info(f"\nHPO Complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"\nBest config saved to: {best_config_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m criteria_bge_hpo.cli {train|eval|hpo} [options]")
        sys.exit(1)

    command = sys.argv[1]
    # Remove command from argv so Hydra can parse the rest
    sys.argv.pop(1)

    if command == "train":
        train()
    elif command == "eval":
        evaluate()
    elif command == "hpo":
        hpo()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval, hpo")
        sys.exit(1)
