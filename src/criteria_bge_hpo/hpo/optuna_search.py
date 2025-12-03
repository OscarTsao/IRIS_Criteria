"""Optuna-based hyperparameter optimization for the IRIS model.

Supports:
- Search space definition for IRIS architecture
- Nested cross-validation with pruning
- Single-objective optimization (F1, accuracy, or loss)
- Best configuration selection
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelSearchSpace:
    """Define search spaces for different model architectures."""

    @staticmethod
    def iris_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        """
        IRIS hyperparameter search space.

        Args:
            trial: Optuna trial

        Returns:
            config: Dictionary of hyperparameters
        """
        config = {
            # Architecture
            "num_queries": trial.suggest_categorical("num_queries", [4, 8, 12, 16]),
            "k_retrieved": trial.suggest_categorical("k_retrieved", [8, 12, 16, 20]),
            "temperature": trial.suggest_float("temperature", 0.05, 0.2),
            "query_penalty_lambda": trial.suggest_float("query_penalty_lambda", 0.05, 0.2),
            "query_penalty_threshold": trial.suggest_float("query_penalty_threshold", 0.3, 0.5),
            # Training
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_epochs": trial.suggest_categorical("num_epochs", [30, 50, 70, 100]),
            # Loss
            "loss_type": trial.suggest_categorical("loss_type", ["focal", "weighted_bce"]),
        }

        if config["loss_type"] == "focal":
            config["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)

        return config


class HPORunner:
    """Run hyperparameter optimization with Optuna."""

    def __init__(
        self,
        model_type: str,
        n_folds: int = 5,
        n_trials: int = 100,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = "sqlite:///optuna.db",
        direction: str = "maximize",
        metric_name: str = "val_f1",
        pruning: bool = True,
        n_warmup_steps: int = 5,
    ):
        """
        Initialize HPO runner.

        Args:
            model_type: "iris"
            n_folds: Number of CV folds
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs
            study_name: Name of the Optuna study
            storage: Storage URL for Optuna
            direction: "maximize" or "minimize"
            metric_name: Metric to optimize
            pruning: Enable pruning for early stopping
            n_warmup_steps: Warmup steps before pruning
        """
        self.model_type = model_type
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name or f"{model_type}_hpo"
        self.storage = storage
        self.direction = direction
        self.metric_name = metric_name

        # Pruner
        self.pruner = None
        if pruning:
            self.pruner = MedianPruner(
                n_startup_trials=n_warmup_steps,
                n_warmup_steps=n_warmup_steps,
            )

        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction=self.direction,
            sampler=TPESampler(seed=42),
            pruner=self.pruner,
        )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial

        Returns:
            metric_value: Objective metric value
        """
        if self.model_type != "iris":
            raise ValueError(f"Only IRIS model_type is supported, got: {self.model_type}")

        # Get hyperparameters
        config = ModelSearchSpace.iris_search_space(trial)

        # Import here to avoid circular imports
        from criteria_bge_hpo.data import (
            load_groundtruth_data,
            load_dsm5_criteria,
            CriterionMatchingDataset,
        )
        from criteria_bge_hpo.training import create_kfold_splits, Trainer, create_loss_function
        from torch.utils.data import DataLoader

        # Load data
        df = load_groundtruth_data("data/groundtruth/criteria_matching_groundtruth.csv")
        criteria = load_dsm5_criteria("data/DSM5/MDD_Criteria.json")

        # K-fold splits
        splits = create_kfold_splits(df, n_folds=self.n_folds, group_column="post_id")

        # Cross-validation
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Early stopping after fold 0 (aggressive pruning)
            if fold_idx == 1 and self.pruner:
                # Report fold 0 score for pruning
                trial.report(fold_scores[0], fold_idx - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Create datasets
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_dataset = CriterionMatchingDataset(
                train_df, criteria, tokenizer=None, max_length=512
            )
            val_dataset = CriterionMatchingDataset(
                val_df, criteria, tokenizer=None, max_length=512
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=2,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"] * 2,
                shuffle=False,
                num_workers=2,
            )

            # Create IRIS model
            from criteria_bge_hpo.models import IRISForCriterionMatching

            model = IRISForCriterionMatching(
                num_queries=config["num_queries"],
                k_retrieved=config["k_retrieved"],
                temperature=config["temperature"],
                query_penalty_lambda=config["query_penalty_lambda"],
                query_penalty_threshold=config["query_penalty_threshold"],
            )
            model.build_retriever(df["post"].unique().tolist())

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )

            # Loss function
            labels = torch.tensor(train_df["groundtruth"].values)
            loss_fn = create_loss_function(
                config["loss_type"],
                labels=labels,
                gamma=config.get("focal_gamma", 2.0),
            )

            # Trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device="cuda" if torch.cuda.is_available() else "cpu",
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
                use_amp=True,
                early_stopping_patience=5,
            )

            # Train
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config["num_epochs"],
            )

            # Get best score
            if "f1" in self.metric_name:
                # Compute F1 from accuracy (approximation)
                best_score = max(history["val_acc"])
            elif "loss" in self.metric_name:
                best_score = min(history["val_loss"])
            else:
                best_score = max(history["val_acc"])

            fold_scores.append(best_score)

            # Report for pruning
            trial.report(best_score, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return mean score
        mean_score = np.mean(fold_scores)
        return mean_score

    def optimize(self) -> optuna.Study:
        """
        Run optimization.

        Returns:
            study: Optimized Optuna study
        """
        logger.info(f"Starting HPO for {self.model_type}")
        logger.info(f"n_trials={self.n_trials}, n_folds={self.n_folds}")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study

    def get_best_config(self) -> Dict[str, Any]:
        """
        Get best configuration.

        Returns:
            config: Best hyperparameter configuration
        """
        return self.study.best_params


def load_best_config(
    study_name: str,
    storage: str = "sqlite:///optuna.db",
) -> Dict[str, Any]:
    """
    Load best configuration from a completed study.

    Args:
        study_name: Name of the study
        storage: Storage URL

    Returns:
        config: Best hyperparameter configuration
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_params
