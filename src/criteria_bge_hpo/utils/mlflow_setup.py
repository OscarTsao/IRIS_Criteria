"""MLflow experiment tracking setup and utilities.

Provides:
- MLflow experiment initialization
- Automatic metric logging
- Model artifact saving
- Hyperparameter logging
- Run management
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """
    Setup MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI (default: ./mlruns)
        artifact_location: Artifact storage location (default: ./mlartifacts)

    Returns:
        experiment_id: MLflow experiment ID
    """
    # Set tracking URI
    if tracking_uri is None:
        tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)

    return experiment_id


class MLflowLogger:
    """MLflow logger for training runs.

    Handles:
    - Metric logging (loss, accuracy, F1, etc.)
    - Parameter logging (hyperparameters)
    - Model artifact saving
    - Run management
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of this run (optional)
            tags: Tags for this run (optional)
            tracking_uri: MLflow tracking URI (optional)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.tracking_uri = tracking_uri

        # Setup experiment
        setup_mlflow(experiment_name, tracking_uri)

        # Start run
        self.run = None
        self._start_run()

    def _start_run(self):
        """Start a new MLflow run."""
        self.run = mlflow.start_run(run_name=self.run_name)

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        logger.info(f"Started MLflow run: {self.run.info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters (hyperparameters).

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step (optional)
        """
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Training step (optional)
        """
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log PyTorch model.

        Args:
            model: PyTorch model
            artifact_path: Path within artifacts (default: "model")
            registered_model_name: Name for model registry (optional)
        """
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file).

        Args:
            local_path: Local file path
            artifact_path: Path within artifacts (optional)
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
        """
        try:
            mlflow.log_dict(dictionary, filename)
            logger.info(f"Logged dictionary as {filename}")
        except Exception as e:
            logger.error(f"Failed to log dictionary: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tags
        """
        for key, value in tags.items():
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Failed to set tag {key}: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.

        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")
        """
        if self.run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run.info.run_id} (status: {status})")
            self.run = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)


def log_training_history(
    history: Dict[str, list],
    logger: Optional[MLflowLogger] = None,
):
    """
    Log training history to MLflow.

    Args:
        history: Training history dict with lists of metrics
        logger: MLflow logger (if None, logs to active run)
    """
    if not history:
        return

    # Get number of epochs
    num_epochs = len(history[list(history.keys())[0]])

    for epoch in range(num_epochs):
        metrics = {}
        for metric_name, values in history.items():
            if epoch < len(values):
                metrics[metric_name] = values[epoch]

        if logger:
            logger.log_metrics(metrics, step=epoch)
        else:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=epoch)


def log_kfold_results(
    fold_results: list,
    metric_names: list,
    logger: Optional[MLflowLogger] = None,
):
    """
    Log K-fold cross-validation results.

    Args:
        fold_results: List of result dicts (one per fold)
        metric_names: Names of metrics to aggregate
        logger: MLflow logger (if None, logs to active run)
    """
    import numpy as np

    # Compute mean and std for each metric
    for metric_name in metric_names:
        values = [fold.get(metric_name, 0.0) for fold in fold_results]
        mean_value = np.mean(values)
        std_value = np.std(values)

        if logger:
            logger.log_metric(f"cv_{metric_name}_mean", mean_value)
            logger.log_metric(f"cv_{metric_name}_std", std_value)
        else:
            mlflow.log_metric(f"cv_{metric_name}_mean", mean_value)
            mlflow.log_metric(f"cv_{metric_name}_std", std_value)

    # Log individual fold results
    if logger:
        logger.log_dict(fold_results, "fold_results.json")
    else:
        mlflow.log_dict(fold_results, "fold_results.json")


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get existing experiment or create new one.

    Args:
        experiment_name: Name of the experiment

    Returns:
        experiment_id: MLflow experiment ID
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
