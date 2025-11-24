#!/usr/bin/env python3
"""
Advanced Optuna HPO pipeline for fine-tuning BERT-style encoders with optional PEFT.

The script supports:
    * Comprehensive CLI for dataset/model specification
    * Rich hyperparameter search space (architecture, optimizer, scheduler, loss, regularization)
    * PEFT methods (LoRA, DoRA, IA3) and partial/full fine-tuning
    * Threshold tuning for binary / multilabel problems
    * Optuna study management with RDB storage and pruning
    * Dynamic worker manager that launches parallel trials based on GPU availability
    * Final model retraining with best hyperparameters
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.samplers import GridSampler, TPESampler
from optuna.trial import Trial
from sklearn import metrics

try:
    import psutil
except ImportError:
    psutil = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from peft import IA3Config, LoraConfig, get_peft_model
except ImportError:
    IA3Config = None
    LoraConfig = None
    get_peft_model = None
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


LOGGER = logging.getLogger("bert_peft_hpo")
GLOBAL_STATE: Dict[str, Any] = {"initialized": False}


# ----------------------------
# CLI
# ----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna PEFT HPO for BERT-style classifiers")
    parser.add_argument("--model-name", required=True, help="HuggingFace encoder checkpoint")
    parser.add_argument("--train-file", required=True, help="Training data file (csv/json/parquet)")
    parser.add_argument("--valid-file", required=True, help="Validation data file (csv/json/parquet)")
    parser.add_argument("--text-column", required=True, help="Name of the text column")
    parser.add_argument("--label-column", required=True, help="Name of the label column")
    parser.add_argument("--num-labels", type=int, required=True, help="Number of labels")
    parser.add_argument(
        "--problem-type",
        choices=["binary", "multiclass", "multilabel"],
        required=True,
        help="Classification problem type",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--study-name", default="bert_peft_hpo", help="Optuna study name")
    parser.add_argument("--storage-url", default="sqlite:///hpo.db", help="Optuna storage URL")
    parser.add_argument("--sampler", choices=["tpe", "grid"], default="tpe", help="Optuna sampler")
    parser.add_argument("--n-trials", type=int, default=500, help="Maximum total trials")
    parser.add_argument("--timeout", type=int, default=None, help="Global optimization timeout (s)")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent worker processes",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help='Comma-separated GPU IDs (e.g., "0,1"); empty string uses all available GPUs',
    )
    parser.add_argument(
        "--min-free-mem-gb",
        type=float,
        default=4.0,
        help="Minimum free GPU memory (GB) required to start a trial",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser


# ----------------------------
# Dataset utilities
# ----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_data_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        return "csv"
    if ext in [".json", ".jsonl"]:
        return "json"
    if ext in [".parquet"]:
        return "parquet"
    raise ValueError(f"Unsupported file extension: {ext}")


def parse_multilabel(value: Any, num_labels: int) -> List[float]:
    indices: List[int] = []
    if isinstance(value, str):
        indices = [int(x) for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple)):
        indices = [int(x) for x in value]
    elif isinstance(value, (int, float)):
        indices = [int(value)]
    else:
        raise ValueError(f"Unsupported multilabel value: {value}")
    vec = np.zeros(num_labels, dtype=np.float32)
    if indices:
        vec[np.clip(indices, 0, num_labels - 1)] = 1.0
    return vec.tolist()


def load_datasets(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if load_dataset is None:
        raise ImportError("datasets library is required. Please install 'datasets' to proceed.")
    data_format = infer_data_format(args.train_file)
    valid_format = infer_data_format(args.valid_file)
    if data_format != valid_format:
        raise ValueError("Train and validation file formats must match")
    data_files = {"train": args.train_file, "valid": args.valid_file}
    dataset_dict = load_dataset(data_format, data_files=data_files)
    train_records: List[Dict[str, Any]] = []
    valid_records: List[Dict[str, Any]] = []
    for split_name, tgt in [("train", train_records), ("valid", valid_records)]:
        for row in dataset_dict[split_name]:
            text = row[args.text_column]
            label_raw = row[args.label_column]
            if args.problem_type == "multilabel":
                label = parse_multilabel(label_raw, args.num_labels)
            elif args.problem_type == "binary":
                label = int(label_raw)
            else:
                label = int(label_raw)
            tgt.append({"text": text, "label": label})
    return train_records, valid_records


def compute_class_stats(
    records: List[Dict[str, Any]], args: argparse.Namespace
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if args.problem_type == "multilabel":
        positives = np.zeros(args.num_labels, dtype=np.float64)
        for row in records:
            positives += np.array(row["label"], dtype=np.float64)
        stats["positives"] = positives
        stats["total"] = len(records)
    else:
        labels = [row["label"] for row in records]
        stats["positives"] = sum(labels) if args.problem_type == "binary" else None
        stats["label_hist"] = np.bincount(labels, minlength=args.num_labels)
        stats["total"] = len(records)
    return stats


class TokenizedDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def tokenize_records(
    records: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    hparams: Dict[str, Any],
) -> TokenizedDataset:
    texts = [row["text"] for row in records]
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=hparams["max_seq_length"],
        return_tensors="pt",
    )
    if tokenizer.model_input_names and "token_type_ids" not in encodings:
        # Some tokenizers (e.g., RoBERTa) do not provide token_type_ids; that's OK.
        pass
    if args.problem_type == "multilabel":
        labels = torch.tensor([row["label"] for row in records], dtype=torch.float32)
    elif args.problem_type == "binary":
        labels = torch.tensor([row["label"] for row in records], dtype=torch.float32).unsqueeze(-1)
    else:
        labels = torch.tensor([row["label"] for row in records], dtype=torch.long)
    return TokenizedDataset(encodings, labels)


def prepare_dataloaders(
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    hparams: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = tokenize_records(GLOBAL_STATE["train_records"], tokenizer, args, hparams)
    valid_dataset = tokenize_records(GLOBAL_STATE["valid_records"], tokenizer, args, hparams)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["train_batch_size"],
        shuffle=True,
        drop_last=False,
    )
    eval_batch_size = max(1, hparams["train_batch_size"] * hparams["eval_batch_factor"])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, valid_loader


# ----------------------------
# Model & PEFT
# ----------------------------


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        head_num_layers: int,
        head_hidden_mult: float,
        activation: str,
        norm_type: str,
        dropout: float,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        hidden_dim = int(round(input_dim * head_hidden_mult))
        act_fn = nn.ReLU if activation == "relu" else nn.GELU
        if head_num_layers == 0:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, num_labels))
        else:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, hidden_dim))
            if norm_type == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BertWithHead(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        problem_type: str,
        hparams: Dict[str, Any],
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        hidden_size = self.encoder.config.hidden_size
        pooling_dim = hidden_size
        if hparams["pooling_type"] == "cls_mean":
            pooling_dim = hidden_size * 2
        self.classifier = ClassificationHead(
            input_dim=pooling_dim,
            num_labels=num_labels,
            head_num_layers=hparams["head_num_layers"],
            head_hidden_mult=hparams.get("head_hidden_mult", 1.0),
            activation=hparams["head_activation"],
            norm_type=hparams["head_norm_type"],
            dropout=hparams["head_dropout"],
        )
        self.problem_type = problem_type
        self.hparams = hparams
        self.num_labels = num_labels

    def _pool_hidden_states(self, hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        last_k = self.hparams["use_last_k_layers"]
        layers = hidden_states[-last_k:]
        if self.hparams["layer_pooling_mode"] == "mean":
            stacked = torch.stack(layers, dim=0).mean(dim=0)
        else:
            stacked = layers[-1]
        pooled_outputs: List[torch.Tensor] = []
        if self.hparams["pooling_type"] in ("cls", "cls_mean"):
            if hasattr(self.encoder.config, "use_pooler") and self.encoder.config.use_pooler:
                pooled_outputs.append(stacked[:, 0])
            else:
                pooled_outputs.append(stacked[:, 0])
        if self.hparams["pooling_type"] in ("mean", "cls_mean"):
            mask = attention_mask.unsqueeze(-1)
            masked = stacked * mask
            mean_embedding = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled_outputs.append(mean_embedding)
        return torch.cat(pooled_outputs, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.hidden_states
        pooled = self._pool_hidden_states(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        return logits


def freeze_encoder_layers(model: BertWithHead, strategy: str) -> None:
    if strategy == "none":
        return
    encoder = model.encoder
    if strategy in ("freeze_embeddings", "freeze_to_layer_6"):
        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is not None:
            for param in embeddings.parameters():
                param.requires_grad = False
    if strategy == "freeze_to_layer_6":
        encoder_module = getattr(encoder, "encoder", None)
        if encoder_module is not None and hasattr(encoder_module, "layer"):
            for layer in encoder_module.layer[:6]:
                for param in layer.parameters():
                    param.requires_grad = False


def resolve_target_modules(profile: str) -> List[str]:
    if profile == "attn_only":
        return ["query", "key", "value", "o_proj", "out_proj"]
    return ["query", "key", "value", "o_proj", "out_proj", "dense", "fc1", "fc2"]


def apply_peft(model: BertWithHead, hparams: Dict[str, Any]) -> None:
    peft_type = hparams["peft_type"]
    if peft_type == "none":
        freeze_encoder_layers(model, hparams["encoder_freeze_strategy"])
        return
    if get_peft_model is None or LoraConfig is None or IA3Config is None:
        raise ImportError("peft library is required for PEFT configurations. Please install 'peft'.")
    target_modules = resolve_target_modules(
        hparams.get("lora_target_profile", hparams.get("ia3_target_profile", "attn_only"))
    )
    for param in model.encoder.parameters():
        param.requires_grad = False
    if peft_type in {"lora", "dora"}:
        config = LoraConfig(
            r=hparams["lora_r"],
            lora_alpha=hparams["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=hparams["lora_dropout"],
            bias="none",
            task_type="SEQ_CLS",
            use_dora=peft_type == "dora",
        )
        model.encoder = get_peft_model(model.encoder, config)
    elif peft_type == "ia3":
        config = IA3Config(target_modules=target_modules, feedforward_modules=["dense", "fc1", "fc2"])
        model.encoder = get_peft_model(model.encoder, config)
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")


def apply_mixout_to_head(model: BertWithHead, mixout_p: float) -> None:
    if mixout_p <= 0.0:
        return

    class MixLinear(nn.Module):
        def __init__(self, linear: nn.Linear, p: float):
            super().__init__()
            self.base = linear
            self.p = p
            self.register_buffer("target_weight", linear.weight.data.clone())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training and self.p > 0:
                mask = torch.bernoulli((1 - self.p) * torch.ones_like(self.base.weight))
                mixed_weight = mask * self.base.weight + (1 - mask) * self.target_weight
            else:
                mixed_weight = self.base.weight
            return F.linear(x, mixed_weight, self.base.bias)

    for name, module in model.classifier.named_children():
        if isinstance(module, nn.Linear):
            setattr(model.classifier, name, MixLinear(module, mixout_p))


# ----------------------------
# Loss functions & thresholding
# ----------------------------


class BCEWrapper(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.type_as(logits)
        return self.loss(logits, labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits)
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            loss = self.alpha * loss * targets + (1 - self.alpha) * loss * (1 - targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AdaptiveFocalLoss(FocalLoss):
    def __init__(self, alpha: float, gamma_base: float, gamma_max: float):
        super().__init__(alpha=alpha, gamma=gamma_base)
        self.gamma_base = gamma_base
        self.gamma_max = gamma_max

    def update_progress(self, progress: float) -> None:
        progress = float(np.clip(progress, 0.0, 1.0))
        self.gamma = self.gamma_base + (self.gamma_max - self.gamma_base) * progress


class HybridLoss(nn.Module):
    def __init__(self, focal: FocalLoss, bce: BCEWrapper, lam: float):
        super().__init__()
        self.focal = focal
        self.bce = bce
        self.lam = lam

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.lam * self.focal(logits, targets) + (1.0 - self.lam) * self.bce(logits, targets)


def build_loss(
    hparams: Dict[str, Any],
    args: argparse.Namespace,
    class_stats: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    loss_type = hparams["loss_type"]
    if args.problem_type == "multiclass":
        return nn.CrossEntropyLoss()
    pos_weight_tensor = None
    if loss_type == "bce_weighted":
        positives = class_stats.get("positives")
        total = class_stats.get("total", 1)
        if positives is None or np.any(positives == 0):
            pos_weight = torch.ones(args.num_labels)
        else:
            pos_weight = torch.tensor((total - positives) / np.clip(positives, 1e-6, None))
        pos_weight *= hparams["bce_pos_weight_scale"]
        pos_weight_tensor = pos_weight.to(device)
        return BCEWrapper(pos_weight_tensor)
    if loss_type == "bce":
        return BCEWrapper()
    focal_alpha = hparams["focal_alpha"]
    focal_gamma = hparams.get("focal_gamma", 2.0)
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if loss_type == "adaptive_focal":
        loss = AdaptiveFocalLoss(
            alpha=focal_alpha,
            gamma_base=hparams["adaptive_gamma_base"],
            gamma_max=hparams["adaptive_gamma_max"],
        )
        return loss
    if loss_type == "hybrid":
        focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        bce = BCEWrapper()
        return HybridLoss(focal, bce, lam=hparams["hybrid_lambda"])
    raise ValueError(f"Unsupported loss type: {loss_type}")


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric_name: str,
    problem_type: str,
) -> float:
    best_thresh = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.1, 0.9, 81)
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        if problem_type == "binary":
            score = metrics.f1_score(labels, preds)
        else:
            score = metrics.f1_score(labels, preds, average="macro")
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh


# ----------------------------
# Metrics
# ----------------------------


def compute_metrics_from_outputs(
    labels: np.ndarray,
    probs: np.ndarray,
    args: argparse.Namespace,
    threshold: float,
) -> Dict[str, float]:
    if args.problem_type == "multiclass":
        preds = np.argmax(probs, axis=1)
        gold = labels
        macro_f1 = metrics.f1_score(gold, preds, average="macro")
        acc = metrics.accuracy_score(gold, preds)
        return {"macro_f1": macro_f1, "accuracy": acc}
    if args.problem_type == "binary":
        preds = (probs >= threshold).astype(int)
        macro_f1 = metrics.f1_score(labels, preds)
        acc = metrics.accuracy_score(labels, preds)
        auroc = metrics.roc_auc_score(labels, probs)
        return {"macro_f1": macro_f1, "accuracy": acc, "auroc": auroc}
    # multilabel
    preds = (probs >= threshold).astype(int)
    macro_f1 = metrics.f1_score(labels, preds, average="macro", zero_division=0)
    acc = metrics.accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


# ----------------------------
# Hyperparameter search space
# ----------------------------


GRID_SPACE = {
    "peft_type": ["none", "lora", "ia3"],
    "pooling_type": ["cls", "mean"],
    "use_last_k_layers": [1, 2],
    "layer_pooling_mode": ["last", "mean"],
    "head_num_layers": [0, 1],
    "head_hidden_mult": [1.0],
    "head_activation": ["relu"],
    "head_norm_type": ["none"],
    "encoder_freeze_strategy": ["none", "freeze_embeddings"],
    "encoder_lr_layer_decay": [0.9, 1.0],
    "optimizer_type": ["adamw"],
    "scheduler_type": ["linear"],
    "train_batch_size": [16],
    "eval_batch_factor": [1, 2],
    "gradient_accumulation_steps": [1, 2],
    "loss_type": ["bce", "focal"],
    "regularization_type": ["none", "rdrop"],
    "threshold_strategy": ["fixed", "opt_on_val"],
    "max_seq_length": [256],
    "num_epochs": [100],
    "weight_decay": [1e-4],
    "warmup_ratio": [0.05],
    "max_grad_norm": [1.0],
    "head_lr_factor": [5.0],
    "decision_threshold": [0.5],
}


def sample_hparams(trial: Trial) -> Dict[str, Any]:
    hparams: Dict[str, Any] = {}
    hparams["peft_type"] = trial.suggest_categorical("peft_type", ["none", "lora", "dora", "ia3"])
    hparams["pooling_type"] = trial.suggest_categorical("pooling_type", ["cls", "mean", "cls_mean"])
    hparams["use_last_k_layers"] = trial.suggest_int("use_last_k_layers", 1, 4)
    hparams["layer_pooling_mode"] = trial.suggest_categorical("layer_pooling_mode", ["last", "mean"])
    hparams["head_num_layers"] = trial.suggest_int("head_num_layers", 0, 1)
    if hparams["head_num_layers"] == 1:
        hparams["head_hidden_mult"] = trial.suggest_categorical("head_hidden_mult", [0.5, 1.0, 2.0])
    else:
        hparams["head_hidden_mult"] = 1.0
    hparams["head_activation"] = trial.suggest_categorical("head_activation", ["relu", "gelu"])
    hparams["head_norm_type"] = trial.suggest_categorical("head_norm_type", ["none", "layernorm"])
    hparams["head_dropout"] = trial.suggest_float("head_dropout", 0.0, 0.5)
    hparams["encoder_freeze_strategy"] = trial.suggest_categorical(
        "encoder_freeze_strategy", ["none", "freeze_embeddings", "freeze_to_layer_6"]
    )
    hparams["encoder_lr_layer_decay"] = trial.suggest_float("encoder_lr_layer_decay", 0.8, 1.0)
    hparams["optimizer_type"] = trial.suggest_categorical("optimizer_type", ["adamw", "adamw_amsgrad"])
    hparams["scheduler_type"] = trial.suggest_categorical(
        "scheduler_type", ["linear", "cosine", "constant_with_warmup"]
    )
    hparams["train_batch_size"] = trial.suggest_categorical("train_batch_size", [8, 16, 32])
    hparams["eval_batch_factor"] = trial.suggest_categorical("eval_batch_factor", [1, 2, 4])
    hparams["gradient_accumulation_steps"] = trial.suggest_categorical(
        "gradient_accumulation_steps", [1, 2, 4]
    )
    hparams["loss_type"] = trial.suggest_categorical(
        "loss_type", ["bce", "bce_weighted", "focal", "adaptive_focal", "hybrid"]
    )
    hparams["regularization_type"] = trial.suggest_categorical("regularization_type", ["none", "mixout", "rdrop"])
    hparams["threshold_strategy"] = trial.suggest_categorical("threshold_strategy", ["fixed", "opt_on_val"])
    hparams["max_seq_length"] = trial.suggest_categorical("max_seq_length", [128, 192, 256, 320, 384])
    hparams["num_epochs"] = trial.suggest_categorical("num_epochs", [100])
    hparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hparams["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    hparams["max_grad_norm"] = trial.suggest_categorical("max_grad_norm", [0.0, 1.0, 2.0])
    hparams["adamw_beta1"] = trial.suggest_float("adamw_beta1", 0.85, 0.98)
    hparams["adamw_beta2"] = trial.suggest_float("adamw_beta2", 0.98, 0.999)
    hparams["adam_epsilon"] = trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True)
    if hparams["peft_type"] == "none":
        hparams["encoder_lr"] = trial.suggest_float("encoder_lr", 1e-5, 5e-5, log=True)
        hparams["head_lr_factor"] = trial.suggest_float("head_lr_factor", 1.0, 10.0, log=True)
    else:
        hparams["peft_lr"] = trial.suggest_float("peft_lr", 1e-4, 1e-2, log=True)
        hparams["head_lr_factor"] = trial.suggest_float("head_lr_factor", 1.0, 5.0, log=True)
    if hparams["peft_type"] in {"lora", "dora"}:
        hparams["lora_r"] = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
        hparams["lora_alpha"] = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        hparams["lora_dropout"] = trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1])
        hparams["lora_target_profile"] = trial.suggest_categorical(
            "lora_target_profile", ["attn_only", "attn_ffn"]
        )
    if hparams["peft_type"] == "ia3":
        hparams["ia3_target_profile"] = trial.suggest_categorical(
            "ia3_target_profile", ["attn_only", "attn_ffn"]
        )
    if hparams["loss_type"] == "bce_weighted":
        hparams["bce_pos_weight_scale"] = trial.suggest_float("bce_pos_weight_scale", 0.5, 2.0, log=True)
    if hparams["loss_type"] in {"focal", "adaptive_focal", "hybrid"}:
        hparams["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 4.0)
        hparams["focal_alpha"] = trial.suggest_categorical("focal_alpha", [0.25, 0.5, 0.75])
    if hparams["loss_type"] == "adaptive_focal":
        hparams["adaptive_gamma_base"] = trial.suggest_float("adaptive_gamma_base", 0.0, 2.0)
        hparams["adaptive_gamma_max"] = trial.suggest_float("adaptive_gamma_max", 2.0, 5.0)
    if hparams["loss_type"] == "hybrid":
        hparams["hybrid_lambda"] = trial.suggest_float("hybrid_lambda", 0.2, 0.8)
    if hparams["regularization_type"] == "mixout":
        hparams["mixout_p"] = trial.suggest_float("mixout_p", 0.7, 0.95)
    if hparams["regularization_type"] == "rdrop":
        hparams["rdrop_alpha"] = trial.suggest_float("rdrop_alpha", 0.1, 5.0, log=True)
    if hparams["threshold_strategy"] == "fixed":
        hparams["decision_threshold"] = trial.suggest_float("decision_threshold", 0.1, 0.9)
    return hparams


# ----------------------------
# Optimizer and scheduler
# ----------------------------


def get_layerwise_lr_params(
    model: BertWithHead,
    base_lr: float,
    decay: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    encoder = model.encoder
    params: List[Dict[str, Any]] = []
    if not hasattr(encoder, "encoder") or not hasattr(encoder.encoder, "layer"):
        params.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": base_lr})
        return params
    layers = encoder.encoder.layer
    num_layers = len(layers)
    lr = base_lr
    for idx in reversed(range(num_layers)):
        layer = layers[idx]
        lr_layer = base_lr * (decay ** (num_layers - idx - 1))
        params.append(
            {
                "params": [p for p in layer.parameters() if p.requires_grad],
                "lr": lr_layer,
                "weight_decay": weight_decay,
            }
        )
    # Pooler / embeddings
    params.append(
        {
            "params": [
                p
                for n, p in encoder.named_parameters()
                if ("embeddings" in n or "pooler" in n) and p.requires_grad
            ],
            "lr": base_lr * (decay ** num_layers),
            "weight_decay": weight_decay,
        }
    )
    return params


def build_optimizer(
    model: BertWithHead,
    hparams: Dict[str, Any],
) -> AdamW:
    weight_decay = hparams["weight_decay"]
    optimizer_params: List[Dict[str, Any]] = []
    if hparams["peft_type"] == "none":
        base_lr = hparams["encoder_lr"]
        if hparams["encoder_lr_layer_decay"] < 0.999:
            optimizer_params.extend(get_layerwise_lr_params(model, base_lr, hparams["encoder_lr_layer_decay"], weight_decay))
        else:
            optimizer_params.append(
                {
                    "params": [p for p in model.encoder.parameters() if p.requires_grad],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            )
    else:
        optimizer_params.append(
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": hparams["peft_lr"],
                "weight_decay": weight_decay,
            }
        )
    head_lr = (
        hparams["encoder_lr"] * hparams["head_lr_factor"]
        if hparams["peft_type"] == "none"
        else hparams["peft_lr"] * hparams["head_lr_factor"]
    )
    optimizer_params.append(
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": head_lr,
            "weight_decay": weight_decay,
        }
    )
    amsgrad = hparams["optimizer_type"] == "adamw_amsgrad"
    optimizer = AdamW(
        optimizer_params,
        lr=head_lr,
        betas=(hparams["adamw_beta1"], hparams["adamw_beta2"]),
        eps=hparams["adam_epsilon"],
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer


def build_scheduler(
    optimizer: AdamW,
    hparams: Dict[str, Any],
    total_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_training_steps * hparams["warmup_ratio"])
    if hparams["scheduler_type"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif hparams["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps)
    else:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


# ----------------------------
# Training & evaluation loop
# ----------------------------


def forward_pass(
    model: BertWithHead,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    return logits


def training_step(
    model: BertWithHead,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: nn.Module,
    args: argparse.Namespace,
    hparams: Dict[str, Any],
    grad_accum: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    current_step: int,
    total_steps: int,
) -> Tuple[torch.Tensor, int]:
    model.train()
    labels = batch["labels"].to(device)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = forward_pass(model, batch, device)
        if isinstance(loss_fn, AdaptiveFocalLoss):
            loss_fn.update_progress(current_step / max(1, total_steps))
        loss = loss_fn(logits, labels)
        if hparams["regularization_type"] == "rdrop":
            logits2 = forward_pass(model, batch, device)
            loss2 = loss_fn(logits2, labels)
            if args.problem_type == "multiclass":
                log_probs1 = F.log_softmax(logits, dim=-1)
                log_probs2 = F.log_softmax(logits2, dim=-1)
            else:
                log_probs1 = F.logsigmoid(logits)
                log_probs2 = F.logsigmoid(logits2)
            kl = F.kl_div(log_probs1, log_probs2, reduction="batchmean") + F.kl_div(
                log_probs2, log_probs1, reduction="batchmean"
            )
            loss = 0.5 * (loss + loss2) + hparams["rdrop_alpha"] * 0.5 * kl
    loss = loss / grad_accum
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return loss.detach(), current_step + 1


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_grad_norm: float,
) -> None:
    if scaler is not None:
        scaler.unscale_(optimizer)
    if max_grad_norm and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(optimizer.parameters(), max_grad_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def evaluate_model(
    model: BertWithHead,
    data_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            logits = forward_pass(model, batch, device)
            labels = batch["labels"]
            if args.problem_type == "multiclass":
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                logits_list.append(probs)
                labels_list.append(labels.numpy())
            else:
                probs = torch.sigmoid(logits).cpu().numpy()
                logits_list.append(probs)
                labels_list.append(labels.numpy())
    probs = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    metrics_dict = compute_metrics_from_outputs(labels, probs, args, threshold)
    metrics_dict["probs"] = probs
    metrics_dict["labels"] = labels
    return metrics_dict


def run_training_loop(
    args: argparse.Namespace,
    hparams: Dict[str, Any],
    trial: Optional[Trial] = None,
) -> Tuple[float, Dict[str, Any]]:
    device = GLOBAL_STATE["device"]
    tokenizer = GLOBAL_STATE["tokenizer"]
    train_loader, valid_loader = prepare_dataloaders(tokenizer, args, hparams)
    model = BertWithHead(args.model_name, args.num_labels, args.problem_type, hparams).to(device)
    apply_peft(model, hparams)
    if hparams["regularization_type"] == "mixout":
        apply_mixout_to_head(model, hparams.get("mixout_p", 0.0))
    optimizer = build_optimizer(model, hparams)
    steps_per_epoch = math.ceil(len(train_loader.dataset) / hparams["train_batch_size"])
    total_steps = max(1, (steps_per_epoch * hparams["num_epochs"]) // hparams["gradient_accumulation_steps"])
    total_steps = max(total_steps, 1)
    scheduler = build_scheduler(optimizer, hparams, total_steps)
    loss_fn = build_loss(hparams, args, GLOBAL_STATE["class_stats"], device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    global_step = 0
    for epoch in range(hparams["num_epochs"]):
        for step, batch in enumerate(train_loader):
            loss, global_step = training_step(
                model,
                batch,
                device,
                loss_fn,
                args,
                hparams,
                hparams["gradient_accumulation_steps"],
                scaler,
                optimizer,
                scheduler,
                global_step,
                total_steps,
            )
            if (step + 1) % hparams["gradient_accumulation_steps"] == 0:
                optimizer_step(optimizer, scheduler, scaler, hparams["max_grad_norm"])
        if trial is not None:
            interim_threshold = hparams.get("decision_threshold", 0.5)
            metrics_dict = evaluate_model(model, valid_loader, device, args, interim_threshold)
            trial.report(metrics_dict["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    eval_threshold = hparams.get("decision_threshold", 0.5)
    metrics_dict = evaluate_model(model, valid_loader, device, args, eval_threshold)
    if args.problem_type != "multiclass" and hparams["threshold_strategy"] == "opt_on_val":
        eval_threshold = find_best_threshold(
            metrics_dict["probs"],
            metrics_dict["labels"],
            metric_name="macro_f1",
            problem_type=args.problem_type,
        )
        metrics_dict = evaluate_model(model, valid_loader, device, args, eval_threshold)
    metrics_dict["threshold"] = eval_threshold
    metrics_dict.pop("probs", None)
    metrics_dict.pop("labels", None)
    return metrics_dict["macro_f1"], {"model": model, "tokenizer": tokenizer, "metrics": metrics_dict, "threshold": eval_threshold}


# ----------------------------
# Optuna objective
# ----------------------------


def ensure_global_state(args: argparse.Namespace) -> None:
    if GLOBAL_STATE["initialized"]:
        return
    set_seed(args.seed)
    train_records, valid_records = load_datasets(args)
    GLOBAL_STATE["train_records"] = train_records
    GLOBAL_STATE["valid_records"] = valid_records
    GLOBAL_STATE["class_stats"] = compute_class_stats(train_records, args)
    GLOBAL_STATE["tokenizer"] = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    GLOBAL_STATE["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_STATE["args"] = args
    GLOBAL_STATE["initialized"] = True


def objective(trial: Trial) -> float:
    args = GLOBAL_STATE["args"]
    hparams = sample_hparams(trial)
    metric, _ = run_training_loop(args, hparams, trial=trial)
    return metric


# ----------------------------
# Optuna Study setup
# ----------------------------


def create_or_load_study(args: argparse.Namespace) -> optuna.Study:
    if args.sampler == "tpe":
        sampler = TPESampler(seed=args.seed)
    else:
        sampler = GridSampler(GRID_SPACE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    return study


# ----------------------------
# Resource-aware worker manager
# ----------------------------


def parse_gpu_list(args: argparse.Namespace) -> List[Optional[int]]:
    if args.gpus.strip():
        return [int(x) for x in args.gpus.split(",")]
    if not torch.cuda.is_available():
        return [None]
    return list(range(torch.cuda.device_count()))


def get_available_gpus(candidate_gpus: List[Optional[int]], min_free_gb: float) -> List[Optional[int]]:
    if not torch.cuda.is_available():
        return [None]
    available: List[Optional[int]] = []
    for gpu_id in candidate_gpus:
        if gpu_id is None:
            available.append(None)
            continue
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            free_gb = free_bytes / (1024 ** 3)
            if free_gb >= min_free_gb:
                available.append(gpu_id)
        except RuntimeError:
            continue
    return available


def worker_entry(
    args: argparse.Namespace,
    study_name: str,
    storage_url: str,
    trials_per_worker: int,
    timeout: Optional[int],
    gpu_id: Optional[int],
):
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ensure_global_state(args)
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        study.optimize(objective, n_trials=trials_per_worker, timeout=timeout, n_jobs=1, gc_after_trial=True)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Worker failed on GPU %s: %s", gpu_id, exc)


@dataclass
class WorkerHandle:
    process: mp.Process
    gpu_id: Optional[int]


class HPOManager:
    def __init__(self, study: optuna.Study, args: argparse.Namespace):
        self.study = study
        self.args = args
        self.workers: List[WorkerHandle] = []
        self.candidate_gpus = parse_gpu_list(args)

    def _cleanup_workers(self) -> None:
        alive_workers: List[WorkerHandle] = []
        for handle in self.workers:
            if handle.process.is_alive():
                alive_workers.append(handle)
            else:
                handle.process.join()
        self.workers = alive_workers

    def _completed_trials(self) -> int:
        return len([t for t in self.study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE])

    def run(self) -> None:
        start_time = time.time()
        trials_target = self.args.n_trials
        while True:
            self._cleanup_workers()
            completed = self._completed_trials()
            if completed >= trials_target:
                LOGGER.info("Completed %s trials; stopping manager", completed)
                break
            if self.args.timeout and (time.time() - start_time) > self.args.timeout:
                LOGGER.info("Timeout reached; stopping manager")
                break
            available_gpus = get_available_gpus(self.candidate_gpus, self.args.min_free_mem_gb)
            if psutil is not None:
                LOGGER.info(
                    "System usage: CPU %.1f%% | RAM %.1f%%",
                    psutil.cpu_percent(interval=None),
                    psutil.virtual_memory().percent,
                )
            if len(self.workers) >= self.args.max_concurrent or not available_gpus:
                time.sleep(5)
                continue
            remaining = trials_target - completed
            gpu_id = available_gpus[0]
            trials_per_worker = min(1, remaining)
            process = mp.Process(
                target=worker_entry,
                args=(self.args, self.study.study_name, self.args.storage_url, trials_per_worker, self.args.timeout, gpu_id),
                daemon=True,
            )
            process.start()
            self.workers.append(WorkerHandle(process=process, gpu_id=gpu_id))
            LOGGER.info("Launched worker on GPU %s (pid=%s)", gpu_id, process.pid)
            time.sleep(2)
        for handle in self.workers:
            handle.process.join()
        self.workers.clear()


# ----------------------------
# Final model training & saving
# ----------------------------


def retrain_best_model(
    args: argparse.Namespace,
    hparams: Dict[str, Any],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Use combined train + validation for final training
    combined_records = GLOBAL_STATE["train_records"] + GLOBAL_STATE["valid_records"]
    GLOBAL_STATE["train_records"] = combined_records
    GLOBAL_STATE["valid_records"] = GLOBAL_STATE["valid_records"]
    metric, artifacts = run_training_loop(args, hparams, trial=None)
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    tokenizer.save_pretrained(output_dir)
    model.encoder.save_pretrained(os.path.join(output_dir, "encoder"))
    torch.save(model.state_dict(), os.path.join(output_dir, "classifier.pt"))
    with open(os.path.join(output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts["metrics"], f, indent=2)
    LOGGER.info("Final model trained with macro F1 %.4f; saved to %s", metric, output_dir)


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "hpo.log")),
        ],
    )
    LOGGER.info("Starting HPO with args: %s", vars(args))
    ensure_global_state(args)
    study = create_or_load_study(args)
    manager = HPOManager(study, args)
    mp.set_start_method("spawn", force=True)
    manager.run()
    if not study.best_trials:
        LOGGER.warning("No completed trials; exiting")
        return
    best_trial = study.best_trial
    best_params = best_trial.params
    with open(os.path.join(args.output_dir, "best_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    LOGGER.info("Best trial score %.4f with params %s", best_trial.value, best_params)
    retrain_best_model(args, best_params, args.output_dir)


if __name__ == "__main__":
    main()
