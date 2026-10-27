"""Utility functions for Phase 1."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Optional
import json


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, backbone_name: str):
    for param in model.parameters():
        param.requires_grad = False
    print(f"Frozen {backbone_name}")


def unfreeze_top_layers(model: nn.Module, backbone_name: str, top_n: int):
    if backbone_name == "biobert":
        for name, param in model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(12 - top_n, 12)):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif backbone_name == "vit":
        for name, param in model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(12 - top_n, 12)):
                param.requires_grad = True
            else:
                param.requires_grad = False
    print(f"Unfrozen top {top_n} layers of {backbone_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    print(f"\n{prefix}Metrics:")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k.capitalize():12s}: {v:.4f}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    is_best: bool = False
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"  Best model saved to {best_path}")


def load_checkpoint(model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop