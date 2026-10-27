"""Configuration loader for Phase 1."""
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    image_root: str = "data/images"
    honest_ooc: str = "data/raw/honest_ooc_clean_long.csv"
    fakehealth: str = "data/raw/fakehealth_content.csv"
    recovery: str = "data/raw/recovery-news-data.csv"
    honest_ooc_cap_per_label: Optional[int] = 1000
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42


@dataclass
class ModelConfig:
    biobert: str = "dmis-lab/biobert-base-cased-v1.1"
    vit: str = "google/vit-base-patch16-224"
    fusion_type: str = "concat"
    hidden_dim: int = 768
    dropout: float = 0.1
    freeze_biobert: bool = True
    freeze_vit: bool = True
    unfreeze_biobert_top_n: int = 4
    unfreeze_vit_top_n: int = 2


@dataclass
class TrainingConfig:
    mode: str = "binary"
    epochs: int = 15
    batch_size: int = 16
    accum_steps: int = 1
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    contrastive_margin: float = 0.5
    temperature: float = 0.07
    use_augmentation: bool = True
    use_amp: bool = True


@dataclass
class EvalConfig:
    threshold_search_min: float = 0.1
    threshold_search_max: float = 0.9
    threshold_search_step: float = 0.01


@dataclass
class SystemConfig:
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42


@dataclass
class LoggingConfig:
    log_dir: str = "checkpoints/logs"
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    log_interval: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> Config:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using defaults.")
        return Config()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
        system=SystemConfig(**raw.get("system", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )