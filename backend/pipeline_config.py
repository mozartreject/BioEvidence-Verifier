from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_path(name: str, default: str) -> Path:
    value = os.environ.get(name, default)
    return Path(value).expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class Phase1Config:
    root: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE1_ROOT",
            "~/Downloads/kaggle_upload",
        )
    )

    checkpoint: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE1_CHECKPOINT",
            "~/Downloads/BiomedicalVerifier/models/phase1_converted.pt",
        )
    )

    config_yaml: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE1_CONFIG",
            "~/Downloads/kaggle_upload/config.yaml",
        )
    )

    export_embeddings: bool = False


@dataclass(frozen=True)
class Phase2Config:
    live_pubmed: bool = field(default_factory=lambda: os.environ.get("BIOVERIFY_LIVE_PUBMED", "1") == "1")
    live_pubmed_limit: int = field(default_factory=lambda: _env_int("BIOVERIFY_PUBMED_LIMIT", 12))
    final_candidate_k: int = field(default_factory=lambda: _env_int("BIOVERIFY_FINAL_EVIDENCE_K", 10))
    pubmed_cache_dir: Path = field(default_factory=lambda: _env_path("BIOVERIFY_PUBMED_CACHE", "~/Downloads/BiomedicalVerifier/data/pubmed_search_cache"))

    index_dir: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE2_INDEX_DIR",
            "~/Downloads/phase2/data/faiss_index_35k",
        )
    )

    root: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE2_ROOT",
            "~/Downloads/phase2",
        )
    )

    config_py: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE2_CONFIG",
            "~/Downloads/phase2/config.py",
        )
    )

    react_prompt: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE2_REACT_PROMPT",
            "~/Downloads/phase2/prompts/react_system_prompt.txt",
        )
    )

    bm25_candidate_k: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_BM25_K",
            50,
        )
    )

    crossencoder_top_k: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_CE_TOP_K",
            10,
        )
    )

    max_react_hops: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_REACT_MAX_HOPS",
            5,
        )
    )


@dataclass(frozen=True)
class Phase3Config:
    root: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE3_ROOT",
            "~/Downloads/Phase3-main",
        )
    )

    config_py: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE3_CONFIG",
            "~/Downloads/Phase3-main/config.py",
        )
    )

    top_k: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_PHASE3_TOP_K",
            3,
        )
    )

    ollama_model: str = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_PHASE3_OLLAMA_MODEL",
            "qwen2.5:7b",
        )
    )

    ollama_chat_url: str = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_PHASE3_OLLAMA_URL",
            "http://localhost:11434/api/chat",
        )
    )


@dataclass(frozen=True)
class Phase4Config:
    backend: str = field(default_factory=lambda: os.environ.get("BIOVERIFY_VERIFIER", "biomedical_nli"))
    nli_model: str = "pritamdeka/PubMedBERT-MNLI-MedNLI"

    root: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE4_ROOT",
            "~/Downloads/Phase4-main",
        )
    )

    checkpoint: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE4_CHECKPOINT",
            "~/Downloads/Phase4-main/results/"
            "e2e_retrieval_conditioned_final/checkpoint/"
            "best_retrieval_conditioned_verifier.pt",
        )
    )

    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_PHASE4_MODEL",
            "microsoft/BiomedNLP-"
            "PubMedBERT-base-uncased-abstract-fulltext",
        )
    )

    max_length: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_PHASE4_MAX_LENGTH",
            384,
        )
    )

    evidence_top_k: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_PHASE4_EVIDENCE_TOP_K",
            3,
        )
    )

    label_refuted: int = 0
    label_nei: int = 1
    label_supported: int = 2

    confidence_name: str = "verifier_score"


@dataclass(frozen=True)
class Phase5Config:
    root: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE5_ROOT",
            "~/Downloads/Phase5-main",
        )
    )

    config_yaml: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PHASE5_CONFIG",
            "~/Downloads/Phase5-main/config.yaml",
        )
    )

    ollama_model: str = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_PHASE5_OLLAMA_MODEL",
            "qwen2.5:7b",
        )
    )

    ollama_chat_url: str = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_PHASE5_OLLAMA_URL",
            "http://localhost:11434/api/chat",
        )
    )

    timeout_seconds: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_PHASE5_TIMEOUT",
            180,
        )
    )

    request_retries: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_PHASE5_RETRIES",
            3,
        )
    )

    repair_attempts: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_PHASE5_REPAIR_ATTEMPTS",
            2,
        )
    )

    semantic_threshold: float = field(
        default_factory=lambda: _env_float(
            "BIOVERIFY_PHASE5_SEMANTIC_THRESHOLD",
            0.45,
        )
    )

    coverage_threshold: float = field(
        default_factory=lambda: _env_float(
            "BIOVERIFY_PHASE5_COVERAGE_THRESHOLD",
            0.45,
        )
    )


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_PROJECT_ROOT",
            "~/Downloads/BiomedicalVerifier",
        )
    )

    outputs_dir: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_OUTPUTS_DIR",
            "~/Downloads/BiomedicalVerifier/outputs",
        )
    )

    audit_dir: Path = field(
        default_factory=lambda: _env_path(
            "BIOVERIFY_AUDIT_DIR",
            "~/Downloads/BiomedicalVerifier/outputs/audit",
        )
    )

    seed: int = field(
        default_factory=lambda: _env_int(
            "BIOVERIFY_SEED",
            42,
        )
    )

    pipeline_version: str = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_PIPELINE_VERSION",
            "0.2.0-claim-anchored-retrieval",
        )
    )

    device_override: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "BIOVERIFY_DEVICE"
        )
    )

    strict_mode: bool = True
    save_audit_json: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    phase1: Phase1Config = field(
        default_factory=Phase1Config
    )

    phase2: Phase2Config = field(
        default_factory=Phase2Config
    )

    phase3: Phase3Config = field(
        default_factory=Phase3Config
    )

    phase4: Phase4Config = field(
        default_factory=Phase4Config
    )

    phase5: Phase5Config = field(
        default_factory=Phase5Config
    )

    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig
    )


def load_pipeline_config() -> PipelineConfig:
    return PipelineConfig()


def ensure_runtime_directories(
    config: PipelineConfig,
) -> None:
    config.runtime.outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.runtime.audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_static_paths(
    config: PipelineConfig,
) -> list[str]:
    checks = {
        "Phase 1 root":
            config.phase1.root,

        "Phase 1 checkpoint":
            config.phase1.checkpoint,

        "Phase 1 config":
            config.phase1.config_yaml,

        "Phase 2 root":
            config.phase2.root,

        "Phase 2 config":
            config.phase2.config_py,

        "Phase 2 ReAct prompt":
            config.phase2.react_prompt,

        "Phase 3 root":
            config.phase3.root,

        "Phase 3 config":
            config.phase3.config_py,

        "Phase 4 root":
            config.phase4.root,

        "Phase 4 checkpoint":
            config.phase4.checkpoint,

        "Phase 5 root":
            config.phase5.root,

        "Phase 5 config":
            config.phase5.config_yaml,
    }

    failures: list[str] = []

    for label, path in checks.items():
        if not path.exists():
            failures.append(
                f"{label} missing: {path}"
            )

    return failures