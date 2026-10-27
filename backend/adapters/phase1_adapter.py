from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import BertTokenizer

from backend.pipeline_config import Phase1Config, RuntimeConfig
from backend.schemas import (
    InferenceRequest,
    Modality,
    ModelArtifact,
    Phase1Result,
    PhaseError,
    PhaseName,
    PhaseStatus,
    PhaseTiming,
)


class Phase1AdapterError(RuntimeError):
    pass


def _load_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise Phase1AdapterError(f"Required Phase 1 source file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise Phase1AdapterError(f"Could not create import spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _choose_device(override: Optional[str] = None) -> torch.device:
    if override:
        requested = override.lower().strip()

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise Phase1AdapterError(
                    "BIOVERIFY_DEVICE=cuda was requested but CUDA is unavailable."
                )
            return torch.device("cuda")

        if requested == "mps":
            if not (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                raise Phase1AdapterError(
                    "BIOVERIFY_DEVICE=mps was requested but Apple MPS is unavailable."
                )
            return torch.device("mps")

        if requested == "cpu":
            return torch.device("cpu")

        raise Phase1AdapterError(
            f"Unsupported BIOVERIFY_DEVICE value: {override!r}. "
            "Use 'cpu', 'mps', or 'cuda'."
        )

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class Phase1Adapter:
    """
    Faithful adapter around the audited Phase 1 ContrastiveFusion model.

    Important research constraint:
    The original binary training path always supplied a real image tensor.
    Although the model contains `no_image_token`, the audited training forward
    path does not use that token. Therefore text-only fusion is supported here
    as a transparent diagnostic path, but is explicitly marked as unvalidated
    and Phase 1 is not allowed to determine the final Phase 4 verdict.
    """

    def __init__(
        self,
        phase1_config: Phase1Config,
        runtime_config: RuntimeConfig,
    ) -> None:
        self.cfg = phase1_config
        self.runtime = runtime_config

        self.device = _choose_device(runtime_config.device_override)

        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[BertTokenizer] = None
        self.image_transform = None

        self._checkpoint: Optional[dict[str, Any]] = None
        self._checkpoint_epoch: Optional[int] = None
        self._checkpoint_metrics: dict[str, Any] = {}

        self._threshold: float = 0.5
        self._loaded = False

        self._phase1_yaml: dict[str, Any] = {}
        self._model_settings: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        return self._threshold

    def load(self) -> None:
        if self._loaded:
            return

        if not self.cfg.root.exists():
            raise Phase1AdapterError(
                f"Phase 1 root does not exist: {self.cfg.root}"
            )

        if not self.cfg.checkpoint.exists():
            raise Phase1AdapterError(
                f"Phase 1 checkpoint does not exist: {self.cfg.checkpoint}"
            )

        if not self.cfg.config_yaml.exists():
            raise Phase1AdapterError(
                f"Phase 1 config does not exist: {self.cfg.config_yaml}"
            )

        with self.cfg.config_yaml.open("r", encoding="utf-8") as handle:
            loaded_yaml = yaml.safe_load(handle)

        if not isinstance(loaded_yaml, dict):
            raise Phase1AdapterError(
                f"Phase 1 YAML did not parse to a dictionary: "
                f"{self.cfg.config_yaml}"
            )

        self._phase1_yaml = loaded_yaml
        self._model_settings = dict(self._phase1_yaml.get("model", {}))

        required_model_keys = {
            "biobert",
            "vit",
            "fusion_type",
            "hidden_dim",
            "dropout",
        }
        missing = sorted(required_model_keys - set(self._model_settings))
        if missing:
            raise Phase1AdapterError(
                "Phase 1 config is missing model fields: "
                + ", ".join(missing)
            )

        models_module = _load_module_from_path(
            "bioverify_phase1_models",
            self.cfg.root / "src" / "models.py",
        )
        transforms_module = _load_module_from_path(
            "bioverify_phase1_transforms",
            self.cfg.root / "src" / "transforms.py",
        )

        ContrastiveFusion = getattr(models_module, "ContrastiveFusion", None)
        if ContrastiveFusion is None:
            raise Phase1AdapterError(
                "ContrastiveFusion was not found in Phase 1 src/models.py."
            )

        get_val_transforms = getattr(
            transforms_module,
            "get_val_transforms",
            None,
        )
        if get_val_transforms is None:
            raise Phase1AdapterError(
                "get_val_transforms was not found in Phase 1 "
                "src/transforms.py."
            )

        self.tokenizer = BertTokenizer.from_pretrained(
            self._model_settings["biobert"]
        )

        self.model = ContrastiveFusion(
            biobert_name=self._model_settings["biobert"],
            vit_name=self._model_settings["vit"],
            fusion_type=self._model_settings["fusion_type"],
            hidden_dim=int(self._model_settings["hidden_dim"]),
            dropout=float(self._model_settings["dropout"]),
        )

        try:
            checkpoint = torch.load(
                self.cfg.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch versions that do not expose
            # the weights_only argument.
            checkpoint = torch.load(
                self.cfg.checkpoint,
                map_location="cpu",
            )

        if not isinstance(checkpoint, dict):
            raise Phase1AdapterError(
                "Phase 1 checkpoint must be a dictionary checkpoint."
            )

        if "model_state_dict" not in checkpoint:
            raise Phase1AdapterError(
                "Phase 1 checkpoint is missing 'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self._checkpoint = checkpoint
        self._checkpoint_epoch = checkpoint.get("epoch")

        metrics = checkpoint.get("metrics", {})
        if isinstance(metrics, dict):
            self._checkpoint_metrics = metrics
            threshold = metrics.get("threshold")
            if threshold is not None:
                self._threshold = float(threshold)

        if not 0.0 <= self._threshold <= 1.0:
            raise Phase1AdapterError(
                f"Invalid binary threshold in checkpoint: {self._threshold}"
            )

        self.image_transform = get_val_transforms()

        self.model.to(self.device)
        self.model.eval()

        self._loaded = True

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise Phase1AdapterError(
                "Phase1Adapter is not loaded. Call load() first."
            )

        if self.model is None:
            raise Phase1AdapterError("Phase 1 model is unexpectedly None.")

        if self.tokenizer is None:
            raise Phase1AdapterError(
                "Phase 1 tokenizer is unexpectedly None."
            )

        if self.image_transform is None:
            raise Phase1AdapterError(
                "Phase 1 image transform is unexpectedly None."
            )

    def _tokenize_claim(
        self,
        claim: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_loaded()
        assert self.tokenizer is not None

        encoded = self.tokenizer(
            claim,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        return input_ids, attention_mask

    def _prepare_image(self, image_path: Path) -> torch.Tensor:
        self._require_loaded()
        assert self.image_transform is not None

        if not image_path.exists():
            raise Phase1AdapterError(
                f"Image path does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise Phase1AdapterError(
                f"Image path is not a file: {image_path}"
            )

        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                tensor = self.image_transform(rgb)
        except Exception as exc:
            raise Phase1AdapterError(
                f"Could not read/process image {image_path}: {exc}"
            ) from exc

        if tensor.ndim != 3 or tuple(tensor.shape) != (3, 224, 224):
            raise Phase1AdapterError(
                "Phase 1 image transform returned unexpected shape: "
                f"{tuple(tensor.shape)}; expected (3, 224, 224)."
            )

        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def _to_vector(tensor: torch.Tensor) -> list[float]:
        return (
            tensor.detach()
            .to("cpu", dtype=torch.float32)
            .squeeze(0)
            .tolist()
        )

    def _model_artifact(self) -> ModelArtifact:
        checkpoint_sha256 = _file_sha256(self.cfg.checkpoint)

        return ModelArtifact(
            component="phase1_multimodal_encoder",
            model_name=(
                f"{self._model_settings.get('biobert')} + "
                f"{self._model_settings.get('vit')}"
            ),
            checkpoint_path=str(self.cfg.checkpoint),
            checkpoint_epoch=self._checkpoint_epoch,
            version=None,
            metadata={
                "fusion_type": self._model_settings.get("fusion_type"),
                "hidden_dim": int(
                    self._model_settings.get("hidden_dim", 768)
                ),
                "binary_threshold": self._threshold,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_metrics": self._checkpoint_metrics,
                "device": str(self.device),
                "binary_label_mapping": {
                    "0": "REAL",
                    "1": "FAKE",
                },
                "phase1_affects_phase4_verdict": False,
            },
        )

    def infer(self, request: InferenceRequest) -> Phase1Result:
        self._require_loaded()
        assert self.model is not None

        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()

        warnings: list[str] = []

        try:
            input_ids, attention_mask = self._tokenize_claim(
                request.claim
            )

            with torch.inference_mode():
                text_emb = self.model.encode_text(
                    input_ids,
                    attention_mask,
                )
                text_emb = F.normalize(text_emb, dim=-1)

                if request.image_path:
                    image_path = Path(
                        request.image_path
                    ).expanduser().resolve()

                    pixel_values = self._prepare_image(image_path)

                    image_emb = self.model.encode_image(pixel_values)
                    image_emb = F.normalize(image_emb, dim=-1)

                    fused_emb = self.model.fuse(
                        text_emb,
                        image_emb,
                    )
                    fused_emb = F.normalize(fused_emb, dim=-1)

                    logits = self.model.classifier(
                        fused_emb
                    ).squeeze(-1)

                    probability = float(
                        torch.sigmoid(logits)
                        .detach()
                        .cpu()
                        .item()
                    )

                    prediction = int(
                        probability > self._threshold
                    )

                    auxiliary_label = (
                        "FAKE" if prediction == 1 else "REAL"
                    )

                    has_image = True
                    modality = Modality.TEXT_IMAGE

                else:
                    # The audited model defines `no_image_token`, but the
                    # audited binary training path never passes it through
                    # forward(). We still expose a transparent diagnostic
                    # missing-modality representation, but it must not be
                    # presented as a validated binary decision.
                    no_image = self.model.no_image_token
                    no_image = no_image.unsqueeze(0).expand(
                        text_emb.shape[0],
                        -1,
                    )
                    no_image = F.normalize(no_image, dim=-1)

                    fused_emb = self.model.fuse(
                        text_emb,
                        no_image,
                    )
                    fused_emb = F.normalize(fused_emb, dim=-1)

                    image_emb = no_image

                    probability = None
                    prediction = None
                    auxiliary_label = None

                    has_image = False
                    modality = Modality.TEXT_ONLY

                    warnings.append(
                        "Phase 1 text-only path used the model's "
                        "no_image_token to create a diagnostic fused "
                        "representation. The audited binary training "
                        "forward path always used real images, so no "
                        "Phase 1 binary probability/prediction is "
                        "reported for text-only input."
                    )

            export_embeddings = bool(self.cfg.export_embeddings)

            result = Phase1Result(
                status=PhaseStatus.SUCCESS,
                has_image=has_image,
                modality=modality,
                text_embedding_dim=int(text_emb.shape[-1]),
                image_embedding_dim=int(image_emb.shape[-1]),
                fused_embedding_dim=int(fused_emb.shape[-1]),
                text_embedding=(
                    self._to_vector(text_emb)
                    if export_embeddings
                    else None
                ),
                image_embedding=(
                    self._to_vector(image_emb)
                    if export_embeddings
                    else None
                ),
                fused_embedding=(
                    self._to_vector(fused_emb)
                    if export_embeddings
                    else None
                ),
                auxiliary_probability=probability,
                auxiliary_prediction=prediction,
                auxiliary_threshold=self._threshold,
                auxiliary_label=auxiliary_label,
                model=self._model_artifact(),
                warnings=warnings,
            )

        except Exception as exc:
            ended_at = datetime.now(timezone.utc)
            latency_ms = (
                time.perf_counter() - started_clock
            ) * 1000.0

            if isinstance(exc, Phase1AdapterError):
                message = str(exc)
                error_type = exc.__class__.__name__
            else:
                message = str(exc)
                error_type = exc.__class__.__name__

            return Phase1Result(
                status=PhaseStatus.FAILED,
                has_image=bool(request.image_path),
                modality=(
                    Modality.TEXT_IMAGE
                    if request.image_path
                    else Modality.TEXT_ONLY
                ),
                model=(
                    self._model_artifact()
                    if self._loaded
                    else None
                ),
                warnings=warnings,
                error=PhaseError(
                    phase=PhaseName.PHASE1,
                    error_type=error_type,
                    message=message,
                    recoverable=False,
                    details={
                        "request_id": request.request_id,
                    },
                ),
                timing=PhaseTiming(
                    phase=PhaseName.PHASE1,
                    started_at_utc=started_at,
                    ended_at_utc=ended_at,
                    latency_ms=latency_ms,
                ),
            )

        ended_at = datetime.now(timezone.utc)
        latency_ms = (
            time.perf_counter() - started_clock
        ) * 1000.0

        result.timing = PhaseTiming(
            phase=PhaseName.PHASE1,
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            latency_ms=latency_ms,
        )

        return result
