from __future__ import annotations

import hashlib
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from backend.pipeline_config import PipelineConfig
from backend.schemas import (
    ClassProbabilities,
    EvidenceRecord,
    ModelArtifact,
    PairwiseVerification,
    Phase4Result,
    PhaseStatus,
    PhaseTiming,
    Verdict,
    verdict_to_user_facing,
)


# ============================================================
# CONSTANTS
# ============================================================

LABEL_ID_TO_VERDICT = {
    0: Verdict.REFUTED,
    1: Verdict.NEI,
    2: Verdict.SUPPORTED,
}

MAX_LENGTH = 384

EXPECTED_STATE_DICT_ENTRIES = 203

EXPECTED_MODEL_NAME = (
    "microsoft/"
    "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class Phase4AdapterError(RuntimeError):
    pass


# ============================================================
# MODEL
# ============================================================

class PubMedBERTVerifier(nn.Module):
    """
    PubMedBERT evidence verifier.

    Architecture shared by the supported Phase 4 checkpoints:

        PubMedBERT
            ->
        CLS representation (768)
            ->
        Linear(768, 384)
            ->
        GELU
            ->
        Dropout
            ->
        Linear(384, 3)

    Output classes:

        0 = REFUTED
        1 = NEI
        2 = SUPPORTED
    """

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        if hidden != 768:
            raise Phase4AdapterError(
                "Unexpected PubMedBERT hidden size: "
                f"{hidden}; expected 768."
            )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden,
                384,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                384,
                3,
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[
            :,
            0,
            :
        ]

        return self.classifier(
            cls
        )


# ============================================================
# UTILITIES
# ============================================================

def utc_now_string() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def choose_device() -> torch.device:
    requested = (
        os.environ
        .get(
            "BIOVERIFY_DEVICE",
            "",
        )
        .strip()
        .lower()
    )

    if requested:
        if requested == "cpu":
            return torch.device(
                "cpu"
            )

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise Phase4AdapterError(
                    "BIOVERIFY_DEVICE=cuda was requested "
                    "but CUDA is unavailable."
                )

            return torch.device(
                "cuda"
            )

        if requested == "mps":
            if not (
                hasattr(
                    torch.backends,
                    "mps",
                )
                and torch.backends.mps.is_available()
            ):
                raise Phase4AdapterError(
                    "BIOVERIFY_DEVICE=mps was requested "
                    "but MPS is unavailable."
                )

            return torch.device(
                "mps"
            )

        raise Phase4AdapterError(
            "Unsupported BIOVERIFY_DEVICE value: "
            f"{requested!r}"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# PHASE 4 ADAPTER
# ============================================================

class Phase4Adapter:
    """
    Phase 4 evidence-grounded verifier.

    Supported checkpoint contracts:

    1. balanced_retrieval_curriculum

       Required metadata:
           epoch
           model_name
           model_state_dict

    2. retrieval_conditioned

       Required metadata:
           epoch
           model_state_dict
           validation_macro_f1
           base_checkpoint
           hard_negative_count
           training_configuration

    Both checkpoint families share the same model architecture.

    Pipeline:

        Phase 3 ranked evidence
                |
                v
        PubMedBERT(claim, evidence)
                |
                v
        pair-level softmax probabilities
                |
                v
        aggregation weight =
            relevance_score * hierarchy_weight
                |
                v
        normalized weighted mean
                |
                v
        REFUTED / NEI / SUPPORTED

    Phase 3 already stores:

        combined_score =
            relevance_score * hierarchy_weight

    Therefore Phase 4 reconstructs the weight from the primitive
    values and does NOT multiply combined_score by hierarchy_weight
    again.

    This prevents hierarchy double-weighting.
    """

    def __init__(
        self,
        config: Optional[
            PipelineConfig
        ] = None,
    ):
        self.config = (
            config
            if config is not None
            else PipelineConfig()
        )

        self.device = (
            choose_device()
        )

        self.model = None
        self.tokenizer = None
        self.checkpoint = None

        self.checkpoint_path: Optional[
            Path
        ] = None

        self.checkpoint_sha256: Optional[
            str
        ] = None

        self.model_name: Optional[
            str
        ] = None

        self.dropout: Optional[
            float
        ] = None

        self.epoch: Optional[
            int
        ] = None

        self.checkpoint_contract: Optional[
            str
        ] = None

        self.loaded = False

    # ========================================================
    # CHECKPOINT PATH
    # ========================================================

    def _resolve_checkpoint_path(
        self,
    ) -> Path:
        phase4_config = (
            self.config.phase4
        )

        raw_path = getattr(
            phase4_config,
            "checkpoint",
            None,
        )

        if raw_path is None:
            raw_path = getattr(
                phase4_config,
                "checkpoint_path",
                None,
            )

        if raw_path is None:
            raise Phase4AdapterError(
                "Phase 4 checkpoint path is missing "
                "from PipelineConfig."
            )

        path = (
            Path(
                raw_path
            )
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise Phase4AdapterError(
                "Phase 4 checkpoint does not exist: "
                f"{path}"
            )

        if not path.is_file():
            raise Phase4AdapterError(
                "Phase 4 checkpoint is not a file: "
                f"{path}"
            )

        return path

    # ========================================================
    # CHECKPOINT LOADING
    # ========================================================

    @staticmethod
    def _load_checkpoint(
        path: Path,
    ):
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:
            checkpoint = torch.load(
                path,
                map_location="cpu",
            )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise Phase4AdapterError(
                "Phase 4 checkpoint root "
                "must be a dictionary."
            )

        return checkpoint

    # ========================================================
    # CHECKPOINT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_checkpoint(
        checkpoint,
    ) -> str:
        if not isinstance(
            checkpoint,
            dict,
        ):
            raise Phase4AdapterError(
                "Phase 4 checkpoint must "
                "be a dictionary."
            )

        keys = set(
            checkpoint.keys()
        )

        # ====================================================
        # CHECKPOINT CONTRACT DETECTION
        # ====================================================

        legacy_required = {
            "epoch",
            "model_name",
            "model_state_dict",
        }

        retrieval_conditioned_required = {
            "epoch",
            "model_state_dict",
            "validation_macro_f1",
            "base_checkpoint",
            "hard_negative_count",
            "training_configuration",
        }

        if legacy_required.issubset(
            keys
        ):
            checkpoint_contract = (
                "balanced_retrieval_curriculum"
            )

            model_name = str(
                checkpoint[
                    "model_name"
                ]
            )

            if (
                model_name
                != EXPECTED_MODEL_NAME
            ):
                raise Phase4AdapterError(
                    "Unexpected Phase 4 "
                    "model name.\n"
                    f"Expected: "
                    f"{EXPECTED_MODEL_NAME}\n"
                    f"Received: "
                    f"{model_name}"
                )

        elif (
            retrieval_conditioned_required
            .issubset(
                keys
            )
        ):
            checkpoint_contract = (
                "retrieval_conditioned"
            )

            epoch = checkpoint[
                "epoch"
            ]

            if (
                not isinstance(
                    epoch,
                    int,
                )
                or epoch < 1
            ):
                raise Phase4AdapterError(
                    "Invalid retrieval-conditioned "
                    "checkpoint epoch."
                )

            validation_macro_f1 = (
                checkpoint[
                    "validation_macro_f1"
                ]
            )

            try:
                validation_macro_f1 = float(
                    validation_macro_f1
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise Phase4AdapterError(
                    "Invalid retrieval-conditioned "
                    "validation_macro_f1."
                ) from exc

            if (
                not math.isfinite(
                    validation_macro_f1
                )
                or not (
                    0.0
                    <= validation_macro_f1
                    <= 1.0
                )
            ):
                raise Phase4AdapterError(
                    "Retrieval-conditioned "
                    "validation_macro_f1 must "
                    "be finite and within [0, 1]."
                )

            hard_negative_count = (
                checkpoint[
                    "hard_negative_count"
                ]
            )

            if (
                not isinstance(
                    hard_negative_count,
                    int,
                )
                or hard_negative_count < 0
            ):
                raise Phase4AdapterError(
                    "Invalid retrieval-conditioned "
                    "hard_negative_count."
                )

            training_configuration = (
                checkpoint[
                    "training_configuration"
                ]
            )

            if not isinstance(
                training_configuration,
                dict,
            ):
                raise Phase4AdapterError(
                    "Retrieval-conditioned "
                    "training_configuration must "
                    "be a dictionary."
                )

            required_training_keys = {
                "learning_rate",
                "weight_decay",
                "label_smoothing",
                "class_weights",
            }

            missing_training_keys = (
                required_training_keys
                - set(
                    training_configuration.keys()
                )
            )

            if missing_training_keys:
                raise Phase4AdapterError(
                    "Retrieval-conditioned "
                    "training configuration "
                    "is missing keys: "
                    f"{sorted(missing_training_keys)}"
                )

            base_checkpoint = str(
                checkpoint[
                    "base_checkpoint"
                ]
            ).strip()

            if not base_checkpoint:
                raise Phase4AdapterError(
                    "Retrieval-conditioned "
                    "base_checkpoint is empty."
                )

        else:
            raise Phase4AdapterError(
                "Unsupported Phase 4 checkpoint "
                "contract.\n"
                f"Checkpoint keys: "
                f"{sorted(keys)}"
            )

        # ====================================================
        # SHARED ARCHITECTURE CONTRACT
        # ====================================================

        state = checkpoint[
            "model_state_dict"
        ]

        if not isinstance(
            state,
            dict,
        ):
            raise Phase4AdapterError(
                "Phase 4 model_state_dict "
                "must be a dictionary."
            )

        if (
            len(state)
            != EXPECTED_STATE_DICT_ENTRIES
        ):
            raise Phase4AdapterError(
                "Unexpected Phase 4 state "
                "dictionary size. "
                f"Expected "
                f"{EXPECTED_STATE_DICT_ENTRIES}, "
                f"received {len(state)}."
            )

        expected_parameters = {
            (
                "encoder.embeddings."
                "word_embeddings.weight"
            ): (
                30522,
                768,
            ),

            (
                "encoder.embeddings."
                "position_embeddings.weight"
            ): (
                512,
                768,
            ),

            "classifier.0.weight": (
                384,
                768,
            ),

            "classifier.0.bias": (
                384,
            ),

            "classifier.3.weight": (
                3,
                384,
            ),

            "classifier.3.bias": (
                3,
            ),
        }

        for (
            parameter_name,
            expected_shape,
        ) in expected_parameters.items():

            if parameter_name not in state:
                raise Phase4AdapterError(
                    "Checkpoint is missing "
                    "parameter: "
                    f"{parameter_name}"
                )

            tensor = state[
                parameter_name
            ]

            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise Phase4AdapterError(
                    "Checkpoint parameter is "
                    "not a tensor: "
                    f"{parameter_name}"
                )

            actual_shape = tuple(
                tensor.shape
            )

            if (
                actual_shape
                != expected_shape
            ):
                raise Phase4AdapterError(
                    "Unexpected shape for "
                    f"{parameter_name}: "
                    f"{actual_shape}; expected "
                    f"{expected_shape}."
                )

            if not torch.isfinite(
                tensor
            ).all():
                raise Phase4AdapterError(
                    "Checkpoint contains "
                    "non-finite values in "
                    "parameter: "
                    f"{parameter_name}"
                )

        return checkpoint_contract

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def load(
        self,
    ) -> None:
        if self.loaded:
            return

        print(
            "Loading Phase 4 production verifier..."
        )

        self.checkpoint_path = (
            self._resolve_checkpoint_path()
        )

        print(
            "Checkpoint:",
            self.checkpoint_path,
        )

        checkpoint = (
            self._load_checkpoint(
                self.checkpoint_path
            )
        )

        self.checkpoint_contract = (
            self._validate_checkpoint(
                checkpoint
            )
        )

        self.checkpoint = (
            checkpoint
        )

        self.epoch = int(
            checkpoint[
                "epoch"
            ]
        )

        # Retrieval-conditioned checkpoint does not
        # contain model_name because architecture was
        # inherited from its base checkpoint.
        if (
            self.checkpoint_contract
            == "retrieval_conditioned"
        ):
            self.model_name = (
                EXPECTED_MODEL_NAME
            )

        else:
            self.model_name = str(
                checkpoint[
                    "model_name"
                ]
            )

        self.dropout = float(
            checkpoint.get(
                "dropout",
                0.20,
            )
        )

        print(
            "Checkpoint contract:",
            self.checkpoint_contract,
        )

        print(
            "Epoch:",
            self.epoch,
        )

        print(
            "Model:",
            self.model_name,
        )

        print(
            "Dropout:",
            self.dropout,
        )

        if (
            self.checkpoint_contract
            == "retrieval_conditioned"
        ):
            print(
                "Validation Macro-F1:",
                checkpoint[
                    "validation_macro_f1"
                ],
            )

            print(
                "Hard negatives:",
                checkpoint[
                    "hard_negative_count"
                ],
            )

            print(
                "Base checkpoint:",
                checkpoint[
                    "base_checkpoint"
                ],
            )

        print(
            "Device:",
            self.device,
        )

        print(
            "Loading tokenizer..."
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.model_name
            )
        )

        print(
            "Instantiating exact verifier architecture..."
        )

        model = (
            PubMedBERTVerifier(
                model_name=
                    self.model_name,

                dropout=
                    self.dropout,
            )
        )

        print(
            "Strictly loading Phase 4 checkpoint..."
        )

        load_result = (
            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )
        )

        if (
            load_result.missing_keys
            or load_result.unexpected_keys
        ):
            raise Phase4AdapterError(
                "Strict checkpoint loading "
                "reported missing or unexpected keys."
            )

        model.to(
            self.device
        )

        model.eval()

        self.model = model
        self.tokenizer = tokenizer

        print(
            "Calculating checkpoint SHA-256..."
        )

        self.checkpoint_sha256 = (
            sha256_file(
                self.checkpoint_path
            )
        )

        self.loaded = True

        print(
            "Phase 4 loaded successfully."
        )

    # ========================================================
    # CLAIM VALIDATION
    # ========================================================

    @staticmethod
    def _validate_claim(
        claim: str,
    ) -> str:
        claim = str(
            claim or ""
        ).strip()

        if not claim:
            raise Phase4AdapterError(
                "Phase 4 received an empty claim."
            )

        return claim

    # ========================================================
    # EVIDENCE VALIDATION
    # ========================================================

    @staticmethod
    def _validate_evidence(
        evidence_list: List[
            EvidenceRecord
        ],
    ) -> None:
        if not evidence_list:
            raise Phase4AdapterError(
                "Phase 4 cannot run "
                "with zero evidence."
            )

        seen_ids = set()

        for evidence in evidence_list:

            if not isinstance(
                evidence,
                EvidenceRecord,
            ):
                raise Phase4AdapterError(
                    "Phase 4 expects EvidenceRecord "
                    "objects."
                )

            evidence_id = str(
                evidence.evidence_id
            ).strip()

            if not evidence_id:
                raise Phase4AdapterError(
                    "Phase 4 received evidence "
                    "with an empty ID."
                )

            if evidence_id in seen_ids:
                raise Phase4AdapterError(
                    "Duplicate evidence ID supplied "
                    "to Phase 4: "
                    f"{evidence_id}"
                )

            seen_ids.add(
                evidence_id
            )

            text = str(
                evidence.text or ""
            ).strip()

            if not text:
                raise Phase4AdapterError(
                    "Evidence text is empty for "
                    f"{evidence_id}."
                )

            if (
                evidence.relevance_score
                is None
            ):
                raise Phase4AdapterError(
                    "Missing relevance_score for "
                    f"{evidence_id}."
                )

            if (
                evidence.hierarchy_weight
                is None
            ):
                raise Phase4AdapterError(
                    "Missing hierarchy_weight for "
                    f"{evidence_id}."
                )

            relevance = float(
                evidence.relevance_score
            )

            hierarchy = float(
                evidence.hierarchy_weight
            )

            if not math.isfinite(
                relevance
            ):
                raise Phase4AdapterError(
                    "Non-finite relevance_score "
                    f"for {evidence_id}."
                )

            if not math.isfinite(
                hierarchy
            ):
                raise Phase4AdapterError(
                    "Non-finite hierarchy_weight "
                    f"for {evidence_id}."
                )

            if not (
                0.0
                <= relevance
                <= 1.0
            ):
                raise Phase4AdapterError(
                    "relevance_score must be "
                    "within [0, 1] for "
                    f"{evidence_id}."
                )

            if not (
                0.0
                <= hierarchy
                <= 1.0
            ):
                raise Phase4AdapterError(
                    "hierarchy_weight must be "
                    "within [0, 1] for "
                    f"{evidence_id}."
                )

    # ========================================================
    # PAIR SCORING
    # ========================================================

    def _score_pairs(
        self,
        claim: str,
        evidence_list: List[
            EvidenceRecord
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        claims = [
            claim
            for _ in evidence_list
        ]

        evidence_texts = [
            str(
                evidence.text
            ).strip()
            for evidence in evidence_list
        ]

        encoded = (
            self.tokenizer(
                claims,
                evidence_texts,
                truncation=True,
                padding=True,
                max_length=
                    MAX_LENGTH,
                return_tensors="pt",
            )
        )

        input_ids = (
            encoded[
                "input_ids"
            ]
            .to(
                self.device
            )
        )

        attention_mask = (
            encoded[
                "attention_mask"
            ]
            .to(
                self.device
            )
        )

        with torch.inference_mode():

            logits = self.model(
                input_ids,
                attention_mask,
            )

            if (
                logits.ndim != 2
                or logits.shape[
                    1
                ] != 3
            ):
                raise Phase4AdapterError(
                    "Expected Phase 4 logits "
                    "with shape [N, 3], received "
                    f"{tuple(logits.shape)}."
                )

            if not torch.isfinite(
                logits
            ).all():
                raise Phase4AdapterError(
                    "Phase 4 produced "
                    "non-finite logits."
                )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
            )

        if not torch.isfinite(
            probabilities
        ).all():
            raise Phase4AdapterError(
                "Phase 4 produced "
                "non-finite probabilities."
            )

        return (
            logits
            .detach()
            .cpu(),

            probabilities
            .detach()
            .cpu(),
        )

    # ========================================================
    # PROBABILITY VALIDATION
    # ========================================================

    @staticmethod
    def _make_probabilities(
        values,
    ) -> ClassProbabilities:

        if len(
            values
        ) != 3:
            raise Phase4AdapterError(
                "Expected exactly three "
                "class probabilities."
            )

        refuted = float(
            values[
                0
            ]
        )

        nei = float(
            values[
                1
            ]
        )

        supported = float(
            values[
                2
            ]
        )

        for value in (
            refuted,
            nei,
            supported,
        ):
            if not math.isfinite(
                value
            ):
                raise Phase4AdapterError(
                    "Non-finite class probability."
                )

            if not (
                0.0
                <= value
                <= 1.0
            ):
                raise Phase4AdapterError(
                    "Class probability outside "
                    "[0, 1]."
                )

        total = (
            refuted
            + nei
            + supported
        )

        if abs(
            total
            - 1.0
        ) > 1e-5:
            raise Phase4AdapterError(
                "Class probabilities do not "
                "sum to 1. "
                f"Total={total}"
            )

        return ClassProbabilities(
            refuted=refuted,
            nei=nei,
            supported=supported,
        )

    # ========================================================
    # EVIDENCE AGGREGATION
    # ========================================================

    @staticmethod
    def _aggregation_weights(
        evidence_list: List[
            EvidenceRecord
        ],
    ) -> torch.Tensor:
        """
        Deployment aggregation weight:

            relevance_score
            *
            hierarchy_weight

        Phase 2 relevance_score already corresponds to
        normalized cross-encoder relevance.

        Phase 3 combined_score already contains the hierarchy
        multiplication, so combined_score must NOT be multiplied
        by hierarchy_weight again.
        """

        raw_weights = []

        for evidence in evidence_list:

            relevance = float(
                evidence.relevance_score
            )

            hierarchy = float(
                evidence.hierarchy_weight
            )

            weight = (
                relevance
                * hierarchy
            )

            if not math.isfinite(
                weight
            ):
                raise Phase4AdapterError(
                    "Non-finite aggregation weight "
                    "for evidence "
                    f"{evidence.evidence_id}."
                )

            if weight < 0.0:
                raise Phase4AdapterError(
                    "Negative aggregation weight "
                    "for evidence "
                    f"{evidence.evidence_id}."
                )

            raw_weights.append(
                weight
            )

        weights = torch.tensor(
            raw_weights,
            dtype=torch.float64,
        )

        total = float(
            weights.sum().item()
        )

        if total <= 0.0:
            raise Phase4AdapterError(
                "Phase 4 aggregation weights "
                "sum to zero."
            )

        normalized = (
            weights
            / weights.sum()
        )

        if not torch.isfinite(
            normalized
        ).all():
            raise Phase4AdapterError(
                "Normalized aggregation weights "
                "contain non-finite values."
            )

        normalization_sum = float(
            normalized.sum().item()
        )

        if abs(
            normalization_sum
            - 1.0
        ) > 1e-10:
            raise Phase4AdapterError(
                "Normalized aggregation weights "
                "do not sum to 1."
            )

        return normalized

    # ========================================================
    # VERDICT
    # ========================================================

    @staticmethod
    def _verdict_from_probabilities(
        probabilities: ClassProbabilities,
    ) -> Verdict:

        values = [
            float(
                probabilities.refuted
            ),
            float(
                probabilities.nei
            ),
            float(
                probabilities.supported
            ),
        ]

        prediction_id = max(
            range(
                3
            ),
            key=lambda index:
                values[
                    index
                ],
        )

        return (
            LABEL_ID_TO_VERDICT[
                prediction_id
            ]
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def _selected_probability(
        probabilities: ClassProbabilities,
        verdict: Verdict,
    ) -> float:

        if (
            verdict
            == Verdict.REFUTED
        ):
            return float(
                probabilities.refuted
            )

        if (
            verdict
            == Verdict.NEI
        ):
            return float(
                probabilities.nei
            )

        if (
            verdict
            == Verdict.SUPPORTED
        ):
            return float(
                probabilities.supported
            )

        raise Phase4AdapterError(
            "Unsupported Phase 4 verdict."
        )

    # ========================================================
    # MODEL PROVENANCE
    # ========================================================

    def _build_model_artifact(
        self,
    ) -> ModelArtifact:

        if (
            self.checkpoint_path
            is None
            or self.checkpoint_sha256
            is None
            or self.model_name
            is None
            or self.epoch
            is None
            or self.checkpoint_contract
            is None
        ):
            raise Phase4AdapterError(
                "Phase 4 provenance requested "
                "before model loading completed."
            )

        metadata = {
            "checkpoint_sha256":
                self.checkpoint_sha256,

            "checkpoint_contract":
                self.checkpoint_contract,

            "architecture":
                "PubMedBERTVerifier",

            "max_length":
                MAX_LENGTH,

            "dropout":
                self.dropout,

            "state_dict_entries":
                EXPECTED_STATE_DICT_ENTRIES,

            "aggregation":
                (
                    "normalized weighted mean "
                    "of pair softmax probabilities"
                ),

            "aggregation_weight":
                (
                    "relevance_score * "
                    "hierarchy_weight"
                ),

            "label_mapping": {
                "0": "REFUTED",
                "1": "NEI",
                "2": "SUPPORTED",
            },
        }

        if (
            self.checkpoint_contract
            == "balanced_retrieval_curriculum"
        ):
            metadata[
                "training_method"
            ] = self.checkpoint.get(
                "training_method"
            )

            metadata[
                "oracle_validation_metrics"
            ] = self.checkpoint.get(
                "oracle_validation_metrics"
            )

        elif (
            self.checkpoint_contract
            == "retrieval_conditioned"
        ):
            metadata[
                "validation_macro_f1"
            ] = self.checkpoint.get(
                "validation_macro_f1"
            )

            metadata[
                "base_checkpoint"
            ] = self.checkpoint.get(
                "base_checkpoint"
            )

            metadata[
                "hard_negative_count"
            ] = self.checkpoint.get(
                "hard_negative_count"
            )

            metadata[
                "training_configuration"
            ] = self.checkpoint.get(
                "training_configuration"
            )

        return ModelArtifact(
            component=
                "phase4_verifier",

            model_name=
                self.model_name,

            checkpoint_path=
                str(
                    self.checkpoint_path
                ),

            checkpoint_epoch=
                self.epoch,

            version=
                self.checkpoint_contract,

            metadata=
                metadata,
        )

    # ========================================================
    # RUN PHASE 4
    # ========================================================

    def run(
        self,
        claim: str,
        evidence_list: List[
            EvidenceRecord
        ],
    ) -> Phase4Result:

        if not self.loaded:
            self.load()

        claim = (
            self._validate_claim(
                claim
            )
        )

        self._validate_evidence(
            evidence_list
        )

        started_at = (
            utc_now_string()
        )

        start_clock = (
            time.perf_counter()
        )

        print()
        print(
            "=" * 70
        )

        print(
            "PHASE 4 EVIDENCE-GROUNDED VERIFICATION"
        )

        print(
            "=" * 70
        )

        print(
            "Checkpoint contract:",
            self.checkpoint_contract,
        )

        print(
            "Evidence pairs:",
            len(
                evidence_list
            ),
        )

        logits, pair_probabilities = (
            self._score_pairs(
                claim,
                evidence_list,
            )
        )

        normalized_weights = (
            self._aggregation_weights(
                evidence_list
            )
        )

        # ====================================================
        # PAIRWISE DIAGNOSTIC
        # ====================================================

        print()
        print(
            "=" * 90
        )

        print(
            "PHASE 4 — PAIRWISE + AGGREGATION DIAGNOSTIC"
        )

        print(
            "=" * 90
        )

        for (
            diagnostic_index,
            diagnostic_evidence,
        ) in enumerate(
            evidence_list
        ):

            diagnostic_probs = (
                pair_probabilities[
                    diagnostic_index
                ]
            )

            diagnostic_weight = float(
                normalized_weights[
                    diagnostic_index
                ].item()
            )

            diagnostic_relevance = float(
                diagnostic_evidence
                .relevance_score
            )

            diagnostic_hierarchy = float(
                diagnostic_evidence
                .hierarchy_weight
            )

            diagnostic_raw_weight = (
                diagnostic_relevance
                * diagnostic_hierarchy
            )

            diagnostic_refuted = float(
                diagnostic_probs[
                    0
                ].item()
            )

            diagnostic_nei = float(
                diagnostic_probs[
                    1
                ].item()
            )

            diagnostic_supported = float(
                diagnostic_probs[
                    2
                ].item()
            )

            diagnostic_logits = [
                float(
                    value
                )
                for value
                in logits[
                    diagnostic_index
                ].tolist()
            ]

            print()
            print(
                "-" * 90
            )

            print(
                f"EVIDENCE "
                f"{diagnostic_index + 1}"
            )

            print(
                "-" * 90
            )

            print(
                "Evidence ID:",
                diagnostic_evidence
                .evidence_id,
            )

            print(
                "Authority tier:",
                diagnostic_evidence
                .authority_tier,
            )

            print(
                "Relevance score:",
                f"{diagnostic_relevance:.8f}",
            )

            print(
                "Hierarchy weight:",
                f"{diagnostic_hierarchy:.8f}",
            )

            print(
                "Phase 3 combined score:",
                diagnostic_evidence
                .combined_score,
            )

            print(
                "Raw Phase 4 aggregation weight:",
                f"{diagnostic_raw_weight:.8f}",
            )

            print(
                "Normalized Phase 4 weight:",
                f"{diagnostic_weight:.8f}",
            )

            print(
                "Raw logits:",
                diagnostic_logits,
            )

            print()
            print(
                "PAIRWISE PROBABILITIES"
            )

            print(
                "  REFUTED   =",
                f"{diagnostic_refuted:.8f}",
            )

            print(
                "  NEI       =",
                f"{diagnostic_nei:.8f}",
            )

            print(
                "  SUPPORTED =",
                f"{diagnostic_supported:.8f}",
            )

            diagnostic_pair_argmax = int(
                torch.argmax(
                    diagnostic_probs
                ).item()
            )

            diagnostic_pair_labels = {
                0: "REFUTED",
                1: "NEI",
                2: "SUPPORTED",
            }

            print(
                "Pairwise predicted class:",
                diagnostic_pair_labels[
                    diagnostic_pair_argmax
                ],
            )

            print()
            print(
                "WEIGHTED CONTRIBUTION"
            )

            print(
                "  REFUTED   =",
                f"{diagnostic_weight * diagnostic_refuted:.8f}",
            )

            print(
                "  NEI       =",
                f"{diagnostic_weight * diagnostic_nei:.8f}",
            )

            print(
                "  SUPPORTED =",
                f"{diagnostic_weight * diagnostic_supported:.8f}",
            )

            evidence_preview = (
                diagnostic_evidence
                .text
                .replace(
                    "\n",
                    " ",
                )
                .strip()
            )

            if (
                len(
                    evidence_preview
                )
                > 700
            ):
                evidence_preview = (
                    evidence_preview[
                        :700
                    ]
                    + "..."
                )

            print()
            print(
                "Evidence text:",
                evidence_preview,
            )

        print()
        print(
            "=" * 90
        )

        # ====================================================
        # CLAIM-LEVEL AGGREGATION
        # ====================================================

        pair_probabilities_64 = (
            pair_probabilities.to(
                dtype=torch.float64
            )
        )

        claim_probability_tensor = (
            torch.sum(
                pair_probabilities_64
                * normalized_weights[
                    :,
                    None
                ],
                dim=0,
            )
        )

        if (
            tuple(
                claim_probability_tensor
                .shape
            )
            != (
                3,
            )
        ):
            raise Phase4AdapterError(
                "Aggregated claim probability "
                "vector must have shape (3,)."
            )

        if not torch.isfinite(
            claim_probability_tensor
        ).all():
            raise Phase4AdapterError(
                "Aggregated claim probabilities "
                "contain non-finite values."
            )

        claim_probability_sum = float(
            claim_probability_tensor
            .sum()
            .item()
        )

        if abs(
            claim_probability_sum
            - 1.0
        ) > 1e-5:
            raise Phase4AdapterError(
                "Aggregated claim probabilities "
                "do not sum to 1. "
                f"Total={claim_probability_sum}"
            )

        # ====================================================
        # FINAL DIAGNOSTIC
        # ====================================================

        print()
        print(
            "=" * 90
        )

        print(
            "FINAL AGGREGATED PHASE 4 PROBABILITIES"
        )

        print(
            "=" * 90
        )

        diagnostic_final = (
            claim_probability_tensor
            .tolist()
        )

        print(
            "REFUTED   =",
            f"{float(diagnostic_final[0]):.8f}",
        )

        print(
            "NEI       =",
            f"{float(diagnostic_final[1]):.8f}",
        )

        print(
            "SUPPORTED =",
            f"{float(diagnostic_final[2]):.8f}",
        )

        diagnostic_final_argmax = int(
            torch.argmax(
                claim_probability_tensor
            ).item()
        )

        diagnostic_final_labels = {
            0: "REFUTED",
            1: "NEI",
            2: "SUPPORTED",
        }

        print(
            "ARGMAX CLASS ID:",
            diagnostic_final_argmax,
        )

        print(
            "ARGMAX LABEL:",
            diagnostic_final_labels[
                diagnostic_final_argmax
            ],
        )

        sorted_diagnostic = sorted(
            enumerate(
                diagnostic_final
            ),
            key=lambda item:
                item[
                    1
                ],
            reverse=True,
        )

        diagnostic_margin = (
            float(
                sorted_diagnostic[
                    0
                ][
                    1
                ]
            )
            - float(
                sorted_diagnostic[
                    1
                ][
                    1
                ]
            )
        )

        print(
            "TOP-2 PROBABILITY MARGIN:",
            f"{diagnostic_margin:.8f}",
        )

        print(
            "PROBABILITY SUM:",
            f"{claim_probability_sum:.8f}",
        )

        print(
            "=" * 90
        )

        print()

        # ====================================================
        # FINAL VERDICT
        # ====================================================

        claim_probabilities = (
            self._make_probabilities(
                claim_probability_tensor
                .tolist()
            )
        )

        verdict = (
            self._verdict_from_probabilities(
                claim_probabilities
            )
        )

        verifier_score = (
            self._selected_probability(
                claim_probabilities,
                verdict,
            )
        )

        # ====================================================
        # PAIRWISE OUTPUT RECORDS
        # ====================================================

        pairwise_predictions = []

        for (
            index,
            evidence,
        ) in enumerate(
            evidence_list
        ):

            pair_probs = (
                self._make_probabilities(
                    pair_probabilities[
                        index
                    ].tolist()
                )
            )

            pair_verdict = (
                self._verdict_from_probabilities(
                    pair_probs
                )
            )

            raw_logits = [
                float(
                    value
                )
                for value
                in logits[
                    index
                ].tolist()
            ]

            pairwise_predictions.append(
                PairwiseVerification(
                    evidence_id=
                        evidence.evidence_id,

                    probabilities=
                        pair_probs,

                    predicted_label=
                        pair_verdict,

                    raw_logits=
                        raw_logits,
                )
            )

        # ====================================================
        # EVIDENCE LINEAGE
        # ====================================================

        evidence_used_ids = [
            evidence.evidence_id
            for evidence
            in evidence_list
        ]

        for evidence in evidence_list:
            evidence.lineage.used_by_phase4 = (
                True
            )

        # ====================================================
        # TIMING
        # ====================================================

        end_clock = (
            time.perf_counter()
        )

        ended_at = (
            utc_now_string()
        )

        timing = PhaseTiming(
            phase=
                "phase4",

            started_at_utc=
                started_at,

            ended_at_utc=
                ended_at,

            latency_ms=(
                end_clock
                - start_clock
            )
            * 1000.0,
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = Phase4Result(
            status=
                PhaseStatus.SUCCESS,

            verdict=
                verdict,

            user_facing_verdict=
                verdict_to_user_facing(
                    verdict
                ),

            probabilities=
                claim_probabilities,

            verifier_score=
                verifier_score,

            pairwise_predictions=
                pairwise_predictions,

            evidence_used_ids=
                evidence_used_ids,

            aggregation_method=(
                "normalized weighted mean of "
                "Phase4 softmax probabilities; "
                "weight = relevance_score * "
                "hierarchy_weight"
            ),

            label_mapping={
                0:
                    Verdict.REFUTED,

                1:
                    Verdict.NEI,

                2:
                    Verdict.SUPPORTED,
            },

            model=
                self._build_model_artifact(),

            verdict_locked=
                True,

            timing=
                timing,

            warnings=[],

            error=None,
        )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print(
            "Final verdict:",
            result.verdict.value,
        )

        print(
            "User-facing verdict:",
            result
            .user_facing_verdict
            .value,
        )

        print(
            "REFUTED:",
            f"{result.probabilities.refuted:.6f}",
        )

        print(
            "NEI:",
            f"{result.probabilities.nei:.6f}",
        )

        print(
            "SUPPORTED:",
            f"{result.probabilities.supported:.6f}",
        )

        print(
            "Verifier score:",
            f"{result.verifier_score:.6f}",
        )

        print(
            "Checkpoint contract:",
            self.checkpoint_contract,
        )

        print()

        print(
            "Aggregation weights:"
        )

        for (
            evidence,
            normalized_weight,
        ) in zip(
            evidence_list,
            normalized_weights.tolist(),
        ):

            raw_weight = (
                float(
                    evidence.relevance_score
                )
                * float(
                    evidence.hierarchy_weight
                )
            )

            print(
                f"  {evidence.evidence_id}: "
                f"raw={raw_weight:.6f} "
                f"normalized="
                f"{normalized_weight:.6f}"
            )

        return result