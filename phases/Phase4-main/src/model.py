from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from .checkpoint import (
    extract_prefixed_state_dict,
    get_model_state_dict,
    load_torch_checkpoint,
    validate_phase1_checkpoint,
)


HIDDEN_SIZE = 768


class Phase4Verifier(nn.Module):

    def __init__(
        self,
        phase1_checkpoint: str | Path,
        classifier_dropout: float = 0.20,
        freeze_claim_biobert: bool = True,
        freeze_vit: bool = True,
        freeze_projection: bool = True,
        train_no_image_token: bool = True,
    ) -> None:

        super().__init__()

        self.vit = None

        bert_config = BertConfig(
            vocab_size=28996,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",

            # Standard hidden-state dropout is retained.
            hidden_dropout_prob=0.1,

            # IMPORTANT:
            # MPS scaled_dot_product_attention does not support
            # attention dropout in the current runtime.
            attention_probs_dropout_prob=0.0,

            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
        )

        # ====================================================
        # PHASE 1 CLAIM ENCODER
        # ====================================================

        self.biobert = BertModel(
            bert_config
        )

        # ====================================================
        # PHASE 1 PROJECTION / FUSION
        # ====================================================

        self.text_proj = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.img_proj = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                HIDDEN_SIZE * 2,
                HIDDEN_SIZE,
            ),
            nn.LayerNorm(
                HIDDEN_SIZE
            ),
        )

        # ====================================================
        # PHASE 1 LEARNED NO-IMAGE TOKEN
        # ====================================================

        self.no_image_token = nn.Parameter(
            torch.zeros(
                HIDDEN_SIZE,
                dtype=torch.float32,
            )
        )

        # ====================================================
        # PHASE 4 EVIDENCE ENCODER
        # ====================================================

        self.evidence_encoder = BertModel(
            copy.deepcopy(
                bert_config
            )
        )

        # ====================================================
        # PHASE 4 CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(
            nn.Linear(
                HIDDEN_SIZE * 2,
                768,
            ),
            nn.GELU(),
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                768,
                384,
            ),
            nn.GELU(),
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                384,
                3,
            ),
        )

        self._load_phase1(
            phase1_checkpoint
        )

        self._configure_trainability(
            freeze_claim_biobert=
                freeze_claim_biobert,

            freeze_vit=
                freeze_vit,

            freeze_projection=
                freeze_projection,

            train_no_image_token=
                train_no_image_token,
        )

    # ========================================================
    # PHASE 1 CHECKPOINT LOADING
    # ========================================================

    def _load_phase1(
        self,
        checkpoint_path: str | Path,
    ) -> None:

        checkpoint = load_torch_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )

        state_dict = get_model_state_dict(
            checkpoint
        )

        validate_phase1_checkpoint(
            state_dict
        )

        biobert_state = (
            extract_prefixed_state_dict(
                state_dict,
                "biobert",
            )
        )

        result = self.biobert.load_state_dict(
            biobert_state,
            strict=True,
        )

        if (
            result.missing_keys
            or result.unexpected_keys
        ):
            raise RuntimeError(
                "Strict Phase 1 BioBERT "
                "loading failed."
            )

        result = (
            self.evidence_encoder
            .load_state_dict(
                biobert_state,
                strict=True,
            )
        )

        if (
            result.missing_keys
            or result.unexpected_keys
        ):
            raise RuntimeError(
                "Strict evidence encoder "
                "initialization failed."
            )

        self.text_proj.load_state_dict(
            extract_prefixed_state_dict(
                state_dict,
                "text_proj",
            ),
            strict=True,
        )

        self.img_proj.load_state_dict(
            extract_prefixed_state_dict(
                state_dict,
                "img_proj",
            ),
            strict=True,
        )

        self.fusion.load_state_dict(
            extract_prefixed_state_dict(
                state_dict,
                "fusion",
            ),
            strict=True,
        )

        no_image = state_dict[
            "no_image_token"
        ]

        if tuple(
            no_image.shape
        ) != (
            HIDDEN_SIZE,
        ):
            raise ValueError(
                "Unexpected Phase 1 "
                "no_image_token shape: "
                f"{tuple(no_image.shape)}"
            )

        if not torch.isfinite(
            no_image
        ).all():
            raise ValueError(
                "Phase 1 no_image_token "
                "contains non-finite values."
            )

        with torch.no_grad():

            self.no_image_token.copy_(
                no_image
            )

    # ========================================================
    # TRAINABILITY
    # ========================================================

    @staticmethod
    def _set_requires_grad(
        module: nn.Module,
        value: bool,
    ) -> None:

        for parameter in (
            module.parameters()
        ):
            parameter.requires_grad = value

    def _configure_trainability(
        self,
        freeze_claim_biobert: bool,
        freeze_vit: bool,
        freeze_projection: bool,
        train_no_image_token: bool,
    ) -> None:

        self._set_requires_grad(
            self.biobert,
            not freeze_claim_biobert,
        )

        self._set_requires_grad(
            self.text_proj,
            not freeze_projection,
        )

        self._set_requires_grad(
            self.img_proj,
            not freeze_projection,
        )

        self._set_requires_grad(
            self.fusion,
            not freeze_projection,
        )

        self._set_requires_grad(
            self.evidence_encoder,
            True,
        )

        self._set_requires_grad(
            self.classifier,
            True,
        )

        self.no_image_token.requires_grad = (
            train_no_image_token
        )

    # ========================================================
    # CLAIM ENCODING
    # ========================================================

    def encode_claim(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids:
            Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = self.biobert(
            input_ids=input_ids,
            attention_mask=
                attention_mask,
            token_type_ids=
                token_type_ids,
            return_dict=True,
        )

        claim_embedding = (
            output.last_hidden_state[
                :,
                0,
                :
            ]
        )

        if (
            claim_embedding.ndim != 2
            or
            claim_embedding.shape[-1]
            != HIDDEN_SIZE
        ):
            raise RuntimeError(
                "Unexpected claim embedding "
                f"shape: "
                f"{tuple(claim_embedding.shape)}"
            )

        return claim_embedding

    # ========================================================
    # EVIDENCE ENCODING
    # ========================================================

    def encode_evidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids:
            Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = self.evidence_encoder(
            input_ids=input_ids,
            attention_mask=
                attention_mask,
            token_type_ids=
                token_type_ids,
            return_dict=True,
        )

        evidence_embedding = (
            output.last_hidden_state[
                :,
                0,
                :
            ]
        )

        if (
            evidence_embedding.ndim != 2
            or
            evidence_embedding.shape[-1]
            != HIDDEN_SIZE
        ):
            raise RuntimeError(
                "Unexpected evidence embedding "
                f"shape: "
                f"{tuple(evidence_embedding.shape)}"
            )

        return evidence_embedding

    # ========================================================
    # TEXT-ONLY FUSION
    # ========================================================

    def build_text_only_fusion(
        self,
        claim_embedding:
            torch.Tensor,
    ) -> torch.Tensor:

        if (
            claim_embedding.ndim != 2
            or
            claim_embedding.shape[-1]
            != HIDDEN_SIZE
        ):
            raise ValueError(
                "claim_embedding must have "
                "shape [batch, 768]."
            )

        batch_size = (
            claim_embedding.shape[0]
        )

        text_feature = self.text_proj(
            claim_embedding
        )

        no_image = (
            self.no_image_token
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        image_feature = self.img_proj(
            no_image
        )

        combined = torch.cat(
            [
                text_feature,
                image_feature,
            ],
            dim=-1,
        )

        if combined.shape[-1] != 1536:
            raise RuntimeError(
                "Unexpected fusion input "
                f"shape: {tuple(combined.shape)}"
            )

        fused_claim = self.fusion(
            combined
        )

        if fused_claim.shape[-1] != 768:
            raise RuntimeError(
                "Unexpected fused claim "
                f"shape: "
                f"{tuple(fused_claim.shape)}"
            )

        return fused_claim

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        claim_input_ids:
            torch.Tensor,
        claim_attention_mask:
            torch.Tensor,
        evidence_input_ids:
            torch.Tensor,
        evidence_attention_mask:
            torch.Tensor,
        claim_token_type_ids:
            Optional[
                torch.Tensor
            ] = None,
        evidence_token_type_ids:
            Optional[
                torch.Tensor
            ] = None,
    ) -> Dict[str, torch.Tensor]:

        claim_embedding = (
            self.encode_claim(
                input_ids=
                    claim_input_ids,

                attention_mask=
                    claim_attention_mask,

                token_type_ids=
                    claim_token_type_ids,
            )
        )

        fused_claim = (
            self.build_text_only_fusion(
                claim_embedding
            )
        )

        evidence_embedding = (
            self.encode_evidence(
                input_ids=
                    evidence_input_ids,

                attention_mask=
                    evidence_attention_mask,

                token_type_ids=
                    evidence_token_type_ids,
            )
        )

        verifier_input = torch.cat(
            [
                fused_claim,
                evidence_embedding,
            ],
            dim=-1,
        )

        if verifier_input.shape[-1] != 1536:
            raise RuntimeError(
                "Unexpected verifier input "
                f"shape: "
                f"{tuple(verifier_input.shape)}"
            )

        logits = self.classifier(
            verifier_input
        )

        if (
            logits.ndim != 2
            or logits.shape[-1] != 3
        ):
            raise RuntimeError(
                "Expected classifier logits "
                "with shape [batch,3], got "
                f"{tuple(logits.shape)}"
            )

        return {
            "logits":
                logits,

            "claim_embedding":
                claim_embedding,

            "fused_claim":
                fused_claim,

            "evidence_embedding":
                evidence_embedding,

            "verifier_input":
                verifier_input,
        }

    # ========================================================
    # PARAMETER REPORT
    # ========================================================

    def parameter_report(
        self,
    ) -> Dict[str, int]:

        total = 0
        trainable = 0

        for parameter in (
            self.parameters()
        ):

            count = parameter.numel()

            total += count

            if parameter.requires_grad:
                trainable += count

        return {
            "total":
                total,

            "trainable":
                trainable,

            "frozen":
                total - trainable,
        }

    # ========================================================
    # TRAINABILITY REPORT
    # ========================================================

    def component_trainability(
        self,
    ) -> Dict[str, bool]:

        def any_trainable(
            module: nn.Module,
        ) -> bool:

            return any(
                parameter.requires_grad

                for parameter
                in module.parameters()
            )

        return {
            "biobert":
                any_trainable(
                    self.biobert
                ),

            "vit":
                False,

            "text_proj":
                any_trainable(
                    self.text_proj
                ),

            "img_proj":
                any_trainable(
                    self.img_proj
                ),

            "fusion":
                any_trainable(
                    self.fusion
                ),

            "evidence_encoder":
                any_trainable(
                    self.evidence_encoder
                ),

            "no_image_token":
                bool(
                    self.no_image_token
                    .requires_grad
                ),

            "classifier":
                any_trainable(
                    self.classifier
                ),
        }